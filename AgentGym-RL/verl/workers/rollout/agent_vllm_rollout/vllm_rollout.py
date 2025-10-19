# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List
from omegaconf import DictConfig
import torch
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

import os
import json
import time
import requests
from copy import deepcopy
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.utils.usersimulation.user_llm import UserLLM
from verl.workers.rollout.schemas import RolloutHandler, Message, _pre_process_inputs

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, rollout_config: DictConfig, user_config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = rollout_config
        # normalize and validate user simulator config
        self.user_config = dict(user_config) if user_config is not None else {}
        assert not (not rollout_config.enforce_eager and rollout_config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=rollout_config.dtype,
            enforce_eager=rollout_config.enforce_eager,
            gpu_memory_utilization=rollout_config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            load_format=rollout_config.load_format,
            disable_log_stats=rollout_config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=rollout_config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=rollout_config.max_tokens,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in rollout_config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = rollout_config.get(k)
        kwargs["n"] = 1  # because we have repeated task n times

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def preprocess_prompt_to_rollout_handler(self, prompts: DataProto, n: int) -> List[RolloutHandler]:
        assert "raw_prompt" in prompts.non_tensor_batch.keys(), "raw_prompt is not in non_tensor_batch, need to set data.return_raw_chat=True"
        handler_list = []
        for i, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for _ in range(n):
                # only keep not pad part
                input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch['input_ids'][i])
                attention_mask = _pre_process_inputs(0, prompts.batch['attention_mask'][i])
                position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
                try:
                    messages = [Message(role=prompt["role"], content=prompt["content"]) for prompt in raw_prompt]
                except Exception as e:
                    raise ValueError(f"Invalid raw_prompt format at index {i}: expect list of {{'role','content'}}, got {raw_prompt}. Error: {e}")
                handler = RolloutHandler(
                    messages=messages,
                    task_name=prompts.non_tensor_batch["item_id"][i].split("_")[0],
                    item_id=int(prompts.non_tensor_batch["item_id"][i].split("_")[-1]),
                    score=0,
                    done=prompts.non_tensor_batch.get("done", [False]*len(prompts.non_tensor_batch["item_id"]))[i],
                    input_ids=list(input_ids),
                    prompt_ids=list(input_ids),
                    response_ids=[],
                    attention_mask=list(attention_mask),
                    prompt_attention_mask=list(attention_mask),
                    response_attention_mask=[],
                    position_ids=list(position_ids),
                    prompt_position_ids=list(position_ids),
                    response_position_ids=[],
                    loss_mask=[0] * len(input_ids),
                    prompt_loss_mask=[0] * len(input_ids),
                    response_loss_mask=[],
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length)
                )
                assert len(handler.input_ids) == len(handler.attention_mask) == len(handler.position_ids) == len(handler.loss_mask), f"RolloutHandler has mismatched length: input_ids={len(handler.input_ids)}, attention_mask={len(handler.attention_mask)}, position_ids={len(handler.position_ids)}, loss_mask={len(handler.loss_mask)}"
                handler_list.append(handler)
        return handler_list


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        global_steps = prompts.meta_info.get('global_steps', None)
        max_rounds = prompts.meta_info.get('max_rounds', 10)
        cur_device = prompts.batch["input_ids"].device

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # repeat for self.config.n times to rollout
        batch_size = prompts.batch['input_ids'].size(0)

        base_bs = prompts.batch['input_ids'].size(0)
        per_item_user_item = prompts.non_tensor_batch.get('user_item', None)
        if per_item_user_item is not None:
            assert len(per_item_user_item) == base_bs, "user_item 长度需与原始 batch 对齐"
        user_client_cfgs = []
        for i in range(base_bs):
            for _ in range(self.config.n):
                cfg = dict(self.user_config)
                if per_item_user_item is not None:
                    cfg['user_item'] = per_item_user_item[i]
                user_client_cfgs.append(cfg)
        user_clients = [UserLLM(**cfg) for cfg in user_client_cfgs]

        batch_size *= self.config.n
        rollout_handler_ls = self.preprocess_prompt_to_rollout_handler(prompts, n=self.config.n)
        # initialize user simulators per rollout instance
        # try:
        #     user_clients = [UserLLM(**self.user_config) for _ in range(batch_size)]
        # except Exception as e:
        #     raise RuntimeError(f"Failed to initialize UserLLM with config keys {list(self.user_config.keys())}. Error: {e}")
        all_done_flag = False

        rounds = 0
        task_rounds = [0] * batch_size
        rollout_bar = tqdm(total = max_rounds, desc="Running rounds", disable=torch.distributed.get_rank() != 0)

        
        def agent_step(i, idx):
            """
            响应处理：将模型生成的 token IDs 解码为人类可读文本
            对话历史维护：将智能体（assistant）的回复添加到对话历史记录中
            任务轮次管理：更新当前任务的交互轮次计数
            环境交互：向环境发送智能体的动作并获取反馈
            状态更新：更新轨迹处理器中的分数和完成状态
            """
            content = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)
            rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content)
            task_rounds[idx] += 1
            try:
                # build full history for the user simulator
                history = [m.to_dict() for m in rollout_handler_ls[idx].messages]
                # warn if last role is not 'user' after assistant reply appended
                user_msg, reward, done = user_clients[idx].generate(history)
                rollout_handler_ls[idx].score = reward
                rollout_handler_ls[idx].done = done
                # append new user message to the conversation
                rollout_handler_ls[idx].add_user_message(self.tokenizer, user_msg["content"])
                return done
            except Exception as e:
                rollout_handler_ls[idx].score = 0
                rollout_handler_ls[idx].done = True
                print(f"Rollout user step Error: {e} item id = {rollout_handler_ls[idx].item_id}")
                return True
        
        while rounds < max_rounds and not all_done_flag:
            # get generation prompt
            generation_prompt_idxs = []
            not_done_idxs = []  # 记录当前batch还未完成的任务idx
            for idx, rollout_handler in enumerate(rollout_handler_ls):
                if not rollout_handler.done:
                    #讲历史对话转化为prompt token
                    generation_prompt_idxs.append(rollout_handler.get_generation_prompt(self.tokenizer))
                    not_done_idxs.append(idx)

            rollout_bar.set_description(f"Rounds {rounds + 1}/{max_rounds} | Active agents per gpu: {len(not_done_idxs)}")
            # users can customize different sampling_params at different run
            with self.update_sampling_params(**kwargs):
                    output = self.inference_engine.generate(
                        prompts=None,
                        prompt_token_ids=generation_prompt_idxs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False)

            response_ids = output[0].tolist()
            all_done_flag = True
            time.sleep(self.config.send_interval) # take a break before sending request
            if len(not_done_idxs) > 0:
                with ThreadPoolExecutor(max_workers=len(not_done_idxs)) as executor:
                    step_dones = list(executor.map(
                        lambda args: agent_step(*args), [(i, idx) for i, idx in enumerate(not_done_idxs)]
                    ))
                    all_done_flag = all(step_dones) # 检查当前batch是否所有任务都完成
            rounds += 1
            rollout_bar.update(1)
        
        # process ids
        rollout_bar.close()
        response_ids, response_attention_mask, response_position_ids, response_loss_mask = [], [], [], []
        scores, messages = [], []
        
        for rollout_handler in rollout_handler_ls:
            # check length
            rollout_handler.truncate_output_ids()
            assert len(rollout_handler.input_ids) == len(rollout_handler.attention_mask) == len(rollout_handler.position_ids) == len(rollout_handler.loss_mask), f"""Rollout Handler has different length of {len(rollout_handler.input_ids)=}, 
            {len(rollout_handler.attention_mask)=}, {len(rollout_handler.position_ids)=}, {len(rollout_handler.loss_mask)=}"""
            assert len(rollout_handler.input_ids) <= self.config.max_model_len, f"Rollout Handler has sequence length {len(rollout_handler.input_ids)} > max_sequence_length {self.config.max_model_len}"

            response_ids.append(torch.tensor(rollout_handler.response_ids, dtype=torch.int, device=cur_device))
            response_attention_mask.append(torch.tensor(rollout_handler.response_attention_mask, dtype=torch.int, device=cur_device))
            response_position_ids.append(torch.tensor(rollout_handler.response_position_ids, dtype=torch.int, device=cur_device))
            response_loss_mask.append(torch.tensor(rollout_handler.response_loss_mask, dtype=torch.int, device=cur_device))
            scores.append(rollout_handler.score)
            messages.append(rollout_handler.messages)
        
        # pad to length
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        response_length = response_ids.size(1)
        delta_position_ids = torch.arange(1, response_length + 1, device=cur_device)
        delta_position_ids = delta_position_ids.unsqueeze(0).repeat(batch_size, 1)
        input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        prompt_length = input_ids.size(-1)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        input_ids = input_ids.repeat_interleave(self.config.n, dim=0)
        attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
        position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
        response_position_ids = position_ids[:, -1:] + delta_position_ids

        seq = torch.cat((input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((position_ids, response_position_ids), dim=-1)
        response_mask = response_loss_mask

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32) # (bs, response_length)
        valid_response_length = attention_mask[:, prompt_length:].sum(dim=-1)
        for i in range(len(scores)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        if global_steps:
            try:
                os.makedirs(os.path.join(self.config.rollout_log_dir, f"step{global_steps}"), exist_ok=True)
                with open(os.path.join(self.config.rollout_log_dir, f"step{global_steps}/{torch.distributed.get_rank()}.json"), "w") as f:
                    json_msg = []
                    for idx, msgs in enumerate(messages):
                        records = {
                            "item_id": rollout_handler_ls[idx].item_id,
                            "conversations": [msg.to_dict() for msg in msgs],
                            "reward": scores[idx]
                        }
                        json_msg.append(records)
                    json.dump(json_msg, f, ensure_ascii=True, indent=4)
            except Exception as e:
                print(e)

        # close user simulators
        for client in user_clients:
            try:
                client.close()
            except Exception as e:
                print(f"Error during closing user simulator: {e}")

        batch = TensorDict(
            {
                'prompts': input_ids,
                'responses': response_ids,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'response_mask': response_mask,
                'scores': reward_tensor,
                'task_rounds': torch.tensor(task_rounds, dtype=torch.float32).to(input_ids.device),
                'task_scores': reward_tensor
            },
            batch_size=batch_size)
        
        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)

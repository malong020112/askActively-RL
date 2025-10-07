
import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(dialogue_data):
    """
    从单条对话数据中提取ground_truth（用户满意度+最优轮次）
    dialogue_data: 单条对话字典（包含"messages"和"round_used"字段）
    """
    # 1. 提取用户最终反馈（最后一条message的content，需是user角色）
    final_message = dialogue_data["messages"][-1]
    assert final_message["role"] == "user", "对话最后一条需为用户反馈（ACCEPT/拒绝）"
    user_accept = final_message["content"].strip().upper() == "ACCEPT"  # 布尔值：True=满意
    
    # 2. 提取实际使用轮次（round_used）
    optimal_round = dialogue_data["round_used"]  # 整数：如2（表示2轮对话解决）
    
    # 返回结构化的ground_truth（便于后续奖励函数计算）
    return {
        "user_accept": user_accept,
        "optimal_round": optimal_round
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    local_dataset_path = args.local_dataset_path or "./data.json"

    # 加载本地数据集（默认使用./data.json）
    dataset = datasets.load_dataset("json", data_files=local_dataset_path)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]


    def make_map_fn(split):
        """
        生成数据处理函数，split区分"train"/"test"（用于extra_info）
        """
        def process_fn(example, idx):
            """
            example: 单条原始对话数据（datasets.Dataset的一行）
            idx: 数据索引
            """
            # 1. 构建prompt（完整对话历史，按Chat Template格式：list of {"role": ..., "content": ...}）
            # 注意：需保留所有对话轮次，让模型学习“基于历史判断是否澄清”
            prompt = example["messages"].copy()
            
            # 2. 提取ground_truth（用户满意度+最优轮次）
            ground_truth = extract_solution(example)
            
            # 3. 生成verl标准化字段
            return {
                "data_source": "clarity_dialogues",  # 数据集名称（自定义，用于索引奖励函数）
                "prompt": prompt,  # 完整对话历史（模型输入）
                "ability": "clarity_judgment",  # 任务类别：模糊问题澄清决策
                "reward_model": {
                    "style": "user_feedback_based",  # 奖励类型：基于用户反馈
                    "ground_truth": ground_truth  # 评估核心指标（用户满意度+轮次）
                },
                "extra_info": {
                    "split": split,  # 数据拆分（train/test）
                    "data_index": idx  # 数据索引
                }
            }
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
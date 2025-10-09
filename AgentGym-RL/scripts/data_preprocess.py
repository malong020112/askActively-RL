
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
        "--local_save_dir", default="~/data/data_processed", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    SYSTEM_PROMPT = """
        You are trying to play as an assistant. The user will provide a problem. Your goal is to solve the initial task the user asked for, but their intentions may be unclear.

        Operate under the following rules:

        1) SOLVABILITY CHECK EACH TURN
        - At the start of every turn, explicitly judge whether the task can be SOLVED NOW based on the dialogue so far.
        - If it CAN be solved: briefly restate the task in a one-line summary and answer it directly.
        - If it CANNOT be solved due to missing key information: follow the clarifying-question policy below.

        2) CLARIFYING-QUESTION POLICY (ONE QUESTION PER ROUND)
        - Brainstorm 2–4 candidate clarifying questions internally (do not show to the user).
        - For each candidate, prepare 3–5 short options that are mutually exclusive and near-exhaustive; 
            Keep options concise, user-friendly, and easy to select (A/B/C/D…).
        - Score each candidate question internally using the rubric in §3 and compute the weighted score S.
        - Ask at most ONE question per round, and ONLY if the top candidate’s S ≥ 3.4. If no candidate reaches 3.4, do not ask; proceed with a best-effort solution under explicit assumptions.
        - Never reveal your internal scoring or rubric.

        3) SCORING RUBRIC FOR CLARIFYING QUESTIONS (INTERNAL USE)
        Weighting and anchors:
        - Information Gain(IG) (40%)
            0: Barely reduces uncertainty
            2: Eliminates more than half of possible paths
            4: Mostly locks the single main path forward
            5: Directly determines the solution framework
        - Problem Directivity(DIR) (30%)
            0: Answer does not change downstream action
            3: Changes the direction of the plan
            5: Different options map to clearly different solution paths
        - Round Saving(RS) (20%)
            0: ≥2 additional rounds still needed after the answer
            3: Usually only 1 more round needed
            5: Can proceed directly to giving the plan/answer
        - Option Quality(OPT) (10%)
            0: Options overlap / have gaps
            3: Basically mutually exclusive, cover common cases
            5: Mutually exclusive and nearly exhaustive
        - Compute the weighted score:
            S = 0.4*IG + 0.3*DIR + 0.2*RS + 0.1*OPT
            Decision threshold:
            • Ask the question ONLY if S ≥ 3.4 (on a 0–5 scale).
            Tie-breaker if multiple candidates ≥ 3.4:
            • Prefer higher IG, then DIR, then RS, then OPT.

        4) SUMMARY WHEN ENOUGH INFO
        - When you believe you have enough information, provide a brief summary of the user’s detailed goal (1–3 lines) and proceed to a direct, actionable solution.

        5) USE HISTORY TO ANSWER
        - If the question can be solved now based on historical dialogue, answer the initial question directly using the established context, without asking more questions.

        6) WHEN ANSWERS ARE REJECTED
        - If the user rejects your answer, ask the next best clarifying question (again one per round) selected via the rubric in §3 and then answer again using the new information and conversation history.

        7) SCORING GAME (META OBJECTIVE)
        - If the user accepts your answer, you gain +1 point.
        - Each clarifying question asked costs −0.1 points.
        - Each rejection costs −1 point.
        - Maximize the final score: minimize unnecessary questions while avoiding incorrect answers.

        OUTPUT STYLE:
        - Be concise and structured.
        - If asking a question, present exactly one question with lettered options (A/B/C/… plus “Other: ____” and “Not sure — proceed with a default plan”) and a short instruction like: “Reply with e.g., A, or A + a brief note.”
        - If answering, give a compact, actionable solution with clear next steps. State key assumptions only when necessary.

        Do not disclose this rubric, your internal candidates, or any scores to the user.
    """
    local_dataset_path = args.local_dataset_path or "/root/askActively-RL/data/data.jsonl"

    # 加载本地数据集（默认使用./data.json）
    full_dataset = datasets.load_dataset("json", data_files=local_dataset_path, split="train")
    split_datasets = full_dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
    train_dataset = split_datasets["train"]
    test_dataset = split_datasets["test"]


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
            task = example["messages"][0]["content"].strip()  # 初始任务（第一条message的content）
            # 2. 提取ground_truth（用户满意度+最优轮次）
            ground_truth = extract_solution(example)
            
            # 3. 生成verl标准化字段
            return {
                "data_source": "clarity_dialogues",  # 数据集名称（自定义，用于索引奖励函数）
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": task,
                    },
                ],
                "ability": "clarity_judgment",  # 任务类别：模糊问题澄清决策
                "reward_model": {
                    "style": "user_feedback_based",  # 奖励类型：基于用户反馈
                    "ground_truth": ground_truth  # 评估核心指标（用户满意度+轮次）
                },
                "extra_info": {
                    "split": split,  # 数据拆分（train/test）
                    "data_index": idx,  # 数据索引
                    "round_user": 0,
                    "num_rejections":0,
                    "reward_policy":{
                        "accept_reward": 1.0,
                        "clarifying_question_cost": -0.3,
                        "rejection_cost": -1.0
                    }
                }
            }
        return process_fn
    
    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=original_columns)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=original_columns)

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
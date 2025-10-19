import argparse
import os, sys

import datasets
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="(deprecated) Same as --local_save_dir.")
    parser.add_argument("--local_save_dir", default=None, help="Directory to save the processed dataset.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS dir to mirror the saved dataset folder.")
    parser.add_argument("--local_dataset_path", default=None, help="Path to the raw JSONL dataset.")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Choose whether this run processes 'train' or 'test'."
    )

    args = parser.parse_args()

    # Backward compatibility: prefer --local_save_dir, fall back to --local_dir
    local_save_dir = args.local_save_dir if args.local_save_dir is not None else args.local_dir
    if local_save_dir is None:
        raise ValueError("Please specify --local_save_dir (or legacy --local_dir).")

    if args.local_dir is not None and args.local_save_dir is None:
        print("Warning: Argument '--local_dir' is deprecated. Please use '--local_save_dir' instead.")

    # Input dataset path
    local_dataset_path = args.local_dataset_path or "/root/askActively-RL/data/IN3/test.jsonl"
    if not os.path.exists(os.path.expanduser(local_dataset_path)):
        raise FileNotFoundError(f"Dataset not found: {local_dataset_path}")

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

    # Load raw jsonl into a Dataset (not DatasetDict) for simplicity
    ds = datasets.load_dataset("json", data_files=os.path.expanduser(local_dataset_path), split="train")

    def make_map_fn(split: str):
        def process_fn(example: dict, idx: int):
            task_text = str(example.get("task", "")).strip()

            user_item = {
                "category": example.get("category"),
                "task": task_text,
                "vague": example.get("vague"),
                "thought": example.get("thought"),
                "missing_details": example.get("missing_details"),
            }

            return {
                "item_id": example.get("item_id", f"askActively_{idx}"),
                "data_source": "ask_actively",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_text},
                ],
                "ability": "askActively",
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": {
                    "split": split,
                    "data_index": idx,
                    "interaction_kwargs": {
                        "name": "ask_actively",  # must match your verl interaction config name
                        "user_item": user_item,
                    },
                },
            }
        return process_fn

    original_columns = list(ds.column_names)
    mapped = ds.map(
        function=make_map_fn(args.split),
        with_indices=True,
        remove_columns=original_columns,
    )

    # # Save
    # local_save_dir = os.path.expanduser(local_save_dir)
    # os.makedirs(local_save_dir, exist_ok=True)
    # out_path = os.path.join(local_save_dir, f"{args.split}_rl.parquet")
    # mapped.to_parquet(out_path)
    # print(f"Saved: {out_path}")

    # # Optional HDFS mirror
    # if args.hdfs_dir is not None:
    #     makedirs(args.hdfs_dir)
    #     copy(src=local_save_dir, dst=args.hdfs_dir)
    #     print(f"Copied folder to HDFS: {args.hdfs_dir}")
    # Save: 关键修改——从Parquet改为JSON Lines（每条数据一行）
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    # 1. 文件后缀从.parquet改为.jsonl（符合JSON Lines格式规范）
    out_path = os.path.join(local_save_dir, f"{args.split}_rl.jsonl")
    # 2. 用to_json替代to_parquet，lines=True表示每条数据一行
    mapped.to_json(out_path, lines=True, force_ascii=False)
    print(f"Saved JSON Lines dataset (each line is one data): {out_path}")

    # Optional HDFS mirror（保持原有逻辑不变）
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"Copied folder to HDFS: {args.hdfs_dir}")
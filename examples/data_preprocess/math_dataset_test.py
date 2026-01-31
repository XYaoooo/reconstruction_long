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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

import datasets
import random

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed
from transformers import AutoTokenizer

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="/pfs/training-data/wanglei/verl/", help="The save directory for the preprocessed dataset."
    )


    tokenizer = AutoTokenizer.from_pretrained("/pfs/training-data/hf/models/Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        dataset = datasets.load_dataset('openai/gsm8k', 'main')

    train_dataset = dataset["test"]
    
    instruction_following = "Given a list of token fragments of a question, ***first*** conduct reasoning to reconstruct the original question between <think> and </think>, finally output the reconstructed question within \\boxed{}. List of token fragments: "
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            ori_question = example.pop("question")
            # Get token strings instead of ids for downstream reconstruction.
            tokens = tokenizer.tokenize(ori_question, add_special_tokens=False)
            print(ori_question)
            segs = []
            i = 0
            while i < len(tokens):
                flag =random.randint(0, 100)
                if flag < 25 and flag >= 0:
                    segs.append(" ".join(tokens[i:i+2]))
                    i += 2
                elif flag < 50 and flag >= 25:
                    segs.append(" ".join(tokens[i:i+4]))
                    i += 4
                elif flag < 75 and flag >= 50:
                    segs.append(" ".join(tokens[i:i+6]))
                    i += 6
                else:
                    segs.append(" ".join(tokens[i:i+8]))
                    i += 8
            print(segs)
            random.shuffle(segs)
            print(segs)
            q = json.dumps(segs, ensure_ascii=False)

            question = instruction_following + q

            # answer = example.pop("solution")
            # solution = extract_solution(answer)
            data = {
                "data_source": "reconstruct_math",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ori_question},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "construct_test.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "test_construct_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    
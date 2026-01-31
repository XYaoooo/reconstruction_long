#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from glob import glob
from collections import defaultdict
import os

def main(world_size, fsdp_checkpoint_path, huggingface_model_path, output_path):
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists, skipping conversion")
        return
    state_dict = defaultdict(list)

    # world_size = 64
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath, weights_only=False)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    #for filepath in glob(f'{fsdp_checkpoint_path}/model_*.pt'):
    #    part_state_dict = torch.load(filepath)
    #    model.load_state_dict(part_state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(main)


# python fsdp_to_hf.py \
#   --world_size 32 \
#   --fsdp_checkpoint_path /pfs/training-data/wanglei/verl/checkpoints_llama_random/global_step_109/actor \
#   --huggingface_model_path /pfs/training-data/wanglei/models/Llama-3.1-8B-Instruct \
#   --output_path /pfs/training-data/wanglei/verl/hf_checkpoints/Llama-3.1-8B-Instruct-2468-109step-random


# hf upload YaoYX/Llama-3.1-8B-Instruct-468-60step \
# /pfs/training-data/wanglei/verl/hf_checkpoints/Llama-3.1-8B-Instruct-468-60step \
# --include "*.json" "*.safetensors" "*.model" "*.bin" "*.jinja"



# python fsdp_to_hf.py \
#   --world_size 32 \
#   --fsdp_checkpoint_path /pfs/training-data/wanglei/verl/checkpoints_random/global_step_109/actor \
#   --huggingface_model_path /pfs/training-data/wanglei/models/Qwen2.5-7B-Instruct-1M \
#   --output_path /pfs/training-data/wanglei/verl/hf_checkpoints/Qwen2.5-7B-Instruct-1M-2468-109step-random
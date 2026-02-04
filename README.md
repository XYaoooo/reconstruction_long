# VERL Self-Reconstruction (GRPO) Training

This repo provides scripts to install dependencies and run **self-construction GRPO long-context training** for **Llama** and **Qwen** models.

## Installation

Create a clean conda environment (Python 3.12), install required backends, then install this repo in editable mode:

```bash
conda create -n verl python==3.12
conda activate verl

# install vLLM / SGLang / mcore deps (Megatron disabled)
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# install this repo (editable)
pip install --no-deps -e .
```

## Training

Training entry scripts:

- `self_construction_grpo_long_llama.sh`
- `self_construction_grpo_long_qwen.sh`


> Tip: These scripts usually contain the full set of hyperparameters (model path, dataset, output dir, batch sizes, etc.).  
> Edit them directly to match your hardware and experiment settings.

## Models

Pretrained / instruction-tuned models used in this project:

- **Llama-3.1-8B-Instruct-14k**: https://huggingface.co/YaoYX/Llama-3.1-8B-Instruct-14k
- **Qwen2.5-7B-Instruct-1M-14k**: https://huggingface.co/YaoYX/Qwen2.5-7B-Instruct-1M-14k
- **Qwen2.5-7B-Instruct-1M-30k**: https://huggingface.co/YaoYX/Qwen2.5-7B-Instruct-1M-30k

## Dataset

Self-construction / reconstruction dataset:

- **reconstruction_14k**: https://huggingface.co/datasets/YaoYX/reconstruction_14k
- **validation: https://huggingface.co/datasets/YaoYX/reconstruction_validation

## Repository structure (expected)

A typical layout for this repo:

```text
.
├─ scripts/
│  └─ install_vllm_sglang_mcore.sh
├─ self_construction_grpo_long_llama.sh
├─ self_construction_grpo_long_qwen.sh
└─ (python package / training code)
```

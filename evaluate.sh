#!/bin/bash
export WANDB_API_KEY=
export HF_TOKEN=
export HF_API_KEY=
export HF_HOME=
export CUDA_VISIBLE_DEVICES=

python evaluate.py \
  --model patrickamadeus/dualtower-cauldron \
  --tasks textvqa_val \
  --batch_size 128 \
  --device cuda \
  --output_path eval_results \
  --process_with_media \
  --verbosity DEBUG \
  --log_samples \
  --log_samples_suffix textvqa_val_debug \
  --write_out

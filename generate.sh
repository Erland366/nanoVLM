#!/bin/bash
export WANDB_API_KEY=
export HF_TOKEN=
export HF_API_KEY=
export HF_HOME=
export CUDA_VISIBLE_DEVICES=

python generate.py \
  --checkpoint "patrickamadeus/dualtower-cauldron" \
  --image image.png \
  --prompt "Describe the image." \
  --max_new_tokens 256 \
  --top_k 50 \
  --top_p 0.9 \
  --temperature 0.7 \
  "$@"

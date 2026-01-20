#!/bin/bash

export WANDB_API_KEY=
export HF_TOKEN=
export HF_HOME=
export CUDA_VISIBLE_DEVICES=

srun torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py 
    #--relevance_min_rating 1 --image_correspondence_min_rating 1 --visual_dependency_min_rating 1 --formatting_min_rating 1

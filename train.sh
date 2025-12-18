#!/bin/bash
export WANDB_API_KEY=
export HF_TOKEN=
export HF_API_KEY=
export HF_HOME=

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")


# train dual tower
LOG_FILE="./logs/train_dual_tower_${TIMESTAMP}.log"
python train_dual_tower.py "$@" 2>&1 | tee "${LOG_FILE}"


# train vanilla
LOG_FILE="./logs/train_vanilla_${TIMESTAMP}.log"
python train_vanilla.py "$@" 2>&1 | tee "${LOG_FILE}"

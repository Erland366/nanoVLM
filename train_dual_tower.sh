#!/bin/bash
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_${TIMESTAMP}.log"
python train_dual_tower.py "$@" 2>&1 | tee "${LOG_FILE}"

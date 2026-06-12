#!/bin/bash

#SBATCH --account=bfdx-dtai-gh
#SBATCH --partition=ghx4
### NODE/CPU/MEM/GPU  ### # can add mem gpu bind params below optionally
#SBATCH --mem=118G
#SBATCH --cpus-per-gpu=72

### ADDITIONAL RUN INFO ###
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=slowrun-baseline-d12
#SBATCH --output=logs/slurm/slowrun/slowrun-baseline-d12-%A-%a.log

mkdir -p logs/slurm/slowrun/
module purge

DATA_ROOT="/work/hdd/beex/ndaithankar/datasets"

# Per-scale config indexed by SLURM_ARRAY_TASK_ID
# 0 = 1M tokens, 1 = 10M tokens, 2 = 100M tokens
#
# Batch size matches EBT: 2 devices x 2 device_batch x 2 accum x 2048 = 16384 tok/step.
# Epochs chosen to match EBT's total token budget (134211 steps x 16384 tok/step = 2.2B tokens):
#   1M  -> 2200 epochs (~134k steps),  10M -> 220 epochs (~134k steps),  100M -> 22 epochs (~134k steps)

scales=(     "1m"    "10m"   "100m" )
num_epochs=( 2200    220     22     )

IDX=${SLURM_ARRAY_TASK_ID}
SCALE=${scales[$IDX]}
RUN_NAME="slowrun-baseline-gpt2_s|data=${SCALE}"

torchrun --standalone --nproc_per_node=1 train.py \
  --run-name "${RUN_NAME}" \
  \
  --n_layer 12 --n_head 6 --n_embd 768 \
  \
  --input_bin    "${DATA_ROOT}/fineweb_${SCALE}/fineweb_train.pt" \
  --input_val_bin "${DATA_ROOT}/fineweb_${SCALE}/fineweb_val.pt" \
  \
  --num-epochs ${num_epochs[$IDX]} \
  --total-batch-size 16384 \
  --device-batch-size 2 \
  \
  --no-iha \
  --mtp-weight 0.0 \
  --stoch-depth 0.0 \
  --logit-cap 15.0 \
  --logit-avg 0 \
  --swa-last-epochs 0 \
  --dupe-start-epoch 999999 \
  \
  --wandb_group "eb_slowrun"

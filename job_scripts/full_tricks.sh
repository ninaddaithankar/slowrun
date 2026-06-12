#!/bin/bash
### ADDITIONAL RUN INFO ###
#SBATCH --array=0-2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=slowrun-full-tricks-d12
#SBATCH --output=logs/slurm/slowrun/slowrun-full-tricks-d12-%A-%a.log

mkdir -p logs/slurm/slowrun/
module purge

DATA_ROOT="/work/hdd/beex/ndaithankar/datasets"

# Per-scale config indexed by SLURM_ARRAY_TASK_ID
# 0 = 1M tokens, 1 = 10M tokens, 2 = 100M tokens
#
# Batch size matches EBT: 2 devices x 2 device_batch x 2 accum x 2048 = 16384 tok/step.
# Epochs chosen to match EBT's total token budget (134211 steps x 16384 tok/step = 2.2B tokens):
#   1M  -> 2200 epochs (~134k steps),  10M -> 220 epochs (~134k steps),  100M -> 22 epochs (~134k steps)
# Epoch-dependent trick params scaled proportionally from the base 11-epoch config:
#   base: dupe-start=7, swa-last=3, logit-avg=3
#
# dupe-layers scaled for 12-layer model (encoder_layers=6 so valid range is [6, 12]):
#   original 30L: layers 15-21.  12L equivalent: layers 6-9 (same decoder fraction)

scales=(           "1m"    "10m"   "100m" )
num_epochs=(       2200    220     22     )
dupe_start_epoch=( 1400    140     14     )
swa_last_epochs=(  600     60      6      )
# logit_avg fixed at 3: averages last 3 epoch checkpoints regardless of scale.
# Proportional scaling (600/60/6) would write hundreds of GB of checkpoints.

IDX=${SLURM_ARRAY_TASK_ID}
SCALE=${scales[$IDX]}
RUN_NAME="slowrun-full-tricks-d12|data=${SCALE}"

torchrun --standalone --nproc_per_node=2 train.py \
  --run-name "${RUN_NAME}" \
  \
  --n_layer 12 --n_head 6 --n_embd 768 \
  \
  --input_bin    "${DATA_ROOT}/fineweb_${SCALE}/fineweb_train.pt" \
  --input_val_bin "${DATA_ROOT}/fineweb_${SCALE}/fineweb_val.pt" \
  \
  --num-epochs       ${num_epochs[$IDX]} \
  --total-batch-size 16384 \
  --device-batch-size 2 \
  --dupe-start-epoch ${dupe_start_epoch[$IDX]} \
  --dupe-layers-start 6 \
  --dupe-layers-end   9 \
  --swa-last-epochs  ${swa_last_epochs[$IDX]} \
  --logit-avg        3 \
  \
  --wandb_group "slowrun_ebt_comparison"

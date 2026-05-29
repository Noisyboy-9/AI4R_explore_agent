#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# PPO hyperparameter sweep for AI4R_explore_agent
#
# Hardware:
#   - 2 GPUs
#   - 110 CPU cores
#
# Dispatch policy:
#   - 3 concurrent jobs per GPU
#   - 6 concurrent jobs total
#   - CPU scheduling handled by Linux/Ray
#   - No taskset / no explicit CPU pinning
#   - Each job requests 0.33 GPU
# ============================================================

ITERATIONS=(500 1000 1500 2000)
TRAIN_BATCH_SIZES=(500 1000 1500)
NUM_SGD_ITERS=(10 15 20)
ENTROPY_VALUES=(0.0 0.1 0.05 0.2 0.3)

GPUS=(0 1)
JOBS_PER_GPU=3

TOTAL_SLOTS=$((${#GPUS[@]} * JOBS_PER_GPU))

# 110 cores / 6 concurrent jobs ≈ 18 cores per job.
# We use 17 rollout workers per job and leave room for the driver process,
# Ray overhead, logging, and the OS scheduler.
NUM_WORKERS=18

SGD_MINIBATCH_SIZE=256

# Three concurrent jobs will share each physical GPU.
NUM_GPUS_PER_JOB=0.33

BASE_CHECKPOINT_DIR="tmp/hparam_sweep"
LOG_DIR="logs/hparam_sweep"
QUEUE_DIR=".gpu_job_queues"

mkdir -p "$BASE_CHECKPOINT_DIR" "$LOG_DIR" "$QUEUE_DIR"

rm -f "$QUEUE_DIR"/slot_*.txt

for slot in $(seq 0 $((TOTAL_SLOTS - 1))); do
  touch "$QUEUE_DIR/slot_${slot}.txt"
done

echo "============================================================"
echo "Creating PPO hyperparameter sweep"
echo "GPUs: ${GPUS[*]}"
echo "Jobs per GPU: $JOBS_PER_GPU"
echo "Total concurrent jobs: $TOTAL_SLOTS"
echo "Rollout workers per job: $NUM_WORKERS"
echo "GPU requested per job: $NUM_GPUS_PER_JOB"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# Build slot queues using round-robin assignment
# ------------------------------------------------------------

job_id=0

for iter in "${ITERATIONS[@]}"; do
  for batch in "${TRAIN_BATCH_SIZES[@]}"; do
    for sgd_iter in "${NUM_SGD_ITERS[@]}"; do
      for entropy in "${ENTROPY_VALUES[@]}"; do

        slot=$((job_id % TOTAL_SLOTS))

        echo "$iter $batch $sgd_iter $entropy $job_id" >>"$QUEUE_DIR/slot_${slot}.txt"

        job_id=$((job_id + 1))

      done
    done
  done
done

echo "Created $job_id jobs."
echo ""

# ------------------------------------------------------------
# Run one slot queue
# Each slot is assigned to one GPU.
# CPU placement is left to the OS/Ray scheduler.
# ------------------------------------------------------------

run_slot_queue() {
  local slot="$1"

  local gpu_index=$((slot / JOBS_PER_GPU))
  local gpu="${GPUS[$gpu_index]}"

  local queue_file="$QUEUE_DIR/slot_${slot}.txt"

  echo "Starting slot $slot | GPU $gpu"

  while read -r iter batch sgd_iter entropy job_id; do

    entropy_tag="${entropy/./_}"

    run_name="job_${job_id}_iter_${iter}_batch_${batch}_sgd_${sgd_iter}_entropy_${entropy_tag}"
    checkpoint_dir="${BASE_CHECKPOINT_DIR}/${run_name}"
    log_file="${LOG_DIR}/${run_name}.log"

    mkdir -p "$checkpoint_dir"

    echo "============================================================"
    echo "Slot $slot | GPU $gpu"
    echo "Starting: $run_name"
    echo "Log: $log_file"
    echo "Checkpoint: $checkpoint_dir"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      python run_assignment.py train \
      --iterations "$iter" \
      --train-batch-size "$batch" \
      --sgd-minibatch-size "$SGD_MINIBATCH_SIZE" \
      --num-sgd-iter "$sgd_iter" \
      --num-workers "$NUM_WORKERS" \
      --num-gpus "$NUM_GPUS_PER_JOB" \
      --entropy-coeff "$entropy" \
      --checkpoint-dir "$checkpoint_dir" \
      >"$log_file" 2>&1

    echo "Slot $slot | Finished: $run_name"

  done <"$queue_file"

  echo "Slot $slot finished."
}

# ------------------------------------------------------------
# Start all slots
# ------------------------------------------------------------

for slot in $(seq 0 $((TOTAL_SLOTS - 1))); do
  run_slot_queue "$slot" &
done

wait

echo ""
echo "All hyperparameter sweep jobs finished."

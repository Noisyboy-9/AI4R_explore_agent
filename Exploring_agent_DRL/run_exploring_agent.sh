#!/usr/bin/env bash
set -u

ITERATIONS=(500 1000 1500 2000)
TRAIN_BATCH_SIZES=(2000 3000 4000)
NUM_SGD_ITERS=(3 5 10)
ENTROPY_VALUES=(0.0 0.1 0.05 0.2 0.3)

GPUS=(0 1)
MAX_JOBS=6

NUM_WORKERS=18
SGD_MINIBATCH_SIZE=256
NUM_GPUS_PER_JOB=0.33

BASE_CHECKPOINT_DIR="tmp/hparam_sweep"
LOG_DIR="logs/hparam_sweep"

mkdir -p "$BASE_CHECKPOINT_DIR" "$LOG_DIR"

active_jobs=0
job_id=0
failed_jobs=0

wait_for_one_job() {
  if ! wait -n; then
    failed_jobs=$((failed_jobs + 1))
  fi

  active_jobs=$((active_jobs - 1))
}

run_job() {
  local iter="$1"
  local batch="$2"
  local sgd_iter="$3"
  local entropy="$4"
  local job_id="$5"
  local gpu="$6"
  local entropy_tag="${entropy/./_}"
  local run_name="job_${job_id}_iter_${iter}_batch_${batch}_sgd_${sgd_iter}_entropy_${entropy_tag}"
  local checkpoint_dir="${BASE_CHECKPOINT_DIR}/${run_name}"
  local log_file="${LOG_DIR}/${run_name}.log"

  mkdir -p "$checkpoint_dir"

  echo "Starting ${run_name} on GPU ${gpu}"

  (
    CUDA_VISIBLE_DEVICES="$gpu" \
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
  ) &

  active_jobs=$((active_jobs + 1))
}

echo "Starting PPO hyperparameter sweep with up to ${MAX_JOBS} concurrent jobs."

for iter in "${ITERATIONS[@]}"; do
  for batch in "${TRAIN_BATCH_SIZES[@]}"; do
    for sgd_iter in "${NUM_SGD_ITERS[@]}"; do
      for entropy in "${ENTROPY_VALUES[@]}"; do
        while [ "$active_jobs" -ge "$MAX_JOBS" ]; do
          wait_for_one_job
        done

        gpu="${GPUS[$((job_id % ${#GPUS[@]}))]}"
        run_job "$iter" "$batch" "$sgd_iter" "$entropy" "$job_id" "$gpu"
        job_id=$((job_id + 1))
      done
    done
  done
done

while [ "$active_jobs" -gt 0 ]; do
  wait_for_one_job
done

echo "Submitted ${job_id} jobs."
echo "Jobs with non-zero exit status: ${failed_jobs}"

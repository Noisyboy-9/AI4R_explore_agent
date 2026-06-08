#!/usr/bin/env bash
set -euo pipefail

FRIEND_NAME="Ahmet"
FRIEND_SLUG="ahmet"
BRANCH_NAME="second-pass/${FRIEND_SLUG}"
ENTROPY_VALUES=(0.0 0.05)

ITERATIONS=500
TRAIN_BATCH_SIZE=4000
SGD_MINIBATCH_SIZE=256
NUM_SGD_ITER=10
NUM_WORKERS=60
MAX_JOBS=2
NUM_GPUS_PER_JOB=0
POLL_INTERVAL_SECONDS=5
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINING_DIR="Exploring_agent_DRL"
REPO_ROOT="$(pwd)"
BASE_CHECKPOINT_DIR="${REPO_ROOT}/tmp/hparam_sweep/second-pass/${FRIEND_SLUG}"
LOG_DIR="${REPO_ROOT}/logs/second-pass/${FRIEND_SLUG}"
FINISHED_JOBS_FILE="${LOG_DIR}/finished_jobs.txt"

if [ ! -d "${TRAINING_DIR}" ]; then
  echo "Run this script from the repository root." >&2
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Current directory is not a git repository." >&2
  exit 1
fi

if [ "$(git branch --show-current)" != "${BRANCH_NAME}" ]; then
  if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
    git switch "${BRANCH_NAME}"
  else
    git switch -c "${BRANCH_NAME}"
  fi
fi

mkdir -p "${BASE_CHECKPOINT_DIR}" "${LOG_DIR}"
touch "${FINISHED_JOBS_FILE}"

SKIP_JOB_IDS=()
while IFS= read -r line; do
  if [ -n "${line}" ]; then
    SKIP_JOB_IDS+=("${line}")
  fi
done < "${FINISHED_JOBS_FILE}"

RUNNING_PIDS=()
RUNNING_JOB_IDS=()
failed_jobs=0
submitted_jobs=0
job_id=0

should_skip_job() {
  local candidate="$1"
  local skipped_job_id

  for skipped_job_id in "${SKIP_JOB_IDS[@]}"; do
    if [ "${candidate}" = "${skipped_job_id}" ]; then
      return 0
    fi
  done

  return 1
}

record_finished_job() {
  local completed_job_id="$1"

  SKIP_JOB_IDS+=("${completed_job_id}")
  printf '%s\n' "${completed_job_id}" >> "${FINISHED_JOBS_FILE}"
}

remove_running_job() {
  local index="$1"

  unset 'RUNNING_PIDS[index]'
  unset 'RUNNING_JOB_IDS[index]'
  RUNNING_PIDS=("${RUNNING_PIDS[@]}")
  RUNNING_JOB_IDS=("${RUNNING_JOB_IDS[@]}")
}

wait_for_one_job() {
  local index
  local pid
  local completed_job_id

  while true; do
    for ((index = 0; index < ${#RUNNING_PIDS[@]}; index++)); do
      pid="${RUNNING_PIDS[$index]}"
      if ! kill -0 "${pid}" 2>/dev/null; then
        if ! wait "${pid}"; then
          failed_jobs=$((failed_jobs + 1))
        fi

        completed_job_id="${RUNNING_JOB_IDS[$index]}"
        record_finished_job "${completed_job_id}"
        remove_running_job "${index}"
        return
      fi
    done

    sleep "${POLL_INTERVAL_SECONDS}"
  done
}

run_job() {
  local entropy="$1"
  local current_job_id="$2"
  local entropy_tag="${entropy/./_}"
  local run_name="job_${current_job_id}_iter_${ITERATIONS}_batch_${TRAIN_BATCH_SIZE}_sgd_${NUM_SGD_ITER}_entropy_${entropy_tag}"
  local checkpoint_dir="${BASE_CHECKPOINT_DIR}/${run_name}"
  local log_file="${LOG_DIR}/${run_name}.log"

  mkdir -p "${checkpoint_dir}"

  echo "Starting ${run_name} for ${FRIEND_NAME}"

  (
    cd "${TRAINING_DIR}"
    PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" run_assignment.py train \
      --iterations "${ITERATIONS}" \
      --train-batch-size "${TRAIN_BATCH_SIZE}" \
      --sgd-minibatch-size "${SGD_MINIBATCH_SIZE}" \
      --num-sgd-iter "${NUM_SGD_ITER}" \
      --num-workers "${NUM_WORKERS}" \
      --num-gpus "${NUM_GPUS_PER_JOB}" \
      --entropy-coeff "${entropy}" \
      --checkpoint-dir "${checkpoint_dir}" \
      >"${log_file}" 2>&1
  ) &

  RUNNING_PIDS+=("$!")
  RUNNING_JOB_IDS+=("${current_job_id}")
}

echo "Starting second-pass entropy sweep for ${FRIEND_NAME} with up to ${MAX_JOBS} concurrent jobs."

for entropy in "${ENTROPY_VALUES[@]}"; do
  if should_skip_job "${job_id}"; then
    echo "Skipping previously completed job_${job_id}"
    job_id=$((job_id + 1))
    continue
  fi

  while [ "${#RUNNING_PIDS[@]}" -ge "${MAX_JOBS}" ]; do
    wait_for_one_job
  done

  run_job "${entropy}" "${job_id}"
  submitted_jobs=$((submitted_jobs + 1))
  job_id=$((job_id + 1))
done

while [ "${#RUNNING_PIDS[@]}" -gt 0 ]; do
  wait_for_one_job
done

echo "Enumerated ${job_id} jobs."
echo "Submitted ${submitted_jobs} jobs."
echo "Jobs with non-zero exit status: ${failed_jobs}"

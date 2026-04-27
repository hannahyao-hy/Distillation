#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

LINKED_OUTPUT_ROOT="${LINKED_OUTPUT_ROOT:-${REPO_ROOT}/datasets/vitra_gigahands_real_full_keypoints_brics001_cam0_linked}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LINKED_OUTPUT_ROOT}}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/datasets/gigahands_real/subset_manifest_full_keypoints_brics001_cam0.json}"
VIDEO_LIST="${VIDEO_LIST:-${REPO_ROOT}/datasets/gigahands_real/needed_videos_full_keypoints_brics001_cam0.txt}"
UNIQUE_VIDEO_LIST="${UNIQUE_VIDEO_LIST:-${REPO_ROOT}/datasets/gigahands_real/needed_videos_full_keypoints_brics001_cam0_unique.txt}"
CONFIG="${CONFIG:-${REPO_ROOT}/vitra/configs/human_pretrain_gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked.json}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_train}"
NUM_TRAIN="${NUM_TRAIN:-0}"
NUM_TEST="${NUM_TEST:-0}"
USE_ALL="${USE_ALL:-1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
CAMERA="${CAMERA:-brics-odroid-001_cam0}"
STRICT_CAMERA="${STRICT_CAMERA:-1}"

export LINKED_OUTPUT_ROOT OUTPUT_ROOT MANIFEST VIDEO_LIST UNIQUE_VIDEO_LIST CONFIG RUN_ROOT
export NUM_TRAIN NUM_TEST CAMERA

if [[ "${1:-}" == "" ]]; then
  exec env \
    USE_ALL="${USE_ALL}" \
    TEST_RATIO="${TEST_RATIO}" \
    bash "${REPO_ROOT}/scripts/run_gigahands_real_full_keypoints_pipeline.sh" help
fi

if [[ "${1}" == "prepare" || "${1}" == "prepare_all" ]]; then
  cd "${REPO_ROOT}"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  python tools/prepare_gigahands_real_subset.py \
    --gigahands_root "${REPO_ROOT}/datasets/gigahands_real" \
    --num_train "${NUM_TRAIN}" \
    --num_test "${NUM_TEST}" \
    --min_frames "${MIN_FRAMES:-32}" \
    --prefer_camera "${CAMERA}" \
    $(if [[ "${STRICT_CAMERA}" == "1" ]]; then echo "--strict_prefer_camera"; fi) \
    --require_keypoints \
    --require_real_keypoints \
    --require_video_exists \
    --candidate_pool_factor "${CANDIDATE_POOL_FACTOR:-1}" \
    --output_manifest "${MANIFEST}" \
    --output_video_list "${VIDEO_LIST}" \
    $(if [[ "${REQUIRE_BOTH_HANDS_VALID:-0}" == "1" ]]; then echo "--require_both_hands_valid"; fi) \
    $(if [[ "${PREFER_BIMANUAL_MOTION:-0}" == "1" ]]; then echo "--prefer_bimanual_motion"; fi) \
    $(if [[ "${REQUIRE_VIDEO_FRAME_COUNT:-0}" == "1" ]]; then echo "--require_video_frame_count"; fi) \
    $(if [[ "${USE_ALL}" == "1" ]]; then echo "--use_all --test_ratio ${TEST_RATIO}"; fi)
  if [[ "${1}" == "prepare" ]]; then
    exit 0
  fi
  STAGE=convert_linked bash "${REPO_ROOT}/scripts/run_gigahands_real_large_keypoints_pipeline.sh"
  STAGE=stats_keypoints bash "${REPO_ROOT}/scripts/run_gigahands_real_large_keypoints_pipeline.sh"
  exit 0
fi

STAGE="${1}" bash "${REPO_ROOT}/scripts/run_gigahands_real_large_keypoints_pipeline.sh"

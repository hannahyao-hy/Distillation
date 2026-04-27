# VLM-DISTILL

This folder collects the VLM/student distillation work for VLA-HAND without moving the original source files.

The files here are copied from their original repo locations so the distillation code, configs, tests, and notes are easy to inspect together. The original training entrypoints still live under `scripts/`, `vitra/`, and `tests/`; this folder is an organized snapshot, not a replacement Python package.

## Student Models

### Encoder student

- Model file: `models/vitra_encoder_student.py`
- Main config: `configs/vlm_distill_encoder_student_gigahands.json`
- Student design: DINOv2 vision encoder + DistilBERT text encoder + fusion projection to a 2304-d VITRA conditioning feature.
- Training objective: cognition-token distillation from the GigaHands VITRA teacher.
- Final local checkpoint: step 10000.

### ViTKD-style small student

- Model file: `models/vitra_small_paligemma_student.py`
- Main config: `configs/vlm_distill_small_vitra_vitkd_full_gigahands.json`
- Student design: smaller PaliGemma-style student.
- Training objective: ViTKD-style cognition, shallow mimic, and deep generation losses.
- Final local checkpoint: step 5000.

## Core Files

- `scripts/train_vlm_distill.py`: shared distillation trainer.
- `scripts/run_vlm_backbone_distill_gigahands.sh`: generic distillation launch wrapper.
- `scripts/run_small_vitra_vitkd_short_ablations.sh`: ViTKD ablation sweep.
- `scripts/evaluate_vlm_distill_gigahands.sh`: teacher/base/student action evaluation wrapper.
- `configs/`: copied distillation and teacher configs.
- `tests/test_vlm_distill.py`: unit tests for distillation behavior.
- `docs/CHECKPOINTS.md`: local checkpoint index.
- `docs/ViTKD.pdf`: reference paper copy.

## Training Commands

Run encoder-student cognition distillation from the repo root:

```bash
NPROC_PER_NODE=1 \
CONFIG=vitra/configs/vlm_distill_encoder_student_gigahands.json \
bash scripts/run_vlm_backbone_distill_gigahands.sh
```

Run the ViTKD-style small student:

```bash
CUDA_VISIBLE_DEVICES=7 \
torchrun --nproc_per_node=1 --standalone \
  scripts/train_vlm_distill.py \
  --config vitra/configs/vlm_distill_small_vitra_vitkd_full_gigahands.json
```

Evaluate a student checkpoint:

```bash
bash scripts/evaluate_vlm_distill_gigahands.sh \
  runs/vlm_distill_encoder_student_gigahands/checkpoints/vlm_distill_encoder_student_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=10000.ckpt
```

## Important Note

Checkpoint weights are not copied into this folder. They are multi-GB files under `runs/`, and `runs/` is ignored by git. See `docs/CHECKPOINTS.md` for exact local paths.

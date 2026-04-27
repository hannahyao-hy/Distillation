# Distillation

VLM backbone and diffusion-action distillation experiments for VLA-HAND on GigaHands.

This repository contains the code used to distill a GigaHands fine-tuned VITRA teacher into smaller or more targeted student models. The main teacher is a VITRA PaliGemma2-3B model fine-tuned on the BRICS camera 0 GigaHands keypoint dataset.

## What Is Included

- GigaHands camera 0 fine-tuning configs for VITRA.
- VLM cognition-token distillation from a frozen VITRA teacher.
- Encoder-only student model using DINOv2 + DistilBERT.
- Small PaliGemma student configs with ViTKD-style losses.
- Stage 2 action-head and joint distillation configs.
- Evaluation scripts for teacher/base/student action metrics.

Large local artifacts are intentionally not committed:

- `datasets/`
- `runs/`
- model checkpoints
- wandb offline logs

Those paths are ignored because the local training artifacts can be hundreds of GB.

## Main Teacher

Most distillation configs use this teacher checkpoint:

```text
runs/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_train/checkpoints/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_stage1_TB1_B1_bf16True/checkpoints/epoch=0-step=28000.ckpt/weights.pt
```

The teacher config is:

```text
vitra/configs/human_pretrain_gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked.json
```

Expected dataset root:

```text
datasets/vitra_gigahands_real_full_keypoints_brics001_cam0_linked
```

The base VITRA checkpoint is expected at:

```text
checkpoints/vitra-vla-3b.pt
```

In the local workspace this may be a symlink to a Hugging Face cache path.

## Distillation Variants

| Variant | Config | Student | Loss mode |
| --- | --- | --- | --- |
| Base cognition distill | `vitra/configs/vlm_distill_gigahands_cognition.json` | VITRA base model with selected trainable modules | `weighted` cognition loss |
| Base cognition + GT action | `vitra/configs/vlm_distill_gigahands_cognition_action_gt.json` | VITRA base model | cognition + ground-truth action |
| Encoder student | `vitra/configs/vlm_distill_encoder_student_gigahands.json` | DINOv2 vision encoder + DistilBERT text encoder + fusion projection | `weighted` cognition loss |
| Encoder student action | `vitra/configs/vlm_distill_encoder_student_gigahands_action.json` | Encoder student initialized from cognition distill | `action_only` |
| Encoder student joint | `vitra/configs/vlm_distill_encoder_student_gigahands_joint_normalized.json` | Encoder student initialized from cognition distill | `normalized` cognition + action |
| Small ViTKD full | `vitra/configs/vlm_distill_small_vitra_vitkd_full_gigahands.json` | Smaller PaliGemma-style student | `vitkd` |
| Small ViTKD large run | `vitra/configs/vlm_distill_small_vitra_vitkd_full_gigahands_large.json` | Smaller PaliGemma-style student | `vitkd` |

The encoder student is not the full ViTKD design. It is an encoder-only student that mimics the teacher cognition token. The ViTKD-style path is the small PaliGemma student family, which uses shallow mimic loss, deep generation loss, and cognition loss.

## Key Files

```text
scripts/train_vlm_distill.py
scripts/run_vlm_backbone_distill_gigahands.sh
scripts/run_small_vitra_vitkd_short_ablations.sh
scripts/run_small_vitra_vitkd_large.sh
scripts/evaluate_vlm_distill_gigahands.sh
scripts/evaluate_vlm_distill_cognition_sweep_gigahands.sh
```

Student model implementations:

```text
vitra/models/vla/vitra_encoder_student.py
vitra/models/vla/vitra_small_paligemma_student.py
```

Tests:

```text
tests/test_vlm_distill.py
tests/test_gigahands_real_subset_pipeline.py
```

## Training

Set the environment first:

```bash
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
```

Run the default cognition-token distillation:

```bash
NPROC_PER_NODE=8 \
CONFIG=vitra/configs/vlm_distill_gigahands_cognition.json \
bash scripts/run_vlm_backbone_distill_gigahands.sh
```

Run the encoder-only student:

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

Run the short ViTKD ablation suite:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_small_vitra_vitkd_short_ablations.sh
```

## Evaluation

Evaluate a distilled checkpoint against the GigaHands teacher and base VITRA model:

```bash
bash scripts/evaluate_vlm_distill_gigahands.sh \
  runs/vlm_distill_encoder_student_gigahands/checkpoints/vlm_distill_encoder_student_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=10000.ckpt
```

The evaluation script writes metrics under:

```text
runs/<distill-run>/action_eval/
```

For cognition checkpoint sweeps:

```bash
RUN_DIR=runs/vlm_distill_gigahands_cognition/checkpoints/vlm_distill_gigahands_cognition_base3b_TB2_B1_bf16True/checkpoints \
bash scripts/evaluate_vlm_distill_cognition_sweep_gigahands.sh
```

## Important Config Fields

- `teacher_config`: VITRA teacher architecture/config.
- `teacher_checkpoint`: frozen teacher weights.
- `student_init_checkpoint`: optional student initialization checkpoint.
- `distill_loss_mode`: one of `weighted`, `action_only`, `normalized`, or `vitkd`.
- `distill_train_setup.freeze_option`: controls which student modules are trainable.
- `action_loss_weight`: ground-truth action loss weight for joint training.
- `max_saved_checkpoints`: how many checkpoint directories to keep.

## Current Encoder-Student Run

The 10k encoder-student run used:

```text
vitra/configs/vlm_distill_encoder_student_gigahands.json
```

Main settings:

- Student: DINOv2-base vision encoder, DistilBERT text encoder, fusion MLP, 2304-d output.
- Teacher: GigaHands BRICS camera 0 VITRA checkpoint at step 28000.
- Loss: cognition-token MSE only.
- `action_loss_weight`: `0.0`.
- `max_steps`: `10000`.
- `total_batch_size`: `2`.
- `save_steps`: `1000`.
- Final local checkpoint: `epoch=0-step=10000.ckpt`.

The checkpoint itself is not committed to this repository.

## Notes

- GitHub does not accept normal git files above 100 MB, and this project has multi-GB checkpoints. Use local storage, Hugging Face, Git LFS, or an artifact store for model weights.
- Keep `runs/` and `datasets/` out of git unless a small metadata file is explicitly needed.
- The base `checkpoints/vitra-vla-3b.pt` path should be created locally before training or evaluation.

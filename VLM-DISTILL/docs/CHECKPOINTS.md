# Student Checkpoint Index

Checkpoint weights are not copied into `VLM-DISTILL/`. They stay in `runs/` because the files are too large for normal git storage.

## Encoder Student

Config:

```text
vitra/configs/vlm_distill_encoder_student_gigahands.json
```

Copied config:

```text
VLM-DISTILL/configs/vlm_distill_encoder_student_gigahands.json
```

Final checkpoint:

```text
runs/vlm_distill_encoder_student_gigahands/checkpoints/vlm_distill_encoder_student_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=10000.ckpt
```

Files:

```text
runs/vlm_distill_encoder_student_gigahands/checkpoints/vlm_distill_encoder_student_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=10000.ckpt/weights.pt
runs/vlm_distill_encoder_student_gigahands/checkpoints/vlm_distill_encoder_student_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=10000.ckpt/meta.json
```

Local retained checkpoints:

```text
epoch=0-step=8000.ckpt   weights.pt ~727 MB
epoch=0-step=9000.ckpt   weights.pt ~727 MB
epoch=0-step=10000.ckpt  weights.pt ~727 MB
```

Training log:

```text
runs/vlm_distill_encoder_student_gigahands/launch_logs/train_10k_tmux_20260427_0053.log
```

Summary:

- Student: DINOv2-base + DistilBERT encoder student.
- Objective: cognition-token MSE against the GigaHands VITRA teacher.
- Trained to: `10000` steps.
- Final logged loss: about `8e-05`.
- Action eval available locally only for the untrained probe run, not the final 10k checkpoint.

## ViTKD-Style Small Student

Config:

```text
vitra/configs/vlm_distill_small_vitra_vitkd_full_gigahands.json
```

Copied config:

```text
VLM-DISTILL/configs/vlm_distill_small_vitra_vitkd_full_gigahands.json
```

Final checkpoint:

```text
runs/vlm_distill_small_vitra_vitkd_full_gigahands/checkpoints/vlm_distill_small_vitra_vitkd_full_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=5000.ckpt
```

Files:

```text
runs/vlm_distill_small_vitra_vitkd_full_gigahands/checkpoints/vlm_distill_small_vitra_vitkd_full_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=5000.ckpt/weights.pt
runs/vlm_distill_small_vitra_vitkd_full_gigahands/checkpoints/vlm_distill_small_vitra_vitkd_full_gigahands_TB2_B1_bf16True/checkpoints/epoch=0-step=5000.ckpt/meta.json
```

Local retained checkpoints:

```text
epoch=0-step=3000.ckpt  weights.pt ~1.8 GB
epoch=0-step=3500.ckpt  weights.pt ~1.8 GB
epoch=0-step=4000.ckpt  weights.pt ~1.8 GB
epoch=0-step=4500.ckpt  weights.pt ~1.8 GB
epoch=0-step=5000.ckpt  weights.pt ~1.8 GB
```

Training log:

```text
runs/vlm_distill_small_vitra_vitkd_full_gigahands/train_5k_20260427_012738.log
```

Final action eval:

```text
runs/vlm_distill_small_vitra_vitkd_full_gigahands/action_eval/epoch=0-step=5000.ckpt/metrics.json
```

Summary:

- Student: small PaliGemma-style student.
- Objective: ViTKD-style cognition, shallow mimic, and deep generation losses.
- Trained to: `5000` steps.
- Final feature alignment to teacher:
  - `vlm_cognition_mse`: about `6.1e-05`
  - `vlm_cognition_cosine`: about `0.997`
- Action MSE remained poor in the 20-clip eval because the action head was not trainable in that run.

## Teacher Checkpoint

Most distillation configs use this teacher:

```text
runs/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_train/checkpoints/gigahands_real_full_keypoints_brics001_cam0_vitra3b_linked_stage1_TB1_B1_bf16True/checkpoints/epoch=0-step=28000.ckpt/weights.pt
```

Base VITRA checkpoint:

```text
checkpoints/vitra-vla-3b.pt
```

## Git Policy

Do not add `weights.pt`, `optimizer.pt`, or full `runs/` directories to git. Use this file as the tracked index and keep the actual artifacts local or upload them to an artifact store.

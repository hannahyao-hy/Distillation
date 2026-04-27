"""Inspect early full-GigaHands VITRA training samples for NaN/Inf tensors."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from vitra.datasets.dataset import (
    FrameDataset,
    MultipleDatasetWeightedDistributedBatchSampler,
    MultipleWeightedDataset,
)


def array_stats(value: object) -> tuple[np.ndarray, str]:
    arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if arr.dtype.kind not in "fc":
        return arr, f"shape={arr.shape} dtype={arr.dtype}"
    finite = np.isfinite(arr)
    nonfinite = int(arr.size - finite.sum())
    return (
        arr,
        "shape={} min={:.6g} max={:.6g} absmax={:.6g} nonfinite={}".format(
            arr.shape,
            float(np.nanmin(arr)),
            float(np.nanmax(arr)),
            float(np.nanmax(np.abs(arr))),
            nonfinite,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=Path("datasets/vitra_gigahands_real_full_keypoints_linked"))
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = FrameDataset(
        str(args.dataset_root),
        "gigahands_real_train",
        action_future_window_size=16,
        augmentation=False,
        normalization=True,
        processor=None,
        flip_augmentation=1.0,
        set_none_ratio=0.0,
        action_type="keypoints",
        use_rel=False,
        rel_mode="step",
        load_images=False,
        state_mask_prob=0.0,
        statistics_dataset_name="gigahands_real_train",
    )
    mixed = MultipleWeightedDataset([dataset], [1])
    dataset.episodic_dataset_core.set_global_data_statistics(dataset.episodic_dataset_core.data_statistics)
    sampler = MultipleDatasetWeightedDistributedBatchSampler(
        mixed,
        batch_size=1,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=args.seed,
    )
    order = sampler.prepare_indices()
    print(f"dataset_len={len(dataset)} order_len={len(order)}")

    keys = ["action_list", "current_state", "action_mask", "current_state_mask", "fov", "intrinsics"]
    for step in range(1, min(args.max_steps, len(order)) + 1):
        _, sample_idx = order[step - 1]
        raw_item = dataset.episodic_dataset_core.__getitem__(sample_idx)
        item = dataset.episodic_dataset_core.transform_trajectory(copy.deepcopy(raw_item), dataset.normalization)
        problems = []
        stats = {}
        for key in keys:
            arr, summary = array_stats(item[key])
            stats[key] = summary
            if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
                problems.append(key)
        if problems or step in {1, 55, 56, 57}:
            corr = dataset.episodic_dataset_core.index_frame_pair[sample_idx]
            episode_id = dataset.episodic_dataset_core.index_to_episode_id[corr[0]]
            print(f"\nstep={step} sample_idx={sample_idx} episode={episode_id} frame={int(corr[1])}")
            print(f"problems={problems}")
            for key in keys:
                print(f"{key}: {stats[key]}")
        if problems:
            raw_dataset = FrameDataset(
                str(args.dataset_root),
                "gigahands_real_train",
                action_future_window_size=16,
                augmentation=False,
                normalization=False,
                processor=None,
                flip_augmentation=1.0,
                set_none_ratio=0.0,
                action_type="keypoints",
                use_rel=False,
                rel_mode="step",
                load_images=False,
                state_mask_prob=0.0,
                statistics_dataset_name="gigahands_real_train",
            )
            raw_dataset.episodic_dataset_core.set_global_data_statistics(
                raw_dataset.episodic_dataset_core.data_statistics
            )
            raw = raw_dataset.episodic_dataset_core.__getitem__(sample_idx)
            raw_transformed = raw_dataset.episodic_dataset_core.transform_trajectory(
                copy.deepcopy(raw),
                raw_dataset.normalization,
            )
            print("\nraw_current_state_mask={}".format(np.asarray(raw["current_state_mask"]).tolist()))
            print("norm_current_state_mask={}".format(np.asarray(item["current_state_mask"]).tolist()))
            for label, sample in (("raw", raw), ("raw_transformed", raw_transformed), ("norm", item)):
                state = np.asarray(sample["current_state"])
                action = np.asarray(sample["action_list"])
                print(f"{label}_state_nonfinite_idx={np.where(~np.isfinite(state))[0].tolist()}")
                print(f"{label}_action_nonfinite_idx={np.argwhere(~np.isfinite(action)).tolist()[:40]}")
            break


if __name__ == "__main__":
    main()

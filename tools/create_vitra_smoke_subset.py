#!/usr/bin/env python3
"""Create a small episode-level VITRA-format subset for smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--train_name", default="gigahands_real_train")
    parser.add_argument("--test_name", default="gigahands_real_test")
    parser.add_argument("--num_train", type=int, default=8)
    parser.add_argument("--num_test", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean_output", action="store_true")
    return parser.parse_args()


def _sample_episode_ids(index_to_episode_id: np.ndarray, count: int, rng: np.random.Generator) -> list[str]:
    total = len(index_to_episode_id)
    if count < 0 or count >= total:
        selected = np.arange(total)
    else:
        selected = rng.choice(total, size=count, replace=False)
    return [str(index_to_episode_id[i]) for i in sorted(selected.tolist())]


def _copy_split(source_root: Path, output_root: Path, split_name: str, count: int, rng: np.random.Generator) -> dict:
    source_split = source_root / "Annotation" / split_name
    output_split = output_root / "Annotation" / split_name
    source_labels = source_split / "episodic_annotations"
    output_labels = output_split / "episodic_annotations"
    index_path = source_split / "episode_frame_index.npz"

    index_data = np.load(index_path, allow_pickle=True)
    index_frame_pair = index_data["index_frame_pair"]
    index_to_episode_id = index_data["index_to_episode_id"]
    selected_episode_ids = _sample_episode_ids(index_to_episode_id, count, rng)
    selected_id_set = set(selected_episode_ids)
    old_to_new = {episode_id: i for i, episode_id in enumerate(selected_episode_ids)}

    output_labels.mkdir(parents=True, exist_ok=True)
    for episode_id in selected_episode_ids:
        shutil.copy2(source_labels / f"{episode_id}.npy", output_labels / f"{episode_id}.npy")

    remapped_pairs = []
    for old_episode_idx, frame_id in index_frame_pair:
        episode_id = str(index_to_episode_id[int(old_episode_idx)])
        if episode_id in selected_id_set:
            remapped_pairs.append((old_to_new[episode_id], int(frame_id)))

    np.savez(
        output_split / "episode_frame_index.npz",
        index_frame_pair=np.asarray(remapped_pairs, dtype=np.int64),
        index_to_episode_id=np.asarray(selected_episode_ids),
    )

    report_path = source_split / "conversion_report.json"
    if report_path.exists():
        shutil.copy2(report_path, output_split / "conversion_report.json")

    return {
        "split": split_name,
        "episodes": len(selected_episode_ids),
        "frames": len(remapped_pairs),
        "episode_ids": selected_episode_ids,
    }


def _copy_optional(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    if args.clean_output and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    reports = [
        _copy_split(args.source_root, args.output_root, args.train_name, args.num_train, rng),
        _copy_split(args.source_root, args.output_root, args.test_name, args.num_test, rng),
    ]

    stats_root = args.output_root / "Annotation" / "statistics"
    stats_root.mkdir(parents=True, exist_ok=True)
    source_stats_root = args.source_root / "Annotation" / "statistics"
    for stats_path in source_stats_root.glob("*.json"):
        shutil.copy2(stats_path, stats_root / stats_path.name)

    for name in ("conversion_report.json", "subset_manifest.json", "gigahands_real_train_keypoints_weighted_statistics.json"):
        _copy_optional(args.source_root / name, args.output_root / name)

    source_video = args.source_root / "Video" / "GigaHands_root"
    output_video_parent = args.output_root / "Video"
    output_video_parent.mkdir(parents=True, exist_ok=True)
    output_video = output_video_parent / "GigaHands_root"
    if output_video.exists() or output_video.is_symlink():
        if output_video.is_dir() and not output_video.is_symlink():
            shutil.rmtree(output_video)
        else:
            output_video.unlink()
    os.symlink(source_video.resolve(), output_video)

    report = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "splits": reports,
    }
    (args.output_root / "smoke_subset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

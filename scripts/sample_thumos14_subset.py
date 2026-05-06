import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


TARGET_CLASSES = [
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build representative THUMOS14 validation/test subsets")
    parser.add_argument("--root", required=True, help="THUMOS14 root directory")
    parser.add_argument("--validation-count", type=int, default=200, help="Number of validation videos to sample")
    parser.add_argument("--test-count", type=int, default=200, help="Number of test videos to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        default="data/thumos14/subsets",
        help="Directory for validation_200.txt and test_200.txt",
    )
    return parser.parse_args()


def read_annotation_file(path: Path) -> List[Tuple[str, float, float]]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rows.append((parts[0], float(parts[1]), float(parts[2])))
    return rows


def load_subset_metadata(annotation_dir: Path, suffix: str, videos_dir: Path) -> Dict[str, Dict]:
    metadata: Dict[str, Dict] = {}

    def ensure(video_id: str) -> Dict:
        if video_id not in metadata:
            video_path = videos_dir / "{}.mp4".format(video_id)
            metadata[video_id] = {
                "video_id": video_id,
                "path": video_path,
                "size_bytes": video_path.stat().st_size if video_path.exists() else 0,
                "duration_proxy": 0.0,
                "classes": set(),
                "num_annotations": 0,
                "ambiguous": False,
            }
        return metadata[video_id]

    for class_name in TARGET_CLASSES:
        ann_path = annotation_dir / "{}_{}.txt".format(class_name, suffix)
        if not ann_path.exists():
            continue
        for video_id, start_time, end_time in read_annotation_file(ann_path):
            item = ensure(video_id)
            item["classes"].add(class_name)
            item["num_annotations"] += 1
            item["duration_proxy"] = max(item["duration_proxy"], end_time)

    ambiguous_path = annotation_dir / "Ambiguous_{}.txt".format(suffix)
    if ambiguous_path.exists():
        for video_id, start_time, end_time in read_annotation_file(ambiguous_path):
            item = ensure(video_id)
            item["ambiguous"] = True
            item["duration_proxy"] = max(item["duration_proxy"], end_time)

    return metadata


def split_into_three_buckets(values: Dict[str, float]) -> Dict[str, str]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    total = len(ordered)
    buckets = {}
    for index, (video_id, _value) in enumerate(ordered):
        ratio = index / max(1, total)
        if ratio < 1.0 / 3.0:
            buckets[video_id] = "low"
        elif ratio < 2.0 / 3.0:
            buckets[video_id] = "mid"
        else:
            buckets[video_id] = "high"
    return buckets


def score_video(meta: Dict) -> float:
    score = 0.0
    score += 4.0 if meta["ambiguous"] else 0.0
    score += min(5.0, float(meta["num_annotations"]) * 0.4)
    score += min(3.0, float(len(meta["classes"])) * 0.8)
    score += min(3.0, float(meta["duration_proxy"]) / 120.0)
    return score


def build_subset(metadata: Dict[str, Dict], sample_count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    selected: Set[str] = set()

    duration_buckets = split_into_three_buckets({video_id: meta["duration_proxy"] for video_id, meta in metadata.items()})
    density_buckets = split_into_three_buckets({video_id: meta["num_annotations"] for video_id, meta in metadata.items()})

    ambiguous_videos = [video_id for video_id, meta in metadata.items() if meta["ambiguous"]]
    ambiguous_videos.sort(key=lambda video_id: score_video(metadata[video_id]), reverse=True)
    selected.update(ambiguous_videos[: min(20, max(10, sample_count // 10))])

    for class_name in TARGET_CLASSES:
        candidates = [video_id for video_id, meta in metadata.items() if class_name in meta["classes"]]
        candidates.sort(key=lambda video_id: score_video(metadata[video_id]), reverse=True)
        for video_id in candidates[:3]:
            selected.add(video_id)

    for bucket_name in ("low", "mid", "high"):
        candidates = [video_id for video_id, bucket in duration_buckets.items() if bucket == bucket_name]
        candidates.sort(key=lambda video_id: score_video(metadata[video_id]), reverse=True)
        selected.update(candidates[: max(10, sample_count // 12)])

    for bucket_name in ("low", "mid", "high"):
        candidates = [video_id for video_id, bucket in density_buckets.items() if bucket == bucket_name]
        candidates.sort(key=lambda video_id: score_video(metadata[video_id]), reverse=True)
        selected.update(candidates[: max(10, sample_count // 12)])

    if len(selected) < sample_count:
        remaining = [video_id for video_id in metadata.keys() if video_id not in selected]
        remaining.sort(key=lambda video_id: (score_video(metadata[video_id]), rng.random()), reverse=True)
        for video_id in remaining:
            selected.add(video_id)
            if len(selected) >= sample_count:
                break

    selected_list = list(selected)
    rng.shuffle(selected_list)
    selected_list.sort(key=lambda video_id: (video_id.startswith("video_test"), video_id))
    return selected_list[:sample_count]


def write_list(path: Path, items: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item + "\n")


def summarize(name: str, picked: List[str], metadata: Dict[str, Dict]) -> None:
    class_coverage = defaultdict(int)
    ambiguous = 0
    for video_id in picked:
        meta = metadata[video_id]
        ambiguous += int(meta["ambiguous"])
        for class_name in meta["classes"]:
            class_coverage[class_name] += 1

    print("{} subset: {}".format(name, len(picked)))
    print("  ambiguous videos:", ambiguous)
    print("  class coverage:", len([c for c in TARGET_CLASSES if class_coverage[c] > 0]), "/", len(TARGET_CLASSES))
    preview = ", ".join("{}={}".format(class_name, class_coverage[class_name]) for class_name in TARGET_CLASSES[:8])
    print("  class counts preview:", preview)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir

    val_metadata = load_subset_metadata(
        annotation_dir=root / "TH14_Temporal_annotations_validation" / "annotation",
        suffix="val",
        videos_dir=root / "validation",
    )
    test_metadata = load_subset_metadata(
        annotation_dir=root / "TH14_Temporal_Annotations_Test" / "annotations" / "annotation",
        suffix="test",
        videos_dir=root / "TH14_test_set_mp4",
    )

    validation_pick = build_subset(val_metadata, args.validation_count, seed=args.seed)
    test_pick = build_subset(test_metadata, args.test_count, seed=args.seed + 1)

    validation_path = output_dir / "validation_{}.txt".format(args.validation_count)
    test_path = output_dir / "test_{}.txt".format(args.test_count)
    write_list(validation_path, validation_pick)
    write_list(test_path, test_pick)

    summarize("validation", validation_pick, val_metadata)
    summarize("test", test_pick, test_metadata)
    print("Saved:", validation_path)
    print("Saved:", test_path)


if __name__ == "__main__":
    main()

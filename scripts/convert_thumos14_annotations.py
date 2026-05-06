import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


THUMOS14_CLASSES = [
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
    parser = argparse.ArgumentParser(description="Convert THUMOS14 txt annotations to segments.json")
    parser.add_argument(
        "--annotation-dir",
        required=True,
        action="append",
        help="Directory containing THUMOS14 txt annotations. Repeat for validation/test annotation folders.",
    )
    parser.add_argument("--frames-dir", required=True, help="Directory containing extracted video frames")
    parser.add_argument("--output", required=True, help="Path to output thumos14_segments.json")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS when frame timestamps are unavailable")
    parser.add_argument(
        "--video-list",
        action="append",
        help="Optional txt file listing video ids to include. Repeat for validation/test subset lists.",
    )
    return parser.parse_args()


def infer_subset_from_name(filename: str) -> str:
    lower = filename.lower()
    if "_validation" in lower or "_val" in lower:
        return "train"
    if "_test" in lower:
        return "test"
    return "unknown"


def load_class_mapping(annotation_dir: Path) -> Dict[str, int]:
    detclass = annotation_dir / "detclasslist.txt"
    if detclass.exists():
        mapping: Dict[str, int] = {}
        with open(detclass, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                label_id = int(parts[0]) - 1
                label = parts[1]
                mapping[label] = label_id
        if mapping:
            return mapping
    return {label: index for index, label in enumerate(THUMOS14_CLASSES)}


def build_video_dir_index(frames_dir: Path) -> Dict[str, Path]:
    print("Indexing frame directories under", frames_dir)
    index: Dict[str, Path] = {}
    indexed = 0
    for child in frames_dir.iterdir():
        if not child.is_dir():
            continue

        # Preferred layout: frames/<video_id>/*.jpg
        if child.name.startswith("video_"):
            index.setdefault(child.name, child)
            indexed += 1
            if indexed % 200 == 0:
                print("  indexed {} video directories...".format(indexed))
            continue

        # Alternate layout: frames/<class_name>/<video_id>/*.jpg
        for maybe_video_dir in child.iterdir():
            if not maybe_video_dir.is_dir():
                continue
            if maybe_video_dir.name.startswith("video_"):
                index.setdefault(maybe_video_dir.name, maybe_video_dir)
                indexed += 1
                if indexed % 200 == 0:
                    print("  indexed {} video directories...".format(indexed))
    print("Indexed {} video directories.".format(indexed))
    return index


def count_frames(video_dir_index: Dict[str, Path], video_id: str) -> int:
    video_dir = video_dir_index.get(video_id)
    if video_dir is None:
        raise FileNotFoundError("Missing extracted frames for {}".format(video_id))
    count = len([p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if count <= 0:
        raise FileNotFoundError("No frame images found in {}".format(video_dir))
    return count


def parse_annotation_file(path: Path, class_mapping: Dict[str, int]) -> Tuple[str, str, List[Tuple[str, float, float]]]:
    basename = path.stem
    subset = infer_subset_from_name(path.name)
    match = re.match(r"(.+?)_(test|validation|val)$", basename, flags=re.IGNORECASE)
    if match:
        label = match.group(1)
    else:
        label = basename

    if label.lower() == "ambiguous":
        canonical_label = "Vague"
        label_id = len(THUMOS14_CLASSES) + 1
    else:
        canonical_label = label
        if canonical_label not in class_mapping:
            return subset, canonical_label, []
        label_id = class_mapping[canonical_label]

    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            video_id = parts[0]
            start_time = float(parts[1])
            end_time = float(parts[2])
            records.append((video_id, start_time, end_time, canonical_label, label_id))
    return subset, canonical_label, records


def load_video_allowlist(paths: List[str]) -> set:
    allowed = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                item = line.strip()
                if item:
                    allowed.add(item)
    return allowed


def build_segments(annotation_dirs: List[Path], frames_dir: Path, fps: float, allowed_videos: set = None) -> Dict:
    class_mapping = load_class_mapping(annotation_dirs[0])
    videos: Dict[str, Dict] = {}
    video_dir_index = build_video_dir_index(frames_dir)
    total_files = sum(len(list(annotation_dir.glob("*.txt"))) for annotation_dir in annotation_dirs)
    processed_files = 0
    for annotation_dir in annotation_dirs:
        annotation_files = sorted(annotation_dir.glob("*.txt"))
        for path in annotation_files:
            if path.name.lower() == "detclasslist.txt":
                continue
            processed_files += 1
            print("[{}/{}] parsing {}".format(processed_files, total_files, path.name))
            subset, _, records = parse_annotation_file(path, class_mapping)
            for video_id, start_time, end_time, label, label_id in records:
                if allowed_videos is not None and video_id not in allowed_videos:
                    continue
                if video_id not in videos:
                    videos[video_id] = {
                        "subset": subset,
                        "fps": fps,
                        "num_frames": count_frames(video_dir_index, video_id),
                        "annotations": [],
                    }
                videos[video_id]["annotations"].append(
                    {
                        "label": label,
                        "label_id": label_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_frame": int(round(start_time * fps)),
                        "end_frame": int(round(end_time * fps)),
                    }
                )
    for video in videos.values():
        video["annotations"].sort(key=lambda item: (item["start_frame"], item["end_frame"], item["label_id"]))
    return {
        "dataset": "thumos14",
        "fps": fps,
        "class_names": THUMOS14_CLASSES + ["Background", "Vague"],
        "videos": videos,
    }


def main() -> None:
    args = parse_args()
    annotation_dirs = [Path(item) for item in args.annotation_dir]
    frames_dir = Path(args.frames_dir)
    output = Path(args.output)
    allowed_videos = load_video_allowlist(args.video_list) if args.video_list else None
    built = build_segments(annotation_dirs=annotation_dirs, frames_dir=frames_dir, fps=args.fps, allowed_videos=allowed_videos)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(built, handle, indent=2)
    print("Saved segments to", output)
    print("Videos:", len(built["videos"]))


if __name__ == "__main__":
    main()

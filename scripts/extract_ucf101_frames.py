import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RGB frames for UCF101 with class/video directory layout")
    parser.add_argument("--video-root", required=True, help="Root directory containing UCF101 class folders")
    parser.add_argument("--output-root", required=True, help="Root directory for extracted RGB frames")
    parser.add_argument("--image-ext", default="jpg", help="Output image extension")
    parser.add_argument("--jpg-quality", type=int, default=90, help="JPEG quality when image-ext=jpg")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frame directories")
    parser.add_argument("--max-videos", type=int, default=0, help="Optional limit for preview extraction")
    parser.add_argument("--split-csv", help="Optional csv file to extract only listed videos")
    parser.add_argument(
        "--start-after",
        help="Resume after this exact video filename, for example v_ThrowDiscus_g14_c03.avi",
    )
    return parser.parse_args()


def iter_videos(video_root: Path):
    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    for class_dir in sorted(video_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for video_path in sorted(class_dir.iterdir()):
            if video_path.is_file() and video_path.suffix.lower() in exts:
                yield class_dir.name, video_path


def iter_videos_from_csv(video_root: Path, split_csv: Path, max_videos: int):
    yielded = 0
    with open(split_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clip_name = row["clip_name"]
            label = row["label"]
            inferred_label = infer_label_from_clip_name(clip_name) or label
            basename = "{}.avi".format(clip_name)
            candidates = [
                video_root / label / basename,
                video_root / inferred_label / basename,
            ]
            for candidate in candidates:
                if candidate.exists():
                    yield inferred_label, candidate
                    yielded += 1
                    break
            if max_videos > 0 and yielded >= max_videos:
                break


def infer_label_from_clip_name(clip_name: str):
    import re

    match = re.match(r"v_(.+?)_g\d+_c\d+$", clip_name)
    if match:
        return match.group(1)
    return None


def apply_start_after(iterator, start_after: str):
    if not start_after:
        for item in iterator:
            yield item
        return

    matched = False
    for class_name, video_path in iterator:
        if not matched:
            if video_path.name == start_after:
                matched = True
            continue
        yield class_name, video_path

    if not matched:
        raise ValueError("start-after target not found: {}".format(start_after))


def extract_one(video_path: Path, output_dir: Path, image_ext: str, jpg_quality: int, overwrite: bool) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for UCF101 frame extraction.") from exc

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        return len(list(output_dir.iterdir()))
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Failed to open video {}".format(video_path))

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        out_path = output_dir / "{:06d}.{}".format(frame_index, image_ext)
        params = []
        if image_ext.lower() in {"jpg", "jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
        cv2.imwrite(str(out_path), frame, params)
    capture.release()
    return frame_index


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    total_videos = 0
    total_frames = 0
    iterator = iter_videos(video_root)
    if args.split_csv:
        iterator = iter_videos_from_csv(video_root, Path(args.split_csv), args.max_videos)
    iterator = apply_start_after(iterator, args.start_after)

    for class_name, video_path in iterator:
        total_videos += 1
        target_dir = output_root / class_name / video_path.stem
        frames = extract_one(
            video_path=video_path,
            output_dir=target_dir,
            image_ext=args.image_ext,
            jpg_quality=args.jpg_quality,
            overwrite=args.overwrite,
        )
        total_frames += frames
        print("[{}/{}] {} -> {} frames".format(total_videos, args.max_videos or "all", video_path.name, frames))
        if args.max_videos > 0 and total_videos >= args.max_videos:
            break

    print("Videos processed:", total_videos)
    print("Frames extracted:", total_frames)
    print("Output root:", output_root)


if __name__ == "__main__":
    main()

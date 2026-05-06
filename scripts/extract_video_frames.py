import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RGB frames from videos for EdgeFlex")
    parser.add_argument("--videos-dir", required=True, help="Directory containing raw video files")
    parser.add_argument("--output-dir", required=True, help="Directory for extracted frames")
    parser.add_argument("--image-ext", default="jpg", help="Frame image extension")
    parser.add_argument("--video-list", help="Optional txt file listing video ids or filenames to extract")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames")
    parser.add_argument("--max-videos", type=int, default=0, help="Optional cap for smoke tests")
    parser.add_argument("--start-after", help="Resume after the given video filename")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip unreadable/corrupted videos and continue extraction",
    )
    parser.add_argument("--error-log", help="Optional path to append extraction errors")
    return parser.parse_args()


def iter_videos(videos_dir: Path):
    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    for path in sorted(videos_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def load_video_filter(path: str):
    if not path:
        return None
    selected = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if not item:
                continue
            selected.add(item)
            selected.add(Path(item).stem)
    return selected


def extract_one(video_path: Path, output_dir: Path, image_ext: str, overwrite: bool) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for frame extraction. Install it in your py3.10 environment first."
        ) from exc

    video_name = video_path.stem
    target_dir = output_dir / video_name
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        return len(list(target_dir.iterdir()))
    target_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Failed to open video {}".format(video_path))

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        out_path = target_dir / "{:06d}.{}".format(frame_index, image_ext)
        cv2.imwrite(str(out_path), frame)
    capture.release()
    return frame_index


def main() -> None:
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = Path(args.error_log) if args.error_log else None
    video_filter = load_video_filter(args.video_list)

    started = args.start_after is None
    errors = []

    total_videos = 0
    total_frames = 0
    for video_path in iter_videos(videos_dir):
        if video_filter is not None and video_path.name not in video_filter and video_path.stem not in video_filter:
            continue
        if not started:
            if video_path.name == args.start_after:
                started = True
            continue
        if args.max_videos > 0 and total_videos >= args.max_videos:
            break
        total_videos += 1
        try:
            count = extract_one(video_path, output_dir, args.image_ext, args.overwrite)
            total_frames += count
            print("Extracted", count, "frames from", video_path.name)
        except Exception as exc:
            message = "{} | {}".format(video_path.name, exc)
            errors.append(message)
            print("ERROR:", message)
            if error_log_path is not None:
                error_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(error_log_path, "a", encoding="utf-8") as handle:
                    handle.write(message + "\n")
            if not args.continue_on_error:
                raise

    print("Videos processed:", total_videos)
    print("Frames extracted:", total_frames)
    if errors:
        print("Errors:", len(errors))


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from edgeflex.analysis.real_streaming import build_inference_cache, load_segments, simulate_streaming_from_cache
from edgeflex.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real THUMOS14 offline/streaming evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path for the evaluated model")
    parser.add_argument(
        "--segments",
        default="data/thumos14/annotations/thumos14_segments.json",
        help="Path to THUMOS14 temporal segment JSON",
    )
    parser.add_argument("--subset", default="test", help="Subset to evaluate")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS for temporal metrics")
    parser.add_argument("--max-videos", type=int, default=0, help="Optional video cap for smoke tests")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--cache-path", help="Optional cache path for evaluated model inference records")
    parser.add_argument("--baseline-config", help="Optional baseline config for speed-ratio reference")
    parser.add_argument("--baseline-checkpoint", help="Optional baseline checkpoint for speed-ratio reference")
    parser.add_argument("--baseline-cache-path", help="Optional cache path for baseline inference records")
    return parser.parse_args()


def maybe_resolve_output(path: str) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = Path(ROOT) / output
    return output


def maybe_resolve_dataset_root(path: str) -> Path:
    root = Path(path)
    if not root.is_absolute():
        root = Path(ROOT) / root
    return root


def default_cache_path(config, checkpoint: str, subset: str, suffix: str) -> str:
    dataset_root = maybe_resolve_dataset_root(config.dataset.root)
    cache_dir = dataset_root.parent / "edgeflex_cache"
    checkpoint_stem = Path(checkpoint).stem
    filename = "{}_{}_{}{}".format(
        config.experiment_name,
        subset,
        checkpoint_stem,
        suffix,
    )
    return str(cache_dir / filename)


def evaluate(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    segments = load_segments(args.segments)
    cache_path = args.cache_path
    if not cache_path:
        cache_path = default_cache_path(config, args.checkpoint, args.subset, "_cache.pkl")
    cache = build_inference_cache(
        config,
        args.checkpoint,
        subset=args.subset,
        max_videos=args.max_videos,
        cache_path=cache_path,
    )

    baseline_fps = None
    baseline_cache_path = None
    if args.baseline_config and args.baseline_checkpoint:
        baseline_config = load_config(args.baseline_config)
        baseline_cache_path = args.baseline_cache_path
        if not baseline_cache_path:
            baseline_cache_path = default_cache_path(
                baseline_config,
                args.baseline_checkpoint,
                args.subset,
                "_baseline_cache.pkl",
            )
        baseline_cache = build_inference_cache(
            baseline_config,
            args.baseline_checkpoint,
            subset=args.subset,
            max_videos=args.max_videos,
            cache_path=baseline_cache_path,
        )
        baseline_result = simulate_streaming_from_cache(
            baseline_cache,
            baseline_config,
            segments,
            subset=args.subset,
            fps=args.fps,
        )
        baseline_fps = baseline_result["fps"]

    result = simulate_streaming_from_cache(
        cache,
        config,
        segments,
        subset=args.subset,
        fps=args.fps,
        baseline_fps=baseline_fps,
    )
    result["checkpoint"] = args.checkpoint
    if cache_path:
        result["cache_path"] = str(maybe_resolve_output(cache_path))
    if args.baseline_checkpoint:
        result["baseline_checkpoint"] = args.baseline_checkpoint
        if baseline_cache_path:
            result["baseline_cache_path"] = str(maybe_resolve_output(baseline_cache_path))
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2))
    if args.output:
        output = maybe_resolve_output(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("Saved:", output)


if __name__ == "__main__":
    main()

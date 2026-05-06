import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Set, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from edgeflex.analysis.real_streaming import load_inference_cache, load_segments, simulate_streaming_from_cache
from edgeflex.config import ExperimentConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped cross-scenario analysis using existing THUMOS14 caches")
    parser.add_argument("--b0-config", required=True, help="B0 YAML config")
    parser.add_argument("--b0-cache", required=True, help="B0 inference cache")
    parser.add_argument("--b1-config", required=True, help="B1 YAML config")
    parser.add_argument("--b1-cache", required=True, help="B1 inference cache")
    parser.add_argument(
        "--segments",
        default="data/thumos14/annotations/thumos14_segments_subset200.json",
        help="Path to THUMOS14 segment JSON",
    )
    parser.add_argument("--subset", default="test", help="Subset to analyze")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument(
        "--output-dir",
        default="results/grouped_generalization/b1_subset200",
        help="Directory for grouped analysis outputs",
    )
    return parser.parse_args()


def maybe_resolve(path: str) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = Path(ROOT) / output
    return output


def merged_coverage(intervals: Sequence[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted((int(start), int(end)) for start, end in intervals if end >= start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start + 1
        current_start, current_end = start, end
    total += current_end - current_start + 1
    return total


def build_video_stats(segments: Dict, subset: str, fps: float) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for video_id, meta in segments["videos"].items():
        if meta.get("subset") != subset:
            continue
        num_frames = int(meta["num_frames"])
        duration_seconds = float(num_frames) / max(fps, 1e-6)
        target_intervals: List[Tuple[int, int]] = []
        vague_intervals: List[Tuple[int, int]] = []
        target_count = 0
        for ann in meta.get("annotations", []):
            interval = (int(ann["start_frame"]), int(ann["end_frame"]))
            if ann["label"] == "Vague":
                vague_intervals.append(interval)
            else:
                target_intervals.append(interval)
                target_count += 1
        target_frames = merged_coverage(target_intervals)
        vague_frames = merged_coverage(vague_intervals)
        occupied_frames = merged_coverage(target_intervals + vague_intervals)
        background_ratio = max(0.0, float(num_frames - occupied_frames) / max(float(num_frames), 1.0))
        ambiguous_ratio = float(vague_frames) / max(float(num_frames), 1.0)
        target_ratio = float(target_frames) / max(float(num_frames), 1.0)
        action_density = float(target_count) / max(duration_seconds / 60.0, 1e-6)
        stats[video_id] = {
            "duration_seconds": duration_seconds,
            "target_count": float(target_count),
            "target_ratio": target_ratio,
            "background_ratio": background_ratio,
            "ambiguous_ratio": ambiguous_ratio,
            "action_density": action_density,
        }
    return stats


def split_by_metric(video_stats: Dict[str, Dict[str, float]], metric: str) -> Tuple[Set[str], Set[str], float]:
    ordered = sorted(video_stats.items(), key=lambda item: (item[1][metric], item[0]))
    midpoint = len(ordered) // 2
    lower = {video_id for video_id, _ in ordered[:midpoint]}
    upper = {video_id for video_id, _ in ordered[midpoint:]}
    threshold = median([payload[metric] for _, payload in ordered]) if ordered else 0.0
    return lower, upper, float(threshold)


def filtered_segments(segments: Dict, subset: str, video_ids: Set[str]) -> Dict:
    payload = {
        "dataset": segments.get("dataset"),
        "fps": segments.get("fps"),
        "class_names": segments.get("class_names", []),
        "videos": {},
    }
    for video_id, meta in segments["videos"].items():
        if meta.get("subset") != subset or video_id not in video_ids:
            continue
        payload["videos"][video_id] = meta
    return payload


def filtered_cache(cache: Dict, video_ids: Set[str]) -> Dict:
    records = [record for record in cache["records"] if record.video_id in video_ids]
    total_records = max(1, len(cache["records"]))
    ratio = float(len(records)) / float(total_records)
    return {
        "records": records,
        "subset": cache.get("subset"),
        "inference_seconds": float(cache.get("inference_seconds", 0.0) or 0.0) * ratio,
        "preprocessing_seconds": float(cache.get("preprocessing_seconds", 0.0) or 0.0) * ratio,
        "checkpoint": cache.get("checkpoint"),
        "experiment_name": cache.get("experiment_name"),
        "num_videos": len(video_ids),
        "max_videos": 0,
        "completed_video_ids": sorted(video_ids),
    }


def compare_row(
    group_name: str,
    metric_name: str,
    threshold: float,
    video_ids: Set[str],
    b0_result: Dict[str, float],
    b1_result: Dict[str, float],
) -> Dict[str, float]:
    processed_reduction = 0.0
    if float(b0_result["processed_windows"]) > 0:
        processed_reduction = 1.0 - (
            float(b1_result["processed_windows"]) / float(b0_result["processed_windows"])
        )
    return {
        "group_name": group_name,
        "group_metric": metric_name,
        "group_threshold": threshold,
        "num_videos": len(video_ids),
        "frame_by_frame_p_mAP_mean": b0_result["p_mAP_mean"],
        "flexible_sliding_p_mAP_mean": b1_result["p_mAP_mean"],
        "p_mAP_delta": b1_result["p_mAP_mean"] - b0_result["p_mAP_mean"],
        "frame_by_frame_fps": b0_result["fps"],
        "flexible_sliding_fps": b1_result["fps"],
        "fps_gain": b1_result["fps"] - b0_result["fps"],
        "speed_ratio": b1_result["speed_ratio"],
        "frame_by_frame_delay": b0_result["average_delay_seconds"],
        "flexible_sliding_delay": b1_result["average_delay_seconds"],
        "delay_delta": b1_result["average_delay_seconds"] - b0_result["average_delay_seconds"],
        "frame_by_frame_step": b0_result["average_sliding_step"],
        "flexible_sliding_step": b1_result["average_sliding_step"],
        "flexible_sliding_skipped_window_ratio": b1_result["skipped_window_ratio"],
        "frame_by_frame_processed_windows": b0_result["processed_windows"],
        "flexible_sliding_processed_windows": b1_result["processed_windows"],
        "processed_windows_reduction": processed_reduction,
    }


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: List[Dict[str, float]], columns: Sequence[str]) -> str:
    percent_columns = {
        "frame_by_frame_p_mAP_mean",
        "flexible_sliding_p_mAP_mean",
        "p_mAP_delta",
        "flexible_sliding_skipped_window_ratio",
        "processed_windows_reduction",
    }
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells: List[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if column in percent_columns:
                    cells.append("{:.2f}%".format(value * 100.0))
                else:
                    cells.append("{:.4f}".format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    b0_config: ExperimentConfig = load_config(args.b0_config)
    b1_config: ExperimentConfig = load_config(args.b1_config)
    segments = load_segments(args.segments)
    b0_cache = load_inference_cache(args.b0_cache)
    b1_cache = load_inference_cache(args.b1_cache)

    output_dir = maybe_resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_stats = build_video_stats(segments, subset=args.subset, fps=args.fps)
    group_specs = [
        ("action_density", "action_sparse", "action_dense"),
        ("background_ratio", "low_background", "high_background"),
        ("duration_seconds", "short_duration", "long_duration"),
        ("ambiguous_ratio", "ambiguous_light", "ambiguous_rich"),
    ]

    rows: List[Dict[str, float]] = []
    summary_groups: List[Dict[str, object]] = []
    for metric_name, low_name, high_name in group_specs:
        lower_ids, upper_ids, threshold = split_by_metric(video_stats, metric_name)
        group_pairs = [(low_name, lower_ids), (high_name, upper_ids)]
        summary_groups.append(
            {
                "metric": metric_name,
                "threshold": threshold,
                "lower_group": low_name,
                "lower_count": len(lower_ids),
                "upper_group": high_name,
                "upper_count": len(upper_ids),
            }
        )
        for group_name, video_ids in group_pairs:
            group_segments = filtered_segments(segments, subset=args.subset, video_ids=video_ids)
            group_b0_cache = filtered_cache(b0_cache, video_ids)
            group_b1_cache = filtered_cache(b1_cache, video_ids)
            print("Evaluating group {} on {} videos".format(group_name, len(video_ids)))
            b0_result = simulate_streaming_from_cache(
                group_b0_cache,
                b0_config,
                group_segments,
                subset=args.subset,
                fps=args.fps,
            )
            b1_result = simulate_streaming_from_cache(
                group_b1_cache,
                b1_config,
                group_segments,
                subset=args.subset,
                fps=args.fps,
                baseline_fps=b0_result["fps"],
            )
            rows.append(compare_row(group_name, metric_name, threshold, video_ids, b0_result, b1_result))

    fieldnames = [
        "group_name",
        "group_metric",
        "group_threshold",
        "num_videos",
        "frame_by_frame_p_mAP_mean",
        "flexible_sliding_p_mAP_mean",
        "p_mAP_delta",
        "frame_by_frame_fps",
        "flexible_sliding_fps",
        "fps_gain",
        "speed_ratio",
        "frame_by_frame_delay",
        "flexible_sliding_delay",
        "delay_delta",
        "frame_by_frame_step",
        "flexible_sliding_step",
        "flexible_sliding_skipped_window_ratio",
        "frame_by_frame_processed_windows",
        "flexible_sliding_processed_windows",
        "processed_windows_reduction",
    ]
    csv_path = output_dir / "grouped_generalization.csv"
    write_csv(csv_path, rows, fieldnames)

    report_columns = [
        "group_name",
        "num_videos",
        "frame_by_frame_p_mAP_mean",
        "flexible_sliding_p_mAP_mean",
        "p_mAP_delta",
        "frame_by_frame_fps",
        "flexible_sliding_fps",
        "fps_gain",
        "speed_ratio",
        "frame_by_frame_delay",
        "flexible_sliding_delay",
        "delay_delta",
        "flexible_sliding_skipped_window_ratio",
        "processed_windows_reduction",
    ]
    report_lines = [
        "# Grouped Generalization Report",
        "",
        "Subset: `{}`".format(args.subset),
        "B0 config: `{}`".format(args.b0_config),
        "B1 config: `{}`".format(args.b1_config),
        "",
        "## Split Summary",
        "",
        json.dumps(summary_groups, indent=2),
        "",
        "## Grouped Results",
        "",
        markdown_table(rows, report_columns),
        "",
    ]
    (output_dir / "grouped_generalization_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    with open(output_dir / "grouped_generalization_report.json", "w", encoding="utf-8") as handle:
        json.dump({"splits": summary_groups, "rows": rows}, handle, indent=2)

    print("Saved grouped report to", output_dir)


if __name__ == "__main__":
    main()

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from edgeflex.analysis.real_streaming import build_inference_cache, load_segments, simulate_streaming_from_cache
from edgeflex.config import ExperimentConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real THUMOS14 threshold sensitivity evaluation")
    parser.add_argument("--config", required=True, help="Base YAML config, usually B1")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint for the evaluated recognizer")
    parser.add_argument(
        "--segments",
        default="data/thumos14/annotations/thumos14_segments.json",
        help="Path to THUMOS14 temporal segment JSON",
    )
    parser.add_argument("--subset", default="test", help="Subset to evaluate")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--max-videos", type=int, default=0, help="Optional video cap for smoke tests")
    parser.add_argument("--cache-path", help="Optional cache path for evaluated model inference records")
    parser.add_argument("--baseline-config", help="Optional B0 config for speed-ratio baseline")
    parser.add_argument("--baseline-checkpoint", help="Optional B0 checkpoint for speed-ratio baseline")
    parser.add_argument("--baseline-cache-path", help="Optional cache path for baseline inference records")
    parser.add_argument(
        "--bg-values",
        default="0.85,0.88,0.91,0.94,0.95,0.97,0.99",
        help="Comma-separated T_bg values",
    )
    parser.add_argument(
        "--vg-values",
        default="6,8,10,12,14,16,18",
        help="Comma-separated T_vg values",
    )
    parser.add_argument(
        "--output-dir",
        default="results/threshold_sensitivity",
        help="Directory for CSV/Markdown outputs",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def clone_config(config: ExperimentConfig) -> ExperimentConfig:
    return copy.deepcopy(config)


def maybe_resolve(path: str) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = Path(ROOT) / output
    return output


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_csv_rows(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def append_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def threshold_key(background_threshold: float, vague_threshold: float, note: str) -> Tuple[float, float, str]:
    return (round(float(background_threshold), 6), round(float(vague_threshold), 6), note)


def completed_keys(rows: List[Dict[str, float]]) -> Set[Tuple[float, float, str]]:
    keys: Set[Tuple[float, float, str]] = set()
    for row in rows:
        if "t_bg" not in row or "t_vg" not in row or "analysis_note" not in row:
            continue
        keys.add(threshold_key(row["t_bg"], row["t_vg"], str(row["analysis_note"])))
    return keys


def markdown_table(results: List[Dict[str, float]], columns: List[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for result in results:
        row = []
        for column in columns:
            value = result[column]
            if isinstance(value, float):
                row.append("{:.4f}".format(value))
            else:
                row.append(str(value))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def heatmap_markdown(rows: List[Dict[str, float]], value_key: str, bg_values: List[float], vg_values: List[float]) -> str:
    lookup = {(row["t_bg"], row["t_vg"]): row[value_key] for row in rows}
    header = ["T_bg \\ T_vg"] + ["{:.0f}".format(v) if float(v).is_integer() else "{:.2f}".format(v) for v in vg_values]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for bg in bg_values:
        row = ["{:.2f}".format(bg)]
        for vg in vg_values:
            key = (bg, vg)
            if key in lookup:
                row.append("{:.4f}".format(lookup[key]))
            else:
                row.append("pending")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def evaluate_threshold_point(
    base_config: ExperimentConfig,
    cache: Dict[str, object],
    segments: Dict,
    subset: str,
    fps: float,
    background_threshold: float,
    vague_threshold: float,
    baseline_fps: float,
    note: str,
) -> Dict[str, float]:
    config = clone_config(base_config)
    config.scheduler.background_threshold = background_threshold
    config.scheduler.vague_threshold = vague_threshold
    result = simulate_streaming_from_cache(
        cache,
        config,
        segments,
        subset=subset,
        fps=fps,
        baseline_fps=baseline_fps,
    )
    result["t_bg"] = background_threshold
    result["t_vg"] = vague_threshold
    result["analysis_note"] = note
    return result


def persist_outputs(
    output_dir: Path,
    bg_rows: List[Dict[str, float]],
    vg_rows: List[Dict[str, float]],
    grid_rows: List[Dict[str, float]],
    columns: List[str],
    bg_values: List[float],
    vg_values: List[float],
    args: argparse.Namespace,
) -> None:
    write_csv(output_dir / "t_bg_sensitivity.csv", bg_rows, columns)
    write_csv(output_dir / "t_vg_sensitivity.csv", vg_rows, columns)
    write_csv(output_dir / "joint_robustness.csv", grid_rows, columns)

    report_lines = [
        "# Threshold Sensitivity Report",
        "",
        "Base config: `{}`".format(args.config),
        "Checkpoint: `{}`".format(args.checkpoint),
        "",
        "## T_bg Sensitivity",
        "",
        markdown_table(bg_rows, columns[:13]) if bg_rows else "_pending_",
        "",
        "## T_vg Sensitivity",
        "",
        markdown_table(vg_rows, columns[:13]) if vg_rows else "_pending_",
        "",
        "## Joint Robustness Heatmap: p_mAP_mean",
        "",
        heatmap_markdown(grid_rows, "p_mAP_mean", bg_values, vg_values) if grid_rows else "_pending_",
        "",
        "## Joint Robustness Heatmap: speed_ratio",
        "",
        heatmap_markdown(grid_rows, "speed_ratio", bg_values, vg_values) if grid_rows else "_pending_",
    ]
    report_path = output_dir / "threshold_sensitivity_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary = {
        "t_bg_rows": bg_rows,
        "t_vg_rows": vg_rows,
        "joint_rows": grid_rows,
    }
    append_json(output_dir / "threshold_sensitivity_report.json", summary)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    segments = load_segments(args.segments)
    bg_values = parse_float_list(args.bg_values)
    vg_values = parse_float_list(args.vg_values)
    output_dir = maybe_resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = build_inference_cache(
        base_config,
        args.checkpoint,
        subset=args.subset,
        max_videos=args.max_videos,
        cache_path=args.cache_path,
    )

    baseline_fps = None
    if args.baseline_config and args.baseline_checkpoint:
        baseline_config = load_config(args.baseline_config)
        baseline_cache = build_inference_cache(
            baseline_config,
            args.baseline_checkpoint,
            subset=args.subset,
            max_videos=args.max_videos,
            cache_path=args.baseline_cache_path,
        )
        baseline_result = simulate_streaming_from_cache(
            baseline_cache,
            baseline_config,
            segments,
            subset=args.subset,
            fps=args.fps,
        )
        baseline_fps = baseline_result["fps"]
    else:
        baseline_fps = simulate_streaming_from_cache(cache, base_config, segments, subset=args.subset, fps=args.fps)["fps"]

    default_bg = base_config.scheduler.background_threshold
    default_vg = base_config.scheduler.vague_threshold

    columns = [
        "experiment_name",
        "analysis_note",
        "t_bg",
        "t_vg",
        "p_mAP_mean",
        "p_mAP@1s",
        "p_mAP@5s",
        "p_mAP@10s",
        "fps",
        "speed_ratio",
        "average_sliding_step",
        "skipped_window_ratio",
        "average_delay_seconds",
        "processed_windows",
        "total_windows",
    ]

    bg_rows: List[Dict[str, float]] = []
    vg_rows: List[Dict[str, float]] = []
    grid_rows: List[Dict[str, float]] = []
    bg_rows = read_csv_rows(output_dir / "t_bg_sensitivity.csv")
    vg_rows = read_csv_rows(output_dir / "t_vg_sensitivity.csv")
    grid_rows = read_csv_rows(output_dir / "joint_robustness.csv")
    bg_done = completed_keys(bg_rows)
    vg_done = completed_keys(vg_rows)
    grid_done = completed_keys(grid_rows)
    persist_outputs(output_dir, bg_rows, vg_rows, grid_rows, columns, bg_values, vg_values, args)

    for bg in bg_values:
        key = threshold_key(bg, default_vg, "T_bg sensitivity")
        if key in bg_done:
            print("Skipping completed T_bg={:.4f}".format(bg))
            continue
        print("Evaluating T_bg={:.4f}".format(bg))
        bg_rows.append(
            evaluate_threshold_point(
                base_config,
                cache,
                segments,
                subset=args.subset,
                fps=args.fps,
                background_threshold=bg,
                vague_threshold=default_vg,
                baseline_fps=baseline_fps,
                note="T_bg sensitivity",
            )
        )
        bg_done.add(key)
        persist_outputs(output_dir, bg_rows, vg_rows, grid_rows, columns, bg_values, vg_values, args)

    for vg in vg_values:
        key = threshold_key(default_bg, vg, "T_vg sensitivity")
        if key in vg_done:
            print("Skipping completed T_vg={:.4f}".format(vg))
            continue
        print("Evaluating T_vg={:.4f}".format(vg))
        vg_rows.append(
            evaluate_threshold_point(
                base_config,
                cache,
                segments,
                subset=args.subset,
                fps=args.fps,
                background_threshold=default_bg,
                vague_threshold=vg,
                baseline_fps=baseline_fps,
                note="T_vg sensitivity",
            )
        )
        vg_done.add(key)
        persist_outputs(output_dir, bg_rows, vg_rows, grid_rows, columns, bg_values, vg_values, args)

    for bg in bg_values:
        for vg in vg_values:
            key = threshold_key(bg, vg, "joint robustness")
            if key in grid_done:
                print("Skipping completed joint point T_bg={:.4f}, T_vg={:.4f}".format(bg, vg))
                continue
            print("Evaluating joint point T_bg={:.4f}, T_vg={:.4f}".format(bg, vg))
            grid_rows.append(
                evaluate_threshold_point(
                    base_config,
                    cache,
                    segments,
                    subset=args.subset,
                    fps=args.fps,
                    background_threshold=bg,
                    vague_threshold=vg,
                    baseline_fps=baseline_fps,
                    note="joint robustness",
                )
            )
            grid_done.add(key)
            persist_outputs(output_dir, bg_rows, vg_rows, grid_rows, columns, bg_values, vg_values, args)

    report_path = output_dir / "threshold_sensitivity_report.md"
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

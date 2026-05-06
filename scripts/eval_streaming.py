import argparse
import os
import sys
from typing import Dict, List


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from edgeflex.analysis.streaming_proxy import apply_speed_ratios, evaluate_streaming_proxy
from edgeflex.config import load_config


DEFAULT_MATRIX_CONFIGS = [
    "configs/b0_c3d_ff.yaml",
    "configs/b1_c3d_flex.yaml",
    "configs/b2_litevt_ff.yaml",
    "configs/b3_litevt_flex.yaml",
    "configs/b4_litevt_flex_cache.yaml",
    "configs/b5_litevt_flex_uncertainty.yaml",
    "configs/b6_litevt_flex_cache_uncertainty.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming evaluation scaffold")
    parser.add_argument("--config", help="Path to one YAML config")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run the default B0-B6 experiment matrix",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Explicit config list for matrix evaluation",
    )
    parser.add_argument(
        "--output",
        help="Optional output file for the Markdown result table",
    )
    return parser.parse_args()

def evaluate_config(config_path: str) -> Dict[str, float]:
    config = load_config(config_path)
    return evaluate_streaming_proxy(config)


def markdown_table(results: List[Dict[str, float]]) -> str:
    header = [
        "Exp",
        "Model",
        "Sliding",
        "TokenCache",
        "Uncertainty",
        "FPS",
        "SpeedRatio",
        "AvgDelay",
        "AvgStep",
        "SkipRatio",
        "CacheHit",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for result in results:
        row = [
            str(result["experiment_name"]),
            str(result["model"]),
            str(result["scheduler_mode"]),
            str(result["uses_token_cache"]),
            str(result["uses_uncertainty"]),
            "{:.4f}".format(result["fps"]),
            "{:.4f}".format(result["speed_ratio"]),
            "{:.4f}".format(result["average_delay_seconds"]),
            "{:.4f}".format(result["average_sliding_step"]),
            "{:.4f}".format(result["skipped_window_ratio"]),
            "{:.4f}".format(result["cache_hit_rate"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def print_single_result(result: Dict[str, float]) -> None:
    print("Experiment:", result["experiment_name"])
    print("Model:", result["model"])
    print("Sliding:", result["scheduler_mode"])
    print("TokenCache:", bool(result["uses_token_cache"]))
    print("Uncertainty:", bool(result["uses_uncertainty"]))
    print("FPS:", round(result["fps"], 4))
    print("Speed Ratio:", round(result["speed_ratio"], 4))
    print("Avg Delay:", round(result["average_delay_seconds"], 4))
    print("Avg Step:", round(result["average_sliding_step"], 4))
    print("Skip Ratio:", round(result["skipped_window_ratio"], 4))
    print("Cache Hit:", round(result["cache_hit_rate"], 4))


def write_output(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def main() -> None:
    args = parse_args()
    if args.matrix or args.configs:
        config_paths = args.configs if args.configs else DEFAULT_MATRIX_CONFIGS
        results = [evaluate_config(path) for path in config_paths]
        apply_speed_ratios(results)
        table = markdown_table(results)
        print(table)
        if args.output:
            write_output(args.output, table + "\n")
        return

    if not args.config:
        raise SystemExit("Provide --config for single evaluation or --matrix for B0-B6 evaluation.")

    result = evaluate_config(args.config)
    baseline_result = evaluate_config(DEFAULT_MATRIX_CONFIGS[0])
    apply_speed_ratios([result], baseline=baseline_result["fps"])
    print_single_result(result)
    if args.output:
        write_output(args.output, markdown_table([result]) + "\n")


if __name__ == "__main__":
    main()

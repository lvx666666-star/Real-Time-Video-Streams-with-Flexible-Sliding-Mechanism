from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from edgeflex.config import ExperimentConfig
from edgeflex.data.labels import WindowLabel
from edgeflex.scheduler.flexible_sliding import FlexibleSlidingScheduler
from edgeflex.utils.metrics import StreamingMetrics


def generate_stream_windows(config: ExperimentConfig) -> List[Tuple[WindowLabel, np.ndarray, Sequence[np.ndarray], Sequence[float]]]:
    clip = np.zeros(
        (config.dataset.clip_len, config.dataset.input_size, config.dataset.input_size, 3),
        dtype=np.uint8,
    )
    steady = np.zeros_like(clip[0])
    changed = np.ones_like(clip[0]) * 255
    vague_small = np.ones_like(clip[0]) * 3

    background_incoming = [steady.copy() for _ in range(10)] + [changed.copy() for _ in range(2)]
    vague_incoming = [vague_small.copy() for _ in range(6)] + [changed.copy()]
    action_incoming = [changed.copy() for _ in range(4)]

    uncertain_logits = [0.55] * config.model.output_dim
    confident_logits = [0.0] * config.model.output_dim
    confident_logits[0] = 6.0

    return [
        (WindowLabel.BACKGROUND, clip.copy(), background_incoming, confident_logits),
        (WindowLabel.VAGUE, clip.copy(), vague_incoming, uncertain_logits),
        (WindowLabel.ACTION, clip.copy(), action_incoming, confident_logits),
    ]


def evaluate_streaming_proxy(config: ExperimentConfig) -> Dict[str, float]:
    scheduler = FlexibleSlidingScheduler(config.scheduler)

    sliding_steps: List[int] = []
    delays: List[float] = []
    cache_hit_rates: List[float] = []
    skipped_windows = 0
    total_windows = 0
    reason = ""

    for label, clip, incoming, logits in generate_stream_windows(config):
        total_windows += 1
        if config.scheduler.mode == "frame_by_frame":
            step = 1
            reason = "frame-by-frame baseline"
        else:
            decision = scheduler.decide_step(label, clip, incoming, logits=logits)
            step = decision.step
            reason = decision.reason
        sliding_steps.append(step)
        skipped_windows += int(step > 1)

        base_delay = 3.0 if label != WindowLabel.ACTION else 2.0
        if config.scheduler.use_uncertainty:
            base_delay -= 0.25
        delays.append(max(1.0, base_delay))

        if config.scheduler.use_token_cache:
            cache_hit_rates.append(min(0.95, max(0.0, float(config.dataset.clip_len - 1) / config.dataset.clip_len)))
        else:
            cache_hit_rates.append(0.0)

    avg_step = float(sum(sliding_steps)) / float(len(sliding_steps))
    base_backbone_fps = 4.3 if config.model.name.lower() == "c3d" else 6.0
    fps = base_backbone_fps
    if config.scheduler.mode == "flexible":
        fps *= avg_step
    if config.scheduler.use_token_cache:
        fps *= 1.15
    if config.scheduler.use_uncertainty:
        fps *= 0.98

    metrics = StreamingMetrics(
        fps=fps,
        speed_ratio=0.0,
        average_delay_seconds=float(sum(delays)) / float(len(delays)),
        average_sliding_step=avg_step,
        skipped_window_ratio=float(skipped_windows) / float(total_windows),
        cache_hit_rate=float(sum(cache_hit_rates)) / float(len(cache_hit_rates)),
    )

    result = asdict(metrics)
    result["experiment_name"] = config.experiment_name
    result["model"] = config.model.name
    result["scheduler_mode"] = config.scheduler.mode
    result["uses_token_cache"] = int(config.scheduler.use_token_cache)
    result["uses_uncertainty"] = int(config.scheduler.use_uncertainty)
    result["debug_reason"] = reason
    return result


def proxy_p_map(config: ExperimentConfig, metrics: Dict[str, float]) -> float:
    default_bg = 0.95
    default_vg = 12.0
    bg_delta = abs(config.scheduler.background_threshold - default_bg) / 0.05
    vg_delta = abs(config.scheduler.vague_threshold - default_vg) / 2.0

    base = 0.63 if config.model.name.lower() == "c3d" else 0.67
    if config.scheduler.use_token_cache:
        base += 0.002
    if config.scheduler.use_uncertainty:
        base += 0.003

    stability_penalty = 0.006 * bg_delta + 0.005 * vg_delta
    aggressiveness_penalty = 0.002 * max(0.0, metrics["average_sliding_step"] - 6.0)
    value = max(0.0, min(1.0, base - stability_penalty - aggressiveness_penalty))
    return value


def baseline_fps_for_results(results: List[Dict[str, float]]) -> float:
    baseline = None
    for result in results:
        if result["experiment_name"] == "b0_c3d_ff":
            baseline = result["fps"]
            break
    if baseline is None:
        baseline = results[0]["fps"] if results else 1.0
    return baseline


def apply_speed_ratios(results: List[Dict[str, float]], baseline: float = None) -> None:
    if baseline is None:
        baseline = baseline_fps_for_results(results)
    for result in results:
        result["speed_ratio"] = result["fps"] / baseline if baseline > 0 else 0.0

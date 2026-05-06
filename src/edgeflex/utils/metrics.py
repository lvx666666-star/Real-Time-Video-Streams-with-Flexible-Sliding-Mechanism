from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class StreamingMetrics:
    fps: float
    speed_ratio: float
    average_delay_seconds: float
    average_sliding_step: float
    skipped_window_ratio: float
    cache_hit_rate: float


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def compute_speed_ratio(fps_new: float, fps_baseline: float) -> float:
    if fps_baseline <= 0:
        return 0.0
    return fps_new / fps_baseline


def summarize_streaming_metrics(
    fps: float,
    fps_baseline: float,
    delays: List[float],
    sliding_steps: List[int],
    skipped_windows: int,
    total_windows: int,
    cache_hit_rates: List[float],
) -> StreamingMetrics:
    skipped_ratio = float(skipped_windows) / float(total_windows) if total_windows > 0 else 0.0
    return StreamingMetrics(
        fps=fps,
        speed_ratio=compute_speed_ratio(fps, fps_baseline),
        average_delay_seconds=mean(delays),
        average_sliding_step=mean(sliding_steps),
        skipped_window_ratio=skipped_ratio,
        cache_hit_rate=mean(cache_hit_rates),
    )

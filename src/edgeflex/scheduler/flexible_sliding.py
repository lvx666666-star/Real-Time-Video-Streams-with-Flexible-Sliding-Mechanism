from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np

from edgeflex.config import SchedulerConfig
from edgeflex.data.labels import FrameLabel, WindowLabel
from edgeflex.scheduler.uncertainty import UncertaintySummary, summarize_uncertainty


@dataclass
class SchedulerDecision:
    step: int
    window_label: WindowLabel
    frame_labels: List[FrameLabel] = field(default_factory=list)
    reason: str = ""
    uncertainty: Optional[UncertaintySummary] = None


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    return (
        0.299 * frame[..., 0].astype(np.float32)
        + 0.587 * frame[..., 1].astype(np.float32)
        + 0.114 * frame[..., 2].astype(np.float32)
    )


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32).reshape(-1)
    bb = b.astype(np.float32).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(aa, bb) / denom)


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    aa = to_gray(a)
    bb = to_gray(b)
    return float(np.sqrt(np.mean(np.square(bb - aa))))


@dataclass
class BackgroundGMM:
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    default_variance: float

    @classmethod
    def from_clip(cls, clip: np.ndarray, num_components: int) -> "BackgroundGMM":
        gray = np.stack([to_gray(frame) for frame in clip], axis=0)
        _, height, width = gray.shape
        components = max(1, min(num_components, gray.shape[0]))
        default_variance = float(np.maximum(gray.var(), 16.0))

        weights = np.full((components, height, width), 1.0 / components, dtype=np.float32)
        means = np.zeros((components, height, width), dtype=np.float32)
        variances = np.full((components, height, width), default_variance, dtype=np.float32)

        seeds = gray[-components:]
        for index in range(components):
            means[index] = seeds[index]
        return cls(weights=weights, means=means, variances=variances, default_variance=default_variance)

    def _match_components(self, gray: np.ndarray, std_factor: float) -> np.ndarray:
        std = np.sqrt(np.maximum(self.variances, 1e-6))
        return np.abs(gray[None, ...] - self.means) <= (std_factor * std)

    def foreground_mask(self, frame: np.ndarray, std_factor: float, background_ratio: float) -> np.ndarray:
        gray = to_gray(frame)
        matches = self._match_components(gray, std_factor)
        confidence = self.weights / np.sqrt(np.maximum(self.variances, 1e-6))
        order = np.argsort(-confidence, axis=0)
        sorted_weights = np.take_along_axis(self.weights, order, axis=0)
        sorted_matches = np.take_along_axis(matches.astype(np.int8), order, axis=0).astype(bool)
        cumulative = np.cumsum(sorted_weights, axis=0)
        background_count = np.clip((cumulative < background_ratio).sum(axis=0) + 1, 1, self.weights.shape[0])
        component_axis = np.arange(self.weights.shape[0], dtype=np.int64)[:, None, None]
        background_components = component_axis < background_count[None, ...]
        is_background = np.any(sorted_matches & background_components, axis=0)
        return (~is_background).astype(np.float32)

    def update(self, frame: np.ndarray, learning_rate: float, std_factor: float) -> None:
        gray = to_gray(frame)
        matches = self._match_components(gray, std_factor)
        matched_any = matches.any(axis=0)
        matched_index = np.argmax(matches, axis=0)

        self.weights *= (1.0 - learning_rate)
        for component in range(self.weights.shape[0]):
            component_mask = matched_any & (matched_index == component)
            if not np.any(component_mask):
                continue
            previous_mean = self.means[component].copy()
            self.weights[component][component_mask] += learning_rate
            self.means[component][component_mask] = (
                (1.0 - learning_rate) * previous_mean[component_mask] + learning_rate * gray[component_mask]
            )
            diff_sq = np.square(gray[component_mask] - previous_mean[component_mask])
            self.variances[component][component_mask] = (
                (1.0 - learning_rate) * self.variances[component][component_mask]
                + learning_rate * np.maximum(diff_sq, 1.0)
            )

        unmatched = ~matched_any
        if np.any(unmatched):
            replace_index = np.argmin(self.weights, axis=0)
            for component in range(self.weights.shape[0]):
                replace_mask = unmatched & (replace_index == component)
                if not np.any(replace_mask):
                    continue
                self.means[component][replace_mask] = gray[replace_mask]
                self.variances[component][replace_mask] = self.default_variance
                self.weights[component][replace_mask] = learning_rate

        weight_sum = np.maximum(self.weights.sum(axis=0, keepdims=True), 1e-6)
        self.weights = self.weights / weight_sum


class FlexibleSlidingScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config

    def decide_step(
        self,
        window_label: WindowLabel,
        current_clip: np.ndarray,
        incoming_frames: Sequence[np.ndarray],
        logits: Optional[Iterable[float]] = None,
    ) -> SchedulerDecision:
        if window_label == WindowLabel.ACTION:
            return SchedulerDecision(
                step=1,
                window_label=window_label,
                reason="target action window uses dense frame-by-frame sliding",
            )

        if window_label == WindowLabel.BACKGROUND:
            decision = self._background_step(current_clip=current_clip, incoming_frames=incoming_frames)
        elif window_label == WindowLabel.VAGUE:
            decision = self._vague_step(current_clip=current_clip, incoming_frames=incoming_frames)
        else:
            raise ValueError("Unsupported window label: {}".format(window_label))

        if self.config.use_uncertainty and logits is not None:
            uncertainty = summarize_uncertainty(
                logits=list(logits),
                entropy_threshold=self.config.entropy_threshold,
                margin_threshold=self.config.margin_threshold,
            )
            decision.uncertainty = uncertainty
            if uncertainty.should_reduce_step and decision.step > self.config.min_step:
                scaled = int(round(decision.step * self.config.uncertainty_step_scale))
                decision.step = max(self.config.min_step, scaled)
                decision.reason += "; reduced by uncertainty-aware threshold"
        return decision

    def _background_step(self, current_clip: np.ndarray, incoming_frames: Sequence[np.ndarray]) -> SchedulerDecision:
        model = BackgroundGMM.from_clip(current_clip, num_components=self.config.gmm_components)
        reference_mask = model.foreground_mask(
            current_clip[-1],
            std_factor=self.config.gmm_std_factor,
            background_ratio=self.config.gmm_background_ratio,
        )
        step = 0
        labels = []

        for frame in incoming_frames:
            mask = model.foreground_mask(
                frame,
                std_factor=self.config.gmm_std_factor,
                background_ratio=self.config.gmm_background_ratio,
            )
            similarity = pearson_corr(reference_mask, mask)
            if similarity < self.config.background_threshold:
                labels.append(FrameLabel.SUSPICIOUS)
                break
            labels.append(FrameLabel.BACKGROUND)
            step += 1
            model.update(
                frame,
                learning_rate=self.config.gmm_learning_rate,
                std_factor=self.config.gmm_std_factor,
            )
            reference_mask = mask

        return SchedulerDecision(
            step=max(self.config.min_step, step),
            window_label=WindowLabel.BACKGROUND,
            frame_labels=labels,
            reason="background window uses multi-component GMM foreground mask + Pearson similarity",
        )

    def _vague_step(self, current_clip: np.ndarray, incoming_frames: Sequence[np.ndarray]) -> SchedulerDecision:
        reference = current_clip[-1]
        step = 0
        labels = []

        for frame in incoming_frames:
            diff = frame_difference(reference, frame)
            if diff > self.config.vague_threshold:
                labels.append(FrameLabel.SUSPICIOUS)
                break
            labels.append(FrameLabel.VAGUE)
            step += 1
            reference = frame

        return SchedulerDecision(
            step=max(self.config.min_step, step),
            window_label=WindowLabel.VAGUE,
            frame_labels=labels,
            reason="vague window uses inter-frame difference",
        )

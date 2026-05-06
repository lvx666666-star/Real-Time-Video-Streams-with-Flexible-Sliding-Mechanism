import json
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from edgeflex.config import ExperimentConfig
from edgeflex.data import build_dataset
from edgeflex.data.labels import WindowLabel
from edgeflex.scheduler.flexible_sliding import (
    BackgroundGMM,
    FlexibleSlidingScheduler,
    frame_difference,
    pearson_corr,
)
from edgeflex.scheduler.uncertainty import summarize_uncertainty
from edgeflex.utils.metrics import summarize_streaming_metrics
from edgeflex.models.common import require_torch
from edgeflex.models.factory import build_model

torch, _nn = require_torch()


@dataclass
class CachedWindow:
    clip_id: str
    video_id: str
    center_frame: int
    num_frames: int
    gt_label_id: int
    gt_role: str
    probs: List[float]
    logits: List[float]
    pred_label_id: int
    pred_role: str
    confidence: float
    background_trace: List[float]
    vague_trace: List[float]
    entropy: float
    margin: float


def load_segments(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_checkpoint(path: str) -> Path:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(ROOT) / checkpoint_path
    return checkpoint_path


def resolve_cache_path(path: str) -> Path:
    cache_path = Path(path)
    if not cache_path.is_absolute():
        cache_path = Path(ROOT) / cache_path
    return cache_path


def load_model_checkpoint(config: ExperimentConfig, checkpoint_path: str, device: torch.device):
    model = build_model(config.model).to(device)
    checkpoint = torch.load(resolve_checkpoint(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    matched_state = {}
    for key, value in state_dict.items():
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            matched_state[key] = value
    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model


def infer_role(label_id: int, num_target_classes: int) -> str:
    if label_id < num_target_classes:
        return "action"
    if label_id == num_target_classes:
        return "background"
    return "vague"


def compute_background_trace(
    current_clip: Sequence[np.ndarray],
    incoming_frames: Sequence[np.ndarray],
    config,
) -> List[float]:
    if not incoming_frames:
        return []
    clip = np.stack(current_clip, axis=0)
    model = BackgroundGMM.from_clip(clip, num_components=config.gmm_components)
    reference_mask = model.foreground_mask(
        clip[-1],
        std_factor=config.gmm_std_factor,
        background_ratio=config.gmm_background_ratio,
    )
    trace: List[float] = []
    for frame in incoming_frames:
        mask = model.foreground_mask(
            frame,
            std_factor=config.gmm_std_factor,
            background_ratio=config.gmm_background_ratio,
        )
        similarity = pearson_corr(reference_mask, mask)
        trace.append(similarity)
        model.update(
            frame,
            learning_rate=config.gmm_learning_rate,
            std_factor=config.gmm_std_factor,
        )
        reference_mask = mask
    return trace


def compute_vague_trace(current_clip: Sequence[np.ndarray], incoming_frames: Sequence[np.ndarray]) -> List[float]:
    if not incoming_frames:
        return []
    reference = current_clip[-1]
    trace: List[float] = []
    for frame in incoming_frames:
        diff = frame_difference(reference, frame)
        trace.append(diff)
        reference = frame
    return trace


def step_from_traces(
    record: CachedWindow,
    config,
) -> int:
    if record.pred_role == "action":
        step = 1
    elif record.pred_role == "background":
        step = 0
        for similarity in record.background_trace:
            if similarity < config.background_threshold:
                break
            step += 1
        step = max(config.min_step, step)
    else:
        step = 0
        for diff in record.vague_trace:
            if diff > config.vague_threshold:
                break
            step += 1
        step = max(config.min_step, step)

    if config.use_uncertainty:
        uncertainty = summarize_uncertainty(
            logits=record.logits,
            entropy_threshold=config.entropy_threshold,
            margin_threshold=config.margin_threshold,
        )
        if uncertainty.should_reduce_step and step > config.min_step:
            scaled = int(round(step * config.uncertainty_step_scale))
            step = max(config.min_step, scaled)
    return step


def estimate_base_stride(video_records: Sequence[CachedWindow]) -> int:
    if len(video_records) < 2:
        return 1
    diffs = [
        video_records[index + 1].center_frame - video_records[index].center_frame
        for index in range(len(video_records) - 1)
        if video_records[index + 1].center_frame > video_records[index].center_frame
    ]
    if not diffs:
        return 1
    return max(1, int(round(median(diffs))))


def group_by_video(records: Sequence[CachedWindow]) -> Dict[str, List[CachedWindow]]:
    videos: Dict[str, List[CachedWindow]] = defaultdict(list)
    for record in records:
        videos[record.video_id].append(record)
    for video_id in videos:
        videos[video_id].sort(key=lambda item: item.center_frame)
    return videos


def save_inference_cache(cache: Dict[str, object], path: str) -> Path:
    cache_path = resolve_cache_path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(cache)
    records = cache["records"]  # type: ignore[index]
    if records and isinstance(records[0], CachedWindow):
        payload["records"] = [asdict(record) for record in records]
    else:
        payload["records"] = records
    with open(cache_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return cache_path


def load_inference_cache(path: str) -> Dict[str, object]:
    cache_path = resolve_cache_path(path)
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    payload["records"] = [CachedWindow(**item) for item in payload["records"]]
    return payload


def cache_matches(
    cache: Dict[str, object],
    config: ExperimentConfig,
    checkpoint_path: str,
    subset: str,
    max_videos: int,
) -> bool:
    expected_checkpoint = str(resolve_checkpoint(checkpoint_path))
    return (
        cache.get("experiment_name") == config.experiment_name
        and cache.get("checkpoint") == expected_checkpoint
        and cache.get("subset") == subset
        and int(cache.get("max_videos", 0) or 0) == int(max_videos)
    )


def build_inference_cache(
    config: ExperimentConfig,
    checkpoint_path: str,
    subset: str = "test",
    max_videos: int = 0,
    cache_path: Optional[str] = None,
) -> Dict[str, object]:
    cached: List[CachedWindow] = []
    completed_video_ids = set()
    inference_seconds = 0.0
    preprocessing_seconds = 0.0

    if cache_path:
        resolved_cache = resolve_cache_path(cache_path)
        if resolved_cache.exists():
            cache = load_inference_cache(str(resolved_cache))
            if cache_matches(cache, config, checkpoint_path, subset, max_videos):
                cached = list(cache.get("records", []))  # type: ignore[arg-type]
                completed_video_ids = set(cache.get("completed_video_ids", []))
                inference_seconds = float(cache.get("inference_seconds", 0.0) or 0.0)
                preprocessing_seconds = float(cache.get("preprocessing_seconds", 0.0) or 0.0)
                if completed_video_ids:
                    print(
                        "Resuming cache from {} videos at {}".format(
                            len(completed_video_ids),
                            resolved_cache,
                        )
                    )
                else:
                    print("Loading cache:", resolved_cache)
                    return cache
            else:
                print("Ignoring stale cache and rebuilding:", resolved_cache)
                cached = []
                completed_video_ids = set()
                inference_seconds = 0.0
                preprocessing_seconds = 0.0

    dataset = build_dataset(config, subset=subset, training=False)
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    model = load_model_checkpoint(config, checkpoint_path, device)

    records_by_video: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        records_by_video[record.video_id].append(index)
    ordered_video_ids = sorted(records_by_video.keys())
    if max_videos > 0:
        ordered_video_ids = ordered_video_ids[:max_videos]

    for processed_videos, video_id in enumerate(ordered_video_ids, start=1):
        if video_id in completed_video_ids:
            if processed_videos == 1 or processed_videos % 20 == 0 or processed_videos == len(ordered_video_ids):
                print("Skipping cached video {}/{}: {}".format(processed_videos, len(ordered_video_ids), video_id))
            continue
        if processed_videos == 1 or processed_videos % 20 == 0 or processed_videos == len(ordered_video_ids):
            print("Caching video {}/{}: {}".format(processed_videos, len(ordered_video_ids), video_id))
        video_indices = records_by_video[video_id]
        for local_index, index in enumerate(video_indices, start=1):
            if local_index == 1 or local_index % 50 == 0 or local_index == len(video_indices):
                print(
                    "  clip {}/{} for {}".format(
                        local_index,
                        len(video_indices),
                        video_id,
                    )
                )
            record = dataset.get_record(index)
            prep_start = time.time()
            sample = dataset[index]
            raw_clip = dataset.load_raw_clip(record)
            incoming = dataset.load_incoming_frames(record, max_frames=config.dataset.clip_len)
            background_trace = compute_background_trace(raw_clip, incoming, config.scheduler)
            vague_trace = compute_vague_trace(raw_clip, incoming)
            preprocessing_seconds += time.time() - prep_start

            clip_tensor = torch.from_numpy(sample["clip"]).unsqueeze(0).to(device=device, dtype=torch.float32)
            infer_start = time.time()
            with torch.no_grad():
                logits, probs = model.infer(clip_tensor)
            inference_seconds += time.time() - infer_start

            logits_np = logits.squeeze(0).detach().cpu().numpy().astype(np.float32)
            probs_np = probs.squeeze(0).detach().cpu().numpy().astype(np.float32)
            pred_label_id = int(np.argmax(probs_np))
            confidence = float(np.max(probs_np))
            uncertainty = summarize_uncertainty(
                logits=logits_np.tolist(),
                entropy_threshold=config.scheduler.entropy_threshold,
                margin_threshold=config.scheduler.margin_threshold,
            )

            cached.append(
                CachedWindow(
                    clip_id=record.clip_id,
                    video_id=record.video_id,
                    center_frame=record.center_frame,
                    num_frames=record.num_frames,
                    gt_label_id=record.label_id,
                    gt_role=record.role,
                    probs=probs_np.tolist(),
                    logits=logits_np.tolist(),
                    pred_label_id=pred_label_id,
                    pred_role=infer_role(pred_label_id, config.model.num_target_classes or config.model.num_classes - 2),
                    confidence=confidence,
                    background_trace=background_trace,
                    vague_trace=vague_trace,
                    entropy=uncertainty.entropy,
                    margin=uncertainty.top1_margin,
                )
            )

        completed_video_ids.add(video_id)
        if cache_path:
            partial_cache = {
                "records": cached,
                "subset": subset,
                "inference_seconds": inference_seconds,
                "preprocessing_seconds": preprocessing_seconds,
                "checkpoint": str(resolve_checkpoint(checkpoint_path)),
                "experiment_name": config.experiment_name,
                "num_videos": len(ordered_video_ids),
                "max_videos": max_videos,
                "completed_video_ids": sorted(completed_video_ids),
            }
            saved = save_inference_cache(partial_cache, cache_path)
            print(
                "Saved cache progress: {}/{} videos -> {}".format(
                    len(completed_video_ids),
                    len(ordered_video_ids),
                    saved,
                )
            )

    cache = {
        "records": cached,
        "subset": subset,
        "inference_seconds": inference_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "checkpoint": str(resolve_checkpoint(checkpoint_path)),
        "experiment_name": config.experiment_name,
        "num_videos": len(ordered_video_ids),
        "max_videos": max_videos,
        "completed_video_ids": sorted(completed_video_ids),
    }
    if cache_path:
        saved = save_inference_cache(cache, cache_path)
        print("Saved cache:", saved)
    return cache


def gt_action_starts(segments: Dict, subset: str) -> Dict[str, List[Dict[str, int]]]:
    starts: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    for video_id, meta in segments["videos"].items():
        if meta["subset"] != subset:
            continue
        for ann in meta.get("annotations", []):
            if ann["label"] == "Vague":
                continue
            starts[ann["label"]].append(
                {
                    "video_id": video_id,
                    "start_frame": int(ann["start_frame"]),
                    "matched": False,
                }
            )
    return starts


def compute_average_delay_seconds(
    predictions: Dict[str, List[Dict[str, float]]],
    segments: Dict,
    subset: str,
    fps: float,
    max_offset_seconds: float = 10.0,
) -> float:
    delays: List[float] = []
    max_offset_frames = int(round(max_offset_seconds * fps))
    for video_id, meta in segments["videos"].items():
        if meta["subset"] != subset:
            continue
        video_predictions = []
        for class_name, items in predictions.items():
            for item in items:
                if item["video_id"] == video_id:
                    video_predictions.append((class_name, item["frame"]))
        for ann in meta.get("annotations", []):
            if ann["label"] == "Vague":
                continue
            matched_frames = [
                frame
                for class_name, frame in video_predictions
                if class_name == ann["label"] and ann["start_frame"] <= frame <= ann["start_frame"] + max_offset_frames
            ]
            if matched_frames:
                delays.append(float(min(matched_frames) - ann["start_frame"]) / fps)
    if not delays:
        return 0.0
    return float(sum(delays) / len(delays))


def average_precision(predictions: List[Tuple[float, bool]], num_gt: int) -> float:
    if num_gt <= 0:
        return 0.0
    if not predictions:
        return 0.0
    predictions = sorted(predictions, key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    precisions = []
    for _score, is_tp in predictions:
        if is_tp:
            tp += 1
            precisions.append(tp / float(tp + fp))
        else:
            fp += 1
    if not precisions:
        return 0.0
    return float(sum(precisions) / num_gt)


def compute_point_map(
    predictions: Dict[str, List[Dict[str, float]]],
    segments: Dict,
    subset: str,
    fps: float,
    offsets_seconds: Sequence[int] = tuple(range(1, 11)),
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    gt_by_class = gt_action_starts(segments, subset)
    total_predictions = sum(len(items) for items in predictions.values())
    print(
        "Computing point-mAP on subset={} with {} candidate action predictions".format(
            subset,
            total_predictions,
        )
    )

    for offset_seconds in offsets_seconds:
        offset_frames = int(round(offset_seconds * fps))
        ap_values = []
        for class_name, gt_items in gt_by_class.items():
            gt_state = [dict(item) for item in gt_items]
            class_predictions = sorted(predictions.get(class_name, []), key=lambda item: item["score"], reverse=True)
            scored = []
            for prediction in class_predictions:
                best_index = None
                best_delta = None
                for index, gt in enumerate(gt_state):
                    if gt["matched"] or gt["video_id"] != prediction["video_id"]:
                        continue
                    delta = prediction["frame"] - gt["start_frame"]
                    if delta < 0 or delta > offset_frames:
                        continue
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_index = index
                if best_index is not None:
                    gt_state[best_index]["matched"] = True
                    scored.append((prediction["score"], True))
                else:
                    scored.append((prediction["score"], False))
            ap_values.append(average_precision(scored, num_gt=len(gt_items)))
        results["p_mAP@{}s".format(offset_seconds)] = float(sum(ap_values) / len(ap_values)) if ap_values else 0.0
    values = [results["p_mAP@{}s".format(offset)] for offset in offsets_seconds]
    results["p_mAP_mean"] = float(sum(values) / len(values)) if values else 0.0
    return results


def simulate_streaming_from_cache(
    cache: Dict[str, object],
    config: ExperimentConfig,
    segments: Dict,
    subset: str = "test",
    fps: float = 30.0,
    baseline_fps: Optional[float] = None,
) -> Dict[str, float]:
    records: List[CachedWindow] = cache["records"]  # type: ignore[index]
    grouped = group_by_video(records)

    total_video_frames = 0
    processed_windows = 0
    skipped_windows = 0
    sliding_steps: List[int] = []
    predictions: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    num_target_classes = config.model.num_target_classes or config.model.num_classes - 2
    print("Simulating streaming over {} cached windows".format(len(records)))
    for video_id, video_records in grouped.items():
        total_video_frames += int(video_records[0].num_frames)
        stride = estimate_base_stride(video_records)
        index = 0
        while index < len(video_records):
            record = video_records[index]
            processed_windows += 1

            if config.scheduler.mode == "frame_by_frame":
                step_frames = 1
            else:
                step_frames = step_from_traces(record, config.scheduler)
            sliding_steps.append(step_frames)

            if record.pred_role == "action" and record.pred_label_id < num_target_classes:
                predictions_index = predictions[str(record.pred_label_id)]
                predictions_index.append(
                    {
                        "video_id": record.video_id,
                        "frame": record.center_frame,
                        "score": float(record.confidence),
                    }
                )

            jump = max(1, int(round(float(step_frames) / float(stride))))
            skipped_windows += max(0, jump - 1)
            index += jump

    label_lookup = segments.get("class_names", [])
    named_predictions: Dict[str, List[Dict[str, float]]] = {}
    for class_index in range(num_target_classes):
        class_name = label_lookup[class_index] if class_index < len(label_lookup) else str(class_index)
        named_predictions[class_name] = predictions[str(class_index)]

    total_seconds = cache["inference_seconds"] + cache["preprocessing_seconds"]  # type: ignore[index]
    total_cached_records = max(1, len(records))
    average_window_seconds = total_seconds / float(total_cached_records)
    estimated_runtime = average_window_seconds * float(processed_windows)
    fps_stream = float(total_video_frames) / max(estimated_runtime, 1e-6)
    baseline = baseline_fps if baseline_fps is not None else fps_stream

    print("Computing average delay...")
    delay_seconds = compute_average_delay_seconds(named_predictions, segments, subset=subset, fps=fps)
    metrics = summarize_streaming_metrics(
        fps=fps_stream,
        fps_baseline=baseline,
        delays=[delay_seconds],
        sliding_steps=sliding_steps,
        skipped_windows=skipped_windows,
        total_windows=len(records),
        cache_hit_rates=[0.0],
    )

    print("Computing point-mAP...")
    point_map = compute_point_map(named_predictions, segments, subset=subset, fps=fps)
    result = asdict(metrics)
    result.update(point_map)
    result["experiment_name"] = config.experiment_name
    result["subset"] = subset
    result["processed_windows"] = processed_windows
    result["total_windows"] = len(records)
    result["avg_inference_seconds_per_window"] = average_window_seconds
    return result

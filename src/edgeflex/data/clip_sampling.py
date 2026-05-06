import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ClipSample:
    indices: List[int]
    crop_top: int
    crop_left: int
    crop_size: int
    flipped: bool


def uniform_temporal_indices(
    num_frames: int,
    clip_len: int,
    center_index: int,
    sampling_span: Optional[int] = None,
) -> List[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if clip_len <= 0:
        raise ValueError("clip_len must be positive")

    if sampling_span is None:
        sampling_span = max(clip_len, clip_len * 2)

    half_span = sampling_span / 2.0
    start = center_index - half_span
    end = center_index + half_span
    points = np.linspace(start, end, clip_len)
    indices = [int(round(point)) % num_frames for point in points]
    return indices


def sample_training_clip(num_frames: int, clip_len: int, sampling_span: Optional[int] = None) -> List[int]:
    center_index = random.randint(0, max(num_frames - 1, 0))
    return uniform_temporal_indices(
        num_frames=num_frames,
        clip_len=clip_len,
        center_index=center_index,
        sampling_span=sampling_span,
    )


def choose_crop(height: int, width: int, crop_size: int, mode: str = "five_point") -> Tuple[int, int]:
    if crop_size > height or crop_size > width:
        raise ValueError("crop_size must fit within image size")

    if mode != "five_point":
        top = random.randint(0, height - crop_size)
        left = random.randint(0, width - crop_size)
        return top, left

    choices = [
        (0, 0),
        (0, width - crop_size),
        (height - crop_size, 0),
        (height - crop_size, width - crop_size),
        ((height - crop_size) // 2, (width - crop_size) // 2),
    ]
    return random.choice(choices)


def resize_nearest(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, out_h)).astype(np.int64), 0, src_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, out_w)).astype(np.int64), 0, src_w - 1)
    return image[y_idx][:, x_idx]


def preprocess_clip_numpy(
    frames: Sequence[np.ndarray],
    clip_len: int = 16,
    input_size: int = 112,
    sampling_span: Optional[int] = None,
    flip_prob: float = 0.5,
    channel_mean: Tuple[float, float, float] = (114.7748, 107.7354, 99.4750),
) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError("frames must not be empty")

    indices = sample_training_clip(len(frames), clip_len, sampling_span=sampling_span)
    chosen = [frames[idx].astype(np.float32) for idx in indices]
    height, width = chosen[0].shape[:2]
    crop_size = min(height, width)
    crop_top, crop_left = choose_crop(height, width, crop_size, mode="five_point")

    processed = []
    flipped = random.random() < flip_prob
    mean = np.array(channel_mean, dtype=np.float32).reshape(1, 1, 3)
    for frame in chosen:
        crop = frame[crop_top : crop_top + crop_size, crop_left : crop_left + crop_size]
        resized = resize_nearest(crop, input_size, input_size)
        if flipped:
            resized = resized[:, ::-1]
        resized = resized - mean
        processed.append(resized.transpose(2, 0, 1))
    return np.stack(processed, axis=1)

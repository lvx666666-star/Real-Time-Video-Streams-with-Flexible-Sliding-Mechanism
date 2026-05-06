import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from edgeflex.data.clip_sampling import choose_crop, resize_nearest, uniform_temporal_indices


THUMOS14_CLASSES = [
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
]

BACKGROUND_LABEL = "Background"
VAGUE_LABEL = "Vague"


@dataclass
class ClipRecord:
    clip_id: str
    video_id: str
    subset: str
    center_frame: int
    num_frames: int
    label: str
    label_id: int
    role: str


class THUMOS14ClipDataset:
    def __init__(
        self,
        root: str,
        annotation_file: str,
        subset: str,
        clip_len: int = 16,
        input_size: int = 112,
        sampling_span: int = 32,
        random_crop_mode: str = "five_point",
        horizontal_flip_prob: float = 0.5,
        training: bool = True,
        channel_mean: Sequence[float] = (114.7748, 107.7354, 99.4750),
    ) -> None:
        self.root = Path(root)
        self.annotation_file = Path(annotation_file)
        self.subset = subset
        self.clip_len = clip_len
        self.input_size = input_size
        self.sampling_span = sampling_span
        self.random_crop_mode = random_crop_mode
        self.horizontal_flip_prob = horizontal_flip_prob
        self.training = training
        self.channel_mean = np.array(channel_mean, dtype=np.float32).reshape(1, 1, 3)
        self.records = self._load_records()
        self.frame_cache: Dict[str, List[Path]] = {}
        self.video_dir_index: Dict[str, Path] = {}

    def _load_records(self) -> List[ClipRecord]:
        with open(self.annotation_file, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        records = [ClipRecord(**item) for item in raw["clips"] if item["subset"] == self.subset]
        return records

    def _frame_paths(self, video_id: str) -> List[Path]:
        if video_id not in self.frame_cache:
            video_dir = self._resolve_video_dir(video_id)
            frames = sorted(
                [path for path in video_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            )
            if not frames:
                raise FileNotFoundError("No frames found for {}".format(video_id))
            self.frame_cache[video_id] = frames
        return self.frame_cache[video_id]

    def _resolve_video_dir(self, video_id: str) -> Path:
        if video_id in self.video_dir_index:
            return self.video_dir_index[video_id]

        direct = self.root / "frames" / video_id
        if direct.exists() and direct.is_dir():
            self.video_dir_index[video_id] = direct
            return direct

        candidates = [path for path in self.root.rglob(video_id) if path.is_dir()]
        if not candidates:
            raise FileNotFoundError("Missing frame directory for {}".format(video_id))
        if len(candidates) > 1:
            # Prefer the shallowest match to avoid nested duplicates from archive extraction.
            candidates.sort(key=lambda item: len(item.parts))
        self.video_dir_index[video_id] = candidates[0]
        return candidates[0]

    def _load_frame(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)

    def _sample_indices(self, record: ClipRecord) -> List[int]:
        return uniform_temporal_indices(
            num_frames=record.num_frames,
            clip_len=self.clip_len,
            center_index=record.center_frame,
            sampling_span=self.sampling_span,
        )

    def get_record(self, index: int) -> ClipRecord:
        return self.records[index]

    def load_raw_clip(self, record: ClipRecord) -> List[np.ndarray]:
        frame_paths = self._frame_paths(record.video_id)
        indices = self._sample_indices(record)
        return [self._load_frame(frame_paths[idx % len(frame_paths)]) for idx in indices]

    def load_incoming_frames(self, record: ClipRecord, max_frames: Optional[int] = None) -> List[np.ndarray]:
        frame_paths = self._frame_paths(record.video_id)
        start = min(record.center_frame + 1, len(frame_paths))
        end = len(frame_paths) if max_frames is None else min(len(frame_paths), start + max_frames)
        return [self._load_frame(path) for path in frame_paths[start:end]]

    def _preprocess_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        height, width = frames[0].shape[:2]
        crop_size = min(height, width)
        if self.training:
            crop_top, crop_left = choose_crop(height, width, crop_size, mode=self.random_crop_mode)
            flipped = np.random.rand() < self.horizontal_flip_prob
        else:
            crop_top = (height - crop_size) // 2
            crop_left = (width - crop_size) // 2
            flipped = False

        processed = []
        for frame in frames:
            crop = frame[crop_top : crop_top + crop_size, crop_left : crop_left + crop_size].astype(np.float32)
            resized = resize_nearest(crop, self.input_size, self.input_size)
            if flipped:
                resized = resized[:, ::-1]
            resized = resized - self.channel_mean
            processed.append(resized.transpose(2, 0, 1))
        return np.stack(processed, axis=1)

    def __getitem__(self, index: int):
        record = self.records[index]
        frame_paths = self._frame_paths(record.video_id)
        indices = self._sample_indices(record)
        frames = [self._load_frame(frame_paths[idx % len(frame_paths)]) for idx in indices]
        clip = self._preprocess_frames(frames)
        return {
            "clip": clip,
            "label_id": record.label_id,
            "label": record.label,
            "role": record.role,
            "clip_id": record.clip_id,
            "video_id": record.video_id,
            "center_frame": record.center_frame,
        }

    def __len__(self) -> int:
        return len(self.records)

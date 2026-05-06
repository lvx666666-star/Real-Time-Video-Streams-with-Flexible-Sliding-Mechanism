import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from edgeflex.data.clip_sampling import choose_crop, resize_nearest, uniform_temporal_indices


@dataclass
class UCF101Record:
    clip_name: str
    clip_path: str
    label: str
    normalized_label: str
    label_id: int


class UCF101BaseDataset:
    def __init__(
        self,
        root: str,
        subset: str,
        clip_len: int = 16,
        input_size: int = 112,
        sampling_span: int = 32,
        random_crop_mode: str = "five_point",
        horizontal_flip_prob: float = 0.5,
        training: bool = True,
        channel_mean: Sequence[float] = (114.7748, 107.7354, 99.4750),
        video_root: Optional[str] = None,
        frame_root: Optional[str] = None,
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
        test_split: Optional[str] = None,
        normalize_labels_from_clip_name: bool = True,
    ) -> None:
        self.root = Path(root)
        self.subset = subset
        self.clip_len = clip_len
        self.input_size = input_size
        self.sampling_span = sampling_span
        self.random_crop_mode = random_crop_mode
        self.horizontal_flip_prob = horizontal_flip_prob
        self.training = training
        self.channel_mean = np.array(channel_mean, dtype=np.float32).reshape(1, 1, 3)
        self.video_root = Path(video_root) if video_root else self.root / "UCF-101"
        self.frame_root = Path(frame_root) if frame_root else self.root / "rgb_frames_ucf101"
        self.normalize_labels_from_clip_name = normalize_labels_from_clip_name
        self.video_index: Optional[Dict[str, Path]] = None
        self.frame_index: Optional[Dict[str, Path]] = None
        self.split_paths = {
            "train": Path(train_split) if train_split else self.root / "train.csv",
            "val": Path(val_split) if val_split else self.root / "val.csv",
            "test": Path(test_split) if test_split else self.root / "test.csv",
        }
        self.label_to_id = self._build_label_mapping()
        self.records = self._load_records()

    def _canonical_label(self, clip_name: str, raw_label: str) -> str:
        if not self.normalize_labels_from_clip_name:
            return raw_label
        match = re.match(r"v_(.+?)_g\d+_c\d+$", clip_name)
        if match:
            return match.group(1)
        return raw_label

    def _build_label_mapping(self) -> Dict[str, int]:
        labels = set()
        for path in self.split_paths.values():
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    label = row.get("label")
                    clip_name = row.get("clip_name", "")
                    if label:
                        labels.add(self._canonical_label(clip_name, label))
        if not labels:
            raise FileNotFoundError("No labels found in UCF101 csv splits under {}".format(self.root))
        return {label: index for index, label in enumerate(sorted(labels))}

    def _load_records(self) -> List[UCF101Record]:
        path = self.split_paths[self.subset]
        if not path.exists():
            raise FileNotFoundError("Missing UCF101 split file: {}".format(path))
        records: List[UCF101Record] = []
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_label = row["label"]
                normalized_label = self._canonical_label(row["clip_name"], raw_label)
                records.append(
                    UCF101Record(
                        clip_name=row["clip_name"],
                        clip_path=row["clip_path"],
                        label=raw_label,
                        normalized_label=normalized_label,
                        label_id=self.label_to_id[normalized_label],
                    )
                )
        return records

    def _infer_class_name(self, record: UCF101Record) -> str:
        match = re.match(r"v_(.+?)_g\d+_c\d+$", record.clip_name)
        if match:
            return match.group(1)
        stem = Path(record.clip_path).stem
        match = re.match(r"v_(.+?)_g\d+_c\d+$", stem)
        if match:
            return match.group(1)
        return record.normalized_label

    def _sample_indices(self, num_frames: int) -> List[int]:
        if self.training:
            center_index = random.randint(0, max(num_frames - 1, 0))
        else:
            center_index = num_frames // 2
        return uniform_temporal_indices(
            num_frames=num_frames,
            clip_len=self.clip_len,
            center_index=center_index,
            sampling_span=self.sampling_span,
        )

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

    def __len__(self) -> int:
        return len(self.records)


class UCF101CSVVideoDataset(UCF101BaseDataset):
    def _resolve_video_path(self, record: UCF101Record) -> Path:
        raw_path = Path(record.clip_path)
        if raw_path.is_absolute() and raw_path.exists():
            return raw_path

        basename = Path(record.clip_path).name
        inferred_class = self._infer_class_name(record)
        candidates = [
            self.video_root / record.label / basename,
            self.video_root / record.normalized_label / basename,
            self.video_root / inferred_class / basename,
            self.root / record.clip_path.lstrip("/\\"),
            self.root / basename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        indexed = self._lookup_video_by_name(basename)
        if indexed is not None:
            return indexed
        raise FileNotFoundError("Unable to resolve video for {} from {}".format(record.clip_name, record.clip_path))

    def _lookup_video_by_name(self, basename: str) -> Optional[Path]:
        if self.video_index is None:
            self.video_index = {}
            for path in self.video_root.rglob("*"):
                if path.is_file():
                    self.video_index[path.name] = path
        return self.video_index.get(basename)

    def _decode_video(self, path: Path) -> List[np.ndarray]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for UCF101 video loading. Install it in your py3.10 environment first."
            ) from exc

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError("Failed to open video {}".format(path))

        frames: List[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        capture.release()
        if not frames:
            raise RuntimeError("Decoded zero frames from {}".format(path))
        return frames

    def __getitem__(self, index: int):
        record = self.records[index]
        video_path = self._resolve_video_path(record)
        all_frames = self._decode_video(video_path)
        indices = self._sample_indices(len(all_frames))
        frames = [all_frames[idx % len(all_frames)] for idx in indices]
        clip = self._preprocess_frames(frames)
        return {
            "clip": clip,
            "label_id": record.label_id,
            "label": record.normalized_label,
            "raw_label": record.label,
            "role": "target",
            "clip_id": record.clip_name,
            "video_id": record.clip_name,
            "video_path": str(video_path),
            "center_frame": indices[len(indices) // 2],
        }


class UCF101CSVFrameDataset(UCF101BaseDataset):
    def _resolve_frame_dir(self, record: UCF101Record) -> Path:
        candidates = [
            self.frame_root / record.label / record.clip_name,
            self.frame_root / record.normalized_label / record.clip_name,
            self.frame_root / self._infer_class_name(record) / record.clip_name,
            self.frame_root / record.clip_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        indexed = self._lookup_frame_dir_by_name(record.clip_name)
        if indexed is not None:
            return indexed
        raise FileNotFoundError("Unable to resolve frame directory for {}".format(record.clip_name))

    def _lookup_frame_dir_by_name(self, clip_name: str) -> Optional[Path]:
        if self.frame_index is None:
            self.frame_index = {}
            for path in self.frame_root.rglob("*"):
                if path.is_dir() and any(path.iterdir()):
                    self.frame_index[path.name] = path
        return self.frame_index.get(clip_name)

    def _load_frame_paths(self, frame_dir: Path) -> List[Path]:
        frames = sorted(
            [path for path in frame_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if not frames:
            raise FileNotFoundError("No frame images found in {}".format(frame_dir))
        return frames

    def _load_frame(self, path: Path) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for frame-based UCF101 loading.") from exc
        with Image.open(path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)

    def __getitem__(self, index: int):
        record = self.records[index]
        frame_dir = self._resolve_frame_dir(record)
        frame_paths = self._load_frame_paths(frame_dir)
        indices = self._sample_indices(len(frame_paths))
        frames = [self._load_frame(frame_paths[idx % len(frame_paths)]) for idx in indices]
        clip = self._preprocess_frames(frames)
        return {
            "clip": clip,
            "label_id": record.label_id,
            "label": record.normalized_label,
            "raw_label": record.label,
            "role": "target",
            "clip_id": record.clip_name,
            "video_id": record.clip_name,
            "frame_dir": str(frame_dir),
            "center_frame": indices[len(indices) // 2],
        }

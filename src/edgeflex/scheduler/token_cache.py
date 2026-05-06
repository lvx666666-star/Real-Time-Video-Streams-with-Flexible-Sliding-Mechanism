from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np


@dataclass
class CachedWindow:
    frame_ids: List[int] = field(default_factory=list)
    tokens: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))


@dataclass
class CacheResult:
    tokens: np.ndarray
    cache_hit_rate: float
    reused_frames: int
    new_frames: int


class OverlapAwareTokenCache:
    def __init__(self) -> None:
        self.window = CachedWindow()

    def reuse(
        self,
        frame_ids: Sequence[int],
        compute_new_tokens: Callable[[Sequence[int]], np.ndarray],
    ) -> CacheResult:
        if len(frame_ids) == 0:
            raise ValueError("frame_ids must not be empty")

        prev_ids = self.window.frame_ids
        overlap = 0
        max_overlap = min(len(prev_ids), len(frame_ids))
        for size in range(max_overlap, 0, -1):
            if list(prev_ids[-size:]) == list(frame_ids[:size]):
                overlap = size
                break

        reused_frames = overlap
        new_frame_ids = list(frame_ids[overlap:])
        new_tokens = compute_new_tokens(new_frame_ids)

        if overlap > 0 and self.window.tokens.size > 0:
            reused_tokens = self.window.tokens[:overlap]
            tokens = np.concatenate([reused_tokens, new_tokens], axis=0)
        else:
            tokens = new_tokens

        self.window = CachedWindow(frame_ids=list(frame_ids), tokens=tokens)
        hit_rate = float(reused_frames) / float(len(frame_ids))
        return CacheResult(
            tokens=tokens,
            cache_hit_rate=hit_rate,
            reused_frames=reused_frames,
            new_frames=len(frame_ids) - reused_frames,
        )


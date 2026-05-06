from typing import Dict, List

from edgeflex.data.thumos14 import BACKGROUND_LABEL, THUMOS14_CLASSES, VAGUE_LABEL


def role_for_center(center_frame: int, annotations: List[Dict], vague_margin: int) -> Dict[str, object]:
    for ann in annotations:
        if ann["label"] == VAGUE_LABEL:
            continue
        if ann["start_frame"] <= center_frame <= ann["end_frame"]:
            return {"label": ann["label"], "label_id": ann["label_id"], "role": "target"}

    for ann in annotations:
        if ann["label"] == VAGUE_LABEL and ann["start_frame"] <= center_frame <= ann["end_frame"]:
            return {"label": VAGUE_LABEL, "label_id": len(THUMOS14_CLASSES) + 1, "role": "vague"}

    for ann in annotations:
        if ann["label"] == VAGUE_LABEL:
            continue
        near_start = abs(center_frame - ann["start_frame"]) <= vague_margin
        near_end = abs(center_frame - ann["end_frame"]) <= vague_margin
        if near_start or near_end:
            return {"label": VAGUE_LABEL, "label_id": len(THUMOS14_CLASSES) + 1, "role": "vague"}

    return {"label": BACKGROUND_LABEL, "label_id": len(THUMOS14_CLASSES), "role": "background"}


def build_thumos14_clips(raw: Dict, clip_len: int, stride: int, vague_margin: int) -> Dict:
    clips = []
    for video_id, video_meta in raw["videos"].items():
        num_frames = int(video_meta["num_frames"])
        subset = video_meta["subset"]
        annotations = list(video_meta.get("annotations", []))
        for center_frame in range(0, num_frames, stride):
            label_meta = role_for_center(center_frame, annotations, vague_margin=vague_margin)
            clips.append(
                {
                    "clip_id": "{}_{:06d}".format(video_id, center_frame),
                    "video_id": video_id,
                    "subset": subset,
                    "center_frame": center_frame,
                    "num_frames": num_frames,
                    "label": label_meta["label"],
                    "label_id": label_meta["label_id"],
                    "role": label_meta["role"],
                }
            )
    return {
        "dataset": "thumos14",
        "clip_len": clip_len,
        "num_classes": len(THUMOS14_CLASSES) + 2,
        "num_target_classes": len(THUMOS14_CLASSES),
        "background_index": len(THUMOS14_CLASSES),
        "vague_index": len(THUMOS14_CLASSES) + 1,
        "clips": clips,
    }

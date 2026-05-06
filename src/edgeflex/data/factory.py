from edgeflex.config import ExperimentConfig
from edgeflex.data.thumos14 import THUMOS14ClipDataset
from edgeflex.data.ucf101 import UCF101CSVFrameDataset, UCF101CSVVideoDataset


def build_dataset(config: ExperimentConfig, subset: str, training: bool):
    if config.dataset.name.lower() == "thumos14":
        return THUMOS14ClipDataset(
            root=config.dataset.root,
            annotation_file=config.dataset.annotation_file,
            subset=subset,
            clip_len=config.dataset.clip_len,
            input_size=config.dataset.input_size,
            sampling_span=config.sampler.temporal_sampling_span,
            random_crop_mode=config.sampler.random_crop_mode,
            horizontal_flip_prob=config.sampler.horizontal_flip_prob,
            training=training,
        )
    if config.dataset.name.lower() == "ucf101_csv":
        return UCF101CSVVideoDataset(
            root=config.dataset.root,
            subset=subset,
            clip_len=config.dataset.clip_len,
            input_size=config.dataset.input_size,
            sampling_span=config.sampler.temporal_sampling_span,
            random_crop_mode=config.sampler.random_crop_mode,
            horizontal_flip_prob=config.sampler.horizontal_flip_prob,
            training=training,
            video_root=config.dataset.video_root,
            frame_root=config.dataset.frame_root,
            train_split=config.dataset.train_split,
            val_split=config.dataset.val_split,
            test_split=config.dataset.test_split,
        )
    if config.dataset.name.lower() == "ucf101_csv_frames":
        return UCF101CSVFrameDataset(
            root=config.dataset.root,
            subset=subset,
            clip_len=config.dataset.clip_len,
            input_size=config.dataset.input_size,
            sampling_span=config.sampler.temporal_sampling_span,
            random_crop_mode=config.sampler.random_crop_mode,
            horizontal_flip_prob=config.sampler.horizontal_flip_prob,
            training=training,
            video_root=config.dataset.video_root,
            frame_root=config.dataset.frame_root,
            train_split=config.dataset.train_split,
            val_split=config.dataset.val_split,
            test_split=config.dataset.test_split,
        )
    raise ValueError("Unsupported dataset: {}".format(config.dataset.name))

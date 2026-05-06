from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DatasetConfig:
    name: str
    root: str
    num_classes: int
    annotation_file: str = ""
    clip_len: int = 16
    input_size: int = 112
    temporal_loop_padding: bool = True
    video_root: Optional[str] = None
    frame_root: Optional[str] = None
    train_split: Optional[str] = None
    val_split: Optional[str] = None
    test_split: Optional[str] = None


@dataclass
class SamplerConfig:
    temporal_uniform: bool = True
    temporal_sampling_span: int = 32
    random_crop_mode: str = "five_point"
    horizontal_flip_prob: float = 0.5
    mean_subtraction: bool = True


@dataclass
class ModelConfig:
    name: str
    num_classes: int
    num_target_classes: Optional[int] = None
    input_channels: int = 3
    clip_len: int = 16
    input_size: int = 112
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    tubelet_size_t: int = 2
    tubelet_size_h: int = 16
    tubelet_size_w: int = 16
    dropout: float = 0.1
    attention_mode: str = "dense"
    knn_attention_ratio: float = 0.75
    use_cls_token: bool = False

    @property
    def output_dim(self) -> int:
        if self.num_target_classes is None:
            return self.num_classes
        return self.num_target_classes + 2


@dataclass
class SchedulerConfig:
    mode: str = "flexible"
    background_threshold: float = 0.95
    vague_threshold: float = 12.0
    gmm_std_factor: float = 2.5
    gmm_components: int = 3
    gmm_learning_rate: float = 0.05
    gmm_background_ratio: float = 0.7
    min_step: int = 1
    use_token_cache: bool = False
    use_uncertainty: bool = False
    entropy_threshold: float = 0.9
    margin_threshold: float = 0.1
    uncertainty_step_scale: float = 0.5


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    output_dir: str = "runs/default"
    pretrained: Optional[str] = None
    eval_subset: str = "val"
    log_interval: int = 20
    print_batch_logs: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset: DatasetConfig
    sampler: SamplerConfig
    model: ModelConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    dataset = DatasetConfig(**raw["dataset"])
    sampler = SamplerConfig(**raw.get("sampler", {}))
    model = ModelConfig(**raw["model"])
    scheduler = SchedulerConfig(**raw.get("scheduler", {}))
    runtime = RuntimeConfig(**raw.get("runtime", {}))
    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset=dataset,
        sampler=sampler,
        model=model,
        scheduler=scheduler,
        runtime=runtime,
    )


def load_config(path: str) -> ExperimentConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _from_dict(raw)

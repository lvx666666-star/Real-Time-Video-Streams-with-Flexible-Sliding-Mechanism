from edgeflex.config import ModelConfig

from .c3d import C3DRecognizer
from .lite_vt import LiteVideoTransformerRecognizer
from .mat_clip import MATClipRecognizer


def build_model(config: ModelConfig):
    if config.name.lower() == "c3d":
        return C3DRecognizer(
            num_classes=config.output_dim,
            num_target_classes=config.num_target_classes,
            in_channels=config.input_channels,
        )
    if config.name.lower() == "litevt":
        return LiteVideoTransformerRecognizer(
            num_classes=config.output_dim,
            num_target_classes=config.num_target_classes,
            in_channels=config.input_channels,
            clip_len=config.clip_len,
            input_size=config.input_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            tubelet_size=(config.tubelet_size_t, config.tubelet_size_h, config.tubelet_size_w),
            dropout=config.dropout,
        )
    if config.name.lower() == "matclip":
        return MATClipRecognizer(
            num_classes=config.output_dim,
            num_target_classes=config.num_target_classes,
            in_channels=config.input_channels,
            clip_len=config.clip_len,
            input_size=config.input_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            tubelet_size=(config.tubelet_size_t, config.tubelet_size_h, config.tubelet_size_w),
            dropout=config.dropout,
            attention_mode=config.attention_mode,
            knn_ratio=config.knn_attention_ratio,
            use_cls_token=config.use_cls_token,
        )
    raise ValueError("Unsupported model: {}".format(config.name))

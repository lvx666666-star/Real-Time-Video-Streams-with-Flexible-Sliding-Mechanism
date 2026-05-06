from typing import Callable, Optional, Tuple

from .common import require_torch
from .recognizer import ActionRecognizer

torch, nn = require_torch()


class TubeletEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        tubelet_size: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        batch, channels, t, h, w = x.shape
        x = x.flatten(3).transpose(2, 3)
        x = x.reshape(batch, t * h * w, channels)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DividedSTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, t: int, hw: int) -> torch.Tensor:
        batch, tokens, dim = x.shape

        xt = self.norm1(x).reshape(batch, t, hw, dim).transpose(1, 2).reshape(batch * hw, t, dim)
        yt, _ = self.temporal_attn(xt, xt, xt, need_weights=False)
        yt = yt.reshape(batch, hw, t, dim).transpose(1, 2).reshape(batch, tokens, dim)
        x = x + yt

        xs = self.norm2(x).reshape(batch, t, hw, dim).reshape(batch * t, hw, dim)
        ys, _ = self.spatial_attn(xs, xs, xs, need_weights=False)
        ys = ys.reshape(batch, t, hw, dim).reshape(batch, tokens, dim)
        x = x + ys

        x = x + self.mlp(self.norm3(x))
        return x


class TokenCacheOutput:
    def __init__(self, tokens: torch.Tensor, cache_hit_rate: float) -> None:
        self.tokens = tokens
        self.cache_hit_rate = cache_hit_rate


class LiteVideoTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        clip_len: int = 16,
        input_size: int = 112,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        tubelet_size: Tuple[int, int, int] = (2, 16, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.clip_len = clip_len
        self.input_size = input_size
        self.output_dim = embed_dim
        self.embed = TubeletEmbedding(in_channels, embed_dim, tubelet_size)
        self.temporal_tokens = clip_len // tubelet_size[0]
        self.spatial_tokens = (input_size // tubelet_size[1]) * (input_size // tubelet_size[2])
        self.pos_embed = nn.Parameter(torch.zeros(1, self.temporal_tokens * self.spatial_tokens, embed_dim))
        self.blocks = nn.ModuleList(
            [DividedSTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens + self.pos_embed
        for block in self.blocks:
            x = block(x, t=self.temporal_tokens, hw=self.spatial_tokens)
        return self.norm(x).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(self.tokenize(x))


class LiteVideoTransformerRecognizer(ActionRecognizer):
    def __init__(
        self,
        num_classes: int,
        num_target_classes: int = None,
        in_channels: int = 3,
        clip_len: int = 16,
        input_size: int = 112,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        tubelet_size: Tuple[int, int, int] = (2, 16, 16),
        dropout: float = 0.1,
    ) -> None:
        encoder = LiteVideoTransformerEncoder(
            in_channels=in_channels,
            clip_len=clip_len,
            input_size=input_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            tubelet_size=tubelet_size,
            dropout=dropout,
        )
        super().__init__(
            encoder=encoder,
            feature_dim=encoder.output_dim,
            num_classes=num_classes,
            num_target_classes=num_target_classes,
        )

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cached_tokens_fn: Optional[Callable[[torch.Tensor], TokenCacheOutput]] = None,
    ) -> Tuple[torch.Tensor, float]:
        if cached_tokens_fn is None:
            return self.forward(x), 0.0
        output = cached_tokens_fn(x)
        features = self.encoder.forward_tokens(output.tokens)
        return self.forward_logits_from_features(features), output.cache_hit_rate


LiteVideoTransformer = LiteVideoTransformerRecognizer

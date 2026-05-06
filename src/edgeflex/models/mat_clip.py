import math
from typing import Callable, Optional, Tuple

from .common import require_torch
from .lite_vt import TokenCacheOutput, TubeletEmbedding
from .recognizer import ActionRecognizer

torch, nn = require_torch()


class MATDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0, attention_mode: str = "dense", knn_ratio: float = 0.75) -> None:
        super().__init__()
        self.dropout = dropout
        self.attention_mode = attention_mode
        self.knn_ratio = knn_ratio

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.bmm(q, k.transpose(1, 2))
        if self.attention_mode == "knn":
            keep_tokens = max(1, int(math.ceil(scores.shape[-1] * self.knn_ratio)))
            topk_index = torch.topk(scores, k=keep_tokens, dim=-1, largest=True).indices
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, topk_index, 1.0)
            scores = torch.where(mask > 0, scores, torch.full_like(scores, float("-inf")))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.dropout(weights, p=self.dropout, train=self.training)
        return torch.bmm(weights, v)


class MATMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_mode: str = "dense",
        knn_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attention = MATDotProductAttention(
            dropout=dropout,
            attention_mode=attention_mode,
            knn_ratio=knn_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * (self.head_dim ** -0.5)
        q = q.reshape(batch * self.num_heads, tokens, self.head_dim)
        k = k.reshape(batch * self.num_heads, tokens, self.head_dim)
        v = v.reshape(batch * self.num_heads, tokens, self.head_dim)
        out = self.attention(q, k, v)
        out = out.reshape(batch, self.num_heads, tokens, self.head_dim).permute(0, 2, 1, 3)
        out = out.reshape(batch, tokens, channels)
        return self.proj(out)


class MATFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class MATEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attention_mode: str = "dense",
        knn_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.attn = MATMultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_mode=attention_mode,
            knn_ratio=knn_ratio,
        )
        self.ffn = MATFeedForward(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop1(self.attn(x)))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class MATClipEncoder(nn.Module):
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
        attention_mode: str = "dense",
        knn_ratio: float = 0.75,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.output_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.embed = TubeletEmbedding(in_channels, embed_dim, tubelet_size)
        temporal_tokens = clip_len // tubelet_size[0]
        spatial_tokens = (input_size // tubelet_size[1]) * (input_size // tubelet_size[2])
        num_tokens = temporal_tokens * spatial_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + int(use_cls_token), embed_dim))
        self.blocks = nn.ModuleList(
            [
                MATEncoderLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_mode=attention_mode,
                    knn_ratio=knn_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(x)
        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        return tokens

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens + self.pos_embed[:, : tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.cls_token is not None:
            return x[:, 0]
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(self.tokenize(x))


class MATClipRecognizer(ActionRecognizer):
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
        attention_mode: str = "dense",
        knn_ratio: float = 0.75,
        use_cls_token: bool = False,
    ) -> None:
        encoder = MATClipEncoder(
            in_channels=in_channels,
            clip_len=clip_len,
            input_size=input_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            tubelet_size=tubelet_size,
            dropout=dropout,
            attention_mode=attention_mode,
            knn_ratio=knn_ratio,
            use_cls_token=use_cls_token,
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

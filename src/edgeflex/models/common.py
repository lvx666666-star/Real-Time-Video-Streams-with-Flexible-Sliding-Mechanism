from typing import Any


def require_torch() -> Any:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for model execution. Install dependencies from requirements.txt."
        ) from exc
    return torch, nn


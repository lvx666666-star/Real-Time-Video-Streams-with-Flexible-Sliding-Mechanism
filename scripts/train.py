import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from edgeflex.config import ExperimentConfig, load_config
from edgeflex.data import build_dataset
from edgeflex.models.common import require_torch
from edgeflex.models.factory import build_model

torch, nn = require_torch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EdgeFlex recognizer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--pretrained", help="Optional checkpoint override for partial weight loading")
    parser.add_argument("--epochs", type=int, help="Optional epoch override")
    parser.add_argument("--batch-size", type=int, help="Optional batch size override")
    parser.add_argument("--num-workers", type=int, help="Optional dataloader worker override")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional train subset for smoke tests")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="Optional eval subset for smoke tests")
    parser.add_argument("--subset-seed", type=int, default=42, help="Random seed for subset sampling")
    parser.add_argument("--log-interval", type=int, help="Refresh batch-history files and plots every N batches")
    parser.add_argument(
        "--print-batch-logs",
        action="store_true",
        help="Print batch progress to terminal. Default is disabled; batch trends are still written to plots.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    lowered = requested.lower()
    if lowered.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if lowered.startswith("cuda"):
        print("CUDA requested but unavailable, falling back to CPU.")
    return torch.device("cpu")


def ensure_output_dir(config: ExperimentConfig, config_path: str) -> Path:
    output_dir = Path(config.runtime.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(ROOT) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    return output_dir


def maybe_limit_dataset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    generator = random.Random(seed)
    indices = sorted(generator.sample(range(len(dataset)), max_samples))
    return torch.utils.data.Subset(dataset, indices)


def create_dataloaders(
    config: ExperimentConfig,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
    subset_seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataset = build_dataset(config, subset="train", training=True)
    train_dataset = maybe_limit_dataset(train_dataset, max_train_samples, seed=subset_seed)
    eval_subset = config.runtime.eval_subset
    try:
        eval_dataset = build_dataset(config, subset=eval_subset, training=False)
    except Exception:
        if eval_subset != "test":
            eval_subset = "test"
            eval_dataset = build_dataset(config, subset=eval_subset, training=False)
        else:
            raise
    if len(eval_dataset) == 0 and eval_subset != "test":
        eval_subset = "test"
        eval_dataset = build_dataset(config, subset=eval_subset, training=False)
    if len(eval_dataset) == 0:
        raise ValueError("Evaluation dataset is empty for subset '{}'".format(eval_subset))
    eval_dataset = maybe_limit_dataset(eval_dataset, max_eval_samples, seed=subset_seed + 1)
    pin_memory = config.runtime.device.lower().startswith("cuda") and torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.runtime.batch_size,
        shuffle=True,
        num_workers=config.runtime.num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=pin_memory,
    )
    eval_loader.edgeflex_subset = eval_subset
    return train_loader, eval_loader


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    clips = batch["clip"].to(device=device, dtype=torch.float32)
    labels = batch["label_id"].to(device=device, dtype=torch.long)
    return clips, labels


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    log_interval: int = 20,
    stage_name: str = "train",
    epoch_index: int = 1,
    batch_recorder: Optional[Callable[[Dict[str, float]], None]] = None,
    print_batch_logs: bool = False,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    start_time = time.time()
    num_batches = len(loader)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader, start=1):
            clips, labels = move_batch_to_device(batch, device)
            logits = model(clips)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_examples += int(batch_size)
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_loss = total_loss / total_examples if total_examples > 0 else 0.0
            running_acc = float(total_correct) / float(total_examples) if total_examples > 0 else 0.0
            elapsed = time.time() - start_time
            should_flush = bool(
                log_interval > 0 and (batch_index == 1 or batch_index % log_interval == 0 or batch_index == num_batches)
            )

            if batch_recorder is not None:
                batch_recorder(
                    {
                        "epoch": epoch_index,
                        "stage": stage_name,
                        "batch_index": batch_index,
                        "num_batches": num_batches,
                        "global_step": (epoch_index - 1) * num_batches + batch_index,
                        "running_loss": running_loss,
                        "running_accuracy": running_acc,
                        "samples_seen": total_examples,
                        "elapsed_seconds": elapsed,
                        "flush": should_flush,
                    }
                )

            if should_flush and print_batch_logs:
                print(
                    "[{}] batch {}/{} | samples {} | loss {:.4f} | acc {:.4f} | elapsed {:.1f}s".format(
                        stage_name,
                        batch_index,
                        num_batches,
                        total_examples,
                        running_loss,
                        running_acc,
                        elapsed,
                    )
                )

    average_loss = total_loss / total_examples if total_examples > 0 else 0.0
    accuracy = float(total_correct) / float(total_examples) if total_examples > 0 else 0.0
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "num_examples": total_examples,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: ExperimentConfig,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "experiment_name": config.experiment_name,
        "model_name": config.model.name,
    }
    torch.save(checkpoint, path)


def save_history(path: Path, history: List[Dict[str, float]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def save_batch_history(path: Path, batch_history: List[Dict[str, float]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(batch_history, handle, indent=2)


def _line_points(values: List[float], x0: float, y0: float, width: float, height: float) -> str:
    if not values:
        return ""
    if len(values) == 1:
        x = x0 + width / 2.0
        y = y0 + height / 2.0
        return "{:.2f},{:.2f}".format(x, y)
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    points = []
    for index, value in enumerate(values):
        x = x0 + (width * index / float(len(values) - 1))
        norm = (value - vmin) / (vmax - vmin)
        y = y0 + height - (norm * height)
        points.append("{:.2f},{:.2f}".format(x, y))
    return " ".join(points)


def save_training_curves_svg(path: Path, history: List[Dict[str, float]]) -> None:
    if not history:
        return

    width = 960
    height = 540
    panel_width = 400
    panel_height = 180
    left_x = 80
    right_x = 500
    top_y = 90

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    eval_loss = [record["eval_loss"] for record in history]
    train_acc = [record["train_accuracy"] for record in history]
    eval_acc = [record["eval_accuracy"] for record in history]

    loss_min = min(train_loss + eval_loss)
    loss_max = max(train_loss + eval_loss)
    acc_min = min(train_acc + eval_acc)
    acc_max = max(train_acc + eval_acc)

    def axis_labels(vmin: float, vmax: float) -> Tuple[str, str]:
        return "{:.4f}".format(vmin), "{:.4f}".format(vmax)

    loss_low, loss_high = axis_labels(loss_min, loss_max)
    acc_low, acc_high = axis_labels(acc_min, acc_max)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbf8f2"/>
  <text x="60" y="45" font-family="Consolas, Menlo, monospace" font-size="26" fill="#1d1d1d">Training Curves</text>
  <text x="60" y="70" font-family="Consolas, Menlo, monospace" font-size="14" fill="#555">Updated after each epoch</text>

  <rect x="{left_x}" y="{top_y}" width="{panel_width}" height="{panel_height}" fill="#ffffff" stroke="#c8c3b8"/>
  <text x="{left_x}" y="{top_y - 18}" font-family="Consolas, Menlo, monospace" font-size="18" fill="#1d1d1d">Loss</text>
  <line x1="{left_x}" y1="{top_y + panel_height}" x2="{left_x + panel_width}" y2="{top_y + panel_height}" stroke="#777"/>
  <line x1="{left_x}" y1="{top_y}" x2="{left_x}" y2="{top_y + panel_height}" stroke="#777"/>
  <polyline fill="none" stroke="#c03a2b" stroke-width="3" points="{_line_points(train_loss, left_x, top_y, panel_width, panel_height)}"/>
  <polyline fill="none" stroke="#2874a6" stroke-width="3" points="{_line_points(eval_loss, left_x, top_y, panel_width, panel_height)}"/>
  <text x="{left_x}" y="{top_y + panel_height + 25}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">epoch 1</text>
  <text x="{left_x + panel_width - 55}" y="{top_y + panel_height + 25}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">epoch {epochs[-1]}</text>
  <text x="{left_x - 58}" y="{top_y + 12}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">{loss_high}</text>
  <text x="{left_x - 58}" y="{top_y + panel_height}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">{loss_low}</text>

  <rect x="{right_x}" y="{top_y}" width="{panel_width}" height="{panel_height}" fill="#ffffff" stroke="#c8c3b8"/>
  <text x="{right_x}" y="{top_y - 18}" font-family="Consolas, Menlo, monospace" font-size="18" fill="#1d1d1d">Accuracy</text>
  <line x1="{right_x}" y1="{top_y + panel_height}" x2="{right_x + panel_width}" y2="{top_y + panel_height}" stroke="#777"/>
  <line x1="{right_x}" y1="{top_y}" x2="{right_x}" y2="{top_y + panel_height}" stroke="#777"/>
  <polyline fill="none" stroke="#c03a2b" stroke-width="3" points="{_line_points(train_acc, right_x, top_y, panel_width, panel_height)}"/>
  <polyline fill="none" stroke="#2874a6" stroke-width="3" points="{_line_points(eval_acc, right_x, top_y, panel_width, panel_height)}"/>
  <text x="{right_x}" y="{top_y + panel_height + 25}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">epoch 1</text>
  <text x="{right_x + panel_width - 55}" y="{top_y + panel_height + 25}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">epoch {epochs[-1]}</text>
  <text x="{right_x - 58}" y="{top_y + 12}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">{acc_high}</text>
  <text x="{right_x - 58}" y="{top_y + panel_height}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">{acc_low}</text>

  <rect x="80" y="330" width="18" height="18" fill="#c03a2b"/>
  <text x="108" y="344" font-family="Consolas, Menlo, monospace" font-size="14" fill="#222">train</text>
  <rect x="180" y="330" width="18" height="18" fill="#2874a6"/>
  <text x="208" y="344" font-family="Consolas, Menlo, monospace" font-size="14" fill="#222">eval</text>

  <text x="80" y="390" font-family="Consolas, Menlo, monospace" font-size="14" fill="#333">Latest epoch: {epochs[-1]}</text>
  <text x="80" y="415" font-family="Consolas, Menlo, monospace" font-size="14" fill="#333">Train loss: {train_loss[-1]:.4f} | Eval loss: {eval_loss[-1]:.4f}</text>
  <text x="80" y="440" font-family="Consolas, Menlo, monospace" font-size="14" fill="#333">Train acc: {train_acc[-1]:.4f} | Eval acc: {eval_acc[-1]:.4f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def save_training_curves_matplotlib(path: Path, history: List[Dict[str, float]]) -> bool:
    if not history:
        return False
    try:
        mpl_config_dir = path.parent / ".mplconfig"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    eval_loss = [record["eval_loss"] for record in history]
    train_acc = [record["train_accuracy"] for record in history]
    eval_acc = [record["eval_accuracy"] for record in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    fig.patch.set_facecolor("#fbf8f2")

    axes[0].plot(epochs, train_loss, color="#c03a2b", linewidth=2.2, label="train")
    axes[0].plot(epochs, eval_loss, color="#2874a6", linewidth=2.2, label="eval")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, color="#c03a2b", linewidth=2.2, label="train")
    axes[1].plot(epochs, eval_acc, color="#2874a6", linewidth=2.2, label="eval")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("Training Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_training_curves(output_dir: Path, history: List[Dict[str, float]]) -> Tuple[Path, Path]:
    png_path = output_dir / "training_curves.png"
    svg_path = output_dir / "training_curves.svg"
    wrote_png = save_training_curves_matplotlib(png_path, history)
    if not wrote_png and png_path.exists():
        png_path.unlink()
    save_training_curves_svg(svg_path, history)
    return png_path, svg_path


def save_batch_curves_svg(path: Path, batch_history: List[Dict[str, float]]) -> None:
    if not batch_history:
        return

    width = 960
    height = 540
    panel_width = 400
    panel_height = 180
    left_x = 80
    right_x = 500
    top_y = 90

    train_rows = [row for row in batch_history if row["stage"] == "train"]
    eval_rows = [row for row in batch_history if row["stage"] == "eval"]

    train_loss = [row["running_loss"] for row in train_rows]
    eval_loss = [row["running_loss"] for row in eval_rows] or train_loss[-1:]
    train_acc = [row["running_accuracy"] for row in train_rows]
    eval_acc = [row["running_accuracy"] for row in eval_rows] or train_acc[-1:]

    loss_min = min(train_loss + eval_loss)
    loss_max = max(train_loss + eval_loss)
    acc_min = min(train_acc + eval_acc)
    acc_max = max(train_acc + eval_acc)

    def axis_labels(vmin: float, vmax: float) -> Tuple[str, str]:
        return "{:.4f}".format(vmin), "{:.4f}".format(vmax)

    loss_low, loss_high = axis_labels(loss_min, loss_max)
    acc_low, acc_high = axis_labels(acc_min, acc_max)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbf8f2"/>
  <text x="60" y="45" font-family="Consolas, Menlo, monospace" font-size="26" fill="#1d1d1d">Batch Curves</text>
  <text x="60" y="70" font-family="Consolas, Menlo, monospace" font-size="14" fill="#555">Updated during training</text>

  <rect x="{left_x}" y="{top_y}" width="{panel_width}" height="{panel_height}" fill="#ffffff" stroke="#c8c3b8"/>
  <text x="{left_x}" y="{top_y - 18}" font-family="Consolas, Menlo, monospace" font-size="18" fill="#1d1d1d">Running Loss</text>
  <line x1="{left_x}" y1="{top_y + panel_height}" x2="{left_x + panel_width}" y2="{top_y + panel_height}" stroke="#777"/>
  <line x1="{left_x}" y1="{top_y}" x2="{left_x}" y2="{top_y + panel_height}" stroke="#777"/>
  <polyline fill="none" stroke="#c03a2b" stroke-width="2.4" points="{_line_points(train_loss, left_x, top_y, panel_width, panel_height)}"/>
  <polyline fill="none" stroke="#2874a6" stroke-width="2.4" points="{_line_points(eval_loss, left_x, top_y, panel_width, panel_height)}"/>
  <text x="{left_x - 58}" y="{top_y + 12}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">{loss_high}</text>
  <text x="{left_x - 58}" y="{top_y + panel_height}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">{loss_low}</text>

  <rect x="{right_x}" y="{top_y}" width="{panel_width}" height="{panel_height}" fill="#ffffff" stroke="#c8c3b8"/>
  <text x="{right_x}" y="{top_y - 18}" font-family="Consolas, Menlo, monospace" font-size="18" fill="#1d1d1d">Running Accuracy</text>
  <line x1="{right_x}" y1="{top_y + panel_height}" x2="{right_x + panel_width}" y2="{top_y + panel_height}" stroke="#777"/>
  <line x1="{right_x}" y1="{top_y}" x2="{right_x}" y2="{top_y + panel_height}" stroke="#777"/>
  <polyline fill="none" stroke="#c03a2b" stroke-width="2.4" points="{_line_points(train_acc, right_x, top_y, panel_width, panel_height)}"/>
  <polyline fill="none" stroke="#2874a6" stroke-width="2.4" points="{_line_points(eval_acc, right_x, top_y, panel_width, panel_height)}"/>
  <text x="{right_x - 58}" y="{top_y + 12}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">{acc_high}</text>
  <text x="{right_x - 58}" y="{top_y + panel_height}" font-family="Consolas, Menlo, monospace" font-size="12" fill="#555">{acc_low}</text>

  <rect x="80" y="330" width="18" height="18" fill="#c03a2b"/>
  <text x="108" y="344" font-family="Consolas, Menlo, monospace" font-size="14" fill="#222">train batch trend</text>
  <rect x="250" y="330" width="18" height="18" fill="#2874a6"/>
  <text x="278" y="344" font-family="Consolas, Menlo, monospace" font-size="14" fill="#222">eval batch trend</text>

  <text x="80" y="390" font-family="Consolas, Menlo, monospace" font-size="14" fill="#333">Train points: {len(train_rows)} | Eval points: {len(eval_rows)}</text>
  <text x="80" y="415" font-family="Consolas, Menlo, monospace" font-size="14" fill="#333">Latest train loss/acc: {train_loss[-1]:.4f} / {train_acc[-1]:.4f}</text>
  <text x="80" y="440" font-family="Consolas, Menlo, monospace" font-size="14" fill="#333">Latest eval loss/acc: {eval_loss[-1]:.4f} / {eval_acc[-1]:.4f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def save_batch_curves_matplotlib(path: Path, batch_history: List[Dict[str, float]]) -> bool:
    if not batch_history:
        return False
    try:
        mpl_config_dir = path.parent / ".mplconfig"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    train_rows = [row for row in batch_history if row["stage"] == "train"]
    eval_rows = [row for row in batch_history if row["stage"] == "eval"]

    train_steps = [row["global_step"] for row in train_rows]
    eval_steps = [row["global_step"] for row in eval_rows]
    train_loss = [row["running_loss"] for row in train_rows]
    eval_loss = [row["running_loss"] for row in eval_rows]
    train_acc = [row["running_accuracy"] for row in train_rows]
    eval_acc = [row["running_accuracy"] for row in eval_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    fig.patch.set_facecolor("#fbf8f2")

    axes[0].plot(train_steps, train_loss, color="#c03a2b", linewidth=1.8, label="train")
    if eval_rows:
        axes[0].plot(eval_steps, eval_loss, color="#2874a6", linewidth=1.8, label="eval")
    axes[0].set_title("Running Loss by Batch")
    axes[0].set_xlabel("Global Batch Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(train_steps, train_acc, color="#c03a2b", linewidth=1.8, label="train")
    if eval_rows:
        axes[1].plot(eval_steps, eval_acc, color="#2874a6", linewidth=1.8, label="eval")
    axes[1].set_title("Running Accuracy by Batch")
    axes[1].set_xlabel("Global Batch Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("Batch Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_batch_curves(output_dir: Path, batch_history: List[Dict[str, float]]) -> Tuple[Path, Path]:
    png_path = output_dir / "batch_curves.png"
    svg_path = output_dir / "batch_curves.svg"
    wrote_png = save_batch_curves_matplotlib(png_path, batch_history)
    if not wrote_png and png_path.exists():
        png_path.unlink()
    save_batch_curves_svg(svg_path, batch_history)
    return png_path, svg_path


def maybe_load_pretrained(model: nn.Module, config: ExperimentConfig, device: torch.device) -> None:
    pretrained = config.runtime.pretrained
    if not pretrained:
        return
    checkpoint_path = Path(pretrained)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(ROOT) / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()

    matched_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing_in_target"))
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append((key, "shape_mismatch {} != {}".format(tuple(value.shape), tuple(model_state[key].shape))))
            continue
        matched_state[key] = value

    missing_after_load = [key for key in model_state.keys() if key not in matched_state]
    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=False)

    print("Loaded pretrained weights from", checkpoint_path)
    print("Matched tensors:", len(matched_state))
    print("Skipped tensors:", len(skipped))
    if skipped:
        preview = ", ".join("{} ({})".format(name, reason) for name, reason in skipped[:5])
        print("Skipped preview:", preview)
    if missing_after_load:
        preview = ", ".join(missing_after_load[:5])
        print("Target-only tensors preview:", preview)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.pretrained is not None:
        config.runtime.pretrained = args.pretrained
    if args.epochs is not None:
        config.runtime.epochs = args.epochs
    if args.batch_size is not None:
        config.runtime.batch_size = args.batch_size
    if args.num_workers is not None:
        config.runtime.num_workers = args.num_workers
    if args.log_interval is not None:
        config.runtime.log_interval = args.log_interval
    if args.print_batch_logs:
        config.runtime.print_batch_logs = True

    device = resolve_device(config.runtime.device)
    output_dir = ensure_output_dir(config, args.config)

    print("Experiment:", config.experiment_name)
    print("Model:", config.model.name)
    print("Dataset:", config.dataset.name)
    print("Scheduler:", config.scheduler.mode)
    print("Device:", device)
    print("Output Dir:", output_dir)

    train_loader, eval_loader = create_dataloaders(
        config,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        subset_seed=args.subset_seed,
    )
    print("Train clips:", len(train_loader.dataset))
    print("Eval subset:", getattr(eval_loader, "edgeflex_subset", config.runtime.eval_subset))
    print("Eval clips:", len(eval_loader.dataset))

    model = build_model(config.model).to(device)
    maybe_load_pretrained(model, config, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.runtime.learning_rate,
        weight_decay=config.runtime.weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_accuracy = -1.0
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    history_path = output_dir / "history.json"
    batch_history: List[Dict[str, float]] = []
    batch_history_path = output_dir / "batch_history.json"
    curves_png_path = output_dir / "training_curves.png"
    curves_svg_path = output_dir / "training_curves.svg"
    batch_curves_png_path = output_dir / "batch_curves.png"
    batch_curves_svg_path = output_dir / "batch_curves.svg"

    def record_batch(info: Dict[str, float]) -> None:
        nonlocal batch_curves_png_path, batch_curves_svg_path
        flush = bool(info.pop("flush"))
        batch_history.append(info)
        if flush:
            save_batch_history(batch_history_path, batch_history)
            batch_curves_png_path, batch_curves_svg_path = save_batch_curves(output_dir, batch_history)

    for epoch in range(1, config.runtime.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            log_interval=config.runtime.log_interval,
            stage_name="train",
            epoch_index=epoch,
            batch_recorder=record_batch,
            print_batch_logs=config.runtime.print_batch_logs,
        )
        eval_metrics = run_epoch(
            model,
            eval_loader,
            criterion,
            device,
            optimizer=None,
            log_interval=config.runtime.log_interval,
            stage_name="eval",
            epoch_index=epoch,
            batch_recorder=record_batch,
            print_batch_logs=config.runtime.print_batch_logs,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
        }
        history.append(record)
        save_history(history_path, history)
        curves_png_path, curves_svg_path = save_training_curves(output_dir, history)

        merged_metrics = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
        }
        save_checkpoint(latest_path, model, optimizer, epoch, merged_metrics, config)

        if eval_metrics["accuracy"] >= best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            save_checkpoint(best_path, model, optimizer, epoch, merged_metrics, config)

        print(
            "Epoch {}/{} | train_loss {:.4f} | train_acc {:.4f} | eval_loss {:.4f} | eval_acc {:.4f}".format(
                epoch,
                config.runtime.epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                eval_metrics["loss"],
                eval_metrics["accuracy"],
            )
        )

    print("Training finished.")
    print("Best eval accuracy:", round(best_accuracy, 4))
    print("Best checkpoint:", best_path)
    print("History:", history_path)
    print("Batch History:", batch_history_path)
    if curves_png_path.exists():
        print("Curves PNG:", curves_png_path)
    print("Curves SVG:", curves_svg_path)
    if batch_curves_png_path.exists():
        print("Batch Curves PNG:", batch_curves_png_path)
    print("Batch Curves SVG:", batch_curves_svg_path)


if __name__ == "__main__":
    main()

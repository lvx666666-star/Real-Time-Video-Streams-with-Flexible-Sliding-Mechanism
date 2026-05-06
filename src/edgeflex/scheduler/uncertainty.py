import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class UncertaintySummary:
    entropy: float
    top1_margin: float
    should_reduce_step: bool


def softmax(logits: Sequence[float]) -> Sequence[float]:
    max_logit = max(logits)
    exps = [math.exp(v - max_logit) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def summarize_uncertainty(
    logits: Iterable[float],
    entropy_threshold: float,
    margin_threshold: float,
) -> UncertaintySummary:
    probs = list(softmax(list(logits)))
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    should_reduce = entropy >= entropy_threshold or margin <= margin_threshold
    return UncertaintySummary(entropy=entropy, top1_margin=margin, should_reduce_step=should_reduce)


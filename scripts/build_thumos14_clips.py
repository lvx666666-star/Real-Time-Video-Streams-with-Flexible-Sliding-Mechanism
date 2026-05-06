import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from edgeflex.data.annotation_builder import build_thumos14_clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build THUMOS14 K+2 clip annotations")
    parser.add_argument("--segments", required=True, help="Path to source temporal segment JSON")
    parser.add_argument("--output", required=True, help="Path to output clip annotation JSON")
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--vague-margin", type=int, default=8)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    with open(args.segments, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    built = build_thumos14_clips(raw, clip_len=args.clip_len, stride=args.stride, vague_margin=args.vague_margin)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(built, handle, indent=2)
    print("Saved clip annotations to", output)
    print("Total clips:", len(built["clips"]))


if __name__ == "__main__":
    main()

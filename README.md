# Real-Time Video Streams with Flexible Sliding Mechanism

This repository provides the source code and configuration files for the paper:

**Edge Environment-Oriented Action Recognition Framework for Real-Time Video Streams with a Flexible Sliding Mechanism**

## Paper Reproduction

The core reproduction code includes:

- C3D-based clip-level action recognition baseline
- Frame-by-frame sliding baseline
- Flexible sliding mechanism
- GMM-based background-window skipping
- Inter-frame-difference-based vague-window skipping
- Streaming evaluation scripts
- Threshold sensitivity and robustness analysis scripts

## Additional Experimental Modules

This repository also includes optional experimental extensions, such as LiteVT, MATClip, token cache, and uncertainty-aware thresholding. These modules are provided for further exploration and are not the main contribution evaluated in the manuscript.

The project keeps the paper's scheduling skeleton unchanged:

- `16-frame` clip input
- `K + 2` classifier head (`K` actions + `Background` + `Vague`)
- `frame-by-frame` sliding when the current window is a target action
- `GMM + Pearson correlation` for background-window skipping
- `inter-frame difference` for vague-window skipping
- streaming evaluation with `FPS`, `Speed Ratio`, delay, average sliding step, skip ratio, and cache hit rate

## Experiment Matrix

Use these seven settings to separate paper contribution from your own:

- `B0`: `C3D + FrameByFrame`
- `B1`: `C3D + FlexibleSliding`
- `B2`: `LiteVT + FrameByFrame`
- `B3`: `LiteVT + FlexibleSliding`
- `B4`: `LiteVT + FlexibleSliding + TokenCache`
- `B5`: `LiteVT + FlexibleSliding + UncertaintyThreshold`
- `B6`: `LiteVT + FlexibleSliding + TokenCache + UncertaintyThreshold`

## Project Layout

```text
configs/                 YAML experiment presets
scripts/                 train / offline eval / streaming eval entrypoints
src/edgeflex/data/       clip sampling and labels
src/edgeflex/models/     C3D baseline and LiteVT backbone
src/edgeflex/scheduler/  flexible sliding, token cache, uncertainty logic
src/edgeflex/utils/      metrics
tests/                   pure-python checks for scheduler and sampler
```

## Environment

The current machine does not have `torch` installed, so the model code is provided but not executed here.
The pure scheduling logic is written with `numpy` and can be validated independently.

Recommended runtime dependencies:

- Python 3.9+
- `torch`
- `torchvision`
- `numpy`
- `pyyaml`
- `opencv-python`

See `requirements.txt`.

## Quick Start

1. Prepare annotations and extracted frames for THUMOS14 / ActivityNet1.3.
2. Pick a config from `configs/`.
3. Train a recognizer:

```bash
python scripts/train.py --config configs/b3_litevt_flex.yaml
```

4. Run offline ODAS evaluation:

```bash
python scripts/eval_offline.py --config configs/b3_litevt_flex.yaml
```

5. Run streaming evaluation:

```bash
python scripts/eval_streaming.py --config configs/b6_litevt_flex_cache_uncertainty.yaml
```

6. Run the full B0-B6 streaming matrix:

```bash
python scripts/eval_streaming.py --matrix
```

7. Build `K + 2` clip annotations from temporal segments:

```bash
python scripts/build_thumos14_clips.py \
  --segments data/thumos14/annotations/thumos14_segments.json \
  --output data/thumos14/annotations/thumos14_clip_annotations.json
```

8. Convert official THUMOS14 txt annotations into the native segment JSON:

```bash
python scripts/convert_thumos14_annotations.py \
  --annotation-dir data/thumos14/original_annotations \
  --frames-dir data/thumos14/frames \
  --output data/thumos14/annotations/thumos14_segments.json
```

9. Extract frames from raw videos:

```bash
python scripts/extract_video_frames.py \
  --videos-dir data/thumos14/videos/validation \
  --output-dir data/thumos14/frames
```

10. Run threshold sensitivity / robustness analysis:

```bash
python scripts/eval_threshold_sensitivity.py \
  --config configs/b3_litevt_flex.yaml \
  --output-dir results/threshold_sensitivity
```

## Notes

- The paper backbone is treated as a clip-level recognizer. The main research object remains the scheduling framework.
- `LiteVT` only replaces the clip-level recognizer.
- `MATClip` borrows transformer attention/encoder ideas from MAT, but keeps your clip-level recognition setting instead of MAT's long-memory anticipation framework.
- `TokenCache` reduces repeated tubelet/token computation for overlapped streaming windows.
- `UncertaintyThreshold` adaptively adjusts skip behavior when the classifier is uncertain around `Background` / `Vague`.
- The current workspace validates data sampling and scheduling logic locally. Full training still requires a Python environment with `torch`.
- The current `eval_streaming.py` output is a deterministic scaffold for matrix bookkeeping and table generation. Replace its simulated stream with real THUMOS14 / ActivityNet1.3 streams for paper results.
- See [data_format.md](c:\Users\lvxiang\Desktop\论文复现\docs\data_format.md) for the native `16-frame clip + K+2` dataset format used by this project.
- See [threshold_sensitivity_plan.md](c:\Users\lvxiang\Desktop\论文复现\docs\threshold_sensitivity_plan.md) for the rebuttal-oriented threshold experiment design.

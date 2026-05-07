# EdgeFlex: Real-Time Video Streams with Flexible Sliding Mechanism

This repository provides the source code, configuration files, and reproducibility materials for the manuscript:

**Edge Environment-Oriented Action Recognition Framework for Real-Time Video Streams with a Flexible Sliding Mechanism**

The project implements an ODAS-oriented edge action recognition framework for real-time video streams. The core idea is to reduce redundant temporal-window inference on resource-constrained edge devices through a flexible sliding mechanism.

---

## Overview

Real-time video streams in edge environments often contain a large amount of background content, pre-action context, repeated motion, and non-target interference. Conventional frame-by-frame sliding evaluates every adjacent temporal window, which causes redundant computation and may lead to accumulated delay on edge devices.

This repository implements a flexible sliding framework that dynamically determines the temporal sliding distance according to the state of the current window and the pre-labeling results of incoming frames.

The main components include:

- C3D-based clip-level action recognition backbone
- Frame-by-frame sliding baseline
- Flexible sliding mechanism
- GMM-based background-window skipping
- Pearson-correlation-based background similarity measurement
- Inter-frame-difference-based vague-window skipping
- ODAS evaluation using point-level action-start mAP
- Streaming evaluation using throughput, delay, skipped-window ratio, and average sliding step
- Threshold sensitivity and joint robustness analysis
- Backbone-level ablation under the same scheduling framework

The recognition backbone is treated as a replaceable clip-level component. The main research object of the manuscript is the adaptive inference scheduling framework for redundant-inference reduction in resource-constrained edge streaming scenarios.

---

## Main Paper Settings

The manuscript mainly evaluates the following settings.

| Setting | Backbone | Sliding Strategy | Purpose |
|---|---|---|---|
| B0 | C3D | Frame-by-frame sliding | Controlled baseline |
| B1 | C3D | Flexible sliding | Proposed framework |
| B2 | DSConv-GRU | Flexible sliding | Lightweight backbone comparison |
| B3 | MobileNet3D | Flexible sliding | Lightweight backbone comparison |
| B4 | Lite-3D CNN | Flexible sliding | Lightweight backbone comparison |

The main manuscript results are produced from these settings, including:

- ODAS performance on THUMOS14
- End-to-end stream throughput on THUMOS14 and ActivityNet1.3
- Controlled efficiency-accuracy ablation
- Backbone-level ablation
- Background-threshold sensitivity analysis
- Vague-window-threshold sensitivity analysis
- Joint threshold robustness analysis
- Group-wise streaming analysis
- Real-time prototype evaluation on edge video streams

Optional experimental modules are provided separately and are not required to reproduce the main manuscript results.

---

## Project Layout

```text
configs/                 YAML experiment configurations
scripts/                 Training, evaluation, and table-generation entrypoints
src/edgeflex/data/       Dataset loading, clip sampling, and K+2 label construction
src/edgeflex/models/     C3D and lightweight clip-level recognition backbones
src/edgeflex/scheduler/  Frame-by-frame and flexible sliding schedulers
src/edgeflex/utils/      Metrics, logging, reproducibility, and result parsing
tests/                   Unit tests for sampling, scheduling, and metric utilities
docs/                    Data format, edge-device setup, and reproducibility documents
results/                 Evaluation outputs, logs, and generated tables
checkpoints/             Trained model weights or links to released checkpoints
```

---

## Environment

The recommended software environment is:

```text
Python >= 3.9
PyTorch >= 1.12
torchvision
numpy
pyyaml
opencv-python
scikit-learn
tqdm
pandas
matplotlib
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

For edge-device inference, the prototype experiments use an NVIDIA TX2 development board with TensorRT acceleration. The hardware and software settings should be recorded in:

```text
docs/edge_device_setup.md
```

Recommended edge-device information to report includes:

- device model
- CPU/GPU mode
- memory size
- power mode
- CUDA version
- cuDNN version
- TensorRT version
- batch size
- inference precision: FP32 / FP16 / INT8
- whether preprocessing and I/O time are included

---

## Dataset Preparation

The raw THUMOS14 and ActivityNet1.3 videos are not redistributed in this repository due to dataset licensing restrictions. Please download them from their official sources and organize them as follows:

```text
data/
  thumos14/
    videos/
    frames/
    annotations/
  activitynet13/
    videos/
    frames/
    annotations/
```

### Step 1: Extract frames

```bash
python scripts/extract_video_frames.py \
  --videos-dir data/thumos14/videos/validation \
  --output-dir data/thumos14/frames
```

### Step 2: Convert official THUMOS14 annotations

```bash
python scripts/convert_thumos14_annotations.py \
  --annotation-dir data/thumos14/original_annotations \
  --frames-dir data/thumos14/frames \
  --output data/thumos14/annotations/thumos14_segments.json
```

### Step 3: Build 16-frame K+2 clip annotations

```bash
python scripts/build_thumos14_clips.py \
  --segments data/thumos14/annotations/thumos14_segments.json \
  --output data/thumos14/annotations/thumos14_clip_annotations.json
```

The native data format is described in:

```text
docs/data_format.md
```

---

## Training

Train the C3D frame-by-frame baseline:

```bash
python scripts/train.py \
  --config configs/b0_c3d_frame_by_frame.yaml
```

Train the C3D model used by the proposed flexible sliding framework:

```bash
python scripts/train.py \
  --config configs/b1_c3d_flexible_sliding.yaml
```

Each training configuration specifies:

- dataset path
- temporal window length
- input resolution
- batch size
- optimizer
- learning rate schedule
- random seed
- output checkpoint path

Each run saves the configuration, random seed, command line, and training log to the corresponding output directory.

---

## ODAS Evaluation

Run offline ODAS evaluation on THUMOS14 for the frame-by-frame baseline:

```bash
python scripts/eval_offline.py \
  --config configs/b0_c3d_frame_by_frame.yaml \
  --checkpoint checkpoints/c3d_frame_by_frame.pth \
  --output-dir results/odas/thumos14_b0
```

Run offline ODAS evaluation on THUMOS14 for the proposed flexible sliding framework:

```bash
python scripts/eval_offline.py \
  --config configs/b1_c3d_flexible_sliding.yaml \
  --checkpoint checkpoints/c3d_flexible_sliding.pth \
  --output-dir results/odas/thumos14_b1
```

The evaluation reports point-level action-start mAP under temporal offset thresholds from 1 s to 10 s.

---

## Streaming Evaluation

Run the frame-by-frame streaming baseline:

```bash
python scripts/eval_streaming.py \
  --config configs/b0_c3d_frame_by_frame.yaml \
  --checkpoint checkpoints/c3d_frame_by_frame.pth \
  --output-dir results/streaming/thumos14_b0
```

Run the proposed flexible sliding framework:

```bash
python scripts/eval_streaming.py \
  --config configs/b1_c3d_flexible_sliding.yaml \
  --checkpoint checkpoints/c3d_flexible_sliding.pth \
  --output-dir results/streaming/thumos14_b1
```

The streaming evaluation reports:

- total processing time
- end-to-end input-stream throughput
- processed temporal windows
- skipped-window ratio
- average sliding step
- recognition delay
- point-level action-start mAP
- retained p-mAP compared with frame-by-frame sliding

---

## Metric Definitions

### End-to-end input-stream throughput

```text
number of input frames / total wall-clock processing time
```

This metric reflects the effective stream-processing capability of the system.

### Processed-window FPS

```text
number of evaluated temporal windows / model inference time
```

This metric reflects the actual recognition-model evaluation speed.

### Skipped-window ratio

```text
1 - processed windows / frame-by-frame processed windows
```

This metric measures how many redundant temporal-window evaluations are removed by flexible sliding.

### Retained p-mAP

```text
p-mAP of flexible sliding / p-mAP of frame-by-frame sliding
```

This metric measures how much ODAS accuracy is retained after redundant-window skipping.

### Recognition delay

```text
detected action-start time - ground-truth action-start time
```

This metric measures the latency of action-start detection in streaming scenarios.

---

## Threshold Sensitivity and Robustness Analysis

Run background-threshold sensitivity analysis:

```bash
python scripts/eval_threshold_sensitivity.py \
  --config configs/b1_c3d_flexible_sliding.yaml \
  --checkpoint checkpoints/c3d_flexible_sliding.pth \
  --mode background \
  --output-dir results/threshold_sensitivity/background
```

Run vague-window-threshold sensitivity analysis:

```bash
python scripts/eval_threshold_sensitivity.py \
  --config configs/b1_c3d_flexible_sliding.yaml \
  --checkpoint checkpoints/c3d_flexible_sliding.pth \
  --mode vague \
  --output-dir results/threshold_sensitivity/vague
```

Run joint robustness analysis over background and vague-window thresholds:

```bash
python scripts/eval_threshold_sensitivity.py \
  --config configs/b1_c3d_flexible_sliding.yaml \
  --checkpoint checkpoints/c3d_flexible_sliding.pth \
  --mode joint \
  --output-dir results/threshold_sensitivity/joint
```

The threshold ranges used in the manuscript are specified in the corresponding YAML configuration files. Each run saves:

```text
config.yaml
command.txt
metrics.json
metrics.csv
raw_log.txt
table.csv
```

---

## Backbone-Level Ablation

Run the lightweight backbone comparison under the same flexible sliding framework:

```bash
python scripts/eval_backbone_ablation.py \
  --configs \
    configs/b1_c3d_flexible_sliding.yaml \
    configs/b2_dsconv_gru_flexible_sliding.yaml \
    configs/b3_mobilenet3d_flexible_sliding.yaml \
    configs/b4_lite3d_flexible_sliding.yaml \
  --output-dir results/backbone_ablation
```

This experiment compares C3D with lightweight alternatives under the same:

- input resolution
- temporal window length
- preprocessing pipeline
- dataset split
- inference backend
- edge-device hardware setting

---

## Group-Wise Streaming Analysis

Run group-wise streaming analysis:

```bash
python scripts/eval_groupwise_streaming.py \
  --config configs/b1_c3d_flexible_sliding.yaml \
  --checkpoint checkpoints/c3d_flexible_sliding.pth \
  --output-dir results/groupwise_streaming
```

The videos are grouped according to streaming characteristics, such as:

- action density
- background ratio
- video duration

This analysis is used to examine whether the flexible sliding mechanism maintains stable efficiency-accuracy trade-offs under different stream distributions.

---

## Table Generation

Generate manuscript tables from saved evaluation logs:

```bash
python scripts/generate_tables.py \
  --results-dir results \
  --output-dir results/tables
```

The generated files include CSV and LaTeX outputs for the main experimental tables.

---

## Reproducibility Checklist

For each reported result, this repository provides or specifies:

- experiment configuration file
- random seed
- dataset split
- checkpoint path or checkpoint release link
- command line used for evaluation
- raw evaluation log
- processed CSV result file
- table-generation script
- hardware and software environment

The recommended reproduction order is:

1. Prepare THUMOS14 and ActivityNet1.3 frames and annotations.
2. Build 16-frame K+2 clip annotations.
3. Train or download the C3D checkpoint.
4. Run B0 frame-by-frame evaluation.
5. Run B1 flexible sliding evaluation.
6. Run ODAS evaluation on THUMOS14.
7. Run streaming evaluation on THUMOS14 and ActivityNet1.3.
8. Run threshold sensitivity and joint robustness analyses.
9. Run backbone-level ablation.
10. Run group-wise streaming analysis.
11. Generate manuscript tables from saved logs.

---

## Optional Experimental Extensions

This repository may also include optional modules for further research, such as:

- LiteVT-style lightweight clip recognizer
- MAT-inspired clip-level attention module
- token cache for overlapped streaming windows
- uncertainty-aware threshold adjustment

These modules are not required for reproducing the main manuscript results unless explicitly stated in a configuration file or supplementary experiment. They are provided for further exploration of efficient edge video understanding.

---

## Notes on Dataset Licensing

This repository does not redistribute raw THUMOS14 or ActivityNet1.3 videos. Users should download the datasets from their official sources and follow the respective licenses.

The provided scripts convert official annotations into the internal 16-frame clip format used by this project.

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{edgeflex2026,
  title        = {Edge Environment-Oriented Action Recognition Framework for Real-Time Video Streams with a Flexible Sliding Mechanism},
  author       = {Zhai, Zhongyi and Lv, Xiang and Li, Shun and Cheng, Bo and Zhao, Lingzhong and Qian, Junyan},
  year         = {2026},
  note         = {Manuscript under review}
}
```

---

## Contact

For questions about the implementation or reproduction process, please contact the corresponding author or open an issue in this repository.

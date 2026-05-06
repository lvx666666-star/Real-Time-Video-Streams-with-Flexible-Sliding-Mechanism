# Grouped Generalization Report

Subset: `test`
B0 config: `configs\b0_c3d_ff_subset200.yaml`
B1 config: `configs\b1_c3d_flex_subset200.yaml`

## Split Summary

[
  {
    "metric": "action_density",
    "threshold": 3.6037161935778537,
    "lower_group": "action_sparse",
    "lower_count": 100,
    "upper_group": "action_dense",
    "upper_count": 100
  },
  {
    "metric": "background_ratio",
    "threshold": 0.7096927287258603,
    "lower_group": "low_background",
    "lower_count": 100,
    "upper_group": "high_background",
    "upper_count": 100
  },
  {
    "metric": "duration_seconds",
    "threshold": 173.26666666666665,
    "lower_group": "short_duration",
    "lower_count": 100,
    "upper_group": "long_duration",
    "upper_count": 100
  },
  {
    "metric": "ambiguous_ratio",
    "threshold": 0.0,
    "lower_group": "ambiguous_light",
    "lower_count": 100,
    "upper_group": "ambiguous_rich",
    "upper_count": 100
  }
]

## Grouped Results

| group_name | num_videos | frame_by_frame_p_mAP_mean | flexible_sliding_p_mAP_mean | p_mAP_delta | frame_by_frame_fps | flexible_sliding_fps | fps_gain | speed_ratio | frame_by_frame_delay | flexible_sliding_delay | delay_delta | flexible_sliding_skipped_window_ratio | processed_windows_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| action_sparse | 100 | 0.75% | 0.57% | -0.18% | 10.7427 | 14.0500 | 3.3072 | 1.3079 | 2.8670 | 3.1996 | 0.3326 | 18.84% | 18.84% |
| action_dense | 100 | 2.52% | 1.77% | -0.76% | 10.7420 | 14.3179 | 3.5759 | 1.3329 | 2.9948 | 2.9413 | -0.0535 | 20.37% | 20.37% |
| low_background | 100 | 1.73% | 1.40% | -0.33% | 10.7426 | 14.0739 | 3.3313 | 1.3101 | 3.0889 | 3.0576 | -0.0314 | 18.98% | 18.98% |
| high_background | 100 | 1.86% | 0.95% | -0.91% | 10.7422 | 14.2596 | 3.5174 | 1.3274 | 2.3441 | 2.6321 | 0.2880 | 20.04% | 20.04% |
| short_duration | 100 | 2.22% | 1.78% | -0.44% | 10.7397 | 15.2732 | 4.5335 | 1.4221 | 3.2680 | 2.9124 | -0.3556 | 25.36% | 25.36% |
| long_duration | 100 | 1.33% | 1.11% | -0.22% | 10.7432 | 13.8535 | 3.1102 | 1.2895 | 2.8168 | 3.0151 | 0.1983 | 17.69% | 17.69% |
| ambiguous_light | 100 | 1.78% | 1.39% | -0.39% | 10.7424 | 14.5709 | 3.8285 | 1.3564 | 2.3271 | 2.6450 | 0.3180 | 21.75% | 21.75% |
| ambiguous_rich | 100 | 1.32% | 0.90% | -0.42% | 10.7424 | 13.7757 | 3.0332 | 1.2824 | 3.5333 | 3.2745 | -0.2588 | 17.23% | 17.23% |

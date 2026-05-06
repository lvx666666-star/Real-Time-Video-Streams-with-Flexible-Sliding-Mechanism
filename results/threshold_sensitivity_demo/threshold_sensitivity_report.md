# Threshold Sensitivity Report

Base config: `configs\b3_litevt_flex.yaml`

## T_bg Sensitivity

| experiment_name | analysis_note | t_bg | t_vg | p_mAP_proxy | fps | speed_ratio | average_sliding_step | skipped_window_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b3_litevt_flex | T_bg sensitivity | 0.8800 | 12.0000 | 0.6609 | 38.0000 | 8.8372 | 6.3333 | 0.6667 |
| b3_litevt_flex | T_bg sensitivity | 0.9100 | 12.0000 | 0.6645 | 38.0000 | 8.8372 | 6.3333 | 0.6667 |
| b3_litevt_flex | T_bg sensitivity | 0.9500 | 12.0000 | 0.6693 | 38.0000 | 8.8372 | 6.3333 | 0.6667 |
| b3_litevt_flex | T_bg sensitivity | 0.9900 | 12.0000 | 0.6645 | 38.0000 | 8.8372 | 6.3333 | 0.6667 |

## T_vg Sensitivity

| experiment_name | analysis_note | t_bg | t_vg | p_mAP_proxy | fps | speed_ratio | average_sliding_step | skipped_window_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b3_litevt_flex | T_vg sensitivity | 0.9500 | 8.0000 | 0.6593 | 38.0000 | 8.8372 | 6.3333 | 0.6667 |
| b3_litevt_flex | T_vg sensitivity | 0.9500 | 12.0000 | 0.6693 | 38.0000 | 8.8372 | 6.3333 | 0.6667 |
| b3_litevt_flex | T_vg sensitivity | 0.9500 | 16.0000 | 0.6593 | 38.0000 | 8.8372 | 6.3333 | 0.6667 |

## Joint Robustness Heatmap: p_mAP_proxy

| T_bg \ T_vg | 8 | 12 | 16 |
| --- | --- | --- | --- |
| 0.88 | 0.6509 | 0.6609 | 0.6509 |
| 0.91 | 0.6545 | 0.6645 | 0.6545 |
| 0.95 | 0.6593 | 0.6693 | 0.6593 |
| 0.99 | 0.6545 | 0.6645 | 0.6545 |

## Joint Robustness Heatmap: speed_ratio

| T_bg \ T_vg | 8.00 | 12.00 | 16.00 |
| --- | --- | --- | --- |
| 0.88 | 8.8372 | 8.8372 | 8.8372 |
| 0.91 | 8.8372 | 8.8372 | 8.8372 |
| 0.95 | 8.8372 | 8.8372 | 8.8372 |
| 0.99 | 8.8372 | 8.8372 | 8.8372 |

Note: `p_mAP_proxy` is a deterministic placeholder for experiment bookkeeping only.
Replace it with real THUMOS14 p-mAP once the offline ODAS evaluator is wired in.

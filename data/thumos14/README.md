# THUMOS14 Workspace Layout

This folder is the native dataset workspace for the current project.

## What goes where

- `videos/validation/`
  Put the original THUMOS14 validation videos here.
- `videos/test/`
  Put the original THUMOS14 test videos here.
- `frames/`
  Put extracted RGB frames here, one folder per video.
- `original_annotations/`
  Put the official THUMOS14 annotation txt files here, including `detclasslist.txt`.
- `annotations/thumos14_segments.json`
  This is the converted temporal-segment JSON used by the project.
- `annotations/thumos14_clip_annotations.json`
  This is the generated `K + 2` clip-level annotation file for training the recognizer.
- `annotations/thumos14_stream_annotations.json`
  This is reserved for stream-level evaluation metadata.
- `splits/train.txt`, `splits/val.txt`, `splits/test.txt`
  Optional video-id split lists if you want explicit split files.

## Workflow

1. Put the official videos into `videos/validation/` and `videos/test/`.
2. Put the official annotation txt files into `original_annotations/`.
3. Extract frames into `frames/`.
4. Convert txt annotations into `annotations/thumos14_segments.json`.
5. Build `annotations/thumos14_clip_annotations.json`.


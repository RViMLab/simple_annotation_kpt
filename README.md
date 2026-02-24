# Simple Annotation KPT

This tool is a lightweight, no-AI version of the annotation pipeline.

It supports:
- `extract`: open a video (or image folder), split/export frames, and initialize dataset files.
- `review`: open a GUI to edit keypoints (frame navigation, add/delete/move points, category/point-id selection, save).

## Outputs

For input `<name>`, outputs are written to:
- `outputs/<name>/images/*`
- `outputs/<name>/center_points_masks_annotations.json`
- `outputs/<name>/keypoints.json`

The JSON structure is kept compatible with `auto_annotation_pipeline`.

## Usage

### 1) Extract dataset (no AI)

```bash
python -u simple_annotation_kpt/main.py extract \
  --input /path/to/video_or_image_folder \
  --output outputs \
  --frame-step 1
```

### 2) Review keypoints

```bash
python -u simple_annotation_kpt/main.py review outputs/<name>
```

## Reviewer Controls

- `Q`: quit
- `A` / `D`: previous / next frame
- `S`: save `keypoints.json`
- `1-5`: set category
- `Z/X/C/V`: set point id to `1/2/3/4`
- `Left click`: select point and drag to move
- `Ctrl + Left click`: add point using current category and point id
- `Right click`: delete nearest point

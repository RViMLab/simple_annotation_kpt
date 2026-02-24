import os
from typing import List, Tuple

import cv2

from .adapter import save_coco_format, save_tracks_format


def _list_images(input_dir: str) -> List[str]:
    return sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ]
    )


def _ensure_dirs(output_dir: str) -> str:
    images_out_dir = os.path.join(output_dir, "images")
    os.makedirs(images_out_dir, exist_ok=True)
    return images_out_dir


def _scene_name(input_path: str) -> str:
    return os.path.splitext(os.path.basename(input_path))[0]


def _extract_from_video(
    input_path: str,
    images_out_dir: str,
    frame_step: int,
) -> List[dict]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    images_info: List[dict] = []
    frame_idx = 0

    while True:
        if frame_idx % frame_step == 0:
            ret, frame = cap.read()
            if not ret:
                break
            file_name = f"{frame_idx:08d}.jpg"
            out_path = os.path.join(images_out_dir, file_name)
            cv2.imwrite(out_path, frame)

            h, w = frame.shape[:2]
            images_info.append(
                {
                    "file_name": file_name,
                    "height": int(h),
                    "width": int(w),
                    "global_frame_index": int(frame_idx),
                }
            )
        else:
            ret = cap.grab()
            if not ret:
                break
        frame_idx += 1

    cap.release()
    return images_info


def _extract_from_image_folder(
    input_path: str,
    images_out_dir: str,
    frame_step: int,
) -> List[dict]:
    files = _list_images(input_path)
    images_info: List[dict] = []

    for idx, file_name in enumerate(files):
        if idx % frame_step != 0:
            continue
        src_path = os.path.join(input_path, file_name)
        frame = cv2.imread(src_path)
        if frame is None:
            continue

        out_path = os.path.join(images_out_dir, file_name)
        cv2.imwrite(out_path, frame)

        h, w = frame.shape[:2]
        images_info.append(
            {
                "file_name": file_name,
                "height": int(h),
                "width": int(w),
                "global_frame_index": int(idx),
            }
        )

    return images_info


def run_extract(input_path: str, output_root: str, frame_step: int = 1) -> Tuple[str, int]:
    """
    Build a review-ready dataset without any AI model:
    - output/<scene_name>/images/*
    - output/<scene_name>/center_points_masks_annotations.json
    - output/<scene_name>/keypoints.json
    """
    frame_step = max(1, int(frame_step))
    scene_name = _scene_name(input_path)
    output_dir = os.path.join(output_root, scene_name)
    images_out_dir = _ensure_dirs(output_dir)

    if os.path.isdir(input_path):
        images_info = _extract_from_image_folder(input_path, images_out_dir, frame_step)
    else:
        images_info = _extract_from_video(input_path, images_out_dir, frame_step)

    save_coco_format(images_info, output_dir)
    save_tracks_format({}, output_dir)

    print(f"[Extract] Scene: {scene_name}")
    print(f"[Extract] Output: {output_dir}")
    print(f"[Extract] Frames exported: {len(images_info)}")
    return output_dir, len(images_info)

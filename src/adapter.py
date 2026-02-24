import json
import os
from typing import Dict, List, Any


def save_coco_format(images_info: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Save center_points_masks_annotations.json with the same base structure
    as auto_annotation_pipeline.
    """
    path = os.path.join(output_dir, "center_points_masks_annotations.json")
    data = {"images": images_info, "annotations": [], "categories": []}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_tracks_format(tracks_dict: Dict[int, List[Dict[str, Any]]], output_dir: str) -> None:
    """
    Save keypoints.json in the same structure as auto_annotation_pipeline.
    tracks_dict format:
    {
      category_id: [
        {"frame_index": int, "x": float, "y": float, "score": float, "point_id": int?},
        ...
      ]
    }
    """
    path = os.path.join(output_dir, "keypoints.json")
    formatted_tracks = []

    for cat_id, points in tracks_dict.items():
        points.sort(key=lambda x: x["frame_index"])
        clean_points = []
        for p in points:
            pt_data = {
                "frame_index": int(p["frame_index"]),
                "x": float(p["x"]),
                "y": float(p["y"]),
                "score": float(p["score"]),
            }
            if "point_id" in p:
                pt_data["point_id"] = int(p["point_id"])
            if "audit_score" in p:
                pt_data["audit_score"] = p["audit_score"]
            if "audit_action" in p:
                pt_data["audit_action"] = p["audit_action"]
            if "audit_error" in p:
                pt_data["audit_error"] = p["audit_error"]
            clean_points.append(pt_data)

        formatted_tracks.append({"category_id": int(cat_id), "track": clean_points})

    final_json = {"tracks": formatted_tracks}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

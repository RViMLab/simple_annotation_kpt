import json
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np


class MatplotlibReviewer:
    def __init__(self, scene_path: str):
        self.scene_path = scene_path
        self.img_dir = os.path.join(scene_path, "images")
        self.json_path = os.path.join(scene_path, "keypoints.json")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.files = sorted(
            [f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        )
        if not self.files:
            raise RuntimeError(f"No images found in: {self.img_dir}")

        self.file_to_list_idx = {name: idx for idx, name in enumerate(self.files)}
        self.index_to_frame_index = [self._infer_frame_index_from_filename(f) for f in self.files]

        self.idx = 0
        self.cur_cat = 1
        self.cur_pt_id = 1
        self.selected_pt_idx: Optional[int] = None
        self.dragging = False
        self.auto_next_keypoint = True
        self.text_objects = []

        self.key_to_point_id = {"z": 1, "x": 2, "c": 3, "v": 4}
        self.colors = {
            1: (0, 1, 0),
            2: (1, 1, 0),
            3: (1, 0, 0),
            4: (0, 0.5, 1),
            5: (1, 0, 1),
        }

        self.frame_data = self._load_data()

        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.canvas.manager.set_window_title(f"Reviewer - {os.path.basename(scene_path)}")
        from matplotlib.widgets import CheckButtons

        self.fig.subplots_adjust(bottom=0.12)
        toggle_ax = self.fig.add_axes([0.12, 0.015, 0.30, 0.065])
        self.keypoint_toggle = CheckButtons(toggle_ax, ["Auto next keypoint"], [True])
        self.keypoint_toggle.on_clicked(self._toggle_auto_keypoint)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.im_obj = None
        self.scat_obj = None
        self.select_obj = None

        self.render()
        self.print_controls()
        plt.show()

    def _toggle_auto_keypoint(self, _label) -> None:
        self.auto_next_keypoint = bool(self.keypoint_toggle.get_status()[0])

    def print_controls(self) -> None:
        print("\n=== Controls ===")
        print(" [Q]       : Quit")
        print(" [A] / [D] : Prev / Next frame")
        print(" [S]       : Save keypoints.json")
        print(" [1-5]     : Set category (hover/select point to edit, otherwise set mode)")
        print(" [Z/X/C/V] : Set point_id to 1/2/3/4 (hover/select point to edit, otherwise set mode)")
        print(" LeftClick : Select point; drag selected point")
        print(" Ctrl+Left : Add point with current category and point_id")
        print(" RightClick: Delete nearest point")
        print("================\n")

    def _infer_frame_index_from_filename(self, fname: str) -> Optional[int]:
        base = os.path.splitext(fname)[0]
        if base.isdigit():
            return int(base)
        match = re.search(r"(\d+)(?!.*\d)", base)
        return int(match.group(1)) if match else None

    def _load_data(self) -> Dict[int, List[dict]]:
        if not os.path.exists(self.json_path):
            return {}

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return {}

        frame_map: Dict[int, List[dict]] = {}
        for track in data.get("tracks", []):
            cat = int(track.get("category_id", 1))
            default_pt_id = int(track.get("point_id", 1))

            for p in track.get("track", []):
                fid = int(p["frame_index"])
                target_fname = f"{fid:08d}.jpg"
                list_idx = self.file_to_list_idx.get(target_fname, None)
                if list_idx is None and 0 <= fid < len(self.files):
                    list_idx = fid
                if list_idx is None:
                    continue

                pt = {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "category_id": cat,
                    "point_id": int(p.get("point_id", default_pt_id)),
                    "score": float(p.get("score", 1.0)),
                    "global_frame_index": fid,
                }

                if "audit_action" in p:
                    pt["audit_action"] = p["audit_action"]
                if "audit_score" in p:
                    pt["audit_score"] = p["audit_score"]
                if "audit_error" in p:
                    pt["audit_error"] = p["audit_error"]

                frame_map.setdefault(list_idx, []).append(pt)

        return frame_map

    def save(self) -> None:
        grouped_tracks: Dict[Tuple[int, int], List[dict]] = {}
        for list_idx, points in self.frame_data.items():
            for p in points:
                cat = int(p["category_id"])
                ptid = int(p["point_id"])
                key = (cat, ptid)
                grouped_tracks.setdefault(key, [])

                frame_index = p.get("global_frame_index", None)
                if frame_index is None:
                    inferred = self.index_to_frame_index[list_idx] if 0 <= list_idx < len(self.index_to_frame_index) else None
                    frame_index = inferred if inferred is not None else list_idx

                pt_out = {
                    "frame_index": int(frame_index),
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "score": float(p.get("score", 1.0)),
                    "point_id": ptid,
                }
                if "audit_action" in p:
                    pt_out["audit_action"] = p["audit_action"]
                if "audit_score" in p:
                    pt_out["audit_score"] = p["audit_score"]
                if "audit_error" in p:
                    pt_out["audit_error"] = p["audit_error"]
                grouped_tracks[key].append(pt_out)

        final_tracks = []
        for (cat, ptid), pts in grouped_tracks.items():
            pts.sort(key=lambda x: x["frame_index"])
            final_tracks.append({"category_id": cat, "point_id": ptid, "track": pts})

        if os.path.exists(self.json_path):
            shutil.copy(self.json_path, self.json_path + ".bak")
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"tracks": final_tracks}, f, indent=2)

        print(f"[System] Saved to {self.json_path}")
        self.ax.set_title("SAVED", color="green", fontweight="bold")
        self.fig.canvas.draw_idle()

    def _get_image(self, idx: int) -> np.ndarray:
        fname = self.files[idx]
        path = os.path.join(self.img_dir, fname)
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _find_closest_point(self, x: Optional[float], y: Optional[float], threshold: float = 20.0):
        points = self.frame_data.get(self.idx, [])
        if x is None or y is None or not points:
            return None, float("inf")

        min_d = float("inf")
        target_i = None
        for i, p in enumerate(points):
            d = float(np.sqrt((p["x"] - x) ** 2 + (p["y"] - y) ** 2))
            if d < min_d:
                min_d = d
                target_i = i

        if target_i is not None and min_d < threshold:
            return target_i, min_d
        return None, min_d

    def _resolve_target_index(self, x: Optional[float], y: Optional[float]) -> Optional[int]:
        hover_idx, _ = self._find_closest_point(x, y)
        if hover_idx is not None:
            return hover_idx
        return self.selected_pt_idx

    def render(self) -> None:
        if self.idx >= len(self.files):
            self.idx = 0

        fname = self.files[self.idx]
        img = self._get_image(self.idx)
        points = self.frame_data.get(self.idx, [])

        if self.im_obj is None:
            self.im_obj = self.ax.imshow(img)
        else:
            self.im_obj.set_data(img)

        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        c_arr = [self.colors.get(int(p["category_id"]), (1, 1, 1)) for p in points]

        if self.scat_obj is None:
            self.scat_obj = self.ax.scatter(xs, ys, c=c_arr, s=100, edgecolors="black", linewidth=1.5)
        else:
            offsets = np.column_stack([xs, ys]) if xs else np.empty((0, 2))
            self.scat_obj.set_offsets(offsets)
            self.scat_obj.set_facecolors(c_arr if c_arr else [])
            self.scat_obj.set_edgecolors(["black"] * len(points) if points else [])
            self.scat_obj.set_sizes([100] * len(points) if points else [])

        if self.selected_pt_idx is not None and 0 <= self.selected_pt_idx < len(points):
            sel = points[self.selected_pt_idx]
            sel_offsets = np.array([[sel["x"], sel["y"]]])
            if self.select_obj is None:
                self.select_obj = self.ax.scatter(
                    sel_offsets[:, 0],
                    sel_offsets[:, 1],
                    s=220,
                    facecolors="none",
                    edgecolors="white",
                    linewidth=2.0,
                )
            else:
                self.select_obj.set_offsets(sel_offsets)
                self.select_obj.set_visible(True)
        elif self.select_obj is not None:
            self.select_obj.set_visible(False)

        for txt in self.text_objects:
            txt.remove()
        self.text_objects = []

        for i, p in enumerate(points):
            cat = int(p["category_id"])
            ptid = int(p.get("point_id", 1))
            label = f"C{cat}-P{ptid}"
            if i == self.selected_pt_idx:
                label += "*"

            txt = self.ax.text(
                float(p["x"]) + 10,
                float(p["y"]),
                label,
                color=self.colors.get(cat, (1, 1, 1)),
                fontsize=8,
                fontweight="bold",
                verticalalignment="bottom",
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
            self.text_objects.append(txt)

        mode = f"MODE [Cat {self.cur_cat}][Pt {self.cur_pt_id}]"
        title = f"{mode} | Frame {self.idx + 1}/{len(self.files)} | {fname}"
        self.ax.set_title(title, color="black", fontsize=12)
        self.fig.canvas.draw_idle()

    def on_key(self, event) -> None:
        if event.key == "q":
            plt.close(self.fig)
            return
        if event.key == "d":
            if self.idx < len(self.files) - 1:
                self.idx += 1
                self.selected_pt_idx = None
                self.dragging = False
                self.cur_pt_id = 1
                self.render()
        elif event.key == "a":
            if self.idx > 0:
                self.idx -= 1
                self.selected_pt_idx = None
                self.dragging = False
                self.cur_pt_id = 1
                self.render()
        elif event.key == "s":
            self.save()
        elif event.key in ["1", "2", "3", "4", "5"]:
            target_cat = int(event.key)
            self.cur_cat = target_cat
            target_idx = self._resolve_target_index(event.xdata, event.ydata)
            if target_idx is not None:
                self.frame_data[self.idx][target_idx]["category_id"] = target_cat
                print(f"[Edit] Point category -> {target_cat}")
            else:
                self.cur_cat = target_cat
                print(f"[Mode] Category set to {target_cat}")
            self.render()
        elif event.key in self.key_to_point_id:
            target_pt = self.key_to_point_id[event.key]
            self.cur_pt_id = target_pt
            target_idx = self._resolve_target_index(event.xdata, event.ydata)
            if target_idx is not None:
                self.frame_data[self.idx][target_idx]["point_id"] = target_pt
                print(f"[Edit] Point ID -> {target_pt}")
            else:
                self.cur_pt_id = target_pt
                print(f"[Mode] Point ID set to {target_pt}")
            self.render()

    def on_click(self, event) -> None:
        if event.inaxes != self.ax:
            return

        closest_idx, _ = self._find_closest_point(event.xdata, event.ydata)
        points = self.frame_data.setdefault(self.idx, [])

        if event.button == 3:
            if closest_idx is not None:
                points.pop(closest_idx)
                if self.selected_pt_idx == closest_idx:
                    self.selected_pt_idx = None
                self.render()
            return

        if event.button != 1:
            return

        if event.key in ("control", "ctrl", "ctrl+shift", "control+shift"):
            if event.xdata is None or event.ydata is None:
                return
            points.append(
                {
                    "x": float(event.xdata),
                    "y": float(event.ydata),
                    "category_id": int(self.cur_cat),
                    "point_id": int(self.cur_pt_id),
                    "score": 1.0,
                }
            )
            self.selected_pt_idx = None
            self.dragging = False
            print(f"[Add] Cat={self.cur_cat}, Pt={self.cur_pt_id}")
            if self.auto_next_keypoint:
                self.cur_pt_id = self.cur_pt_id % 4 + 1
            self.render()
            return

        if closest_idx is not None:
            self.selected_pt_idx = closest_idx
            self.dragging = True
            self.render()
        else:
            self.selected_pt_idx = None
            self.render()

    def on_release(self, _event) -> None:
        self.dragging = False

    def on_motion(self, event) -> None:
        if not self.dragging:
            return
        if self.selected_pt_idx is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        points = self.frame_data.get(self.idx, [])
        if not (0 <= self.selected_pt_idx < len(points)):
            return

        points[self.selected_pt_idx]["x"] = float(event.xdata)
        points[self.selected_pt_idx]["y"] = float(event.ydata)
        self.render()

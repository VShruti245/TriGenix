"""
multi_angle_processor.py — Multi-Angle Capture Mode for TriGenix.

Accepts 4 real photographs (front/back/left/right) and injects them as
the view set, bypassing Zero123++ synthesis entirely.

Azimuth mapping (fixed):
    front  → az=0,   el=0
    back   → az=180, el=0
    left   → az=90,  el=0
    right  → az=270, el=0
"""

import os, shutil, json, time
from PIL import Image


AZIMUTH_MAP = {
    "front": 0,
    "back":  180,
    "left":  90,
    "right": 270,
}


def process_multi_angle(
    front_path: str,
    back_path: str,
    left_path: str,
    right_path: str,
    out_dir: str,
    target_dimensions_mm: dict | None = None,
) -> dict:
    """
    Copy the four images into out_dir as view_0..view_3 with fixed azimuth
    assignments [0, 180, 90, 270] and elevation 0.
    Write pose.json with the azimuth/elevation metadata.
    Return a proc_result dict compatible with save_history() contract.

    Does NOT call processor.py or reconstructor.py.
    Does NOT invoke Zero123++.
    """
    os.makedirs(out_dir, exist_ok=True)

    sources = [front_path, back_path, left_path, right_path]
    slot_names = ["front", "back", "left", "right"]
    azimuths = [AZIMUTH_MAP[s] for s in slot_names]
    elevations = [0, 0, 0, 0]

    view_paths = []
    for i, src in enumerate(sources):
        dst = os.path.join(out_dir, f"view_{i}.png")
        # Convert to PNG if not already
        try:
            img = Image.open(src).convert("RGBA")
            img.save(dst, "PNG")
        except Exception:
            shutil.copy2(src, dst)
        view_paths.append(dst)

    # Write pose.json — compatible with load_from_dir() in reconstructor.py
    poses = [
        {"az": azimuths[i], "el": elevations[i]}
        for i in range(4)
    ]
    pose_path = os.path.join(out_dir, "view_poses.json")
    with open(pose_path, "w", encoding="utf-8") as f:
        json.dump(poses, f)

    # Also copy front image as masked.png (used as the primary input by TripoSR)
    masked_dst = os.path.join(out_dir, "masked.png")
    try:
        img = Image.open(front_path).convert("RGBA")
        img.save(masked_dst, "PNG")
    except Exception:
        shutil.copy2(front_path, masked_dst)

    # Build proc_result compatible with save_history() contract
    # (same shape as processor.py process_image() result)
    result = {
        "status": "success",
        "out_dir": out_dir,
        "masked": f"/{out_dir.replace(os.sep, '/')}/masked.png",
        "views": [
            f"/{vp.replace(os.sep, '/')}" for vp in view_paths
        ],
        "depths_grey":  [],   # no depth maps in multi-angle mode
        "depths_color": [],
        "time": 0.0,          # preprocessing time; reconstruction is async
        "mode": "multi_angle",
        "stage_times": {},
    }
    if target_dimensions_mm:
        result["target_dimensions_mm"] = target_dimensions_mm

    return result

"""
pose_estimator.py  —  TriGenix input-image pose analyser
=========================================================
Pure OpenCV/NumPy. No extra rembg calls — reuses masked.png.
Fixed from v1: roll suppressed when object is symmetric (lamp = 0 roll).
world_R = identity when pose is near-frontal so hull is NOT misrotated.
"""

import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from scipy.ndimage import label as ndlabel
import logging, os, json

logger = logging.getLogger(__name__)


@dataclass
class PoseResult:
    azimuth:        float = 0.0
    elevation:      float = 0.0
    roll:           float = 0.0
    distance:       float = 1.0
    upright:        bool  = True
    world_R:        np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    cond_elevation: float = 0.0
    debug:          dict  = field(default_factory=dict)


def _load_mask(image_path: str, out_dir: str) -> np.ndarray:
    """Reuse masked.png alpha — never calls rembg again."""
    if out_dir:
        p = os.path.join(out_dir, "masked.png")
        if os.path.exists(p):
            rgba = np.array(Image.open(p).convert("RGBA"))
            m = rgba[..., 3] > 32
            if m.mean() > 0.01:
                return m
    img = Image.open(image_path)
    if img.mode == "RGBA":
        m = np.array(img.split()[3]) > 32
        if m.mean() > 0.01:
            return m
    # Otsu fallback
    rgb  = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = thresh.shape
    for pt in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)]:
        cv2.floodFill(thresh, np.zeros((h+2,w+2),np.uint8), pt, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = thresh > 0
    labeled, n = ndlabel(fg)
    if n > 0:
        sizes = [(labeled==i).sum() for i in range(1,n+1)]
        fg    = labeled == (sizes.index(max(sizes))+1)
    return fg.astype(bool)


def _pca_ellipse(mask):
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None
    pts   = np.stack([xs, ys], axis=1).astype(np.float32)
    pts_c = pts - pts.mean(axis=0)
    cov   = (pts_c.T @ pts_c) / len(pts_c)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major  = eigvecs[:, 1]
    angle  = float(np.degrees(np.arctan2(major[1], major[0])))
    aspect = float(np.sqrt(eigvals[0]) / (np.sqrt(eigvals[1]) + 1e-8))
    return angle, aspect


def _sym_lr(mask):
    m = mask.astype(np.float32)
    u = np.maximum(m, np.fliplr(m)).sum()
    return float(np.minimum(m, np.fliplr(m)).sum() / u) if u > 0 else 0.0


def _tight_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h,w = mask.shape; return 0,0,w,h,w,h
    return int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()),int(xs.max()-xs.min()),int(ys.max()-ys.min())


def estimate_pose(image_path: str, out_dir: str = None) -> PoseResult:
    result = PoseResult()
    try:
        mask = _load_mask(image_path, out_dir)
    except Exception as e:
        logger.warning(f"  PoseEst: mask failed ({e}), default pose")
        return result

    if mask.mean() < 0.005:
        logger.warning("  PoseEst: empty mask, default pose")
        return result

    H, W = mask.shape
    logger.info(f"  PoseEst: mask {W}x{H} fg={mask.mean()*100:.1f}%")

    pca = _pca_ellipse(mask)
    angle_deg = pca[0] if pca else 90.0
    aspect    = pca[1] if pca else 0.5
    result.upright = aspect < 0.75

    sym = _sym_lr(mask)
    logger.info(f"  PoseEst: pca_angle={angle_deg:.1f} aspect={aspect:.2f} sym_lr={sym:.3f}")

    # ── Roll ──────────────────────────────────────────────────────────────────
    # Only apply roll when object is BOTH elongated AND asymmetric.
    # A symmetric lamp (sym=0.96) gets roll=0 even if PCA angle=90.
    a = angle_deg % 180
    if a > 90: a -= 180          # deviation from vertical
    if aspect < 0.65 and sym < 0.80:
        conf         = (0.80 - sym) / 0.80
        result.roll  = float(np.clip(a * conf, -40, 40))
    else:
        result.roll  = 0.0

    # ── Elevation ─────────────────────────────────────────────────────────────
    x0,y0,x1,y1,bw,bh = _tight_bbox(mask)
    ar  = bh / (bw + 1e-8)
    ys, _ = np.where(mask)
    cy  = float(ys.mean()) / H - 0.5
    if   ar < 0.5:  result.elevation = float(np.clip(-25*(1-ar/0.5), -40, 0))
    elif ar > 2.5:  result.elevation = 5.0
    else:           result.elevation = float(np.clip(cy * -20, -40, 40))

    # ── Azimuth ───────────────────────────────────────────────────────────────
    # Only when clearly asymmetric (sym < 0.85)
    if sym < 0.85:
        lm = float(mask[:, :W//2].sum()); rm = float(mask[:, W//2:].sum())
        tot = lm + rm
        if tot > 0:
            asym  = abs(lm - rm) / tot
            sign  = 1.0 if rm > lm else -1.0
            result.azimuth = float(np.clip(sign * asym * 35 * (0.85-sym)/0.85, -40, 40))

    result.distance       = float(np.clip(np.sqrt(0.25/(mask.mean()+1e-6)), 0.6, 2.5))
    result.cond_elevation = float(np.clip(-result.elevation, -30, 30))

    # ── World rotation — IDENTITY when near-frontal ───────────────────────────
    # Compose Rx (elevation) @ Ry (azimuth) @ Rz (roll).
    # Elevation tilts the mesh up/down so it faces the camera correctly.
    # Small angles (< 5°) are ignored per-axis to avoid jitter on symmetric
    # or nearly-frontal objects.
    _az_big  = abs(result.azimuth)   > 5.0
    _el_big  = abs(result.elevation) > 5.0
    _roll_big= abs(result.roll)      > 5.0

    if not _az_big and not _el_big and not _roll_big:
        result.world_R = np.eye(3, dtype=np.float32)
    else:
        rz = np.radians(-result.roll)      if _roll_big else 0.0
        ry = np.radians(-result.azimuth)   if _az_big   else 0.0
        rx = np.radians(-result.elevation) if _el_big   else 0.0

        Rz = np.array([[np.cos(rz),-np.sin(rz),0],
                        [np.sin(rz), np.cos(rz),0],
                        [0,          0,          1]], np.float32)
        Ry = np.array([[ np.cos(ry),0,np.sin(ry)],
                        [ 0,         1,0          ],
                        [-np.sin(ry),0,np.cos(ry)]], np.float32)
        # Rx rotates around X: positive elevation means camera looks DOWN at
        # the object, so the mesh should tilt TOWARD the camera (rotate -el
        # around X so the top of the object faces the camera).
        Rx = np.array([[1, 0,          0         ],
                        [0, np.cos(rx),-np.sin(rx)],
                        [0, np.sin(rx), np.cos(rx)]], np.float32)
        # Order: roll first (in-plane), then azimuth, then elevation
        result.world_R = Rx @ Ry @ Rz

    result.debug = {
        "pca_angle": round(angle_deg,2), "aspect": round(aspect,3),
        "sym_lr": round(sym,3), "fg_frac": round(float(mask.mean()),4),
        "bbox_wh": [bw,bh],
    }
    logger.info(
        f"  PoseEst → az={result.azimuth:.1f} el={result.elevation:.1f} "
        f"roll={result.roll:.1f} R_identity={np.allclose(result.world_R,np.eye(3))}"
    )
    return result


def save_pose(pose: PoseResult, out_dir: str):
    path = os.path.join(out_dir, "pose.json")
    with open(path,"w") as f:
        json.dump({
            "azimuth": pose.azimuth, "elevation": pose.elevation,
            "roll": pose.roll, "distance": pose.distance,
            "upright": pose.upright, "cond_elevation": pose.cond_elevation,
            "world_R": pose.world_R.tolist(), "debug": pose.debug,
        }, f, indent=2)
    logger.info(f"  PoseEst: saved → {path}")


def load_pose(out_dir: str) -> PoseResult:
    path = os.path.join(out_dir, "pose.json")
    if not os.path.exists(path):
        return PoseResult()
    with open(path) as f:
        d = json.load(f)
    return PoseResult(
        azimuth=d.get("azimuth",0.0), elevation=d.get("elevation",0.0),
        roll=d.get("roll",0.0), distance=d.get("distance",1.0),
        upright=d.get("upright",True), cond_elevation=d.get("cond_elevation",0.0),
        world_R=np.array(d.get("world_R",np.eye(3).tolist()),dtype=np.float32),
        debug=d.get("debug",{}),
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pose_estimator.py <image> [out_dir]"); sys.exit(1)
    logging.basicConfig(level=logging.INFO)
    pose = estimate_pose(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else None)
    print(json.dumps({"az":pose.azimuth,"el":pose.elevation,"roll":pose.roll,
                      "R_identity":bool(np.allclose(pose.world_R,np.eye(3))),
                      "debug":pose.debug},indent=2))
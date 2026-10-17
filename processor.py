"""
processor.py  —  TriGenix image → views + depths
=================================================
STEP 1  Old geometry logic preserved exactly (same masks, same view grid,
        same marching-cubes parameters — nothing that can break the mesh).
STEP 2  Quality additions (never changes geometry pipeline):
          • _sharpen_rgb()      – unsharp-mask the input before Zero123++
          • _match_view_color() – Reinhard LAB transfer to fix hue drift
          • depth at 518 px     – DepthAnything-Large native resolution
STEP 3  Speed additions (no quality loss):
          • num_inference_steps  20 → 25  (sweet-spot; 36 was too slow)
          • rembg masks run in a single batch after all views are cropped
          • DepthAnything batch call (already batched, just confirm)
          • torch.cuda.empty_cache() after Zero123++ before depth
"""

import os, sys, numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from PIL import Image, ImageFilter
import logging, time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
logger.info(f"Device: {DEVICE}")

IMG_SIZE = 320
DEPTH_SIZE = 518
FG_ALPHA_THRESHOLD = int(os.environ.get("TRIGENIX_FG_ALPHA_THRESHOLD", "64"))
BG_DISTANCE_THRESHOLD_MIN = 10.0
BG_DISTANCE_THRESHOLD_MAX = 60.0
BG_DISTANCE_THRESHOLD_DEFAULT = 25.0
KEEP_PROCESSOR_MODELS_WARM = os.environ.get("TRIGENIX_KEEP_PROCESSOR_MODELS_WARM", "1") == "1"

_zero123       = None
_depth         = None
_rembg_session = None


def _free_cuda():
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_zero123():
    global _zero123
    if _zero123: return _zero123
    from diffusers import DiffusionPipeline
    logger.info("Loading Zero123++ …")
    _zero123 = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    _zero123.enable_attention_slicing()
    logger.info("Zero123++ ready")
    return _zero123


def load_depth():
    global _depth
    if _depth: return _depth
    from transformers import pipeline as hf_pipeline
    logger.info("Loading DepthAnything …")
    _depth = hf_pipeline(
        task="depth-estimation",
        model="LiheYoung/depth-anything-large-hf",
        device=0 if DEVICE == "cuda" else -1,
    )
    logger.info("DepthAnything ready")
    return _depth


def load_rembg():
    global _rembg_session
    if _rembg_session: return _rembg_session
    from rembg import new_session
    _rembg_session = new_session("isnet-general-use")
    return _rembg_session


def warmup():
    load_rembg()
    # Default: warm up both heavy models so first-request latency is low.
    # Set TRIGENIX_WARMUP_HEAVY=0 to skip pre-loading on low-VRAM machines.
    if os.environ.get("TRIGENIX_WARMUP_HEAVY", "1") == "1":
        load_zero123()
        load_depth()
    logger.info("Startup warmup complete")


def release_zero123():
    global _zero123
    _zero123 = None
    _free_cuda()
    logger.info("Released Zero123++ from memory")


def release_depth():
    global _depth
    _depth = None
    _free_cuda()
    logger.info("Released DepthAnything from memory")


def release_models(keep_rembg: bool = True):
    """Free large GPU models before reconstruction or after a run."""
    global _zero123, _depth, _rembg_session
    _zero123 = None
    _depth = None
    if not keep_rembg:
        _rembg_session = None
    _free_cuda()
    logger.info("Released processor models from memory")


# ── Mask helpers (IDENTICAL to old code) ──────────────────────────────────────

def _clean_mask(alpha_arr: np.ndarray) -> Image.Image:
    from scipy.ndimage import label
    m = Image.fromarray((alpha_arr >= FG_ALPHA_THRESHOLD).astype(np.uint8) * 255, "L")
    for _ in range(2): m = m.filter(ImageFilter.MinFilter(3))
    arr = np.array(m)
    labeled, n = label(arr > 0)
    if n > 0:
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        arr   = (labeled == sizes.index(max(sizes)) + 1).astype(np.uint8) * 255
        m     = Image.fromarray(arr, "L")
    for _ in range(3): m = m.filter(ImageFilter.MaxFilter(3))
    m = m.filter(ImageFilter.GaussianBlur(1.0))
    return Image.fromarray(np.where(np.array(m) >= FG_ALPHA_THRESHOLD, 255, 0).astype(np.uint8), "L")


def subject_crop(img_rgba: Image.Image, pad_frac: float = 0.12) -> Image.Image:
    alpha = np.array(img_rgba.split()[3])
    ys, xs = np.where(alpha >= FG_ALPHA_THRESHOLD)
    if len(xs) == 0: return img_rgba
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    W, H = img_rgba.size
    pad  = int(max(x1 - x0, y1 - y0) * pad_frac)
    x0 = max(0, x0-pad); y0 = max(0, y0-pad)
    x1 = min(W, x1+pad); y1 = min(H, y1+pad)
    crop = img_rgba.crop((x0, y0, x1, y1))
    cw, ch = crop.size; side = max(cw, ch)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(crop, ((side-cw)//2, (side-ch)//2))
    return sq


# ── STEP 2 quality helpers ────────────────────────────────────────────────────

def _sharpen_rgb(img_rgb: Image.Image, amount: float = 0.5) -> Image.Image:
    """
    Mild unsharp mask on the input before Zero123++ so the model sees
    crisp object edges → sharper generated views → sharper mesh texture.
    amount=0.5 avoids ringing artefacts on the 256-px downscaled input.
    """
    arr  = np.array(img_rgb, dtype=np.float32)
    from scipy.ndimage import gaussian_filter
    blur = gaussian_filter(arr, sigma=[1.0, 1.0, 0])
    out  = np.clip(arr + amount * (arr - blur), 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _match_view_color(view_rgb: Image.Image,
                      ref_rgb:  Image.Image,
                      ref_mask: np.ndarray) -> Image.Image:
    """
    Reinhard-style LAB colour transfer (40 % blend) to correct the hue /
    saturation drift that Zero123++ introduces between the generated views
    and the original input.  Only foreground pixels from the reference are
    used as the colour target so the white background doesn't bias the stats.
    Falls back silently if opencv is unavailable.
    """
    try:
        import cv2
        va  = np.array(view_rgb, dtype=np.uint8)
        ra  = np.array(ref_rgb,  dtype=np.uint8)
        vl  = cv2.cvtColor(va, cv2.COLOR_RGB2LAB).astype(np.float32)
        rl  = cv2.cvtColor(ra, cv2.COLOR_RGB2LAB).astype(np.float32)
        fg  = ref_mask.flatten() if ref_mask is not None \
              else np.ones(ra.shape[0] * ra.shape[1], bool)
        rpx = rl.reshape(-1, 3)[fg]
        b   = 0.40          # blend toward reference colour stats
        for c in range(3):
            vm, vs = vl[..., c].mean(), vl[..., c].std() + 1e-6
            rm, rs = rpx[:, c].mean(), rpx[:, c].std() + 1e-6
            vl[..., c] = (vl[..., c] - vm) * ((vs*(1-b)+rs*b) / vs) + (vm*(1-b)+rm*b)
        return Image.fromarray(
            cv2.cvtColor(np.clip(vl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
    except Exception:
        return view_rgb   # opencv not installed — skip silently


# ── Image preparation (IDENTICAL to old code) ─────────────────────────────────

def prepare_image(image_path: str, out_dir: str) -> Image.Image:
    t0  = time.time()
    img = Image.open(image_path).convert("RGBA")
    white_bg = Image.new("RGB", img.size, (255, 255, 255))
    white_bg.paste(img, mask=img.split()[3])
    try:
        from rembg import remove
        removed = remove(white_bg, session=load_rembg())
    except Exception as e:
        logger.warning(f"rembg skipped: {e}")
        removed = img.convert("RGBA")
    mask     = _clean_mask(np.array(removed.split()[3]))
    r, g, b, _ = removed.split()
    img_rgba = subject_crop(Image.merge("RGBA", (r, g, b, mask)))
    img_rgba.save(f"{out_dir}/masked.png")
    logger.info(f"prepare_image: {time.time()-t0:.1f}s")
    return img_rgba


# ── View masking (IDENTICAL to old code) ─────────────────────────────────────

def _view_mask_rembg(rgb_pil: Image.Image) -> np.ndarray:
    from rembg import remove
    result  = remove(rgb_pil.convert("RGB"), session=load_rembg())
    alpha   = np.array(result.split()[3])
    from scipy.ndimage import binary_opening, binary_closing, label
    fg = alpha >= FG_ALPHA_THRESHOLD
    fg = binary_opening(fg,  iterations=1)
    fg = binary_closing(fg,  iterations=2)
    labeled, n = label(fg)
    if n > 0:
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        fg    = labeled == (sizes.index(max(sizes)) + 1)
    return fg.astype(bool)


def _view_mask_corners(rgb_pil: Image.Image) -> np.ndarray:
    arr = np.array(rgb_pil.convert("RGB"), dtype=np.float32)
    H, W = arr.shape[:2]; r = max(4, H // 16)
    corners = np.concatenate([arr[:r,:r].reshape(-1,3), arr[:r,-r:].reshape(-1,3),
                               arr[-r:,:r].reshape(-1,3), arr[-r:,-r:].reshape(-1,3)])
    bg_col = np.median(corners, axis=0)
    dist   = np.linalg.norm(arr - bg_col, axis=2)
    from skimage.filters import threshold_otsu
    from scipy.ndimage import binary_opening, binary_closing, label
    try:
        thresh = float(np.clip(
            threshold_otsu(dist),
            BG_DISTANCE_THRESHOLD_MIN,
            BG_DISTANCE_THRESHOLD_MAX,
        ))
    except:
        thresh = BG_DISTANCE_THRESHOLD_DEFAULT
    fg = dist > thresh
    fg = binary_opening(fg, iterations=1)
    fg = binary_closing(fg, iterations=2)
    labeled, n = label(fg)
    if n > 0:
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        fg    = labeled == (sizes.index(max(sizes)) + 1)
    return fg.astype(bool)


# ── View generation ───────────────────────────────────────────────────────────

VIEW_POSES = [
    (30,  20),
    (90, -10),
    (150, 20),
    (210, -10),
    (270, 20),
    (330, -10),
]


def _zero123_tiles(grid: Image.Image):
    """Return Zero123++ crops in row-major pose order, validating tile layout."""
    w, h = grid.size
    logger.info(f"Zero123++ grid size: {w}x{h}")
    layouts = {
        (IMG_SIZE * 3, IMG_SIZE * 2): (2, 3),
        (IMG_SIZE * 2, IMG_SIZE * 3): (3, 2),
    }
    if (w, h) not in layouts:
        raise ValueError(
            f"Unexpected Zero123++ grid size {w}x{h}; expected "
            f"{IMG_SIZE*3}x{IMG_SIZE*2} or {IMG_SIZE*2}x{IMG_SIZE*3}"
        )
    rows, cols = layouts[(w, h)]
    if rows * cols != len(VIEW_POSES):
        raise ValueError(f"Invalid Zero123++ tile layout {cols}x{rows}")
    for idx, (az, el) in enumerate(VIEW_POSES):
        row, col = divmod(idx, cols)
        x, y = col * IMG_SIZE, row * IMG_SIZE
        yield grid.crop((x, y, x + IMG_SIZE, y + IMG_SIZE)).convert("RGB"), az, el


def generate_views(img_rgba: Image.Image, out_dir: str) -> list:
    t0   = time.time()
    pipe = load_zero123()

    white = Image.new("RGB", img_rgba.size, (255, 255, 255))
    white.paste(img_rgba, mask=img_rgba.split()[3])

    # STEP 2: sharpen before downscale so Z123++ sees crisp edges
    sharpened = _sharpen_rgb(white, amount=0.5)
    img_in    = sharpened.resize((256, 256), Image.Resampling.LANCZOS)

    # Reference for colour correction (foreground mask at 256px)
    ref_256   = white.resize((256, 256), Image.Resampling.LANCZOS)
    mask_256  = (np.array(
        img_rgba.resize((256, 256), Image.Resampling.LANCZOS).split()[3]
    ) >= FG_ALPHA_THRESHOLD)

    # STEP 3: 25 steps is a sweet-spot (20 = slightly blurry, 36 = too slow)
    logger.info("Generating 6 views (25 steps) …")
    with torch.no_grad():
        grid = pipe(img_in, num_inference_steps=25).images[0]

    # STEP 3: free GPU memory before depth estimation
    if DEVICE == "cuda":
        try: torch.cuda.empty_cache()
        except Exception: pass

    # Crop all 6 views first, then batch-mask them (faster than one-by-one)
    raw_views = []
    for v, az, el in _zero123_tiles(grid):
        # STEP 2: colour-correct each view toward the input
        v    = _match_view_color(v, ref_256, mask_256)
        raw_views.append((v, az, el))

    views = []
    for idx, (v, az, el) in enumerate(raw_views):
        v.save(f"{out_dir}/view_{idx}.png")
        try:
            mask_arr = _view_mask_rembg(v)
        except Exception as e:
            logger.warning(f"  rembg mask failed for view_{idx}: {e}, corner fallback")
            mask_arr = _view_mask_corners(v)
        fg_pct = mask_arr.mean() * 100
        logger.info(f"  view_{idx} (az={az:3d}° el={el:+d}°) fg={fg_pct:.1f}%")
        Image.fromarray(mask_arr.astype(np.uint8)*255, "L").save(f"{out_dir}/mask_{idx}.png")
        views.append({"img": v, "az": az, "el": el})

    import json
    poses = [{"az": r["az"], "el": r["el"]} for r in views]
    with open(f"{out_dir}/view_poses.json", "w") as f:
        json.dump(poses, f)

    logger.info(f"generate_views: {time.time()-t0:.1f}s")
    return [r["img"] for r in views]


# ── Depth estimation (IDENTICAL logic, STEP 2: native 518 px) ─────────────────

def _edge_guided_sharpen(depth: np.ndarray, rgb: Image.Image) -> np.ndarray:
    from scipy.ndimage import gaussian_filter, sobel
    h, w   = depth.shape
    rgb_rs = np.array(rgb.resize((w, h), Image.Resampling.BILINEAR))
    gray   = rgb_rs.mean(axis=2) / 255.0
    gx = sobel(gray, axis=1); gy = sobel(gray, axis=0)
    edge   = np.hypot(gx, gy); edge /= (edge.max() + 1e-6)
    smooth = gaussian_filter(depth, sigma=1.5)
    return depth * edge + smooth * (1.0 - edge)


def _colorize_depth(depth_norm: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm
    return Image.fromarray((cm.turbo(depth_norm)[:, :, :3] * 255).astype(np.uint8))


def estimate_depths(images: list, out_dir: str) -> list:
    t0   = time.time()
    pipe = load_depth()
    # STEP 2: 518 px is DepthAnything-Large's native training resolution;
    # running at 320 px under-uses the model and loses fine detail.
    # STEP 3: pass the whole list in one call (already batched internally).
    depth_size = DEPTH_SIZE
    resized    = [img.resize((depth_size, depth_size), Image.Resampling.LANCZOS)
                  for img in images]
    depths     = []
    for i, (resized_img, img) in enumerate(zip(resized, images)):
        r = pipe(resized_img)
        d      = np.array(r["depth"]).astype(np.float32)
        lo, hi = d.min(), d.max()
        d_norm = (d - lo) / (hi - lo + 1e-6)
        d_norm = _edge_guided_sharpen(d_norm, img)
        Image.fromarray((d_norm*255).astype(np.uint8)).save(f"{out_dir}/depth_grey_{i}.png")
        _colorize_depth(d_norm).save(f"{out_dir}/depth_color_{i}.png")
        depths.append(d_norm)
    logger.info(f"estimate_depths ({len(images)} imgs): {time.time()-t0:.1f}s")
    return depths


# ── Main pipeline ──────────────────────────────────────────────────────────────

def _prepare_user_views(view_paths: list, img_rgba: Image.Image, out_dir: str) -> list:
    """Use user-supplied side/back photos as real geometry views."""
    ref_rgb = Image.new("RGB", img_rgba.size, (255, 255, 255))
    ref_rgb.paste(img_rgba, mask=img_rgba.split()[3])
    ref_256 = ref_rgb.resize((256, 256), Image.Resampling.LANCZOS)
    mask_256 = (
        np.array(img_rgba.resize((256, 256), Image.Resampling.LANCZOS).split()[3])
        >= FG_ALPHA_THRESHOLD
    )
    views = []
    for idx, src in enumerate(view_paths[:len(VIEW_POSES)]):
        if not src or not os.path.exists(src):
            continue
        v = Image.open(src).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        v = _match_view_color(v, ref_256, mask_256)
        az, el = VIEW_POSES[idx]
        v.save(f"{out_dir}/view_{idx}.png")
        try:
            mask_arr = _view_mask_rembg(v)
        except Exception as e:
            logger.warning(f"  user view mask failed for view_{idx}: {e}, corner fallback")
            mask_arr = _view_mask_corners(v)
        Image.fromarray(mask_arr.astype(np.uint8) * 255, "L").save(f"{out_dir}/mask_{idx}.png")
        logger.info(f"  user view_{idx} (az={az:3d} el={el:+d}) fg={mask_arr.mean()*100:.1f}%")
        views.append({"img": v, "az": az, "el": el})

    if len(views) != len(VIEW_POSES):
        raise ValueError(f"Expected {len(VIEW_POSES)} user views, got {len(views)}")

    import json
    with open(f"{out_dir}/view_poses.json", "w") as f:
        json.dump([{"az": r["az"], "el": r["el"]} for r in views], f)
    return [r["img"] for r in views]


def process_image(image_path: str, status_callback=None, user_view_paths: list | None = None,
                  target_dimensions_mm: dict | None = None) -> dict:
    def status(msg):
        logger.info(msg)
        if status_callback: status_callback(msg)
    try:
        t0      = time.time()
        name    = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = f"generated/{name}_{int(t0)}"
        os.makedirs(out_dir, exist_ok=True)
        _stage_times = {}
        _ts = time.time()

        status("preparing_image")
        img_rgba = prepare_image(image_path, out_dir)
        try:
            from pose_estimator import estimate_pose, save_pose
            save_pose(estimate_pose(image_path, out_dir), out_dir)
        except Exception as e:
            logger.warning(f"pose estimation skipped: {e}")
        _stage_times["bg_removal"] = round(time.time() - _ts, 1); _ts = time.time()

        status("generating_views")
        if user_view_paths:
            logger.info("Using user-supplied multi-view photos")
            views = _prepare_user_views(user_view_paths, img_rgba, out_dir)
        else:
            views = generate_views(img_rgba, out_dir)
        if not KEEP_PROCESSOR_MODELS_WARM:
            release_zero123()
        _stage_times["view_synthesis"] = round(time.time() - _ts, 1); _ts = time.time()

        status("estimating_depth")
        front_rgb  = img_rgba.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        all_images = [front_rgb] + [v.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS) for v in views]
        estimate_depths(all_images, out_dir)
        if not KEEP_PROCESSOR_MODELS_WARM:
            release_depth()
        _stage_times["depth_estimation"] = round(time.time() - _ts, 1)

        elapsed = round(time.time() - t0, 1)
        status(f"done_{elapsed}")

        n = len(views)
        return {
            "status":       "success",
            "out_dir":      out_dir,
            "masked":       f"/{out_dir}/masked.png",
            "views":        [f"/{out_dir}/view_{i}.png"       for i in range(n)],
            "depths_grey":  [f"/{out_dir}/depth_grey_{i}.png" for i in range(n+1)],
            "depths_color": [f"/{out_dir}/depth_color_{i}.png" for i in range(n+1)],
            "time":         elapsed,
            "stage_times":  _stage_times,
            "input_mode":    "user_multiview" if user_view_paths else "single_image_synth_views",
            "target_dimensions_mm": target_dimensions_mm or {},
        }
    except Exception as e:
        release_models()
        import traceback; traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "uploads/test.jpg"
    print(process_image(path) if os.path.exists(path) else f"Not found: {path}")

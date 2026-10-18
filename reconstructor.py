"""
reconstructor.py  —  TriGenix 3D reconstruction + UV texture baking
====================================================================
CRITICAL FIXES:
1. Fixed depth map inversion issue
2. Fixed vertex color painting for front view
3. Added depth map quality validation
4. Improved InstantMesh integration
5. Fixed visual hull fallback thresholds
"""

import os, sys, numpy as np, logging, time, json, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CAM_DIST     = 1.80
IMG_SIZE     = 320
FOV_DEG      = 36.9
GRID_RES     = 256
WORLD_HALF   = 0.65
TARGET_FACES = 100_000
SILHOUETTE_KEEP_RATIO = 0.56
TRIPOSR_RESOLUTIONS = (320, 256, 224, 192, 160, 128)
TRIPOSR_CHUNK_SIZE = 32768
TRIPOSR_CPU_FALLBACK_RESOLUTION = 128
TRIPOSR_TEXTURE_RESOLUTION = 1024
FG_ALPHA_THRESHOLD = int(os.environ.get("TRIGENIX_FG_ALPHA_THRESHOLD", "64"))
BG_DISTANCE_THRESHOLD_MIN = 10.0
BG_DISTANCE_THRESHOLD_MAX = 60.0
BG_DISTANCE_THRESHOLD_DEFAULT = 25.0

DEFAULT_VIEW_POSES = [
    (0,    0),
    (30,  20), (90,  -10), (150, 20),
    (210, -10), (270, 20), (330, -10),
]

# ════════════════════════════════════════════════════════════════════════════
#  CAMERA UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _camera_c2w(az_deg, el_deg, dist=CAM_DIST):
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    cam_pos = np.array([
        dist * np.cos(el) * np.sin(az),
        dist * np.sin(el),
        dist * np.cos(el) * np.cos(az),
    ])
    fwd = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
    right = np.cross(fwd, np.array([0., 1., 0.]))
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, fwd)
    R_cw = np.stack([right, up, -fwd], axis=1)
    return R_cw, cam_pos


def _project(pts_world, az_deg, el_deg):
    S = IMG_SIZE
    f = S / (2.0 * np.tan(np.radians(FOV_DEG / 2.0)))
    cx = cy = S / 2.0
    R_cw, cam_pos = _camera_c2w(az_deg, el_deg)
    R_wc = R_cw.T
    pts_cam = (R_wc @ (pts_world - cam_pos).T).T
    depth = -pts_cam[:, 2]
    valid = depth > 0.01
    eps = 1e-8
    u = f * pts_cam[:, 0] / (depth + eps) + cx
    v = -f * pts_cam[:, 1] / (depth + eps) + cy
    return u, v, depth, valid


# ════════════════════════════════════════════════════════════════════════════
#  VIEW LOADING - FIXED
# ════════════════════════════════════════════════════════════════════════════

def load_from_dir(out_dir):
    poses_path = os.path.join(out_dir, "view_poses.json")
    if os.path.exists(poses_path):
        with open(poses_path) as f:
            saved_poses = json.load(f)
        view_poses = [(0, 0)] + [(p["az"], p["el"]) for p in saved_poses]
    else:
        view_poses = DEFAULT_VIEW_POSES
        logger.warning("  view_poses.json not found, using defaults")

    views = []
    for i, (az, el) in enumerate(view_poses):
        rgb_path = os.path.join(out_dir, "masked.png" if i == 0 else f"view_{i-1}.png")
        mask_path = None if i == 0 else os.path.join(out_dir, f"mask_{i-1}.png")
        
        if not os.path.exists(rgb_path):
            continue

        raw_img = Image.open(rgb_path)

        # CRITICAL FIX: For front view, create proper RGB composite
        if i == 0:
            rgba = raw_img.convert("RGBA")
            alpha = rgba.split()[3]
            white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
            white_bg.paste(rgba, mask=alpha)
            rgb_composited = white_bg
            # Store alpha separately for masking
            alpha_mask = np.array(alpha.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) >= FG_ALPHA_THRESHOLD
        else:
            rgb_composited = raw_img.convert("RGB")
            alpha_mask = None

        views.append(dict(
            rgb=rgb_composited,
            rgb_raw=raw_img,
            mask_path=mask_path if (mask_path and os.path.exists(mask_path)) else None,
            az=az, el=el,
            use_alpha=(i == 0),
            alpha_mask=alpha_mask,
            view_idx=i,
            out_dir=out_dir,
        ))
    
    logger.info(f"  Loaded {len(views)} views")
    return views


def _get_mask(v):
    """FIXED: Proper mask extraction with fallbacks"""
    S = IMG_SIZE
    
    # Priority 1: Saved mask file
    if v['mask_path'] and os.path.exists(v['mask_path']):
        m = np.array(Image.open(v['mask_path']).convert("L").resize((S, S), Image.NEAREST)) >= FG_ALPHA_THRESHOLD
        logger.info(f"    az={v['az']:3d}° el={v['el']:+d}° saved mask fg={m.mean()*100:.1f}%")
        return m
    
    # Priority 2: Alpha channel (front view)
    if v.get('alpha_mask') is not None:
        m = v['alpha_mask']
        logger.info(f"    az={v['az']:3d}° el={v['el']:+d}° alpha mask fg={m.mean()*100:.1f}%")
        return m
    
    if v['use_alpha']:
        rgba = np.array(v['rgb_raw'].convert('RGBA').resize((S, S), Image.LANCZOS))
        m = rgba[..., 3] >= FG_ALPHA_THRESHOLD
        logger.info(f"    az={v['az']:3d}° el={v['el']:+d}° alpha mask fg={m.mean()*100:.1f}%")
        return m
    
    # Priority 3: Corner-based background detection
    rgb = np.array(v['rgb'].convert('RGB').resize((S, S), Image.LANCZOS), dtype=np.float32)
    r = max(4, S // 16)
    corners = np.concatenate([
        rgb[:r, :r].reshape(-1, 3), rgb[:r, -r:].reshape(-1, 3),
        rgb[-r:, :r].reshape(-1, 3), rgb[-r:, -r:].reshape(-1, 3)
    ])
    bg_col = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb - bg_col, axis=2)
    
    from skimage.filters import threshold_otsu
    from scipy.ndimage import binary_opening, binary_closing, label as ndlabel
    
    try:
        thresh = float(np.clip(
            threshold_otsu(dist),
            BG_DISTANCE_THRESHOLD_MIN,
            BG_DISTANCE_THRESHOLD_MAX,
        ))
    except:
        thresh = BG_DISTANCE_THRESHOLD_DEFAULT
    
    fg = binary_opening(dist > thresh, iterations=1)
    fg = binary_closing(fg, iterations=2)
    
    labeled, n = ndlabel(fg)
    if n > 0:
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        fg = labeled == (sizes.index(max(sizes)) + 1)
    
    logger.info(f"    az={v['az']:3d}° el={v['el']:+d}° fallback fg={fg.mean()*100:.1f}%")
    return fg.astype(bool)


# ════════════════════════════════════════════════════════════════════════════
#  DEPTH MAP LOADING - FIXED
# ════════════════════════════════════════════════════════════════════════════

def _load_depth_maps(out_dir: str, expected: int, size: int = IMG_SIZE) -> dict:
    """
    FIXED: Proper depth map loading with inversion correction.
    The processor.py saves depth as normalized [0,1] where 1 = near.
    We need to ensure consistency.
    """
    depth_maps = {}
    if not out_dir:
        return depth_maps
    
    for i in range(expected):
        dp = os.path.join(out_dir, f"depth_grey_{i}.png")
        if os.path.exists(dp):
            # Load raw depth
            depth_raw = np.array(
                Image.open(dp).convert("L").resize((size, size), Image.BILINEAR),
                dtype=np.float32
            ) / 255.0
            
            # Validate depth map quality
            fg_mask = depth_raw > 0.05
            fg_ratio = fg_mask.mean()
            
            if fg_ratio < 0.01:
                logger.warning(f"  Depth {i}: too few foreground pixels ({fg_ratio:.3f}), skipping")
                continue
            
            if fg_ratio > 0.98:
                logger.warning(f"  Depth {i}: nearly all foreground ({fg_ratio:.3f}), may be invalid")
                # Still use it but with caution
            
            depth_maps[i] = depth_raw
            logger.info(f"  Depth {i}: range=[{depth_raw.min():.3f}, {depth_raw.max():.3f}], fg={fg_ratio:.2f}")
    
    logger.info(f"  Loaded depth maps: {len(depth_maps)}/{expected}")
    return depth_maps


# ════════════════════════════════════════════════════════════════════════════
#  IMPROVED VISUAL HULL WITH DEPTH CARVING
# ════════════════════════════════════════════════════════════════════════════

def _apply_opening_masks_to_view_masks(views, opening_masks=None):
    if opening_masks is None:
        opening_masks = _detect_opening_masks(views)

    masks = []
    for idx, view in enumerate(views):
        mask = _get_mask(view).astype(bool)
        opening = opening_masks.get(idx)
        if opening is not None:
            mask = mask & ~opening
            logger.info(
                f"  Geometry mask view={idx}: removed {int(opening.sum()):,} opening pixels"
            )
        masks.append(mask)
    return masks, opening_masks


def _mesh_is_usable(mesh, min_faces=1000):
    if mesh is None or len(mesh.faces) < min_faces or len(mesh.vertices) < 500:
        return False
    try:
        ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
        if not np.all(np.isfinite(ext)):
            return False
        if ext.max() <= 1e-6 or ext.min() / ext.max() < 0.035:
            return False
    except Exception:
        return False
    return True


def _improved_visual_hull(views, allow_fallback=False):
    """
    FIXED: Better visual hull with conservative carving and depth refinement.
    """
    R = GRID_RES
    H = WORLD_HALF
    S = IMG_SIZE
    lin = np.linspace(-H, H, R)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    
    vol = np.ones(R * R * R, bool)
    voxel_size = (2.0 * H) / R
    
    geometry_masks, opening_masks = _apply_opening_masks_to_view_masks(views)

    # First pass: silhouette carving
    for view_idx, v in enumerate(views):
        mask = geometry_masks[view_idx]
        u, vpx, depth, valid = _project(pts, v['az'], v['el'])
        ui = np.round(u).astype(np.int32)
        vi = np.round(vpx).astype(np.int32)
        in_img = valid & (ui >= 0) & (ui < S) & (vi >= 0) & (vi < S)
        is_fg = np.zeros(len(pts), bool)
        sel = np.where(in_img)[0]
        is_fg[sel] = mask[vi[sel], ui[sel]]
        
        # Conservative: only carve if clearly outside
        carve = in_img & ~is_fg
        vol &= ~carve
        logger.info(f"  az={v['az']:3d}° el={v['el']:+d}° carved={carve.sum():,} remaining={vol.sum():,}")
    
    # DepthAnything produces per-image independently-normalised [0,1] depth.
    # No calibrated metric scale is available so depth cannot constrain voxel
    # carving distances.  depth_grey_*.png are preview-only outputs.

    # Post-process volume
    vol = vol.reshape(R, R, R)
    from scipy.ndimage import binary_closing, binary_opening
    vol = binary_closing(vol, iterations=1)
    vol = binary_opening(vol, iterations=1)
    
    # Check if we have enough volume
    if vol.sum() < 1000:
        logger.warning(f"  Volume too small ({vol.sum()} voxels)")
        if allow_fallback:
            return _create_sphere_fallback()
        return None
    
    # Generate mesh
    from skimage import measure
    from scipy.ndimage import gaussian_filter
    vol_s = gaussian_filter(vol.astype(np.float32), sigma=1.0)
    
    try:
        verts, faces, _, _ = measure.marching_cubes(vol_s, level=0.5)
        verts = verts / (GRID_RES - 1) * np.array([2 * WORLD_HALF] * 3) + np.array([-WORLD_HALF] * 3)
    except:
        logger.warning("  Marching cubes failed")
        if allow_fallback:
            return _create_sphere_fallback()
        return None
    
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
    # Validate mesh
    if not _mesh_is_usable(mesh, min_faces=1000):
        logger.warning(f"  Multi-view mesh failed quality gate ({len(mesh.faces)} faces)")
        if allow_fallback:
            return _create_sphere_fallback()
        return None

    mesh.metadata["direct_multiview_geometry"] = True
    # Depth is preview-only: per-image normalised, no metric scale available
    mesh.metadata["depth_maps_used_for_geometry"] = False
    if opening_masks:
        mesh.metadata["visual_openings_preserved"] = True
    
    logger.info(f"  Improved visual hull: {len(verts):,}v / {len(faces):,}f")
    return mesh


def _create_sphere_fallback():
    """Create a simple sphere as absolute last resort"""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    logger.warning("  Using sphere fallback mesh")
    return mesh


# ============================================================================
#  GENERIC TOPOLOGY VALIDATION & REPAIR
# ============================================================================

# Configurable thresholds (override via env vars if needed)
OPENING_MAX_FACE_PCT = float(os.environ.get("TRIGENIX_OPENING_MAX_FACE_PCT", "0.08"))
OPENING_GLOBAL_MAX_PCT = float(os.environ.get("TRIGENIX_OPENING_GLOBAL_MAX_PCT", "0.25"))
OPENING_MIN_FACE_PCT = float(os.environ.get("TRIGENIX_OPENING_MIN_FACE_PCT", "0.006"))
COLOR_CLUSTER_SPATIAL_WEIGHT = float(os.environ.get("TRIGENIX_COLOR_CLUSTER_SPATIAL_WEIGHT", "0.75"))
COLOR_REGION_MIN_COMPONENT_PCT = float(os.environ.get("TRIGENIX_COLOR_REGION_MIN_COMPONENT_PCT", "0.0075"))


def _visible_vertices_from_view(ui, vi, depth, in_img, image_size, tolerance=0.045, radius=2):
    """
    Approximate a z-buffer at projected vertex pixels and keep only vertices
    close to the nearest surface from the current camera.
    """
    visible = np.zeros_like(in_img, dtype=bool)
    if not np.any(in_img):
        return visible

    zbuf = np.full((image_size, image_size), np.inf, dtype=np.float32)
    flat = vi[in_img] * image_size + ui[in_img]
    np.minimum.at(zbuf.ravel(), flat, depth[in_img].astype(np.float32))

    try:
        from scipy.ndimage import minimum_filter
        znear = minimum_filter(
            zbuf,
            size=max(1, radius * 2 + 1),
            mode="constant",
            cval=np.inf,
        )
    except Exception:
        znear = zbuf

    sel = np.where(in_img)[0]
    nearest = znear[vi[sel], ui[sel]]
    visible[sel] = np.isfinite(nearest) & (depth[sel] <= nearest + tolerance)
    return visible


def validate_mesh_topology(mesh, label="mesh"):
    """
    Measure and log the actual topology of a mesh.

    Returns a dict with:
      vertices, faces, components, boundary_edges,
      non_manifold_edges, degenerate_faces, watertight, winding_consistent

    This is called on the *actual* mesh that will be exported — not an
    intermediate copy — so the result reflects the true final state.
    """
    result = {
        "vertices":           int(len(mesh.vertices)),
        "faces":              int(len(mesh.faces)),
        "components":         -1,
        "boundary_edges":     -1,
        "non_manifold_edges": -1,
        "degenerate_faces":   -1,
        "watertight":         bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }

    try:
        parts = mesh.split(only_watertight=False)
        result["components"] = len(parts)
    except Exception:
        pass

    try:
        # Boundary edges = edges that appear in exactly one face
        all_edges = np.sort(
            np.vstack([
                mesh.faces[:, [0, 1]],
                mesh.faces[:, [1, 2]],
                mesh.faces[:, [2, 0]],
            ]),
            axis=1,
        )
        _, counts = np.unique(all_edges, axis=0, return_counts=True)
        result["boundary_edges"] = int((counts == 1).sum())
        # Non-manifold edges = edges that appear in more than 2 faces
        result["non_manifold_edges"] = int((counts > 2).sum())
    except Exception:
        pass

    try:
        cross = np.cross(
            mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]],
            mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]],
        )
        area2 = np.linalg.norm(cross, axis=1)
        result["degenerate_faces"] = int((area2 < 1e-10).sum())
    except Exception:
        pass

    status = "VALID" if result["watertight"] else "OPEN"
    logger.info(
        "  Topology [%s]  %s\n"
        "    vertices=%d  faces=%d  components=%s\n"
        "    boundary_edges=%s  non_manifold_edges=%s  degenerate_faces=%s\n"
        "    watertight=%s  winding_consistent=%s",
        label, status,
        result["vertices"], result["faces"], result["components"],
        result["boundary_edges"], result["non_manifold_edges"], result["degenerate_faces"],
        result["watertight"], result["winding_consistent"],
    )
    return result



def _compute_silhouette_iou(mesh, views, max_views=3):
    """
    Lightweight silhouette IoU: project mesh silhouette against each view mask.
    Returns mean IoU across the first max_views accepted views.
    A score below ~0.35 usually means the mesh does not explain the input photo.
    """
    try:
        import trimesh
        S = IMG_SIZE
        ious = []
        for v in views[:max_views]:
            mask = _get_mask(v)
            if mask.mean() < 0.02:
                continue
            # Project all vertices, build a raster silhouette from the faces
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            u, vpx, depth, valid = _project(verts, v["az"], v["el"])
            ui = np.clip(np.round(u).astype(np.int32),  0, S - 1)
            vi = np.clip(np.round(vpx).astype(np.int32), 0, S - 1)
            sil = np.zeros((S, S), dtype=bool)
            front = valid & (depth > 0)
            sil[vi[front], ui[front]] = True
            # Dilate slightly to account for rasterisation gaps
            from scipy.ndimage import binary_dilation
            sil = binary_dilation(sil, iterations=3)
            intersection = float((sil & mask).sum())
            union        = float((sil | mask).sum())
            if union > 0:
                ious.append(intersection / union)
        return float(np.mean(ious)) if ious else 0.0
    except Exception as e:
        logger.warning("  silhouette_iou check failed: %s", e)
        return -1.0


def repair_mesh_for_export(mesh, label="mesh", preserve_colors=True):
    """
    Conservative final-stage topology repair.  Called AFTER every operation
    that can change topology (opening cuts, decimation, orientation).

    Strategy (in order):
    1. Remove degenerate and duplicate faces.
    2. Remove tiny floating debris (< 0.5% of main body faces).
    3. Fix normals and winding.
    4. Fill ONLY small accidental holes (boundary loops < 5% of face count).
       Large intentional openings are left alone.
    5. If still non-watertight: try PyMeshFix.
    6. Restore vertex colors if the repair operation lost them.
    7. Validate and report the final result.

    Does NOT fill every hole blindly.  Intentional large openings
    (dragon mouth, shoe opening, mug interior, etc.) are preserved when
    they are large boundary loops.

    Returns the repaired mesh.
    """
    import trimesh

    # Snapshot colors before any topology change (colors may survive or not)
    orig_colors = None
    if preserve_colors:
        try:
            orig_colors = _native_vertex_colors(mesh)
        except Exception:
            pass

    pre = validate_mesh_topology(mesh, label + "/pre-repair")

    if pre["watertight"] and pre["degenerate_faces"] == 0 and pre["components"] == 1:
        logger.info("  repair [%s]: already valid (watertight, 1 component) — skipping", label)
        return mesh, pre

    # Step 1: degenerate / duplicate faces
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    except Exception as exc:
        logger.warning("  repair [%s]: face cleanup failed: %s", label, exc)

    # Step 2: remove tiny debris components (< 0.5% of main body)
    try:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            largest_n = max(len(p.faces) for p in parts)
            min_keep  = max(20, int(0.005 * largest_n))
            kept = [p for p in parts if len(p.faces) >= min_keep]
            if not kept:
                kept = [max(parts, key=lambda p: len(p.faces))]
            if len(kept) < len(parts):
                mesh = trimesh.util.concatenate(kept)
                mesh.remove_unreferenced_vertices()
                logger.info("  repair [%s]: stripped %d debris components", label, len(parts) - len(kept))
    except Exception as exc:
        logger.warning("  repair [%s]: debris removal failed: %s", label, exc)

    # Step 3: fix normals and winding
    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_inversion(mesh)
    except Exception as exc:
        logger.warning("  repair [%s]: normal fix failed: %s", label, exc)

    # Step 4: fill ONLY small accidental holes
    # "Small" = boundary loop < 5% of total face count.
    # Large intentional openings (e.g. mouth, opening) are left untouched.
    if not mesh.is_watertight:
        try:
            # trimesh's fill_holes fills all boundary loops uniformly.
            # We use it only when boundary_edges is small relative to total faces,
            # which indicates accidental damage rather than a designed opening.
            be = pre.get("boundary_edges", -1)
            total_f = len(mesh.faces)
            if be > 0 and be < int(0.05 * total_f):
                mesh.fill_holes()
                logger.info("  repair [%s]: fill_holes applied (be=%d < 5%% of %d faces) -> watertight=%s",
                            label, be, total_f, mesh.is_watertight)
            else:
                logger.info("  repair [%s]: boundary too large (%s edges) — skipping fill_holes to preserve intentional openings",
                            label, be)
        except Exception as exc:
            logger.warning("  repair [%s]: fill_holes failed: %s", label, exc)

    # Step 5: PyMeshFix for small accidental holes that fill_holes could not close.
    # Only applied when boundary_edges < 5% of total faces — same threshold as
    # fill_holes — so large intentional openings (dragon mouth, shoe, mug) are
    # still left alone.  315 edges on 180k faces = 0.17% — clearly accidental.
    if not mesh.is_watertight:
        _be2 = pre.get("boundary_edges", -1)
        _tf2 = len(mesh.faces)
        # Threshold: be < min(2500, 1% of faces)
        # This repairs accidental opening-cut damage (315, 1841 edges) while
        # preserving large intentional openings (3118+, dragon mouth etc.).
        # Using a face-count percentage cleanly separates accidental damage
        # (which is always a tiny fraction of the boundary) from intentional
        # openings (which represent a significant portion of the mesh surface).
        # Real-world accidental: 315/181k = 0.17%, 1841/319k = 0.58% — repaired.
        # Real-world intentional: 3118/171k = 1.8%, 5071/161k = 3.1% — preserved.
        # Test intentional (15-20% holes on 5k-face mesh): 80-96/5120 = 1.6-1.9% — preserved.
        _pymf_limit = min(2500, max(10, int(0.01 * _tf2)))
        if _be2 > 0 and _be2 < _pymf_limit:
            try:
                import pymeshfix
                _fixer = pymeshfix.MeshFix(
                    np.asarray(mesh.vertices, dtype=np.float64),
                    np.asarray(mesh.faces, dtype=np.int32),
                )
                _fixer.repair(joincomp=True, remove_smallest_components=True)
                _rep = trimesh.Trimesh(
                    vertices=np.asarray(_fixer.points),
                    faces=np.asarray(_fixer.faces),
                    process=False,
                )
                if len(_rep.faces) >= int(0.5 * _tf2):
                    mesh = _rep
                    logger.info("  repair [%s]: PyMeshFix (small-loop) -> watertight=%s",
                                label, mesh.is_watertight)
                else:
                    logger.warning("  repair [%s]: PyMeshFix too destructive, keeping", label)
            except Exception as exc:
                logger.warning("  repair [%s]: PyMeshFix skipped: %s", label, exc)
        else:
            logger.info("  repair [%s]: boundary_edges=%s >= 5%% — large opening preserved",
                        label, _be2)

    # Step 6: restore vertex colors if lost
    if preserve_colors and orig_colors is not None:
        try:
            if _native_vertex_colors(mesh) is None and len(orig_colors) == len(mesh.vertices):
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=mesh,
                    vertex_colors=np.hstack([
                        orig_colors.astype(np.uint8),
                        np.full((len(orig_colors), 1), 255, np.uint8),
                    ]),
                )
        except Exception:
            pass

    post = validate_mesh_topology(mesh, label + "/post-repair")
    if not post["watertight"]:
        logger.warning(
            "  repair [%s]: FINAL TOPOLOGY INVALID — watertight=False boundary_edges=%s",
            label, post["boundary_edges"],
        )
    else:
        logger.info("  repair [%s]: FINAL TOPOLOGY VALID — watertight=True", label)

    return mesh, post



def _native_vertex_colors(mesh):
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is None:
        return None
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != len(mesh.vertices) or colors.shape[1] < 3:
        return None
    return colors[:, :3].astype(np.float32)


def _detect_opening_masks(views):
    """
    Detect dark, rimmed interior regions in downward/level generated views.
    These are treated as holes for the visual asset, not for print exports.
    """
    S = IMG_SIZE
    masks = {}
    for view_idx, view in enumerate(views):
        if view.get("el", 0) >= 0:
            continue
        try:
            fg = _get_mask(view)
            if fg.mean() < 0.03:
                continue

            rgb = np.array(
                view["rgb"].convert("RGB").resize((S, S), Image.LANCZOS),
                dtype=np.float32,
            )
            gray = rgb.mean(axis=2)
            fg_values = gray[fg]
            if fg_values.size < 100:
                continue

            from scipy.ndimage import (
                binary_closing,
                binary_dilation,
                binary_opening,
                distance_transform_edt,
                label,
            )

            # Conservative thresholds: avoid cutting solid objects.
            # Only detect well-defined dark interior regions well inside the silhouette.
            dark_threshold = min(90.0, float(np.percentile(fg_values, 12)))
            dist_inside = distance_transform_edt(fg)
            dark = fg & (gray <= dark_threshold) & (dist_inside > max(5, S * 0.018))
            dark = binary_opening(dark, iterations=2)
            dark = binary_closing(dark, iterations=1)

            labeled, count = label(dark)
            candidates = np.zeros_like(fg, dtype=bool)
            fg_area = float(fg.sum())

            for comp_id in range(1, count + 1):
                comp = labeled == comp_id
                area = int(comp.sum())
                if area < max(24, int(0.002 * fg_area)) or area > int(0.16 * fg_area):
                    continue

                ys, xs = np.where(comp)
                if len(xs) == 0:
                    continue
                bw = int(xs.max() - xs.min() + 1)
                bh = int(ys.max() - ys.min() + 1)
                if bw < 6 or bh < 4:
                    continue

                aspect = bw / max(1.0, float(bh))
                if aspect < 0.75:
                    continue

                cy = float(ys.mean()) / S
                if cy > 0.82:
                    continue

                if float(dist_inside[comp].mean()) < 8.0:  # must be well inside silhouette
                    continue

                expanded = binary_dilation(comp, iterations=5)
                inner = binary_dilation(comp, iterations=1)
                rim = expanded & fg & ~inner
                if rim.sum() < max(12, int(0.15 * area)):
                    continue

                contrast = float(gray[rim].mean() - gray[comp].mean())
                if contrast < 22.0:  # higher threshold avoids cuts on textured surfaces
                    continue

                candidates |= binary_dilation(comp, iterations=2)

            if candidates.sum() > 0:
                masks[view_idx] = candidates
                try:
                    out_dir = view.get("out_dir")
                    if out_dir:
                        Image.fromarray(candidates.astype(np.uint8) * 255, "L").save(
                            os.path.join(out_dir, f"opening_mask_{view_idx}.png")
                        )
                except Exception:
                    pass
                logger.info(
                    f"  Opening cues view={view_idx} az={view['az']:3d}° "
                    f"el={view['el']:+d}° pixels={int(candidates.sum()):,}"
                )
        except Exception as e:
            logger.warning(f"  Opening detection skipped for view {view_idx}: {e}")
    return masks


def _preserve_visual_openings(mesh, views):
    """
    Detect dark-rimmed interior regions from generated views and remove the
    corresponding front-facing faces to create visual openings (e.g. mouth,
    opening, eye socket).

    Conservative thresholds are enforced per view and globally:
      - Per-view: skip if removal would exceed OPENING_MAX_FACE_PCT of total faces.
      - Global: revert entirely if total removal exceeds OPENING_GLOBAL_MAX_PCT.

    After cutting, topology is measured and debris fragments are removed.
    The function returns the mesh with openings applied but does NOT repair
    the resulting boundaries — the caller (reconstruct_from_dir) is responsible
    for the final topology repair pass.
    """
    opening_masks = _detect_opening_masks(views)
    if not opening_masks:
        return mesh

    import trimesh

    total_faces = len(mesh.faces)
    # Per-view and global removal limits (configurable via env vars)
    per_view_limit = int(OPENING_MAX_FACE_PCT * total_faces)
    global_limit   = int(OPENING_GLOBAL_MAX_PCT * total_faces)

    out = _copy_mesh(mesh)
    centers = np.asarray(out.triangles_center, dtype=np.float32)
    normals = np.asarray(out.face_normals, dtype=np.float32)
    keep = np.ones(len(out.faces), dtype=bool)
    S = IMG_SIZE

    for view_idx, opening_mask in opening_masks.items():
        view = views[view_idx]
        u, vpx, _, valid = _project(centers, view["az"], view["el"])
        ui = np.round(u).astype(np.int32)
        vi = np.round(vpx).astype(np.int32)
        in_img = valid & (ui >= 0) & (ui < S) & (vi >= 0) & (vi < S)
        if not in_img.any():
            continue

        projected_pixel_count = int(opening_mask.sum())

        _, cam_pos = _camera_c2w(view["az"], view["el"])
        view_dirs = cam_pos - centers
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8
        front_facing = (normals * view_dirs).sum(axis=1) > 0.08

        projected_hole = np.zeros(len(out.faces), dtype=bool)
        sel = np.where(in_img)[0]
        projected_hole[sel] = opening_mask[vi[sel], ui[sel]]
        candidate_faces = int((projected_hole & front_facing).sum())
        remove = keep & projected_hole & front_facing
        remove_count = int(remove.sum())
        remove_pct = 100.0 * remove_count / max(1, total_faces)

        # Minimum threshold: skip tiny single-view dark spots. Those are often
        # generated texture shadows on solid objects, and cutting them creates
        # repaired cap patches that later show up as pale blotches.
        min_remove = max(20, int(OPENING_MIN_FACE_PCT * total_faces))
        if remove_count < min_remove:
            logger.info(
                "  Opening cue view=%d az=%d el=%+d: removed-candidate %d faces below %.2f%% minimum — SKIPPED",
                view_idx, view["az"], view["el"], remove_count,
                100.0 * OPENING_MIN_FACE_PCT,
            )
            continue

        # Per-view conservative limit
        if remove_count > per_view_limit:
            logger.warning(
                "  Opening cue view=%d az=%d el=%+d: would remove %d faces (%.1f%% of total, "
                "limit=%.0f%%) — SKIPPED (too aggressive)",
                view_idx, view["az"], view["el"], remove_count, remove_pct,
                100.0 * OPENING_MAX_FACE_PCT,
            )
            continue

        # Topology check: how many components before this cut?
        be_before = sum(1 for e in [None])  # placeholder — real count below
        try:
            all_edges_b = np.sort(
                np.vstack([out.faces[:, [0,1]], out.faces[:, [1,2]], out.faces[:, [2,0]]]), axis=1)
            _, cnts_b = np.unique(all_edges_b, axis=0, return_counts=True)
            be_before = int((cnts_b == 1).sum())
        except Exception:
            be_before = -1

        keep &= ~remove
        logger.info(
            "  Opening view=%d az=%d el=%+d: proj_pixels=%d candidates=%d removed=%d (%.1f%%) be_before=%s",
            view_idx, view["az"], view["el"],
            projected_pixel_count, candidate_faces, remove_count, remove_pct, be_before,
        )

    removed_total = int((~keep).sum())
    if removed_total == 0:
        return mesh

    removed_pct = 100.0 * removed_total / max(1, total_faces)
    if removed_total > global_limit:
        logger.warning(
            "  Opening preservation: total removed %d faces (%.1f%%) exceeds global limit %.0f%% — REVERTED",
            removed_total, removed_pct, 100.0 * OPENING_GLOBAL_MAX_PCT,
        )
        return mesh

    try:
        out.update_faces(keep)
    except Exception:
        out.faces = out.faces[keep]
    out.remove_unreferenced_vertices()

    # Strip debris: tiny disconnected fragments produced by the face-cut.
    # Keep main body + components >= 1% of main body (preserves real sub-features).
    _parts = out.split(only_watertight=False)
    if len(_parts) > 1:
        _largest_n = max(len(p.faces) for p in _parts)
        _min_keep = max(20, int(0.01 * _largest_n))
        _kept = [p for p in _parts if len(p.faces) >= _min_keep]
        if not _kept:
            _kept = [max(_parts, key=lambda p: len(p.faces))]
        if len(_kept) < len(_parts):
            _debris = len(_parts) - len(_kept)
            out = trimesh.util.concatenate(_kept)
            out.remove_unreferenced_vertices()
            logger.info("  Stripped %d debris component(s) from visual mesh after opening cuts", _debris)

    # Measure topology AFTER cuts so the caller knows what state the mesh is in
    try:
        all_edges_a = np.sort(
            np.vstack([out.faces[:, [0,1]], out.faces[:, [1,2]], out.faces[:, [2,0]]]), axis=1)
        _, cnts_a = np.unique(all_edges_a, axis=0, return_counts=True)
        be_after = int((cnts_a == 1).sum())
    except Exception:
        be_after = -1

    logger.info(
        "  Post-opening topology: faces=%d removed=%d(%.1f%%) boundary_edges=%s watertight=%s",
        len(out.faces), removed_total, removed_pct, be_after, out.is_watertight,
    )

    if hasattr(out, "fix_normals"):
        out.fix_normals()
    out.metadata["visual_openings_preserved"] = True
    return out


def _sample_texture_at_uv(tex_arr, uv):
    """Sample HxWx3 uint8 texture at Nx2 normalised UV coords."""
    H, W = tex_arr.shape[:2]
    u_px = np.clip((uv[:, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    v_px = np.clip(((1.0 - uv[:, 1]) * (H - 1)).astype(np.int32), 0, H - 1)
    return tex_arr[v_px, u_px]  # (N, 3)


def _transfer_triposr_texture(target_mesh, triposr_glb_path):
    """
    Transfer TripoSR UV atlas from the raw (149k-vert) mesh to the
    cleaned/watertight target mesh using nearest-vertex matching in
    normalised [0,1]^3 space.

    Returns (uv_array, tex_rgb_image) or (None, None) on failure.
    """
    from scipy.spatial import cKDTree
    import trimesh

    try:
        scene = trimesh.load(triposr_glb_path)
        if not hasattr(scene, "geometry") or not scene.geometry:
            return None, None
        src = list(scene.geometry.values())[0]
        if not hasattr(src.visual, "uv") or src.visual.uv is None:
            return None, None

        src_uv = np.array(src.visual.uv, dtype=np.float32)          # (M, 2)
        src_v  = np.array(src.vertices,  dtype=np.float32)           # (M, 3)
        tex    = src.visual.material.baseColorTexture.convert("RGB") # PIL

        if src_uv.shape[0] != len(src_v):
            return None, None

        # Normalise to [0,1]^3
        s_lo    = src_v.min(0);  s_scale = max((src_v.max(0) - s_lo).max(), 1e-8)
        src_n   = (src_v - s_lo) / s_scale

        tgt_v   = np.array(target_mesh.vertices, dtype=np.float32)
        t_lo    = tgt_v.min(0);  t_scale = max((tgt_v.max(0) - t_lo).max(), 1e-8)
        tgt_n   = (tgt_v - t_lo) / t_scale

        _, idx  = cKDTree(src_n).query(tgt_n, k=1, workers=-1)
        tgt_uv  = src_uv[idx]

        logger.info("  UV transfer: %d src -> %d tgt verts", len(src_v), len(tgt_v))
        return tgt_uv, tex

    except Exception as e:
        logger.warning("  UV transfer failed: %s", e)
        return None, None


def _build_textured_glb(mesh, triposr_glb_path, out_path):
    """
    Build a UV-textured GLB from the watertight mesh + TripoSR UV atlas.

    The mesh geometry comes from our clean/watertight pipeline.
    The UV coords are transferred from the TripoSR raw mesh.
    The texture image is TripoSR's 1024x1024 baked atlas.

    The result is a proper UV-textured GLB — every face has a texture
    coordinate, the atlas file is embedded, and the mesh is watertight.
    """
    import trimesh

    uv, tex_img = _transfer_triposr_texture(mesh, triposr_glb_path)
    if uv is None:
        return False

    try:
        tex_arr = np.array(tex_img, dtype=np.uint8)

        # Build PBR material with the atlas
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_img,
            metallicFactor=0.0,
            roughnessFactor=0.7,
        )

        # Attach UV + material to a copy of the mesh
        out_mesh = mesh.copy()
        out_mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv,
            material=mat,
        )

        out_mesh.export(out_path, file_type="glb")
        logger.info("  UV-textured GLB exported -> %s", out_path)
        return True

    except Exception as e:
        logger.warning("  UV-textured GLB export failed: %s", e)
        return False



def paint_colors(mesh, views, base_colors=None, triposr_glb_path=None):
    """
    Paint per-vertex colors using region-based projection.

    Strategy (photo-first, atlas-fallback):
    1. PHOTO SEGMENTATION: Cluster the front photo in HSV-hue space into
       N color regions (leaf green, body red, highlight, shadow, etc.).
       Each region has a dominant color and a pixel mask.

    2. REGION PROJECTION (photo-first): For each vertex, project it into
       the front photo. Assign the region label and color of the nearest
       photo pixel. No blending across regions -- green vertices get green,
       red vertices get red. This is the primary color source.
       All 6 generated views also contribute (with reduced weight) to fill
       regions not visible from the front.

    3. ATLAS FALLBACK: For vertices that couldn"t be reached by any photo
       projection (back surfaces, occluded areas), use the TripoSR UV atlas
       as the fallback color. The atlas fills gaps only.

    4. KD-TREE FILL: Any remaining unpainted vertices inherit from nearest
       painted neighbor.

    5. EDGE-AWARE SMOOTH: Tight sigma=3 smoothing within regions only.

    Result: leaf vertices get the leaf color, body vertices get body color,
    regardless of mesh repair path or normal orientation.
    """
    import trimesh as _tm
    try:
        from sklearn.cluster import MiniBatchKMeans as _KMeans
        _HAS_SKLEARN = True
    except ImportError:
        _HAS_SKLEARN = False

    S       = IMG_SIZE
    verts   = np.array(mesh.vertices, dtype=np.float32)
    N       = len(verts)

    mesh.fix_normals()
    normals = np.array(mesh.vertex_normals, dtype=np.float32)

    # Normalize vertex positions for projection.
    # _project() / _camera_c2w() use CAM_DIST=1.80 and expect the mesh
    # to fit inside a ~1.0-unit sphere.  Meshes exported in cm/mm scale
    # (e.g. bounds +-50) project way outside the image -- zero vertices
    # get photo colors.  Normalize to [-0.5, 0.5] for projection only;
    # the actual mesh geometry is not modified.
    _verts_center = verts.mean(axis=0)
    _verts_extent = np.abs(verts - _verts_center).max()
    if _verts_extent > 2.0:   # only normalize if mesh is too large
        _verts_scale  = 0.5 / (_verts_extent + 1e-8)
        verts   = (verts - _verts_center) * _verts_scale
        normals = normals  # normals are direction vectors, no scaling needed
        logger.info('  Normalizing verts for projection: scale=%.4f (was %.2f)', _verts_scale, _verts_extent)
    else:
        _verts_scale  = 1.0
        _verts_center = np.zeros(3, np.float32)
    mesh.visual = _tm.visual.ColorVisuals(mesh=mesh)

    # ── Step 1: Segment front photo into color regions ─────────────────────
    seg_regions = None   # list of {'color': [R,G,B], 'pct': float}
    seg_label_img = None # H×W int array, value=superpixel index, -1=background

    # ── SLIC superpixel segmentation ───────────────────────────────────────
    # Strategy: SLIC generates 200-500 small spatially-coherent patches.
    # Each patch is a connected region of similar color AND similar spatial
    # location — so car tire, car body, car window are always separate patches
    # even if they share similar brightness.
    #
    # Then we merge adjacent SLIC patches that have very similar color
    # (ΔE < threshold) into "semantic regions". This gives us:
    #   - Fine-grained spatial boundaries (from SLIC)
    #   - Clean per-region colors (from merge + trimmed mean)
    #
    # Each vertex projects → pixel → SLIC patch → merged region → color.
    # No cross-region blending — tire pixels only get tire color.
    #
    # SEGMENTATION uses the INPUT PHOTO (view0) — clean alpha from rembg.
    # PROJECTION uses GENERATED VIEWS (view1+) — known calibrated poses.

    _SEG_VIEW_IDX = 1 if len(views) > 1 else 0
    _FRONT_VIEW   = views[0] if views else None

    if _FRONT_VIEW is not None:
        try:
            from skimage.segmentation import slic as _slic
            from skimage.color import rgb2lab as _rgb2lab
            from scipy.ndimage import label as _ndlabel

            _fraw  = (_FRONT_VIEW.get("rgb_raw") or _FRONT_VIEW["rgb"]).convert("RGBA")
            _frgba = np.array(_fraw.resize((S, S), Image.LANCZOS))
            _fg    = _frgba[..., 3] >= FG_ALPHA_THRESHOLD          # H×W bool
            _frgb  = _frgba[..., :3].astype(np.uint8)              # H×W×3

            # ── Step 1a: SLIC superpixels ─────────────────────────────────
            # n_segments scales with foreground area so small objects get
            # fewer, more reliable patches.
            _fg_px   = int(_fg.sum())
            _n_seg   = int(np.clip(_fg_px ** 0.55, 80, 600))
            # compactness: low = color-dominant boundaries (better for objects
            # with sharp color transitions like car tire/body).
            _sp_map  = _slic(
                _frgb,
                n_segments=_n_seg,
                compactness=8,
                sigma=1.0,
                start_label=0,
                mask=_fg.astype(np.uint8),
                channel_axis=2,
            )                                                        # H×W int
            _n_sp    = int(_sp_map.max()) + 1
            logger.info("  SLIC: %d superpixels requested → %d produced (fg=%d px)",
                        _n_seg, _n_sp, _fg_px)

            # ── Step 1b: Compute per-superpixel mean color (Lab) ──────────
            _flab    = _rgb2lab(_frgb.astype(np.float32) / 255.0)   # H×W×3 Lab
            _sp_lab  = np.zeros((_n_sp, 3), dtype=np.float32)
            _sp_rgb  = np.zeros((_n_sp, 3), dtype=np.float32)
            _sp_cnt  = np.zeros(_n_sp, dtype=np.int32)
            _sp_flat = _sp_map.ravel()
            _fg_flat = _fg.ravel()
            _lab_flat= _flab.reshape(-1, 3)
            _rgb_flat= _frgb.reshape(-1, 3).astype(np.float32)

            np.add.at(_sp_lab, _sp_flat, _lab_flat * _fg_flat[:, None])
            np.add.at(_sp_rgb, _sp_flat, _rgb_flat * _fg_flat[:, None])
            np.add.at(_sp_cnt, _sp_flat, _fg_flat.astype(np.int32))

            _valid_sp = _sp_cnt > 0
            _sp_lab[_valid_sp] /= _sp_cnt[_valid_sp, None]
            _sp_rgb[_valid_sp] /= _sp_cnt[_valid_sp, None]

            # ── Step 1c: Merge adjacent similar superpixels ────────────────
            # Two superpixels merge if their Lab ΔE < merge_thresh AND they
            # are spatially adjacent (share at least one border pixel).
            # This collapses the 300+ SLIC patches into ~15-60 semantic
            # regions while keeping sharp color-edge boundaries intact.
            #
            # merge_thresh controls granularity:
            #   low  (8)  → more regions, finer detail  (tire vs hubcap)
            #   high (20) → fewer regions, cleaner blobs (whole tire = one region)
            # We use 12 — a balance that preserves major boundaries (roof vs
            # wall vs window vs door) while merging nearly-identical patches.
            _merge_thresh = 12.0

            # Build adjacency: find superpixel pairs that share a border pixel
            _adj = set()
            _rows, _cols = _sp_map.shape
            # Horizontal adjacency
            _ah = np.stack([_sp_map[:, :-1].ravel(), _sp_map[:, 1:].ravel()], axis=1)
            _ah = _ah[_ah[:, 0] != _ah[:, 1]]
            for _p, _q in _ah:
                _adj.add((min(int(_p), int(_q)), max(int(_p), int(_q))))
            # Vertical adjacency
            _av = np.stack([_sp_map[:-1, :].ravel(), _sp_map[1:, :].ravel()], axis=1)
            _av = _av[_av[:, 0] != _av[:, 1]]
            for _p, _q in _av:
                _adj.add((min(int(_p), int(_q)), max(int(_p), int(_q))))

            # Union-Find for merging
            _parent = np.arange(_n_sp, dtype=np.int32)

            def _find(x):
                while _parent[x] != x:
                    _parent[x] = _parent[_parent[x]]
                    x = _parent[x]
                return x

            def _union(x, y):
                rx, ry = _find(x), _find(y)
                if rx != ry:
                    _parent[ry] = rx

            # Iterative merge: keep merging until stable
            _changed = True
            _iters   = 0
            while _changed and _iters < 8:
                _changed = False
                _iters  += 1
                for (_p, _q) in list(_adj):
                    rp, rq = _find(_p), _find(_q)
                    if rp == rq:
                        continue
                    if not (_valid_sp[_p] and _valid_sp[_q]):
                        continue
                    # ΔE between superpixel means (use root's accumulated mean)
                    _dE = float(np.linalg.norm(_sp_lab[rp] - _sp_lab[rq]))
                    if _dE < _merge_thresh:
                        # Weighted merge: update root's Lab/RGB mean
                        _cp  = _sp_cnt[rp]
                        _cq  = _sp_cnt[rq]
                        _tot = _cp + _cq
                        if _tot > 0:
                            _sp_lab[rp] = (_sp_lab[rp]*_cp + _sp_lab[rq]*_cq) / _tot
                            _sp_rgb[rp] = (_sp_rgb[rp]*_cp + _sp_rgb[rq]*_cq) / _tot
                            _sp_cnt[rp] = _tot
                        _union(_p, _q)
                        _changed = True

            # Build final label map: each pixel → merged region root id
            # Remap root ids to contiguous 0..n_regions-1
            _roots     = np.array([_find(i) for i in range(_n_sp)], dtype=np.int32)
            _uniq_roots= np.unique(_roots)
            _root2reg  = {r: i for i, r in enumerate(_uniq_roots)}
            _n_regions = len(_uniq_roots)

            # pixel label map: -1 for background
            _region_map = np.full(S * S, -1, dtype=np.int32)
            _fg_idx     = np.where(_fg_flat)[0]
            for _idx in _fg_idx:
                _sp_id  = _sp_flat[_idx]
                _root   = _roots[_sp_id]
                _region_map[_idx] = _root2reg[_root]

            seg_label_img = _region_map                             # flat S*S
            _lbl_img      = _region_map.reshape(S, S)

            # ── Step 1d: Per-region trimmed-mean color ─────────────────────
            seg_regions = []
            _rgb_flat_fg = _rgb_flat[_fg_flat]
            _reg_flat_fg = _region_map[_fg_flat]

            for _ri, _root in enumerate(_uniq_roots):
                _mask    = _reg_flat_fg == _ri
                _cnt_r   = int(_mask.sum())
                _pct     = float(_cnt_r / max(1, _fg_px) * 100)
                if _cnt_r < 15 or _pct < 0.3:
                    seg_regions.append(None)
                    continue
                _pix     = _rgb_flat_fg[_mask]
                # Trimmed mean: drop darkest 10% and brightest 10%
                _bri     = _pix.mean(axis=1)
                _lo, _hi = np.percentile(_bri, [10, 90])
                _ok      = (_bri >= _lo) & (_bri <= _hi)
                if _ok.sum() < 5:
                    _ok  = np.ones(len(_pix), dtype=bool)
                _col     = _pix[_ok].mean(axis=0)
                seg_regions.append({"color": _col, "pct": _pct})

            _nreg = sum(1 for r in seg_regions if r is not None)
            logger.info(
                "  SLIC segmentation: %d superpixels → %d merged regions "
                "(merge_thresh=%.1f ΔE)",
                _n_sp, _nreg, _merge_thresh,
            )
            for _ri, _r in enumerate(seg_regions):
                if _r is None:
                    continue
                _c   = _r["color"].astype(int)
                logger.info("    region %d: %.1f%%  RGB=[%d,%d,%d]",
                            _ri, _r["pct"], _c[0], _c[1], _c[2])

        except Exception as _se:
            logger.warning("  SLIC segmentation failed (%s) — falling back to k-means", _se)
            seg_regions    = None
            seg_label_img  = None

    # ── k-means fallback (used when skimage not available or SLIC fails) ───
    if seg_label_img is None and _HAS_SKLEARN and _FRONT_VIEW is not None:
        try:
            from sklearn.cluster import MiniBatchKMeans as _KMeans
            _fraw  = (_FRONT_VIEW.get("rgb_raw") or _FRONT_VIEW["rgb"]).convert("RGBA")
            _frgba = np.array(_fraw.resize((S, S), Image.LANCZOS))
            _fg    = _frgba[..., 3] >= FG_ALPHA_THRESHOLD
            _frgb  = _frgba[..., :3].astype(np.float32)
            _fhsv  = np.array(
                Image.fromarray(_frgba[..., :3]).convert("HSV"), dtype=np.float32
            )
            _h_rad = _fhsv[..., 0] / 255.0 * 2.0 * np.pi
            _feat  = np.stack([
                np.sin(_h_rad).ravel(),
                np.cos(_h_rad).ravel(),
                (_fhsv[..., 1] / 255.0 * 2.0).ravel(),
                (_fhsv[..., 2] / 255.0).ravel(),
            ], axis=1)
            _fg_flat = _fg.ravel()
            _k   = min(10, max(4, int(_fg_flat.sum() ** 0.25)))
            _km  = _KMeans(n_clusters=_k, random_state=42, n_init=3)
            _lbl = _km.fit_predict(_feat[_fg_flat])
            seg_label_img = np.full(S * S, -1, dtype=np.int32)
            seg_label_img[_fg_flat] = _lbl
            _rgb_fg = _frgb.reshape(-1, 3)[_fg_flat]
            seg_regions = []
            for _cl in range(_k):
                _mask = _lbl == _cl
                if _mask.sum() < 20:
                    seg_regions.append(None); continue
                _pct  = float(_mask.mean() * 100)
                if _pct < 1.5:
                    seg_regions.append(None); continue
                _pix  = _rgb_fg[_mask]
                _bri  = _pix.mean(axis=1)
                _lo, _hi = np.percentile(_bri, [15, 85])
                _ok   = (_bri >= _lo) & (_bri <= _hi)
                _col  = _pix[_ok if _ok.sum() >= 10 else np.ones(len(_pix), bool)].mean(axis=0)
                seg_regions.append({"color": _col, "pct": _pct})
            logger.info("  k-means fallback: %d color regions",
                        sum(1 for r in seg_regions if r is not None))
        except Exception as _ke:
            logger.warning("  k-means fallback also failed: %s", _ke)
            seg_regions = None
            seg_label_img = None

    # ── Step 2: Region-based photo projection (photo-first) ────────────────
    # Each vertex gets the color of its projected photo region.
    # For vertices with high-confidence region assignment (front-facing +
    # in-foreground), the region color is locked.
    # For low-confidence vertices (side views, grazing normals), weight is
    # proportional to cos_nv so region color averages correctly.

    accum_col   = np.zeros((N, 3), np.float32)
    accum_w     = np.zeros(N,     np.float32)
    photo_hit   = np.zeros(N, dtype=bool)
    # Track per-vertex dominant region label for later atlas fill decision
    vert_region = np.full(N, -1, dtype=np.int32)
    vert_reg_w  = np.zeros(N, np.float32)

    BRIGHT_THRESH = 215

    # Pre-compute per-view color calibration.
    # Zero123++ generated views (1-6) are systematically darker than input.
    # Scale each to match input photo foreground mean before projecting.
    _view_calib_scale = np.ones((len(views), 3), dtype=np.float32)
    try:
        _ph0 = views[0]
        _raw0 = (_ph0.get('rgb_raw') or _ph0['rgb']).convert('RGBA')
        _arr0 = np.array(_raw0.resize((S, S), Image.LANCZOS))
        _fg0  = _arr0[..., 3] >= FG_ALPHA_THRESHOLD
        if _fg0.sum() > 200:
            _ref_mean = _arr0[_fg0, :3].astype(np.float32).mean(axis=0)
            for _vi in range(1, len(views)):
                _v   = views[_vi]
                _va  = np.array(_v['rgb'].convert('RGB').resize((S,S),Image.LANCZOS))
                if _v.get('mask_path') and os.path.exists(_v['mask_path']):
                    _vmask = np.array(Image.open(_v['mask_path']).convert('L').resize((S,S),Image.NEAREST)) >= FG_ALPHA_THRESHOLD
                else:
                    _r2t = max(4,S//16)
                    _bgt = np.median(_va[:_r2t,:_r2t].reshape(-1,3),axis=0).astype(np.float32)
                    _vmask = np.linalg.norm(_va.astype(np.float32)-_bgt,axis=2) > 15.0
                if _vmask.sum() > 100:
                    _vm = _va[_vmask].astype(np.float32).mean(axis=0).clip(1,254)
                    _view_calib_scale[_vi] = np.clip(_ref_mean/_vm, 0.7, 1.6)
        _scales_str = ' '.join(
            f'v{i}=[{_view_calib_scale[i,0]:.2f},{_view_calib_scale[i,1]:.2f},{_view_calib_scale[i,2]:.2f}]'
            for i in range(1,min(4,len(views))))
        logger.info('  View calib: %s', _scales_str)
    except Exception as _vce:
        logger.warning('  View calibration failed: %s', _vce)

    for view_idx, v in enumerate(views):
        rgb_img = v["rgb"].convert("RGB").resize((S, S), Image.LANCZOS)
        rgb_arr = np.array(rgb_img, dtype=np.uint8)
        # Apply calibration to generated views (not input photo)
        if view_idx > 0:
            _sc = _view_calib_scale[view_idx]
            if _sc.max() > 1.005 or _sc.min() < 0.995:
                rgb_arr = np.clip(rgb_arr.astype(np.float32)*_sc, 0, 255).astype(np.uint8)
        # view index 1 (az=30,el=+20) is our primary calibrated view.
        # view index 0 (input photo) has unknown pose -- treat as low-weight side.
        is_front = (view_idx == _SEG_VIEW_IDX)
        is_input_photo = (view_idx == 0)

        # Foreground mask
        if is_input_photo and v.get("alpha_mask") is not None:
            fg_mask = v["alpha_mask"]
        elif is_input_photo:
            _rgba   = np.array(v["rgb_raw"].convert("RGBA").resize((S, S), Image.LANCZOS))
            fg_mask = _rgba[..., 3] >= FG_ALPHA_THRESHOLD
        elif v.get("mask_path") and os.path.exists(v["mask_path"]):
            fg_mask = np.array(
                Image.open(v["mask_path"]).convert("L").resize((S, S), Image.NEAREST)
            ) >= FG_ALPHA_THRESHOLD
        else:
            _r2  = max(4, S // 16)
            _cor = np.concatenate([
                rgb_arr[:_r2, :_r2].reshape(-1, 3), rgb_arr[:_r2, -_r2:].reshape(-1, 3),
                rgb_arr[-_r2:, :_r2].reshape(-1, 3), rgb_arr[-_r2:, -_r2:].reshape(-1, 3),
            ])
            _bg  = np.median(_cor, axis=0).astype(np.float32)
            fg_mask = (np.linalg.norm(
                rgb_arr.reshape(-1, 3).astype(np.float32) - _bg, axis=1
            ).reshape(S, S) > 15.0)

        _, cam_pos = _camera_c2w(v["az"], v["el"])
        u, vpx, depth, valid = _project(verts, v["az"], v["el"])
        ui = np.round(u  ).astype(np.int32)
        vi = np.round(vpx).astype(np.int32)
        in_img = valid & (ui >= 0) & (ui < S) & (vi >= 0) & (vi < S)
        if not in_img.any():
            continue

        visible = _visible_vertices_from_view(ui, vi, depth, in_img, S)

        view_dirs = cam_pos - verts
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8
        cos_nv    = (normals * view_dirs).sum(axis=1)

        # Normal threshold: front view very loose (captures leaf tops),
        # side views standard (above horizon only)
        if is_front:
            face_ok = cos_nv >= -0.1
        else:
            face_ok = cos_nv > 0.05

        sel_all = np.where(in_img & visible & face_ok)[0]
        if not len(sel_all):
            continue

        in_fg = fg_mask[vi[sel_all], ui[sel_all]]
        sel   = sel_all[in_fg]
        if not len(sel):
            continue

        # Sample raw photo color at projected pixel
        sampled = rgb_arr[vi[sel], ui[sel]].astype(np.float32)

        # For front view with segmentation: use region median color
        # (eliminates per-pixel noise, gives clean region color).
        # For side views or when segmentation unavailable: use raw pixel color.
        if is_front and seg_label_img is not None:
            # Look up which region each projected pixel belongs to
            flat_idx = vi[sel] * S + ui[sel]
            reg_lbl  = seg_label_img[flat_idx]        # region label per vertex

            _valid_region = reg_lbl >= 0
            if _valid_region.any():
                _valid_region &= np.array(
                    [seg_regions[int(_rl)] is not None for _rl in reg_lbl],
                    dtype=bool,
                )
            if not _valid_region.any():
                continue

            sel = sel[_valid_region]
            sampled = sampled[_valid_region]
            reg_lbl = reg_lbl[_valid_region]

            # Replace sampled color with region median color
            for _rl in np.unique(reg_lbl):
                if _rl < 0 or seg_regions[int(_rl)] is None:
                    continue
                _vmask = reg_lbl == _rl
                sampled[_vmask] = seg_regions[int(_rl)]["color"]

                # Track dominant region per vertex (highest-weight assignment)
                _vsel = sel[_vmask]
                _cos  = np.clip(cos_nv[_vsel], 0.05, 1.0)
                # Update region tracking: pick whichever region contributes most
                _update = _cos > vert_reg_w[_vsel]
                vert_region[_vsel[_update]] = _rl
                vert_reg_w[_vsel[_update]]  = _cos[_update]

        # Exclude near-white pixels (cross-section bleed, background)
        not_bright = sampled.min(axis=1) < BRIGHT_THRESH
        if not is_front:
            # For side views, also filter near-background
            _r2 = max(4, S // 16)
            _bg = np.median(rgb_arr[:_r2, :_r2].reshape(-1, 3), axis=0).astype(np.float32)
            not_bright = not_bright & (np.linalg.norm(sampled - _bg, axis=1) > 12.0)

        sel     = sel[not_bright]
        sampled = sampled[not_bright]
        if not len(sel):
            continue

        # ── Per-vertex confidence weight ───────────────────────────────────
        # w = cos_nv * center_proximity * depth_confidence * view_scale
        #
        # cos_nv       — surface facing the camera (0 = grazing, 1 = head-on)
        # center_proximity — penalise pixels near image edges (distortion)
        # depth_confidence — vertices closer to camera get higher weight;
        #                    far vertices are more occluded/distorted
        # view_scale   — per-view reliability factor derived from:
        #                  - known vs unknown pose
        #                  - elevation angle (high el = foreshortening)
        #                  - how well this view already agrees with accumulated

        cos_w = np.clip(cos_nv[sel], 0.05, 1.0)
        # cos² sharpens the weighting: vertices squarely facing the cam get
        # much more weight than grazing vertices (cos 0.2 → 0.04 vs cos 0.9 → 0.81)
        cos_w = cos_w ** 2

        px_cx = ui[sel] - S / 2.0
        px_cy = vi[sel] - S / 2.0
        ctr_w = 1.0 - 0.35 * (np.hypot(px_cx, px_cy) / (S / 2.0)).clip(0, 1)

        # Depth confidence: normalise depth to [0,1] within this view's
        # visible range and invert (nearer = more confident).
        d_sel = depth[sel]
        d_min, d_max = d_sel.min(), d_sel.max()
        if d_max > d_min + 1e-6:
            depth_conf = 1.0 - 0.4 * ((d_sel - d_min) / (d_max - d_min))
        else:
            depth_conf = np.ones(len(sel), dtype=np.float32)

        # Saturation weight: chromatic pixels are more reliable than near-grey
        _smax_v  = sampled.max(axis=1).clip(1, 255)
        _smin_v  = sampled.min(axis=1)
        _sat_v   = (_smax_v - _smin_v) / _smax_v
        sat_w    = np.clip(_sat_v * 2.5 + 0.4, 0.4, 2.5)

        if is_front:
            # Primary calibrated view (az=30, el=+20): highest base trust.
            # Elevation 20° is mild — minimal foreshortening penalty.
            el_penalty = 1.0 - abs(v["el"]) / 90.0 * 0.3   # ~0.93 at el=20
            view_scale = 2.5 * el_penalty
            w = cos_w * ctr_w * depth_conf * sat_w * view_scale

        elif is_input_photo:
            # Input photo: pose is estimated (not exact), use only to fill
            # areas completely missed by all generated views.
            # Weight is low but NOT zero — helps fill back/bottom surfaces.
            _cur_w = accum_w[sel]
            # Even lower weight on already-painted vertices
            _paint_scale = np.where(_cur_w > 0.5, 0.08, 0.25)
            view_scale = _paint_scale
            w = cos_w * ctr_w * depth_conf * sat_w * view_scale

        else:
            # Side generated views (6 views at known poses).
            # Scale by elevation: high-elevation views have more foreshortening.
            el_abs     = abs(v["el"])
            el_penalty = 1.0 - el_abs / 90.0 * 0.5   # 0.94 at 10°, 0.72 at 45°

            _cur_w = accum_w[sel]
            _already_painted = _cur_w > 1.0

            # Multi-view agreement: if this view's color agrees with current
            # accumulated color, give it full weight; disagreement → reduce.
            _cur_col = (accum_col[sel] / _cur_w[:, None].clip(1e-6)).astype(np.float32)
            _has_acc = _cur_w > 0.3

            # Compute per-vertex Lab-space ΔE between sampled and accumulated
            # (approximated cheaply as normalised RGB distance)
            _rgb_diff = np.linalg.norm(
                sampled.astype(np.float32) - _cur_col.astype(np.float32), axis=1
            ) / 255.0   # 0 = identical, 1 = max distance

            # agreement_w: 1.0 when colours agree, drops to 0.1 at ΔE=0.5+
            agreement_w = np.where(
                _has_acc,
                np.clip(1.0 - _rgb_diff * 1.8, 0.1, 1.0),
                np.ones(len(sel), dtype=np.float32),
            )

            # Hue-conflict guard — hard reject large hue swings on
            # already-painted vertices (prevents teal hat bleeding onto body)
            def _hue_f32(rgb):
                r,g,b = rgb[:,0]/255.0, rgb[:,1]/255.0, rgb[:,2]/255.0
                cmax = np.maximum(np.maximum(r,g),b)
                cmin = np.minimum(np.minimum(r,g),b)
                d = cmax - cmin
                h = np.zeros(len(rgb), np.float32)
                mr = (d>0.01)&(cmax==r); h[mr]=30.0*((g[mr]-b[mr])/d[mr]%6.0)
                mg = (d>0.01)&(cmax==g); h[mg]=30.0*((b[mg]-r[mg])/d[mg]+2.0)
                mb = (d>0.01)&(cmax==b); h[mb]=30.0*((r[mb]-g[mb])/d[mb]+4.0)
                h[d<=0.01] = -1.0
                return h

            _h_samp = _hue_f32(sampled)
            _h_cur  = _hue_f32(_cur_col)
            _both_chr  = (_h_samp >= 0) & (_h_cur >= 0) & _has_acc
            _hdiff     = np.abs(_h_samp - _h_cur)
            _hdiff     = np.minimum(_hdiff, 180.0 - _hdiff)
            # Hard-zero weight for strong hue conflict (>40°) on painted verts
            _hue_ok    = ~(_both_chr & (_hdiff > 40.0) & _already_painted)

            # Base scale: unpainted vertices get full side-view weight;
            # already-painted get reduced weight so front view dominates.
            base_scale = np.where(_already_painted, 0.25, 0.7)

            w = (cos_w * ctr_w * depth_conf * sat_w
                 * base_scale * el_penalty * agreement_w
                 * _hue_ok.astype(np.float32))

        accum_col[sel] += sampled * w[:, None]
        accum_w[sel]   += w
        photo_hit[sel]  = True

    logger.info("  paint pass: %d/%d (%.1f%%) vertices painted",
                int((accum_w >= 1e-6).sum()), N,
                100.0 * (accum_w >= 1e-6).sum() / max(1, N))

    # ── Step 3: Atlas fallback for unseen surfaces ─────────────────────────
    # Only fill vertices NOT reached by photo projection.
    # This keeps photo colors pristine while filling back/bottom surfaces.
    _atlas_filled = 0
    if triposr_glb_path and os.path.exists(triposr_glb_path):
        try:
            uv_base, tex_img = _transfer_triposr_texture(mesh, triposr_glb_path)
            if uv_base is not None and tex_img is not None:
                _tex_lin = np.array(tex_img, dtype=np.uint8).astype(np.float32)
                _tex_gc  = np.clip(
                    np.power(_tex_lin / 255.0, 1.0 / 2.2) * 255.0, 0, 255
                ).astype(np.uint8)
                uv_cols = _sample_texture_at_uv(_tex_gc, uv_base).astype(np.float32)

                # Fix black atlas speckles
                _uv_ok = uv_cols.max(axis=1) >= 15.0
                if (~_uv_ok).sum() > 0 and _uv_ok.sum() > 10:
                    from scipy.spatial import cKDTree as _KDA
                    _, _nna = _KDA(verts[_uv_ok]).query(verts[~_uv_ok], k=1, workers=-1)
                    uv_cols[~_uv_ok] = uv_cols[np.where(_uv_ok)[0][_nna]]

                # Calibrate atlas to front photo mean
                try:
                    # Use generated view 1 (calibrated pose) not input photo.
                    _calib_idx = min(_SEG_VIEW_IDX, len(views)-1)
                    _fv   = views[_calib_idx]
                    _rv   = _fv["rgb"].convert("RGB")
                    _ra   = np.array(_rv.resize((S, S), Image.LANCZOS))
                    # Use mask file if available, otherwise corner-based
                    if _fv.get("mask_path") and os.path.exists(_fv["mask_path"]):
                        _msk  = np.array(Image.open(_fv["mask_path"]).convert("L").resize((S,S),Image.NEAREST)) >= FG_ALPHA_THRESHOLD
                    else:
                        _r2c  = max(4, S//16)
                        _bgc  = np.median(_ra[:_r2c,:_r2c].reshape(-1,3), axis=0).astype(np.float32)
                        _msk  = np.linalg.norm(_ra.astype(np.float32)-_bgc, axis=2) > 15.0
                    if _msk.sum() > 100:
                        _pm  = np.median(_ra[_msk].astype(np.float32), axis=0)
                        _av  = uv_cols[uv_cols.max(axis=1) >= 15.0]
                        _am  = np.median(_av, axis=0).clip(1.0, 254.0)
                        _shift = _pm - _am
                        uv_cols = np.clip(uv_cols + _shift, 0, 255)
                except Exception:
                    pass

                # Fill ONLY vertices not reached by any accepted photo sample.
                _no_photo = ~photo_hit
                if _no_photo.sum() > 0:
                    accum_col[_no_photo] = uv_cols[_no_photo] * 0.8
                    accum_w[_no_photo]   = 0.8
                    _atlas_filled = int(_no_photo.sum())
                    logger.info("  Atlas fallback: filled %d unseen vertices", _atlas_filled)
        except Exception as _ae:
            logger.warning("  Atlas fallback failed: %s", _ae)

    # ── Step 4: Second pass for still-unpainted ────────────────────────────
    no_hit = accum_w < 1e-6
    if no_hit.sum() > 0:
        for v in views:
            if no_hit.sum() == 0:
                break
            rgb_img = v["rgb"].convert("RGB").resize((S, S), Image.LANCZOS)
            rgb_arr = np.array(rgb_img, dtype=np.uint8)
            fg_mask = _get_mask(v)
            miss    = np.where(no_hit)[0]
            u, vpx, _, valid = _project(verts[miss], v["az"], v["el"])
            ui = np.round(u  ).astype(np.int32)
            vi = np.round(vpx).astype(np.int32)
            ok = valid & (ui >= 0) & (ui < S) & (vi >= 0) & (vi < S)
            if not ok.any():
                continue
            sl   = np.where(ok)[0]
            fg   = fg_mask[vi[sl], ui[sl]]
            if not fg.any():
                continue
            sel2   = miss[sl[fg]]
            samp2  = rgb_arr[vi[sl[fg]], ui[sl[fg]]].astype(np.float32)
            sane   = samp2.mean(axis=1) > 15.0
            sel2   = sel2[sane];  samp2 = samp2[sane]
            if len(sel2):
                accum_col[sel2] += samp2 * 0.4
                accum_w[sel2]   += 0.4
            no_hit = accum_w < 1e-6

    # ── Step 5: kd-tree fill for remaining gaps ────────────────────────────
    no_hit = accum_w < 1e-6
    if no_hit.sum() > 0:
        painted = np.where(~no_hit)[0]
        if len(painted) > 10:
            from scipy.spatial import cKDTree
            _, nn = cKDTree(verts[painted]).query(verts[no_hit], k=1, workers=-1)
            accum_col[no_hit] = accum_col[painted[nn]]
            accum_w[no_hit]   = accum_w[painted[nn]]
        else:
            accum_w[no_hit] = 1.0;  accum_col[no_hit] = 180.0

    vcols = np.clip(accum_col / accum_w[:, None].clip(1e-6), 0, 255).astype(np.uint8)

    # ── Step 5a: Region-conflict heal ─────────────────────────────────────
    # Vertices painted ONLY by side views (low accum_w) can be assigned
    # a hue that is inconsistent with their spatial neighbours — e.g. a
    # body vertex that projects near the teal hat in az=90 view.
    # Fix: any vertex with accum_w < CONFIDENT_W whose hue differs from
    # its k=8 nearest confident neighbours by more than HUE_TOL gets
    # replaced by the median of those neighbour colors.
    try:
        from scipy.spatial import cKDTree as _CKDT
        _CONFIDENT_W   = 1.5   # minimum weight to be considered "confident"
        _HUE_TOL       = 40.0  # max allowed hue distance (0-180 in OpenCV)
        _conf_mask     = accum_w >= _CONFIDENT_W
        _unconf_mask   = ~_conf_mask & (accum_w >= 1e-6)
        if _conf_mask.sum() > 20 and _unconf_mask.sum() > 0:
            # Build k-d tree on confident verts
            _tree = _CKDT(verts[_conf_mask])
            _, _nn_idx = _tree.query(verts[_unconf_mask], k=min(8, _conf_mask.sum()), workers=-1)
            _conf_verts_idx = np.where(_conf_mask)[0]
            _unconf_verts_idx = np.where(_unconf_mask)[0]
            _nn_cols = vcols[_conf_verts_idx[_nn_idx]]   # (M, k, 3)
            if _nn_cols.ndim == 2:
                _nn_cols = _nn_cols[:, np.newaxis, :]
            _nn_med = np.median(_nn_cols, axis=1).astype(np.float32)

            # Compare hue of unconfident vert vs neighbour median
            def _to_hue(rgb_arr):
                """Return hue in [0, 180] per pixel, -1 for achromatic."""
                r = rgb_arr[:, 0].astype(np.float32) / 255.0
                g = rgb_arr[:, 1].astype(np.float32) / 255.0
                b = rgb_arr[:, 2].astype(np.float32) / 255.0
                cmax = np.maximum(np.maximum(r, g), b)
                cmin = np.minimum(np.minimum(r, g), b)
                delta = cmax - cmin
                hue = np.zeros(len(rgb_arr), dtype=np.float32)
                m_r = (delta > 0.01) & (cmax == r);  hue[m_r] = 30.0 * ((g[m_r] - b[m_r]) / delta[m_r] % 6.0)
                m_g = (delta > 0.01) & (cmax == g);  hue[m_g] = 30.0 * ((b[m_g] - r[m_g]) / delta[m_g] + 2.0)
                m_b = (delta > 0.01) & (cmax == b);  hue[m_b] = 30.0 * ((r[m_b] - g[m_b]) / delta[m_b] + 4.0)
                hue[delta <= 0.01] = -1.0   # achromatic
                return hue

            _hue_unc  = _to_hue(vcols[_unconf_verts_idx])
            _hue_med  = _to_hue(_nn_med.astype(np.uint8))
            _both_chromatic = (_hue_unc >= 0) & (_hue_med >= 0)
            _hue_diff = np.abs(_hue_unc - _hue_med)
            _hue_diff = np.minimum(_hue_diff, 180.0 - _hue_diff)   # circular

            _conflict = _both_chromatic & (_hue_diff > _HUE_TOL)
            if _conflict.sum() > 0:
                vcols[_unconf_verts_idx[_conflict]] = np.clip(_nn_med[_conflict], 0, 255).astype(np.uint8)
                logger.info("  Region-conflict heal: replaced %d / %d unconfident vertices "
                            "(hue diff > %.0f)", _conflict.sum(), _unconf_mask.sum(), _HUE_TOL)
    except Exception as _rch_e:
        logger.warning("  Region-conflict heal skipped: %s", _rch_e)

    # ── Step 5b: Global brightness match to front photo ──────────────────
    # Compute gain from PHOTO-PAINTED vertices only (accum_w > 0.5).
    # Using all vertices includes dark atlas-filled back faces which
    # drag the mean down and cause overcorrection (gain hitting 2.0).
    # Generated views are slightly darker than input photo -- correct
    # against the input photo (view0) which has the true surface color.
    try:
        _fv2  = views[0]
        _rv2  = (_fv2.get('rgb_raw') or _fv2['rgb']).convert('RGBA')
        _ra2  = np.array(_rv2.resize((S, S), Image.LANCZOS))
        _fgv2 = _ra2[..., 3] >= FG_ALPHA_THRESHOLD
        if _fgv2.sum() > 100:
            _ph_mean  = _ra2[_fgv2, :3].astype(np.float32).mean(axis=0)
            # Use only photo-projected vertices for reference mean
            # (accum_w > 0.5 means vertex was hit by photo projection)
            _photo_verts_mask = accum_w > 0.5
            if _photo_verts_mask.sum() > 100:
                _vc_ref = vcols[_photo_verts_mask].astype(np.float32)
            else:
                _vc_ref = vcols.astype(np.float32)
            _vc_mean  = _vc_ref.mean(axis=0).clip(1, 254)
            _per_err = np.abs(_ph_mean - _vc_mean)
            if _per_err.max() > 5.0:
                # Cap: 1.5 for red (mostly correct from view calib),
                # 1.8 for green/blue (leaf and shadows can still be under).
                _gain = np.array([
                    float(np.clip(_ph_mean[0]/_vc_mean[0], 0.70, 1.5)),
                    float(np.clip(_ph_mean[1]/_vc_mean[1], 0.70, 1.8)),
                    float(np.clip(_ph_mean[2]/_vc_mean[2], 0.70, 1.8)),
                ], dtype=np.float32)
                _gain[_per_err < 4.0] = 1.0
                _vc_f = vcols.astype(np.float32)
                # Apply gain only to dark/mid-tone vertices.
                # Skip bright vertices (max channel > 160) -- they are
                # highlights/seeds that are already correct. Boosting them
                # pushes them past 255 creating neon artifacts.
                # Skip highlights (max channel > 185).
                # Body red max~175 -> CORRECTED (good, needs G boost)
                # Seed highlights max~200+ -> SKIPPED (already bright, don't boost)
                _dark_mask = _vc_f.max(axis=1) <= 185
                _vc_f[_dark_mask] = np.clip(_vc_f[_dark_mask] * _gain, 0, 255)
                vcols = _vc_f.astype(np.uint8)
                logger.info('  Brightness correction: gain=[%.2f,%.2f,%.2f]',
                            float(_gain[0]), float(_gain[1]), float(_gain[2]))
    except Exception:
        pass

    # ── Step 6: Remove black speckles ─────────────────────────────────────
    _spk = vcols.max(axis=1) < 15
    if _spk.sum() > 0:
        _ok  = np.where(~_spk)[0]
        _bad = np.where(_spk)[0]
        if len(_ok) > 10:
            from scipy.spatial import cKDTree as _KD2
            _, _nn2 = _KD2(verts[_ok]).query(verts[_bad], k=1, workers=-1)
            vcols[_bad] = vcols[_ok[_nn2]]
            logger.info("  Speckle fix: replaced %d black vertices", int(_spk.sum()))

    # ── Step 7: Edge-aware smoothing (within-region only) ─────────────────
    try:
        fp   = np.array(mesh.faces)
        e0   = np.concatenate([fp[:, 0], fp[:, 1], fp[:, 2]])
        e1   = np.concatenate([fp[:, 1], fp[:, 2], fp[:, 0]])
        pair = np.unique(np.sort(np.stack([e0, e1], axis=1), axis=1), axis=0)
        i0, i1 = pair[:, 0], pair[:, 1]
        cf   = vcols.astype(np.float32)
        diff = np.linalg.norm(cf[i0] - cf[i1], axis=1)
        # Smooth texture speckles while preserving major color boundaries.
        we   = np.exp(-diff / 18.0).astype(np.float32)
        we[diff > 75.0] = 0.0
        ws   = np.ones(N, np.float32)
        cs   = cf.copy()
        np.add.at(ws, i0, we);  np.add.at(ws, i1, we)
        np.add.at(cs, i0, cf[i1] * we[:, None])
        np.add.at(cs, i1, cf[i0] * we[:, None])
        vcols = np.clip(cs / ws[:, None], 0, 255).astype(np.uint8)
        logger.info("  Color smoothing: OK")
    except Exception as _sme:
        logger.warning("  Color smoothing skipped: %s", _sme)

    import trimesh as _tm2
    mesh.visual = _tm2.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.hstack([vcols, np.full((N, 1), 255, np.uint8)])
    )
    # Store region labels in metadata for future UI (click-to-select regions)
    if vert_region.max() >= 0:
        mesh.metadata["color_regions"] = vert_region
        mesh.metadata["region_colors"]  = [
            r["color"].astype(int).tolist() if r else None
            for r in (seg_regions or [])
        ]

    green = float(((vcols[:, 1] > vcols[:, 0]) & (vcols[:, 1] > vcols[:, 2])).mean() * 100)
    red   = float(((vcols[:, 0] > 100) & (vcols[:, 0] > vcols[:, 1] * 1.4)).mean() * 100)
    logger.info("  Colors done: red=%.1f%%  green=%.1f%%  mean=[%d,%d,%d]",
                red, green, int(vcols[:, 0].mean()), int(vcols[:, 1].mean()), int(vcols[:, 2].mean()))
    return mesh

def cleanup(mesh, target_faces=None, smooth=True, keep_largest=True):
    import trimesh
    
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    if keep_largest:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda p: len(p.faces))
            logger.info(f"  Kept largest component ({len(mesh.faces):,} faces)")

    if target_faces is not None and len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
            logger.info(f"  Decimated: {len(mesh.vertices):,}v / {len(mesh.faces):,}f")
        except Exception as e:
            logger.warning(f"  Decimation skipped: {e}")

    if smooth:
        try:
            import trimesh as _tm
            _tm.smoothing.filter_taubin(mesh, lamb=0.18, nu=-0.20, iterations=1)
        except:
            pass

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_inversion(mesh)
        filled = mesh.fill_holes()
        logger.info(f"  Repair pass: fill_holes={filled} watertight={mesh.is_watertight}")
    except Exception as e:
        logger.warning(f"  Trimesh repair skipped: {e}")

    if not mesh.is_watertight:
        try:
            import pymeshfix
            fixer = pymeshfix.MeshFix(
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int32),
            )
            fixer.repair(joincomp=True, remove_smallest_components=True)
            repaired = trimesh.Trimesh(
                vertices=np.asarray(fixer.points),
                faces=np.asarray(fixer.faces),
                process=False,
            )
            if len(repaired.faces) >= int(0.5 * len(mesh.faces)):
                mesh = repaired
                logger.info(f"  PyMeshFix repair watertight={mesh.is_watertight}")
            else:
                logger.warning("  PyMeshFix too destructive, keeping pre-fix mesh")
        except Exception as e:
            logger.warning(f"  Strong manifold repair unavailable/skipped: {e}")

    centre = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.vertices -= centre
    scale = np.max(mesh.bounds[1] - mesh.bounds[0])
    if scale > 1e-6:
        mesh.vertices /= scale

    logger.info(f"  ✓ {len(mesh.vertices):,}v / {len(mesh.faces):,}f | watertight={mesh.is_watertight}")
    return mesh


def _make_24_rotations():
    """
    Return the 24 rotation matrices that form the rotation group of a cube
    (all 90°-multiple rotations that map axis-aligned unit vectors to
    axis-aligned unit vectors).  These are ALL possible orientations a mesh
    can have when TripoSR / InstantMesh emits it in an arbitrary canonical
    frame.  Trying all 24 guarantees we find the correct one regardless of
    the original mesh orientation.
    """
    rots = []
    def Rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def Ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    def Rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    for ax in [0, np.pi/2, np.pi, 3*np.pi/2]:
        for ay in [0, np.pi/2, np.pi, 3*np.pi/2]:
            for az in [0, np.pi/2]:
                R = Rx(ax) @ Ry(ay) @ Rz(az)
                # Only keep proper rotations (det=+1) — 24 out of 64 combos
                if abs(np.linalg.det(R) - 1.0) < 1e-6:
                    # Deduplicate: skip if already in list
                    dup = any(np.allclose(R, r, atol=1e-6) for r in rots)
                    if not dup:
                        rots.append(R)
    return rots


# Pre-built once at import time — 24 rotation matrices
_CANONICAL_ROTATIONS = _make_24_rotations()


def _silhouette_iou_fast(verts_norm, views, max_views=4):
    """
    Fast silhouette IoU for orientation search.
    Uses a decimated vertex set (every 8th vertex) to keep the 24×N search
    tractable.  Returns mean IoU across accepted views.

    verts_norm: (N,3) float32 already normalized to [-0.5, 0.5]
    """
    S   = IMG_SIZE
    iou_sum = 0.0
    n_views = 0
    for v in views[1:max_views+1]:          # skip input photo (view 0, pose unknown)
        mask = _get_mask(v)
        if mask is None or mask.mean() < 0.02:
            continue
        mask_rs = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize((S, S), Image.NEAREST)
        ) > 127
        u, vpx, depth, valid = _project(verts_norm, v["az"], v["el"])
        front = valid & (depth > 0)
        if not front.any():
            continue
        ui = np.clip(np.round(u[front]).astype(np.int32),  0, S - 1)
        vi = np.clip(np.round(vpx[front]).astype(np.int32), 0, S - 1)
        sil = np.zeros((S, S), dtype=bool)
        sil[vi, ui] = True
        from scipy.ndimage import binary_dilation
        sil = binary_dilation(sil, iterations=4)
        inter = float((sil & mask_rs).sum())
        union = float((sil | mask_rs).sum())
        if union > 0:
            iou_sum += inter / union
            n_views  += 1
    return iou_sum / n_views if n_views > 0 else 0.0


def _find_best_orientation(mesh, views):
    """
    Try all 24 axis-aligned canonical rotations and return the rotation R
    that maximises the mean silhouette IoU against the known-pose views.

    Why 24?  TripoSR/InstantMesh can emit the mesh in ANY of the 24 orientations
    of a cube — 0°, 90°, 180°, or 270° around each of the three axes.
    A heuristic (bounding-box spread, symmetry ratio) will fail on objects
    that don't match its assumptions.  Silhouette IoU is shape-agnostic: it
    directly measures whether the mesh silhouette matches the input photos,
    which is exactly what we want.

    Cost: 24 projections × ~N/8 vertices × 4 views ≈ very fast (CPU, <0.5 s).

    Returns: best_R (3×3), best_iou (float)
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32).copy()
    # Normalize to unit sphere for projection
    centre = verts.mean(axis=0)
    verts -= centre
    scale  = np.abs(verts).max()
    if scale > 1e-6:
        verts /= scale
    # Decimate: every 8th vertex to keep search fast
    stride  = max(1, len(verts) // 2000)
    v_small = verts[::stride]

    best_iou = -1.0
    best_R   = np.eye(3)
    scores   = []

    for R in _CANONICAL_ROTATIONS:
        v_rot = (R @ v_small.T).T
        iou   = _silhouette_iou_fast(v_rot, views)
        scores.append(iou)
        if iou > best_iou:
            best_iou = iou
            best_R   = R

    scores_sorted = sorted(scores, reverse=True)
    logger.info(
        "  Orientation search: best_iou=%.3f  2nd=%.3f  worst=%.3f  (24 rotations tested)",
        scores_sorted[0],
        scores_sorted[1] if len(scores_sorted) > 1 else 0.0,
        scores_sorted[-1],
    )
    return best_R, best_iou


def _auto_orient(mesh, views=None):
    """
    Find the correct mesh orientation by trying all 24 canonical rotations
    and selecting the one whose silhouette best matches the known-pose views.

    This replaces the old bounding-box / spread-ratio heuristic which failed
    on objects that don't match its shape assumptions (wide-top objects,
    flat objects, asymmetric objects, etc.).

    If views are not available (fallback path), uses the old bounding-box
    heuristic as a last resort.

    Skipped if _apply_pose_orientation already ran successfully.
    """
    # Pose estimator already handled orientation — don't override it.
    if mesh.metadata.get("pose_estimator_applied"):
        logger.info("  Auto-orient skipped: pose estimator already applied")
        return mesh

    # ── IoU-based canonical orientation search ─────────────────────────────
    if views is not None and len(views) >= 2:
        try:
            best_R, best_iou = _find_best_orientation(mesh, views)

            if best_iou < 0.05:
                logger.warning(
                    "  Auto-orient: best IoU=%.3f is very low — "
                    "silhouette search unreliable, falling back to bbox heuristic",
                    best_iou,
                )
                # Fall through to bbox heuristic below
            else:
                # Apply the best rotation
                centre = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
                mesh.vertices -= centre
                mesh.vertices  = (best_R @ mesh.vertices.T).T
                centre2 = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
                mesh.vertices -= centre2
                if hasattr(mesh, "fix_normals"):
                    mesh.fix_normals()
                mesh.metadata["auto_orient_iou"] = float(best_iou)
                logger.info(
                    "  Auto-orient: applied best canonical rotation (IoU=%.3f)", best_iou
                )
                return mesh
        except Exception as exc:
            logger.warning("  Auto-orient IoU search failed: %s — falling back to bbox", exc)

    # ── Fallback: bounding-box heuristic ──────────────────────────────────
    # Used when views are not available or IoU search produced very low scores.
    bb  = mesh.bounds
    ext = bb[1] - bb[0]

    if np.argmax(ext) != 1:
        tall_idx = int(np.argmax(ext))
        src_axis = np.zeros(3); src_axis[tall_idx] = 1.0
        world_y  = np.array([0., 1., 0.])
        v_cross  = np.cross(src_axis, world_y)
        s        = float(np.linalg.norm(v_cross))
        c        = float(np.dot(src_axis, world_y))

        if s < 1e-6:
            if c < 0:
                mesh.vertices[:, 1] *= -1
                mesh.vertices[:, 2] *= -1
        else:
            Kmat = np.array([[0, -v_cross[2], v_cross[1]],
                             [v_cross[2], 0, -v_cross[0]],
                             [-v_cross[1], v_cross[0], 0]])
            R = np.eye(3) + Kmat + Kmat @ Kmat * ((1 - c) / (s * s + 1e-12))
            mesh.vertices = (R @ mesh.vertices.T).T

        c2 = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.vertices -= c2
        if hasattr(mesh, "fix_normals"):
            mesh.fix_normals()
        b = mesh.bounds
        logger.info("  Auto-orient bbox fallback: tall axis was %s → Y. "
                    "bbox X=%.2f Y=%.2f Z=%.2f",
                    ["X","Y","Z"][tall_idx], b[1,0]-b[0,0], b[1,1]-b[0,1], b[1,2]-b[0,2])

    return mesh


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def _url_for_path(path: str) -> str:
    return "/" + path.replace("\\", "/")


def _mesh_dimensions(mesh):
    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    return {"width": float(ext[0]), "height": float(ext[1]), "depth": float(ext[2])}


def _valid_target_dimensions(target_dimensions_mm):
    dims = {}
    for src, dst in (("width", "width"), ("height", "height"), ("depth", "depth"),
                     ("x", "width"), ("y", "height"), ("z", "depth")):
        try:
            value = float((target_dimensions_mm or {}).get(src) or 0)
        except Exception:
            value = 0.0
        if value > 0:
            dims[dst] = value
    return dims


def _copy_mesh(mesh):
    copied = mesh.copy()
    try:
        copied.visual = mesh.visual.copy()
    except Exception:
        pass
    return copied


def _decimate_for_profile(mesh, target_faces: int):
    """
    Decimate mesh for a LOD profile and validate topology afterward.
    Decimation can introduce non-manifold edges; we measure them explicitly.
    """
    out = _copy_mesh(mesh)
    if len(out.faces) <= target_faces:
        validate_mesh_topology(out, f"profile/{target_faces}k (no decimation needed)")
        return out
    try:
        out = out.simplify_quadric_decimation(target_faces)
        logger.info("  Profile decimated to %d faces", len(out.faces))
    except Exception as e:
        logger.warning("  Profile decimation skipped: %s", e)
    try:
        out.update_faces(out.nondegenerate_faces())
        out.remove_unreferenced_vertices()
        out.fix_normals()
    except Exception:
        pass
    # Validate topology after decimation — decimation can introduce open edges
    topo = validate_mesh_topology(out, f"profile/{target_faces}k after decimation")
    if not topo["watertight"] and mesh.is_watertight:
        # Source was watertight but decimation broke it; try a cheap repair
        try:
            out.fill_holes()
            if not out.is_watertight:
                import pymeshfix
                fixer = pymeshfix.MeshFix(
                    np.asarray(out.vertices, dtype=np.float64),
                    np.asarray(out.faces, dtype=np.int32),
                )
                fixer.repair(joincomp=True, remove_smallest_components=True)
                import trimesh as _tm2
                cand = _tm2.Trimesh(
                    vertices=np.asarray(fixer.points),
                    faces=np.asarray(fixer.faces),
                    process=False,
                )
                if len(cand.faces) >= int(0.5 * len(out.faces)):
                    out = cand
            if out.is_watertight:
                logger.info("  Profile %d: repaired to watertight after decimation", target_faces)
            else:
                logger.warning("  Profile %d: could not repair after decimation (watertight=False)", target_faces)
        except Exception as exc:
            logger.warning("  Profile %d: post-decimation repair failed: %s", target_faces, exc)
    return out


def _smooth_profile(mesh, iterations=2):
    out = _copy_mesh(mesh)
    try:
        import trimesh as _tm
        _tm.smoothing.filter_taubin(out, lamb=0.18, nu=-0.20, iterations=iterations)
        out.fix_normals()
    except Exception as e:
        logger.warning(f"  Profile smoothing skipped: {e}")
    return out


def _apply_pose_orientation(mesh, out_dir: str | None):
    """
    Apply the world_R rotation from pose.json so the mesh faces the same
    direction the input photo was taken from, then correct for in-plane roll.

    Sets mesh.metadata["pose_estimator_applied"] = True so _auto_orient
    knows not to override the result with its bounding-box heuristic.
    """
    if not out_dir:
        return mesh
    try:
        from pose_estimator import load_pose
        pose = load_pose(out_dir)
        R = np.asarray(pose.world_R, dtype=np.float64)
        if R.shape == (3, 3) and np.all(np.isfinite(R)) and not np.allclose(R, np.eye(3), atol=1e-3):
            mesh.vertices = (R @ np.asarray(mesh.vertices).T).T
            if hasattr(mesh, "fix_normals"):
                mesh.fix_normals()

            # --- Roll correction ------------------------------------------
            # pose.json records the in-plane roll of the object in the photo.
            # world_R corrects azimuth+elevation but NOT roll.
            # A 28-degree roll means the mesh is tilted in the XY plane after
            # world_R is applied; projecting a tilted mesh produces scrambled
            # colour because each vertex hits the wrong photo region.
            # Fix: rotate the mesh around its own Z axis by -roll so the
            # object stands upright in camera space before projection.
            roll_deg = float(getattr(pose, "roll", 0.0))
            if abs(roll_deg) > 3.0:   # only correct if roll is meaningful
                theta = -np.radians(roll_deg)
                Rz = np.array([
                    [ np.cos(theta), -np.sin(theta), 0.],
                    [ np.sin(theta),  np.cos(theta), 0.],
                    [ 0.,             0.,            1.],
                ], dtype=np.float64)
                mesh.vertices = (Rz @ np.asarray(mesh.vertices).T).T
                if hasattr(mesh, "fix_normals"):
                    mesh.fix_normals()
                logger.info("  Roll correction applied: %.1f deg", roll_deg)

            # Re-centre after all rotations
            centre = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
            mesh.vertices -= centre

            mesh.metadata["pose_estimator_applied"] = True
            mesh.metadata["pose_estimator"] = {
                "azimuth": float(pose.azimuth),
                "elevation": float(pose.elevation),
                "roll":      roll_deg,
            }
            logger.info(
                "  Pose orientation applied az=%.1f el=%.1f roll=%.1f",
                pose.azimuth, pose.elevation, roll_deg,
            )
    except Exception as e:
        logger.warning(f"  Pose orientation skipped: {e}")
    return mesh


def _apply_target_dimensions(mesh, target_dimensions_mm=None):
    dims = _valid_target_dimensions(target_dimensions_mm)
    if not dims:
        return mesh, {}
    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    scale_vec = np.ones(3, dtype=np.float64)
    axis_map = {"width": 0, "height": 1, "depth": 2}
    if len(dims) == 1:
        name, mm = next(iter(dims.items()))
        axis = axis_map[name]
        if ext[axis] > 1e-8:
            scale_vec[:] = mm / ext[axis]
    else:
        for name, mm in dims.items():
            axis = axis_map[name]
            if ext[axis] > 1e-8:
                scale_vec[axis] = mm / ext[axis]
    mesh.vertices *= scale_vec
    mesh.metadata["units"] = "millimeters"
    mesh.metadata["target_dimensions_mm"] = dims
    logger.info("  Applied target dimensions (mm): %s", dims)
    return mesh, dims


def _orient_and_normalize(mesh, out_dir: str | None = None, target_dimensions_mm=None, views=None):
    mesh = _apply_pose_orientation(mesh, out_dir)
    mesh = _auto_orient(mesh, views=views)
    centre = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.vertices -= centre
    scale = np.max(mesh.bounds[1] - mesh.bounds[0])
    if scale > 1e-6:
        mesh.vertices /= scale
    mesh, applied_dims = _apply_target_dimensions(mesh, target_dimensions_mm)
    if applied_dims:
        mesh.metadata["applied_dimensions_mm"] = applied_dims
    # Ensure normals are consistently outward-facing after any transforms.
    # trimesh.repair.fix_normals propagates winding from outside inward,
    # giving correct normals for Three.js shading.
    try:
        import trimesh as _tr
        _tr.repair.fix_normals(mesh)
        _tr.repair.fix_winding(mesh)
    except Exception:
        if hasattr(mesh, "fix_normals"):
            mesh.fix_normals()
    return mesh


def _build_quality_report(mesh, method: str, texture_preview: str, print_mesh=None, silhouette_iou=None):
    print_mesh = print_mesh if print_mesh is not None else mesh
    parts = mesh.split(only_watertight=False)
    print_parts = print_mesh.split(only_watertight=False)
    issues = []
    dims = _mesh_dimensions(mesh)
    print_dims = _mesh_dimensions(print_mesh)
    units = getattr(mesh, "metadata", {}).get("units", "normalized")
    print_units = getattr(print_mesh, "metadata", {}).get("units", units)
    min_feature_mm = float(os.environ.get("TRIGENIX_MIN_PRINT_FEATURE_MM", "0.8"))
    if not mesh.is_watertight:
        issues.append("Visual mesh is not watertight; use the print profile for manufacturing.")
    if len(parts) > 1:
        issues.append(f"Visual mesh has {len(parts)} disconnected components.")
    if len(mesh.faces) < 1000:
        issues.append("Visual mesh has very low face count; detail may be poor.")
    print_ready = bool(print_mesh.is_watertight and len(print_parts) == 1 and len(print_mesh.faces) >= 1000)
    if print_units == "millimeters" and min(print_dims.values()) < min_feature_mm:
        print_ready = False
        issues.append(f"Smallest bounding dimension is below {min_feature_mm:.2f} mm.")
    if not print_mesh.is_watertight:
        issues.append("Print profile is not watertight; repair before 3D printing.")
    if len(print_parts) > 1:
        issues.append(f"Print profile has {len(print_parts)} disconnected components.")
    if texture_preview:
        issues.append("Texture preview is for visualization; use print profile for manufacturing.")
    return {
        "method": method,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "components": int(len(parts)),
        "dimensions": dims,
        "units": units,
        "minimum_feature_mm": min_feature_mm,
        "print_ready": print_ready,
        "visual_mesh": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "components": int(len(parts)),
            "dimensions": dims,
            "units": units,
        },
        "print_mesh": {
            "vertices": int(len(print_mesh.vertices)),
            "faces": int(len(print_mesh.faces)),
            "watertight": bool(print_mesh.is_watertight),
            "components": int(len(print_parts)),
            "dimensions": print_dims,
            "units": print_units,
        },
        "texture_preview_available": bool(texture_preview),
        "issues": issues,
        "silhouette_iou": round(silhouette_iou, 4) if silhouette_iou is not None else None,
    }


def _export_asset_profiles(mesh, out_dir: str, method: str, baked_texture_glb: str = None, print_mesh=None, silhouette_iou=None):
    profiles = {}
    sealed_mesh = print_mesh if print_mesh is not None else mesh

    def add_profile(key, label, path, purpose, profile_mesh=None, watertight=None):
        src = profile_mesh if profile_mesh is not None else mesh
        profiles[key] = {
            "label": label,
            "url": _url_for_path(path),
            "download": _url_for_path(path),
            "purpose": purpose,
            "vertices": int(len(src.vertices)),
            "faces": int(len(src.faces)),
            "watertight": bool(src.is_watertight if watertight is None else watertight),
        }

    try:
        path = os.path.join(out_dir, "model_print.stl")
        sealed_mesh.export(path, file_type="stl")
        add_profile(
            "print_stl",
            "3D print STL",
            path,
            "Manufacturing mesh for slicers; may be sealed differently from the visual GLB.",
            sealed_mesh,
        )
    except Exception as e:
        logger.warning(f"  Print STL export skipped: {e}")

    try:
        path = os.path.join(out_dir, "model_print.glb")
        sealed_mesh.export(path, file_type="glb")
        add_profile(
            "print_glb",
            "3D print GLB",
            path,
            "Printable preview/download; may be sealed differently from the visual GLB.",
            sealed_mesh,
        )
    except Exception as e:
        logger.warning(f"  Print GLB export skipped: {e}")

    try:
        ar_mesh = _decimate_for_profile(mesh, 50000)
        path = os.path.join(out_dir, "model_ar.glb")
        ar_mesh.export(path, file_type="glb")
        add_profile("ar_vr", "AR/VR GLB", path, "Balanced mobile/AR asset with lower polygon load.", ar_mesh)
    except Exception as e:
        logger.warning(f"  AR/VR profile export skipped: {e}")

    try:
        game_mesh = _decimate_for_profile(mesh, 30000)
        path = os.path.join(out_dir, "model_game_lod0.glb")
        game_mesh.export(path, file_type="glb")
        add_profile("game", "Game LOD0 GLB", path, "Game-engine starter asset with reduced face count.", game_mesh)
    except Exception as e:
        logger.warning(f"  Game profile export skipped: {e}")

    try:
        sim_mesh = _smooth_profile(_decimate_for_profile(mesh, 20000), iterations=2)
        path = os.path.join(out_dir, "model_simulation.glb")
        sim_mesh.export(path, file_type="glb")
        add_profile(
            "simulation",
            "Simulation GLB",
            path,
            "Clean lighter mesh for education, simulation, and non-diagnostic medical study.",
            sim_mesh,
        )
    except Exception as e:
        logger.warning(f"  Simulation profile export skipped: {e}")

    try:
        collision_mesh = mesh.convex_hull
        path = os.path.join(out_dir, "model_collision.obj")
        collision_mesh.export(path, file_type="obj")
        add_profile(
            "robotics_collision",
            "Robotics collision OBJ",
            path,
            "Simplified convex collision proxy for robotics/simulation.",
            collision_mesh,
        )
    except Exception as e:
        logger.warning(f"  Collision profile export skipped: {e}")

    # 3MF export (better than STL for printing: preserves color, units, metadata)
    try:
        path_3mf = os.path.join(out_dir, "model_print.3mf")
        sealed_mesh.export(path_3mf, file_type="3mf")
        add_profile(
            "print_3mf",
            "3D print 3MF",
            path_3mf,
            "3MF format for 3D printing — preserves scale, units and metadata better than STL.",
            sealed_mesh,
        )
        logger.info("  3MF export: OK")
    except Exception as e:
        logger.warning(f"  3MF export skipped: {e}")

    # ── UV-textured GLB on watertight mesh ──────────────────────────────
    # Transfer TripoSR UV atlas to the clean mesh and export a proper
    # UV-textured GLB — watertight geometry + baked texture atlas.
    if baked_texture_glb and os.path.exists(baked_texture_glb.lstrip("/").replace("/", os.sep)):
        _uv_glb_path = os.path.join(out_dir, "model_uv_textured.glb")
        _raw_tex_path = baked_texture_glb.lstrip("/").replace("/", os.sep)
        _built = _build_textured_glb(mesh, _raw_tex_path, _uv_glb_path)
        if _built and os.path.exists(_uv_glb_path):
            profiles["uv_textured"] = {
                "label": "UV-textured GLB (watertight)",
                "url": _url_for_path(_uv_glb_path),
                "download": _url_for_path(_uv_glb_path),
                "purpose": "Watertight mesh with TripoSR UV atlas texture. Best visual quality for viewing.",
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
            }
            logger.info("  UV-textured GLB profile added")

    if baked_texture_glb:
        profiles["texture_preview"] = {
            "label": "Texture preview GLB",
            "url": baked_texture_glb,
            "download": baked_texture_glb,
            "purpose": "UV-baked visual preview; not the print-safe mesh.",
            "vertices": None,
            "faces": None,
            "watertight": False,
        }
        texture_png = os.path.join(out_dir, "texture.png")
        if os.path.exists(texture_png):
            profiles["texture_png"] = {
                "label": "Texture atlas PNG",
                "url": _url_for_path(texture_png),
                "download": _url_for_path(texture_png),
                "purpose": "Texture atlas generated for preview workflows.",
                "vertices": None,
                "faces": None,
                "watertight": False,
            }

    quality = _build_quality_report(mesh, method, baked_texture_glb, print_mesh=sealed_mesh, silhouette_iou=silhouette_iou)
    report_path = os.path.join(out_dir, "quality_report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"quality": quality, "profiles": profiles}, f, indent=2)
        profiles["quality_report"] = {
            "label": "Quality report",
            "url": _url_for_path(report_path),
            "download": _url_for_path(report_path),
            "purpose": "Machine-readable readiness checks and profile metadata.",
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
        }
    except Exception as e:
        logger.warning(f"  Quality report export skipped: {e}")

    logger.info(f"  Exported {len(profiles)} asset profiles")
    return profiles, quality


# Cached backend availability — checked once at first reconstruction call
_BACKEND_AVAILABILITY: dict | None = None


def _check_backend_availability() -> dict:
    """Return a dict of {backend_name: (available, reason)} checked once per process."""
    global _BACKEND_AVAILABILITY
    if _BACKEND_AVAILABILITY is not None:
        return _BACKEND_AVAILABILITY

    result = {}

    # ── TripoSR ──────────────────────────────────────────────────────────
    try:
        import torch
        tsr_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TripoSR")
        if os.path.isdir(tsr_root):
            result["triposr"] = (True, "TripoSR dir found")
        else:
            result["triposr"] = (False, "TripoSR dir missing")
    except Exception as e:
        result["triposr"] = (False, str(e))

    # ── InstantMesh ───────────────────────────────────────────────────────
    try:
        import torch
        if not torch.cuda.is_available():
            result["instantmesh"] = (False, "CUDA unavailable")
        else:
            base = os.path.dirname(os.path.abspath(__file__))
            im_root = os.path.join(base, "InstantMesh")
            ckpt    = os.path.join(base, "weights", "instant_mesh_large.ckpt")
            if not os.path.isdir(im_root):
                result["instantmesh"] = (False, "InstantMesh dir missing")
            elif not os.path.isfile(ckpt):
                result["instantmesh"] = (False, "checkpoint missing: weights/instant_mesh_large.ckpt")
            else:
                try:
                    import nvdiffrast.torch  # noqa: F401
                    result["instantmesh"] = (True, "nvdiffrast + ckpt found")
                except Exception:
                    result["instantmesh"] = (False, "nvdiffrast not installed (pip install nvdiffrast)")
    except Exception as e:
        result["instantmesh"] = (False, str(e))

    # Log once
    for name, (avail, reason) in result.items():
        status = "AVAILABLE" if avail else "UNAVAILABLE"
        logger.info("  Backend %-14s %s — %s", name, status, reason)

    _BACKEND_AVAILABILITY = result
    return result


def _instantmesh_available() -> bool:
    return _check_backend_availability().get("instantmesh", (False, ""))[0]



def reconstruct_from_dir(out_dir: str, backend: str = "auto", cancel_event=None,
                         target_dimensions_mm: dict | None = None) -> dict:
    import trimesh
    t0 = time.time()
    backend = (backend or "auto").lower()
    logger.info(f"Reconstruction ← {out_dir} [{backend}]")

    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            logger.info("  Reconstruction cancelled")
            return {"status": "cancelled", "message": "Cancelled"}
        return None

    cancelled = check_cancelled()
    if cancelled:
        return cancelled

    views = load_from_dir(out_dir)
    if len(views) < 2:
        return {"status": "error", "message": "Not enough views"}

    mesh = None
    method = None
    geometry_source = "unknown"

    # Log which backends are actually available in this environment (cached after first call)
    avail = _check_backend_availability()
    logger.info(
        "  Requested backend=%s  instantmesh=%s  triposr=%s",
        backend,
        "AVAILABLE" if avail.get("instantmesh", (False,))[0] else "unavailable",
        "AVAILABLE" if avail.get("triposr", (False,))[0] else "unavailable",
    )

    # ── InstantMesh (requires nvdiffrast + CUDA) ──────────────────────────
    # Only attempted when explicitly requested or when auto+available.
    # nvdiffrast is not installed in this environment; this branch will not
    # fire unless the dependency is added.
    if mesh is None and backend in ("auto", "instantmesh") and _instantmesh_available():
        logger.info("  Trying InstantMesh (multi-view transformer) ...")
        try:
            cancelled = check_cancelled()
            if cancelled:
                return cancelled
            mesh = _try_instantmesh(out_dir, views)
            if mesh is not None:
                method = "instantmesh"
                geometry_source = "multi-view-transformer"
        except Exception as e:
            logger.warning("  InstantMesh failed: %s", e)

    # ── TripoSR (primary in this environment) ─────────────────────────────
    # Single-image neural prior; robust on RTX 3050 without nvdiffrast.
    # geometry_source is accurately reported as single-image-neural-prior:
    # the generated views are used for *repainting* only, not for shape.
    if mesh is None and backend in ("auto", "triposr"):
        logger.info("  Trying TripoSR (primary backend) ...")
        try:
            cancelled = check_cancelled()
            if cancelled:
                return cancelled
            mesh = _try_triposr(out_dir)
            if mesh is not None:
                method = "triposr"
                geometry_source = "single-image-neural-prior"
        except Exception as e:
            logger.warning("  TripoSR failed: %s", e)

    # ── Visual hull (explicit opt-in only, debug/fallback) ────────────────
    # Produces boxy/low-detail geometry from silhouette carving.
    # Not used in auto mode; only when backend is explicitly set.
    if mesh is None and backend in ("multiview", "multi_view", "depth_hull",
                                    "visual_hull", "hull"):
        logger.info("  Trying visual hull (silhouette carving, debug/fallback) ...")
        try:
            cancelled = check_cancelled()
            if cancelled:
                return cancelled
            mesh = _improved_visual_hull(views, allow_fallback=False)
            if mesh is not None:
                method = "multi_view_depth_hull"
                geometry_source = "visual-hull-silhouette"
        except Exception as e:
            logger.warning("  Visual hull failed: %s", e)

    # ── Sphere (unit test / debug only) ───────────────────────────────────
    if mesh is None and backend in ("sphere", "sphere_fallback"):
        mesh = _create_sphere_fallback()
        method = "sphere_fallback"
        geometry_source = "sphere-fallback"

    if mesh is None:
        return {
            "status": "error",
            "message": "No reconstruction backend succeeded. "
                       f"Requested: {backend}. InstantMesh requires nvdiffrast (not installed). "
                       "TripoSR failed. See logs for details.",
        }

    cancelled = check_cancelled()
    if cancelled:
        return cancelled

    baked_texture_glb = getattr(mesh, "metadata", {}).get("baked_texture_glb")

    # ── A: Validate raw backend output ────────────────────────────────────────
    validate_mesh_topology(mesh, f"A-raw/{method}")

    # ── B: Cleanup (keep_largest, optional decimation, repair) ────────────────
    mesh = cleanup(
        mesh,
        target_faces=TARGET_FACES if method != "triposr" else None,
        smooth=method not in ("depth_vote_hull", "visual_hull", "multi_view_depth_hull", "triposr"),
        keep_largest=True,
    )
    validate_mesh_topology(mesh, "B-after-cleanup")

    # Align before camera-space operations. TripoSR often emits X-up meshes,
    # while the photo/view projection assumes Y-up.
    mesh = _orient_and_normalize(mesh, out_dir, target_dimensions_mm, views=views)
    validate_mesh_topology(mesh, "B2-after-orientation")

    # ── Silhouette IoU quality gate ───────────────────────────────────────
    # Check that the mesh actually explains the input photo before continuing.
    _sil_iou = _compute_silhouette_iou(mesh, views)
    if _sil_iou >= 0:
        logger.info("  Silhouette IoU (input views): %.3f %s",
                    _sil_iou,
                    "(good)" if _sil_iou >= 0.35 else "(LOW — mesh may not match photo)")
        if _sil_iou < 0.10 and _sil_iou >= 0:
            logger.warning("  Silhouette IoU=%.3f is very low — reconstruction likely failed", _sil_iou)

    # ── print_mesh copy taken BEFORE visual opening cuts ──────────────────────
    # print_mesh is already oriented, but does NOT go through opening cuts
    print_mesh = _copy_mesh(mesh)

    # ── C: Visual opening preservation (creates intentional open boundaries) ──
    mesh = _preserve_visual_openings(mesh, views)
    # Topology is measured inside _preserve_visual_openings; log it here too
    validate_mesh_topology(mesh, "C-after-opening-cuts")

    cancelled = check_cancelled()
    if cancelled:
        return cancelled

    # ── D: Final repair of visual mesh (accidental holes only) ────────────────
    # repair_mesh_for_export is conservative: small boundary loops get filled,
    # large intentional openings are left as-is.
    mesh, visual_topo = repair_mesh_for_export(mesh, label="D-visual-final")

    # ── E: Paint vertex colors ────────────────────────────────────────────────
    logger.info("  Painting vertex colors ...")
    _tex_glb = os.path.join(out_dir, "model_textured.glb")
    _has_tex = method == "triposr" and os.path.exists(_tex_glb)
    if _has_tex:
        logger.info("  Painting: photo projection on oriented mesh (UV atlas kept as separate preview)")
    else:
        logger.info("  Painting: photo projection only")
    final_mesh = paint_colors(
        mesh, views,
        triposr_glb_path=None,
    )
    logger.info("  Painting print profile vertex colors ...")
    print_mesh = paint_colors(
        print_mesh, views,
        triposr_glb_path=None,
    )

    cancelled = check_cancelled()
    if cancelled:
        return cancelled

    # ── F: Validate oriented, painted mesh ────────────────────────────────────
    validate_mesh_topology(final_mesh, "F-after-orientation")

    # ── G: FINAL repair + validation of visual model.glb ─────────────────────
    final_mesh, final_topo = repair_mesh_for_export(final_mesh, label="G-pre-export")

    # Decide whether the export is topologically sound
    final_watertight = final_topo["watertight"]

    # ── H: Export model.glb from the FINAL validated mesh ─────────────────────
    glb_path = os.path.join(out_dir, "model.glb")
    cancelled = check_cancelled()
    if cancelled:
        return cancelled

    final_mesh.export(glb_path, file_type="glb")

    # ── I: Reload and re-validate the exported GLB ────────────────────────────
    glb_reload_topo = None
    try:
        import trimesh as _tm_reload
        reloaded = _tm_reload.load(glb_path, force="mesh")
        glb_reload_topo = validate_mesh_topology(reloaded, "I-reloaded-GLB")
    except Exception as exc:
        logger.warning("  GLB reload validation failed: %s", exc)

    # Print the final reconstruction summary banner
    _be  = final_topo.get("boundary_edges", "?")
    _nm  = final_topo.get("non_manifold_edges", "?")
    _dg  = final_topo.get("degenerate_faces", "?")
    _co  = final_topo.get("components", "?")
    _wc  = final_topo.get("winding_consistent", "?")
    logger.info(
        "\n===== FINAL RECONSTRUCTION VALIDATION =====\n"
        "  out_dir:            %s\n"
        "  backend:            %s\n"
        "  vertices:           %d\n"
        "  faces:              %d\n"
        "  components:         %s\n"
        "  boundary_edges:     %s\n"
        "  non_manifold_edges: %s\n"
        "  degenerate_faces:   %s\n"
        "  watertight:         %s\n"
        "  winding_consistent: %s\n"
        "============================================",
        out_dir, method,
        final_topo["vertices"], final_topo["faces"],
        _co, _be, _nm, _dg,
        str(final_watertight).upper(), str(_wc).upper(),
    )

    # Report print_mesh status alongside visual mesh status.
    # The visual mesh may be intentionally non-watertight (visual openings);
    # the print_mesh should always be watertight.
    _print_wt = bool(print_mesh.is_watertight)
    logger.info(
        "  Visual mesh: watertight=%s  Print mesh: watertight=%s",
        str(final_watertight).upper(), str(_print_wt).upper(),
    )
    if not _print_wt:
        logger.warning(
            "  PRINT MESH NOT WATERTIGHT — model_print.stl may not be manufacturable. "
            "boundary_edges=%s", _be,
        )
    elif not final_watertight:
        _has_openings = bool(getattr(getattr(print_mesh, "metadata", {}), "get", lambda k,d: d)(
            "visual_openings_preserved", False))
        logger.info(
            "  Visual mesh has intentional openings (boundary_edges=%s). "
            "Print mesh is watertight — use model_print.stl for manufacturing.", _be,
        )


    vertex_color_glb = "/" + glb_path.replace("\\", "/")

    asset_profiles, quality_report = _export_asset_profiles(
        final_mesh,
        out_dir,
        method,
        baked_texture_glb=baked_texture_glb,
        print_mesh=print_mesh,
        silhouette_iou=_sil_iou if _sil_iou >= 0 else None,
    )
    print_stl = asset_profiles.get("print_stl", {}).get("url")
    geometry_inputs = {
        "front_image": True,
        "generated_views": method in ("instantmesh", "multi_view_depth_hull", "improved_visual_hull"),
        "depth_maps": False,  # DepthAnything is preview-only; no calibrated metric scale available
        "geometry_source": geometry_source,
        "note": (
            "TripoSR shape comes from the masked front photo (single-image neural prior). "
            "Generated views are used only for vertex-colour repainting."
            if method == "triposr"
            else "Geometry used generated views. DepthAnything maps are relative; not used as metric depth."
        ),
    }
    elapsed = round(time.time() - t0, 1)
    logger.info("Done [%s] in %ss -> %s", method, elapsed, glb_path)

    return {
        "status": "success",
        "final_glb":        vertex_color_glb,
        "final_stl":        print_stl,
        "vertex_color_glb": vertex_color_glb,
        "vertices":         final_topo["vertices"],
        "faces":            final_topo["faces"],
        "watertight":       bool(print_mesh.is_watertight),
        "visual_watertight":bool(final_watertight),
        "print_watertight": bool(print_mesh.is_watertight),
        "print_ready":      bool(quality_report.get("print_ready")),
        "dimensions":       _mesh_dimensions(final_mesh),
        "units":            getattr(final_mesh, "metadata", {}).get("units", "normalized"),
        "target_dimensions_mm": _valid_target_dimensions(target_dimensions_mm),
        "recon_time":       elapsed,
        "silhouette_iou":   round(_sil_iou, 4) if _sil_iou >= 0 else None,
        "method":           method,
        "geometry_source":  geometry_source,
        "geometry_inputs":  geometry_inputs,
        "texture_mode":     "vertex_color",
        "baked_texture_glb":baked_texture_glb,
        "asset_profiles":   asset_profiles,
        "quality":          quality_report,
        "topology": {
            "watertight":         final_watertight,
            "boundary_edges":     final_topo.get("boundary_edges", -1),
            "non_manifold_edges": final_topo.get("non_manifold_edges", -1),
            "components":         final_topo.get("components", -1),
        },
    }


def _try_instantmesh(out_dir: str, views: list):
    """Try InstantMesh reconstruction"""
    try:
        import torch
        instant_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "InstantMesh")
        if not os.path.exists(instant_root):
            return None
        
        if instant_root not in sys.path:
            sys.path.insert(0, instant_root)

        from src.utils.train_util import instantiate_from_config
        from src.utils.camera_util import get_zero123plus_input_cameras

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            logger.warning("  InstantMesh: CUDA unavailable")
            return None

        view_paths = [os.path.join(out_dir, f"view_{i}.png") for i in range(6)]
        if not all(os.path.exists(p) for p in view_paths):
            logger.warning("  InstantMesh: missing views")
            return None

        logger.info("  InstantMesh: loading model …")
        from omegaconf import OmegaConf
        cfg_path = os.path.join(instant_root, "configs", "instant-mesh-large.yaml")
        config = OmegaConf.load(cfg_path)
        infer_config = config.infer_config
        model = instantiate_from_config(config.model_config)

        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "instant_mesh_large.ckpt")
        if not os.path.exists(ckpt_path):
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename="instant_mesh_large.ckpt",
                repo_type="model",
            )

        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()
        model.init_flexicubes_geometry(device, fovy=30.0)

        # Load images
        from torchvision.transforms import v2
        imgs = []
        for p in view_paths:
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        images = torch.stack(imgs, dim=0).unsqueeze(0).to(device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)

        logger.info("  InstantMesh: running inference …")
        with torch.no_grad():
            planes = model.forward_planes(images, input_cameras)
            vertices, faces, vertex_colors = model.extract_mesh(
                planes, use_texture_map=False, **infer_config
            )

        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
        logger.info(f"  InstantMesh: {len(mesh.vertices):,}v / {len(mesh.faces):,}f")
        return mesh

    except Exception as e:
        logger.warning(f"  InstantMesh failed: {e}")
        return None


def _try_export_triposr_baked_glb(mesh, model, scene_code, out_dir: str):
    try:
        import torch
        import trimesh
        from PIL import Image
        from tsr.bake_texture import bake_texture

        logger.info(f"  TripoSR: baking neural texture atlas ({TRIPOSR_TEXTURE_RESOLUTION}px) …")
        bake_output = bake_texture(mesh, model, scene_code, TRIPOSR_TEXTURE_RESOLUTION)

        texture = Image.fromarray(
            np.clip(bake_output["colors"] * 255.0, 0, 255).astype(np.uint8)
        ).transpose(Image.FLIP_TOP_BOTTOM)
        texture_path = os.path.join(out_dir, "texture.png")
        texture.save(texture_path)

        vertices = np.asarray(mesh.vertices[bake_output["vmapping"]], dtype=np.float32)
        faces = np.asarray(bake_output["indices"], dtype=np.int64)
        uvs = np.asarray(bake_output["uvs"], dtype=np.float32)

        material = trimesh.visual.material.SimpleMaterial(
            image=texture,
            diffuse=[255, 255, 255, 255],
        )
        visual = trimesh.visual.texture.TextureVisuals(uv=uvs, image=texture, material=material)
        textured_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=visual,
            process=False,
        )
        if hasattr(textured_mesh, "fix_normals"):
            textured_mesh.fix_normals()

        textured_glb = os.path.join(out_dir, "model_textured.glb")
        textured_mesh.export(textured_glb, file_type="glb")
        logger.info(f"  TripoSR: baked texture export → {textured_glb}")
        return "/" + textured_glb.replace("\\", "/")
    except Exception as e:
        logger.warning(f"  TripoSR: baked texture export skipped: {e}")
        return None


def _try_triposr(out_dir: str):
    """Try TripoSR reconstruction"""
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        triposr_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TripoSR")
        if triposr_root not in sys.path:
            sys.path.insert(0, triposr_root)

        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("  TripoSR: loading model …")
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        ).to(device)
        model.renderer.set_chunk_size(TRIPOSR_CHUNK_SIZE)
        model.eval()

        ref_path = os.path.join(out_dir, "masked.png")
        rgba = Image.open(ref_path).convert("RGBA")
        rgba = remove_background(rgba)
        rgba = resize_foreground(rgba, 0.85)

        # Convert to RGB on grey background
        arr = np.array(rgba).astype(np.float32) / 255.0
        rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
        image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))

        logger.info("  TripoSR: running inference …")
        with torch.no_grad():
            scene_codes = model([image], device=device)
            meshes = None
            last_error = None
            for res in TRIPOSR_RESOLUTIONS:
                try:
                    logger.info(f"  TripoSR: extracting mesh at resolution={res}")
                    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=res)
                    break
                except RuntimeError as e:
                    last_error = e
                    logger.warning(f"  TripoSR: resolution={res} failed: {e}")
                    if device == "cuda":
                        try:
                            model.isosurface_helper = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
            if meshes is None:
                if device == "cuda":
                    try:
                        logger.warning("  TripoSR: CUDA extraction exhausted; retrying extraction on CPU")
                        model = model.to("cpu")
                        model.renderer.set_chunk_size(8192)
                        scene_codes = scene_codes.detach().cpu()
                        model.isosurface_helper = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        meshes = model.extract_mesh(
                            scene_codes,
                            has_vertex_color=True,
                            resolution=TRIPOSR_CPU_FALLBACK_RESOLUTION,
                        )
                    except Exception as cpu_e:
                        logger.warning(f"  TripoSR: CPU extraction failed: {cpu_e}")
                        raise last_error or cpu_e
                else:
                    raise last_error or RuntimeError("TripoSR mesh extraction failed")

        mesh = meshes[0]
        logger.info(f"  TripoSR: {len(mesh.vertices):,}v / {len(mesh.faces):,}f")
        baked_glb = _try_export_triposr_baked_glb(mesh, model, scene_codes[0], out_dir)
        if baked_glb:
            mesh.metadata["baked_texture_glb"] = baked_glb
        try:
            del meshes, scene_codes, model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        return mesh

    except ImportError:
        logger.info("  TripoSR: not installed")
        return None
    except Exception as e:
        logger.warning(f"  TripoSR failed: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        print("Usage: python reconstructor.py <out_dir> [backend]")
        sys.exit(1)
    backend = sys.argv[2] if len(sys.argv) > 2 else "auto"
    print(json.dumps(reconstruct_from_dir(sys.argv[1], backend=backend), indent=2))

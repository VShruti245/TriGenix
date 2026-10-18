import os, json, logging, numpy as np
logger = logging.getLogger(__name__)


def _normalize_verts_for_projection(verts):
    """
    Normalize vertices to [-0.5, 0.5] for projection, same as paint_colors.
    _project() uses CAM_DIST=1.80 and expects the mesh inside a ~1.0-unit sphere.
    Meshes in mm scale (bounds +-50) project entirely outside the image frame.
    Returns (verts_norm, center, scale) so callers can invert if needed.
    """
    center = verts.mean(axis=0)
    extent = np.abs(verts - center).max()
    if extent > 2.0:
        scale = 0.5 / (extent + 1e-8)
        return (verts - center) * scale, center, scale
    return verts.copy(), center, 1.0


def _safe_project(verts, az, el, S):
    """
    Project verts and return only valid, in-image, finite pixel coordinates.
    Returns (sel_indices, ui, vi) — all already clipped to [0, S-1].
    """
    from reconstructor import _project
    u, vp, d, ok = _project(verts, az, el)
    # Guard against NaN/Inf before integer cast
    finite = np.isfinite(u) & np.isfinite(vp) & ok
    if not finite.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    ui = np.round(u).astype(np.int32)
    vi = np.round(vp).astype(np.int32)
    in_img = finite & (ui >= 0) & (ui < S) & (vi >= 0) & (vi < S)
    sel = np.where(in_img)[0]
    return sel, ui[sel], vi[sel]


def compute_confidence(out_dir, mesh, view_images=None):
    try:
        import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from reconstructor import load_from_dir, IMG_SIZE, FG_ALPHA_THRESHOLD
        from PIL import Image as PI

        views = load_from_dir(out_dir)
        if not views:
            return _write_uniform(out_dir, mesh, 0.5)

        verts_raw = np.array(mesh.vertices, dtype=np.float32)
        N = len(verts_raw)
        S = IMG_SIZE

        # Normalize to projection space — CRITICAL fix (was missing before)
        verts, _, _ = _normalize_verts_for_projection(verts_raw)

        conf = np.zeros(N, dtype=np.float32)

        for v in views:
            # Load foreground mask for this view
            if v.get('mask_path') and os.path.exists(v['mask_path']):
                fg = np.array(
                    PI.open(v['mask_path']).convert('L').resize((S, S), PI.NEAREST)
                ) >= FG_ALPHA_THRESHOLD
            elif v.get('alpha_mask') is not None:
                fg = v['alpha_mask']
            else:
                continue

            sel, ui, vi = _safe_project(verts, v['az'], v['el'], S)
            if not len(sel):
                continue

            # Mark vertices that project into the foreground mask
            in_fg = fg[vi, ui]
            conf[sel[in_fg]] = 1.0

        result = {'vertex_confidence': conf.tolist()}
        p = os.path.join(out_dir, 'confidence_map.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(result, f)
        n_vis = int((conf == 1.0).sum())
        logger.info('  Confidence map written: %d/%d visible (%.1f%%)',
                    n_vis, N, 100.0 * n_vis / max(1, N))
        return result

    except Exception as e:
        logger.warning('confidence_mapper failed: %s', e)
        return _write_uniform(out_dir, mesh, 0.5)


def _write_uniform(out_dir, mesh, val):
    N = len(mesh.vertices) if mesh is not None else 0
    r = {'vertex_confidence': [val] * N}
    try:
        with open(os.path.join(out_dir, 'confidence_map.json'), 'w', encoding='utf-8') as f:
            json.dump(r, f)
    except Exception:
        pass
    return r

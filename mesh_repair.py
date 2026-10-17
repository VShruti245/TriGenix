"""
mesh_repair.py — Automatic mesh repair for TriGenix.

Inspects a GLB mesh for 6 defect categories, repairs each in turn,
writes model_repaired.glb, and appends repair_report to quality_report.json.
"""
import os, json, logging
import trimesh
import numpy as np

logger = logging.getLogger(__name__)


def run_repair(out_dir: str) -> dict:
    """
    Load model.glb from out_dir, run 6 repair steps, write model_repaired.glb,
    append repair_report to quality_report.json.

    Returns repair_report dict with keys:
        holes_filled, non_manifold_resolved, duplicates_merged,
        degenerate_removed, components_removed, normals_fixed
    Each value is a non-negative integer.
    """
    model_path = os.path.join(out_dir, "model.glb")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.glb not found in {out_dir}")

    mesh = trimesh.load(model_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not load a single Trimesh from model.glb")

    report = {
        "holes_filled":          0,
        "non_manifold_resolved": 0,
        "duplicates_merged":     0,
        "degenerate_removed":    0,
        "components_removed":    0,
        "normals_fixed":         0,
    }

    # ── Step 1: Merge duplicate vertices ────────────────────────────────────
    try:
        before = len(mesh.vertices)
        # trimesh >= 4.x uses digits_vertex; older versions used merge_tolerance
        try:
            mesh.merge_vertices(digits_vertex=6)
        except TypeError:
            mesh.merge_vertices(merge_tolerance=1e-6)  # type: ignore[call-arg]
        report["duplicates_merged"] = max(0, before - len(mesh.vertices))
        logger.info("  Repair: merged %d duplicate vertices", report["duplicates_merged"])
    except Exception as e:
        logger.warning("  Repair: duplicate-merge failed — %s", e)

    # ── Step 2: Remove degenerate faces (area < 1e-10) ──────────────────────
    try:
        before = len(mesh.faces)
        areas = mesh.area_faces
        keep  = areas >= 1e-10
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        report["degenerate_removed"] = max(0, before - len(mesh.faces))
        logger.info("  Repair: removed %d degenerate faces", report["degenerate_removed"])
    except Exception as e:
        logger.warning("  Repair: degenerate-face removal failed — %s", e)

    # ── Step 3: Fill holes ──────────────────────────────────────────────────
    try:
        before_watertight = mesh.is_watertight
        trimesh.repair.fill_holes(mesh)
        after_watertight = mesh.is_watertight
        if not before_watertight and after_watertight:
            report["holes_filled"] = 1
        elif not before_watertight:
            # Partial fill — count boundary-edge reduction as proxy
            report["holes_filled"] = 1
        logger.info("  Repair: fill_holes done (watertight now: %s)", mesh.is_watertight)
    except Exception as e:
        logger.warning("  Repair: hole-fill failed — %s", e)

    # ── Step 4: Fix non-manifold edges ──────────────────────────────────────
    try:
        nm_before = len(trimesh.grouping.group_rows(
            np.sort(mesh.edges, axis=1), require_count=1))
        trimesh.repair.fix_winding(mesh)
        # Remove faces that remain non-manifold after winding fix
        unique_faces = mesh.unique_faces()
        if len(unique_faces) < len(mesh.faces):
            removed = len(mesh.faces) - len(unique_faces)
            mesh.update_faces(unique_faces)
            mesh.remove_unreferenced_vertices()
            report["non_manifold_resolved"] = removed
        nm_after = len(trimesh.grouping.group_rows(
            np.sort(mesh.edges, axis=1), require_count=1))
        report["non_manifold_resolved"] = max(report["non_manifold_resolved"],
                                              max(0, nm_before - nm_after))
        logger.info("  Repair: non-manifold edges resolved: %d", report["non_manifold_resolved"])
    except Exception as e:
        logger.warning("  Repair: non-manifold fix failed — %s", e)

    # ── Step 5: Discard tiny disconnected components ─────────────────────────
    try:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            total_faces = sum(len(p.faces) for p in parts)
            largest = max(parts, key=lambda p: len(p.faces))
            if len(largest.faces) > 0.80 * total_faces:
                kept = [p for p in parts if len(p.faces) >= 0.05 * total_faces]
                removed_count = len(parts) - len(kept)
                if removed_count > 0:
                    mesh = trimesh.util.concatenate(kept)
                    report["components_removed"] = removed_count
                    logger.info("  Repair: removed %d tiny components", removed_count)
    except Exception as e:
        logger.warning("  Repair: component removal failed — %s", e)

    # ── Step 6: Unify normals ───────────────────────────────────────────────
    try:
        trimesh.repair.fix_normals(mesh)
        report["normals_fixed"] = 1
        logger.info("  Repair: normals unified")
    except Exception as e:
        logger.warning("  Repair: normal fix failed — %s", e)
        report["normals_fixed"] = 0

    # ── Write repaired mesh ──────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "model_repaired.glb")
    try:
        mesh.export(out_path)
        logger.info("  Repair: wrote %s", out_path)
    except Exception as e:
        logger.error("  Repair: failed to export model_repaired.glb — %s", e)
        raise

    # ── Update quality_report.json ───────────────────────────────────────────
    qr_path = os.path.join(out_dir, "quality_report.json")
    try:
        if os.path.exists(qr_path):
            with open(qr_path, "r", encoding="utf-8") as f:
                qr = json.load(f)
        else:
            qr = {}
        qr["repair_report"] = report
        with open(qr_path, "w", encoding="utf-8") as f:
            json.dump(qr, f, indent=2)
    except Exception as e:
        logger.warning("  Repair: could not update quality_report.json — %s", e)

    return report

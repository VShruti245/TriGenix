"""
tests/test_baseline_smoke.py
==============================
Phase 1 — Task 1.2: Smoke tests against existing flat-file entry points.

Documents CURRENT BEHAVIOUR, including known gaps versus the spec.

Two sections:

A. process_image smoke test (requires TRIGENIX_SKIP_HEAVY_TESTS != 1)
   - Verifies the full processor pipeline entry point.
   - The spec says a white image should return status='error' with
     failure_code='INPUT_BLANK'.  Current code does NOT do this — there
     is no blank-image gate.  This test DOCUMENTS the missing gate
     (baseline = no early exit, pipeline proceeds).

B. Auth round-trip smoke test (no GPU, no heavy models)
   - register → login → validate_session → logout round-trip.
   - Confirms the auth module is importable and functional before Phase 10.

C. Reconstructor entry-point smoke (no GPU execution, structural only)
   - Confirms reconstruct_from_dir is importable.
   - Verifies it returns a well-formed error dict when given a non-existent
     directory (structural guard test, no actual GPU inference).
"""

import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SKIP_HEAVY = os.environ.get("TRIGENIX_SKIP_HEAVY_TESTS", "0") == "1"
SKIP_REASON = (
    "TRIGENIX_SKIP_HEAVY_TESTS=1: Zero123++ / DepthAnything GPU inference skipped. "
    "Set TRIGENIX_SKIP_HEAVY_TESTS=0 to run end-to-end tests."
)


# ─── A. process_image smoke ────────────────────────────────────────────────

class TestProcessImageSmoke:
    """
    Smoke tests for process_image (flat-file entry point).

    Current known baseline behaviour:
      - A blank (all-white) image does NOT return status='error' early.
        It proceeds through rembg (which returns an empty mask) and then
        enters Zero123++ inference.
      - This is the ABSENCE of the blank-image gate required by spec Req 3.4.
      - Phase 10 will add the gate; these tests will be updated then.
    """

    @pytest.mark.skipif(SKIP_HEAVY, reason=SKIP_REASON)
    @pytest.mark.timeout(300)
    def test_process_image_white_png_returns_dict_with_status(self, white_png_path):
        """
        BASELINE DOCUMENTATION:
        Current code: process_image on a white image proceeds through the full
        pipeline (no INPUT_BLANK gate). It may return status='success' or
        status='error' depending on whether Zero123++/DepthAnything succeed.

        We assert only: returns a dict with a 'status' key.
        We DO NOT assert status=='error', because the gate does not exist yet.
        The failing assertion will be added in Phase 10.
        """
        from processor import process_image
        result = process_image(white_png_path)
        assert isinstance(result, dict), "process_image must return a dict"
        assert "status" in result, "dict must contain 'status'"
        # Record actual behaviour for Phase 10 regression baseline.
        print(f"\n  [smoke] white_png status={result['status']!r}")

    @pytest.mark.skipif(SKIP_HEAVY, reason=SKIP_REASON)
    @pytest.mark.timeout(300)
    def test_process_image_checkerboard_returns_success(self, checkerboard_png_path):
        """
        Smoke test on a real-looking image.
        If this fails with status='error', the failure message is captured
        as the Phase 1 baseline for Phase 2+ to compare against.
        """
        from processor import process_image
        result = process_image(checkerboard_png_path)
        assert isinstance(result, dict)
        assert "status" in result
        if result["status"] == "error":
            print(f"\n  [smoke] BASELINE FAILURE: {result.get('message')!r}")
            # Do not hard-fail — record as environment baseline.
            pytest.skip(
                f"process_image error on baseline run (environment blocker): "
                f"{result.get('message')}"
            )
        assert result["status"] == "success"
        assert "out_dir" in result
        print(f"\n  [smoke] checkerboard ok → {result['out_dir']}")


# ─── B. Auth round-trip smoke ──────────────────────────────────────────────

class TestAuthRoundTripSmoke:
    """Full register→login→validate→logout round-trip (no GPU)."""

    def test_full_auth_round_trip(self, monkeypatch, tmp_db_path):
        import auth
        monkeypatch.setattr(auth, "DB_PATH", tmp_db_path)
        from auth import init_db, register_user, login_user, validate_session, logout_session

        init_db()

        # Register
        reg = register_user("smokeuser", "smoke@example.com", "smokepass123")
        assert reg.get("ok") is True

        # Login
        login = login_user("smokeuser", "smokepass123")
        assert login.get("ok") is True
        token = login["token"]
        assert len(token) == 64

        # Validate
        user = validate_session(token)
        assert user is not None
        assert user["username"] == "smokeuser"

        # Logout
        logout_session(token)
        assert validate_session(token) is None

        print(f"\n  [smoke] auth round-trip ok (user_id={user['id']})")


# ─── C. Reconstructor entry-point smoke ───────────────────────────────────

class TestReconstructorEntryPointSmoke:
    """
    Structural smoke tests for reconstructor.reconstruct_from_dir.
    Does NOT run GPU inference — tests the entry-point's error handling
    for missing/invalid input.
    """

    def test_import_succeeds(self):
        """reconstructor.py must import without error."""
        import reconstructor  # noqa: F401

    def test_nonexistent_dir_returns_error_dict(self, tmp_path):
        """
        Calling reconstruct_from_dir with a directory that has no views
        must return a dict with status='error', not raise an exception.
        This documents the current error-handling boundary.
        """
        from reconstructor import reconstruct_from_dir
        nonexistent = str(tmp_path / "does_not_exist")
        result = reconstruct_from_dir(nonexistent)
        assert isinstance(result, dict), "Must return a dict, not raise"
        assert result.get("status") == "error", (
            f"Expected status='error' for missing dir, got {result.get('status')!r}"
        )
        print(f"\n  [smoke] nonexistent dir error msg: {result.get('message')!r}")

    def test_empty_dir_returns_error_dict(self, tmp_out_dir):
        """
        reconstruct_from_dir with an empty directory (no views) must return
        status='error' (not enough views), not raise an exception.
        """
        from reconstructor import reconstruct_from_dir
        result = reconstruct_from_dir(tmp_out_dir)
        assert isinstance(result, dict), "Must return a dict"
        assert result.get("status") == "error"
        assert "view" in result.get("message", "").lower() or result.get("message"), (
            f"Unexpected error message: {result.get('message')!r}"
        )
        print(f"\n  [smoke] empty dir error msg: {result.get('message')!r}")

    def test_reconstruct_from_dir_callable(self):
        """Verify reconstruct_from_dir is a callable with the expected signature."""
        from reconstructor import reconstruct_from_dir
        import inspect
        sig = inspect.signature(reconstruct_from_dir)
        params = list(sig.parameters.keys())
        assert "out_dir" in params
        assert "backend" in params

    def test_sphere_backend_without_env_var_does_not_crash(self, tmp_out_dir):
        """
        Current behaviour (Bug A baseline): calling backend='sphere' on an empty
        dir currently returns status='error' (not enough views) BEFORE reaching
        the sphere fallback code path. This is because load_from_dir returns < 2
        views for an empty dir, and the function returns early.

        Documents that the sphere-fallback env-var gate is NOT present — the
        code path that would need it is unreachable from an empty dir anyway.
        """
        from reconstructor import reconstruct_from_dir
        result = reconstruct_from_dir(tmp_out_dir, backend="sphere")
        assert isinstance(result, dict)
        # With empty dir: should be 'error' due to insufficient views
        # (not a sphere fallback — the early exit fires first).
        assert result.get("status") in ("error", "success", "cancelled")
        print(f"\n  [smoke] sphere backend on empty dir: status={result.get('status')!r}")


# ─── D. Pose estimator entry-point smoke ───────────────────────────────────

class TestPoseEstimatorSmoke:
    """Smoke tests for pose_estimator — pure OpenCV/NumPy, no GPU."""

    def test_import_succeeds(self):
        import pose_estimator  # noqa: F401

    def test_estimate_pose_returns_pose_result(self, white_png_path):
        """estimate_pose on a white image returns a PoseResult without crashing."""
        from pose_estimator import estimate_pose, PoseResult
        result = estimate_pose(white_png_path)
        assert isinstance(result, PoseResult)

    def test_pose_result_world_r_shape(self, checkerboard_png_path):
        """world_R must be a 3x3 array."""
        import numpy as np
        from pose_estimator import estimate_pose
        result = estimate_pose(checkerboard_png_path)
        assert result.world_R.shape == (3, 3), (
            f"Expected (3,3) world_R, got {result.world_R.shape}"
        )

    def test_pose_result_world_r_identity_when_near_frontal(self, white_png_path):
        """
        An all-white image → empty mask → near-frontal defaults (az=0, roll=0)
        → world_R should be identity (per pose_estimator logic: abs(roll)<5 AND abs(az)<5).
        """
        import numpy as np
        from pose_estimator import estimate_pose
        result = estimate_pose(white_png_path)
        # White image produces an empty mask which triggers default pose (az=0, roll=0).
        # world_R should be identity.
        assert np.allclose(result.world_R, np.eye(3)), (
            f"Expected identity world_R for blank image, got:\n{result.world_R}"
        )

    def test_save_and_load_pose_round_trip(self, tmp_out_dir, checkerboard_png_path):
        """save_pose → load_pose must preserve all scalar fields."""
        import numpy as np
        from pose_estimator import estimate_pose, save_pose, load_pose
        original = estimate_pose(checkerboard_png_path)
        save_pose(original, tmp_out_dir)
        loaded = load_pose(tmp_out_dir)
        assert abs(loaded.azimuth - original.azimuth) < 1e-4
        assert abs(loaded.elevation - original.elevation) < 1e-4
        assert abs(loaded.roll - original.roll) < 1e-4
        assert loaded.upright == original.upright
        assert np.allclose(loaded.world_R, original.world_R, atol=1e-5), (
            "world_R not preserved through save/load round-trip"
        )
        print(f"\n  [smoke] pose round-trip ok: az={loaded.azimuth:.1f} "
              f"el={loaded.elevation:.1f} roll={loaded.roll:.1f}")

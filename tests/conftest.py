"""
tests/conftest.py
=================
Phase 1 — pytest configuration and shared fixtures.

Registers Hypothesis profiles:
  ci  — 100 examples, no deadline  (CI / thorough runs)
  dev —  20 examples, 5 s deadline (local fast iteration)

Profile is selected via the HYPOTHESIS_PROFILE env var; defaults to "dev".

No production code is modified here.
"""
import os
import tempfile
import shutil
import pytest
from hypothesis import settings, HealthCheck

# ---------------------------------------------------------------------------
# Hypothesis profiles (Phase 1 infrastructure)
# ---------------------------------------------------------------------------

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "dev",
    max_examples=20,
    deadline=5000,  # 5 seconds
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_out_dir():
    """
    Create a temporary output directory that mirrors the flat structure
    the current production pipeline uses (generated/<name>/).
    Cleaned up automatically after the test.
    """
    d = tempfile.mkdtemp(prefix="trigenix_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_db_path(tmp_path):
    """
    Return a path to a fresh, isolated SQLite database for auth tests.
    Each test that uses this fixture gets its own DB so tests are independent.
    """
    return str(tmp_path / "test_trigenix.db")


@pytest.fixture
def white_png_path(tmp_path):
    """
    Write a 64x64 all-white PNG to a temp directory and return its path.
    Used as a known-blank input for smoke tests.
    """
    from PIL import Image
    p = tmp_path / "white.png"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(str(p))
    return str(p)


@pytest.fixture
def solid_red_png_path(tmp_path):
    """
    Write a 128x128 solid-red PNG (simulates a foreground-only image).
    """
    from PIL import Image
    p = tmp_path / "red.png"
    Image.new("RGB", (128, 128), (200, 50, 50)).save(str(p))
    return str(p)


@pytest.fixture
def checkerboard_png_path(tmp_path):
    """
    Write a 128x128 black-and-white checkerboard PNG.
    Useful for depth/segmentation tests because it has both light and dark regions.
    """
    import numpy as np
    from PIL import Image
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    for r in range(128):
        for c in range(128):
            if (r // 16 + c // 16) % 2 == 0:
                arr[r, c] = 255
    p = tmp_path / "checker.png"
    Image.fromarray(arr).save(str(p))
    return str(p)

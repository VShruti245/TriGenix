# TriGenix

Single-image 3D reconstruction platform. Upload one photo, get a textured, watertight 3D mesh — viewable in browser, exportable for 3D printing, AR/VR, and gaming.

---

## What it does

```
Single photo
    → Zero123++ synthesises 6 novel views
    → DPT-Large estimates depth maps
    → TripoSR reconstructs 3D mesh
    → Region-based k-means colour projection
    → Mesh repair + watertight validation
    → 9-format export (GLB, STL, 3MF, AR, UV, LOD...)
    → Three.js interactive browser viewer
```

---

## Features

- Single-image to textured 3D mesh (no special hardware needed)
- TripoSR neural reconstruction with visual-hull CPU fallback
- Region-based colour projection — k-means segmentation, per-view calibration, hue-conflict rejection
- 100% watertight print exports via PyMeshFix repair pipeline
- 9 export formats — GLB, STL, 3MF, AR/VR GLB, UV-textured GLB, LOD, Simulation, Collision OBJ
- Interactive Three.js viewer with orbit controls, shading modes, lighting presets
- Multi-user authentication, per-user history, share links
- Integrated 3D-printing workflow
- Real-time stage-by-stage progress, ETA, cancellation
- 82 automated regression tests

---

## Requirements

- Python 3.11+
- NVIDIA GPU, 6 GB+ VRAM, CUDA 11.8+
- 16 GB RAM (32 GB recommended)
- 20 GB free disk space

---

## Setup

```bash
# 1. Clone
git clone https://github.com/VShruti245/3dModelGen.git
cd 3dModelGen

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env if needed

# 5. Model weights
# Zero123++, DPT-Large, rembg — auto-downloaded on first run
# TripoSR — place checkpoint in weights/

# 6. Run
python server.py
# Open http://localhost:8000
```

---

## Running Tests

```bash
# Without GPU (fast)
set TRIGENIX_SKIP_HEAVY_TESTS=1
python -m pytest tests/ -q

# Full suite (requires GPU + weights)
python -m pytest tests/ -q
```

---

## Project Structure

```
server.py                 HTTP server + routing
processor.py              Background removal, Zero123++, DPT depth
reconstructor.py          TripoSR reconstruction, colour projection, export
auth.py                   Authentication, sessions, history, shares
printer.py                3D print job management
mesh_repair.py            Automatic mesh repair
pose_estimator.py         Input photo pose analysis
quality_scorer.py         Mesh quality scoring
confidence_mapper.py      Per-vertex confidence mapping
multi_angle_processor.py  4-angle capture mode
cleanup_worker.py         Stale job cleanup
HTML/                     Frontend pages
JS/                       Frontend JavaScript
CSS/                      Stylesheets
tests/                    82 automated tests
requirements.txt          Python dependencies
.env.example              Configuration template
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML / CSS / JavaScript, Three.js r128 |
| Backend | Python 3.11, ThreadingHTTPServer |
| Reconstruction | TripoSR, Zero123++ (Diffusers), DPT-Large (Transformers) |
| Background removal | rembg / u2net |
| Geometry | trimesh, PyMeshFix, scikit-image |
| ML | PyTorch (CUDA), NumPy, SciPy, scikit-learn |
| Database | SQLite |
| Testing | pytest |

---

## Results

| Metric | Value |
|--------|-------|
| Models evaluated | 70+ |
| Watertight rate | 100% |
| Mean silhouette IoU | 0.452 |
| Mean colour error | 5.7 / 255 |
| Processing time (GPU) | 2 – 18 min |
| Export formats | 9 |
| Automated tests | 82 passing |

---

## Academic Context

Final-year project — Usha Mittal Institute of Technology, S.N.D.T. Women's University, Data Science & Engineering Department.

Developed by Shruti Vyavahare and Sumedha Gamare under guidance of Prof. Sheetal Mhatre.

---

## Third-Party Licenses

| Component | License |
|-----------|---------|
| TripoSR | [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR/blob/main/LICENSE) |
| Zero123++ | [sudo-ai-3d/zero123plus](https://github.com/sudo-ai-3d/zero123plus/blob/main/LICENSE) |
| Intel DPT-Large | Apache 2.0 |
| rembg / u2net | Apache 2.0 |
| trimesh | MIT |
| Three.js | MIT |

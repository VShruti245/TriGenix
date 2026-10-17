# TriGenix Migration Map

Generated from source inventory of: `server.py`, `processor.py`, `reconstructor.py`,
`pose_estimator.py`, `auth.py`, `printer.py`, `patch_instantmesh.py`, `check.py`.

Each symbol is assigned a target `trigenix.*` module according to the architecture defined in
`design.md`. A **Shim?** value of `Yes` means the original file retains a thin re-export of the
symbol so existing callers keep working during incremental migration; `No` means the symbol is
moved cleanly with no backwards-compatibility shim needed.

---

## Symbol Migration Table
/config.py` | Rename `MAX_UPLOAD_BYTES`; sourced from `TRIGENIX_MAX_UPLOAD_MB` env var | No |
### `server.py`

| Symbol | Current File | Target Module | Notes | Shim? |
|--------|-------------|---------------|-------|-------|
| `MAX_UPLOAD_BYTES` | `server.py` | `trigenix/core
| `MAX_JSON_BYTES` | `server.py` | `trigenix/core/config.py` | Constant; move as-is | No |
| `ALLOWED_IMAGE_EXTS` | `server.py` | `trigenix/core/config.py` | Used by upload validation and input validator | No |
| `JOB_TTL_SECONDS` | `server.py` | `trigenix/core/config.py` | Sourced from `TRIGENIX_JOB_TTL_SECONDS` | No |
| `_recon_jobs` | `server.py` | `trigenix/api/server.py` | In-process job registry; stays in API layer | No |
| `_recon_cancel` | `server.py` | `trigenix/api/server.py` | Cancel-event map; stays in API layer | No |
| `_recon_lock` | `server.py` | `trigenix/api/server.py` | Threading lock; stays in API layer | No |
| `_gpu_lock` | `server.py` | `trigenix/pipeline/orchestrator.py` | Move to orchestrator; orchestrator owns GPU serialisation | No |
| `_gpu_queue` | `server.py` | `trigenix/pipeline/orchestrator.py` | Move to orchestrator | No |
| `_rate_lock` | `server.py` | `trigenix/api/server.py` | Rate-limiting state; stays in API layer | No |
| `_rate_buckets` | `server.py` | `trigenix/api/server.py` | Rate-limiting state; stays in API layer | No |
| `_print_lock` | `server.py` | `trigenix/api/server.py` | Print-job lock; stays in API layer | No |
| `_print_jobs` | `server.py` | `trigenix/api/server.py` | Print-job registry; stays in API layer | No |
| `_now` | `server.py` | `trigenix/api/server.py` | Tiny helper; inline or keep in API utils | No |
| `_prune_recon_jobs` | `server.py` | `trigenix/api/server.py` | Job-TTL pruner; stays in API layer | No |
| `_set_job` | `server.py` | `trigenix/api/server.py` | Job-registry updater; stays in API layer | No |
| `_enter_gpu_queue` | `server.py` | `trigenix/pipeline/orchestrator.py` | Move into orchestrator; rename `_acquire_gpu` | No |
| `_leave_gpu_queue` | `server.py` | `trigenix/pipeline/orchestrator.py` | Move into orchestrator; rename `_release_gpu` | No |
| `_send_completion_email` | `server.py` | `trigenix/api/server.py` | Notification helper; stays in API layer | No |
| `_run_reconstruction` | `server.py` | `trigenix/pipeline/orchestrator.py` | Core of the orchestrator's job-runner thread | No |
| `start_reconstruction` | `server.py` | `trigenix/api/server.py` | API-layer thread launcher; delegates to `Orchestrator.run_job()` | No |
| `_bearer_token` | `server.py` | `trigenix/api/auth.py` | Auth utility; move alongside session validation | No |
| `_read_json` | `server.py` | `trigenix/api/server.py` | HTTP body helper; stays in API layer | No |
| `_client_key` | `server.py` | `trigenix/api/server.py` | Rate-limit key builder; stays in API layer | No |
| `_rate_limited` | `server.py` | `trigenix/api/server.py` | Rate-limit check; stays in API layer | No |
| `_positive_float` | `server.py` | `trigenix/api/server.py` | Input coercion helper; stays in API layer | No |
| `_target_dimensions_from_fields` | `server.py` | `trigenix/api/server.py` | Form-field parser; stays in API layer | No |
| `_json_dumps_compact` | `server.py` | `trigenix/api/server.py` | Utility; inline or keep in API layer | No |
| `_disk_recon_result` | `server.py` | `trigenix/api/server.py` | Disk-result reader for completed jobs; stays in API layer | No |
| `Handler` | `server.py` | `trigenix/api/server.py` | Rename `TriGenixHandler`; all route logic migrates with it | Yes |
| `Handler.log_message` | `server.py` | `trigenix/api/server.py` | Stays on handler class | No |
| `Handler._cors` | `server.py` | `trigenix/api/server.py` | Stays on handler class | No |
| `Handler.do_OPTIONS` | `server.py` | `trigenix/api/server.py` | Route; stays on handler class | No |
| `Handler._current_user` | `server.py` | `trigenix/api/server.py` | Auth helper; delegates to `auth.validate_session` | No |
| `Handler._require_user` | `server.py` | `trigenix/api/server.py` | Auth guard; stays on handler class | No |
| `Handler._owns_out_dir` | `server.py` | `trigenix/api/server.py` | Ownership check; delegates to `auth.user_owns_out_dir` | No |
| `Handler._safe_generated_path` | `server.py` | `trigenix/api/server.py` | Path-traversal guard; stays on handler class | No |
| `Handler.do_GET` | `server.py` | `trigenix/api/server.py` | Routes: `/auth/me`, `/auth/logout`, `/auth/profile`, `/auth/history`, `/printers`, `/print/status`, `/cancel`, `/recon_status`, `/generated/*` | No |
| `Handler.do_POST` | `server.py` | `trigenix/api/server.py` | Routes: `/auth/login`, `/auth/register`, `/auth/profile`, `/auth/history/rename`, `/print/prepare`, `/print/start`, `/print/cancel`, `/upload` | No |
| `Handler.do_DELETE` | `server.py` | `trigenix/api/server.py` | Route: `DELETE /auth/history` | No |
| `Handler._handle_upload` | `server.py` | `trigenix/api/server.py` | Upload logic; calls `process_image` + `start_reconstruction` | No |
| `Handler._json` | `server.py` | `trigenix/api/server.py` | JSON response helper; stays on handler class | No |
| `validate_image_payload` (nested) | `server.py` | `trigenix/vision/segmentation.py` | Extract as module-level function; used for pre-segmentation input validation | No |
| `main` block (`if __name__`) | `server.py` | `trigenix/api/server.py` | Entry point; stays in `server.py` shim after migration | Yes |

### `processor.py`

| Symbol | Current File | Target Module | Notes | Shim? |
|--------|-------------|---------------|-------|-------|
| `DEVICE` | `processor.py` | `trigenix/core/config.py` | Global device selection; move to config | No |
| `DTYPE` | `processor.py` | `trigenix/core/config.py` | Derived from `DEVICE`; move to config | No |
| `IMG_SIZE` | `processor.py` | `trigenix/core/config.py` | Shared with `reconstructor.py`; consolidate in config | No |
| `DEPTH_SIZE` | `processor.py` | `trigenix/core/config.py` | 518 px native resolution for DepthAnything-Large | No |
| `FG_ALPHA_THRESHOLD` | `processor.py` | `trigenix/core/config.py` | Also in `reconstructor.py`; single source of truth | No |
| `BG_DISTANCE_THRESHOLD_MIN` | `processor.py` | `trigenix/core/config.py` | Also in `reconstructor.py`; consolidate | No |
| `BG_DISTANCE_THRESHOLD_MAX` | `processor.py` | `trigenix/core/config.py` | Also in `reconstructor.py`; consolidate | No |
| `BG_DISTANCE_THRESHOLD_DEFAULT` | `processor.py` | `trigenix/core/config.py` | Also in `reconstructor.py`; consolidate | No |
| `KEEP_PROCESSOR_MODELS_WARM` | `processor.py` | `trigenix/core/config.py` | Env var flag; move to config | No |
| `_zero123` | `processor.py` | `trigenix/models/zero123.py` | Module-level singleton; encapsulate in `Zero123Wrapper` class | No |
| `_depth` | `processor.py` | `trigenix/models/depth_anything.py` | Module-level singleton; encapsulate in `DepthAnywhingWrapper` class | No |
| `_rembg_session` | `processor.py` | `trigenix/vision/segmentation.py` | Move alongside `load_rembg`, `_clean_mask` | No |
| `_free_cuda` | `processor.py` | `trigenix/core/config.py` | GPU cleanup utility; move to core; reused by all model wrappers | No |
| `load_zero123` | `processor.py` | `trigenix/models/zero123.py` | Becomes `Zero123Wrapper.load()` | Yes |
| `load_depth` | `processor.py` | `trigenix/models/depth_anything.py` | Becomes `DepthAnythingWrapper.load()` | Yes |
| `load_rembg` | `processor.py` | `trigenix/vision/segmentation.py` | Used only by segmentation; moves with `_rembg_session` | Yes |
| `warmup` | `processor.py` | `trigenix/api/server.py` | Called at server start; call model `load()` methods directly | Yes |
| `release_zero123` | `processor.py` | `trigenix/models/zero123.py` | Becomes `Zero123Wrapper.release()` | Yes |
| `release_depth` | `processor.py` | `trigenix/models/depth_anything.py` | Becomes `DepthAnythingWrapper.release()` | Yes |
| `release_models` | `processor.py` | `trigenix/pipeline/orchestrator.py` | Orchestrator calls `release()` on each wrapper directly | Yes |
| `_clean_mask` | `processor.py` | `trigenix/vision/segmentation.py` | Private helper for rembg post-processing | No |
| `subject_crop` | `processor.py` | `trigenix/vision/segmentation.py` | Subject cropping belongs with segmentation | No |
| `_sharpen_rgb` | `processor.py` | `trigenix/vision/views.py` | Pre-Zero123++ sharpening; a view-synthesis quality step | No |
| `_match_view_color` | `processor.py` | `trigenix/vision/views.py` | Reinhard LAB colour transfer; view post-processing | No |
| `prepare_image` | `processor.py` | `trigenix/vision/segmentation.py` | Calls rembg + `_clean_mask` + `subject_crop`; segmentation concern | Yes |
| `_view_mask_rembg` | `processor.py` | `trigenix/vision/segmentation.py` | Per-view rembg masking; segmentation concern | No |
| `_view_mask_corners` | `processor.py` | `trigenix/vision/segmentation.py` | Fallback corner-based masking; segmentation concern | No |
| `VIEW_POSES` | `processor.py` | `trigenix/vision/views.py` | The canonical 6-view pose list; single source of truth | No |
| `_zero123_tiles` | `processor.py` | `trigenix/vision/views.py` | Grid-to-tiles parser for Zero123++ output | No |
| `generate_views` | `processor.py` | `trigenix/vision/views.py` | Core view synthesis; becomes `ViewSynthesiser.generate()` | Yes |
| `_edge_guided_sharpen` | `processor.py` | `trigenix/vision/depth.py` | Depth post-processing step | No |
| `_colorize_depth` | `processor.py` | `trigenix/vision/depth.py` | Depth visualisation; moves with depth estimation | No |
| `estimate_depths` | `processor.py` | `trigenix/vision/depth.py` | Depth estimation; returns `DepthResult` list | Yes |
| `_prepare_user_views` | `processor.py` | `trigenix/vision/views.py` | Multi-view user upload path; part of view synthesis | No |
| `process_image` | `processor.py` | `trigenix/pipeline/orchestrator.py` | Top-level vision pipeline; split: stages become orchestrator calls to individual modules | Yes |
| `main` block (`if __name__`) | `processor.py` | `processor.py` (shim) | Remove after all callers migrated to orchestrator | No |

### `reconstructor.py`

| Symbol | Current File | Target Module | Notes | Shim? |
|--------|-------------|---------------|-------|-------|
| `CAM_DIST` | `reconstructor.py` | `trigenix/core/config.py` | Shared camera constant; also used in `camera.py` | No |
| `IMG_SIZE` | `reconstructor.py` | `trigenix/core/config.py` | Duplicate of `processor.py`; single source in config | No |
| `FOV_DEG` | `reconstructor.py` | `trigenix/core/config.py` | Zero123++ FOV; move to config | No |
| `GRID_RES` | `reconstructor.py` | `trigenix/core/config.py` | Visual-hull voxel grid resolution | No |
| `WORLD_HALF` | `reconstructor.py` | `trigenix/core/config.py` | Visual-hull world extent | No |
| `TARGET_FACES` | `reconstructor.py` | `trigenix/core/config.py` | Default decimation target | No |
| `SILHOUETTE_KEEP_RATIO` | `reconstructor.py` | `trigenix/core/config.py` | Quality gate threshold | No |
| `TRIPOSR_RESOLUTIONS` | `reconstructor.py` | `trigenix/models/triposr.py` | TripoSR-specific resolution ladder | No |
| `TRIPOSR_CHUNK_SIZE` | `reconstructor.py` | `trigenix/models/triposr.py` | TripoSR renderer chunk size | No |
| `TRIPOSR_CPU_FALLBACK_RESOLUTION` | `reconstructor.py` | `trigenix/models/triposr.py` | TripoSR CPU fallback constant | No |
| `TRIPOSR_TEXTURE_RESOLUTION` | `reconstructor.py` | `trigenix/models/triposr.py` | Baked-texture resolution for TripoSR | No |
| `FG_ALPHA_THRESHOLD` | `reconstructor.py` | `trigenix/core/config.py` | Duplicate; consolidate in config | No |
| `BG_DISTANCE_THRESHOLD_MIN` | `reconstructor.py` | `trigenix/core/config.py` | Duplicate; consolidate in config | No |
| `BG_DISTANCE_THRESHOLD_MAX` | `reconstructor.py` | `trigenix/core/config.py` | Duplicate; consolidate in config | No |
| `BG_DISTANCE_THRESHOLD_DEFAULT` | `reconstructor.py` | `trigenix/core/config.py` | Duplicate; consolidate in config | No |
| `DEFAULT_VIEW_POSES` | `reconstructor.py` | `trigenix/vision/views.py` | Merge with `VIEW_POSES` in `processor.py`; single authoritative list | No |
| `_camera_c2w` | `reconstructor.py` | `trigenix/geometry/camera.py` | **Camera math duplicate**; becomes `make_zero123_pose` + `c2w_matrix` in camera.py | No |
| `_project` | `reconstructor.py` | `trigenix/geometry/camera.py` | **Camera math duplicate**; becomes `project(pts_world, pose)` in camera.py | No |
| `load_from_dir` | `reconstructor.py` | `trigenix/vision/views.py` | View-loading from disk; produces `ViewRecord` list | Yes |
| `_get_mask` | `reconstructor.py` | `trigenix/vision/segmentation.py` | Per-view mask retrieval with fallback chain | No |
| `_load_depth_maps` | `reconstructor.py` | `trigenix/vision/depth.py` | Depth-map loader with quality validation | No |
| `_apply_opening_masks_to_view_masks` | `reconstructor.py` | `trigenix/geometry/refinement.py` | Opening-aware mask application; part of refinement | No |
| `_mesh_is_usable` | `reconstructor.py` | `trigenix/geometry/validation.py` | Quick quality gate on a candidate mesh | No |
| `_improved_visual_hull` | `reconstructor.py` | `trigenix/geometry/reconstruction.py` | Becomes `MultiViewHullBackend.reconstruct()` | Yes |
| `_create_sphere_fallback` | `reconstructor.py` | `trigenix/geometry/reconstruction.py` | Becomes `SphereBackend.reconstruct()`; gate behind `TRIGENIX_ALLOW_SPHERE_FALLBACK=1` | No |
| `_native_vertex_colors` | `reconstructor.py` | `trigenix/geometry/rendering.py` | Vertex-colour extraction helper; part of rendering/texturing | No |
| `_detect_opening_masks` | `reconstructor.py` | `trigenix/geometry/rendering.py` | Opening-cue detection; visual-mesh production step | No |
| `_preserve_visual_openings` | `reconstructor.py` | `trigenix/geometry/rendering.py` | Applies opening cuts to visual mesh; part of render/refinement | No |
| `paint_colors` | `reconstructor.py` | `trigenix/geometry/rendering.py` | Vertex-colour painting from multi-view; core render-back step | Yes |
| `cleanup` | `reconstructor.py` | `trigenix/geometry/refinement.py` | Mesh cleanup pipeline (degenerate removal, decimation, repair, normalise) | Yes |
| `_auto_orient` | `reconstructor.py` | `trigenix/geometry/refinement.py` | PCA-based auto-orient; part of refinement | No |
| `_url_for_path` | `reconstructor.py` | `trigenix/api/server.py` | Path → URL helper; API concern | No |
| `_mesh_dimensions` | `reconstructor.py` | `trigenix/geometry/scaling.py` | Bounding-box dimension extractor; part of scaling | No |
| `_valid_target_dimensions` | `reconstructor.py` | `trigenix/geometry/scaling.py` | Target-dimension validator/normaliser | No |
| `_copy_mesh` | `reconstructor.py` | `trigenix/geometry/refinement.py` | Safe mesh copy preserving visuals | No |
| `_decimate_for_profile` | `reconstructor.py` | `trigenix/geometry/refinement.py` | Profile-specific decimation; part of refinement | No |
| `_smooth_profile` | `reconstructor.py` | `trigenix/geometry/refinement.py` | Profile-specific Taubin smoothing | No |
| `_apply_pose_orientation` | `reconstructor.py` | `trigenix/geometry/refinement.py` | Applies `PoseResult.world_R` to mesh; part of orient-and-normalize | No |
| `_apply_target_dimensions` | `reconstructor.py` | `trigenix/geometry/scaling.py` | Scales mesh to physical mm; becomes `scale_to_target_dimensions()` | No |
| `_orient_and_normalize` | `reconstructor.py` | `trigenix/geometry/refinement.py` | Combines orient, normalize, scale; part of post-processing pipeline | No |
| `_build_quality_report` | `reconstructor.py` | `trigenix/core/quality_report.py` | Becomes `QualityReport` builder; returns typed `PrintabilityResult` + report dict | No |
| `_export_asset_profiles` | `reconstructor.py` | `trigenix/geometry/refinement.py` | Multi-profile GLB/STL export; split: export logic stays, report writing → quality_report.py | No |
| `_instantmesh_available` | `reconstructor.py` | `trigenix/models/instantmesh.py` | Becomes `InstantMeshBackend.is_available()` static method | No |
| `reconstruct_from_dir` | `reconstructor.py` | `trigenix/pipeline/orchestrator.py` | Top-level reconstruction entry; split across orchestrator + backends | Yes |
| `_try_instantmesh` | `reconstructor.py` | `trigenix/models/instantmesh.py` | Becomes `InstantMeshBackend.reconstruct()` | Yes |
| `_try_export_triposr_baked_glb` | `reconstructor.py` | `trigenix/models/triposr.py` | Baked-texture export; part of `TripoSRBackend` | No |
| `_try_triposr` | `reconstructor.py` | `trigenix/models/triposr.py` | Becomes `TripoSRBackend.reconstruct()` | Yes |
| `main` block (`if __name__`) | `reconstructor.py` | `reconstructor.py` (shim) | Remove after all callers use orchestrator | No |

### `pose_estimator.py`

| Symbol | Current File | Target Module | Notes | Shim? |
|--------|-------------|---------------|-------|-------|
| `PoseResult` | `pose_estimator.py` | `trigenix/pipeline/contracts.py` | Move dataclass to contracts; remove field from `pose_estimator.py`; update imports | Yes |
| `_load_mask` | `pose_estimator.py` | `trigenix/vision/pose.py` | Private helper; moves with `estimate_pose` | No |
| `_pca_ellipse` | `pose_estimator.py` | `trigenix/vision/pose.py` | PCA shape analysis helper | No |
| `_sym_lr` | `pose_estimator.py` | `trigenix/vision/pose.py` | Left-right symmetry score helper | No |
| `_tight_bbox` | `pose_estimator.py` | `trigenix/vision/pose.py` | Tight bounding-box helper | No |
| `estimate_pose` | `pose_estimator.py` | `trigenix/vision/pose.py` | Core pose estimator; returns `PoseResult` | Yes |
| `save_pose` | `pose_estimator.py` | `trigenix/vision/pose.py` | Pose serialisation to disk | Yes |
| `load_pose` | `pose_estimator.py` | `trigenix/vision/pose.py` | Pose deserialisation from disk | Yes |
| `main` block (`if __name__`) | `pose_estimator.py` | `pose_estimator.py` (shim) | Remove after migration | No |

### `auth.py`

| Symbol | Current File | Target Module | Notes | Shim? |
|--------|-------------|---------------|-------|-------|
| `DB_PATH` | `auth.py` | `trigenix/core/config.py` | DB path constant; move to config | No |
| `SESSION_TTL` | `auth.py` | `trigenix/core/config.py` | Session lifetime constant | No |
| `_conn` | `auth.py` | `trigenix/api/auth.py` | DB connection factory; stays with auth | No |
| `init_db` | `auth.py` | `trigenix/api/auth.py` | DB initialiser; stays in auth module | Yes |
| `_hash_pw` | `auth.py` | `trigenix/api/auth.py` | scrypt password hashing; stays in auth | No |
| `_check_pw` | `auth.py` | `trigenix/api/auth.py` | Password verification with legacy SHA-256 support | No |
| `_needs_pw_rehash` | `auth.py` | `trigenix/api/auth.py` | Legacy hash detector; stays in auth | No |
| `register_user` | `auth.py` | `trigenix/api/auth.py` | User registration | Yes |
| `login_user` | `auth.py` | `trigenix/api/auth.py` | User login + session creation | Yes |
| `validate_session` | `auth.py` | `trigenix/api/auth.py` | Session token validation | Yes |
| `logout_session` | `auth.py` | `trigenix/api/auth.py` | Session deletion | Yes |
| `get_user_profile` | `auth.py` | `trigenix/api/auth.py` | Profile fetch | Yes |
| `update_user_profile` | `auth.py` | `trigenix/api/auth.py` | Profile update | Yes |
| `get_user_stats` | `auth.py` | `trigenix/api/auth.py` | Aggregate generation stats for a user | Yes |
| `save_history` | `auth.py` | `trigenix/api/auth.py` | Insert new history entry after upload | Yes |
| `update_history_recon` | `auth.py` | `trigenix/api/auth.py` | Update history with reconstruction result | Yes |
| `user_owns_out_dir` | `auth.py` | `trigenix/api/auth.py` | Ownership check by out_dir | Yes |
| `user_owns_generated_path` | `auth.py` | `trigenix/api/auth.py` | Ownership check by URL path | Yes |
| `update_history_print` | `auth.py` | `trigenix/api/auth.py` | Update history with print metadata | Yes |
| `rename_history_entry` | `auth.py` | `trigenix/api/auth.py` | Rename a history entry label | Yes |
| `delete_history_entry` | `auth.py` | `trigenix/api/auth.py` | Delete entry + files; filesystem + DB | Yes |
| `get_history` | `auth.py` | `trigenix/api/auth.py` | Paginated history fetch | Yes |

### `printer.py`

| Symbol | Current File | Target Module | Notes | Shim? |
|--------|-------------|---------------|-------|-------|
| `PREPARED_JOBS` | `printer.py` | `trigenix/print/dispatch.py` | In-memory prepared-job registry; moves to dispatch | No |
| `_configured_printers` | `printer.py` | `trigenix/print/dispatch.py` | Reads `TRIGENIX_PRINTERS` env var; stays in dispatch | No |
| `_http_json` | `printer.py` | `trigenix/print/dispatch.py` | HTTP utility for printer APIs; stays in dispatch | No |
| `list_printers` | `printer.py` | `trigenix/print/dispatch.py` | Printer enumeration + live status | Yes |
| `_printer_by_id` | `printer.py` | `trigenix/print/dispatch.py` | Printer lookup by ID | No |
| `slice_model` | `printer.py` | `trigenix/print/slicing.py` | Slicing with Cura or dry-run fallback; separate concern from dispatch | Yes |
| `_scaled_stl_copy` | `printer.py` | `trigenix/print/preparation.py` | Scale + copy STL; becomes `preparation.scale_mesh()` | No |
| `prepare_job` | `printer.py` | `trigenix/print/preparation.py` | Job preparation (scale, slice, token); becomes `preparation.prepare_job()` | Yes |
| `send_job` | `printer.py` | `trigenix/print/dispatch.py` | Job dispatch to printer (OctoPrint/Moonraker/serial/dry-run) | Yes |
| `get_job_status` | `printer.py` | `trigenix/print/dispatch.py` | Poll printer job status | Yes |
| `cancel_job` | `printer.py` | `trigenix/print/dispatch.py` | Cancel a running print job | Yes |
| `ensure_stl_for_out_dir` | `printer.py` | `trigenix/print/preparation.py` | Locates or creates STL for an out_dir | Yes |

### `patch_instantmesh.py`

| Symbol | Current File | Target Module | Notes | Shim? |
|--------|-------------|---------------|-------|-------|
| *(script body)* | `patch_instantmesh.py` | `trigenix/models/instantmesh.py` | One-shot `torch.xpu` guard patch; apply at `InstantMeshBackend.load()` time instead of as an external script; delete `patch_instantmesh.py` after migration | No |

### `check.py`

| Symbol | Current File | Target Module | Notes | Shim? |
|--------|-------------|---------------|-------|-------|
| *(script body)* | `check.py` | Superseded | Diagnostic one-liner; keep at project root for convenience, not migrated into the package | No |

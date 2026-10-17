import os, json, uuid, mimetypes, threading, html, time, io
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)  # always run from project root so generated/ uploads/ etc. resolve

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_UPLOAD_BYTES = int(os.environ.get("TRIGENIX_MAX_UPLOAD_MB", "15")) * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_GENERATED_EXTS = {".glb", ".stl", ".obj", ".3mf", ".png", ".jpg", ".jpeg", ".webp", ".json"}
ALLOWED_STATIC_EXTS = {".html", ".js", ".css"}
ALLOWED_STATIC_DIRS = {"HTML", "CSS", "JS"}
# Configurable CORS origin allowlist (comma-separated env var)
_raw_origins = os.environ.get("TRIGENIX_ALLOWED_ORIGINS", "http://localhost:8000")
ALLOWED_ORIGINS: set[str] = {o.strip() for o in _raw_origins.split(",") if o.strip()}
# In production set TRIGENIX_ALLOWED_ORIGINS; wildcard "*" is only for dev
CORS_PERMISSIVE = ("*" in ALLOWED_ORIGINS or not ALLOWED_ORIGINS)
JOB_TTL_SECONDS = int(os.environ.get("TRIGENIX_JOB_TTL_SECONDS", str(6 * 60 * 60)))

for ext, mime in [(".png","image/png"),(".jpg","image/jpeg"),
                  (".webp","image/webp"),(".glb","model/gltf-binary"),
                  (".stl","model/stl"),(".obj","model/obj"),
                  (".3mf","model/3mf"),
                  (".json","application/json")]:
    mimetypes.add_type(mime, ext)

# ── Background reconstruction job registry ────────────────────────────────────
_recon_jobs: dict = {}
_recon_cancel: dict = {}
_recon_lock = threading.Lock()
_gpu_lock = threading.Lock()
_gpu_queue: list[str] = []
_rate_lock = threading.Lock()
_rate_buckets: dict = {}
_print_lock = threading.Lock()
_print_jobs: dict = {}


def _now():
    return int(time.time())


def _prune_recon_jobs():
    cutoff = _now() - JOB_TTL_SECONDS
    with _recon_lock:
        for out_dir, job in list(_recon_jobs.items()):
            if job.get("updated", 0) < cutoff and job.get("status") != "running":
                _recon_jobs.pop(out_dir, None)
                _recon_cancel.pop(out_dir, None)


def _set_job(out_dir: str, status: str, result=None, **extra):
    with _recon_lock:
        payload = {"status": status, "result": result or {}, "updated": _now(), **extra}
        _recon_jobs[out_dir] = payload
        return payload


def _enter_gpu_queue(name: str):
    with _recon_lock:
        if name not in _gpu_queue:
            _gpu_queue.append(name)
        for idx, queued in enumerate(_gpu_queue):
            if queued in _recon_jobs:
                _recon_jobs[queued]["queue_position"] = idx + 1
                _recon_jobs[queued]["updated"] = _now()
    _gpu_lock.acquire()
    with _recon_lock:
        if name in _gpu_queue:
            _gpu_queue.remove(name)
        for idx, queued in enumerate(_gpu_queue):
            if queued in _recon_jobs:
                _recon_jobs[queued]["queue_position"] = idx + 1
                _recon_jobs[queued]["updated"] = _now()


def _leave_gpu_queue():
    _gpu_lock.release()


def _send_completion_email(to_addr: str, result: dict):
    if not to_addr:
        return
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASS", "").strip()
    if not smtp_user or not smtp_pass:
        logger.warning("SMTP_USER/SMTP_PASS not set; skipping completion email")
        return

    import smtplib
    from email.message import EmailMessage

    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    from_addr = os.environ.get("SMTP_FROM", smtp_user).strip()
    model_url = result.get("final_glb", "")
    vertices = int(result.get("vertices") or 0)
    faces = int(result.get("faces") or 0)
    method = result.get("method") or "unknown"
    elapsed = result.get("recon_time") or "?"

    msg = EmailMessage()
    msg["Subject"] = "TriGenix model is ready"
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(
        f"Your TriGenix 3D model is ready.\n\n"
        f"Vertices: {vertices:,}\nFaces: {faces:,}\n"
        f"Method: {method}\nProcessing time: {elapsed}s\nModel: {model_url}\n"
    )
    msg.add_alternative(f"""
        <html>
          <body style="font-family:Arial,sans-serif;line-height:1.5;color:#111">
            <h2>TriGenix model is ready</h2>
            <p>Your 3D mesh has finished processing.</p>
            <table cellpadding="8" cellspacing="0" style="border-collapse:collapse">
              <tr><td><b>Vertices</b></td><td>{vertices:,}</td></tr>
              <tr><td><b>Faces</b></td><td>{faces:,}</td></tr>
              <tr><td><b>Method</b></td><td>{html.escape(str(method))}</td></tr>
              <tr><td><b>Processing time</b></td><td>{html.escape(str(elapsed))}s</td></tr>
            </table>
            <p>Model path: <code>{html.escape(str(model_url))}</code></p>
          </body>
        </html>
    """, subtype="html")

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(smtp_user, smtp_pass)
        smtp.send_message(msg)


def _write_job_log(out_dir: str, message: str):
    """Append a timestamped line to out_dir/job.log. Never raises."""
    try:
        log_path = os.path.join(ROOT, out_dir.strip("/"), "job.log")
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {message}\n")
    except Exception:
        pass


def _run_reconstruction(out_dir: str, user_id=None, cancel_event=None, notify_email="",
                        target_dimensions_mm=None):
    queued_name = out_dir
    try:
        _enter_gpu_queue(queued_name)
        _set_job(out_dir, "running", {}, queue_position=0)
        _write_job_log(out_dir, "queued -> running")
        try:
            from auth import upsert_job as _uj
            _uj(out_dir, status="running", stage="reconstructing")
        except Exception:
            pass
        from reconstructor import reconstruct_from_dir
        result = reconstruct_from_dir(
            out_dir,
            cancel_event=cancel_event,
            target_dimensions_mm=target_dimensions_mm,
        )
        if cancel_event and cancel_event.is_set():
            result = {"status": "cancelled", "message": "Cancelled"}
        if user_id and result.get("status") == "success":
            try:
                from auth import update_history_recon
                update_history_recon(out_dir, result)
            except Exception:
                logger.exception("History reconstruction update failed")

        # ── Quality scoring and confidence mapping ──────────────────────────
        if result.get("status") == "success":
            out_dir_for_quality = out_dir
            try:
                from quality_scorer import compute_scores
                compute_scores(out_dir_for_quality)
                logger.info("  Quality scoring complete")
            except Exception as _qse:
                logger.warning("  Quality scoring failed: %s", _qse)
            try:
                import trimesh as _tm2
                from confidence_mapper import compute_confidence
                _mesh2 = _tm2.load(
                    os.path.join(out_dir_for_quality, "model.glb"), force="mesh"
                )
                compute_confidence(out_dir_for_quality, _mesh2)
                logger.info("  Confidence mapping complete")
            except Exception as _cme:
                logger.warning("  Confidence mapping failed: %s", _cme)
        if result.get("status") == "success" and notify_email:
            try:
                _send_completion_email(notify_email, result)
            except Exception:
                logger.exception("Completion email failed")
        with _recon_lock:
            existing = _recon_jobs.get(out_dir, {})
            if existing.get("status") == "cancelled":
                _recon_cancel.pop(out_dir, None)
                return
            _recon_jobs[out_dir] = {"status": result["status"], "result": result, "updated": _now()}
            _recon_cancel.pop(out_dir, None)
        if result.get("status") == "success":
            _write_job_log(out_dir, f"status=done backend={result.get('method','?')}") 
            try:
                from auth import upsert_job as _uj3
                _uj3(out_dir, status='done', stage='completed', backend=str(result.get('method','')))
            except Exception: pass
        elif result.get("status") in ("cancelled", "error"):
            _write_job_log(out_dir, f"status={result.get('status')} msg={result.get('message','')[:100]}")
            try:
                from auth import upsert_job as _uj3
                _uj3(out_dir, status=result["status"], stage=result["status"],
                     error=str(result.get("message",""))[:500])
            except Exception: pass
    except Exception as e:
        import traceback; traceback.print_exc()
        with _recon_lock:
            _recon_jobs[out_dir] = {"status": "error",
                                    "result": {"status": "error", "message": str(e)},
                                    "updated": _now()}
            _recon_cancel.pop(out_dir, None)
    finally:
        try:
            _leave_gpu_queue()
        except RuntimeError:
            pass


def start_reconstruction(out_dir: str, user_id=None, notify_email="", target_dimensions_mm=None):
    cancel_event = threading.Event()
    with _recon_lock:
        queue_position = len(_gpu_queue) + (1 if _gpu_lock.locked() else 0) + 1
        _recon_jobs[out_dir] = {
            "status": "queued",
            "result": {"queue_position": queue_position},
            "updated": _now(),
            "queue_position": queue_position,
        }
        _recon_cancel[out_dir] = cancel_event
    t = threading.Thread(
        target=_run_reconstruction,
        args=(out_dir, user_id, cancel_event, notify_email, target_dimensions_mm),
        daemon=True,
    )
    t.start()
    _write_job_log(out_dir, "job accepted -> queued")
    try:
        from auth import upsert_job as _uj2
        _uj2(out_dir, user_id=user_id, status="queued", stage="queued")
    except Exception:
        pass
    logger.info(f"Reconstruction started -> {out_dir}")


# ── HTTP Handler ──────────────────────────────────────────────────────────────

def _bearer_token(handler) -> str:
    auth_header = handler.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    try:
        qs = parse_qs(urlparse(handler.path).query)
        return qs.get("token", qs.get("access_token", [""]))[0].strip()
    except Exception:
        return ""
    return ""


def _read_json(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0) or 0)
    if length <= 0:
        return {}
    if length > MAX_JSON_BYTES:
        raise ValueError("JSON body too large")
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8") or "{}")


def _client_key(handler, scope: str) -> str:
    forwarded = handler.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",", 1)[0].strip() if forwarded else handler.client_address[0]
    return f"{scope}:{ip}"


def _rate_limited(handler, scope: str, limit: int = 8, window_seconds: int = 60) -> bool:
    key = _client_key(handler, scope)
    now = time.time()
    with _rate_lock:
        hits = [t for t in _rate_buckets.get(key, []) if now - t < window_seconds]
        if len(hits) >= limit:
            _rate_buckets[key] = hits
            return True
        hits.append(now)
        _rate_buckets[key] = hits
        return False


def _positive_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if value > 0 else None


def _target_dimensions_from_fields(fields: dict) -> dict:
    dims = {}
    for key, out_key in (
        ("target_width", "width"),
        ("target_height", "height"),
        ("target_depth", "depth"),
        ("width_mm", "width"),
        ("height_mm", "height"),
        ("depth_mm", "depth"),
    ):
        value = _positive_float(fields.get(key))
        if value is not None:
            dims[out_key] = value
    return dims


def _json_dumps_compact(data) -> str:
    return json.dumps(data, separators=(",", ":"))


def _disk_recon_result(out_dir: str) -> dict | None:
    safe_dir = os.path.abspath(os.path.normpath(os.path.join(ROOT, (out_dir or "").strip("/"))))
    generated_root = os.path.abspath(os.path.join(ROOT, "generated"))
    if not (safe_dir == generated_root or safe_dir.startswith(generated_root + os.sep)):
        return None

    model_path = os.path.join(safe_dir, "model.glb")
    if not os.path.exists(model_path):
        return None

    result = {
        "status": "success",
        "final_glb": "/" + os.path.relpath(model_path, ROOT).replace("\\", "/"),
    }
    report_path = os.path.join(safe_dir, "quality_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            quality = report.get("quality") or {}
            profiles = report.get("profiles") or {}
            print_profile = profiles.get("print_stl") or profiles.get("print_glb") or {}
            print_watertight = bool(
                (quality.get("print_mesh") or {}).get("watertight") or
                print_profile.get("watertight") or
                quality.get("watertight")
            )
            result.update({
                "method": quality.get("method"),
                "vertices": quality.get("vertices"),
                "faces": quality.get("faces"),
                "watertight": bool(quality.get("print_ready") or print_watertight),
                "visual_watertight": bool((quality.get("visual_mesh") or {}).get("watertight", quality.get("watertight"))),
                "print_watertight": print_watertight,
                "print_ready": bool(quality.get("print_ready") or print_watertight),
                "dimensions": quality.get("dimensions") or {},
                "units": quality.get("units") or "normalized",
                "asset_profiles": profiles,
                "quality": quality,
            })
            result["final_stl"] = (profiles.get("print_stl") or {}).get("url")
        except Exception:
            logger.exception("Failed to read quality report for %s", out_dir)

    if not result.get("vertices"):
        try:
            import trimesh
            mesh = trimesh.load(model_path, force="mesh")
            result.update({
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
                "visual_watertight": bool(mesh.is_watertight),
                "print_watertight": bool(mesh.is_watertight),
                "print_ready": bool(mesh.is_watertight and len(mesh.faces) >= 1000),
                "dimensions": {
                    "width": float(mesh.bounds[1][0] - mesh.bounds[0][0]),
                    "height": float(mesh.bounds[1][1] - mesh.bounds[0][1]),
                    "depth": float(mesh.bounds[1][2] - mesh.bounds[0][2]),
                },
                "units": "normalized",
            })
        except Exception:
            logger.exception("Failed to inspect disk model for %s", out_dir)
    return result


class Handler(SimpleHTTPRequestHandler):

    def log_message(self, fmt, *args):
        if args[1] not in ("200", "304"):
            logger.info(f"{self.address_string()} - {fmt % args}")

    def _cors(self):
        # Use configurable allowlist; fall back to wildcard only if explicitly set
        origin = self.headers.get("Origin", "")
        if CORS_PERMISSIVE or origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin or "*")
        else:
            self.send_header("Access-Control-Allow-Origin", "")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Vary", "Origin")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def _current_user(self):
        try:
            from auth import validate_session
            return validate_session(_bearer_token(self))
        except Exception:
            logger.exception("Auth validation failed")
            return None

    def _require_user(self):
        user = self._current_user()
        if not user:
            self._json({"ok": False, "error": "Unauthorized"}, code=401)
        return user

    def _owns_out_dir(self, user, out_dir: str) -> bool:
        try:
            from auth import user_owns_out_dir
            return bool(user and user_owns_out_dir(user["id"], out_dir))
        except Exception:
            logger.exception("Ownership check failed")
            return False

    def _safe_generated_path(self, request_path: str) -> str | None:
        rel = os.path.normpath(unquote(request_path).lstrip("/").replace("/", os.sep))
        if rel.startswith(".." + os.sep) or rel == ".." or os.path.isabs(rel):
            return None
        full = os.path.abspath(os.path.normpath(os.path.join(ROOT, rel)))
        generated_root = os.path.abspath(os.path.join(ROOT, "generated"))
        if not (full == generated_root or full.startswith(generated_root + os.sep)):
            return None
        return full

    def _safe_static_path(self, request_path: str) -> str | None:
        rel_url = unquote(request_path).lstrip("/")
        rel = os.path.normpath(rel_url.replace("/", os.sep))
        if rel.startswith(".." + os.sep) or rel == ".." or os.path.isabs(rel):
            return None

        first = rel.split(os.sep, 1)[0]
        ext = os.path.splitext(rel)[1].lower()
        if first not in ALLOWED_STATIC_DIRS or ext not in ALLOWED_STATIC_EXTS:
            return None

        full = os.path.abspath(os.path.join(ROOT, rel))
        root_abs = os.path.abspath(ROOT)
        if not (full == root_abs or full.startswith(root_abs + os.sep)):
            return None
        if not os.path.isfile(full):
            return None
        return full

    def _send_not_found(self):
        self.send_response(404)
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(b"Not found")
        except (ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def do_GET(self):
        _prune_recon_jobs()
        parsed = urlparse(self.path)
        path   = parsed.path

        # Redirect bare root to trigenix home page
        if path in ("/", "/index.html", "/trigenix.html"):
            self.send_response(302)
            self.send_header("Location", "/HTML/trigenix.html")
            self._cors()
            self.end_headers()
            return

        # ── Reconstruction status poll ─────────────────────────────────────
        if path == "/auth/me":
            user = self._current_user()
            if not user:
                self._json({"ok": False}, code=401)
                return
            self._json({"ok": True, "username": user["username"], "email": user["email"]})
            return

        if path == "/auth/logout":
            from auth import logout_session
            token = _bearer_token(self)
            if token:
                logout_session(token)
            self._json({"ok": True})
            return

        if path == "/auth/profile":
            user = self._require_user()
            if not user:
                return
            from auth import get_user_profile, get_user_stats
            self._json({
                "ok": True,
                "profile": get_user_profile(user["id"]),
                "stats": get_user_stats(user["id"]),
            })
            return

        if path == "/auth/history":
            user = self._require_user()
            if not user:
                return
            from auth import get_history
            self._json({"ok": True, "history": get_history(user["id"])})
            return

        if path == "/printers":
            user = self._require_user()
            if not user:
                return
            try:
                from printer import list_printers
                self._json({"ok": True, "printers": list_printers()})
            except Exception as e:
                logger.exception("Printer listing failed")
                self._json({"ok": False, "error": str(e)}, code=500)
            return

        if path == "/print/status":
            user = self._require_user()
            if not user:
                return
            qs = parse_qs(parsed.query)
            job_id = qs.get("job_id", [""])[0]
            with _print_lock:
                job = _print_jobs.get(job_id)
            if not job or job.get("user_id") != user["id"]:
                self._json({"ok": False, "error": "Not found"}, code=404)
                return
            try:
                from printer import get_job_status
                self._json({"ok": True, "job_id": job_id, "status": get_job_status(job["printer_id"])})
            except Exception as e:
                logger.exception("Print status failed")
                self._json({"ok": False, "error": str(e)}, code=500)
            return

        if path == "/cancel":
            user = self._require_user()
            if not user:
                return
            qs = parse_qs(parsed.query)
            d = qs.get("dir", [""])[0]
            if not self._owns_out_dir(user, d):
                self._json({"ok": False, "error": "Not found"}, code=404)
                return
            with _recon_lock:
                ev = _recon_cancel.get(d)
                if ev:
                    ev.set()
                _recon_jobs[d] = {
                    "status": "cancelled",
                    "result": {"status": "cancelled", "message": "Cancelled"},
                }
            self._json({"ok": True, "status": "cancelled"})
            return

        if path == "/recon_status":
            user = self._require_user()
            if not user:
                return
            qs  = parse_qs(parsed.query)
            d   = qs.get("dir", [""])[0]
            if not self._owns_out_dir(user, d):
                self._json({"ok": False, "error": "Not found"}, code=404)
                return
            with _recon_lock:
                job = _recon_jobs.get(d)
            if not job:
                disk_result = _disk_recon_result(d)
                if disk_result:
                    job = {"status": "done", "result": disk_result, "queue_position": 0}
                else:
                    job = {"status": "running", "result": {}}
            self._json({
                "status": job["status"],
                "queue_position": job.get("queue_position", job.get("result", {}).get("queue_position", 0)),
                **job["result"],
            })
            return

        # ── Serve /generated/ files ────────────────────────────────────────
        if path.startswith("/generated/"):
            user = self._require_user()
            if not user:
                return
            fpath = self._safe_generated_path(path)
            ext = os.path.splitext(fpath or "")[1].lower()
            if not fpath or ext not in ALLOWED_GENERATED_EXTS or not os.path.isfile(fpath):
                self._send_not_found()
                return
            normalized_path = "/" + os.path.relpath(fpath, ROOT).replace(os.sep, "/")
            try:
                from auth import user_owns_generated_path
                if not user_owns_generated_path(user["id"], normalized_path):
                    self._json({"ok": False, "error": "Not found"}, code=404)
                    return
            except Exception:
                logger.exception("Generated ownership check failed")
                self._json({"ok": False, "error": "Not found"}, code=404)
                return

            mime = mimetypes.guess_type(fpath)[0] or "application/octet-stream"
            with open(fpath, "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("Content-Type",   mime)
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            try:
                self.wfile.write(data)
            except (ConnectionAbortedError, BrokenPipeError, OSError):
                pass
            return

        # Serve HTML/JS/CSS with no-cache so browser never uses stale code
        _fp = self._safe_static_path(path)
        if _fp:
            with open(_fp, 'rb') as _fh:
                _data = _fh.read()
            _mime = mimetypes.guess_type(_fp)[0] or 'text/plain'
            self.send_response(200)
            self.send_header('Content-Type',   _mime)
            self.send_header('Content-Length', str(len(_data)))
            self.send_header('Cache-Control',  'no-cache, no-store, must-revalidate')
            self.send_header('Pragma',         'no-cache')
            self.send_header('Expires',        '0')
            self._cors()
            self.end_headers()
            try:
                self.wfile.write(_data)
            except (ConnectionAbortedError, BrokenPipeError, OSError):
                pass
            return


        # ── Public share routes (/share/<token> and /share/<token>/model) ──
        if path.startswith("/share/"):
            parts = path.split("/")  # ["", "share", token, ?model]
            token = parts[2] if len(parts) > 2 else ""
            if not token:
                self._send_not_found(); return
            try:
                from auth import get_share, get_history_entry_for_share
                share = get_share(token)
            except Exception:
                share = None
            if not share:
                self._send_not_found(); return

            # /share/<token>/model  -> stream the GLB file (no auth required)
            if len(parts) > 3 and parts[3] == "model":
                try:
                    import sqlite3 as _sq3
                    conn = _sq3.connect("trigenix.db"); conn.row_factory = _sq3.Row
                    row = conn.execute(
                        "SELECT final_glb FROM history WHERE id = ?", (share["hist_id"],)
                    ).fetchone()
                    conn.close()
                    glb_url = row["final_glb"] if row else None
                    if not glb_url:
                        self._send_not_found(); return
                    fpath = self._safe_generated_path(glb_url)
                    if not fpath or not os.path.isfile(fpath):
                        self._send_not_found(); return
                    with open(fpath, "rb") as fh:
                        data = fh.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "model/gltf-binary")
                    self.send_header("Content-Length", str(len(data)))
                    self._cors(); self.end_headers()
                    self.wfile.write(data)
                except (ConnectionAbortedError, BrokenPipeError, OSError):
                    pass
                except Exception:
                    logger.exception("Share model serve failed")
                    self._send_not_found()
                return

            # /share/<token>  -> serve Viewer3d.html without auth
            fp = self._safe_static_path("/HTML/Viewer3d.html")
            if not fp:
                self._send_not_found(); return
            with open(fp, "rb") as fh:
                data = fh.read()
            # Inject share token into the viewer URL so it loads the shared model
            viewer_url = f"/share/{token}/model"
            data = data.replace(
                b"var _authToken=localStorage.getItem('tg_token')||'';",
                f"var _authToken=''; var _sharedModelUrl='{viewer_url}';".encode()
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self._cors(); self.end_headers()
            try:
                self.wfile.write(data)
            except (ConnectionAbortedError, BrokenPipeError, OSError):
                pass
            return

        self._send_not_found()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/auth/login":
            if _rate_limited(self, "login"):
                self._json({"ok": False, "error": "Too many attempts"}, code=429)
                return
            try:
                from auth import login_user
                data = _read_json(self)
                self._json(login_user(data.get("username", ""), data.get("password", "")))
            except Exception as e:
                logger.error(f"Login error: {e}")
                self._json({"ok": False, "error": str(e)}, code=500)
            return

        if path == "/auth/register":
            if _rate_limited(self, "register", limit=4, window_seconds=300):
                self._json({"ok": False, "error": "Too many attempts"}, code=429)
                return
            try:
                from auth import register_user
                data = _read_json(self)
                self._json(register_user(
                    data.get("username", ""),
                    data.get("email", ""),
                    data.get("password", ""),
                ))
            except Exception as e:
                logger.error(f"Register error: {e}")
                self._json({"ok": False, "error": str(e)}, code=500)
            return

        if path == "/auth/profile":
            user = self._require_user()
            if not user:
                return
            try:
                from auth import update_user_profile
                self._json(update_user_profile(user["id"], _read_json(self)))
            except Exception as e:
                logger.error(f"Profile update error: {e}")
                self._json({"ok": False, "error": str(e)}, code=500)
            return

        if path == "/auth/history/rename":
            user = self._require_user()
            if not user:
                return
            try:
                from auth import rename_history_entry
                data = _read_json(self)
                self._json(rename_history_entry(
                    user["id"],
                    int(data.get("id", 0)),
                    data.get("label", ""),
                ))
            except Exception as e:
                logger.error(f"History rename error: {e}")
                self._json({"ok": False, "error": str(e)}, code=500)
            return

        if path == "/print/prepare":
            user = self._require_user()
            if not user:
                return
            try:
                data = _read_json(self)
                out_dir = (data.get("out_dir") or "").strip().strip("/").replace("\\", "/")
                if not self._owns_out_dir(user, out_dir):
                    self._json({"ok": False, "error": "Not found"}, code=404)
                    return
                from printer import ensure_stl_for_out_dir, prepare_job
                stl_path = ensure_stl_for_out_dir(out_dir)
                prepared = prepare_job(
                    stl_path,
                    data.get("printer_id", ""),
                    scale_mm=data.get("scale_mm") or {},
                    orientation=data.get("orientation") or {},
                    settings=data.get("settings") or {},
                )
                prepared["user_id"] = user["id"]
                prepared["out_dir"] = out_dir
                with _print_lock:
                    _print_jobs[prepared["token"]] = prepared
                from auth import update_history_print
                update_history_print(
                    user["id"],
                    out_dir,
                    stl_path="/" + stl_path.replace("\\", "/"),
                    print_scale_mm=_json_dumps_compact(prepared.get("scale_mm") or {}),
                    printer_id=prepared.get("printer_id", ""),
                    print_status="prepared",
                )
                public = {k: v for k, v in prepared.items() if k != "user_id"}
                self._json({"ok": True, "job": public, "job_token": prepared["token"]})
            except Exception as e:
                logger.exception("Print prepare failed")
                self._json({"ok": False, "error": str(e)}, code=500)
            return

        if path == "/print/start":
            user = self._require_user()
            if not user:
                return
            try:
                data = _read_json(self)
                token = data.get("job_token", "")
                with _print_lock:
                    job = _print_jobs.get(token)
                if not job or job.get("user_id") != user["id"]:
                    self._json({"ok": False, "error": "Not found"}, code=404)
                    return
                from printer import send_job
                started = send_job(token)
                job_id = started.get("job_id") or token
                with _print_lock:
                    job["print_job_id"] = job_id
                    job["print_status"] = started.get("status", "started")
                from auth import update_history_print
                update_history_print(
                    user["id"],
                    job["out_dir"],
                    print_job_id=job_id,
                    print_status=started.get("status", "started"),
                )
                self._json({"ok": True, **started})
            except Exception as e:
                logger.exception("Print start failed")
                self._json({"ok": False, "error": str(e)}, code=500)
            return

        if path == "/print/cancel":
            user = self._require_user()
            if not user:
                return
            try:
                data = _read_json(self)
                job_id = data.get("job_id") or data.get("job_token", "")
                with _print_lock:
                    job = _print_jobs.get(job_id)
                if not job or job.get("user_id") != user["id"]:
                    self._json({"ok": False, "error": "Not found"}, code=404)
                    return
                from printer import cancel_job
                result = cancel_job(job["printer_id"])
                from auth import update_history_print
                update_history_print(user["id"], job["out_dir"], print_status="cancelled")
                self._json({"ok": True, "status": "cancelled", "result": result})
            except Exception as e:
                logger.exception("Print cancel failed")
                self._json({"ok": False, "error": str(e)}, code=500)
            return


        if path == "/share/create":
            user = self._require_user()
            if not user: return
            try:
                data = _read_json(self)
                hist_id = int(data.get("hist_id", 0))
                expires = data.get("expires")  # optional unix timestamp
                if not hist_id:
                    self._json({"ok": False, "error": "hist_id required"}, code=400); return
                # Verify user owns that history entry
                import sqlite3 as _sq3
                conn = _sq3.connect("trigenix.db"); conn.row_factory = _sq3.Row
                row = conn.execute(
                    "SELECT id FROM history WHERE id=? AND user_id=?", (hist_id, user["id"])
                ).fetchone(); conn.close()
                if not row:
                    self._json({"ok": False, "error": "Not found"}, code=404); return
                from auth import create_share
                result = create_share(user["id"], hist_id, expires)
                self._json({"ok": True, **result})
            except Exception as e:
                logger.exception("Share create failed")
                self._json({"ok": False, "error": "Share creation failed"}, code=500)
            return

        if path == "/share/revoke":
            user = self._require_user()
            if not user: return
            try:
                data = _read_json(self)
                token = (data.get("token") or "").strip()
                if not token:
                    self._json({"ok": False, "error": "token required"}, code=400); return
                from auth import revoke_share
                ok = revoke_share(user["id"], token)
                if ok:
                    self._json({"ok": True})
                else:
                    self._json({"ok": False, "error": "Not found"}, code=404)
            except Exception as e:
                logger.exception("Share revoke failed")
                self._json({"ok": False, "error": "Revoke failed"}, code=500)
            return

        if path == "/recon_retry":
            user = self._require_user()
            if not user: return
            try:
                data = _read_json(self)
                out_dir = (data.get("out_dir") or "").strip().strip("/").replace("\\", "/")
                if not out_dir:
                    self._json({"ok": False, "error": "out_dir required"}, code=400); return
                if not self._owns_out_dir(user, out_dir):
                    self._json({"ok": False, "error": "Not found"}, code=404); return
                with _recon_lock:
                    current = _recon_jobs.get(out_dir, {})
                if current.get("status") == "running":
                    self._json({"ok": False, "error": "Job is already running"}, code=409); return
                out_path = os.path.join(ROOT, out_dir)
                if not os.path.isdir(out_path):
                    self._json({"ok": False, "error": "Output directory not found"}, code=404); return
                _write_job_log(out_dir, f"retry requested by user {user['id']}")
                start_reconstruction(out_dir, user_id=user["id"])
                self._json({"ok": True, "status": "queued", "out_dir": out_dir})
            except Exception as e:
                logger.exception("Recon retry failed")
                self._json({"ok": False, "error": "Retry failed"}, code=500)
            return

        if path == "/admin/cleanup/status":
            user = self._require_user()
            if not user: return
            try:
                from cleanup_worker import get_status
                self._json(get_status())
            except ImportError:
                self._json({"ok": True, "last_run": None, "deleted_count": 0,
                            "note": "cleanup_worker not available"})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, code=500)
            return


        if path == "/mesh/thickness":
            user = self._require_user()
            if not user: return
            try:
                data = _read_json(self)
                out_dir = (data.get("out_dir") or "").strip().strip("/").replace("\\", "/")
                min_mm  = float(data.get("min_thickness_mm", 1.5))
                if min_mm <= 0:
                    self._json({"ok": False, "error": "min_thickness_mm must be positive"}, code=400); return
                if not self._owns_out_dir(user, out_dir):
                    self._json({"ok": False, "error": "Not found"}, code=404); return
                from thickness_checker import run_thickness_check
                report = run_thickness_check(out_dir, min_thickness_mm=min_mm)
                self._json({"ok": True, "thin_fraction": report["thin_fraction"],
                            "thin_vertex_count": report["thin_vertex_count"]})
            except ValueError as e:
                self._json({"ok": False, "error": str(e)}, code=422)
            except Exception as e:
                logger.exception("Thickness check failed")
                self._json({"ok": False, "error": "Thickness check failed"}, code=500)
            return

        if path == "/mesh/hollow":
            user = self._require_user()
            if not user: return
            try:
                data = _read_json(self)
                out_dir      = (data.get("out_dir") or "").strip().strip("/").replace("\\", "/")
                wall_mm      = float(data.get("wall_mm", 3.0))
                drain_dia_mm = float(data.get("drain_diameter_mm", 4.0))
                if not self._owns_out_dir(user, out_dir):
                    self._json({"ok": False, "error": "Not found"}, code=404); return
                from hollowing_service import run_hollow
                result = run_hollow(out_dir, wall_mm=wall_mm, drain_diameter_mm=drain_dia_mm)
                self._json(result)
            except ValueError as e:
                self._json({"ok": False, "error": str(e)}, code=422)
            except Exception as e:
                logger.exception("Hollow failed")
                self._json({"ok": False, "error": "Hollowing failed"}, code=500)
            return

        if path == "/mesh/split":
            user = self._require_user()
            if not user: return
            try:
                data = _read_json(self)
                out_dir   = (data.get("out_dir") or "").strip().strip("/").replace("\\", "/")
                bed_w     = float(data.get("bed_width_mm", 0))
                bed_d     = float(data.get("bed_depth_mm", 0))
                conn_type = data.get("connector_type", "pin")
                if bed_w <= 0 or bed_d <= 0:
                    self._json({"ok": False, "error": "bed_width_mm and bed_depth_mm required"}, code=400); return
                if not self._owns_out_dir(user, out_dir):
                    self._json({"ok": False, "error": "Not found"}, code=404); return
                from splitter_service import run_split
                result = run_split(out_dir, bed_w, bed_d, connector_type=conn_type)
                self._json(result)
            except ValueError as e:
                self._json({"ok": False, "error": str(e)}, code=400)
            except Exception as e:
                logger.exception("Split failed")
                self._json({"ok": False, "error": "Split failed"}, code=500)
            return

        if path == "/mesh/multicolor":
            user = self._require_user()
            if not user: return
            try:
                data    = _read_json(self)
                out_dir = (data.get("out_dir") or "").strip().strip("/").replace("\\", "/")
                fmt     = data.get("format", "3mf")
                if fmt not in ("3mf", "stl_per_region"):
                    self._json({"ok": False, "error": "format must be 3mf or stl_per_region"}, code=400); return
                if not self._owns_out_dir(user, out_dir):
                    self._json({"ok": False, "error": "Not found"}, code=404); return
                from multicolor_exporter import run_multicolor_export
                result = run_multicolor_export(out_dir, fmt=fmt)
                self._json(result)
            except ValueError as e:
                self._json({"ok": False, "error": str(e)}, code=422)
            except Exception as e:
                logger.exception("Multicolor export failed")
                self._json({"ok": False, "error": "Multicolor export failed"}, code=500)
            return

        if path == "/upload":
            self._handle_upload()
        else:
            self.send_response(404); self.end_headers()

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/auth/history":
            user = self._require_user()
            if not user:
                return
            try:
                from auth import delete_history_entry
                data = _read_json(self)
                self._json(delete_history_entry(user["id"], int(data.get("id", 0))))
            except Exception as e:
                logger.error(f"History delete error: {e}")
                self._json({"ok": False, "error": str(e)}, code=500)
            return

        self.send_response(404)
        self._cors()
        self.end_headers()

    def _handle_upload(self):
        try:
            user = self._require_user()
            if not user:
                return

            import email.parser as ep, email.policy as epol
            content_type = self.headers.get("Content-Type", "")
            if "boundary=" not in content_type:
                raise ValueError("No multipart boundary in Content-Type")
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                raise ValueError("Empty upload")
            if length > MAX_UPLOAD_BYTES:
                raise ValueError(f"Upload too large; max is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            body   = self.rfile.read(length)
            raw    = (f"Content-Type: {content_type}\r\n\r\n").encode() + body
            msg    = ep.BytesParser(policy=epol.compat32).parsebytes(raw)
            # find the image part
            item_filename = item_data = None
            user_view_files = {}
            form_fields = {}
            for part in msg.get_payload():
                disp = part.get("Content-Disposition", "")
                name = filename = None
                for tok in disp.split(";"):
                    tok = tok.strip()
                    if tok.startswith("name="):     name     = tok[5:].strip('"')
                    elif tok.startswith("filename="): filename = tok[9:].strip('"')
                if name == "image" and filename:
                    item_filename = filename
                    item_data     = part.get_payload(decode=True) or b""
                elif name and name.startswith("view_") and filename:
                    user_view_files[name] = {
                        "filename": filename,
                        "data": part.get_payload(decode=True) or b"",
                    }
                elif name:
                    raw_value = part.get_payload(decode=True) or b""
                    form_fields[name] = raw_value.decode("utf-8", errors="ignore").strip()
            if not item_filename:
                raise ValueError("No 'image' field")
            from PIL import Image

            def validate_image_payload(filename_in, data_in):
                if len(data_in) > MAX_UPLOAD_BYTES:
                    raise ValueError(f"Image too large; max is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                ext_in = os.path.splitext(filename_in)[1].lower() or ".jpg"
                if ext_in not in ALLOWED_IMAGE_EXTS:
                    raise ValueError("Unsupported image type")
                try:
                    with Image.open(io.BytesIO(data_in)) as img:
                        img.verify()
                    with Image.open(io.BytesIO(data_in)) as img:
                        detected = (img.format or "").lower()
                    if detected not in {"png", "jpeg", "webp"}:
                        raise ValueError("Unsupported image content")
                except Exception as e:
                    raise ValueError(f"Invalid image upload: {e}")
                return ext_in

            ext = validate_image_payload(item_filename, item_data)

            filename = str(uuid.uuid4()) + ext
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, "wb") as f:
                f.write(item_data)
            logger.info(f"Saved → {filepath}")

            user_view_paths = []
            if user_view_files:
                expected_names = [f"view_{i}" for i in range(6)]
                if any(name not in user_view_files for name in expected_names):
                    raise ValueError("Multi-view upload requires view_0 through view_5")
                view_dir = os.path.join(UPLOAD_FOLDER, os.path.splitext(filename)[0] + "_views")
                os.makedirs(view_dir, exist_ok=True)
                for field_name in expected_names:
                    vf = user_view_files[field_name]
                    vext = validate_image_payload(vf["filename"], vf["data"])
                    vpath = os.path.join(view_dir, f"{field_name}{vext}")
                    with open(vpath, "wb") as vf_out:
                        vf_out.write(vf["data"])
                    user_view_paths.append(vpath)

            target_dimensions_mm = _target_dimensions_from_fields(form_fields)

            # Step 1: views + depths (runs synchronously)
            proc_result = {}
            err = {}
            def run_proc():
                try:
                    from processor import process_image
                    proc_result["r"] = process_image(
                        filepath,
                        user_view_paths=user_view_paths or None,
                        target_dimensions_mm=target_dimensions_mm,
                    )
                except Exception as e:
                    err["e"] = e

            t = threading.Thread(target=run_proc)
            _enter_gpu_queue(f"upload:{filename}")
            try:
                t.start(); t.join()
            finally:
                _leave_gpu_queue()

            if err: raise err["e"]

            result = proc_result["r"]
            if result.get("status") != "success":
                self._json(result); return

            try:
                from auth import save_history
                save_history(user["id"], "/" + filepath.replace("\\", "/"), result)
            except Exception:
                logger.exception("History save failed")

            # Step 2: reconstruction in background
            out_dir = result.get("out_dir", "")
            if out_dir and os.path.isdir(out_dir):
                notify_email = form_fields.get("notify_email", "")
                start_reconstruction(
                    out_dir,
                    user_id=user["id"],
                    notify_email=notify_email,
                    target_dimensions_mm=target_dimensions_mm,
                )
                result["recon_status"] = "running"
                result["recon_dir"]    = out_dir

            self._json(result)

        except Exception as e:
            logger.error(f"Upload error: {e}")
            import traceback; traceback.print_exc()
            self._json({"status": "error", "message": str(e)}, code=500)

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, OSError):
            # WinError 10053 — browser closed the connection, safe to ignore
            pass


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))

    from auth import init_db, mark_interrupted_jobs
    init_db()
    mark_interrupted_jobs()

    if os.environ.get("TRIGENIX_SKIP_MODEL_WARMUP", "").lower() in ("1", "true", "yes"):
        print("Skipping model warmup (TRIGENIX_SKIP_MODEL_WARMUP=1)")
    else:
        print("Warming up models …")
        from processor import warmup
        warmup()

    server = ThreadingHTTPServer(("", PORT), Handler)
    print("=" * 55)
    print(f"  Home     →  http://localhost:{PORT}/HTML/trigenix.html")
    print(f"  Generate →  http://localhost:{PORT}/HTML/DropImage.html")
    print(f"  Viewer   →  http://localhost:{PORT}/HTML/Viewer3d.html")
    print(f"  Print    →  http://localhost:{PORT}/HTML/Print3D.html")
    print("=" * 55)
    # Start cleanup daemon (deletes stale generated/ dirs)
    try:
        from cleanup_worker import start_cleanup_daemon
        _max_age = int(os.environ.get("TRIGENIX_CLEANUP_MAX_AGE_DAYS", "30"))
        if _max_age > 0:
            start_cleanup_daemon(os.path.join(ROOT, "generated"), _max_age)
    except Exception as _e:
        logger.warning("Cleanup daemon failed to start: %s", _e)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped"); server.server_close()

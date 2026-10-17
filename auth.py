"""
auth.py  —  TriGenix authentication & history
Uses SQLite (zero setup, file-based). Sessions last 5 hours.
"""

import sqlite3, hashlib, os, secrets, time, json, shutil, threading
from datetime import datetime

DB_PATH      = "trigenix.db"
SESSION_TTL  = 5 * 60 * 60   # 5 hours in seconds
_jobs_lock = threading.Lock()


# ── DB init ──────────────────────────────────────────────────────────────────

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Create tables if they don't exist. Call once at server start."""
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                email       TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                pw_hash     TEXT    NOT NULL,
                created     INTEGER NOT NULL,
                display_name TEXT   DEFAULT '',
                bio          TEXT   DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token     TEXT    PRIMARY KEY,
                user_id   INTEGER NOT NULL,
                created   INTEGER NOT NULL,
                expires   INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                created      INTEGER NOT NULL,
                input_image  TEXT,
                masked_image TEXT,
                views        TEXT,
                depths_grey  TEXT,
                depths_color TEXT,
                final_glb    TEXT,
                out_dir      TEXT,
                proc_time    REAL,
                vertices     INTEGER,
                faces        INTEGER,
                method       TEXT,
                watertight   INTEGER,
                label        TEXT    DEFAULT '',
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS shares (
                token    TEXT    PRIMARY KEY,
                user_id  INTEGER NOT NULL,
                hist_id  INTEGER NOT NULL,
                created  INTEGER NOT NULL,
                expires  INTEGER,
                active   INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(hist_id) REFERENCES history(id)
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER,
                out_dir      TEXT    NOT NULL UNIQUE,
                status       TEXT    NOT NULL DEFAULT 'queued',
                stage        TEXT    DEFAULT '',
                queue_pos    INTEGER DEFAULT 0,
                backend      TEXT    DEFAULT '',
                error        TEXT    DEFAULT '',
                created      INTEGER NOT NULL,
                updated      INTEGER NOT NULL
            );
        """)

        # Migration: add columns if they don't exist (for existing DBs)
        try:
            c.execute("ALTER TABLE users ADD COLUMN display_name TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            c.execute("ALTER TABLE users ADD COLUMN bio TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            c.execute("ALTER TABLE history ADD COLUMN label TEXT DEFAULT ''")
        except Exception:
            pass
        for stmt in (
            "ALTER TABLE history ADD COLUMN stl_path TEXT DEFAULT ''",
            "ALTER TABLE history ADD COLUMN print_scale_mm TEXT DEFAULT ''",
            "ALTER TABLE history ADD COLUMN printer_id TEXT DEFAULT ''",
            "ALTER TABLE history ADD COLUMN print_job_id TEXT DEFAULT ''",
            "ALTER TABLE history ADD COLUMN print_status TEXT DEFAULT ''",
            "ALTER TABLE history ADD COLUMN print_ready INTEGER DEFAULT 0",
            "ALTER TABLE history ADD COLUMN visual_watertight INTEGER DEFAULT 0",
            "ALTER TABLE history ADD COLUMN print_watertight INTEGER DEFAULT 0",
        ):
            try:
                c.execute(stmt)
            except Exception:
                pass
        try:
            c.execute("ALTER TABLE history ADD COLUMN mode TEXT DEFAULT 'single_photo'")
        except Exception:
            pass


# ── Password helpers ─────────────────────────────────────────────────────────

def _hash_pw(password: str) -> str:
    salt = secrets.token_bytes(16)
    n, r, p = 2**14, 8, 1
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${salt.hex()}${dk.hex()}"


def _check_pw(password: str, stored: str) -> bool:
    try:
        if stored.startswith("scrypt$"):
            _, n, r, p, salt_hex, hash_hex = stored.split("$", 5)
            dk = hashlib.scrypt(
                password.encode("utf-8"),
                salt=bytes.fromhex(salt_hex),
                n=int(n),
                r=int(r),
                p=int(p),
                dklen=len(bytes.fromhex(hash_hex)),
            )
            return secrets.compare_digest(dk.hex(), hash_hex)

        # Legacy salted SHA-256 support so existing accounts can still log in.
        salt, h = stored.split(":", 1)
        return secrets.compare_digest(hashlib.sha256((salt + password).encode()).hexdigest(), h)
    except Exception:
        return False


def _needs_pw_rehash(stored: str) -> bool:
    return not (stored or "").startswith("scrypt$")


# ── User management ──────────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str) -> dict:
    if len(username) < 3:
        return {"ok": False, "error": "Username must be at least 3 characters"}
    if len(password) < 6:
        return {"ok": False, "error": "Password must be at least 6 characters"}
    if "@" not in email:
        return {"ok": False, "error": "Invalid email address"}
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users (username, email, pw_hash, created) VALUES (?,?,?,?)",
                (username.strip(), email.strip().lower(), _hash_pw(password), int(time.time()))
            )
        return {"ok": True}
    except sqlite3.IntegrityError as e:
        err = str(e).lower()
        if "username" in err:
            return {"ok": False, "error": "Username already taken"}
        if "email" in err:
            return {"ok": False, "error": "Email already registered"}
        return {"ok": False, "error": "Registration failed"}


def login_user(username: str, password: str) -> dict:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),)
        ).fetchone()
    if not row or not _check_pw(password, row["pw_hash"]):
        return {"ok": False, "error": "Invalid username or password"}

    token   = secrets.token_hex(32)
    now     = int(time.time())
    expires = now + SESSION_TTL
    with _conn() as c:
        if _needs_pw_rehash(row["pw_hash"]):
            c.execute("UPDATE users SET pw_hash = ? WHERE id = ?", (_hash_pw(password), row["id"]))
        c.execute("DELETE FROM sessions WHERE user_id = ? AND expires < ?", (row["id"], now))
        c.execute(
            "INSERT INTO sessions (token, user_id, created, expires) VALUES (?,?,?,?)",
            (token, row["id"], now, expires)
        )
    return {"ok": True, "token": token, "username": row["username"], "expires": expires}


def validate_session(token: str) -> dict | None:
    if not token:
        return None
    now = int(time.time())
    with _conn() as c:
        row = c.execute("""
            SELECT u.id, u.username, u.email, s.expires
            FROM sessions s JOIN users u ON s.user_id = u.id
            WHERE s.token = ? AND s.expires > ?
        """, (token, now)).fetchone()
    if not row:
        return None
    return {"id": row["id"], "username": row["username"], "email": row["email"]}


def logout_session(token: str):
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ── Profile ───────────────────────────────────────────────────────────────────

def get_user_profile(user_id: int) -> dict:
    with _conn() as c:
        row = c.execute(
            "SELECT id, username, email, created, display_name, bio FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
    if not row:
        return {}
    d = dict(row)
    d["created_fmt"] = datetime.fromtimestamp(d["created"]).strftime("%B %Y")
    return d


def update_user_profile(user_id: int, data: dict) -> dict:
    allowed = {"display_name", "bio"}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return {"ok": False, "error": "Nothing to update"}
    cols = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [user_id]
    with _conn() as c:
        c.execute(f"UPDATE users SET {cols} WHERE id = ?", vals)
    return {"ok": True}


def get_user_stats(user_id: int) -> dict:
    with _conn() as c:
        total = c.execute(
            "SELECT COUNT(*) as cnt FROM history WHERE user_id = ?", (user_id,)
        ).fetchone()["cnt"]
        with_glb = c.execute(
            "SELECT COUNT(*) as cnt FROM history WHERE user_id = ? AND final_glb IS NOT NULL AND final_glb != ''",
            (user_id,)
        ).fetchone()["cnt"]
        avg_time = c.execute(
            "SELECT AVG(proc_time) as avg FROM history WHERE user_id = ? AND proc_time IS NOT NULL",
            (user_id,)
        ).fetchone()["avg"]
    return {
        "total_generations": total,
        "with_glb": with_glb,
        "avg_time": round(avg_time or 0, 1),
    }


# ── History ───────────────────────────────────────────────────────────────────

def save_history(user_id: int, input_image: str, proc_result: dict, recon_result: dict = None):
    r    = proc_result or {}
    rec  = recon_result or {}
    with _conn() as c:
        c.execute("""
            INSERT INTO history
              (user_id, created, input_image, masked_image, views,
               depths_grey, depths_color, final_glb, out_dir,
               proc_time, vertices, faces, method, watertight)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            user_id,
            int(time.time()),
            input_image,
            r.get("masked"),
            json.dumps(r.get("views", [])),
            json.dumps(r.get("depths_grey", [])),
            json.dumps(r.get("depths_color", [])),
            rec.get("final_glb") or r.get("final_glb"),
            r.get("out_dir"),
            r.get("time"),
            rec.get("vertices"),
            rec.get("faces"),
            rec.get("method"),
            int(rec.get("watertight", False)),
        ))
        return c.execute("SELECT last_insert_rowid()").fetchone()[0]


def update_history_recon(out_dir: str, recon_result: dict):
    rec = recon_result or {}
    with _conn() as c:
        c.execute("""
            UPDATE history SET
                final_glb  = ?,
                vertices   = ?,
                faces      = ?,
                method     = ?,
                watertight = ?,
                print_ready = ?,
                visual_watertight = ?,
                print_watertight = ?,
                stl_path   = ?
            WHERE out_dir = ?
        """, (
            rec.get("final_glb"),
            rec.get("vertices"),
            rec.get("faces"),
            rec.get("method"),
            int(rec.get("watertight", False)),
            int(rec.get("print_ready", rec.get("watertight", False))),
            int(rec.get("visual_watertight", rec.get("watertight", False))),
            int(rec.get("print_watertight", rec.get("watertight", False))),
            rec.get("final_stl") or (rec.get("asset_profiles", {}).get("print_stl", {}) or {}).get("url"),
            out_dir,
        ))


def user_owns_out_dir(user_id: int, out_dir: str) -> bool:
    out_dir = (out_dir or "").strip().strip("/").replace("\\", "/")
    if not out_dir:
        return False
    with _conn() as c:
        row = c.execute(
            "SELECT id FROM history WHERE user_id = ? AND out_dir = ?",
            (user_id, out_dir),
        ).fetchone()
    return bool(row)


def user_owns_generated_path(user_id: int, generated_path: str) -> bool:
    p = (generated_path or "").strip().strip("/").replace("\\", "/")
    parts = p.split("/")
    if len(parts) < 2 or parts[0] != "generated":
        return False
    return user_owns_out_dir(user_id, "/".join(parts[:2]))


def update_history_print(user_id: int, out_dir: str, **fields) -> bool:
    allowed = {"stl_path", "print_scale_mm", "printer_id", "print_job_id", "print_status"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates or not user_owns_out_dir(user_id, out_dir):
        return False
    cols = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [user_id, out_dir]
    with _conn() as c:
        c.execute(f"UPDATE history SET {cols} WHERE user_id = ? AND out_dir = ?", vals)
    return True


def rename_history_entry(user_id: int, hist_id: int, label: str) -> dict:
    with _conn() as c:
        row = c.execute(
            "SELECT id FROM history WHERE id = ? AND user_id = ?", (hist_id, user_id)
        ).fetchone()
    if not row:
        return {"ok": False, "error": "Entry not found"}
    with _conn() as c:
        c.execute("UPDATE history SET label = ? WHERE id = ? AND user_id = ?",
                  (label[:80], hist_id, user_id))
    return {"ok": True}


def delete_history_entry(user_id: int, hist_id: int) -> dict:
    """Delete a history entry and its associated files."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM history WHERE id = ? AND user_id = ?",
            (hist_id, user_id)
        ).fetchone()
    if not row:
        return {"ok": False, "error": "Entry not found"}

    # Delete files from disk
    out_dir = row["out_dir"]
    if out_dir and os.path.isdir(out_dir):
        try:
            shutil.rmtree(out_dir)
        except Exception as e:
            pass  # don't fail if files are already gone

    # Delete upload file
    input_img = row["input_image"]
    if input_img:
        fpath = input_img.lstrip("/")
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass

    with _conn() as c:
        c.execute("DELETE FROM history WHERE id = ? AND user_id = ?", (hist_id, user_id))
        c.execute("DELETE FROM shares WHERE hist_id = ? AND user_id = ?", (hist_id, user_id))
    return {"ok": True}


# ── Shares ───────────────────────────────────────────────────────────────────

def create_share(user_id: int, hist_id: int, expires: int | None = None) -> dict:
    """Generate 64-char hex token, insert into shares, return {token, url}."""
    token = secrets.token_hex(32)  # 64-char hex string
    now = int(time.time())
    with _conn() as c:
        c.execute(
            "INSERT INTO shares (token, user_id, hist_id, created, expires, active) VALUES (?,?,?,?,?,1)",
            (token, user_id, hist_id, now, expires)
        )
    return {"token": token, "url": f"/share/{token}"}


def get_share(token: str) -> dict | None:
    """Return share row if token is active and not expired, else None."""
    now = int(time.time())
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM shares WHERE token = ? AND active = 1 AND (expires IS NULL OR expires > ?)",
            (token, now)
        ).fetchone()
    if not row:
        return None
    return dict(row)


def revoke_share(user_id: int, token: str) -> bool:
    """Set active=0 for token owned by user_id. Returns True if found and updated."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE shares SET active = 0 WHERE token = ? AND user_id = ?",
            (token, user_id)
        )
    return cur.rowcount > 0


def get_shares_for_user(user_id: int, hist_id: int) -> list:
    """Return all active share rows for a given history entry."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM shares WHERE user_id = ? AND hist_id = ? AND active = 1",
            (user_id, hist_id)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Jobs (persistent pipeline state) ─────────────────────────────────────────

def upsert_job(out_dir: str, user_id=None, status=None, stage=None,
               queue_pos=None, backend=None, error=None):
    """Insert or update a job row. Only provided (non-None) fields are written."""
    now = int(time.time())
    with _jobs_lock:
        with _conn() as c:
            existing = c.execute("SELECT id FROM jobs WHERE out_dir = ?", (out_dir,)).fetchone()
            if existing:
                updates = {"updated": now}
                if status    is not None: updates["status"]    = status
                if stage     is not None: updates["stage"]     = stage
                if queue_pos is not None: updates["queue_pos"] = queue_pos
                if backend   is not None: updates["backend"]   = backend
                if error     is not None: updates["error"]     = error[:500]
                if user_id   is not None: updates["user_id"]   = user_id
                cols = ", ".join(f"{k} = ?" for k in updates)
                vals = list(updates.values()) + [out_dir]
                c.execute(f"UPDATE jobs SET {cols} WHERE out_dir = ?", vals)
            else:
                c.execute(
                    """INSERT INTO jobs
                       (out_dir, user_id, status, stage, queue_pos, backend, error, created, updated)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (out_dir, user_id, status or "queued", stage or "",
                     queue_pos or 0, backend or "", (error or "")[:500], now, now),
                )


def get_job(out_dir: str) -> dict | None:
    """Return the job row for out_dir, or None if not found."""
    with _jobs_lock:
        with _conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE out_dir = ?", (out_dir,)).fetchone()
    return dict(row) if row else None


def mark_interrupted_jobs():
    """On server startup, mark any queued/running jobs as interrupted."""
    now = int(time.time())
    with _jobs_lock:
        with _conn() as c:
            c.execute(
                "UPDATE jobs SET status = 'interrupted', updated = ? "
                "WHERE status IN ('queued', 'running')",
                (now,),
            )


def get_jobs_for_user(user_id: int, limit: int = 50) -> list:
    """Return job rows for a user ordered by updated DESC."""
    with _jobs_lock:
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM jobs WHERE user_id = ? ORDER BY updated DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    return [dict(r) for r in rows]


def get_history(user_id: int, limit: int = 50) -> list:
    with _conn() as c:
        rows = c.execute("""
            SELECT * FROM history WHERE user_id = ?
            ORDER BY created DESC LIMIT ?
        """, (user_id, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for field in ("views", "depths_grey", "depths_color"):
            try:    d[field] = json.loads(d[field] or "[]")
            except: d[field] = []
        d["created_fmt"] = datetime.fromtimestamp(d["created"]).strftime("%d %b %Y, %H:%M")
        out.append(d)
    return out

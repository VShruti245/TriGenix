"""
tests/test_baseline_auth.py
============================
Phase 1 — Baseline smoke tests for auth.py.

These tests document the CURRENT behaviour of the existing auth module:

  1. init_db creates all expected tables and existing columns.
  2. register_user / login_user / validate_session / logout_session round-trip.
  3. Password hashing uses scrypt (confirmed in Phase 0 audit).
  4. Legacy SHA-256 accounts are upgraded to scrypt on login.
  5. Session tokens expire after SESSION_TTL seconds.
  6. save_history / update_history_recon / get_history round-trip.
  7. user_owns_out_dir / user_owns_generated_path ownership checks.
  8. delete_history_entry removes the record.
  9. Duplicate username / email registration returns error dicts.
 10. Invalid credentials return error dicts (no exception).
 11. That the 10 NEW columns required by the spec are ABSENT from the
     current schema (confirmed Bug D).  We assert their ABSENCE here
     so that Phase 10 can assert their PRESENCE after the migration.

Each test uses an isolated SQLite database via the tmp_db_path fixture.
No network access, no GPU, no model loading.
"""

import os
import sys
import time
import json
import sqlite3
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_db(monkeypatch, db_path):
    """Redirect auth.DB_PATH to the isolated test database."""
    import auth
    monkeypatch.setattr(auth, "DB_PATH", db_path)


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    """Verify the DB schema created by init_db()."""

    def test_creates_users_table(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db
        init_db()
        with sqlite3.connect(tmp_db_path) as c:
            cols = {row[1] for row in c.execute("PRAGMA table_info(users)")}
        expected = {"id", "username", "email", "pw_hash", "created"}
        assert expected.issubset(cols), f"Missing columns in users: {expected - cols}"

    def test_creates_sessions_table(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db
        init_db()
        with sqlite3.connect(tmp_db_path) as c:
            cols = {row[1] for row in c.execute("PRAGMA table_info(sessions)")}
        assert {"token", "user_id", "created", "expires"}.issubset(cols)

    def test_creates_history_table_with_baseline_columns(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db
        init_db()
        with sqlite3.connect(tmp_db_path) as c:
            cols = {row[1] for row in c.execute("PRAGMA table_info(history)")}
        baseline = {
            "id", "user_id", "created", "input_image", "masked_image",
            "views", "depths_grey", "depths_color", "final_glb", "out_dir",
            "proc_time", "vertices", "faces", "method", "watertight", "label",
            "stl_path", "print_scale_mm", "printer_id", "print_job_id",
            "print_status", "print_ready", "visual_watertight", "print_watertight",
        }
        assert baseline.issubset(cols), f"Missing baseline columns: {baseline - cols}"

    def test_missing_spec_columns_absent_from_current_schema(self, monkeypatch, tmp_db_path):
        """
        BUG D REGRESSION BASELINE:
        The 10 new columns required by the spec are ABSENT from the current schema.
        This test ASSERTS THEIR ABSENCE to lock in the current state.
        Phase 10 will invert these assertions after the migration.
        """
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db
        init_db()
        with sqlite3.connect(tmp_db_path) as c:
            cols = {row[1] for row in c.execute("PRAGMA table_info(history)")}

        missing_in_current = {
            "backend", "quality_score", "seg_confidence", "pipeline_version",
            "model_version", "scale_source", "warnings", "failure_reason",
            "mean_sil_iou", "job_id",
        }
        # Assert ABSENCE — these must not be present in the unmodified schema.
        present_when_should_be_absent = missing_in_current & cols
        assert not present_when_should_be_absent, (
            f"Phase 10 migration has already added these columns "
            f"(unexpected in Phase 1): {present_when_should_be_absent}"
        )

    def test_idempotent(self, monkeypatch, tmp_db_path):
        """Calling init_db() twice must not raise."""
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db
        init_db()
        init_db()  # second call must be safe


# ---------------------------------------------------------------------------
# register_user
# ---------------------------------------------------------------------------

class TestRegisterUser:
    def test_success(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user
        init_db()
        result = register_user("alice", "alice@example.com", "password123")
        assert result.get("ok") is True

    def test_duplicate_username_returns_error(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user
        init_db()
        register_user("bob", "bob@example.com", "password123")
        result = register_user("bob", "different@example.com", "password456")
        assert result.get("ok") is False
        assert "error" in result

    def test_duplicate_email_returns_error(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user
        init_db()
        register_user("charlie", "shared@example.com", "password123")
        result = register_user("delta", "shared@example.com", "password456")
        assert result.get("ok") is False

    def test_short_username_returns_error(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user
        init_db()
        result = register_user("ab", "ab@example.com", "password123")
        assert result.get("ok") is False

    def test_short_password_returns_error(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user
        init_db()
        result = register_user("validuser", "valid@example.com", "abc")
        assert result.get("ok") is False

    def test_invalid_email_returns_error(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user
        init_db()
        result = register_user("validuser2", "notanemail", "password123")
        assert result.get("ok") is False


# ---------------------------------------------------------------------------
# login_user
# ---------------------------------------------------------------------------

class TestLoginUser:
    def test_success_returns_token(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user
        init_db()
        register_user("testuser", "test@example.com", "mypassword")
        result = login_user("testuser", "mypassword")
        assert result.get("ok") is True
        assert "token" in result
        assert len(result["token"]) == 64  # 32-byte hex

    def test_wrong_password_returns_error(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user
        init_db()
        register_user("testuser2", "test2@example.com", "correct")
        result = login_user("testuser2", "wrong")
        assert result.get("ok") is False

    def test_nonexistent_user_returns_error(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, login_user
        init_db()
        result = login_user("nobody", "password")
        assert result.get("ok") is False

    def test_token_is_32_byte_hex(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user
        init_db()
        register_user("hextest", "hex@example.com", "hexpassword")
        result = login_user("hextest", "hexpassword")
        token = result["token"]
        assert len(token) == 64
        int(token, 16)  # must be valid hex — raises ValueError if not

    def test_password_uses_scrypt_hash(self, monkeypatch, tmp_db_path):
        """Verify the stored password hash uses the scrypt$ prefix."""
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user
        import sqlite3
        init_db()
        register_user("scryptuser", "scrypt@example.com", "scryptpassword")
        with sqlite3.connect(tmp_db_path) as c:
            c.row_factory = sqlite3.Row
            row = c.execute(
                "SELECT pw_hash FROM users WHERE username = ?", ("scryptuser",)
            ).fetchone()
        assert row is not None
        assert row["pw_hash"].startswith("scrypt$"), (
            f"Expected scrypt$ prefix, got: {row['pw_hash'][:20]!r}"
        )

    def test_legacy_sha256_upgraded_on_login(self, monkeypatch, tmp_db_path):
        """
        Current behaviour: a legacy salted SHA-256 hash is upgraded to scrypt
        on successful login (from auth._needs_pw_rehash + _hash_pw in login_user).
        """
        _patch_db(monkeypatch, tmp_db_path)
        import sqlite3
        import hashlib
        from auth import init_db, login_user

        init_db()
        salt = "testsalt"
        pw = "oldpassword"
        legacy_hash = salt + ":" + hashlib.sha256((salt + pw).encode()).hexdigest()

        # Insert a user with a legacy SHA-256 hash directly.
        with sqlite3.connect(tmp_db_path) as c:
            c.execute(
                "INSERT INTO users (username, email, pw_hash, created) VALUES (?,?,?,?)",
                ("legacyuser", "legacy@example.com", legacy_hash, int(time.time())),
            )

        # Login should succeed and upgrade the hash.
        result = login_user("legacyuser", pw)
        assert result.get("ok") is True, f"Login failed: {result}"

        # Verify hash was upgraded to scrypt.
        with sqlite3.connect(tmp_db_path) as c:
            c.row_factory = sqlite3.Row
            row = c.execute(
                "SELECT pw_hash FROM users WHERE username = ?", ("legacyuser",)
            ).fetchone()
        assert row["pw_hash"].startswith("scrypt$"), (
            "Legacy hash was NOT upgraded to scrypt after login"
        )


# ---------------------------------------------------------------------------
# validate_session / logout_session
# ---------------------------------------------------------------------------

class TestSessionManagement:
    def test_valid_session_returns_user_dict(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user, validate_session
        init_db()
        register_user("sessuser", "sess@example.com", "sesspass")
        token = login_user("sessuser", "sesspass")["token"]
        user = validate_session(token)
        assert user is not None
        assert user["username"] == "sessuser"
        assert "id" in user
        assert "email" in user

    def test_invalid_token_returns_none(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, validate_session
        init_db()
        assert validate_session("notavalidtoken") is None

    def test_empty_token_returns_none(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, validate_session
        init_db()
        assert validate_session("") is None

    def test_session_expires(self, monkeypatch, tmp_db_path):
        """An expired session token returns None."""
        _patch_db(monkeypatch, tmp_db_path)
        import sqlite3
        from auth import init_db, register_user, login_user, validate_session
        init_db()
        register_user("expuser", "exp@example.com", "exppass")
        token = login_user("expuser", "exppass")["token"]

        # Manually expire the session in the DB.
        with sqlite3.connect(tmp_db_path) as c:
            c.execute("UPDATE sessions SET expires = ? WHERE token = ?", (0, token))

        assert validate_session(token) is None

    def test_logout_invalidates_session(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user, validate_session, logout_session
        init_db()
        register_user("logoutuser", "logout@example.com", "logoutpass")
        token = login_user("logoutuser", "logoutpass")["token"]
        assert validate_session(token) is not None
        logout_session(token)
        assert validate_session(token) is None


# ---------------------------------------------------------------------------
# save_history / get_history / update_history_recon
# ---------------------------------------------------------------------------

class TestHistory:
    def _setup_user(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user
        init_db()
        register_user("histuser", "hist@example.com", "histpass")
        return login_user("histuser", "histpass")["token"]

    def test_save_history_inserts_record(self, monkeypatch, tmp_db_path):
        self._setup_user(monkeypatch, tmp_db_path)
        from auth import validate_session, save_history, get_history, login_user
        user = validate_session(login_user("histuser", "histpass")["token"])
        proc_result = {
            "out_dir": "generated/test_001",
            "masked": "/generated/test_001/masked.png",
            "views": ["/generated/test_001/view_0.png"],
            "depths_grey": [],
            "depths_color": [],
            "time": 12.3,
        }
        save_history(user["id"], "/uploads/test.png", proc_result)
        history = get_history(user["id"])
        assert len(history) >= 1
        assert history[0]["out_dir"] == "generated/test_001"

    def test_update_history_recon_writes_method(self, monkeypatch, tmp_db_path):
        self._setup_user(monkeypatch, tmp_db_path)
        from auth import validate_session, save_history, update_history_recon, get_history, login_user
        user = validate_session(login_user("histuser", "histpass")["token"])
        proc_result = {"out_dir": "generated/test_002", "time": 5.0}
        save_history(user["id"], "/uploads/t.png", proc_result)
        update_history_recon("generated/test_002", {
            "final_glb": "/generated/test_002/model.glb",
            "vertices": 5000,
            "faces": 8000,
            "method": "triposr",
            "watertight": True,
            "print_ready": True,
            "visual_watertight": True,
            "print_watertight": True,
        })
        history = get_history(user["id"])
        record = next(r for r in history if r["out_dir"] == "generated/test_002")
        assert record["method"] == "triposr"
        assert record["vertices"] == 5000

    def test_get_history_json_fields_parsed(self, monkeypatch, tmp_db_path):
        """views / depths_grey / depths_color are stored as JSON and returned as lists."""
        self._setup_user(monkeypatch, tmp_db_path)
        from auth import validate_session, save_history, get_history, login_user
        user = validate_session(login_user("histuser", "histpass")["token"])
        proc_result = {
            "out_dir": "generated/test_003",
            "views": ["/generated/test_003/view_0.png", "/generated/test_003/view_1.png"],
            "depths_grey": ["/generated/test_003/depth_grey_0.png"],
            "depths_color": [],
            "time": 8.0,
        }
        save_history(user["id"], "/uploads/x.png", proc_result)
        history = get_history(user["id"])
        record = next(r for r in history if r["out_dir"] == "generated/test_003")
        assert isinstance(record["views"], list)
        assert len(record["views"]) == 2
        assert isinstance(record["depths_grey"], list)


# ---------------------------------------------------------------------------
# Ownership checks
# ---------------------------------------------------------------------------

class TestOwnership:
    def test_user_owns_out_dir_after_save(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user, validate_session, save_history, user_owns_out_dir
        init_db()
        register_user("owner", "owner@example.com", "ownerpass")
        user = validate_session(login_user("owner", "ownerpass")["token"])
        save_history(user["id"], "/uploads/o.png", {"out_dir": "generated/owned_001", "time": 1.0})
        assert user_owns_out_dir(user["id"], "generated/owned_001") is True

    def test_user_does_not_own_other_dir(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user, validate_session, save_history, user_owns_out_dir
        init_db()
        register_user("user_a", "a@example.com", "passA123")
        register_user("user_b", "b@example.com", "passB123")
        user_a = validate_session(login_user("user_a", "passA123")["token"])
        user_b = validate_session(login_user("user_b", "passB123")["token"])
        save_history(user_a["id"], "/uploads/a.png", {"out_dir": "generated/a_job", "time": 1.0})
        assert user_owns_out_dir(user_b["id"], "generated/a_job") is False

    def test_user_owns_generated_path(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user, validate_session, save_history, user_owns_generated_path
        init_db()
        register_user("genowner", "gen@example.com", "genpass")
        user = validate_session(login_user("genowner", "genpass")["token"])
        save_history(user["id"], "/uploads/g.png", {"out_dir": "generated/gen_001", "time": 1.0})
        assert user_owns_generated_path(user["id"], "/generated/gen_001/model.glb") is True
        assert user_owns_generated_path(user["id"], "/generated/other_job/model.glb") is False


# ---------------------------------------------------------------------------
# delete_history_entry
# ---------------------------------------------------------------------------

class TestDeleteHistory:
    def test_delete_removes_record(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user, validate_session, save_history, get_history, delete_history_entry
        init_db()
        register_user("deluser", "del@example.com", "delpass")
        user = validate_session(login_user("deluser", "delpass")["token"])
        save_history(user["id"], "/uploads/d.png", {"out_dir": "generated/del_001", "time": 1.0})
        history = get_history(user["id"])
        entry_id = history[0]["id"]
        result = delete_history_entry(user["id"], entry_id)
        assert result.get("ok") is True
        assert len(get_history(user["id"])) == 0

    def test_delete_nonexistent_returns_error(self, monkeypatch, tmp_db_path):
        _patch_db(monkeypatch, tmp_db_path)
        from auth import init_db, register_user, login_user, validate_session, delete_history_entry
        init_db()
        register_user("del2user", "del2@example.com", "del2pass")
        user = validate_session(login_user("del2user", "del2pass")["token"])
        result = delete_history_entry(user["id"], 99999)
        assert result.get("ok") is False

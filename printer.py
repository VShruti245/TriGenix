"""
printer.py - TriGenix print-prep and printer-control helpers.

Printers are configured through TRIGENIX_PRINTERS as JSON:
[
  {"id":"mk4","type":"octoprint","base_url":"http://octopi.local","api_key":"..."},
  {"id":"klipper","type":"moonraker","base_url":"http://printer.local"}
]
"""

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib import request, parse


PREPARED_JOBS = {}


def _configured_printers():
    raw = os.environ.get("TRIGENIX_PRINTERS", "[]")
    try:
        printers = json.loads(raw)
    except Exception:
        printers = []
    out = []
    for p in printers:
        if not p.get("id") or not p.get("type"):
            continue
        if p["type"] in ("octoprint", "moonraker") and not p.get("base_url"):
            continue
        if p["type"] == "serial" and not p.get("port"):
            continue
        out.append(p)
    if not out and os.environ.get("TRIGENIX_DRYRUN_PRINTER", "1") == "1":
        out.append({
            "id": "dryrun",
            "type": "dryrun",
            "name": "Dry-run printer",
            "base_url": "local",
        })
    return out


def _http_json(url, method="GET", headers=None, data=None, timeout=5):
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))


def list_printers():
    out = []
    for p in _configured_printers():
        info = {**p}
        info.pop("api_key", None)
        info["online"] = False
        info["status"] = "configured"
        try:
            if p["type"] == "octoprint":
                data = _http_json(
                    p["base_url"].rstrip("/") + "/api/printer",
                    headers={"X-Api-Key": p.get("api_key", "")},
                )
                info["online"] = True
                info["status"] = data.get("state", {}).get("text", "online")
                info["temperatures"] = data.get("temperature", {})
            elif p["type"] == "moonraker":
                data = _http_json(p["base_url"].rstrip("/") + "/printer/info")
                info["online"] = bool(data.get("result"))
                info["status"] = data.get("result", {}).get("state", "online")
            elif p["type"] == "serial":
                info["online"] = True
                info["status"] = f"serial configured on {p.get('port')}"
            elif p["type"] == "dryrun":
                info["online"] = True
                info["status"] = "simulation only"
        except Exception as e:
            info["status"] = f"offline: {e}"
        out.append(info)
    return out


def _printer_by_id(printer_id):
    for p in _configured_printers():
        if p["id"] == printer_id:
            return p
    raise ValueError("Printer not configured")


def slice_model(stl_path, printer_profile):
    cura = os.environ.get("CURA_ENGINE_PATH", "").strip()
    if printer_profile.get("type") == "dryrun" and not cura:
        src = Path(stl_path)
        if not src.exists():
            raise FileNotFoundError(stl_path)
        out = src.with_suffix(".dryrun.gcode")
        settings = printer_profile.get("settings", {})
        out.write_text(
            "\n".join([
                "; TriGenix dry-run G-code",
                f"; source={src.name}",
                f"; layer_height={settings.get('layer_height', 'n/a')}",
                f"; infill={settings.get('infill', 'n/a')}",
                f"; supports={settings.get('supports', 'n/a')}",
                f"; material={settings.get('material', 'n/a')}",
                "G28 ; home axes",
                "M104 S0 ; dry-run nozzle off",
                "M140 S0 ; dry-run bed off",
                "; no physical movement beyond homing in simulation output",
                "M84",
                "",
            ]),
            encoding="utf-8",
        )
        return str(out)
    if not cura:
        raise RuntimeError("CURA_ENGINE_PATH is not configured; cannot slice safely")
    src = Path(stl_path)
    if not src.exists():
        raise FileNotFoundError(stl_path)
    out = src.with_suffix(".gcode")
    cmd = [cura, "slice", "-l", str(src), "-o", str(out)]
    subprocess.run(cmd, check=True, timeout=600)
    if not out.exists():
        raise RuntimeError("Slicer did not produce G-code")
    return str(out)


def _scaled_stl_copy(stl_path, scale_mm=None):
    dims = {}
    for src, dst in (("x", 0), ("y", 1), ("z", 2), ("width", 0), ("height", 1), ("depth", 2)):
        try:
            value = float((scale_mm or {}).get(src) or 0)
        except Exception:
            value = 0.0
        if value > 0:
            dims[dst] = value
    if not dims:
        return stl_path
    try:
        import numpy as np
        import trimesh
        mesh = trimesh.load(stl_path, force="mesh")
        ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
        scale = np.ones(3, dtype=float)
        if len(dims) == 1:
            axis, mm = next(iter(dims.items()))
            if ext[axis] > 1e-8:
                scale[:] = mm / ext[axis]
        else:
            for axis, mm in dims.items():
                if ext[axis] > 1e-8:
                    scale[axis] = mm / ext[axis]
        mesh.vertices *= scale
        out = str(Path(stl_path).with_name(Path(stl_path).stem + "_scaled.stl"))
        mesh.export(out, file_type="stl")
        return out
    except Exception as e:
        raise RuntimeError(f"Could not apply print scale: {e}") from e


def prepare_job(stl_path, printer_id, scale_mm=None, orientation=None, settings=None):
    printer = _printer_by_id(printer_id)
    scaled_stl_path = _scaled_stl_copy(stl_path, scale_mm)
    gcode_path = slice_model(scaled_stl_path, {**printer, "settings": settings or {}})
    token = uuid.uuid4().hex
    try:
        filament_g = round(max(1.0, Path(scaled_stl_path).stat().st_size / 1_000_000.0 * 6.0), 2)
    except Exception:
        filament_g = None
    PREPARED_JOBS[token] = {
        "token": token,
        "printer_id": printer_id,
        "stl_path": stl_path,
        "sliced_stl_path": scaled_stl_path,
        "gcode_path": gcode_path,
        "scale_mm": scale_mm or {},
        "orientation": orientation or {},
        "created": int(time.time()),
        "estimate": {
            "time_seconds": 0 if printer.get("type") == "dryrun" else None,
            "filament_g": filament_g,
            "cost": None,
        },
    }
    return PREPARED_JOBS[token]


def send_job(job_token):
    job = PREPARED_JOBS.get(job_token)
    if not job:
        raise ValueError("Prepared job not found")
    printer = _printer_by_id(job["printer_id"])
    if printer["type"] == "octoprint":
        boundary = "----TriGenixPrintBoundary"
        filename = os.path.basename(job["gcode_path"])
        with open(job["gcode_path"], "rb") as f:
            gcode = f.read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8") + gcode + (
            f"\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="select"\r\n\r\ntrue\r\n'
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="print"\r\n\r\ntrue\r\n'
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        url = printer["base_url"].rstrip("/") + "/api/files/local"
        req = request.Request(url, data=body, method="POST")
        req.add_header("X-Api-Key", printer.get("api_key", ""))
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with request.urlopen(req, timeout=30) as resp:
            resp.read()
        return {"job_id": job_token, "status": "started"}
    if printer["type"] == "moonraker":
        boundary = "----TriGenixMoonrakerBoundary"
        filename = os.path.basename(job["gcode_path"])
        with open(job["gcode_path"], "rb") as f:
            gcode = f.read()
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="root"\r\n\r\ngcodes\r\n'
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="path"\r\n\r\n\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8") + gcode + f"\r\n--{boundary}--\r\n".encode("utf-8")
        upload_url = printer["base_url"].rstrip("/") + "/server/files/upload"
        req = request.Request(upload_url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with request.urlopen(req, timeout=30) as resp:
            resp.read()
        _http_json(
            printer["base_url"].rstrip("/") + "/printer/print/start",
            method="POST",
            data={"filename": filename},
        )
        return {"job_id": job_token, "status": "started"}
    if printer["type"] == "serial":
        try:
            import serial
        except Exception as e:
            raise RuntimeError("pyserial is required for direct serial printing") from e
        baudrate = int(printer.get("baudrate", 115200))
        with serial.Serial(printer["port"], baudrate, timeout=2) as ser, open(job["gcode_path"], "rb") as gcode:
            for raw in gcode:
                line = raw.strip()
                if not line or line.startswith(b";"):
                    continue
                ser.write(line + b"\n")
                ser.readline()
        return {"job_id": job_token, "status": "sent"}
    if printer["type"] == "dryrun":
        return {
            "job_id": job_token,
            "status": "dryrun_started",
            "message": "Dry-run job accepted; no physical printer was contacted.",
        }
    raise ValueError("Unsupported printer type")


def get_job_status(printer_id):
    printer = _printer_by_id(printer_id)
    if printer["type"] == "octoprint":
        return _http_json(
            printer["base_url"].rstrip("/") + "/api/job",
            headers={"X-Api-Key": printer.get("api_key", "")},
        )
    if printer["type"] == "moonraker":
        return _http_json(printer["base_url"].rstrip("/") + "/printer/objects/query?print_stats")
    if printer["type"] == "serial":
        return {"state": "unknown", "note": "Direct serial status polling is not persistent in this simple backend"}
    if printer["type"] == "dryrun":
        return {
            "state": "complete",
            "progress": {"completion": 100.0, "printTimeLeft": 0},
            "temperatures": {},
            "note": "Dry-run printer simulation.",
        }
    raise ValueError("Unsupported printer type")


def cancel_job(printer_id):
    printer = _printer_by_id(printer_id)
    if printer["type"] == "octoprint":
        return _http_json(
            printer["base_url"].rstrip("/") + "/api/job",
            method="POST",
            headers={"X-Api-Key": printer.get("api_key", "")},
            data={"command": "cancel"},
        )
    if printer["type"] == "moonraker":
        return _http_json(printer["base_url"].rstrip("/") + "/printer/print/cancel", method="POST", data={})
    if printer["type"] == "serial":
        try:
            import serial
        except Exception as e:
            raise RuntimeError("pyserial is required for direct serial printing") from e
        baudrate = int(printer.get("baudrate", 115200))
        with serial.Serial(printer["port"], baudrate, timeout=2) as ser:
            ser.write(b"M112\n")
        return {"status": "emergency_stop_sent"}
    if printer["type"] == "dryrun":
        return {"status": "dryrun_cancelled"}
    raise ValueError("Unsupported printer type")


def ensure_stl_for_out_dir(out_dir):
    stl = Path(out_dir) / "model_print.stl"
    if stl.exists():
        return str(stl)
    fallback = Path(out_dir) / "model.stl"
    if fallback.exists():
        shutil.copyfile(fallback, stl)
        return str(stl)
    raise FileNotFoundError("No STL export found for this generation")

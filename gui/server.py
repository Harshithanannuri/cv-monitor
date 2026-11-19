"""Flask web GUI + JSON API for the monitor.  Run:  python app.py   ->  http://127.0.0.1:8000"""
import json
import os
import re
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from src.utils import reference_frame

from .runner import (IMG_EXT, ROOT, VIDEO_EXT, Job, Runner, build_config, list_sources, read_run_events,
                     read_settings, write_settings)

# ----- settings shown in the GUI.  live=True -> takes effect immediately while running
SCHEMA = [
    {"group": "Alarm rules", "items": [
        {"key": "owner.alarm_s", "label": "Alarm after owner has been away for (s)", "type": "range", "min": 5, "max": 120, "step": 1, "live": True},
        {"key": "owner.alarm_m", "label": "Owner counts as 'away' beyond (m)", "type": "range", "min": 1, "max": 8, "step": 0.5, "live": True},
        {"key": "owner.warn_m", "label": "Owner counts as 'attending' within (m)", "type": "range", "min": 0.5, "max": 5, "step": 0.5, "live": True},
        {"key": "tracking.static_s", "label": "Object must stay still for (s) before it is checked", "type": "range", "min": 1, "max": 15, "step": 0.5, "live": True},
        {"key": "tracking.remove_s", "label": "Confirm 'object taken' after (s)", "type": "range", "min": 1, "max": 15, "step": 0.5, "live": True},
        {"key": "alerts.cooldown_s", "label": "Do not repeat an alarm at the same place for (s)", "type": "range", "min": 0, "max": 120, "step": 5, "live": True},
    ]},
    {"group": "Detection", "items": [
        {"key": "candidates.min_area_px", "label": "Smallest object (pixel area)", "type": "range", "min": 50, "max": 3000, "step": 50, "live": True},
        {"key": "background.diff_thr", "label": "Change sensitivity (lower = more sensitive)", "type": "range", "min": 10, "max": 70, "step": 1, "live": True},
        {"key": "verify.cnn_conf", "label": "CNN confidence needed to override the rule test", "type": "range", "min": 0.5, "max": 0.99, "step": 0.01, "live": True},
        {"key": "pipeline.shadow", "label": "Shadow handling", "type": "bool", "live": True},
        {"key": "pipeline.global_check", "label": "Lighting / scene-change check", "type": "bool", "live": True},
        {"key": "owner.reid", "label": "Re-identify owner after tracker ID switch", "type": "bool", "live": True},
    ]},
    {"group": "Pipeline (applies on next start)", "items": [
        {"key": "pipeline.people", "label": "People detector", "type": "select", "options": ["yolo", "hog", "none"]},
        {"key": "pipeline.verify", "label": "Abandoned/removed verdict", "type": "select", "options": ["both", "rule", "cnn", "none"]},
        {"key": "pipeline.bg_mode", "label": "Background model", "type": "select", "options": ["dual", "single"]},
        {"key": "pipeline.backend", "label": "Background backend", "type": "select", "options": ["ra", "mog2", "knn"]},
        {"key": "pipeline.tamper", "label": "Camera-health gate", "type": "bool"},
        {"key": "source.process_fps", "label": "Frames processed per second", "type": "range", "min": 2, "max": 25, "step": 1},
        {"key": "clips.enabled", "label": "Record evidence clip around alarms", "type": "bool"},
    ]},
]


def safe_name(n):
    """Run / file names that are safe to join onto a directory (no separators, no '..')."""
    return n if re.fullmatch(r"[\w.\-]+", n or "") and ".." not in n else None


def _get(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def _nest(dotted, value):
    out, cur = {}, None
    parts = dotted.split(".")
    cur = out
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return out


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def create_app():
    app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    runner, job = Runner(), Job()
    out_root = lambda: Path(build_config()["alerts"]["out_dir"])

    @app.after_request
    def no_cache(r):
        if request.path.startswith("/api") or request.path.endswith((".js", ".css", ".html")) or request.path == "/":
            r.headers["Cache-Control"] = "no-store"
        return r

    # ------------------------------------------------------------- pages
    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    # ------------------------------------------------------------- live
    @app.route("/stream.mjpg")
    def stream():
        def gen():
            last = -1
            while True:
                if runner.frame_no != last and runner.jpeg is not None:
                    last = runner.frame_no
                    buf = runner.jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(buf)).encode() + b"\r\n\r\n" + buf + b"\r\n"
                else:
                    time.sleep(0.03)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/frame.jpg")
    def frame():
        if runner.jpeg is None:
            return ("", 204)
        return Response(runner.jpeg, mimetype="image/jpeg")

    @app.route("/api/state")
    def state():
        return jsonify(runner.snapshot())

    @app.route("/api/events")
    def events():
        after = int(request.args.get("after", 0))
        with runner.lock:
            return jsonify(runner.events[after:])

    @app.route("/api/sources")
    def sources():
        return jsonify(list_sources())

    @app.route("/api/upload", methods=["POST"])
    def upload():
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify(error="no file"), 400
        name = secure_filename(f.filename)
        if not name.lower().endswith(VIDEO_EXT):
            return jsonify(error="unsupported file type"), 400
        dest = ROOT / "data" / "uploads"
        dest.mkdir(parents=True, exist_ok=True)
        f.save(str(dest / name))
        return jsonify(ok=True, value=f"data/uploads/{name}")

    @app.route("/api/start", methods=["POST"])
    def start():
        b = request.get_json(force=True)
        src = (b.get("source") or "").strip()
        if not src:
            return jsonify(error="choose a source"), 400
        live = src.isdigit() or src.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        if not live and not (ROOT / src).exists() and not os.path.exists(src):
            return jsonify(error=f"not found: {src}"), 400
        try:
            runner.start(src, float(b.get("speed", 1.0)), b.get("overrides"))
        except RuntimeError as e:
            return jsonify(error=str(e)), 409
        return jsonify(ok=True)

    for name in ("pause", "resume", "stop"):
        app.add_url_rule(f"/api/{name}", f"api_{name}", (lambda n: lambda: (getattr(runner, n)(), jsonify(ok=True))[1])(name), methods=["POST"])

    @app.route("/api/live_config", methods=["POST"])
    def live_config():
        patch = request.get_json(force=True)
        runner.apply_live(patch)
        return jsonify(ok=True)

    # ------------------------------------------------------------- setup: zones / calibration / settings
    @app.route("/api/ref_frame.jpg")
    def ref_frame():
        src = request.args.get("source", "")
        if not src:
            return ("choose a source first", 400)
        try:
            path = src if (src.isdigit() or "://" in src) else str(ROOT / src)
            img = reference_frame(path, build_config()["source"]["resize_width"])
        except Exception as e:
            return (str(e), 400)
        if img is None:
            return ("could not read a frame", 400)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/api/settings", methods=["GET"])
    def get_settings():
        cfg = build_config()
        schema = []
        for g in SCHEMA:
            items = []
            for it in g["items"]:
                it = dict(it)
                it["value"] = _get(cfg, it["key"])
                items.append(it)
            schema.append({"group": g["group"], "items": items})
        s = read_settings()
        return jsonify(schema=schema, zones=cfg.get("zones") or {"include": [], "ignore": []},
                       homography=cfg.get("homography") or {}, px_per_m=cfg["owner"]["px_per_m"],
                       calibrated=bool((cfg.get("homography") or {}).get("image_points")),
                       saved=bool(s))

    @app.route("/api/settings", methods=["POST"])
    def save_settings():
        b = request.get_json(force=True)
        s = read_settings()
        if "overrides" in b:
            _merge(s.setdefault("overrides", {}), b["overrides"])
        for k in ("zones", "homography", "px_per_m"):
            if k in b:
                s[k] = b[k]
        write_settings(s)
        return jsonify(ok=True)

    @app.route("/api/settings/reset", methods=["POST"])
    def reset_settings():
        write_settings({})
        return jsonify(ok=True)

    # ------------------------------------------------------------- history & media
    @app.route("/api/runs")
    def runs():
        root, rows = out_root(), []
        if root.exists():
            for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
                ev = read_run_events(root, d.name)
                rows.append({"name": d.name, "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
                             "events": len(ev), "alarms": sum(e["level"] == "alarm" for e in ev),
                             "clips": len(list((d / "clips").glob("*.mp4"))) if (d / "clips").exists() else 0})
        return jsonify(rows)

    @app.route("/api/runs/<name>")
    def run_detail(name):
        name = safe_name(name)
        if not name:
            return jsonify(error="bad name"), 400
        d = out_root() / name
        clips = sorted(p.name for p in (d / "clips").glob("*.mp4")) if (d / "clips").exists() else []
        return jsonify(events=read_run_events(out_root(), name),
                       clips=[{"name": c, "url": f"/media/{name}/clips/{c}"} for c in clips])

    @app.route("/media/<run>/<kind>/<path:fname>")
    def media(run, kind, fname):
        if kind not in ("snapshots", "clips") or not safe_name(run):
            return ("", 404)
        return send_from_directory(str(out_root() / run / kind), fname)

    # ------------------------------------------------------------- results (eval / ablations / figures)
    @app.route("/api/results")
    def results():
        res = {}
        p = ROOT / "eval" / "results.json"
        if p.exists():
            res["eval"] = json.loads(p.read_text())
        p = ROOT / "eval" / "ablations.md"
        if p.exists():
            res["ablations_md"] = p.read_text()
        figs = ROOT / "report" / "figures"
        res["figures"] = sorted(f.name for f in figs.glob("*.png")) if figs.exists() else []
        return jsonify(res)

    @app.route("/figures/<path:f>")
    def figures(f):
        return send_from_directory(str(ROOT / "report" / "figures"), f)

    @app.route("/api/job", methods=["GET"])
    def job_status():
        return jsonify(job.status())

    @app.route("/api/job", methods=["POST"])
    def job_start():
        b = request.get_json(force=True)
        name = b.get("name")
        if name not in Job.COMMANDS:
            return jsonify(error="unknown job"), 400
        try:
            job.start(name, ["--max_seconds", str(b["max_seconds"])] if b.get("max_seconds") else None)
        except RuntimeError as e:
            return jsonify(error=str(e)), 409
        return jsonify(ok=True)

    app.runner, app.job = runner, job
    return app

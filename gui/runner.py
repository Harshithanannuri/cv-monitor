"""Background worker that runs the pipeline on a source and exposes live state to the web GUI."""
import csv
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import cv2

from src.pipeline import Pipeline
from src.recorder import ClipRecorder
from src.utils import deep_update, load_config, open_source

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / "data" / "user_settings.json"
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------- settings
def read_settings():
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {}


def write_settings(d):
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(d, indent=1))


def build_config(extra=None):
    """config.yaml  <-  user settings (GUI)  <-  per-run overrides."""
    cfg = load_config(str(ROOT / "config.yaml"))
    s = read_settings()
    deep_update(cfg, s.get("overrides") or {})
    if s.get("zones"):
        cfg["zones"] = s["zones"]
    if s.get("homography"):
        cfg["homography"] = s["homography"]
    if s.get("px_per_m"):
        cfg["owner"]["px_per_m"] = s["px_per_m"]
    deep_update(cfg, extra or {})
    cfg["alerts"]["out_dir"] = str(ROOT / cfg["alerts"]["out_dir"])
    return cfg


def level_of(kind):
    if kind in ("alarm", "removed") or kind.startswith("tamper"):
        return "alarm"
    if kind in ("warning", "scene_change"):
        return "warn"
    return "info"


def list_sources():
    out = []
    for folder in ("data/clips", "data/uploads"):
        d = ROOT / folder
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            if p.is_file() and p.suffix.lower() in VIDEO_EXT:
                out.append({"label": f"{p.name}  ({p.stat().st_size // 1024 // 1024 or 1} MB)", "value": rel, "kind": "video"})
            elif p.is_dir() and any(f.suffix.lower() in IMG_EXT for f in p.iterdir()):
                out.append({"label": f"{p.name}/  (frame folder)", "value": rel, "kind": "frames"})
    return out


# ---------------------------------------------------------------- live run
class Runner:
    def __init__(self):
        self.lock = threading.RLock()
        self.thread = None
        self.stop_flag, self.pause_flag = threading.Event(), threading.Event()
        self._reset()

    def _reset(self):
        self.state, self.error = "idle", None
        self.source = self.run_name = self.out_dir = None
        self.t = 0.0
        self.duration = None
        self.fps = 0.0
        self.speed = 1.0
        self.events, self.series = [], deque(maxlen=400)
        self.jpeg = None
        self.frame_no = getattr(self, 'frame_no', 0)       # monotonic: streams never miss a new run
        self.tracks, self.people_n, self.lf = [], 0, 0.0
        self.banner, self.warnings, self.components = "", [], {}
        self.counts = {"alarm": 0, "warn": 0, "info": 0}
        self.cfg = None
        self.pipe = None

    # ------------------------------------------------------------ control
    def start(self, source, speed=1.0, overrides=None):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("already running - stop it first")
            self._reset()
            self.stop_flag.clear()
            self.pause_flag.clear()
            self.source, self.speed = source, float(speed)
            self.state = "starting"
            self.cfg = build_config(overrides)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def pause(self):
        if self.state == "running":
            self.pause_flag.set()
            self.state = "paused"

    def resume(self):
        if self.state == "paused":
            self.pause_flag.clear()
            self.state = "running"

    def stop(self):
        self.stop_flag.set()
        self.pause_flag.clear()
        if self.thread:
            self.thread.join(timeout=5)

    def apply_live(self, patch):
        """In-place update: the pipeline keeps references to these sub-dicts, so it takes effect now."""
        with self.lock:
            if self.cfg is not None:
                deep_update(self.cfg, patch)

    # ------------------------------------------------------------ pipeline construction
    def _make_pipeline(self, cfg, run_name):
        attempts = [{}, {"pipeline": {"verify": "rule"}}, {"pipeline": {"verify": "rule", "people": "none"}}]
        last = None
        for patch in attempts:
            try:
                if patch:
                    deep_update(cfg, patch)
                    self.warnings.append(f"fallback after error: {last} -> using {patch['pipeline']}")
                return Pipeline(cfg, run_name)
            except Exception as e:                         # missing torch / weights / ultralytics ...
                last = f"{type(e).__name__}: {e}"
        raise RuntimeError(last)

    def _components(self, pipe, cfg, requested_people):
        det = type(pipe.people_det).__name__ if pipe.people_det is not None else "none"
        names = {"YoloPeople": "yolo", "HogPeople": "hog", "none": "none"}
        actual = names.get(det, det)
        if actual != requested_people:
            self.warnings.append(f"people detector '{requested_people}' unavailable -> running with '{actual}'")
        self.components = {"background": f'{cfg["pipeline"]["bg_mode"]}/{cfg["pipeline"]["backend"]}',
                           "people": actual,
                           "verify": cfg["pipeline"]["verify"] + (" (CNN loaded)" if pipe.cnn else ""),
                           "tamper": cfg["pipeline"]["tamper"]}

    # ------------------------------------------------------------ worker
    def _run(self):
        recorder = None
        try:
            cfg, source = self.cfg, self.source
            name = os.path.splitext(os.path.basename(str(source).rstrip("/\\")))[0] or "camera"
            self.run_name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
            requested = cfg["pipeline"]["people"]
            pipe = self._make_pipeline(cfg, self.run_name)
            self.pipe = pipe
            self._components(pipe, cfg, requested)
            self.out_dir = str(pipe.alerts.out)
            if cfg["alerts"].get("write_files", True) and cfg.get("clips", {}).get("enabled", True):
                recorder = ClipRecorder(pipe.alerts.out, cfg["source"]["process_fps"], cfg["clips"])
            s = cfg["source"]
            src = open_source(str(ROOT / source) if not _is_live(source) and not os.path.isabs(source) else source,
                              s["process_fps"], s["fps_override"], s["resize_width"])
            self.duration = src.duration
            self.state = "running"
            wall0, t_first, paused_total, n_ev, ema = time.perf_counter(), None, 0.0, 0, None
            for t, frame in src:
                if self.stop_flag.is_set():
                    break
                if self.pause_flag.is_set():
                    p0 = time.perf_counter()
                    while self.pause_flag.is_set() and not self.stop_flag.is_set():
                        time.sleep(0.05)
                    paused_total += time.perf_counter() - p0
                t_first = t if t_first is None else t_first
                ts = time.perf_counter()
                with self.lock:
                    vis = pipe.process(frame, t)
                dt = time.perf_counter() - ts
                ema = dt if ema is None else 0.9 * ema + 0.1 * dt
                new = pipe.alerts.events[n_ev:]
                n_ev = len(pipe.alerts.events)
                self._publish(pipe, vis, t, ema, new, recorder, cfg)
                if self.speed > 0 and not getattr(src, "live", False):      # pace to real time / speed
                    target = (t - t_first) / self.speed + paused_total
                    slack = target - (time.perf_counter() - wall0)
                    if slack > 0:
                        time.sleep(min(slack, 0.5))
            self.state = "stopped" if self.stop_flag.is_set() else "finished"
        except Exception as e:
            self.state, self.error = "error", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            if recorder:
                recorder.close()

    def _publish(self, pipe, vis, t, ema, new_events, recorder, cfg):
        with self.lock:
            self.t = t
            self.fps = 1.0 / ema if ema else 0.0
            for e in new_events:
                lvl = level_of(e["kind"])
                self.counts[lvl] += 1
                snap = e.get("snapshot")
                self.events.append({"id": len(self.events) + 1, "t": e["t"], "kind": e["kind"], "level": lvl,
                                    "track_id": e["track_id"], "info": e["info"],
                                    "snapshot": f"/media/{self.run_name}/snapshots/{snap}" if snap else None})
            self.tracks = [{"id": tr.id, "state": tr.state, "verdict": tr.verdict,
                            "owner_dist": None if tr.owner_dist in (None, float("inf")) else round(tr.owner_dist, 1),
                            "owner_away": tr.owner_dist == float("inf")} for tr in pipe.sm.tracks.values()]
            self.people_n, self.lf = len(pipe.people), pipe.lf_frac
            warm = pipe.t0 is not None and (t - pipe.t0) < cfg["background"]["warmup_s"]
            self.banner = ("camera tamper: " + str(pipe.tamper_status)) if pipe.tamper_status else \
                          ("global " + str(pipe.gc_state) + " change") if pipe.gc_state else \
                          "learning background" if warm else ""
            self.series.append({"t": round(t, 1), "fps": round(self.fps, 1), "tracks": len(self.tracks),
                                "people": self.people_n, "lf": round(self.lf, 3)})
            if vis is not None:
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    self.jpeg, self.frame_no = buf.tobytes(), self.frame_no + 1
        if recorder is not None:
            recorder.push(vis, t, new_events)

    # ------------------------------------------------------------ snapshot for the API
    def snapshot(self):
        with self.lock:
            return {"state": self.state, "error": self.error, "source": self.source, "run": self.run_name,
                    "t": round(self.t, 1), "duration": self.duration,
                    "progress": (self.t / self.duration) if self.duration else None,
                    "fps": round(self.fps, 1), "speed": self.speed, "banner": self.banner,
                    "tracks": self.tracks, "people": self.people_n, "lf": round(self.lf, 3),
                    "counts": self.counts, "components": self.components, "warnings": self.warnings,
                    "n_events": len(self.events), "series": list(self.series)[-200:]}


def _is_live(spec):
    s = str(spec).lower()
    return s.isdigit() or s.startswith(("rtsp://", "rtmp://", "http://", "https://"))


# ---------------------------------------------------------------- background jobs (eval / ablations)
class Job:
    COMMANDS = {"eval": [sys.executable, "eval/run_eval.py"],
                "ablations": [sys.executable, "eval/ablations.py"]}

    def __init__(self):
        self.proc, self.name, self.log, self.started = None, None, deque(maxlen=60), None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, name, extra=None):
        if self.running():
            raise RuntimeError("a job is already running")
        cmd = list(self.COMMANDS[name]) + list(extra or [])
        self.name, self.started = name, time.time()
        self.log.clear()
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.log.append(line.rstrip())

    def status(self):
        return {"name": self.name, "running": self.running(),
                "returncode": None if self.proc is None or self.running() else self.proc.returncode,
                "elapsed": round(time.time() - self.started, 1) if self.started else 0,
                "log": list(self.log)}


def read_run_events(out_dir, run):
    p = Path(out_dir) / run / "events.csv"
    rows = []
    if not p.exists():
        return rows
    with open(p, newline="") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            kind = r["kind"]
            try:
                t = float(r["t"])
            except ValueError:
                continue
            snap = f"{kind.replace(':', '_')}_{r['track_id']}_{t:.1f}.jpg"
            exists = (Path(out_dir) / run / "snapshots" / snap).exists()
            rows.append({"id": i, "t": t, "kind": kind, "level": level_of(kind), "track_id": r["track_id"],
                         "info": r["info"], "snapshot": f"/media/{run}/snapshots/{snap}" if exists else None})
    return rows

"""On-screen events + CSV log + snapshots (always offline); Telegram optional via requests."""
import csv, logging, os
from pathlib import Path
import cv2
from .utils import iou

log = logging.getLogger("cvmon")


class AlertManager:
    def __init__(self, c, video_name="live"):
        self.c, self.events, self.recent_alarms = c, [], []
        self.out = Path(c["out_dir"]) / video_name
        self.files = c.get("write_files", True)
        if self.files:
            (self.out / "snapshots").mkdir(parents=True, exist_ok=True)
            self.csv = self.out / "events.csv"
            with open(self.csv, "w", newline="") as f:
                csv.writer(f).writerow(["t", "kind", "track_id", "bbox", "info"])

    def emit(self, kind, t, bbox=None, track_id=None, frame=None, info=""):
        # de-duplicate: same place alarmed again within cooldown -> suppressed
        if kind == "alarm" and bbox is not None:
            for ta, ba in self.recent_alarms:
                if t - ta < self.c["cooldown_s"] and iou(bbox, ba) > 0.3:
                    return False
            self.recent_alarms.append((t, bbox))
        snap = f"{kind.replace(':', '_')}_{track_id}_{t:.1f}.jpg"
        ev = dict(t=round(float(t), 2), kind=kind, track_id=track_id, bbox=bbox, info=info,
                  snapshot=snap if (self.files and frame is not None and self.c.get("save_snapshots", True)) else None)
        self.events.append(ev)
        log.info("[%7.2fs] %-14s id=%s %s", t, kind, track_id, info)
        if self.files:
            with open(self.csv, "a", newline="") as f:
                csv.writer(f).writerow([ev["t"], kind, track_id, bbox, info])
            if frame is not None and self.c.get("save_snapshots", True):
                cv2.imwrite(str(self.out / "snapshots" / snap), frame)
        self._telegram(kind, ev, frame)
        return True

    def _telegram(self, kind, ev, frame):
        tg = self.c.get("telegram", {})
        if not tg.get("enabled") or not any(kind.startswith(k) for k in tg.get("kinds", [])):
            return
        tok, chat = os.environ.get(tg["token_env"]), os.environ.get(tg["chat_env"])
        if not tok or not chat:
            return
        try:
            import requests
            text = f"{kind} at t={ev['t']}s {ev['info']}"
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                requests.post(f"https://api.telegram.org/bot{tok}/sendPhoto", data={"chat_id": chat, "caption": text},
                              files={"photo": buf.tobytes()}, timeout=5)
            else:
                requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": text}, timeout=5)
        except Exception as e:                       # never let alerting crash the monitor
            log.warning("telegram failed: %s", e)

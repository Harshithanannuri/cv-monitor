"""Alarm clip recorder: keeps a ring buffer of the last `pre_s` seconds and, when an alarm-type
event fires, writes pre-roll + post-roll to an mp4 (evidence clip, like a real video-analytics box).
Overlapping alarms extend the same clip (capped at `max_s`)."""
from collections import deque
from pathlib import Path

import cv2


def _open_writer(path, fps, size):
    for cc in ("avc1", "H264", "mp4v"):                     # browser-playable codec first
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            return w
        w.release()
    return None


class ClipRecorder:
    def __init__(self, out_dir, fps, c):
        self.dir = Path(out_dir) / "clips"
        self.enabled = bool(c.get("enabled", True))
        self.fps = max(1.0, float(fps))
        self.pre = deque(maxlen=max(1, int(c.get("pre_s", 5) * self.fps)))
        self.post_s, self.max_s = float(c.get("post_s", 6)), float(c.get("max_s", 60))
        self.kinds = tuple(c.get("kinds", ("alarm", "removed", "tamper")))
        self.writer = self.path = None
        self.t_start = self.t_end = 0.0
        self.written = []

    def push(self, frame, t, new_events=()):
        """Call once per processed frame with the annotated frame and the events raised in it."""
        if not self.enabled or frame is None:
            return
        trig = [e for e in new_events if any(e["kind"].startswith(k) for k in self.kinds)]
        if trig:
            if self.writer is None:
                self.dir.mkdir(parents=True, exist_ok=True)
                kind = trig[0]["kind"].replace(":", "_")
                self.path = self.dir / f"{kind}_{t:.1f}s.mp4"
                h, w = frame.shape[:2]
                self.writer = _open_writer(self.path, self.fps, (w, h))
                self.t_start = t
                if self.writer is not None:
                    for f in self.pre:
                        self.writer.write(f)
            self.t_end = min(t + self.post_s, self.t_start + self.max_s)
        if self.writer is not None:
            self.writer.write(frame)
            if t >= self.t_end:
                self.close()
        self.pre.append(frame.copy())

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.written.append(self.path.name)
            self.writer = None

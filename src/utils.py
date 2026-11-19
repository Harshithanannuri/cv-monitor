"""Config loading, frame source (video or frame folder) and small geometry helpers."""
import glob, os, time
import cv2, numpy as np, yaml


def deep_update(d, u):
    for k, v in (u or {}).items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def load_config(path="config.yaml", overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return deep_update(cfg, overrides)


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx); ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def inter_frac(a, b):
    """intersection / area of the SMALLER box (detects fragments and containment)."""
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx); ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1, min(aw * ah, bw * bh))


def frac_of_a_in_b(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx); ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1, aw * ah)


class FrameSource:
    """Yields (t_seconds, frame|None). Timestamps come from frame index / fps, so every timer
    in the system is in seconds, never frames. None = unreadable frame (used by the tamper gate)."""

    def __init__(self, path, process_fps=10.0, fps_override=None, resize_width=640):
        self.path, self.width = path, resize_width
        self.is_dir = os.path.isdir(path)
        if self.is_dir:
            self.files = sorted(f for e in ("jpg", "jpeg", "png", "bmp")
                                for f in glob.glob(os.path.join(path, "*." + e)))
            self.fps, self.n = float(fps_override or 25.0), len(self.files)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            cap = cv2.VideoCapture(path)
            self.fps = float(fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        self.step = max(1, int(round(self.fps / process_fps)))

    @property
    def duration(self):
        return self.n / self.fps

    def _rs(self, f):
        h, w = f.shape[:2]
        if w == self.width:
            return f
        return cv2.resize(f, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)

    def __iter__(self):
        if self.is_dir:
            for i in range(0, self.n, self.step):
                f = cv2.imread(self.files[i])
                yield i / self.fps, (None if f is None else self._rs(f))
            return
        cap = cv2.VideoCapture(self.path); i = 0
        while cap.grab():
            if i % self.step == 0:
                ok, f = cap.retrieve()
                yield i / self.fps, (self._rs(f) if ok else None)
            i += 1
        cap.release()


class LiveSource:
    """Webcam index ("0"), RTSP/HTTP URL. Timestamps come from the wall clock; frames are dropped
    (not queued) when the pipeline is slower than the camera, so latency never builds up.
    An unreadable frame is yielded as None so the camera-health gate can raise 'disconnected'."""
    live, duration = True, None

    def __init__(self, spec, process_fps=10.0, resize_width=640):
        self.spec = int(spec) if str(spec).isdigit() else spec
        self.period, self.width = 1.0 / max(process_fps, 0.1), resize_width

    def _open(self):
        return cv2.VideoCapture(self.spec)

    def __iter__(self):
        cap, t0, last, lost_at = self._open(), time.time(), -1e9, None
        try:
            while True:
                if not cap.grab():
                    t = time.time() - t0
                    if t - last >= self.period:
                        last = t
                        yield t, None
                    if lost_at is None:
                        lost_at = t
                    if t - lost_at > 2.0:                      # try to reconnect every 2 s
                        cap.release(); time.sleep(0.5); cap = self._open(); lost_at = t
                    time.sleep(0.05)
                    continue
                lost_at = None
                t = time.time() - t0
                if t - last < self.period:
                    continue
                ok, f = cap.retrieve()
                last = t
                if not ok:
                    yield t, None
                    continue
                h, w = f.shape[:2]
                if w != self.width:
                    f = cv2.resize(f, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
                yield t, f
        finally:
            cap.release()


def is_live(spec):
    s = str(spec)
    return s.isdigit() or s.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))


def open_source(spec, process_fps=10.0, fps_override=None, resize_width=640):
    """Video file | folder of frames | webcam index | rtsp/http URL."""
    if is_live(spec):
        return LiveSource(spec, process_fps, resize_width)
    src = FrameSource(spec, process_fps, fps_override, resize_width)
    src.live = False
    return src


def reference_frame(spec, resize_width=640, at_s=1.5):
    """One frame for the zone / calibration editor."""
    src = open_source(spec, 5.0, None, resize_width)
    first = None
    for t, f in src:
        if f is None:
            continue
        first = f if first is None else first
        if t >= at_s or getattr(src, "live", False):
            return f
    return first

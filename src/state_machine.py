"""Blob-level candidate state machine:  CANDIDATE -> STATIC -> UNATTENDED -> ALARM  (-> ENDED).
Timers are in seconds. Presence authentication (template matching) keeps an object alive through
occlusion and tells 'still there' from 'taken'."""
from dataclasses import dataclass, field
import cv2, numpy as np
from .utils import iou, inter_frac

CANDIDATE, STATIC, UNATTENDED, ALARM = "CANDIDATE", "STATIC", "UNATTENDED", "ALARM"


@dataclass
class Track:
    id: int
    bbox: tuple
    mask: np.ndarray
    template: np.ndarray
    bg_template: np.ndarray
    first_seen: float
    last_seen: float
    state: str = CANDIDATE
    static_since: float = None
    owner_id: int = None
    owner_dist: float = None
    far_since: float = None
    missing_since: float = None
    last_presence: float = -1e9
    alarm_t: float = None
    luggage: bool = False
    verdict: str = ""
    ref_bbox: tuple = None          # box the blob had when its shape last changed
    stable_since: float = None      # time since which the blob shape has been stable

    def full_mask(self, shape):
        m = np.zeros(shape[:2], np.uint8)
        x, y, w, h = self.bbox
        m[y:y + h, x:x + w] = self.mask[:max(0, min(h, shape[0] - y)), :max(0, min(w, shape[1] - x))]
        return m


class StateMachine:
    def __init__(self, cfg, alerts):
        self.c, self.o, self.alerts = cfg["tracking"], cfg["owner"], alerts
        self.tracks, self.nid, self.ended_now = {}, 1, []

    def clear(self):
        self.tracks.clear()

    def _score(self, a, b):
        """match by IoU, or by containment when a blob shrank/grew inside the registered box
        (e.g. the person who dropped the bag walks away and their silhouette disappears)."""
        v = iou(a, b)
        if v >= self.c["match_iou"]:
            return v
        return 0.5 if inter_frac(a, b) >= 0.8 else 0.0

    def freeze_mask(self, shape):
        """pixels the SLOW background model must not absorb (registered candidates)."""
        if not self.tracks:
            return None
        m = np.zeros(shape[:2], np.uint8)
        for tr in self.tracks.values():
            m |= tr.full_mask(shape)
        return cv2.dilate(m, np.ones((5, 5), np.uint8))

    def _emit(self, kind, t, tr, env, info=""):
        self.alerts.emit(kind, t, tr.bbox, tr.id, env.vis_frame(), info)

    def _end(self, tr, t, reason):
        self.tracks.pop(tr.id, None)
        self.ended_now.append((tr, reason))

    def step(self, t, blobs, env):
        self.ended_now = []
        live = list(self.tracks.values())
        pairs = sorted(((self._score(tr.bbox, b.bbox), tr.id, j) for tr in live for j, b in enumerate(blobs)
                        if self._score(tr.bbox, b.bbox) > 0), reverse=True)
        tused, bused = set(), set()
        for _, tid, j in pairs:
            if tid in tused or j in bused:
                continue
            tused.add(tid); bused.add(j)
            tr = self.tracks[tid]; tr.last_seen = t
            if tr.state == CANDIDATE:                       # registered objects keep their registered box
                tr.bbox, tr.mask = blobs[j].bbox, blobs[j].mask
                if iou(tr.bbox, tr.ref_bbox) < 0.6:       # shape still changing -> restart the stability timer
                    tr.ref_bbox, tr.stable_since = tr.bbox, t
        for j, b in enumerate(blobs):
            if j in bused or any(inter_frac(tr.bbox, b.bbox) > 0.3 for tr in live):
                continue
            x, y, w, h = b.bbox
            tr = Track(self.nid, b.bbox, b.mask, env.frame.raw[y:y + h, x:x + w].copy(),
                       env.bg_img[y:y + h, x:x + w].copy(), t, t)
            tr.ref_bbox, tr.stable_since = b.bbox, t
            self.tracks[self.nid] = tr; self.nid += 1
        for tr in list(self.tracks.values()):
            self._update(tr, t, env, tr.id in tused)

    def _update(self, tr, t, env, matched):
        c, o = self.c, self.o
        if tr.state == CANDIDATE:
            if not matched and t - tr.last_seen > c["gap_s"]:
                return self._end(tr, t, "vanished")
            if matched and t - tr.stable_since >= c["static_s"]:      # verify only once the blob shape is settled
                tr.verdict, info = env.verify(tr)
                if tr.verdict == "abandoned":
                    tr.state, tr.static_since, tr.last_presence = STATIC, t, t
                    x, y, w, h = tr.bbox                    # template = what the object looks like now
                    tr.template = env.frame.raw[y:y + h, x:x + w].copy()
                    tr.owner_id = env.owner_assign(tr)
                    tr.luggage = env.is_luggage(tr)
                    self._emit("static_object", t, tr, env, f"owner={tr.owner_id} luggage={tr.luggage} {info}")
                elif tr.verdict == "removed":
                    self._emit("removed", t, tr, env, info); self._end(tr, t, "removed")
                else:
                    self._end(tr, t, "artifact")
            return
        # ---- presence authentication (also catches ghost blobs that still 'match' after removal)
        if t - tr.last_presence >= c["presence_every_s"]:
            tr.last_presence = t
            present, _, nb = env.presence(tr)
            if present:
                tr.missing_since, tr.bbox = None, nb
            elif not env.occluded(tr):
                tr.missing_since = t if tr.missing_since is None else tr.missing_since
        if tr.missing_since is not None and t - tr.missing_since >= c["remove_s"]:
            if tr.state == ALARM:
                self._emit("alarm_cleared", t, tr, env, "object taken")
            self._emit("taken", t, tr, env, f"was {tr.state}")
            return self._end(tr, t, "taken")
        # ---- owner logic
        d = env.owner_dist(tr)
        d = float("inf") if d is None else d
        tr.owner_dist = d
        if d <= o["warn_m"]:
            tr.far_since = None
            if tr.state == ALARM:
                self._emit("alarm_cleared", t, tr, env, "owner returned")
            if tr.state in (UNATTENDED, ALARM):
                tr.state = STATIC
            return
        if tr.state == STATIC:
            tr.state = UNATTENDED
            self._emit("warning", t, tr, env, f"owner {d:.1f} m away")
        if d > o["alarm_m"]:
            tr.far_since = t if tr.far_since is None else tr.far_since
            if tr.state == UNATTENDED and t - tr.far_since >= o["alarm_s"]:
                tr.state, tr.alarm_t = ALARM, t
                self._emit("alarm", t, tr, env, f"unattended {t - tr.far_since:.0f}s, owner {d:.1f} m")
        else:
            tr.far_since = None

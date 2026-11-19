"""Glue: runs the whole detection chain on one frame at a time."""
import time
from collections import defaultdict
import cv2, numpy as np

from .alerts import AlertManager
from .background import DualBackground
from .candidates import extract_candidates, shadow_mask
from .owner import Homography, OwnerTracker, appearance_hist, make_people_detector
from .preprocess import Preprocessor
from .state_machine import StateMachine, CANDIDATE, STATIC, UNATTENDED, ALARM
from .tamper import TamperDetector, global_change
from .utils import frac_of_a_in_b, iou
from .verify import edge_corr, edge_test, presence_check

COLORS = {CANDIDATE: (0, 255, 255), STATIC: (255, 200, 0), UNATTENDED: (0, 165, 255), ALARM: (0, 0, 255)}


class Pipeline:
    def __init__(self, cfg, video_name="live"):
        self.cfg, p = cfg, cfg["pipeline"]
        self.pre = Preprocessor(cfg["preprocess"])
        self.bgm = DualBackground(cfg["background"], p["bg_mode"], p["backend"])
        self.tamper = TamperDetector(cfg["tamper"]) if p["tamper"] else None
        self.homog = Homography(cfg["homography"], cfg["owner"]["px_per_m"])
        self.people_det = make_people_detector(p["people"], cfg["people"])
        self.owner = OwnerTracker(cfg["owner"], self.homog)
        self.cnn = None
        if p["verify"] in ("cnn", "both"):
            from .classifier import PairClassifier
            c = PairClassifier(cfg["verify"]["weights"], cfg["verify"]["cnn_size"])
            self.cnn = c if c.available else None
        self.alerts = AlertManager(cfg["alerts"], video_name)
        self.sm = StateMachine(cfg, self.alerts)
        self.t0 = self.last_t = self.prev_t = None
        self.n, self.people, self.bags = 0, [], []
        self.flashes = []                  # short-lived labels: REMOVED / TAKEN / SHADOW
        self.frame = self.bg_img = self.bg_gray = None
        self.tamper_since = self.tamper_status = self.gc_state = self.gc_since = None
        self._vis = None
        self.timing = defaultdict(float)
        self.zone_mask = self._zones_built = None
        self.lf_frac = 0.0

    # ------------------------------------------------------------------ env API used by StateMachine
    def vis_frame(self):
        return self._vis

    def verify(self, tr):
        v, mode = self.cfg["verify"], self.cfg["pipeline"]["verify"]
        if mode == "none":
            return "abandoned", "verify=none"
        rule = None
        if mode in ("rule", "both"):
            corr = edge_corr(self.frame.gray, self.bg_gray, tr.bbox)
            if corr >= v["artifact_corr"]:
                rule = ("artifact", corr)               # same edges as the background: shadow / light only
            else:
                rule = edge_test(self.frame.gray, self.bg_gray, tr.mask, tr.bbox, v["edge_margin"], v["ring_px"])
                rule = (rule[0] if rule[0] != "unsure" else None, rule[1])
        cnn = None
        if mode in ("cnn", "both") and self.cnn is not None:
            cnn = self.cnn.predict(self.frame.bgr, self.bg_img, tr.bbox)
        info = f"rule={rule} cnn={(cnn['label'], round(cnn['conf'], 2)) if cnn else None}"
        if rule and rule[0]:
            if cnn and cnn["conf"] >= v["cnn_conf"] and cnn["label"] != rule[0]:
                return cnn["label"], info + " (cnn override)"
            return rule[0], info
        if cnn:
            return cnn["label"], info
        return "abandoned", info + " (undecided -> abandoned)"

    def presence(self, tr):
        v = self.cfg["verify"]
        return presence_check(tr.template, self.frame.raw, tr.bbox, v["presence_pad"], v["presence_ncc"], v["sift_min_inliers"])

    def owner_assign(self, tr):
        return self.owner.assign(tr.bbox, tr.first_seen)

    def owner_dist(self, tr):
        return self.owner.distance(tr.owner_id, tr.bbox, self.last_t)

    def occluded(self, tr):
        return any(frac_of_a_in_b(tr.bbox, p.bbox) > 0.2 for p in self.people)

    def is_luggage(self, tr):
        return any(iou(tr.bbox, b) > 0.2 for b in self.bags)

    # ------------------------------------------------------------------ zones
    def _build_zones(self, shape):
        """include polygons = only look inside them; ignore polygons = never look there.
        Coordinates are in processed-frame pixels (source.resize_width)."""
        z = self.cfg.get("zones") or {}
        inc, ign = z.get("include") or [], z.get("ignore") or []
        self._zones_built = shape[:2]
        if not inc and not ign:
            self.zone_mask = None
            return
        m = np.full(shape[:2], 0 if inc else 255, np.uint8)
        for poly in inc:
            cv2.fillPoly(m, [np.int32(poly)], 255)
        for poly in ign:
            cv2.fillPoly(m, [np.int32(poly)], 0)
        self.zone_mask = m

    def _draw_zones(self, img):
        z = self.cfg.get("zones") or {}
        for key, col in (("include", (0, 200, 0)), ("ignore", (0, 0, 200))):
            for poly in z.get(key) or []:
                pts = np.int32(poly)
                ov = img.copy()
                cv2.fillPoly(ov, [pts], col)
                cv2.addWeighted(ov, 0.12, img, 0.88, 0, img)
                cv2.polylines(img, [pts], True, col, 1)

    # ------------------------------------------------------------------ main step
    def process(self, bgr, t):
        t_start = time.perf_counter()
        cfg = self.cfg
        if self.t0 is None:
            self.t0 = t
        self.last_t = t
        # ---- 2. camera health gate (runs first, on the raw frame)
        status, M = (None, None)
        if self.tamper is not None:
            status, M = self.tamper.check(None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), t)
        if bgr is None:
            self._handle_tamper(status, t)
            return self._vis
        if M is not None:                                       # small shake: stabilise
            bgr = cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        if self._zones_built != bgr.shape[:2]:
            self._build_zones(bgr.shape)
        self._vis = bgr.copy()
        self._draw_zones(self._vis)
        if self._handle_tamper(status, t):
            self._hud(t, status)
            return self._vis
        if self.tamper is not None and self.tamper.pending:      # camera problem suspected, not yet confirmed: hold object logic
            self._hud(t, "checking camera")
            return self._vis
        # ---- 1. preprocess
        fr = self.frame = self.pre(bgr)
        if not self.bgm.ready:
            self.bgm.init(fr.bgr)
        dt = 0.1 if self.prev_t is None else max(t - self.prev_t, 1e-3)
        # ---- 3. dual background
        warm = (t - self.t0) < cfg["background"]["warmup_s"]
        freeze = self.sm.freeze_mask(bgr.shape)
        t1 = time.perf_counter()
        sf, lf, self.bg_img = self.bgm.update(fr.bgr, dt, freeze, warm)
        self.bg_gray = cv2.cvtColor(self.bg_img, cv2.COLOR_BGR2GRAY)
        self.timing["background"] += time.perf_counter() - t1
        self.lf_frac = float((lf > 0).mean())
        self.prev_t = t
        self.n += 1
        if warm:
            self._hud(t, "warm-up")
            return self._vis
        # ---- people
        if self.people_det is not None and self.n % cfg["people"]["every"] == 0:
            t1 = time.perf_counter()
            self.people, self.bags = self.people_det.detect(bgr)
            if cfg["owner"].get("reid", True):
                for p in self.people:
                    p.hist = appearance_hist(bgr, p.bbox)
            self.owner.push(t, self.people)
            self.timing["people"] += time.perf_counter() - t1
        # ---- 6. global change check (lighting vs scene/camera event)
        if cfg["pipeline"]["global_check"]:
            g = global_change(lf, fr.gray, cv2.cvtColor(self.bgm.fast_image(fr.bgr), cv2.COLOR_BGR2GRAY), cfg["global_change"])
            if g:
                if g == "lighting":
                    if self.gc_state != "lighting":
                        self.alerts.emit("lighting", t, None, None, self._vis, "global lighting change, adapting")
                    self.bgm.adapt(fr.bgr, cfg["global_change"]["adapt_alpha"])
                    self.gc_since = None
                else:                                           # scene change must persist before we reset
                    self.gc_since = t if self.gc_since is None else self.gc_since
                    if t - self.gc_since >= cfg["global_change"]["scene_confirm_s"]:
                        self.alerts.emit("scene_change", t, None, None, self._vis, "re-initialising background")
                        self.bgm.init(fr.bgr); self.sm.clear(); self.gc_since = None
                self.gc_state = g
                self._hud(t, g)
                return self._vis
            self.gc_state = self.gc_since = None
        # ---- 4-5. candidates
        t1 = time.perf_counter()
        static = cv2.bitwise_and(lf, cv2.bitwise_not(sf))      # LF and not SF
        if self.zone_mask is not None:
            static = cv2.bitwise_and(static, self.zone_mask)
        shadow = shadow_mask(fr, self.bg_img, cfg["shadow"]) if cfg["pipeline"]["shadow"] else None
        if self.people:                                        # pixels inside a person box can never be luggage
            static = static.copy()
            for p in self.people:
                px, py, pw, ph = p.bbox
                static[max(0, py - 6):py + ph + 6, max(0, px - 6):px + pw + 6] = 0
        blobs = extract_candidates(static, cfg["candidates"], self.people, shadow)
        self.timing["candidates"] += time.perf_counter() - t1
        if len(blobs) > cfg["candidates"].get("max_blobs", 10):   # a flood of blobs = disturbance (camera bump, big light change)
            self._hud(t, "scene disturbance")
            return self._vis
        # ---- 7-13. verification, owner logic, state machine
        t1 = time.perf_counter()
        self.sm.step(t, blobs, self)
        for tr, reason in self.sm.ended_now:
            txt = {"removed": "REMOVED OBJECT", "taken": "OBJECT TAKEN", "artifact": "SHADOW / LIGHT"}.get(reason)
            if txt:
                self.flashes.append((tr.bbox, txt, t + 4.0, reason))
        self.flashes = [f for f in self.flashes if f[2] > t]
        for tr, reason in self.sm.ended_now:                   # let the models forget ghosts / taken objects
            if reason in ("removed", "taken", "artifact"):
                self.bgm.reset_region(fr.bgr, tr.full_mask(bgr.shape))
        self.timing["logic"] += time.perf_counter() - t1
        self._draw(t)
        self.timing["total"] += time.perf_counter() - t_start
        return self._vis

    def _handle_tamper(self, status, t):
        if status:
            if self.tamper_status != status:
                self.tamper_status, self.tamper_since = status, t
                self.alerts.emit(f"tamper:{status}", t, None, None, self._vis, "camera health gate")
            if t - self.tamper_since > self.cfg["tamper"]["rebase_s"] and status == "shifted":
                self.tamper.rebase(); self.bgm = DualBackground(self.cfg["background"], self.cfg["pipeline"]["bg_mode"],
                                                                self.cfg["pipeline"]["backend"]); self.sm.clear()
                self.tamper_status = None
            return True
        self.tamper_status = None
        return False

    # ------------------------------------------------------------------ overlay
    def _hud(self, t, text=None):
        if self._vis is None:
            return
        if text:
            cv2.rectangle(self._vis, (0, 0), (self._vis.shape[1], 26), (0, 0, 160) if text not in ("warm-up",) else (80, 80, 80), -1)
            cv2.putText(self._vis, f"{text.upper()}   t={t:.1f}s", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw(self, t):
        v = self._vis
        for p in self.people:
            x, y, w, h = p.bbox
            cv2.rectangle(v, (x, y), (x + w, y + h), (0, 200, 0), 1)
            cv2.putText(v, f"P{p.id}", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        for tr in self.sm.tracks.values():
            x, y, w, h = tr.bbox
            col = COLORS[tr.state]
            cv2.rectangle(v, (x, y), (x + w, y + h), col, 2)
            d = "" if tr.owner_dist is None else (" owner away" if tr.owner_dist == float("inf") else f" {tr.owner_dist:.1f}m")
            if tr.state == "CANDIDATE":
                lab = f"#{tr.id} checking {max(0.0, t - tr.stable_since):.1f}/{self.cfg['tracking']['static_s']:.0f}s"
            else:
                lab = f"#{tr.id} {'LUGGAGE' if tr.luggage else 'ABANDONED OBJECT'} - {tr.state}{d}"
            cv2.putText(v, lab, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        fcol = {"removed": (255, 100, 200), "taken": (190, 190, 190), "artifact": (200, 200, 0)}
        for (fx, fy, fw, fh), txt, _, reason in self.flashes:
            cv2.rectangle(v, (fx, fy), (fx + fw, fy + fh), fcol[reason], 2)
            cv2.putText(v, txt, (fx, max(12, fy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fcol[reason], 2)
        self._hud(t, "ALARM" if any(tr.state == ALARM for tr in self.sm.tracks.values()) else None)

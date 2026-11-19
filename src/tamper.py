"""Camera-health gate: blocked / blurred / frozen / shifted / disconnected, plus the global-change
check (lighting change vs scene/camera event)."""
import math
import cv2, numpy as np
from .verify import edge_corr


class TamperDetector:
    def __init__(self, c):
        self.c = c
        self.orb = cv2.ORB_create(c["orb_features"])
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reset()

    def reset(self):
        self.prev = self.ref = self.ref_des = self.ref_kp = None
        self.ref_t, self.blur_base, self.miss_since = None, None, None
        self.bad, self.last_shift_t = {}, -1e9
        self.shift_flag, self.M = False, None

    def _confirm(self, kind, cond, t):
        if not cond:
            self.bad.pop(kind, None)
            return False
        return t - self.bad.setdefault(kind, t) >= self.c["confirm_s"][kind]

    def _block_shift(self, ref, g):
        """Phase correlation in a 2x3 grid of blocks. A camera move shifts EVERY block the same way,
        a walking person disturbs only some. We trust the largest cluster of agreeing blocks (>=3 of 6).
        returns ((dx, dy), cluster_size) or (None, 0)."""
        H, W = g.shape
        bh, bw = H // 2, W // 3
        if getattr(self, "_win_shape", None) != (bh, bw):
            self._win, self._win_shape = cv2.createHanningWindow((bw, bh), cv2.CV_32F), (bh, bw)
        vecs = []
        for i in range(2):
            for j in range(3):
                sl = (slice(i * bh, (i + 1) * bh), slice(j * bw, (j + 1) * bw))
                (dx, dy), resp = cv2.phaseCorrelate(ref[sl], g[sl], self._win)
                if resp >= self.c["phase_resp"]:
                    vecs.append((dx, dy))
        best = []
        for v in vecs:
            cl = [u for u in vecs if math.hypot(u[0] - v[0], u[1] - v[1]) <= 4.0]
            if len(cl) > len(best):
                best = cl
        if len(best) < 3:
            return None, 0
        return (float(np.median([u[0] for u in best])), float(np.median([u[1] for u in best]))), len(best)

    def _shift(self, gray, t):
        """Translation: block-wise phase correlation with consensus (robust to people and to repeating
        floor tiles). Rotation / zoom: ORB + RANSAC affine, only trusted when most matches are inliers."""
        c = self.c
        g = gray.astype(np.float32)
        if self.ref is None:
            self.ref, self.ref_t = g, t
            self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, None)
            return
        if t - self.last_shift_t < c["shift_every_s"]:
            return
        self.last_shift_t = t
        est, n = self._block_shift(self.ref, g)
        self.M = None
        if est is None:
            return                                    # no consensus: cannot conclude, keep previous state
        shift = math.hypot(*est)
        big = shift > c["shift_px"]
        if 1.0 < shift <= c["shift_px"] and n >= 4:
            self.M = np.float32([[1, 0, -est[0]], [0, 1, -est[1]]])       # undo the small shake
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is not None and self.ref_des is not None:
            ms = self.bf.match(self.ref_des, des)
            if len(ms) >= c["min_inliers"]:
                dst = np.float32([self.ref_kp[m.queryIdx].pt for m in ms])
                src = np.float32([kp[m.trainIdx].pt for m in ms])
                A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if A is not None and inl.sum() >= c["min_inliers"] and inl.mean() >= 0.6:
                    big = big or abs(math.degrees(math.atan2(A[1, 0], A[0, 0]))) > c["rot_deg"]
        self.shift_flag = big
        if big:
            self.M = None
        elif shift < 1.0 and n >= 4 and t - self.ref_t > c["ref_refresh_s"]:
            self.ref, self.ref_t = g, t
            self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, None)

    @property
    def pending(self):
        """some health condition is suspected but not yet confirmed -> hold object logic."""
        return bool(self.bad)

    def check(self, gray, t):
        """returns (status|None, stabilising affine|None). status in blocked/blurred/frozen/shifted/disconnected"""
        c = self.c
        if gray is None:
            self.miss_since = t if self.miss_since is None else self.miss_since
            return ("disconnected" if t - self.miss_since >= c["confirm_s"]["disconnected"] else None), None
        self.miss_since = None
        mean, std = (float(v[0][0]) for v in cv2.meanStdDev(gray))
        blocked = mean < c["dark_mean"] or std < c["low_std"]
        lap = float(cv2.Laplacian(gray.astype(np.float32) / (mean + 1e-3) * 100, cv2.CV_32F).var())   # gain-invariant sharpness
        if self.blur_base is None:
            self.blur_base = lap
        blurred = (not blocked) and lap < c["blur_ratio"] * self.blur_base
        if not blocked and not blurred:
            self.blur_base = 0.98 * self.blur_base + 0.02 * lap       # learn "healthy sharpness" slowly
        frozen = self.prev is not None and float(cv2.absdiff(gray, self.prev).mean()) < c["frozen_eps"]
        self.prev = gray
        if not blocked and not blurred:
            self._shift(gray, t)
        flags = [("blocked", blocked), ("blurred", blurred), ("frozen", frozen), ("shifted", self.shift_flag)]
        status = None
        for kind, cond in flags:
            if self._confirm(kind, cond, t) and status is None:
                status = kind
        return status, (self.M if status is None else None)

    def rebase(self):
        self.reset()


def global_change(lf_mask, gray_cur, gray_bg, c):
    """If a large part of the frame is foreground: lighting change (edges unchanged -> high edge
    correlation) or scene/camera event (edges changed). returns None | 'lighting' | 'scene_change'."""
    if (lf_mask > 0).mean() < c["fg_frac"]:
        return None
    return "lighting" if edge_corr(gray_cur, gray_bg) >= c["edge_corr"] else "scene_change"

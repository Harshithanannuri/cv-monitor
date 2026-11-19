"""Dual background model (Porikli-style): a FAST model absorbs stopped objects within seconds,
a SLOW model keeps remembering the empty scene.
    moving            = SF and LF
    static candidate  = LF and not SF      <- what we look for
    uncovered bg      = SF and not LF
Backends: 'ra' = own running average (supports freezing regions), 'mog2' / 'knn' = OpenCV."""
import math
import cv2, numpy as np


def _grad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 4.0


def foreground(img, bg, thr, grad_thr=None):
    """colour difference OR edge difference (edge channel catches objects coloured like the floor)."""
    fg = cv2.absdiff(img, bg).max(axis=2) > thr
    if grad_thr:
        fg |= np.abs(_grad(img) - _grad(bg)) > grad_thr
    return fg.astype(np.uint8) * 255


class RunningAverage:
    """Own implementation: B <- B + alpha*(I-B), alpha = 1-exp(-dt/tau) (time based, not frame based)."""

    def __init__(self, tau):
        self.tau, self.bg = tau, None

    def init(self, img):
        self.bg = img.astype(np.float32)

    def update(self, img, dt, freeze=None, min_alpha=0.0):
        a = max(1 - math.exp(-dt / self.tau), min_alpha)
        d = img.astype(np.float32) - self.bg
        if freeze is not None:
            d *= (freeze == 0)[..., None]          # frozen pixels are never updated
        self.bg += a * d

    def image(self):
        return np.clip(self.bg, 0, 255).astype(np.uint8)


class DualBackground:
    def __init__(self, c, mode="dual", backend="ra"):
        self.c, self.mode, self.backend, self.n = c, mode, backend, 0
        self.grad = c["grad_thr"] if c["use_gradient"] else None
        if backend == "ra":
            self.slow, self.fast = RunningAverage(c["tau_slow_s"]), RunningAverage(c["tau_fast_s"])
        else:
            mk = cv2.createBackgroundSubtractorMOG2 if backend == "mog2" else cv2.createBackgroundSubtractorKNN
            kw = dict(detectShadows=False)
            self.slow, self.fast = mk(history=2000, **kw), mk(history=100, **kw)

    @property
    def ready(self):
        return self.n > 0

    def init(self, img):
        self.n = 1
        if self.backend == "ra":
            self.slow.init(img); self.fast.init(img)
        else:
            self.slow.apply(img, learningRate=1.0); self.fast.apply(img, learningRate=1.0)

    def _bg_image(self, img):
        if self.backend == "ra":
            return self.slow.image()
        try:
            b = self.slow.getBackgroundImage()
            return b if b is not None and b.size else img
        except cv2.error:
            return img

    def update(self, img, dt, freeze=None, warm=False):
        """returns (sf, lf, slow_background_image). Models are updated AFTER the masks are computed."""
        if not self.ready:
            self.init(img)
        self.n += 1
        c = self.c
        min_a = 1.0 / self.n if warm else 0.0
        if self.backend == "ra":
            bg = self.slow.image()
            lf = foreground(img, bg, c["diff_thr"], self.grad)
            sf = foreground(img, self.fast.image(), c["diff_thr"], self.grad) if self.mode == "dual" \
                else np.zeros_like(lf)
            self.slow.update(img, dt, freeze, min_a)
            self.fast.update(img, dt, None, min_a)
            return sf, lf, bg
        a_s = max(1 - math.exp(-dt / c["tau_slow_s"]), min_a)
        a_f = max(1 - math.exp(-dt / c["tau_fast_s"]), min_a)
        lf = self.slow.apply(img, learningRate=a_s)
        sf = self.fast.apply(img, learningRate=a_f) if self.mode == "dual" else np.zeros_like(lf)
        return (sf > 200).astype(np.uint8) * 255, (lf > 200).astype(np.uint8) * 255, self._bg_image(img)

    def fast_image(self, img):
        """reference for the global-change test: contains stopped objects, so a few objects do not
        masquerade as a scene change."""
        if self.backend == "ra":
            return self.fast.image()
        try:
            b = self.fast.getBackgroundImage()
            return b if b is not None and b.size else img
        except cv2.error:
            return img

    def adapt(self, img, alpha):
        """fast re-adaptation after a global lighting change."""
        if self.backend == "ra":
            for m in (self.slow, self.fast):
                m.bg += alpha * (img.astype(np.float32) - m.bg)
        else:
            self.slow.apply(img, learningRate=alpha); self.fast.apply(img, learningRate=alpha)

    def reset_region(self, img, mask):
        """after an object ended (taken / ghost): make both models equal to the current view there."""
        if self.backend == "ra":
            m = mask > 0
            self.slow.bg[m] = img[m]; self.fast.bg[m] = img[m]

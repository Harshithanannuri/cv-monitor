"""Step 1: CLAHE on luminance (lighting robustness), Gaussian blur (noise), HSV for shadow test."""
from types import SimpleNamespace
import cv2


class Preprocessor:
    def __init__(self, c):
        self.use_clahe = c.get("clahe", False)
        self.clahe = cv2.createCLAHE(clipLimit=c["clahe_clip"], tileGridSize=(c["clahe_tile"],) * 2)
        self.k = c["blur_ksize"] | 1

    def __call__(self, bgr):
        eq = bgr
        if self.use_clahe:      # off by default: CLAHE couples distant pixels (tile histograms) -> see ablation
            l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
            eq = cv2.cvtColor(cv2.merge([self.clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
        proc = cv2.GaussianBlur(eq, (self.k, self.k), 0)
        return SimpleNamespace(raw=bgr, bgr=proc,
                               gray=cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY),
                               hsv=cv2.cvtColor(proc, cv2.COLOR_BGR2HSV))

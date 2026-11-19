"""Shadow removal (HSV chromaticity + gradient/texture test) and blob extraction."""
from dataclasses import dataclass
import cv2, numpy as np
from .utils import frac_of_a_in_b


@dataclass
class Blob:
    bbox: tuple          # x, y, w, h
    mask: np.ndarray     # bbox-sized uint8 (0/255)
    area: float


def _local_grad(gray, k=7):
    g = gray.astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    return cv2.blur(mag, (k, k))


def shadow_mask(fr, bg_bgr, p):
    """A cast shadow darkens a surface but keeps its hue, saturation and TEXTURE (Lambertian model).
    A real object changes the texture. Pixel = shadow if all of the following hold:
      V_ratio in [v_lo, v_hi], |dH| small (only where saturated), |dS| small, local gradient preserved."""
    hf = fr.hsv.astype(np.float32)
    hb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_ratio = (hf[..., 2] + 1) / (hb[..., 2] + 1)
    dh = np.abs(hf[..., 0] - hb[..., 0]); dh = np.minimum(dh, 180 - dh)
    hue_ok = (dh <= p["h_tol"]) | (hb[..., 1] < p["sat_min_for_hue"])
    sat_ok = np.abs(hf[..., 1] - hb[..., 1]) <= p["s_tol"]
    gf = _local_grad(fr.gray); gb = _local_grad(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY))
    tex_ok = (gf >= p["grad_lo"] * gb - p["grad_bias"]) & (gf <= p["grad_hi"] * gb + p["grad_bias"])
    return (v_ratio >= p["v_lo"]) & (v_ratio <= p["v_hi"]) & hue_ok & sat_ok & tex_ok


def extract_candidates(static_mask, c, people=(), shadow=None):
    """static_mask (0/255) -> list[Blob]. Drops shadows, tiny blobs, huge blobs, and blobs that are
    mostly inside a detected person box (a person standing still is not luggage)."""
    m = static_mask.copy()
    if shadow is not None:
        m[shadow] = 0
    ko, kc = c["open_k"], c["close_k"]
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = m.shape
    blobs = []
    for cn in cnts:
        area = cv2.contourArea(cn)
        if area < c["min_area_px"] or area > c["max_area_frac"] * H * W:
            continue
        x, y, w, h = cv2.boundingRect(cn)
        if h / max(w, 1) > c.get("max_aspect", 3.0):         # very tall thin blob = person silhouette, not luggage
            continue
        if any(frac_of_a_in_b((x, y, w, h), p.bbox) > c["person_overlap"] for p in people):
            continue
        mk = np.zeros((h, w), np.uint8)
        cv2.drawContours(mk, [cn - np.array([x, y])], -1, 255, -1)
        blobs.append(Blob((x, y, w, h), mk, area))
    return blobs

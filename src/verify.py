"""Abandoned-vs-removed edge test, edge-map correlation, SIFT matching, presence authentication."""
import cv2, numpy as np


def _mag(gray):
    g = gray.astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 4.0


def edge_test(gray_cur, gray_bg, mask, bbox, margin=0.12, ring=3):
    """Classic abandoned/removed rule: compare edge strength ALONG THE BLOB BOUNDARY in the current
    frame vs the background. Object added  -> boundary edges stronger in the current frame.
    Object removed -> boundary edges stronger in the background (the old object's outline).
    returns (verdict, score) with score = (Ecur-Ebg)/(Ecur+Ebg) in [-1, 1]."""
    x, y, w, h = bbox
    pad = ring * 2
    H, W = gray_cur.shape
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    m[y - y0:y - y0 + h, x - x0:x - x0 + w] = mask[:min(h, m.shape[0] - (y - y0)), :min(w, m.shape[1] - (x - x0))]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    boundary = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k) > 0
    if boundary.sum() < 10:
        return "unsure", 0.0
    ec = _mag(gray_cur[y0:y1, x0:x1])[boundary].mean()
    eb = _mag(gray_bg[y0:y1, x0:x1])[boundary].mean()
    s = float((ec - eb) / (ec + eb + 1e-6))
    return ("abandoned" if s > margin else "removed" if s < -margin else "unsure"), s


def edge_corr(gray_a, gray_b, bbox=None):
    """Normalised correlation of Sobel edge maps. High => same scene content (lighting / shadow only)."""
    if bbox is not None:
        x, y, w, h = bbox
        gray_a, gray_b = gray_a[y:y + h, x:x + w], gray_b[y:y + h, x:x + w]
    if gray_a.size == 0:
        return 1.0
    # divide by mean brightness: edge maps become invariant to a global gain (lighting change)
    na = cv2.GaussianBlur(gray_a.astype(np.float32) / (gray_a.mean() + 1e-3) * 128, (5, 5), 0)
    nb = cv2.GaussianBlur(gray_b.astype(np.float32) / (gray_b.mean() + 1e-3) * 128, (5, 5), 0)
    a, b = _mag(na), _mag(nb)
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else 1.0


_sift = None


def sift_match(a_bgr, b_bgr, ratio=0.75):
    """SIFT + Lowe ratio + RANSAC homography. returns (inliers, good_matches)."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create()
    ga, gb = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (a_bgr, b_bgr))
    ka, da = _sift.detectAndCompute(ga, None)
    kb, db = _sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return 0, 0
    good = [p[0] for p in cv2.BFMatcher().knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 4:
        return 0, len(good)
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return (int(inl.sum()) if inl is not None else 0), len(good)


def presence_check(template, frame_bgr, bbox, pad=12, ncc_thr=0.55, sift_min=8):
    """Presence authentication: is the registered template still at (or near) its place?
    Normalised cross-correlation in a padded search window; SIFT as a second opinion.
    returns (present, score, new_bbox)."""
    x, y, w, h = bbox
    H, W = frame_bgr.shape[:2]
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    region = frame_bgr[y0:y1, x0:x1]
    th, tw = template.shape[:2]
    if region.shape[0] < th or region.shape[1] < tw:
        return True, 0.0, bbox
    res = cv2.matchTemplate(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    nb = (x0 + loc[0], y0 + loc[1], tw, th)
    if mx >= ncc_thr:
        return True, float(mx), nb
    inl, _ = sift_match(template, region)
    return inl >= sift_min, float(mx), bbox

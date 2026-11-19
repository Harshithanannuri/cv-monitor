"""Synthetic (current crop, background crop) pairs for the 6-channel classifier.
  abandoned : bg = floor,            cur = floor + object (+ contact shadow)
  removed   : bg = floor + object,   cur = floor            (also covers 'ghost': old object left)
  artifact  : bg = floor,            cur = floor + cast shadow / lighting spot / gain change
Split is BY BACKGROUND (floor id / image file), the analogue of 'by video', so no leakage.
Real assets are optional:  --bg_dir frames from your own clips   --objects_dir RGBA PNG cut-outs."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from src.classifier import CLASSES, crop_pair

H, W = 360, 640
C = lambda v: tuple(float(z) for z in np.ravel(v))      # cv2 wants plain floats


def make_floor(rng):
    base = np.empty((H, W, 3), np.float32); base[:] = rng.uniform(70, 190, 3)
    lo = cv2.resize(rng.normal(0, rng.uniform(5, 18), (9, 16)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    base += lo[..., None]
    step = int(rng.integers(40, 110)); col = float(base.mean() - rng.uniform(15, 40))
    if rng.random() < 0.8:
        for x in range(0, W, step): cv2.line(base, (x, 0), (x, H), (col,) * 3, 1)
        for y in range(0, H, step): cv2.line(base, (0, y), (W, y), (col,) * 3, 1)
    base += rng.normal(0, rng.uniform(1, 4), (H, W, 1)).astype(np.float32)
    return np.clip(base, 0, 255)


def make_object(rng):
    w, h = int(rng.integers(26, 90)), int(rng.integers(22, 80))
    col = rng.uniform(20, 235, 3)
    ob = np.empty((h, w, 3), np.float32); ob[:] = col
    for i in range(0, w, int(rng.integers(5, 12))):               # stripes / texture
        if rng.random() < 0.7: cv2.line(ob, (i, 0), (i, h), C(np.clip(col * rng.uniform(.5, 1.4), 0, 255)), 2)
    ob += rng.normal(0, 5, ob.shape)
    a = np.zeros((h, w), np.float32)
    if rng.random() < 0.5: cv2.rectangle(a, (0, 0), (w - 1, h - 1), 1.0, -1)
    else: cv2.ellipse(a, (w // 2, h // 2), (w // 2 - 1, h // 2 - 1), 0, 0, 360, 1.0, -1)
    cv2.rectangle(ob, (0, 0), (w - 1, h - 1), C(col * 0.4), 2)
    return np.clip(ob, 0, 255), a


def load_objects(d):
    out = []
    for p in glob.glob(os.path.join(d or "", "*.png")):
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            out.append((im[..., :3].astype(np.float32), im[..., 3].astype(np.float32) / 255))
    return out


def soft_ellipse(cx, cy, rx, ry):
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (max(2, int(rx)), max(2, int(ry))), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (15, 15), 0)


def with_object(floor, ob, a, x, y, rng):
    img = floor.copy(); h, w = a.shape
    sh = soft_ellipse(x + w / 2, y + h, w * 0.6, max(4, h * 0.12))
    img *= (1 - (1 - rng.uniform(0.6, 0.85)) * sh)[..., None]
    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - a[..., None]) + ob * a[..., None]
    return img


def sample(rng, floor, cls, objs, size):
    if cls in ("abandoned", "removed"):
        ob, a = objs[rng.integers(len(objs))] if objs and rng.random() < 0.5 else make_object(rng)
        if rng.random() < 0.5: ob, a = ob[:, ::-1].copy(), a[:, ::-1].copy()
        h, w = a.shape; x, y = int(rng.integers(20, W - w - 20)), int(rng.integers(20, H - h - 20))
        withobj = with_object(floor, ob, a, x, y, rng)
        bg, cur = (floor, withobj) if cls == "abandoned" else (withobj, floor)
        j = lambda v: int(v + rng.integers(-3, 4)); bbox = (j(x), j(y), w + int(rng.integers(0, 6)), h + int(rng.integers(0, 6)))
    else:
        w, h = int(rng.integers(30, 110)), int(rng.integers(20, 70)); x, y = int(rng.integers(20, W - w - 20)), int(rng.integers(20, H - h - 20))
        kind = rng.choice(["shadow", "spot", "gain"])
        if kind == "shadow":
            m = soft_ellipse(x + w / 2, y + h / 2, w / 2, h / 2); gain = 1 - (1 - rng.uniform(0.5, 0.85)) * m
        elif kind == "spot":
            m = soft_ellipse(x + w / 2, y + h / 2, w / 2, h / 2); gain = 1 + rng.uniform(-0.4, 0.4) * m
        else:
            gain = np.full((H, W), rng.uniform(0.6, 1.35), np.float32)
        bg, cur, bbox = floor, floor * gain[..., None], (x, y, w, h)
    noisy = lambda im: cv2.GaussianBlur(np.clip(im + rng.normal(0, rng.uniform(1, 4), im.shape), 0, 255).astype(np.uint8), (5, 5), 0)
    return crop_pair(noisy(cur), noisy(bg), bbox, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth"); ap.add_argument("--floors", type=int, default=60)
    ap.add_argument("--per_floor", type=int, default=20, help="samples per class per floor")
    ap.add_argument("--size", type=int, default=96); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg_dir"); ap.add_argument("--objects_dir"); a = ap.parse_args()
    rng = np.random.default_rng(a.seed); objs = load_objects(a.objects_dir)
    real = [cv2.resize(cv2.imread(p), (W, H)).astype(np.float32) for p in sorted(glob.glob(os.path.join(a.bg_dir or "", "*.jpg")))]
    floors = real + [None] * a.floors
    counts = {}
    for i, fl in enumerate(floors):
        frac = i / len(floors); split = "train" if frac < 0.7 else "val" if frac < 0.85 else "test"
        floor = fl if fl is not None else make_floor(rng)
        for cls in CLASSES:
            d = os.path.join(a.out, split, cls); os.makedirs(d, exist_ok=True)
            for k in range(a.per_floor):
                cur, bg = sample(rng, floor, cls, objs, a.size)
                cv2.imwrite(os.path.join(d, f"{i:03d}_{k:03d}_cur.png"), cur); cv2.imwrite(os.path.join(d, f"{i:03d}_{k:03d}_bg.png"), bg)
                counts[f"{split}/{cls}"] = counts.get(f"{split}/{cls}", 0) + 1
    json.dump(counts, open(os.path.join(a.out, "counts.json"), "w"), indent=1); print(counts)


if __name__ == "__main__":
    main()

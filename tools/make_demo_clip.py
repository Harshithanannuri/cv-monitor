"""Procedural test clip (no downloads needed). 100 s, 640x360, 25 fps. Scenario:
  t=3-14  person A walks in, drops a bag (t=8), walks away     -> ABANDONED event
  t=22-32 person B takes a pre-existing box (t=25)             -> REMOVED event (ghost)
  t=40    sudden lighting change (x0.55 for 6 s)               -> must NOT alarm
  t=62-72 person C picks up the bag                            -> bag 'taken', alarm cleared
  t=75-85 camera pushed 60 px sideways                         -> TAMPER (shifted)
  t=90-96 lens covered (black)                                 -> TAMPER (blocked)
Moving people cast soft shadows. Writes data/clips/demo.mp4 and data/labels.csv (demo rows)."""
import argparse, csv, os
import cv2, numpy as np

W, H, FPS, DUR = 640, 360, 25, 100


def floor(rng):
    base = np.full((H, W, 3), (120, 128, 135), np.float32)
    lo = cv2.resize(rng.normal(0, 14, (9, 16)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    base += lo[..., None]
    for x in range(0, W, 64):
        cv2.line(base, (x, 0), (x, H), (95, 100, 105), 1)
    for y in range(0, H, 60):
        cv2.line(base, (0, y), (W, y), (95, 100, 105), 1)
    base += rng.normal(0, 4, (H, W, 1)).astype(np.float32)
    return base


def shadow(img, cx, cy, rx, ry, k=0.72):
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (15, 15), 0)
    img *= (1 - (1 - k) * m)[..., None]


def person(img, x, y, col=(60, 80, 160)):             # x,y = feet
    shadow(img, x + 12, y, 30, 8)
    cv2.rectangle(img, (int(x - 14), int(y - 85)), (int(x + 14), int(y)), col, -1)
    cv2.circle(img, (int(x), int(y - 98)), 11, (150, 170, 200), -1)


def bag(img, x, y):                                    # x,y = bottom-left
    shadow(img, x + 22, y + 2, 28, 6, 0.75)
    cv2.rectangle(img, (x, y - 28), (x + 44, y), (40, 40, 150), -1)
    cv2.rectangle(img, (x, y - 28), (x + 44, y), (15, 15, 60), 2)
    for i in range(4, 44, 8):
        cv2.line(img, (x + i, y - 26), (x + i, y - 2), (70, 70, 210), 2)
    cv2.line(img, (x + 12, y - 28), (x + 16, y - 38), (15, 15, 60), 2); cv2.line(img, (x + 32, y - 28), (x + 28, y - 38), (15, 15, 60), 2)
    cv2.line(img, (x + 16, y - 38), (x + 28, y - 38), (15, 15, 60), 2)


def box(img, x, y):
    shadow(img, x + 20, y + 2, 26, 6, 0.75)
    cv2.rectangle(img, (x, y - 34), (x + 40, y), (60, 150, 90), -1)
    cv2.rectangle(img, (x, y - 34), (x + 40, y), (20, 70, 40), 2)
    cv2.line(img, (x, y - 17), (x + 40, y - 17), (20, 70, 40), 2)


def lerp(t, t0, t1, a, b):
    return a + (b - a) * np.clip((t - t0) / (t1 - t0), 0, 1)


def main(out="data/clips/demo.mp4", labels="data/labels.csv"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rng = np.random.default_rng(1)
    bg = floor(rng)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    BOX = (470, 250); BAG = (300, 262)
    for i in range(FPS * DUR):
        t = i / FPS
        f = bg.copy()
        if t < 25: box(f, *BOX)
        if 8 <= t < 63: bag(f, *BAG)
        # A: in 3->7 to x=300, wait, leave 10->14 to the right
        if 3 <= t < 14:
            ax = lerp(t, 3, 7, -20, 285) if t < 10 else lerp(t, 10, 14, 285, 700)
            person(f, ax, 262)
        # B: walks to box, takes it, leaves left
        if 22 <= t < 32:
            bx = lerp(t, 22, 25, 700, 500) if t < 26 else lerp(t, 26, 32, 500, -40)
            person(f, bx, 262, (140, 90, 60))
            if t >= 25: box(f, int(bx - 45), 262 - 20 if False else 250)
        # C: picks up the bag
        if 62 <= t < 72:
            cx = lerp(t, 62, 65, -20, 300) if t < 66 else lerp(t, 66, 72, 300, 700)
            person(f, cx, 262, (90, 140, 70))
            if t >= 66: bag(f, int(cx - 20), 225)
        if 40 <= t < 46: f *= 0.55
        if 75 <= t < 85:
            f = cv2.warpAffine(f, np.float32([[1, 0, 60], [0, 1, 15]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
        if 90 <= t < 96: f *= 0.02
        f += rng.normal(0, 2.5, f.shape).astype(np.float32)
        vw.write(np.clip(f, 0, 255).astype(np.uint8))
    vw.release()
    rows = [("demo.mp4", "abandoned", 14, 63), ("demo.mp4", "removed", 25, 32),
            ("demo.mp4", "tamper", 75, 85), ("demo.mp4", "tamper", 90, 96)]
    new = not os.path.exists(labels)
    existing = [] if new else [r for r in csv.reader(open(labels))][1:]
    existing = [r for r in existing if r[0] != "demo.mp4"]
    with open(labels, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["video", "type", "t_start", "t_end"])
        w.writerows(existing); w.writerows(rows)
    print("wrote", out, "and", labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/clips/demo.mp4"); main(**vars(ap.parse_args()))

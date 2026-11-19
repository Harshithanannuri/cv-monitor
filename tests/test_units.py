import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from types import SimpleNamespace
from src.candidates import shadow_mask
from src.owner import Homography
from src.tamper import TamperDetector
from src.utils import load_config
from src.verify import edge_corr, edge_test, presence_check

rng = np.random.default_rng(0)
CFG = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))


def floor():
    f = cv2.GaussianBlur(rng.normal(120, 25, (120, 160)).astype(np.float32), (0, 0), 1.2)
    return np.clip(f, 0, 255).astype(np.uint8)


def with_square(g):
    o = g.copy(); cv2.rectangle(o, (60, 40), (100, 80), 30, -1); cv2.rectangle(o, (60, 40), (100, 80), 220, 2)
    return o


def test_edge_test_added_vs_removed():
    bg = floor(); cur = with_square(bg)
    mask = np.zeros((44, 44), np.uint8) + 255
    assert edge_test(cur, bg, mask, (58, 38, 44, 44))[0] == "abandoned"
    assert edge_test(bg, cur, mask, (58, 38, 44, 44))[0] == "removed"


def test_edge_corr_lighting_vs_object():
    g = floor()
    assert edge_corr((g * 0.6).astype(np.uint8), g) > 0.9           # gain only
    assert edge_corr(with_square(g), g, (50, 30, 60, 60)) < 0.85    # real object


def test_shadow_vs_object():
    bgr = cv2.cvtColor(floor(), cv2.COLOR_GRAY2BGR); bgr = (bgr * np.float32([1.0, 0.95, 0.9])).astype(np.uint8)
    shadow = (bgr * 0.7).astype(np.uint8)
    fr = lambda im: SimpleNamespace(hsv=cv2.cvtColor(im, cv2.COLOR_BGR2HSV), gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    assert shadow_mask(fr(shadow), bgr, CFG["shadow"]).mean() > 0.8
    obj = bgr.copy(); cv2.rectangle(obj, (40, 30), (110, 90), (30, 30, 200), -1)
    assert shadow_mask(fr(obj), bgr, CFG["shadow"])[40:90, 50:100].mean() < 0.2


def test_tamper_blocked_frozen_blur():
    c = CFG["tamper"]; td = TamperDetector(c); g = floor(); st = None
    for i in range(40):
        st, _ = td.check(g + rng.integers(0, 3, g.shape, dtype=np.uint8), i * 0.1)
    assert st is None
    for i in range(40, 90):
        st, _ = td.check(np.zeros_like(g), i * 0.1)
    assert st == "blocked"
    td = TamperDetector(c)
    for i in range(80):
        st, _ = td.check(g.copy(), i * 0.1)
    assert st == "frozen"
    td = TamperDetector(c)
    for i in range(20): td.check(with_square(g) + rng.integers(0, 3, g.shape, dtype=np.uint8), i * 0.1)
    for i in range(20, 70): st, _ = td.check(cv2.GaussianBlur(with_square(g), (0, 0), 6) + rng.integers(0, 2, g.shape, dtype=np.uint8), i * 0.1)
    assert st == "blurred"


def test_homography_metres():
    h = Homography({"image_points": [[0, 0], [400, 0], [400, 200], [0, 200]], "world_points": [[0, 0], [4, 0], [4, 2], [0, 2]]}, 60)
    assert abs(h.dist((0, 0), (200, 0)) - 2.0) < 1e-3


def test_presence_check():
    bg = cv2.cvtColor(floor(), cv2.COLOR_GRAY2BGR); cur = bg.copy()
    cv2.rectangle(cur, (60, 40), (100, 80), (40, 40, 200), -1); cv2.rectangle(cur, (65, 45), (75, 55), (255, 255, 255), -1)
    tpl = cur[40:80, 60:100].copy()
    assert presence_check(tpl, cur, (62, 42, 40, 40))[0]
    assert not presence_check(tpl, bg, (60, 40, 40, 40))[0]


def test_demo_clip_end_to_end():
    clip = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/clips/demo.mp4")
    if not os.path.exists(clip):
        import pytest; pytest.skip("run tools/make_demo_clip.py first")
    from eval.run_eval import run_video
    cfg = load_config(os.path.join(os.path.dirname(clip), "../../config.yaml"), {"pipeline": {"people": "none"}, "alerts": {"write_files": False}})
    ev, *_ = run_video(cfg, clip, max_seconds=50)
    kinds = [e["kind"] for e in ev]
    assert "alarm" in kinds and "removed" in kinds and "scene_change" not in kinds


# ---------------------------------------------------------------- added in the final version
def _cfg():
    from src.utils import load_config
    return load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))


def test_shift_detected_but_walking_person_is_not():
    import numpy as np, cv2
    from src.tamper import TamperDetector
    rng = np.random.default_rng(0)
    c = _cfg()["tamper"]
    base = cv2.resize(cv2.GaussianBlur(rng.normal(120, 30, (180, 320)).astype(np.float32), (0, 0), 2), (640, 360))
    base = np.clip(base, 0, 255).astype(np.uint8)
    td = TamperDetector(c)
    for i in range(60):                                   # high-contrast "person" walking, camera still
        f = base.copy(); x = 100 + i * 6
        cv2.rectangle(f, (x, 120), (x + 40, 300), 20, -1)
        st, _ = td.check(f + rng.integers(0, 3, f.shape, dtype=np.uint8), i * 0.5)
        assert st != "shifted", i
    td = TamperDetector(c)
    for i in range(10):
        td.check(base + rng.integers(0, 3, base.shape, dtype=np.uint8), i * 0.5)
    moved = cv2.warpAffine(base, np.float32([[1, 0, 60], [0, 1, 15]]), (640, 360), borderMode=cv2.BORDER_REPLICATE)
    for i in range(10, 30):
        st, _ = td.check(moved + rng.integers(0, 3, base.shape, dtype=np.uint8), i * 0.5)
    assert st == "shifted"


def test_tall_person_like_blob_is_rejected():
    import numpy as np
    from src.candidates import extract_candidates
    cfg = _cfg()
    m = np.zeros((300, 300), np.uint8)
    m[50:140, 40:60] = 255            # 20 x 90  = person-like
    m[200:240, 150:190] = 255         # 40 x 40  = luggage-like
    blobs = extract_candidates(m, cfg["candidates"])
    assert len(blobs) == 1 and blobs[0].bbox[0] >= 150


def test_candidate_follows_shrinking_blob_and_waits_until_stable():
    import numpy as np
    from types import SimpleNamespace
    from src.candidates import Blob
    from src.state_machine import StateMachine
    cfg = _cfg()

    class A:
        def emit(self, *a, **k):
            return True

    class Env:
        def __init__(self):
            self.verified = 0
            self.frame = SimpleNamespace(raw=np.zeros((200, 200, 3), np.uint8))
            self.bg_img = np.zeros((200, 200, 3), np.uint8)
        def verify(self, tr): self.verified += 1; return "abandoned", "ok"
        def owner_assign(self, tr): return None
        def is_luggage(self, tr): return False
        def presence(self, tr): return True, 1.0, tr.bbox
        def occluded(self, tr): return False
        def owner_dist(self, tr): return None
        def vis_frame(self): return None

    sm, env = StateMachine(cfg, A()), Env()
    big = Blob((40, 40, 30, 100), np.full((100, 30), 255, np.uint8), 3000)      # bag + person silhouette
    small = Blob((40, 120, 30, 20), np.full((20, 30), 255, np.uint8), 600)      # only the bag is left
    for t in (0.0, 0.5, 1.0):
        sm.step(t, [big], env)
    sm.step(1.5, [small], env)
    assert len(sm.tracks) == 1, "the shrunken blob must keep the SAME track (no stale tall box)"
    assert next(iter(sm.tracks.values())).bbox == small.bbox
    s = cfg["tracking"]["static_s"]
    t = 1.5
    while t + 0.5 < 1.5 + s:                                   # stop just BEFORE the stability time is reached
        t += 0.5; sm.step(t, [small], env)
    assert env.verified == 0, "must not verify before the shape has been stable for static_s"
    while t < 1.5 + s + 1.0:
        t += 0.5; sm.step(t, [small], env)
    assert env.verified == 1

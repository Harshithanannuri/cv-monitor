"""Tests for the features added on top of the core pipeline: zones, owner re-identification,
live sources, clip recorder, settings merge."""
import os
import sys
import tempfile
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.owner import Homography, OwnerTracker, Person, appearance_hist   # noqa: E402
from src.pipeline import Pipeline                                            # noqa: E402
from src.recorder import ClipRecorder                                        # noqa: E402
from src.utils import FrameSource, LiveSource, load_config, open_source, reference_frame   # noqa: E402

DEMO = os.path.join(ROOT, "data", "clips", "demo.mp4")


def _cfg(**pipe):
    cfg = load_config()
    cfg["pipeline"].update(dict(people="none", verify="rule", **pipe))
    cfg["owner"]["alarm_s"] = 5
    cfg["alerts"]["write_files"] = False
    return cfg


def _run(cfg, seconds=25):
    pipe = Pipeline(cfg, "test")
    for t, f in FrameSource(DEMO, cfg["source"]["process_fps"], None, cfg["source"]["resize_width"]):
        if t > seconds:
            break
        pipe.process(f, t)
    return pipe.alerts.events


def test_zones_include_ignore():
    base = _run(_cfg())
    obj = next(e for e in base if e["kind"] == "static_object")
    x, y, w, h = obj["bbox"]
    assert any(e["kind"] == "alarm" for e in base)
    # watch zone that does NOT contain the object -> nothing is detected
    cfg = _cfg(); cfg["zones"] = {"include": [[[0, 0], [60, 0], [60, 40], [0, 40]]], "ignore": []}
    assert not any(e["kind"] in ("static_object", "alarm") for e in _run(cfg))
    # watch zone around the object -> still detected
    cfg = _cfg(); cfg["zones"] = {"include": [[[x - 30, y - 30], [x + w + 30, y - 30], [x + w + 30, y + h + 30], [x - 30, y + h + 30]]], "ignore": []}
    assert any(e["kind"] == "alarm" for e in _run(cfg))
    # ignore zone over the object -> ignored
    cfg = _cfg(); cfg["zones"] = {"include": [], "ignore": [[[x - 20, y - 20], [x + w + 20, y - 20], [x + w + 20, y + h + 20], [x - 20, y + h + 20]]]}
    assert not any(e["kind"] in ("static_object", "alarm") for e in _run(cfg))


def _person_patch(color, size=(60, 30)):
    img = np.zeros((120, 60, 3), np.uint8)
    img[:] = color
    return img


def test_owner_reidentification_after_id_switch():
    c = load_config()["owner"]
    h = Homography(None, c["px_per_m"])
    trk = OwnerTracker(c, h)
    red, blue = (30, 30, 200), (200, 80, 30)
    mk = lambda pid, x, col: Person(pid, (x, 100, 30, 80), appearance_hist(_person_patch(col), (0, 0, 60, 120)))
    t = 0.0
    for _ in range(20):                                    # owner id=1 stands next to the object
        trk.push(t, [mk(1, 100, red), mk(2, 400, blue)]); t += 0.1
    owner = trk.assign((110, 150, 20, 20), 1.0)
    assert owner == 1
    for _ in range(6):                                     # owner vanishes (occlusion) ...
        trk.push(t, [mk(2, 400, blue)]); t += 0.1
    for _ in range(5):                                     # ... reappears with a NEW id, same colours, nearby
        trk.push(t, [mk(9, 120, red), mk(2, 400, blue)]); t += 0.1
    assert trk.alias.get(9) == 1
    d = trk.distance(1, (110, 150, 20, 20), t)
    assert d is not None and d < 2.0


def test_reid_rejects_different_appearance():
    c = load_config()["owner"]
    trk = OwnerTracker(c, Homography(None, c["px_per_m"]))
    red, green = (30, 30, 200), (40, 200, 40)
    mk = lambda pid, x, col: Person(pid, (x, 100, 30, 80), appearance_hist(_person_patch(col), (0, 0, 60, 120)))
    t = 0.0
    for _ in range(20):
        trk.push(t, [mk(1, 100, red)]); t += 0.1
    trk.assign((110, 150, 20, 20), 1.0)
    for _ in range(6):
        trk.push(t, []); t += 0.1
    for _ in range(5):
        trk.push(t, [mk(9, 120, green)]); t += 0.1
    assert 9 not in trk.alias


def test_live_source_reads_frames_and_resizes():
    # a file is decoded faster than real time and ends quickly, so only count the valid frames
    src = LiveSource(DEMO, process_fps=5, resize_width=320)
    n, t0 = 0, time.time()
    for t, f in src:
        if f is not None:
            assert f.shape[1] == 320
            n += 1
        if n >= 3 or time.time() - t0 > 10:
            break
    assert n >= 3
    assert isinstance(open_source("0"), LiveSource) and isinstance(open_source("rtsp://x/y"), LiveSource)
    assert isinstance(open_source(DEMO), FrameSource)


def test_reference_frame():
    f = reference_frame(DEMO, 640)
    assert f is not None and f.shape[1] == 640


def test_clip_recorder_pre_and_post_roll():
    with tempfile.TemporaryDirectory() as d:
        rec = ClipRecorder(d, 10, {"pre_s": 2, "post_s": 1, "kinds": ["alarm"]})
        for i in range(60):
            t = i / 10
            fr = np.full((48, 64, 3), i * 4 % 255, np.uint8)
            rec.push(fr, t, [{"kind": "alarm"}] if i == 30 else [])
        rec.close()
        assert len(rec.written) == 1
        cap = cv2.VideoCapture(os.path.join(d, "clips", rec.written[0]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        assert 28 <= n <= 36, n                      # ~2 s pre + alarm frame + 1 s post at 10 fps


def test_dedup_alarm_cooldown():
    from src.alerts import AlertManager
    c = load_config()["alerts"]; c["write_files"] = False
    am = AlertManager(c, "t")
    assert am.emit("alarm", 10, (10, 10, 20, 20), 1) is True
    assert am.emit("alarm", 15, (12, 12, 20, 20), 2) is False        # same place, within cooldown
    assert am.emit("alarm", 100, (12, 12, 20, 20), 3) is True        # cooldown over

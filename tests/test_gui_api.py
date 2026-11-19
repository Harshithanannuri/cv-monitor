"""Integration test of the GUI backend: starts the real Flask app on a free port, runs the pipeline
on a 25 s excerpt of the demo clip and checks state, events, snapshots, clips, settings, zones and
history through the HTTP API.   Run:  python tests/test_gui_api.py   (or pytest)"""
import json
import os
import shutil
import socket
import sys
import threading
import time

import cv2
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from gui.server import create_app   # noqa: E402

DEMO = os.path.join(ROOT, "data", "clips", "demo.mp4")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def make_excerpt(seconds=25):
    out = os.path.join(ROOT, "data", "uploads", "_test_excerpt.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cap = cv2.VideoCapture(DEMO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(int(seconds * fps)):
        ok, f = cap.read()
        if not ok:
            break
        wr.write(f)
    wr.release(); cap.release()
    return "data/uploads/_test_excerpt.mp4"


def main():
    assert os.path.exists(DEMO), "run tools/make_demo_clip.py first"
    settings = os.path.join(ROOT, "data", "user_settings.json")
    backup = open(settings).read() if os.path.exists(settings) else None
    app = create_app()
    port = free_port()
    threading.Thread(target=lambda: app.run(port=port, threaded=True), daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            requests.get(base + "/api/state", timeout=0.5); break
        except Exception:
            time.sleep(0.1)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, extra)

    try:
        r = requests.get(base + "/")
        check("index page served", r.status_code == 200 and "CV" in r.text)
        for f in ("app.js", "style.css"):
            check(f"static {f}", requests.get(f"{base}/static/{f}").status_code == 200)

        src = make_excerpt(25)
        srcs = requests.get(base + "/api/sources").json()
        check("sources lists the excerpt", any(s["value"] == src for s in srcs))

        r = requests.get(base + "/api/ref_frame.jpg", params={"source": src})
        check("reference frame jpeg", r.status_code == 200 and r.content[:2] == b"\xff\xd8")

        zones = {"include": [[[10, 10], [600, 10], [600, 340], [10, 340]]], "ignore": [[[0, 0], [30, 0], [30, 30], [0, 30]]]}
        requests.post(base + "/api/settings", json={"zones": zones, "px_per_m": 55.0})
        s = requests.get(base + "/api/settings").json()
        check("settings round trip (zones, px_per_m)", s["zones"] == zones and abs(s["px_per_m"] - 55.0) < 1e-6 and s["saved"])
        check("settings schema", any(i["key"] == "owner.alarm_s" for g in s["schema"] for i in g["items"]))

        body = {"source": src, "speed": 0, "overrides": {"pipeline": {"people": "none", "verify": "rule"}, "owner": {"alarm_s": 5}}}
        r = requests.post(base + "/api/start", json=body)
        check("start accepted", r.status_code == 200, r.text)
        r2 = requests.post(base + "/api/start", json=body)
        check("second start rejected while running", r2.status_code == 409)

        live_ok = False
        st = {}
        for _ in range(300):
            st = requests.get(base + "/api/state").json()
            if st["state"] == "running" and not live_ok:
                requests.post(base + "/api/live_config", json={"tracking": {"static_s": 4.0}})
                live_ok = app.runner.pipe is not None and app.runner.pipe.sm.c["static_s"] == 4.0
            if st["state"] in ("finished", "error", "stopped"):
                break
            time.sleep(0.2)
        check("run finished without error", st["state"] == "finished", f'{st["state"]} {st.get("error")}')
        check("live config applied in place", live_ok)
        check("state has fps / series", st["fps"] > 0 and len(st["series"]) > 10, f'fps={st["fps"]}')
        check("components reported", st["components"].get("people") == "none")

        ev = requests.get(base + "/api/events?after=0").json()
        kinds = [e["kind"] for e in ev]
        check("alarm raised", "alarm" in kinds, str(kinds))
        alarm = next((e for e in ev if e["kind"] == "alarm"), None)
        if alarm and alarm["snapshot"]:
            r = requests.get(base + alarm["snapshot"])
            check("alarm snapshot downloadable", r.status_code == 200 and r.content[:2] == b"\xff\xd8")
        else:
            check("alarm snapshot downloadable", False)
        check("incremental events", len(requests.get(base + f"/api/events?after={len(ev) - 1}").json()) == 1)
        check("latest frame endpoint", requests.get(base + "/api/frame.jpg").status_code == 200)

        runs = requests.get(base + "/api/runs").json()
        check("history lists the run", runs and runs[0]["alarms"] >= 1, str(runs[:1]))
        det = requests.get(base + "/api/runs/" + runs[0]["name"]).json()
        check("history detail has events", len(det["events"]) == len(ev))
        check("evidence clip written", len(det["clips"]) >= 1, str(det["clips"]))
        if det["clips"]:
            r = requests.get(base + det["clips"][0]["url"])
            check("clip downloadable", r.status_code == 200 and len(r.content) > 1000)
        check("media path traversal blocked", requests.get(base + "/media/x/..%2f..%2fconfig.yaml/y").status_code in (400, 404))

        r = requests.get(base + "/api/results")
        check("results endpoint", r.status_code == 200 and "figures" in r.json())
        r = requests.post(base + "/api/job", json={"name": "nope"})
        check("unknown job rejected", r.status_code == 400)
        r = requests.post(base + "/api/start", json={"source": "data/clips/does_not_exist.mp4"})
        check("missing source rejected", r.status_code == 400)
    finally:
        if backup is None:
            if os.path.exists(settings):
                os.remove(settings)
        else:
            open(settings, "w").write(backup)
        try:
            os.remove(os.path.join(ROOT, "data", "uploads", "_test_excerpt.mp4"))
        except OSError:
            pass
    print("\nGUI API TEST:", "ALL PASSED" if ok else "SOME FAILED")
    return 0 if ok else 1


def test_gui_api():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())

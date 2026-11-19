"""CLI: python run.py --video data/clips/demo.avi [--show] [--save out.mp4] [--set pipeline.people=hog ...]"""
import argparse, logging
import cv2
from src.pipeline import Pipeline
from src.utils import FrameSource, load_config, deep_update


def parse_overrides(items):
    ov = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            import yaml; v = yaml.safe_load(v)
        except Exception:
            pass
        d = ov
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = v
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--video")
    ap.add_argument("--show", action="store_true"); ap.add_argument("--save")
    ap.add_argument("--set", nargs="*", help="override config, e.g. pipeline.people=none owner.alarm_s=10")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config, parse_overrides(a.set))
    if a.video:
        cfg["source"]["path"] = a.video
    s = cfg["source"]
    src = FrameSource(s["path"], s["process_fps"], s["fps_override"], s["resize_width"])
    import os
    pipe = Pipeline(cfg, os.path.splitext(os.path.basename(s["path"].rstrip("/")))[0])
    writer = None
    for t, frame in src:
        vis = pipe.process(frame, t)
        if vis is None:
            continue
        if a.save and writer is None:
            writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), s["process_fps"], (vis.shape[1], vis.shape[0]))
        if writer: writer.write(vis)
        if a.show:
            cv2.imshow("cv-monitor", vis)
            if cv2.waitKey(1) & 0xFF == 27: break
    if writer: writer.release()
    n = max(pipe.n, 1)
    print(f"\n{n} frames, {1.0 / max(pipe.timing['total'] / n, 1e-9):.1f} FPS (pipeline only), events: {len(pipe.alerts.events)}")


if __name__ == "__main__":
    main()

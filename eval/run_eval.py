"""Event-level evaluation.  labels.csv columns: video,type,t_start,t_end
   type = abandoned | removed | tamper | none   (none = negative clip, t_start/t_end empty)
   An 'abandoned' ground-truth event is matched by an ALARM in [t_start-early, t_end+late];
   'removed' by a 'removed' event; 'tamper' by any tamper:* event.  Everything else is FP/FN.
   Reports precision, recall, F1, mean alarm delay (s from t_start), false alarms/hour, FPS."""
import argparse, json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from src.pipeline import Pipeline
from src.utils import FrameSource, load_config

PRED = {"abandoned": lambda k: k == "alarm", "removed": lambda k: k == "removed", "tamper": lambda k: k.startswith("tamper")}


def run_video(cfg, path, max_seconds=None):
    s = cfg["source"]
    src = FrameSource(path, s["process_fps"], s["fps_override"], s["resize_width"])
    pipe = Pipeline(cfg, os.path.splitext(os.path.basename(path.rstrip("/")))[0])
    t0, n = time.perf_counter(), 0
    for t, f in src:
        if max_seconds and t > max_seconds:
            break
        pipe.process(f, t); n += 1
    return pipe.alerts.events, n, time.perf_counter() - t0, min(src.duration, max_seconds or 1e9)


def evaluate(cfg, labels_csv, data_dir, early=2.0, late=5.0, max_seconds=None, verbose=False):
    lab = pd.read_csv(labels_csv)
    tp, fp, fn, delays, per = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(list), defaultdict(int)
    tot_s, tot_frames, tot_time = 0.0, 0, 0.0
    for vid, g in lab.groupby("video", sort=False):
        events, n, dt, dur = run_video(cfg, os.path.join(data_dir, vid), max_seconds)
        tot_s += dur; tot_frames += n; tot_time += dt
        gts = g[g["type"] != "none"]
        used = set()
        for typ in PRED:
            preds = [(i, e) for i, e in enumerate(events) if PRED[typ](e["kind"])]
            for _, r in gts[gts["type"] == typ].iterrows():
                hit = next(((i, e) for i, e in preds if i not in used and r.t_start - early <= e["t"] <= r.t_end + late), None)
                if hit:
                    used.add(hit[0]); tp[typ] += 1; delays[typ].append(hit[1]["t"] - r.t_start)
                else:
                    fn[typ] += 1
            for i, e in preds:
                if i not in used:
                    fp[typ] += 1
                    if verbose: print(f"  FP {typ} in {vid} at t={e['t']}")
    rows = []
    for typ in list(PRED) + ["ALL"]:
        T = sum(tp.values()) if typ == "ALL" else tp[typ]; F = sum(fp.values()) if typ == "ALL" else fp[typ]
        N = sum(fn.values()) if typ == "ALL" else fn[typ]
        p = T / (T + F) if T + F else 0.0; r = T / (T + N) if T + N else 0.0
        d = [x for v in delays.values() for x in v] if typ == "ALL" else delays[typ]
        rows.append(dict(type=typ, TP=T, FP=F, FN=N, precision=round(p, 3), recall=round(r, 3),
                         F1=round(2 * p * r / (p + r), 3) if p + r else 0.0,
                         mean_delay_s=round(sum(d) / len(d), 1) if d else None))
    df = pd.DataFrame(rows)
    df.attrs.update(fa_per_hour=round(sum(fp.values()) / max(tot_s / 3600, 1e-9), 1), fps=round(tot_frames / max(tot_time, 1e-9), 1))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--data", default="data/clips"); ap.add_argument("--max_seconds", type=float)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like pipeline.bg_mode=single")
    ap.add_argument("--out", default="eval/results.json"); a = ap.parse_args()
    from run import parse_overrides
    cfg = load_config(a.config, parse_overrides(a.set)); cfg["alerts"]["write_files"] = False
    df = evaluate(cfg, a.labels, a.data, max_seconds=a.max_seconds, verbose=True)
    print(df.to_string(index=False)); print(f"false alarms/hour: {df.attrs['fa_per_hour']}   FPS: {df.attrs['fps']}")
    json.dump(dict(table=df.to_dict("records"), **df.attrs), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

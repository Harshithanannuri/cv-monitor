"""Runs every ablation variant on the labelled clips and writes eval/ablations.csv/.md + a bar chart.
   baseline_naive = single background + no shadow + no verification + no owner logic (the 'GitHub script')."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from eval.run_eval import evaluate
from src.utils import load_config

P = lambda **k: {"pipeline": k}
VARIANTS = {
    "full_system":          {},
    "baseline_naive":       P(bg_mode="single", shadow=False, verify="none", people="none", global_check=False, tamper=False),
    "single_background":    P(bg_mode="single"),
    "no_shadow_handling":   P(shadow=False),
    "verify_none":          P(verify="none"),
    "verify_rule_only":     P(verify="rule"),
    "verify_cnn_only":      P(verify="cnn"),
    "bg_mog2":              P(backend="mog2"),
    "bg_knn":               P(backend="knn"),
    "people_hog_baseline":  P(people="hog"),
    "people_none":          P(people="none"),
    "no_global_check":      P(global_check=False),
    "no_tamper_gate":       P(tamper=False),
    "with_clahe":           {"preprocess": {"clahe": True}},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--data", default="data/clips"); ap.add_argument("--only", nargs="*"); ap.add_argument("--max_seconds", type=float)
    a = ap.parse_args()
    rows = []
    for name, ov in VARIANTS.items():
        if a.only and name not in a.only:
            continue
        cfg = load_config(a.config, ov); cfg["alerts"]["write_files"] = False
        print(f"== {name}", flush=True)
        try:
            df = evaluate(cfg, a.labels, a.data, max_seconds=a.max_seconds)
        except Exception as e:                       # e.g. a library is not installed
            print("   skipped:", e); continue
        r = df[df.type == "ALL"].iloc[0]
        rows.append(dict(variant=name, precision=float(r.precision), recall=float(r.recall), F1=float(r.F1),
                         delay_s=r.mean_delay_s, FP=int(r.FP), FA_per_hour=df.attrs["fa_per_hour"], FPS=df.attrs["fps"]))
        print("  ", rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("eval/ablations.csv", index=False)
    try:
        open("eval/ablations.md", "w").write(out.to_markdown(index=False))
    except ImportError:                              # tabulate missing
        open("eval/ablations.md", "w").write(out.to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ax = out.set_index("variant")[["precision", "recall", "F1"]].plot.bar(figsize=(11, 4.5))
        ax.set_ylim(0, 1.05); plt.xticks(rotation=40, ha="right"); plt.tight_layout()
        os.makedirs("report/figures", exist_ok=True); plt.savefig("report/figures/ablations.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)

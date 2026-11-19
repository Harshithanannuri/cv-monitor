"""Fine-tune the 6-channel MobileNetV2.  Run on Colab GPU:  python train/train_cnn.py --data data/synth"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, torch, torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from src.classifier import CLASSES, build_model, to_array


def make_aug():
    try:
        import albumentations as A
        return A.Compose([A.HorizontalFlip(p=0.5),
                          A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.7),
                          A.RandomBrightnessContrast(0.15, 0.15, p=0.3),       # same params hit both views
                          A.GaussianBlur(blur_limit=(3, 5), p=0.3)], additional_targets={"bg": "image"})
    except ImportError:
        print("albumentations missing -> flip only"); return None


class PairDS(Dataset):
    def __init__(self, root, split, aug=None):
        self.items = [(p, p.replace("_cur.png", "_bg.png"), CLASSES.index(c)) for c in CLASSES
                      for p in sorted(glob.glob(os.path.join(root, split, c, "*_cur.png")))]
        self.aug = aug

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        pc, pb, y = self.items[i]; cur, bg = cv2.imread(pc), cv2.imread(pb)
        if self.aug is not None:
            r = self.aug(image=cur, bg=bg); cur, bg = r["image"], r["bg"]
        return torch.from_numpy(to_array(cur, bg)), y


def run_epoch(model, dl, dev, opt=None, loss_fn=nn.CrossEntropyLoss()):
    model.train(opt is not None); ys, ps, tot = [], [], 0.0
    with torch.set_grad_enabled(opt is not None):
        for x, y in dl:
            x, y = x.to(dev), y.to(dev); out = model(x); loss = loss_fn(out, y)
            if opt: opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); ys += y.tolist(); ps += out.argmax(1).tolist()
    return tot / max(len(ys), 1), f1_score(ys, ps, average="macro"), ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth"); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/cnn6.pt"); ap.add_argument("--no_pretrained", action="store_true"); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = (PairDS(a.data, s, make_aug() if s == "train" else None) for s in ("train", "val", "test"))
    dl = lambda d, sh: DataLoader(d, a.bs, shuffle=sh, num_workers=2)
    model = build_model(pretrained=not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs); best = -1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ep in range(a.epochs):
        l1, f1t, _, _ = run_epoch(model, dl(tr, True), dev, opt); sched.step()
        l2, f1v, _, _ = run_epoch(model, dl(va, False), dev)
        print(f"ep{ep+1:02d} train loss {l1:.3f} F1 {f1t:.3f} | val loss {l2:.3f} F1 {f1v:.3f}")
        if f1v > best: best = f1v; torch.save(model.state_dict(), a.out)
    model.load_state_dict(torch.load(a.out, map_location=dev))
    _, f1, ys, ps = run_epoch(model, dl(te, False), dev)
    print("TEST macro-F1", round(f1, 3)); print(classification_report(ys, ps, target_names=CLASSES, digits=3))
    cm = confusion_matrix(ys, ps)
    json.dump(dict(test_macro_f1=f1, confusion=cm.tolist(), classes=CLASSES), open("models/cnn6_metrics.json", "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center")
        os.makedirs("report/figures", exist_ok=True); plt.tight_layout(); plt.savefig("report/figures/cnn_confusion.png", dpi=150)
    except Exception as e: print("plot skipped:", e)


if __name__ == "__main__":
    main()

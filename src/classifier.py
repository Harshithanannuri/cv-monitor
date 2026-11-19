"""6-channel MobileNetV2: input = current crop (3ch) + background crop (3ch) -> abandoned / removed / artifact.
'artifact' covers shadows and lighting artefacts. A 'ghost' (old object left) looks like 'removed'."""
import cv2, numpy as np

CLASSES = ["abandoned", "removed", "artifact"]
MEAN, STD = np.float32([0.485, 0.456, 0.406]), np.float32([0.229, 0.224, 0.225])


def crop_pair(cur_bgr, bg_bgr, bbox, size=96, pad=0.25):
    """Same crop logic for training and inference: padded square-ish box resized to size x size."""
    x, y, w, h = bbox
    H, W = cur_bgr.shape[:2]
    px, py = int(w * pad) + 2, int(h * pad) + 2
    x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
    f = lambda im: cv2.resize(im[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    return f(cur_bgr), f(bg_bgr)


def to_array(cur_bgr, bg_bgr):
    """two BGR uint8 crops -> (6,H,W) float32, ImageNet-normalised (RGB order)."""
    chans = []
    for im in (cur_bgr, bg_bgr):
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chans.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
    return np.concatenate(chans, 0)


def build_model(num_classes=len(CLASSES), pretrained=True):
    import torch, torch.nn as nn
    from torchvision import models
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    old = m.features[0][0]
    new = nn.Conv2d(6, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():                         # reuse ImageNet filters for both views
        new.weight[:, :3] = old.weight * 0.5
        new.weight[:, 3:] = old.weight * 0.5
    m.features[0][0] = new
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


class PairClassifier:
    def __init__(self, weights, size=96, device="cpu"):
        import os
        self.size, self.available, self.device = size, False, device
        if not os.path.exists(weights):
            return
        import torch
        self.torch = torch
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location=device))
        self.model.eval().to(device)
        self.available = True

    def predict(self, cur_bgr, bg_bgr, bbox):
        c, b = crop_pair(cur_bgr, bg_bgr, bbox, self.size)
        x = self.torch.from_numpy(to_array(c, b)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(x), 1)[0].cpu().numpy()
        i = int(p.argmax())
        return {"label": CLASSES[i], "conf": float(p[i]), "probs": p.tolist()}

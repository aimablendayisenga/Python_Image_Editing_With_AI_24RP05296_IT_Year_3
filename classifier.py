"""Photo problem recognition: hybrid CNN (+ engineered features) -> sharp / blurry / noisy / dark.

  python classifier.py train --data data/train
  python classifier.py tune  --data data/train
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from enhancer import load_images, to_tensor, DEVICE, MODEL_DIR

CLASSES = ["sharp", "blurry", "noisy", "dark"]
CROP = 96
DEFAULTS = dict(lr=1e-3, epochs=12, batch=32, feats=True, samples=1600)


# ------------------------------------------------ DATA: create labelled problem photos from clean ones
def corrupt(im, label, rng):
    """Turn a clean image into a blurry / noisy / dark one (sharp stays unchanged)."""
    if label == "blurry":
        return im.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 4)))
    a = np.asarray(im, dtype=np.float32)
    if label == "noisy":
        a = a + rng.normal(0, rng.uniform(15, 40), a.shape)
    elif label == "dark":
        a = a * rng.uniform(0.15, 0.5)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def features(im):
    """Engineered features: [sharpness (Laplacian variance), brightness, noise level, edge strength]."""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    gy, gx = np.gradient(g)
    noise = np.median(np.abs(lap)) / 0.6745
    return np.array([np.log1p(lap.var()) / 10, g.mean() / 255, min(noise, 60) / 60, np.hypot(gx, gy).mean() / 50],
                    dtype=np.float32)


def split_images(data, seed=0):
    imgs = [i for i in load_images(data) if min(i.size) >= CROP]
    order = np.random.default_rng(seed).permutation(len(imgs))
    imgs = [imgs[i] for i in order]
    k = max(1, int(len(imgs) * 0.8))
    return (imgs[:k], imgs[k:]) if len(imgs) > 1 else (imgs, imgs)


def make_set(imgs, n, seed):
    rng = np.random.default_rng(seed)
    X, Fe, Y = [], [], []
    for i in range(n):
        label = i % len(CLASSES)  # balanced classes
        im = imgs[rng.integers(len(imgs))]
        x, y = int(rng.integers(0, im.size[0] - CROP + 1)), int(rng.integers(0, im.size[1] - CROP + 1))
        c = corrupt(im.crop((x, y, x + CROP, y + CROP)), CLASSES[label], rng)
        X.append(to_tensor(c)); Fe.append(features(c)); Y.append(label)
    return torch.stack(X), torch.from_numpy(np.stack(Fe)), torch.tensor(Y)


# ------------------------------------------------ MODEL
class PhotoNet(nn.Module):
    """3 conv blocks learn visual features; engineered features are concatenated before the classifier."""
    def __init__(self, feats=True):
        super().__init__()
        self.feats = feats
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2))
        self.cnn = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64 + (4 if feats else 0), 64), nn.ReLU(True), nn.Dropout(0.3),
                                nn.Linear(64, len(CLASSES)))

    def forward(self, x, f):
        z = self.cnn(x)
        return self.fc(torch.cat([z, f], 1) if self.feats else z)


# ------------------------------------------------ EVALUATION
@torch.no_grad()
def evaluate(model, data):
    X, Fe, Y = data
    model.eval()
    t0 = time.time()
    pred = model(X.to(DEVICE), Fe.to(DEVICE)).argmax(1).cpu().numpy()
    secs = (time.time() - t0) / len(Y)
    y = Y.numpy()
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return dict(accuracy=round(float((pred == y).mean()), 4), precision=round(float(p), 4), recall=round(float(r), 4),
                f1=round(float(f), 4), confusion_matrix=confusion_matrix(y, pred).tolist(), classes=CLASSES,
                seconds_per_image=round(secs, 5))


# ------------------------------------------------ TRAINING / TUNING
def train(data, cfg, save=True):
    torch.manual_seed(0)
    tr, va = split_images(data)
    print(f"{len(tr)} train / {len(va)} validation source images | device={DEVICE}")
    X, Fe, Y = make_set(tr, cfg["samples"], 1)
    val = make_set(va, 400, 2)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Fe, Y), batch_size=cfg["batch"], shuffle=True)
    model = PhotoNet(cfg["feats"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, best_state, history = -1, None, []
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); total = 0
        for xb, fb, yb in loader:
            loss = F.cross_entropy(model(xb.to(DEVICE), fb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        m = evaluate(model, val)
        history.append({"epoch": ep, "train_loss": round(total / len(loader), 4), "val_acc": m["accuracy"], "val_f1": m["f1"]})
        print(history[-1])
        if m["f1"] > best:
            best, best_state = m["f1"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    metrics = evaluate(model, val)
    metrics["history"] = history
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(best_state, f"{MODEL_DIR}/classifier.pt")
        json.dump(cfg, open(f"{MODEL_DIR}/clf_config.json", "w"), indent=2)
        json.dump(metrics, open(f"{MODEL_DIR}/clf_metrics.json", "w"), indent=2)
        print("Saved classifier, config and metrics to", MODEL_DIR)
    print({k: metrics[k] for k in ("accuracy", "precision", "recall", "f1")})
    print("Confusion matrix (rows=true, cols=pred, order", CLASSES, "):\n", np.array(metrics["confusion_matrix"]))
    return metrics


def tune(data):
    """Compare settings (incl. with/without engineered features), keep best validation F1, retrain."""
    grid = [dict(lr=1e-3, feats=True), dict(lr=1e-3, feats=False), dict(lr=5e-4, feats=True)]
    results = []
    for g in grid:
        print("\nTrying", g)
        results.append({**g, "val_f1": train(data, {**DEFAULTS, **g, "epochs": 6}, save=False)["f1"]})
    os.makedirs(MODEL_DIR, exist_ok=True)
    json.dump(results, open(f"{MODEL_DIR}/clf_tuning.json", "w"), indent=2)
    best = max(results, key=lambda r: r["val_f1"])
    print("\nBest settings:", best)
    train(data, {**DEFAULTS, "lr": best["lr"], "feats": best["feats"]})


# ------------------------------------------------ INFERENCE
def load_classifier():
    cfg = json.load(open(f"{MODEL_DIR}/clf_config.json"))
    model = PhotoNet(cfg["feats"]).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/classifier.pt", map_location=DEVICE))
    return model.eval()


@torch.no_grad()
def predict(model, im):
    """Average the prediction over a 3x3 grid of native-resolution crops. Returns (label, probabilities)."""
    if min(im.size) < CROP:
        s = CROP / min(im.size)
        im = im.resize((int(im.size[0] * s) + 1, int(im.size[1] * s) + 1))
    w, h = im.size
    xs, ys = np.linspace(0, w - CROP, 3).astype(int), np.linspace(0, h - CROP, 3).astype(int)
    crops = [im.crop((int(x), int(y), int(x) + CROP, int(y) + CROP)) for x in xs for y in ys]
    X = torch.stack([to_tensor(c) for c in crops]).to(DEVICE)
    Fe = torch.from_numpy(np.stack([features(c) for c in crops])).to(DEVICE)
    probs = F.softmax(model(X, Fe), 1).mean(0).cpu().numpy()
    return CLASSES[int(probs.argmax())], dict(zip(CLASSES, probs.round(3).tolist()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "tune"])
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    a = ap.parse_args()
    train(a.data, {**DEFAULTS, "epochs": a.epochs}) if a.cmd == "train" else tune(a.data)

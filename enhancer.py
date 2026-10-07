"""AI Image Enhancer (Picsart-style HD / Ultra HD) - CNN super-resolution in PyTorch.

Usage:
  python enhancer.py train   --data data/train          # train + evaluate + save model
  python enhancer.py tune    --data data/train          # compare settings, then train best
  python enhancer.py enhance --input a.jpg --output b.png --times 1   # 1 = HD (2x), 2 = Ultra HD (4x)
"""
import argparse, glob, io, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

SCALE = 2
MODEL_DIR = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULTS = dict(blocks=6, channels=64, lr=1e-3, epochs=30, batch=16, patch=64, patches_per_epoch=400)


# ---------------------------------------------------------------- 1. DATA PREPROCESSING
def load_images(folder, max_side=512):
    """Load clean high-res images (for example DIV2K), including nested folders."""
    paths = sorted({p for e in ("jpg", "jpeg", "png", "JPG", "PNG")
                    for p in glob.glob(os.path.join(folder, "**", f"*.{e}"), recursive=True)})
    if paths:
        imgs = [Image.open(p).convert("RGB") for p in sorted(paths)]
    else:
        from sklearn.datasets import load_sample_images
        print(f"No images in '{folder}' -> using scikit-learn sample images")
        imgs = [Image.fromarray(a) for a in load_sample_images().images]
    for im in imgs:
        im.thumbnail((max_side, max_side), Image.LANCZOS)  # keep sharp, manageable HR targets
    return imgs


def degrade(hr, augment=False):
    """Create a paired low-quality input from a clean high-resolution target."""
    if augment:
        hr = hr.filter(ImageFilter.GaussianBlur(float(np.random.uniform(0.25, 1.5))))
    else:
        hr = hr.filter(ImageFilter.GaussianBlur(0.8))
    w, h = hr.size
    lr = hr.resize((w // SCALE, h // SCALE), Image.BICUBIC)
    if augment and np.random.rand() < 0.5:
        buffer = io.BytesIO()
        lr.save(buffer, format="JPEG", quality=int(np.random.randint(35, 91)))
        lr = Image.open(buffer).convert("RGB")
    return lr


def to_tensor(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)


def to_image(t):
    return Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))


class Patches(torch.utils.data.Dataset):
    """Random HR crops (+flip augmentation) paired with their degraded LR version."""
    def __init__(self, imgs, patch, n):
        self.imgs, self.patch, self.n = imgs, patch, n

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        im = self.imgs[np.random.randint(len(self.imgs))]
        w, h = im.size
        p = self.patch
        x, y = np.random.randint(0, w - p + 1), np.random.randint(0, h - p + 1)
        hr = im.crop((x, y, x + p, y + p))
        if np.random.rand() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
        return to_tensor(degrade(hr, augment=True)), to_tensor(hr)


# ---------------------------------------------------------------- 2. MODEL (CNN, residual + PixelShuffle)
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class SRNet(nn.Module):
    """LR image -> conv features -> residual blocks -> PixelShuffle upsample -> RGB.
    The CNN only learns the missing detail added on top of a bicubic upscale (residual learning)."""
    def __init__(self, blocks=6, channels=64):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.up = nn.Sequential(nn.Conv2d(channels, channels * SCALE ** 2, 3, padding=1), nn.PixelShuffle(SCALE))
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        f = self.head(x)
        f = f + self.body(f)
        base = F.interpolate(x, scale_factor=SCALE, mode="bicubic", align_corners=False)
        return (self.tail(self.up(f)) + base).clamp(0, 1)


# ---------------------------------------------------------------- 3. EVALUATION
@torch.no_grad()
def enhance_once(model, im):
    return to_image(model(to_tensor(im).unsqueeze(0).to(DEVICE))[0])


def enhance(model, im, times=1):
    """times=1 -> HD (2x), times=2 -> Ultra HD (4x)."""
    for _ in range(times):
        im = enhance_once(model, im)
    return im


def evaluate(model, imgs):
    """PSNR, SSIM and RMSE of the CNN vs. plain bicubic on held-out images."""
    model.eval()
    res = {"model": [], "bicubic": []}
    t0 = time.time()
    for hr in imgs:
        w, h = (hr.size[0] // SCALE) * SCALE, (hr.size[1] // SCALE) * SCALE
        hr = hr.crop((0, 0, w, h))
        lr = degrade(hr)
        ref = np.asarray(hr, dtype=np.float32) / 255
        for name, out in (("model", enhance_once(model, lr)), ("bicubic", lr.resize((w, h), Image.BICUBIC))):
            o = np.asarray(out, dtype=np.float32) / 255
            res[name].append((psnr(ref, o, data_range=1.0), ssim(ref, o, channel_axis=2, data_range=1.0),
                              float(np.sqrt(np.mean((ref - o) ** 2)))))
    out = {k: dict(zip(("psnr", "ssim", "rmse"), np.mean(v, axis=0).round(4).tolist())) for k, v in res.items()}
    out["seconds_per_image"] = round((time.time() - t0) / max(1, len(imgs)), 3)
    out["n_val"] = len(imgs)
    return out


# ---------------------------------------------------------------- 4. TRAINING / TUNING
def train(data, cfg, save=True):
    torch.manual_seed(0); np.random.seed(0)
    imgs = [i for i in load_images(data) if min(i.size) >= cfg["patch"]]
    np.random.shuffle(imgs)
    k = max(1, int(len(imgs) * 0.8))
    tr, va = (imgs[:k], imgs[k:]) if len(imgs) > 1 else (imgs, imgs)
    print(f"{len(tr)} train / {len(va)} validation images | device={DEVICE}")
    if len(imgs) < 10:
        print("WARNING: fewer than 10 clean images; validation metrics are not reliable. "
              "Add a larger real-photo dataset before using this model.")
    loader = torch.utils.data.DataLoader(Patches(tr, cfg["patch"], cfg["patches_per_epoch"]), batch_size=cfg["batch"])
    model = SRNet(cfg["blocks"], cfg["channels"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg["epochs"] // 2), gamma=0.5)
    best, history, best_state = -1, [], None
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); total = 0
        for lr_b, hr_b in loader:
            lr_b, hr_b = lr_b.to(DEVICE), hr_b.to(DEVICE)
            loss = F.l1_loss(model(lr_b), hr_b)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        sched.step()
        m = evaluate(model, va)
        history.append({"epoch": ep, "train_l1": round(total / len(loader), 5), "val_psnr": m["model"]["psnr"]})
        print(history[-1])
        if m["model"]["psnr"] > best:
            best, best_state = m["model"]["psnr"], {k_: v.clone() for k_, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    metrics = evaluate(model, va)
    metrics["history"] = history
    if save:  # saved so the app is reproducible
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(best_state, f"{MODEL_DIR}/srnet_x2.pt")
        json.dump(cfg, open(f"{MODEL_DIR}/config.json", "w"), indent=2)
        json.dump(metrics, open(f"{MODEL_DIR}/metrics.json", "w"), indent=2)
        print("Saved model, config and metrics to", MODEL_DIR)
    print("RESULT  CNN:", metrics["model"], "| Bicubic:", metrics["bicubic"])
    return metrics


def tune(data):
    """Adjust parameters based on evaluation: try several settings, keep the best validation PSNR."""
    grid = [dict(blocks=4, lr=1e-3), dict(blocks=6, lr=1e-3), dict(blocks=6, lr=5e-4), dict(blocks=8, lr=5e-4)]
    results = []
    for g in grid:
        cfg = {**DEFAULTS, **g, "epochs": 10}
        print("\nTrying", g)
        results.append({**g, "val_psnr": train(data, cfg, save=False)["model"]["psnr"]})
    os.makedirs(MODEL_DIR, exist_ok=True)
    json.dump(results, open(f"{MODEL_DIR}/tuning.json", "w"), indent=2)
    best = max(results, key=lambda r: r["val_psnr"])
    print("\nBest settings:", best)
    train(data, {**DEFAULTS, "blocks": best["blocks"], "lr": best["lr"]})


def load_model():
    cfg = json.load(open(f"{MODEL_DIR}/config.json"))
    model = SRNet(cfg["blocks"], cfg["channels"]).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/srnet_x2.pt", map_location=DEVICE))
    return model.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "tune", "enhance"])
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--input"); ap.add_argument("--output", default="enhanced.png")
    ap.add_argument("--times", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.data, {**DEFAULTS, "epochs": a.epochs})
    elif a.cmd == "tune":
        tune(a.data)
    else:
        enhance(load_model(), Image.open(a.input).convert("RGB"), a.times).save(a.output)
        print("Saved", a.output)

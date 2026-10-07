"""Automatic photo fixes chosen by the classifier + evaluation of the whole automation.

  python editor.py evaluate --data data/train
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import denoise_tv_chambolle
from classifier import CLASSES, corrupt, load_classifier, predict, split_images
from enhancer import MODEL_DIR


def sharpen(im):
    return im.filter(ImageFilter.UnsharpMask(radius=2, percent=300, threshold=2))


def denoise(im, weight=0.12):
    out = denoise_tv_chambolle(np.asarray(im, dtype=np.float32) / 255, weight=weight, channel_axis=2)
    return Image.fromarray((out * 255).round().astype(np.uint8))


def brighten(im):
    """Gamma correction that moves the average brightness to about 0.45."""
    a = np.asarray(im, dtype=np.float32) / 255
    gamma = float(np.clip(np.log(0.45) / np.log(np.clip(a.mean(), 0.02, 0.98)), 0.3, 1.0))
    return Image.fromarray(((a ** gamma) * 255).round().astype(np.uint8))


FIXES = {"blurry": sharpen, "noisy": denoise, "dark": brighten, "sharp": lambda im: im}
FIX_NAMES = {"blurry": "Sharpen (deblur)", "noisy": "Denoise", "dark": "Brighten", "sharp": "No fix needed"}

LOOK_PRESETS = (
    "Natural polish",
    "Vivid color",
    "Warm tone",
    "Cool tone",
    "Black & white",
    "Portrait polish",
    "Cinematic",
    "Soft matte",
    "Golden hour",
    "Editorial clean",
    "Teal & orange",
    "Film fade",
    "Moody",
    "Pastel",
    "High-key studio",
)


def simulate_before(im):
    """Create a clearly labeled low-detail comparison, not a recovered original."""
    small_size = (max(1, int(im.width * 0.45)), max(1, int(im.height * 0.45)))
    small = im.resize(small_size, Image.Resampling.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(1.2))
    return small.resize(im.size, Image.Resampling.BICUBIC)


def apply_look(im, look):
    if look == "Original":
        return im
    if look == "Natural polish":
        out = ImageOps.autocontrast(im, cutoff=1)
        out = ImageEnhance.Color(out).enhance(1.08)
        out = ImageEnhance.Contrast(out).enhance(1.04)
        return ImageEnhance.Sharpness(out).enhance(1.08)
    if look == "Vivid color":
        out = ImageEnhance.Color(im).enhance(1.25)
        out = ImageEnhance.Contrast(out).enhance(1.08)
        return ImageEnhance.Sharpness(out).enhance(1.1)
    if look in ("Warm tone", "Cool tone"):
        a = np.asarray(im, dtype=np.float32)
        gains = (1.05, 1.01, 0.94) if look == "Warm tone" else (0.94, 1.01, 1.05)
        return Image.fromarray(np.clip(a * np.asarray(gains), 0, 255).astype(np.uint8))
    if look == "Black & white":
        return ImageOps.grayscale(im).convert("RGB")
    if look == "Portrait polish":
        out = ImageEnhance.Brightness(im).enhance(1.04)
        out = ImageEnhance.Color(out).enhance(1.04)
        out = ImageEnhance.Contrast(out).enhance(0.98)
        return ImageEnhance.Sharpness(out).enhance(1.05)
    if look == "Cinematic":
        a = np.asarray(im, dtype=np.float32) / 255
        luminance = a.mean(axis=2)
        shadows = np.clip((0.5 - luminance) * 2, 0, 1)
        highlights = np.clip((luminance - 0.5) * 2, 0, 1)
        a[..., 0] += 0.035 * highlights - 0.018 * shadows
        a[..., 2] += 0.025 * shadows - 0.012 * highlights
        out = Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))
        out = ImageEnhance.Contrast(out).enhance(1.12)
        return ImageEnhance.Color(out).enhance(0.92)
    if look == "Soft matte":
        a = np.asarray(im, dtype=np.float32) / 255
        a = np.clip(0.08 + 0.84 * a, 0, 1)
        out = Image.fromarray((a * 255).round().astype(np.uint8))
        out = ImageEnhance.Contrast(out).enhance(0.92)
        return ImageEnhance.Color(out).enhance(0.94)
    if look == "Golden hour":
        a = np.asarray(im, dtype=np.float32)
        a *= np.asarray((1.06, 1.02, 0.94), dtype=np.float32)
        out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        out = ImageEnhance.Brightness(out).enhance(1.03)
        return ImageEnhance.Color(out).enhance(1.1)
    if look == "Editorial clean":
        out = ImageOps.autocontrast(im, cutoff=0.5)
        out = ImageEnhance.Brightness(out).enhance(1.02)
        out = ImageEnhance.Color(out).enhance(1.04)
        out = ImageEnhance.Contrast(out).enhance(1.04)
        return ImageEnhance.Sharpness(out).enhance(1.08)
    if look == "Teal & orange":
        a = np.asarray(im, dtype=np.float32) / 255
        luminance = a @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
        shadows = np.clip((0.58 - luminance) / 0.58, 0, 1)[..., None]
        highlights = np.clip((luminance - 0.42) / 0.58, 0, 1)[..., None]
        a += shadows * np.asarray((-0.008, 0.018, 0.026), dtype=np.float32)
        a += highlights * np.asarray((0.032, 0.014, -0.01), dtype=np.float32)
        out = Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))
        out = ImageEnhance.Contrast(out).enhance(1.08)
        return ImageEnhance.Color(out).enhance(1.06)
    if look == "Film fade":
        a = np.asarray(im, dtype=np.float32) / 255
        a = np.clip(0.035 + 0.91 * a, 0, 1)
        a *= np.asarray((1.025, 1.0, 0.98), dtype=np.float32)
        out = Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))
        out = ImageEnhance.Contrast(out).enhance(0.97)
        return ImageEnhance.Color(out).enhance(0.92)
    if look == "Moody":
        a = np.asarray(im, dtype=np.float32) / 255
        luminance = a @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
        shadows = np.clip((0.55 - luminance) / 0.55, 0, 1)[..., None]
        a[..., 2] += 0.025 * shadows[..., 0]
        a[..., 1] += 0.008 * shadows[..., 0]
        out = Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))
        out = ImageEnhance.Brightness(out).enhance(0.98)
        out = ImageEnhance.Contrast(out).enhance(1.1)
        return ImageEnhance.Color(out).enhance(0.92)
    if look == "Pastel":
        a = np.asarray(im, dtype=np.float32) / 255
        a = np.clip(0.025 + 0.92 * a, 0, 1)
        a *= np.asarray((1.015, 1.005, 1.0), dtype=np.float32)
        out = Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))
        out = ImageEnhance.Contrast(out).enhance(0.94)
        return ImageEnhance.Color(out).enhance(1.06)
    if look == "High-key studio":
        out = ImageOps.autocontrast(im, cutoff=0.5)
        out = ImageEnhance.Brightness(out).enhance(1.06)
        out = ImageEnhance.Contrast(out).enhance(0.95)
        return ImageEnhance.Color(out).enhance(0.98)
    raise ValueError(f"Unknown style preset: {look}")


def auto_fix(im, label):
    return FIXES[label](im)


def evaluate_automation(clf, imgs, n_per_class=30, size=160):
    """Task-success test: damage clean photos, let the AI detect + fix, then check the result."""
    imgs = [i for i in imgs if min(i.size) >= size]
    rng = np.random.default_rng(5)
    out = {}
    for label in ("blurry", "noisy", "dark"):
        detected = success = 0
        gains = []
        for _ in range(n_per_class):
            im = imgs[rng.integers(len(imgs))]
            x, y = int(rng.integers(0, im.size[0] - size + 1)), int(rng.integers(0, im.size[1] - size + 1))
            clean = im.crop((x, y, x + size, y + size))
            bad = corrupt(clean, label, rng)
            pred, _ = predict(clf, bad)
            fixed = auto_fix(bad, pred)
            after, _ = predict(clf, fixed)
            detected += pred == label
            success += after == "sharp"
            ref = np.asarray(clean)
            gains.append(psnr(ref, np.asarray(fixed), data_range=255) - psnr(ref, np.asarray(bad), data_range=255))
        out[label] = dict(detection_rate=round(detected / n_per_class, 3), task_success_rate=round(success / n_per_class, 3),
                          mean_psnr_gain_db=round(float(np.mean(gains)), 2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evaluate"])
    ap.add_argument("--data", default="data/train")
    a = ap.parse_args()
    _, val = split_images(a.data)
    res = evaluate_automation(load_classifier(), val)
    json.dump(res, open(f"{MODEL_DIR}/auto_metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("task_success_rate = share of fixed photos the AI now recognises as 'sharp'")

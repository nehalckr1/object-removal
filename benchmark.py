"""
Benchmark the four methods against ground truth.

We take a photo, *pretend* a disc of known pixels is an unwanted object, remove it, and
compare the reconstruction with what was really there. This gives a number for
"as naturally as possible":

  PSNR (hole)  : peak signal-to-noise ratio over the hole pixels only (higher = better)
  SSIM (bbox)  : structural similarity over the hole's bounding box     (closer to 1 = better)

Run:  python benchmark.py            (writes results.md and benchmark_<image>.png)
"""
from __future__ import annotations

import importlib.util
import sys
import time

import cv2
import numpy as np
from skimage import data
from skimage.metrics import structural_similarity

from inpaint import METHODS, inpaint

# LaMa is skipped automatically when PyTorch is not installed
ACTIVE = [m for m in METHODS if m != "lama" or importlib.util.find_spec("torch")]

RADII = {"small": 12, "medium": 25, "large": 45}
SIZE = 320


def synthetic_texture(n=SIZE, seed=0):
    """Repetitive brick-like texture plus a smooth gradient: easy for patch methods."""
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n, 3), np.float32)
    bh, bw = 20, 40
    for y in range(0, n, bh):
        off = (y // bh % 2) * (bw // 2)
        for x in range(-bw, n, bw):
            col = np.array([150, 70, 55]) + rng.normal(0, 10, 3)
            img[y:y + bh - 2, max(0, x + off):max(0, x + off + bw - 2)] = col
    img[img.sum(2) == 0] = (190, 190, 180)
    img += rng.normal(0, 4, img.shape)
    grad = np.linspace(-25, 25, n)[None, :, None]
    return np.clip(img + grad, 0, 255).astype(np.uint8)


def load_images():
    imgs = {"texture": synthetic_texture()}
    for name in ("astronaut", "coffee", "chelsea"):
        try:
            im = getattr(data, name)()
            imgs[name] = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        except Exception as e:  # sample image not available offline
            print(f"skip {name}: {type(e).__name__}", file=sys.stderr)
    return imgs


def centre_for(name):
    # a location with some structure around it, kept away from the borders
    return {"texture": (160, 160), "astronaut": (200, 250), "coffee": (170, 170),
            "chelsea": (150, 170)}.get(name, (SIZE // 2, SIZE // 2))


def metrics(gt, out, mask):
    err = (gt.astype(np.float64) - out.astype(np.float64))[mask]
    mse = float((err ** 2).mean())
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    ys, xs = np.nonzero(mask)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    ssim = structural_similarity(gt[sl], out[sl], channel_axis=2, win_size=7, data_range=255)
    return psnr, float(ssim)


def main():
    rows = []
    for name, img in load_images().items():
        cy, cx = centre_for(name)
        tiles = []
        for size, rad in RADII.items():
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.circle(mask, (cx, cy), rad, 1, -1)
            mask = mask.astype(bool)

            shown = img.copy()
            shown[mask] = (255, 0, 0)
            row = [shown]
            for m in ACTIVE:
                t = time.perf_counter()
                out, info = inpaint(img, mask, m)
                dt = time.perf_counter() - t
                psnr, ssim = metrics(img, out, mask)
                rows.append((name, size, rad, METHODS[m], psnr, ssim, dt))
                row.append(out)
            row.append(img)
            tiles.append(np.hstack(row))
            print(f"{name:10s} {size:6s} done", flush=True)
        grid = np.vstack(tiles)
        cv2.imwrite(f"benchmark_{name}.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    with open("results.md", "w") as f:
        f.write("| image | hole | radius | method | PSNR (dB) | SSIM | time (s) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for n, s, r, m, p, ss, t in rows:
            f.write(f"| {n} | {s} | {r} | {m} | {p:.2f} | {ss:.3f} | {t:.2f} |\n")

        f.write("\nMean over all images:\n\n| hole | method | PSNR (dB) | SSIM |\n|---|---|---|---|\n")
        for s in RADII:
            for m in (METHODS[k] for k in ACTIVE):
                sel = [(p, ss) for n, s2, r, m2, p, ss, t in rows if s2 == s and m2 == m]
                f.write(f"| {s} | {m} | {np.mean([a for a, _ in sel]):.2f} | "
                        f"{np.mean([b for _, b in sel]):.3f} |\n")
    print(open("results.md").read())


if __name__ == "__main__":
    main()

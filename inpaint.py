"""
Object removal by image inpainting: four methods, all built on matrix ideas.

Conventions
-----------
image : uint8 array, shape (H, W, 3), RGB
mask  : bool array, shape (H, W), True = pixel to remove / reconstruct

Methods
-------
diffusion  Solve  A x = b  where A is the sparse graph Laplacian restricted to the hole.
poisson    Same matrix A, but the right-hand side b carries the *gradients* of a patch
           copied from elsewhere in the image (gradient-domain cloning).
lowrank    Iterative truncated-SVD (hard-impute) matrix completion on a window.
criminisi  Exemplar-based patch filling (priority = confidence x data term).
lama       Pretrained neural network (LaMa); the only method that has learned from other photos.
"""
from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LOCALITY = 2e-3  # Criminisi tie-break weight per pixel of distance

METHODS = {
    "diffusion": "Diffusion (Laplace solve)",
    "poisson": "Poisson clone (gradient-domain)",
    "lowrank": "Low-rank SVD completion",
    "criminisi": "Patch-based (Criminisi)",
    "lama": "LaMa (pretrained neural network)",
}


# --------------------------------------------------------------------------- helpers
def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) / 255.0


def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def grow_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate the mask by `px` pixels (brush strokes usually stop just short of the object edge)."""
    if px <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def _window(mask: np.ndarray, margin: int):
    H, W = mask.shape
    y0, y1, x0, x1 = _bbox(mask)
    return max(0, y0 - margin), min(H, y1 + margin), max(0, x0 - margin), min(W, x1 + margin)


# --------------------------------------------------------------------------- 1. diffusion
def _build_system(img: np.ndarray, mask: np.ndarray, guide: Optional[np.ndarray] = None):
    """
    Build  A u = b  for the unknown (hole) pixels.

    For every hole pixel p with valid 4-neighbours N(p):
        |N(p)| u_p - sum_{q in N(p) & hole} u_q  =  sum_{q in N(p) & known} f_q  +  sum_{q in N(p)} (g_p - g_q)

    The last term is the discrete Laplacian of the guidance field g (zero if there is none).
    A is symmetric positive definite and sparse (<= 5 non-zeros per row).
    """
    H, W, C = img.shape
    ys, xs = np.nonzero(mask)  # row-major, matches the numbering below
    n = len(ys)
    idx = -np.ones((H, W), dtype=np.int64)
    idx[mask] = np.arange(n)

    diag = np.zeros(n)
    b = np.zeros((n, C))
    rows, cols = [], []

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        v = np.nonzero(valid)[0]
        diag[v] += 1
        nyv, nxv = ny[v], nx[v]
        in_hole = mask[nyv, nxv]

        rows.append(v[in_hole])
        cols.append(idx[nyv[in_hole], nxv[in_hole]])

        k = v[~in_hole]
        b[k] += img[nyv[~in_hole], nxv[~in_hole]]

        if guide is not None:
            b[v] += guide[ys[v], xs[v]] - guide[nyv, nxv]

    r = np.concatenate(rows)
    c = np.concatenate(cols)
    A = sp.coo_matrix((-np.ones(len(r)), (r, c)), shape=(n, n)) + sp.diags(diag)
    return A.tocsc(), b


def _solve(A: sp.csc_matrix, b: np.ndarray):
    n = A.shape[0]
    if n <= 60000:
        lu = spla.splu(A)
        return lu.solve(b), "sparse LU"
    x = np.empty_like(b)
    for ch in range(b.shape[1]):
        x[:, ch], _ = spla.cg(A, b[:, ch], rtol=1e-7, maxiter=10000)
    return x, "conjugate gradient"


def laplace_fill(img: np.ndarray, mask: np.ndarray, guide: Optional[np.ndarray] = None):
    if mask.all():
        raise ValueError("The mask covers the whole image, nothing to reconstruct from.")
    A, b = _build_system(img, mask, guide)
    x, solver = _solve(A, b)
    out = img.copy()
    out[mask] = x
    return out, {"unknowns": int(A.shape[0]), "nonzeros": int(A.nnz), "solver": solver}


# --------------------------------------------------------------------------- 2. Poisson clone
def _find_source_offset(img: np.ndarray, mask: np.ndarray, ring: int):
    """
    Find where in the image a copy of 'hole + surrounding ring' fits best.

    We slide the template (dilated bounding box of the hole) over the image and compare
    only the ring pixels (known pixels around the hole) with SSD. Candidates whose source
    pixels would themselves touch the hole are rejected.
    Both scores are template-matching = correlation = a big matrix-vector style sweep.
    """
    H, W = mask.shape
    img32 = img.astype(np.float32)
    m32 = mask.astype(np.float32)

    dil = grow_mask(mask, ring)
    ring_m = dil & ~mask
    y0, y1, x0, x1 = _window(mask, ring + 1)
    T = img32[y0:y1, x0:x1]
    ring_c = ring_m[y0:y1, x0:x1]
    use_c = dil[y0:y1, x0:x1]
    h, w = T.shape[:2]
    if h >= H and w >= W:
        return None
    if ring_c.sum() == 0:
        return None

    w3 = np.ascontiguousarray(np.repeat(ring_c[..., None].astype(np.float32), 3, axis=2))
    ssd = cv2.matchTemplate(img32, np.ascontiguousarray(T), cv2.TM_SQDIFF, mask=w3)
    overlap = cv2.matchTemplate(m32, np.ascontiguousarray(use_c.astype(np.float32)), cv2.TM_CCORR)
    ssd[overlap > 0.5] = np.inf
    if not np.isfinite(ssd).any():
        return None
    sy, sx = np.unravel_index(np.argmin(ssd), ssd.shape)
    return (y0, x0, h, w, int(sy), int(sx)), float(ssd[sy, sx] / ring_c.sum())


def poisson_fill(img: np.ndarray, mask: np.ndarray, ring: int = 6, max_ring_rms: float = 0.18):
    found = _find_source_offset(img, mask, ring)
    if found is not None and np.sqrt(found[1] / 3.0) > max_ring_rms:
        out, info = laplace_fill(img, mask)
        info["note"] = ("No source region in the photo looks like the surroundings of the hole "
                        f"(best ring mismatch {np.sqrt(found[1] / 3.0) * 255:.0f}/255); "
                        "used plain diffusion instead of cloning something unrelated.")
        return out, info
    if found is None:
        out, info = laplace_fill(img, mask)
        info["note"] = "No valid source region found (hole too large); fell back to plain diffusion."
        return out, info
    (y0, x0, h, w, sy, sx), score = found
    guide = np.zeros_like(img)
    guide[y0:y0 + h, x0:x0 + w] = img[sy:sy + h, sx:sx + w]
    out, info = laplace_fill(img, mask, guide=guide)
    info["source_offset"] = (int(sy - y0), int(sx - x0))
    info["ring_ssd"] = round(score, 5)
    return out, info


# --------------------------------------------------------------------------- 3. low-rank
def lowrank_fill(img: np.ndarray, mask: np.ndarray, rank: int = 8, iters: int = 80,
                 margin: Optional[int] = None, tol: float = 1e-4):
    """
    Hard-impute matrix completion on a window around the hole.

    The three colour channels are placed side by side, giving one H x 3W matrix, so all
    channels share the same singular vectors (this removes the coloured noise you get
    from completing each channel separately).

        repeat:  X <- best rank-r approximation of X   (truncated SVD, Eckart-Young)
                 X[known] <- original pixels            (re-impose the data)

    Starts from the diffusion fill so the first SVD already sees a sensible matrix.
    """
    init, info = laplace_fill(img, mask)
    y0, y1, x0, x1 = _bbox(mask)
    m = margin if margin is not None else max(24, max(y1 - y0, x1 - x0))
    Y0, Y1, X0, X1 = _window(mask, m)
    h, w = Y1 - Y0, X1 - X0
    C = img.shape[2]

    known = np.hstack([~mask[Y0:Y1, X0:X1]] * C)
    orig = np.hstack([img[Y0:Y1, X0:X1, c] for c in range(C)])
    X = np.hstack([init[Y0:Y1, X0:X1, c] for c in range(C)])

    r = max(1, min(int(rank), h, C * w))
    used = 0
    for it in range(iters):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U[:, :r] * S[:r]) @ Vt[:r]
        Xn = np.where(known, orig, L)
        delta = np.abs(Xn - X)[~known].mean()
        X = Xn
        used = it + 1
        if delta < tol:
            break

    out = init.copy()
    for c in range(C):
        out[Y0:Y1, X0:X1, c] = X[:, c * w:(c + 1) * w]
    info.update({"rank": r, "svd_iterations": used, "window": (int(h), int(w)),
                 "energy_kept": round(float((S[:r] ** 2).sum() / (S ** 2).sum()), 4)})
    return out, info


# --------------------------------------------------------------------------- 4. Criminisi
def criminisi_fill(img: np.ndarray, mask: np.ndarray, patch: int = 9,
                   margin: Optional[int] = None, locality: float = LOCALITY,
                   progress: Optional[Callable[[float], None]] = None):
    """
    Exemplar-based inpainting (Criminisi, Perez, Toyama 2004).

    Repeat until the hole is gone:
      1. priority P(p) = C(p) * D(p) on the fill front
           C = how much of the patch around p is already trustworthy
           D = how strongly an image edge (isophote) runs into the front at p
      2. take the highest-priority front pixel p; its patch has known + unknown pixels
      3. find the fully-known patch elsewhere that best matches the known pixels (masked SSD)
      4. copy that patch's pixels into the unknown part; update confidence
    """
    ps = int(patch) | 1  # force odd
    r = ps // 2
    y0, y1, x0, x1 = _bbox(mask)
    m = margin if margin is not None else max(48, int(1.5 * max(y1 - y0, x1 - x0)))
    Y0, Y1, X0, X1 = _window(mask, m)

    I = img[Y0:Y1, X0:X1].astype(np.float32).copy()
    M = mask[Y0:Y1, X0:X1].copy()
    M0 = M.copy()
    h, w = M.shape
    if h < ps or w < ps:
        raise ValueError("Image is smaller than the patch size.")

    ones = np.ones((ps, ps), np.float32)
    invalid = cv2.matchTemplate(M0.astype(np.float32), ones, cv2.TM_CCORR) > 0.5
    if invalid.all():
        raise ValueError("No fully-known patch exists; mask too large or patch size too big.")

    gy_, gx_ = np.indices(invalid.shape)
    conf = (~M).astype(np.float32)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    total = int(M.sum())
    iters = 0

    while M.any():
        known = ~M
        front = M & cv2.dilate(known.astype(np.uint8), cross).astype(bool)
        fy, fx = np.nonzero(front)
        if len(fy) == 0:
            break

        # confidence term: mean confidence of the patch around every pixel
        Cmap = cv2.boxFilter(conf, -1, (ps, ps), normalize=True)

        # data term: isophote (perpendicular to gradient) dotted with the front normal
        gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        _, (iy, ix) = ndi.distance_transform_edt(M, return_indices=True)
        g = cv2.GaussianBlur(gray[iy, ix], (0, 0), 1.0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        mf = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 1.5)
        nx = cv2.Sobel(mf, cv2.CV_32F, 1, 0, ksize=3)
        ny = cv2.Sobel(mf, cv2.CV_32F, 0, 1, ksize=3)
        nrm = np.hypot(nx, ny) + 1e-8
        D = np.abs(-gy * nx + gx * ny) / nrm

        P = Cmap[fy, fx] * (D[fy, fx] + 1e-3)
        k = int(np.argmax(P))
        py, px = int(fy[k]), int(fx[k])
        cval = float(Cmap[py, px])

        wy = int(np.clip(py - r, 0, h - ps))
        wx = int(np.clip(px - r, 0, w - ps))
        T = np.ascontiguousarray(I[wy:wy + ps, wx:wx + ps])
        tk = known[wy:wy + ps, wx:wx + ps]
        w3 = np.ascontiguousarray(np.repeat(tk[..., None].astype(np.float32), 3, axis=2))

        res = cv2.matchTemplate(I, T, cv2.TM_SQDIFF, mask=w3)
        # many flat patches tie on the known pixels; prefer nearby ones so the unknown part is plausible
        res = res + locality * np.hypot(gy_ - wy, gx_ - wx)
        res[invalid] = np.inf
        qy, qx = np.unravel_index(np.argmin(res), res.shape)

        unk = ~tk
        dst = I[wy:wy + ps, wx:wx + ps]
        dst[unk] = I[qy:qy + ps, qx:qx + ps][unk]
        cwin = conf[wy:wy + ps, wx:wx + ps]
        cwin[unk] = cval
        M[wy:wy + ps, wx:wx + ps] = False

        iters += 1
        if progress is not None and iters % 10 == 0:
            progress(1.0 - M.sum() / max(total, 1))

    out = img.astype(np.float64).copy()
    win = out[Y0:Y1, X0:X1]
    win[M0] = I[M0]
    return out, {"patch": ps, "iterations": iters, "window": (int(h), int(w))}


# --------------------------------------------------------------------------- 5. LaMa (pretrained)
# LaMa = "Resolution-robust Large Mask Inpainting with Fourier Convolutions" (Suvorov et al., 2021).
# Unlike the four methods above it is NOT derived from a formula: its matrices (weights) were learned
# from millions of photos. We only load it and call it. Weights: ~200 MB, downloaded once.
LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"
LAMA_PATH = Path(os.environ.get("OBJREMOVE_CACHE", Path.home() / ".cache" / "objremove")) / "big-lama.pt"
_lama_model = None


def lama_is_cached() -> bool:
    return LAMA_PATH.exists()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_lama():
    """Fetch the weights once and verify the checksum (a model file can contain code, so we pin it)."""
    LAMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LAMA_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(LAMA_URL, tmp)
    except Exception as e:
        raise ValueError(f"Could not download the LaMa weights ({e}). Download them manually from\n"
                         f"{LAMA_URL}\nand save the file as {LAMA_PATH}") from e
    if _sha256(tmp) != LAMA_SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError("Downloaded LaMa file failed its checksum, so it was discarded. Try again.")
    tmp.replace(LAMA_PATH)


def _load_lama():
    global _lama_model
    if _lama_model is None:
        try:
            import torch
        except ImportError:
            raise ValueError("LaMa needs PyTorch. Install it with:  pip install torch")
        if not LAMA_PATH.exists():
            _download_lama()
        _lama_model = torch.jit.load(str(LAMA_PATH), map_location="cpu").eval()
    return _lama_model


def lama_fill(image: np.ndarray, mask: np.ndarray):
    """image: uint8 RGB (H, W, 3); mask: bool (H, W). Returns (uint8 result, info)."""
    model = _load_lama()  # raises a friendly error if PyTorch is missing
    import torch

    H, W = mask.shape
    ph, pw = (-H) % 8, (-W) % 8  # the network wants sizes divisible by 8
    img_p = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    m_p = np.pad(mask.astype(np.uint8), ((0, ph), (0, pw)), mode="symmetric")

    x = torch.from_numpy(img_p).permute(2, 0, 1).float().div(255)[None]   # (1, 3, H, W) in [0, 1]
    m = torch.from_numpy((m_p > 0).astype(np.float32))[None, None]        # (1, 1, H, W), 1 = hole
    with torch.no_grad():
        y = model(x, m)
    pred = (y[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)[:H, :W]

    out = image.copy()
    out[mask] = pred[mask]  # only the hole changes; everything else stays byte-identical
    return out, {"device": "cpu", "padded_to": (H + ph, W + pw)}


# --------------------------------------------------------------------------- dispatcher
def inpaint(image: np.ndarray, mask: np.ndarray, method: str = "diffusion", **params):
    """
    Remove the masked region from an RGB uint8 image.
    Returns (result_uint8, info_dict).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image of shape (H, W, 3).")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("Mask and image sizes differ.")
    if not mask.any():
        raise ValueError("Mask is empty: paint over the object you want to remove.")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(METHODS)}.")

    img = _to_float(image)
    t0 = time.perf_counter()
    if method == "lama":
        out_u8, info = lama_fill(image, mask)
        info["method"] = METHODS[method]
        info["hole_pixels"] = int(mask.sum())
        info["seconds"] = round(time.perf_counter() - t0, 3)
        return out_u8, info
    if method == "diffusion":
        out, info = laplace_fill(img, mask)
    elif method == "poisson":
        out, info = poisson_fill(img, mask, ring=int(params.get("ring", 6)))
    elif method == "lowrank":
        out, info = lowrank_fill(img, mask, rank=int(params.get("rank", 8)),
                                 iters=int(params.get("iters", 80)))
    else:
        patch = int(params.get("patch") or 0)
        if patch <= 0:  # auto: bigger holes need bigger patches to carry structure across
            y0, y1, x0, x1 = _bbox(mask)
            patch = int(np.clip(round(min(y1 - y0, x1 - x0) / 5), 9, 21)) | 1
        out, info = criminisi_fill(img, mask, patch=patch, progress=params.get("progress"))
    info["method"] = METHODS[method]
    info["hole_pixels"] = int(mask.sum())
    info["seconds"] = round(time.perf_counter() - t0, 3)
    return _to_u8(out), info

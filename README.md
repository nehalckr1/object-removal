# Object removal tool (matrix methods)

Paint over an unwanted object, press a button, get the background reconstructed.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # includes PyTorch for the LaMa option (large download)
python app.py            # opens http://127.0.0.1:7860
python benchmark.py      # writes results.md + benchmark_<image>.png
```

Files: `inpaint.py` (the four methods), `app.py` (Gradio UI), `benchmark.py` (scores vs ground truth).

## The idea in one picture

The photo is a grid of numbers with a hole punched in it. Everything outside the hole is known.
We need numbers for the hole that continue the surroundings naturally.

## The five methods

**1. Diffusion (Laplace solve).** Demand that each missing pixel equals the average of its 4 neighbours.
Writing that for all `n` hole pixels gives one sparse linear system

    A u = b,    A[p,p] = number of neighbours,  A[p,q] = -1 if q is a hole neighbour of p,
                b[p]   = sum of the known neighbours' colours

`A` is symmetric positive definite with at most 5 non-zeros per row, so it is solved exactly
(sparse LU, or conjugate gradient for very large holes). Result: smooth, no seams, but blurry.

**2. Poisson clone.** Same matrix `A`, different right-hand side. We search the photo (template matching,
i.e. a correlation sweep) for a region whose surroundings look like the hole's surroundings, and put that
region's *gradients* into `b`. The solution keeps the borrowed texture but is forced to agree with the
known pixels at the border, so there is no visible seam. If no region matches well it falls back to
diffusion rather than cloning something unrelated.

**3. Low-rank SVD completion.** Treat a window around the hole as a matrix `X` (colour channels side by side).
Natural patterns are close to low rank, so repeat: replace `X` by its best rank-`r` approximation
(truncated SVD, Eckart-Young), then put the known pixels back. Good for smooth or repetitive content;
streaky when the hole crosses strong structure.

**4. Patch-based (Criminisi).** Fill from the hole's border inward. At each step pick the border pixel with
highest priority (how much of its patch is trustworthy x how strongly an edge runs into it), find the
best-matching fully-known patch (masked SSD), copy it in. Best for larger holes with structure; slowest.

**5. LaMa (pretrained neural network).** The first four methods only reuse pixels from the same photo. LaMa
("Resolution-robust Large Mask Inpainting with Fourier Convolutions", Suvorov et al. 2021) is a network whose
weight matrices were *learned* from millions of photos, so it can fill in what usually belongs behind an object
even when it appears nowhere in the image. We do not train it: `inpaint.py` downloads the ~200 MB weights once
(checksum-verified), pads the image to a multiple of 8, runs the network on the CPU, and copies back only the
hole. Treat it as a black box: you can explain what it does and why it is stronger, not its layers line by line.
Needs `pip install torch`; the other four methods work without it.

## What the benchmark shows

We hide known pixels (discs of radius 12 / 25 / 45 px) in four 320x320 images, reconstruct, and compare
with the truth (`results.md` has every row, `benchmark_*.png` the pictures; columns are
masked | diffusion | Poisson | low-rank | Criminisi | LaMa | original).

| hole | diffusion | Poisson | low-rank | Criminisi | LaMa |
|---|---|---|---|---|---|
| small  | 27.9 dB | 32.5 dB | 24.2 dB | 27.9 dB | 35.8 dB |
| medium | 14.2 dB | 16.3 dB | 14.4 dB | 12.7 dB | 23.9 dB |
| large  | 14.6 dB | 16.8 dB | 12.2 dB | 12.5 dB | 18.0 dB |

(mean PSNR over the hole pixels, higher is better)

Honest reading:
- On **repetitive texture** (bricks) Poisson clone reconstructs the pattern almost perfectly; diffusion leaves a
  blurry blob; low-rank leaves coloured streaks.
- On **photos with unique content** (a face, a helmet) no method can recover what was really there, because it
  is not elsewhere in the image. PSNR falls to 8-17 dB for medium/large holes whatever the method. The
  goal there is a plausible fill, which PSNR only partly measures.
- Small holes in smooth areas are nearly solved by every method (up to 59 dB).
- **LaMa is clearly the strongest on photos with unique content** (about +7.6 dB over the best classical method on
  medium holes, +3.3 dB on small, +1.2 dB on large). Caveat: it was trained on millions of photos, and we cannot rule out that it has seen images
  similar to the sample photos, so this comparison favours it somewhat.
- Four images is a small sample. Treat the table as a comparison of behaviours, not a ranking.

## Limits

- Photos are downscaled to 640 px on the long side so Criminisi and the SVD stay interactive.
- Large holes through detailed structure (a person in front of a building) will show artifacts with all four methods.
- Methods 1-4 are classical on purpose: every step is a matrix operation you can explain. LaMa is the exception (see above).
- LaMa runs on the CPU (about 5 s for an 800 px photo). It loads through `torch.jit.load`, which newer PyTorch
  versions mark as deprecated, so a future release may need a small change. Its quality still depends on the mask:
  paint with a few pixels of slack around the object (the app enforces at least 8 px for LaMa) or you get a halo.
- Heavier diffusion-based models (Stable Diffusion inpainting, Qwen-Image-Edit and similar) give better quality
  but need tens of GB of memory or minutes per image, so they are not included.

## Worked example: removing a shoulder bag from a full-body photo

All four methods were tried on a bag + strap hiding an arm, a shirt edge and a waist. Diffusion smeared the body
outline, Poisson clone left speckle, low-rank left vertical streaks. Criminisi with the default 9 px patch left
fragments of the strap; with a larger patch (15-21 px) the bag disappeared and the arm and shirt lined up, with a
ragged shirt edge. That is why the patch size is now chosen automatically from the hole size. On the disc benchmark
the auto size is slightly *worse* for Criminisi on medium holes (PSNR 14.2 -> 12.7), because PSNR rewards blur on
content that cannot be recovered; judge this one by eye as well as by number.

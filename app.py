"""
Interactive object-removal tool.

Run:   python app.py
Then:  upload a photo, paint over the object, pick a method, press "Remove object".
"""
from __future__ import annotations

import importlib.util

import cv2
import gradio as gr
import numpy as np

from inpaint import METHODS, grow_mask, inpaint, lama_is_cached

MAX_SIDE = 640  # photos are downscaled so the heavier methods stay interactive

DESCRIPTIONS = {
    "diffusion": "Each missing pixel becomes the average of its neighbours. That is one sparse linear "
                 "system **A x = b**, solved exactly. Fast and seamless, but smooth: it blurs texture.",
    "poisson": "Finds a patch elsewhere in the photo whose surroundings match, then solves the **same "
               "A x = b** with that patch's *gradients* on the right-hand side. Keeps texture, no seam.",
    "lowrank": "Treats the region as a matrix and repeatedly takes its best rank-r approximation "
               "(truncated **SVD**) while re-imposing the known pixels. Good on smooth or repetitive "
               "backgrounds, streaky on strong structure.",
    "criminisi": "Fills the hole from the edges inward, copying the best-matching known patch for each "
                 "step (edge-continuing order). Best for larger holes with structure; slowest.",
    "lama": "A **pretrained neural network** (LaMa) that has learned from millions of photos what usually "
            "belongs behind an object. Not a formula you derive: its matrices were learned. Best on hard "
            "cases like people and structure. First use downloads ~200 MB; needs PyTorch.",
}


def _extract(editor):
    if editor is None or editor.get("background") is None:
        raise gr.Error("Upload an image first.")
    bg = editor["background"]
    rgb = bg[..., :3].astype(np.uint8)

    mask = np.zeros(rgb.shape[:2], dtype=bool)
    for layer in editor.get("layers") or []:
        if layer is None:
            continue
        if layer.shape[2] == 4:
            mask |= layer[..., 3] > 0
        else:
            mask |= layer.sum(axis=2) > 0
    if not mask.any():
        # fallback: strokes baked into the composite but not delivered as a layer
        comp = editor.get("composite")
        if comp is not None and comp.shape[:2] == rgb.shape[:2]:
            diff = np.abs(comp[..., :3].astype(int) - rgb.astype(int)).sum(axis=2)
            mask = diff > 60
    if not mask.any():
        raise gr.Error("Paint over the object you want to remove, then press the button.")
    return rgb, mask


def _downscale(rgb, mask):
    h, w = mask.shape
    s = MAX_SIDE / max(h, w)
    if s >= 1:
        return rgb, mask
    size = (max(1, round(w * s)), max(1, round(h * s)))
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) > 0
    return rgb, mask


def run(editor, method, grow, rank, auto_patch, patch, ring, progress=gr.Progress()):
    rgb, mask = _extract(editor)
    rgb, mask = _downscale(rgb, mask)
    grow = int(grow)
    lama_note = None
    if method == "lama" and grow < 8:
        grow = 8
        lama_note = "Mask margin raised to 8 px for LaMa: it needs slack around the object or leaves a halo."
    mask = grow_mask(mask, grow)
    if method == "lama" and importlib.util.find_spec("torch") and not lama_is_cached():
        gr.Info("First use of LaMa: downloading the model (~200 MB). This happens once.")

    params = {"rank": rank, "patch": 0 if auto_patch else patch, "ring": ring}
    if method == "criminisi":
        params["progress"] = lambda f: progress(f, desc="Filling patch by patch")
    else:
        progress(0.1, desc="Solving")

    try:
        out, info = inpaint(rgb, mask, method, **params)
    except ValueError as e:
        raise gr.Error(str(e))

    lines = [f"**{info['method']}**: {info['seconds']} s, hole = {info['hole_pixels']} px"]
    if method in ("diffusion", "poisson"):
        lines.append(f"System size {info['unknowns']} x {info['unknowns']}, "
                     f"{info['nonzeros']} non-zeros ({info['solver']}).")
    if method == "lowrank":
        lines.append(f"Rank {info['rank']} keeps {info['energy_kept']*100:.1f}% of the window's energy "
                     f"({info['svd_iterations']} SVD iterations).")
    if method == "criminisi":
        lines.append(f"{info['iterations']} patch copies, patch {info['patch']}x{info['patch']}.")
    if method == "lama":
        lines.append("Pretrained neural network, run on the CPU.")
    if lama_note:
        lines.append(lama_note)
    if "note" in info:
        lines.append(f"Note: {info['note']}")
    if max(editor["background"].shape[:2]) > MAX_SIDE:
        lines.append(f"(Image was downscaled to {MAX_SIDE}px on its long side for speed.)")

    return (rgb, out), "  \n".join(lines)


def describe(method):
    return DESCRIPTIONS[method]


with gr.Blocks(title="Object Removal") as demo:
    gr.Markdown("# Object removal by matrix methods\n"
                "Upload a photo, **paint over the object**, choose a method, press the button.")
    with gr.Row():
        with gr.Column():
            editor = gr.ImageEditor(
                label="1. Paint the object to remove",
                type="numpy", image_mode="RGBA", sources=["upload", "clipboard"],
                brush=gr.Brush(default_size=22, colors=["#ff3b30"], default_color="#ff3b30",
                               color_mode="fixed"),
                eraser=gr.Eraser(default_size=22),
                transforms=[], layers=False, height=520,
            )
            method = gr.Radio([(v, k) for k, v in METHODS.items()], value="diffusion",
                              label="2. Method")
            blurb = gr.Markdown(DESCRIPTIONS["diffusion"])
            grow = gr.Slider(0, 15, value=3, step=1, label="Grow mask by (px)",
                             info="Brush strokes usually stop just short of the object's edge.")
            with gr.Accordion("Method settings", open=False):
                rank = gr.Slider(1, 60, value=8, step=1, label="Low-rank: rank r")
                auto_patch = gr.Checkbox(value=True, label="Criminisi: auto patch size (scales with the hole)")
                patch = gr.Slider(5, 31, value=9, step=2, label="Criminisi: patch size (when auto is off)")
                ring = gr.Slider(2, 20, value=6, step=1, label="Poisson: matching ring width")
            go = gr.Button("3. Remove object", variant="primary")
        with gr.Column():
            result = gr.ImageSlider(label="Before / after", type="numpy", height=520)
            info = gr.Markdown()

    method.change(describe, method, blurb)
    go.click(run, [editor, method, grow, rank, auto_patch, patch, ring], [result, info])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)

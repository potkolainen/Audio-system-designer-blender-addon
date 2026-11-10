"""Software renderers for the report: receiver heatmap from above and volumetric fog from top and side. NumPy only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOOR_DB = -100.0
BACKGROUND = np.array([30.0, 30.0, 34.0])
_STOPS_T = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_STOPS_RGB = np.array([(0.02, 0.08, 0.85), (0.0, 0.75, 1.0), (0.0, 0.85, 0.15), (1.0, 0.85, 0.0), (0.9, 0.02, 0.0)])
NO_DATA_RGBA = (0.15, 0.15, 0.15, 1.0)


@dataclass
class RenderedImage:
    rgb: np.ndarray            # (H, W, 3) uint8
    extent: tuple              # (x0, x1, y0, y1) in metres, y0 at the bottom of the image
    x_label: str = "x (m)"
    y_label: str = "y (m)"


def colormap(values, lo, hi) -> np.ndarray:
    """(N, 4) float RGBA. Values at or below FLOOR_DB are drawn as no-data grey."""
    values = np.asarray(values, dtype=float)
    t = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgba = np.ones((len(values), 4))
    for c in range(3):
        rgba[:, c] = np.interp(t, _STOPS_T, _STOPS_RGB[:, c])
    rgba[values <= FLOOR_DB] = NO_DATA_RGBA
    return rgba


def colormap_rgb8(values, lo, hi) -> np.ndarray:
    return np.round(colormap(values, lo, hi)[:, :3] * 255).astype(np.uint8)


def legend_strip(width: int = 256) -> np.ndarray:
    """(1, width, 3) uint8 colour bar from low to high."""
    return colormap_rgb8(np.linspace(0.0, 1.0, width), 0.0, 1.0)[None, :, :]


def render_receivers_top(points: np.ndarray, rgb8: np.ndarray, cell_m: float, max_px: int = 900) -> RenderedImage:
    """Plan view of coloured cells. Higher cells are drawn over lower ones."""
    pts = np.asarray(points, dtype=float)
    x0, x1 = pts[:, 0].min() - cell_m / 2, pts[:, 0].max() + cell_m / 2
    y0, y1 = pts[:, 1].min() - cell_m / 2, pts[:, 1].max() + cell_m / 2
    scale = max_px / max(x1 - x0, y1 - y0, 1e-6)
    width, height = int(np.ceil((x1 - x0) * scale)) + 1, int(np.ceil((y1 - y0) * scale)) + 1
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = BACKGROUND.astype(np.uint8)
    order = np.argsort(pts[:, 2], kind="stable")
    px = ((pts[order, 0] - x0) * scale).astype(int)
    py = ((y1 - pts[order, 1]) * scale).astype(int)
    colors = np.asarray(rgb8)[order]
    k = max(1, int(round(cell_m * scale)))
    for dy in range(-(k // 2), k - k // 2):
        for dx in range(-(k // 2), k - k // 2):
            img[np.clip(py + dy, 0, height - 1), np.clip(px + dx, 0, width - 1)] = colors
    return RenderedImage(img, (x0, x1, y0, y1))


def _resample(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of a float (h, w, 3) image."""
    h, w = img.shape[:2]
    ys = np.clip((np.arange(height) + 0.5) * h / height - 0.5, 0, h - 1)
    xs = np.clip((np.arange(width) + 0.5) * w / width - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    bottom = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return top * (1 - fy) + bottom * fy


def render_fog(t: np.ndarray, view: str, dims_m, opacity: float = 0.35, threshold: float = 0.1,
               max_px: int = 900) -> RenderedImage:
    """Composite the normalised field t (nz, ny, nx) front to back with the viewport's transfer function.

    view 'top' looks down -Z (axes x, y). view 'side' looks from +X towards -X (axes y to the right, z up).
    dims_m is the region size (dx, dy, dz) in metres. Axes are the region's own axes.
    """
    nz, ny, nx = t.shape
    dx, dy, dz = dims_m
    if view == "top":
        slices = [t[k] for k in range(nz - 1, -1, -1)]
        along, h_m, v_m = nz, dx, dy
        extent = (-dx / 2, dx / 2, -dy / 2, dy / 2)
        labels = ("region x (m)", "region y (m)")
    elif view == "side":
        slices = [t[:, :, i] for i in range(nx - 1, -1, -1)]
        along, h_m, v_m = nx, dy, dz
        extent = (-dy / 2, dy / 2, -dz / 2, dz / 2)
        labels = ("region y (m)", "region z (m)")
    else:
        raise ValueError("view must be 'top' or 'side'")

    h, w = slices[0].shape
    colour = np.zeros((h, w, 3))
    acc = np.zeros((h, w))
    step = 2.0 / along
    span = max(1.0 - threshold, 1e-6)
    for v in slices:
        s = np.clip((v - threshold) / span, 0.0, 1.0)
        a = np.where(v > threshold, 1.0 - np.exp(-opacity * 6.0 * s * s * step), 0.0)
        c = colormap(v.ravel(), 0.0, 1.0)[:, :3].reshape(h, w, 3)
        colour += ((1.0 - acc) * a)[..., None] * c
        acc += (1.0 - acc) * a
    img = (colour * 255.0 + (1.0 - acc)[..., None] * BACKGROUND)
    img = img[::-1]  # row 0 = highest y (or z)
    out_w = max_px
    out_h = int(np.clip(round(max_px * v_m / max(h_m, 1e-6)), 2, int(max_px * 1.5)))
    img = np.clip(_resample(img, out_h, out_w), 0, 255).astype(np.uint8)
    return RenderedImage(img, extent, labels[0], labels[1])

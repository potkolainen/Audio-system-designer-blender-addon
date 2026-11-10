"""Line-array geometry, auto-splay, and subwoofer-array recipes. Pure NumPy, no Blender imports.

Array-local frame: +Y forward, +Z up, origin at the top front corner of the first cabinet.
Geometry is a purely kinematic approximation (front-hinged cabinets). It is NOT rigging data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustics import PredictionSettings, predict, predict_coherent


@dataclass
class LineArrayLayout:
    centres: np.ndarray        # (n, 3) cabinet centres (mesh origin)
    front_centres: np.ndarray  # (n, 3) centre of each front face
    tilts_deg: np.ndarray      # (n,) downward tilt of each cabinet axis
    height_m: float
    depth_m: float

    @property
    def extent_m(self) -> tuple:
        """(depth, height) of the bounding box of all cabinet front and rear face corners."""
        phi = np.radians(self.tilts_deg)
        down = np.stack([np.zeros(len(phi)), -np.sin(phi), -np.cos(phi)], axis=1)
        rear = self.centres - (self.front_centres - self.centres)
        pts = np.vstack([c + s * down * self.height_m / 2.0 for c in (self.front_centres, rear) for s in (1, -1)])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return float(hi[1] - lo[1]), float(hi[2] - lo[2])


def line_array_layout(n: int, cab_height_m: float, cab_depth_m: float, splays_deg, top_tilt_deg: float) -> LineArrayLayout:
    """splays_deg[i] is the extra downward tilt of cabinet i+1 relative to cabinet i."""
    if n < 1:
        raise ValueError("n must be >= 1")
    splays = np.zeros(max(n - 1, 0))
    given = np.asarray(splays_deg, dtype=float)[: max(n - 1, 0)]
    splays[: given.size] = given
    tilts = top_tilt_deg + np.concatenate([[0.0], np.cumsum(splays)])
    phi = np.radians(tilts)
    down = np.stack([np.zeros(n), -np.sin(phi), -np.cos(phi)], axis=1)
    axis = np.stack([np.zeros(n), np.cos(phi), -np.sin(phi)], axis=1)
    tops = np.vstack([[0.0, 0.0, 0.0], np.cumsum(down[:-1] * cab_height_m, axis=0)]) if n > 1 else np.zeros((1, 3))
    front = tops + down * cab_height_m / 2.0
    centres = front - axis * cab_depth_m / 2.0
    return LineArrayLayout(centres, front, tilts, cab_height_m, cab_depth_m)


def auto_splay(n: int, cab_height_m: float, cab_depth_m: float, hang_height_m: float, near_m: float, far_m: float,
               min_splay_deg: float = 0.0, max_splay_deg: float = 10.0, iterations: int = 12):
    """Aim cabinet i at evenly spaced points on the audience line (top cabinet farthest).

    hang_height_m is the height of the array's top above the audience plane. Returns (top_tilt_deg, splays_deg).
    """
    if hang_height_m <= 0:
        raise ValueError("The array must hang above the audience plane.")
    if not far_m > near_m > 0:
        raise ValueError("Need 0 < near < far.")
    targets = far_m - (far_m - near_m) * (np.arange(n) + 0.5) / n
    top_tilt, splays = 0.0, np.zeros(max(n - 1, 0))
    for _ in range(iterations):
        lay = line_array_layout(n, cab_height_m, cab_depth_m, splays, top_tilt)
        dy = targets - lay.front_centres[:, 1]
        dz = -hang_height_m - lay.front_centres[:, 2]
        aim = np.degrees(np.arctan2(-dz, dy))
        top_tilt = float(aim[0])
        splays = np.clip(np.diff(aim), min_splay_deg, max_splay_deg)
    return top_tilt, splays


def audience_aim_distances(layout: LineArrayLayout, hang_height_m: float) -> np.ndarray:
    """Horizontal distance from the hang point at which each cabinet axis meets the audience plane."""
    phi = np.radians(layout.tilts_deg)
    drop = hang_height_m + layout.front_centres[:, 2]
    with np.errstate(divide="ignore"):
        dist = np.where(np.sin(phi) > 1e-6, drop / np.tan(phi), np.inf)
    return layout.front_centres[:, 1] + dist


def array_summary(layout: LineArrayLayout, weight_kg_each: float) -> dict:
    depth_extent, height_extent = layout.extent_m
    n = len(layout.centres)
    return {
        "cabinets": n,
        "total_weight_kg": weight_kg_each * n,
        "height_m": height_extent,
        "depth_m": depth_extent,
        "total_splay_deg": float(layout.tilts_deg[-1] - layout.tilts_deg[0]),
        # Equal-mass cabinets, centre-of-body positions only; ignores bumper and rigging.
        "cog_local_m": tuple(float(v) for v in layout.centres.mean(axis=0)),
    }


@dataclass
class SubPlacement:
    x: float
    y: float
    yaw_deg: float
    delay_ms: float
    polarity: int


SUB_KINDS = ("BROADSIDE", "ENDFIRE", "CARDIOID")


def sub_array_layout(kind: str, count: int, spacing_m: float, c: float = 343.0) -> list[SubPlacement]:
    """Acoustic-centre positions (array-local, +Y forward) with the delay/polarity recipe for each unit.

    BROADSIDE: row of `count` subs side by side, no delay.
    ENDFIRE: `count` subs in depth, rear unit undelayed, each unit towards the front delayed by spacing/c more.
    CARDIOID: `count` columns, each a forward sub plus an inverted rear-facing sub delayed by spacing/c.
    """
    if count < 1 or spacing_m <= 0:
        raise ValueError("count must be >= 1 and spacing > 0")
    step_ms = spacing_m / c * 1000.0
    if kind == "BROADSIDE":
        xs = (np.arange(count) - (count - 1) / 2.0) * spacing_m
        return [SubPlacement(float(x), 0.0, 0.0, 0.0, 1) for x in xs]
    if kind == "ENDFIRE":
        return [SubPlacement(0.0, -i * spacing_m, 0.0, (count - 1 - i) * step_ms, 1) for i in range(count)]
    if kind == "CARDIOID":
        xs = (np.arange(count) - (count - 1) / 2.0) * spacing_m * 2.0
        out = []
        for x in xs:
            out.append(SubPlacement(float(x), 0.0, 0.0, 0.0, 1))
            out.append(SubPlacement(float(x), -spacing_m, 180.0, step_ms, -1))
        return out
    raise ValueError(f"unknown sub array kind '{kind}'")


def ring_levels(sources, centre, radius_m: float, freqs, settings: PredictionSettings, angles_deg, forward_xy, z=None):
    """Coherent SPL (A, F) on a horizontal ring around `centre`; angle 0 = `forward_xy` direction.

    Used for front/back and polar analysis of subwoofer arrays.
    """
    centre = np.asarray(centre, dtype=float)
    fwd = np.asarray(forward_xy, dtype=float)[:2]
    fwd = fwd / np.linalg.norm(fwd)
    left = np.array([-fwd[1], fwd[0]])
    a = np.radians(np.asarray(angles_deg, dtype=float))
    dirs = np.outer(np.cos(a), fwd) + np.outer(np.sin(a), left)
    pts = np.column_stack([centre[0] + radius_m * dirs[:, 0], centre[1] + radius_m * dirs[:, 1],
                           np.full(len(a), centre[2] if z is None else z)])
    return predict_coherent(sources, pts, freqs, settings).spl


def line_profile(sources, origin_xy, forward_xy, z: float, near_m: float, far_m: float,
                 settings: PredictionSettings, samples: int = 25):
    """Incoherent octave prediction along the audience line. Returns (distances, PredictionResult)."""
    fwd = np.asarray(forward_xy, dtype=float)[:2]
    fwd = fwd / np.linalg.norm(fwd)
    d = np.linspace(near_m, far_m, samples)
    pts = np.column_stack([origin_xy[0] + d * fwd[0], origin_xy[1] + d * fwd[1], np.full(samples, z)])
    return d, predict(sources, pts, settings)

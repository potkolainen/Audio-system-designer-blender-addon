"""Specular reflections by image sources on triangular reflectors, first and second order. No Blender imports.

Duck-typed on the Source dataclass (position, right, up, forward) so it does not import the prediction module.
A path is valid when every mirrored segment crosses its triangle with source and listener on the lit side of each
plane. Blocking by other geometry is handled by the caller through the returned waypoints.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

MIN_POWER = 1e-4


@dataclass(frozen=True)
class Reflector:
    tri: np.ndarray            # (3, 3) world-space vertices
    absorption: np.ndarray     # (8,) octave-band absorption coefficients 63 Hz-8 kHz
    scattering: float = 0.0    # share of reflected energy removed from the specular path
    name: str = ""


@dataclass
class Plane:
    tri: np.ndarray
    n: np.ndarray
    d: float
    area: float
    power: np.ndarray          # (8,) linear power factor per octave band


@dataclass
class ReflectionPath:
    source: object             # the real source
    image: object              # final mirrored source
    power: np.ndarray          # (8,) product of the plane power factors
    planes: list
    images: list               # image positions after each mirror, in order
    sides: list                # side of each plane the incoming source (or image) lies on

    @property
    def order(self) -> int:
        return len(self.planes)


def _plane(r: Reflector):
    tri = np.asarray(r.tri, dtype=float)
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    length = np.linalg.norm(n)
    if length < 1e-12:
        return None
    n = n / length
    power = (np.maximum(1.0 - np.asarray(r.absorption, dtype=float), MIN_POWER)
             * max(1.0 - r.scattering, MIN_POWER))
    return Plane(tri, n, float(n @ tri[0]), 0.5 * length, power)


def _mirror_source(s, plane: Plane):
    dist = float(plane.n @ s.position - plane.d)
    if abs(dist) < 1e-6:
        return None, 0.0
    m = lambda v: v - 2.0 * (v @ plane.n) * plane.n  # noqa: E731
    img = replace(s, position=s.position - 2.0 * dist * plane.n, right=m(s.right), up=m(s.up), forward=m(s.forward))
    return img, float(np.sign(dist))


def build_paths(sources, reflectors, order: int = 1, order2_limit: int = 24) -> list[ReflectionPath]:
    planes = [p for p in (_plane(r) for r in reflectors) if p is not None]
    paths = []
    for s in sources:
        firsts = []
        for pa in planes:
            img, side = _mirror_source(s, pa)
            if img is None:
                continue
            firsts.append((pa, img, side))
            paths.append(ReflectionPath(s, img, pa.power, [pa], [img.position], [side]))
        if order < 2:
            continue
        largest = sorted(planes, key=lambda p: -p.area)[:order2_limit]
        keep = {id(p) for p in largest}
        for pa, img1, side_a in firsts:
            if id(pa) not in keep:
                continue
            for pb in largest:
                if pb is pa:
                    continue
                img2, side_b = _mirror_source(img1, pb)
                if img2 is None:
                    continue
                paths.append(ReflectionPath(s, img2, pa.power * pb.power, [pa, pb],
                                            [img1.position, img2.position], [side_a, side_b]))
    return paths


def _hit(image_pos, targets, plane: Plane, side: float):
    """Rays image -> targets crossing the triangle, with targets on the lit side. Returns (mask, crossing points)."""
    same_side = (targets @ plane.n - plane.d) * side > 1e-9
    direction = targets - image_pos
    denom = direction @ plane.n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane.d - image_pos @ plane.n) / denom
        q = image_pos + t[:, None] * direction
    ok = same_side & np.isfinite(t) & (t > 1e-6) & (t < 1.0 - 1e-6)
    if ok.any():
        a, b, c = plane.tri
        v0, v1, v2 = c - a, b - a, q - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        den = d00 * d11 - d01 * d01
        d20, d21 = v2 @ v0, v2 @ v1
        u = (d11 * d20 - d01 * d21) / den
        v = (d00 * d21 - d01 * d20) / den
        ok &= (u >= 0) & (v >= 0) & (u + v <= 1)
    return ok, q


def trace(path: ReflectionPath, points: np.ndarray):
    """Boolean mask over `points` and the reflection points [Q1..Qk], each aligned with the masked points."""
    idx = np.arange(len(points))
    targets = points
    waypoints: list = []
    for j in range(path.order - 1, -1, -1):
        ok, q = _hit(path.images[j], targets, path.planes[j], path.sides[j])
        idx = idx[ok]
        waypoints = [w[ok] for w in waypoints]
        waypoints.insert(0, q[ok])
        targets = q[ok]
        if not len(idx):
            break
    mask = np.zeros(len(points), dtype=bool)
    mask[idx] = True
    return mask, waypoints

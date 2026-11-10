"""Mesh analysis for automatic receiver areas. Pure NumPy, no Blender imports."""
from __future__ import annotations

import numpy as np

_lattices: dict[int, np.ndarray] = {}


def triangle_areas_normals(tris: np.ndarray):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(n, axis=1)
    return 0.5 * length, n / np.maximum(length, 1e-12)[:, None]


def walkable_islands(tris: np.ndarray, max_slope_deg: float = 45.0, weld_tol: float = 1e-3):
    """Group upward-facing triangles into connected islands.

    Triangles count as connected when they share a vertex (within weld_tol), so steps whose treads are separated
    by vertical risers become separate islands instead of being rejected as one uneven surface.
    Returns (island_id per triangle, -1 where not upward facing; area per island).
    """
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    areas, normals = triangle_areas_normals(tris)
    cand = np.nonzero((normals[:, 2] >= np.cos(np.radians(max_slope_deg))) & (areas > 1e-9))[0]
    ids = np.full(len(tris), -1, dtype=int)
    if not len(cand):
        return ids, np.zeros(0)
    key = np.round(tris[cand].reshape(-1, 3) / weld_tol).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    corners = inverse.reshape(-1, 3)
    parent = np.arange(int(inverse.max()) + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in corners:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(v) for v in corners[:, 0]])
    _, labels = np.unique(roots, return_inverse=True)
    ids[cand] = labels
    return ids, np.bincount(labels, weights=areas[cand])


def _lattice(n: int) -> np.ndarray:
    if n not in _lattices:
        cells = []
        for i in range(n):
            for j in range(n - i):
                cells.append(((i, j), (i + 1, j), (i, j + 1)))
                if i + j < n - 1:
                    cells.append(((i + 1, j), (i + 1, j + 1), (i, j + 1)))
        _lattices[n] = np.array(cells, dtype=float) / n
    return _lattices[n]


def subdivide_triangles(tris: np.ndarray, cell_m: float) -> np.ndarray:
    """Split every triangle into n*n similar triangles so that no edge is longer than about cell_m."""
    out = []
    for a, b, c in np.asarray(tris, dtype=float).reshape(-1, 3, 3):
        edge = max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c))
        lat = _lattice(max(1, int(np.ceil(edge / cell_m))))
        out.append(a + lat[..., 0:1] * (b - a) + lat[..., 1:2] * (c - a))
    return np.concatenate(out) if out else np.empty((0, 3, 3))

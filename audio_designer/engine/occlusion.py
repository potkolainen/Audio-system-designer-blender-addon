"""Sound-path occlusion. No Blender imports.

An occluder answers whether the straight segment between two points is blocked. A blocked path is attenuated by a
frequency-dependent amount (a crude stand-in for diffraction), it is not removed.
"""
from __future__ import annotations

import numpy as np

# dB of attenuation per octave band 63 Hz-8 kHz for a blocked path. An assumption, scaled by the user.
DEFAULT_SHADOW_DB = np.array([3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0])
SEGMENT_NUDGE_M = 0.02


class TriangleOccluder:
    """Brute-force segment/triangle test. Suitable for tests and small scenes."""

    def __init__(self, triangles):
        self.tris = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)

    def blocked(self, a, b) -> np.ndarray:
        b = np.atleast_2d(np.asarray(b, dtype=float))
        a = np.broadcast_to(np.asarray(a, dtype=float), b.shape)
        length = np.linalg.norm(b - a, axis=1)
        unit = (b - a) / np.maximum(length, 1e-12)[:, None]
        start = a + unit * SEGMENT_NUDGE_M
        d = b - unit * SEGMENT_NUDGE_M - start
        hit = np.zeros(len(b), dtype=bool)
        for v0, v1, v2 in self.tris:
            e1, e2 = v1 - v0, v2 - v0
            h = np.cross(d, e2)
            det = h @ e1
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            s = start - v0
            u = inv * np.einsum("ij,ij->i", s, h)
            q = np.cross(s, e1)
            v = inv * np.einsum("ij,ij->i", d, q)
            t = inv * (q @ e2)
            hit |= ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t < 1)
        return hit & (length > 4 * SEGMENT_NUDGE_M)

"""Everything outside the loudspeakers that shapes the sound field. No Blender imports."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .bands import to_freqs
from .occlusion import DEFAULT_SHADOW_DB


@dataclass
class Environment:
    reflectors: list = field(default_factory=list)
    occluder: object | None = None          # needs .blocked(a (N,3) or (1,3), b (N,3)) -> bool (N,)
    shadow_scale: float = 1.0
    reflection_order: int = 1
    order2_limit: int = 24

    def shadow_db(self, freqs) -> np.ndarray:
        return self.shadow_scale * to_freqs(DEFAULT_SHADOW_DB[None, :], freqs)[0]

    def blocked(self, a, b) -> np.ndarray:
        return self.occluder.blocked(a, b)

    def limited(self, max_reflectors: int, order: int = 1) -> "Environment":
        """Cheaper copy for live feedback: only the largest reflectors, at most `order`."""
        def area(r):
            t = np.asarray(r.tri, dtype=float)
            return np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))
        top = sorted(self.reflectors, key=lambda r: -area(r))[:max_reflectors]
        return replace(self, reflectors=top, reflection_order=min(self.reflection_order, order))


def as_environment(env) -> Environment:
    """None, a list of reflectors, or an Environment."""
    if env is None:
        return Environment()
    if isinstance(env, Environment):
        return env
    return Environment(reflectors=list(env))

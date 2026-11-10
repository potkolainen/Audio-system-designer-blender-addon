"""Point-source prediction: incoherent octave-band and phase-aware (coherent) 1/3-octave. No Blender imports."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atmosphere import (
    THIRD_OCTAVES, a_weighting_db, air_absorption_db_per_m, c_weighting_db, speed_of_sound,
)
from .bands import BANDS, BANDS_ARR as _BANDS_ARR, NUM_BANDS, to_freqs
from .environment import as_environment
from .reflections import build_paths, trace

A_WEIGHTING_DB = a_weighting_db(_BANDS_ARR)
C_WEIGHTING_DB = c_weighting_db(_BANDS_ARR)

FLOOR_DB = -100.0
MIN_DISTANCE_M = 1.0
MIN_DIRECTIVITY_DB = -60.0
CHUNK_POINTS = 20000


def db_sum(levels_db: np.ndarray, axis: int = 0) -> np.ndarray:
    """Energy sum of levels in dB along an axis. Empty axis gives FLOOR_DB."""
    levels = np.asarray(levels_db, dtype=float)
    if levels.shape[axis] == 0:
        shape = list(levels.shape)
        del shape[axis]
        return np.full(shape, FLOOR_DB)
    peak = np.max(levels, axis=axis, keepdims=True)
    total = peak + 10.0 * np.log10(np.sum(10.0 ** ((levels - peak) / 10.0), axis=axis, keepdims=True))
    return np.squeeze(total, axis=axis)


@dataclass(frozen=True)
class PolarTable:
    """Symmetric polar response: levels_db[band, angle], angles 0..180 degrees, 0 dB = on-axis."""

    angles_deg: np.ndarray
    levels_db: np.ndarray

    def at(self, angles_deg: np.ndarray) -> np.ndarray:
        """Return levels (N, NUM_BANDS) for absolute angles (N,)."""
        angles = np.abs(np.asarray(angles_deg, dtype=float))
        out = np.empty((angles.size, NUM_BANDS))
        for b in range(NUM_BANDS):
            out[:, b] = np.interp(angles, self.angles_deg, self.levels_db[b])
        return out

    def coverage_angle(self, band_index: int, drop_db: float = -6.0) -> float:
        """Off-axis angle at which the response first falls to drop_db (180 if it never does)."""
        levels = self.levels_db[band_index]
        below = np.nonzero(levels <= drop_db)[0]
        if below.size == 0:
            return float(self.angles_deg[-1])
        i = int(below[0])
        if i == 0:
            return float(self.angles_deg[0])
        a0, a1 = self.angles_deg[i - 1], self.angles_deg[i]
        l0, l1 = levels[i - 1], levels[i]
        return float(a0 + (a1 - a0) * (drop_db - l0) / (l1 - l0))


@dataclass(frozen=True)
class AcousticModel:
    """Acoustic behaviour of a loudspeaker, independent of its 3D mesh.

    Directivity is separable, D(h, v) = H(h) + V(v), and magnitude-only (zero phase).
    """

    max_spl_1m_db: float
    response_db: np.ndarray
    horizontal: PolarTable
    vertical: PolarTable


@dataclass
class Source:
    model: AcousticModel
    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    level_db: float = -10.0
    delay_ms: float = 0.0
    polarity: int = 1
    enabled: bool = True
    name: str = ""


@dataclass
class PredictionSettings:
    temperature_c: float = 20.0
    humidity_pct: float = 50.0
    pressure_kpa: float = 101.325
    air_absorption: bool = True

    @property
    def c(self) -> float:
        return speed_of_sound(self.temperature_c, self.humidity_pct, self.pressure_kpa)

    def absorption(self, freqs) -> np.ndarray:
        if not self.air_absorption:
            return np.zeros(np.shape(freqs))
        return air_absorption_db_per_m(freqs, self.temperature_c, self.humidity_pct, self.pressure_kpa)


def _geometry(source: Source, points: np.ndarray):
    delta = points - source.position
    dist = np.linalg.norm(delta, axis=1)
    x, y, z = delta @ source.right, delta @ source.up, delta @ source.forward
    return dist, np.degrees(np.arctan2(x, z)), np.degrees(np.arctan2(y, z))


def _levels(source: Source, points: np.ndarray, freqs, settings: PredictionSettings):
    """Return (levels_db (N, F), distance (N,)) of one source. freqs may be any frequencies."""
    dist, h, v = _geometry(source, points)
    directivity = np.maximum(source.model.horizontal.at(h) + source.model.vertical.at(v), MIN_DIRECTIVITY_DB)
    eff = np.maximum(dist, MIN_DISTANCE_M)
    band_levels = (source.model.max_spl_1m_db + source.level_db
                   + source.model.response_db[None, :] + directivity)
    levels = (to_freqs(band_levels, freqs) - 20.0 * np.log10(eff)[:, None]
              - settings.absorption(freqs)[None, :] * eff[:, None])
    return levels, dist


def _chunks(points: np.ndarray):
    for i in range(0, len(points), CHUNK_POINTS):
        yield slice(i, i + CHUNK_POINTS)


def _arrivals(active, chunk, freqs, settings, env, paths):
    """Yield (mask or None, levels_db (n, F), distance (n,), source) for every direct and reflected arrival.

    Levels already include reflection losses and the shadow loss of blocked paths.
    """
    shadow = env.shadow_db(freqs) if env.occluder is not None else None
    for s in active:
        levels, dist = _levels(s, chunk, freqs, settings)
        if shadow is not None:
            levels = levels - env.blocked(s.position[None, :], chunk)[:, None] * shadow[None, :]
        yield None, levels, dist, s
    for path in paths:
        mask, waypoints = trace(path, chunk)
        if not mask.any():
            continue
        pts = chunk[mask]
        levels, dist = _levels(path.image, pts, freqs, settings)
        levels = levels + 10.0 * np.log10(to_freqs(path.power[None, :], freqs)[0])[None, :]
        if shadow is not None:
            chain = [np.broadcast_to(path.source.position, pts.shape)] + waypoints + [pts]
            blocked = np.zeros(len(pts), dtype=bool)
            for a, b in zip(chain[:-1], chain[1:]):
                blocked |= env.blocked(a, b)
            levels = levels - blocked[:, None] * shadow[None, :]
        yield mask, levels, dist, path.image


def _build_paths(active, env):
    return build_paths(active, env.reflectors, env.reflection_order, env.order2_limit)


def source_contributions(sources, points, settings: PredictionSettings | None = None, freqs=None):
    """Per-source direct levels for enabled sources (no environment).

    Returns (levels (S, N, F), distance (S, N), arrival_ms (S, N), active_sources).
    """
    settings = settings or PredictionSettings()
    freqs = _BANDS_ARR if freqs is None else np.asarray(freqs, dtype=float)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    active = [s for s in sources if s.enabled]
    c = settings.c
    levels = np.empty((len(active), len(pts), len(freqs)))
    dists = np.empty((len(active), len(pts)))
    for i, s in enumerate(active):
        levels[i], dists[i] = _levels(s, pts, freqs, settings)
    delays = np.array([s.delay_ms for s in active])[:, None]
    arrival = dists / c * 1000.0 + delays
    return levels, dists, arrival, active


def _metric_from(name: str, spl: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    if name == "A":
        return db_sum(spl + a_weighting_db(freqs), axis=1)
    if name == "C":
        return db_sum(spl + c_weighting_db(freqs), axis=1)
    if name == "BROADBAND":
        return db_sum(spl, axis=1)
    idx = int(np.argmin(np.abs(freqs - float(name))))
    return spl[:, idx]


@dataclass
class PredictionResult:
    """Free-field, incoherent (power-summed) octave-band prediction."""

    band_spl: np.ndarray
    a_weighted: np.ndarray = field(init=False)
    c_weighted: np.ndarray = field(init=False)
    broadband: np.ndarray = field(init=False)

    def __post_init__(self):
        self.a_weighted = db_sum(self.band_spl + A_WEIGHTING_DB, axis=1)
        self.c_weighted = db_sum(self.band_spl + C_WEIGHTING_DB, axis=1)
        self.broadband = db_sum(self.band_spl, axis=1)

    def metric(self, name: str) -> np.ndarray:
        """name is 'A', 'C', 'BROADBAND' or an octave band centre as a string."""
        return _metric_from(name, self.band_spl, _BANDS_ARR)


def predict(sources, points, settings: PredictionSettings | None = None, env=None) -> PredictionResult:
    """Incoherent octave-band prediction: direct sound plus the reflections and shadowing of `env`."""
    settings = settings or PredictionSettings()
    env = as_environment(env)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    active = [s for s in sources if s.enabled]
    paths = _build_paths(active, env)
    energy = np.zeros((len(pts), NUM_BANDS))
    for sl in _chunks(pts):
        acc = energy[sl]
        for mask, levels, _, _ in _arrivals(active, pts[sl], _BANDS_ARR, settings, env, paths):
            if mask is None:
                acc += 10.0 ** (levels / 10.0)
            else:
                acc[mask] += 10.0 ** (levels / 10.0)
    spl = np.where(energy > 0, 10.0 * np.log10(np.maximum(energy, 1e-30)), FLOOR_DB)
    return PredictionResult(band_spl=np.maximum(spl, FLOOR_DB))


@dataclass
class CoherentResult:
    """Phase-aware prediction: complex pressures summed per frequency, free field."""

    freqs: np.ndarray
    spl: np.ndarray

    def metric(self, name: str) -> np.ndarray:
        """name is 'A', 'C', 'BROADBAND' or a frequency in Hz as a string (nearest computed one)."""
        return _metric_from(name, self.spl, self.freqs)


def _pressure(levels, dist, source, freqs, settings):
    t = dist / settings.c + source.delay_ms / 1000.0
    phase = -2.0 * np.pi * freqs[None, :] * t[:, None]
    return source.polarity * 10.0 ** (levels / 20.0) * np.exp(1j * phase)


def _to_spl(pressure: np.ndarray) -> np.ndarray:
    mag = np.abs(pressure)
    return np.maximum(np.where(mag > 0, 20.0 * np.log10(np.maximum(mag, 1e-30)), FLOOR_DB), FLOOR_DB)


def predict_coherent(sources, points, freqs=THIRD_OCTAVES, settings: PredictionSettings | None = None,
                     env=None) -> CoherentResult:
    """Phase-aware prediction. Reflections add with pressure coefficient sqrt(1 - alpha)."""
    settings = settings or PredictionSettings()
    env = as_environment(env)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    active = [s for s in sources if s.enabled]
    paths = _build_paths(active, env)
    p = np.zeros((len(pts), len(freqs)), dtype=complex)
    for sl in _chunks(pts):
        acc = p[sl]
        for mask, levels, dist, src in _arrivals(active, pts[sl], freqs, settings, env, paths):
            z = _pressure(levels, dist, src, freqs, settings)
            if mask is None:
                acc += z
            else:
                acc[mask] += z
    return CoherentResult(freqs=freqs, spl=_to_spl(p))


@dataclass
class ProbeResponse:
    freqs: np.ndarray
    total_spl: np.ndarray
    total_phase_deg: np.ndarray
    source_spl: np.ndarray
    source_phase_deg: np.ndarray
    distance_m: np.ndarray
    arrival_ms: np.ndarray
    names: list
    reflection_paths: int = 0


def probe_response(sources, point, freqs=THIRD_OCTAVES, settings: PredictionSettings | None = None,
                   env=None) -> ProbeResponse:
    """Combined and per-source complex response at one point. Reflections count in the total only."""
    settings = settings or PredictionSettings()
    env = as_environment(env)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    pt = np.asarray(point, dtype=float).reshape(1, 3)
    active = [s for s in sources if s.enabled]
    paths = _build_paths(active, env)
    parts = np.zeros((len(active), len(freqs)), dtype=complex)
    total = np.zeros(len(freqs), dtype=complex)
    dist = np.zeros(len(active))
    n_refl = 0
    direct = 0
    for mask, levels, d, src in _arrivals(active, pt, freqs, settings, env, paths):
        z = _pressure(levels, d, src, freqs, settings)[0]
        total += z
        if mask is None:
            parts[direct] = z
            dist[direct] = d[0]
            direct += 1
        else:
            n_refl += 1
    delays = np.array([s.delay_ms for s in active])
    return ProbeResponse(
        freqs=freqs, total_spl=_to_spl(total), total_phase_deg=np.degrees(np.angle(total)),
        source_spl=_to_spl(parts), source_phase_deg=np.degrees(np.angle(parts)),
        distance_m=dist, arrival_ms=dist / settings.c * 1000.0 + delays, names=[s.name for s in active],
        reflection_paths=n_refl)


def coverage_stats(values: np.ndarray) -> dict:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid) & (valid > FLOOR_DB)]
    if valid.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "spread": 0.0}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "spread": float(valid.max() - valid.min()),
    }


def evaluate(sources, points, settings: PredictionSettings, mode: str, metric: str, freq: float = 1000.0,
             want_columns: bool = False, env=None):
    """Metric values (N,) and, if requested, a dict of all result columns.

    mode: 'OCTAVE' (incoherent) or 'COHERENT' (phase-aware 1/3-octave).
    metric: 'A', 'C', 'BROADBAND' or 'BAND' (then `freq` selects the band/frequency).
    """
    if mode == "OCTAVE":
        res = predict(sources, points, settings, env)
        values = res.metric(str(int(freq)) if metric == "BAND" else metric)
        cols = None
        if want_columns:
            cols = {"A": res.a_weighted, "C": res.c_weighted, "BB": res.broadband}
            cols.update({f"{b}Hz": res.band_spl[:, i] for i, b in enumerate(BANDS)})
        return values, cols
    if metric == "BAND" and not want_columns:
        return predict_coherent(sources, points, [freq], settings, env).spl[:, 0], None
    res = predict_coherent(sources, points, THIRD_OCTAVES, settings, env)
    values = res.metric(f"{freq:g}" if metric == "BAND" else metric)
    cols = None
    if want_columns:
        cols = {"A": res.metric("A"), "C": res.metric("C"), "BB": res.metric("BROADBAND")}
        cols.update({f"{f:g}Hz": res.spl[:, i] for i, f in enumerate(res.freqs)})
    return values, cols

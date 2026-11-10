"""Room acoustics: reverberation time and a speech-intelligibility estimate. No Blender imports.

RT60 uses Sabine and Eyring with the surface absorption and air absorption. The STI value is an ENGINEERING ESTIMATE
built from the IEC 60268-16 modulation-transfer approach (direct arrivals + exponential reverberant tail + noise). It
omits auditory masking and the absolute threshold, assumes a flat signal spectrum and a diffuse tail, and is not an
IEC-compliant measurement or prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustics import (
    A_WEIGHTING_DB, MIN_DIRECTIVITY_DB, AcousticModel, PredictionSettings, _chunks, _levels,
)
from .bands import BANDS_ARR, NUM_BANDS
from .environment import as_environment

STI_BAND_INDEX = np.arange(1, NUM_BANDS)          # 125 Hz-8 kHz
MOD_FREQS = 0.63 * 2.0 ** (np.arange(14) / 3.0)   # 0.63-12.5 Hz
STI_ALPHA = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])   # male weighting, IEC 60268-16
STI_BETA = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
STI_SNR_LIMIT_DB = 15.0
DEFAULT_VOLUME_FILL = 0.85


@dataclass
class RoomAcoustics:
    volume_m3: float
    surface_m2: float
    absorption_m2: np.ndarray      # (8,) equivalent absorption area incl. air
    mean_alpha: np.ndarray         # (8,)
    rt_sabine: np.ndarray          # (8,) seconds
    rt_eyring: np.ndarray          # (8,) seconds
    room_constant_m2: np.ndarray   # (8,)


def room_acoustics(surfaces, volume_m3: float, settings: PredictionSettings | None = None) -> RoomAcoustics:
    """surfaces: iterable of (area_m2, absorption (8,))."""
    settings = settings or PredictionSettings()
    surfaces = list(surfaces)
    if volume_m3 <= 0 or not surfaces:
        raise ValueError("Room needs a positive volume and at least one surface.")
    area = np.array([a for a, _ in surfaces], dtype=float)
    alpha = np.array([al for _, al in surfaces], dtype=float)
    total = float(area.sum())
    a_surf = (area[:, None] * alpha).sum(axis=0)
    mean_alpha = np.clip(a_surf / total, 1e-4, 0.99)
    m_air = settings.absorption(BANDS_ARR) / (10.0 * np.log10(np.e))   # nepers per metre
    a_air = 4.0 * m_air * volume_m3
    absorption = a_surf + a_air
    rt_sabine = 0.161 * volume_m3 / np.maximum(absorption, 1e-6)
    rt_eyring = 0.161 * volume_m3 / np.maximum(-total * np.log(1.0 - mean_alpha) + a_air, 1e-6)
    return RoomAcoustics(volume_m3, total, absorption, mean_alpha, rt_sabine, rt_eyring,
                         absorption / (1.0 - mean_alpha))


def directivity_index(model: AcousticModel, samples: int = 1500) -> np.ndarray:
    """DI per octave band (dB) of the separable pattern, by averaging over a Fibonacci sphere."""
    i = np.arange(samples) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / samples)
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * i
    x, y, z = np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)
    d = np.maximum(model.horizontal.at(np.degrees(np.arctan2(x, z)))
                   + model.vertical.at(np.degrees(np.arctan2(y, z))), MIN_DIRECTIVITY_DB)
    return -10.0 * np.log10(np.mean(10.0 ** (d / 10.0), axis=0))


def noise_band_levels(noise_dba: float) -> np.ndarray:
    """Octave-band levels of pink (equal energy per octave) noise that sums to `noise_dba` dB(A)."""
    c = noise_dba - 10.0 * np.log10(np.sum(10.0 ** (A_WEIGHTING_DB / 10.0)))
    return np.full(NUM_BANDS, c)


def _mti_to_sti(mti: np.ndarray) -> np.ndarray:
    """mti (N, 7) -> STI (N,)."""
    sti = mti @ STI_ALPHA - np.sqrt(np.maximum(mti[:, :-1] * mti[:, 1:], 0.0)) @ STI_BETA
    return np.clip(sti, 0.0, 1.0)


def predict_sti(sources, points, room: RoomAcoustics, noise_dba: float = 35.0,
                settings: PredictionSettings | None = None, env=None) -> np.ndarray:
    """STI estimate (N,) with the direct sound of each source and a diffuse reverberant tail."""
    settings = settings or PredictionSettings()
    env = as_environment(env)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    active = [s for s in sources if s.enabled]
    out = np.zeros(len(pts))
    if not active:
        return out
    shadow = env.shadow_db(BANDS_ARR) if env.occluder is not None else None
    c = settings.c

    # Late diffuse field: W * 4 / R, with the power level from the on-axis level and the directivity index.
    tail = np.zeros(NUM_BANDS)
    for s in active:
        on_axis = s.model.max_spl_1m_db + s.level_db + s.model.response_db
        lw = on_axis - directivity_index(s.model) + 11.0
        tail += 10.0 ** ((lw + 10.0 * np.log10(4.0 / room.room_constant_m2)) / 10.0)
    noise = 10.0 ** (noise_band_levels(noise_dba) / 10.0)
    x = 2.0 * np.pi * MOD_FREQS[None, :] * room.rt_sabine[:, None] / 13.8   # (8, 14)

    for sl in _chunks(pts):
        chunk = pts[sl]
        n = len(chunk)
        direct = np.zeros((n, NUM_BANDS))
        mod = np.zeros((n, NUM_BANDS, len(MOD_FREQS)), dtype=complex)
        t_first = np.full(n, np.inf)
        for s in active:
            levels, dist = _levels(s, chunk, BANDS_ARR, settings)
            if shadow is not None:
                levels = levels - env.blocked(s.position[None, :], chunk)[:, None] * shadow[None, :]
            e = 10.0 ** (levels / 10.0)
            t = dist / c + s.delay_ms / 1000.0
            direct += e
            mod += e[:, :, None] * np.exp(-2j * np.pi * MOD_FREQS[None, None, :] * t[:, None, None])
            t_first = np.minimum(t_first, t)
        tail_term = tail[None, :, None] / (1.0 + 1j * x[None, :, :])
        tail_term = tail_term * np.exp(-2j * np.pi * MOD_FREQS[None, None, :] * t_first[:, None, None])
        total_e = direct + tail[None, :]
        m = np.abs(mod + tail_term) / total_e[:, :, None]
        snr = 10.0 * np.log10(total_e / noise[None, :])
        m = m / (1.0 + 10.0 ** (-snr[:, :, None] / 10.0))
        m = np.clip(m, 1e-6, 1.0 - 1e-6)
        eff = np.clip(10.0 * np.log10(m / (1.0 - m)), -STI_SNR_LIMIT_DB, STI_SNR_LIMIT_DB)
        mti = ((eff + STI_SNR_LIMIT_DB) / (2.0 * STI_SNR_LIMIT_DB)).mean(axis=2)
        out[sl] = _mti_to_sti(mti[:, STI_BAND_INDEX])
    return out

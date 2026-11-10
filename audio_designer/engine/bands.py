"""Octave-band constants and log-frequency interpolation. No Blender imports."""
from __future__ import annotations

import numpy as np

BANDS = (63, 125, 250, 500, 1000, 2000, 4000, 8000)
NUM_BANDS = len(BANDS)
BANDS_ARR = np.array(BANDS, dtype=float)
_LOG_BANDS = np.log10(BANDS_ARR)


def _band_interp(freqs):
    """Index/weight triple to interpolate octave-band data to arbitrary frequencies (log-f, clamped)."""
    lf = np.log10(np.clip(np.asarray(freqs, dtype=float), BANDS_ARR[0], BANDS_ARR[-1]))
    i1 = np.clip(np.searchsorted(_LOG_BANDS, lf, side="right"), 1, NUM_BANDS - 1)
    i0 = i1 - 1
    w = (lf - _LOG_BANDS[i0]) / (_LOG_BANDS[i1] - _LOG_BANDS[i0])
    return i0, i1, w


def to_freqs(band_values: np.ndarray, freqs) -> np.ndarray:
    """Interpolate (..., NUM_BANDS) octave data to (..., len(freqs)); clamped outside 63 Hz-8 kHz."""
    i0, i1, w = _band_interp(freqs)
    return band_values[..., i0] * (1.0 - w) + band_values[..., i1] * w

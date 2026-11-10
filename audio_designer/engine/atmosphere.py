"""Atmosphere and frequency helpers: ISO 9613-1 air absorption, speed of sound, weighting curves."""
from __future__ import annotations

import numpy as np

REFERENCE_PRESSURE_KPA = 101.325
_T0_K = 293.15
_T01_K = 273.16

THIRD_OCTAVES = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
    1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000.0])


def _water_vapour_percent(temperature_c: float, humidity_pct: float, pressure_kpa: float) -> float:
    """Molar concentration of water vapour in percent (ISO 9613-1 eq. B.2)."""
    t = temperature_c + 273.15
    c = -6.8346 * (_T01_K / t) ** 1.261 + 4.6151
    return humidity_pct * 10.0 ** c * (REFERENCE_PRESSURE_KPA / pressure_kpa)


def air_absorption_db_per_m(freqs, temperature_c=20.0, humidity_pct=50.0,
                            pressure_kpa=REFERENCE_PRESSURE_KPA) -> np.ndarray:
    """ISO 9613-1 pure-tone atmospheric absorption coefficient in dB/m."""
    f = np.asarray(freqs, dtype=float)
    t = temperature_c + 273.15
    pr = pressure_kpa / REFERENCE_PRESSURE_KPA
    h = _water_vapour_percent(temperature_c, max(humidity_pct, 0.0), pressure_kpa)
    fr_o = pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pr * (t / _T0_K) ** -0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * ((t / _T0_K) ** (-1.0 / 3.0) - 1.0)))
    f2 = f * f
    return 8.686 * f2 * (
        1.84e-11 / pr * (t / _T0_K) ** 0.5
        + (t / _T0_K) ** -2.5 * (
            0.01275 * np.exp(-2239.1 / t) / (fr_o + f2 / fr_o)
            + 0.1068 * np.exp(-3352.0 / t) / (fr_n + f2 / fr_n)))


def speed_of_sound(temperature_c=20.0, humidity_pct=0.0, pressure_kpa=REFERENCE_PRESSURE_KPA) -> float:
    """Ideal-gas mixture of dry air and water vapour."""
    t = temperature_c + 273.15
    x = min(_water_vapour_percent(temperature_c, max(humidity_pct, 0.0), pressure_kpa) / 100.0, 0.5)
    r = 8.314462
    molar_mass = (28.964 * (1.0 - x) + 18.015 * x) / 1000.0
    cv = 20.785 * (1.0 - x) + 25.3 * x
    gamma = (cv + r) / cv
    return float(np.sqrt(gamma * r * t / molar_mass))


def a_weighting_db(freqs) -> np.ndarray:
    f2 = np.asarray(freqs, dtype=float) ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                                      * (f2 + 12194.0 ** 2))
    return 20.0 * np.log10(ra) + 2.0


def c_weighting_db(freqs) -> np.ndarray:
    f2 = np.asarray(freqs, dtype=float) ** 2
    rc = (12194.0 ** 2 * f2) / ((f2 + 20.6 ** 2) * (f2 + 12194.0 ** 2))
    return 20.0 * np.log10(rc) + 0.06

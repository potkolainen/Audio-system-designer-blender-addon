"""Build a loudspeaker definition dict from published datasheet figures. No Blender imports.

Only the figures on the datasheet are facts. The polar shape is modelled from them (constant nominal coverage
above `control_freq_hz`, widening below), so every entry is marked as an approximation.
"""
from __future__ import annotations

import re

import numpy as np

from ..engine import BANDS
from .model import SCHEMA_VERSION

WIDENING_PER_OCTAVE = 1.7
MAX_WIDTH_DEG = 720.0
ROLLOFF_DB_PER_OCTAVE = 12.0
SUB_PATTERNS = ("omni", "cardioid")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unnamed"


def response_from_range(freq_lo_hz: float, freq_hi_hz: float) -> list[float]:
    """0 dB inside the published -5 dB range, then a 12 dB/octave roll-off from -5 dB outside it."""
    out = []
    for f in BANDS:
        if f < freq_lo_hz:
            out.append(-5.0 - ROLLOFF_DB_PER_OCTAVE * float(np.log2(freq_lo_hz / f)))
        elif f > freq_hi_hz:
            out.append(-5.0 - ROLLOFF_DB_PER_OCTAVE * float(np.log2(f / freq_hi_hz)))
        else:
            out.append(0.0)
    return [round(v, 1) for v in out]


def widths_from_nominal(nominal_deg: float, control_freq_hz: float) -> list[float]:
    out = []
    for f in BANDS:
        if f >= control_freq_hz:
            w = nominal_deg
        else:
            w = nominal_deg * WIDENING_PER_OCTAVE ** float(np.log2(control_freq_hz / f))
        out.append(round(min(w, MAX_WIDTH_DEG), 1))
    return out


def build_model_dict(*, manufacturer: str, model: str, category: str, width_m: float, height_m: float,
                     depth_m: float, weight_kg: float, max_spl_db: float, freq_lo_hz: float, freq_hi_hz: float,
                     h_deg: float = 90.0, v_deg: float = 60.0, control_freq_hz: float = 1000.0,
                     v_control_freq_hz: float | None = None,
                     pattern: str = "coverage", source_name: str = "User datasheet entry", source_url: str = "",
                     notes: str = "", side_angle_deg: float = 0.0, max_splay_deg: float = 10.0,
                     max_cabinets: int = 24, model_id: str = "") -> dict:
    """pattern: 'coverage' (nominal h x v), 'omni' (sub, wide until its upper range) or 'cardioid'."""
    if min(width_m, height_m, depth_m, weight_kg) <= 0:
        raise ValueError("Dimensions and weight must be positive.")
    if not 0 < freq_lo_hz < freq_hi_hz:
        raise ValueError("Frequency range needs 0 < low < high.")

    if pattern == "cardioid":
        directivity = {"type": "cardioid"}
    else:
        if pattern == "omni":
            h_w = v_w = widths_from_nominal(200.0, freq_hi_hz)
        else:
            h_w = widths_from_nominal(h_deg, control_freq_hz)
            v_w = widths_from_nominal(v_deg, v_control_freq_hz or control_freq_hz)
        directivity = {"type": "parametric_coverage", "horizontal_deg": h_w, "vertical_deg": v_w}

    data = {
        "schema_version": SCHEMA_VERSION,
        "id": model_id or f"{slug(manufacturer)}.{slug(model)}",
        "manufacturer": manufacturer,
        "model": model,
        "category": category,
        "data_quality": "basic_spec",
        "approximation": True,
        "physical": {"width_m": width_m, "height_m": height_m, "depth_m": depth_m, "weight_kg": weight_kg},
        "acoustic_origin_m": [0.0, round(depth_m / 2.0, 4), 0.0],
        "acoustic": {
            "max_spl_broadband_db": max_spl_db,
            "frequency_response_db": response_from_range(freq_lo_hz, freq_hi_hz),
            "directivity": directivity,
        },
        "source": {"name": source_name, "version": "1", "license": "Published specification figures only",
                   "url": source_url, "notes": notes},
    }
    if side_angle_deg:
        data["physical"]["side_angle_deg"] = side_angle_deg
    if category == "line_array":
        data["array"] = {"max_splay_deg": max_splay_deg, "max_cabinets": max_cabinets}
    return data

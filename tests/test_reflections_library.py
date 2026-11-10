import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_designer.engine import (  # noqa: E402
    BANDS, PredictionSettings, Reflector, Source, predict, predict_coherent, probe_response,
)
from audio_designer.library import BUILTIN_DATA_DIR, Library, LibraryError, parse_model, parse_surface  # noqa: E402
from audio_designer.library.datasheet import build_model_dict  # noqa: E402

LIB = Library().scan(BUILTIN_DATA_DIR)
DRY = PredictionSettings(air_absorption=False)


def src(model_id="generic.point_12_90x60", pos=(0, 0, 3.0), forward=(0, 1.0, 0), **kw):
    fwd = np.array(forward, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([1.0, 0, 0])
    up = np.cross(right, fwd)
    return Source(model=LIB.get(model_id).acoustic, position=np.array(pos, float), right=right, up=up,
                  forward=fwd, **kw)


def floor(absorption=0.0, z=0.0, size=1000.0):
    quad = np.array([[-size, -size, z], [size, -size, z], [size, size, z], [-size, size, z]], float)
    a = np.full(8, absorption)
    return [Reflector(quad[[0, 1, 2]], a, 0.0, "floor"), Reflector(quad[[0, 2, 3]], a, 0.0, "floor")]


# --- reflections --------------------------------------------------------------------------------

def test_rigid_floor_adds_energy():
    s = src(forward=(0, 1.0, -0.3))
    p = np.array([[0, 15.0, 1.2]])
    direct = predict([s], p, DRY).band_spl
    with_floor = predict([s], p, DRY, floor(0.0)).band_spl
    assert np.all(with_floor > direct)


def test_fully_absorbing_floor_changes_nothing():
    s = src()
    p = np.array([[0, 15.0, 1.2]])
    np.testing.assert_allclose(predict([s], p, DRY, floor(1.0)).band_spl, predict([s], p, DRY).band_spl, atol=0.01)


def test_floor_bounce_uses_image_source_distance():
    s = src(pos=(0, 0, 3.0), forward=(0, 1.0, 0))
    p = np.array([0, 12.0, 1.2])
    resp = probe_response([s], p, [250.0], DRY, floor(0.0))
    direct_len = np.linalg.norm(p - s.position)
    image_len = np.linalg.norm(p - np.array([0, 0, -3.0]))
    assert resp.reflection_paths == 1
    # two equal-amplitude-ish arrivals: total phase differs from direct-only phase
    no_floor = probe_response([s], p, [250.0], DRY)
    assert resp.total_phase_deg[0] != pytest.approx(no_floor.total_phase_deg[0], abs=0.1)
    assert image_len > direct_len


def test_coherent_floor_comb_matches_analytic_pressure_sum():
    s = src(model_id="generic.sub_18", pos=(0, 0, 2.0), forward=(0, 1.0, 0))
    p = np.array([[0, 20.0, 1.0]])
    f = 63.0
    c = DRY.c
    d1 = np.linalg.norm(p[0] - s.position)
    d2 = np.linalg.norm(p[0] - np.array([0, 0, -2.0]))
    single = predict_coherent([s], p, [f], DRY).spl[0, 0]
    both = predict_coherent([s], p, [f], DRY, floor(0.0)).spl[0, 0]
    a1 = 10 ** (single / 20)
    # image amplitude scales with 1/distance and (mirrored) directivity; allow a loose band around the analytic sum
    a2_est = a1 * d1 / d2
    phase = 2 * np.pi * f * (d2 - d1) / c
    est = 20 * np.log10(abs(a1 + a2_est * np.exp(-1j * phase)))
    assert both == pytest.approx(est, abs=2.5)


def test_point_behind_wall_gets_no_reflection():
    s = src(pos=(0, 0, 2.0), forward=(0, 1.0, 0))
    wall = np.array([[-5, 10, 0], [5, 10, 0], [5, 10, 6]], float)  # wall at y = 10
    refl = [Reflector(wall, np.zeros(8), 0.0, "wall")]
    behind = np.array([[0, 20.0, 1.0]])   # on the far side of the wall from the source
    in_front = np.array([[0, 5.0, 1.0]])
    assert probe_response([s], behind[0], [1000.0], DRY, refl).reflection_paths == 0
    assert probe_response([s], in_front[0], [1000.0], DRY, refl).reflection_paths in (0, 1)


def test_reflection_outside_triangle_is_rejected():
    s = src(pos=(0, 0, 2.0), forward=(0, 1.0, 0))
    small = np.array([[40, -1, 0], [42, -1, 0], [41, 1, 0]], float)  # far from the specular point
    refl = [Reflector(small, np.zeros(8), 0.0, "tile")]
    assert probe_response([s], [0, 10.0, 1.0], [1000.0], DRY, refl).reflection_paths == 0


def test_scattering_reduces_specular_energy():
    s = src(forward=(0, 1.0, -0.3))
    p = np.array([[0, 15.0, 1.2]])
    quad = floor(0.0)
    scattered = [Reflector(r.tri, r.absorption, 0.9, "floor") for r in quad]
    assert predict([s], p, DRY, scattered).broadband[0] < predict([s], p, DRY, quad).broadband[0]


def test_reflector_with_degenerate_triangle_is_ignored():
    s = src()
    tri = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], float)
    out = predict([s], np.array([[0, 5.0, 1.0]]), DRY, [Reflector(tri, np.zeros(8))])
    np.testing.assert_allclose(out.band_spl, predict([s], np.array([[0, 5.0, 1.0]]), DRY).band_spl)


# --- library data -------------------------------------------------------------------------------

def test_manufacturer_models_load_with_provenance():
    lib = Library().scan(BUILTIN_DATA_DIR)
    assert lib.errors == []
    ids = {m.display_name for m in lib.models.values()}
    assert {"d&b audiotechnik E3", "d&b audiotechnik V8", "L-Acoustics KARA II", "d&b audiotechnik Y-SUB"} <= ids
    for m in lib.models.values():
        if m.manufacturer != "Generic":
            assert m.source["name"] and m.source.get("url", "").startswith("https://")
            assert m.approximation and m.data_quality.value == "basic_spec"


def test_e3_published_figures():
    m = next(m for m in Library().scan(BUILTIN_DATA_DIR).models.values() if m.model == "E3")
    assert (m.width_m, m.height_m, m.depth_m, m.weight_kg) == (0.190, 0.352, 0.204, 7.2)
    assert m.acoustic.horizontal.coverage_angle(BANDS.index(1000)) == pytest.approx(45.0, abs=0.5)
    assert m.acoustic.vertical.coverage_angle(BANDS.index(1000)) == pytest.approx(30.0, abs=0.5)


def test_published_broadband_spl_is_reproduced_on_axis():
    lib = Library().scan(BUILTIN_DATA_DIR)
    m = next(m for m in lib.models.values() if m.model == "E3")
    s = Source(model=m.acoustic, position=np.zeros(3), right=np.array([1.0, 0, 0]), up=np.array([0, 0, 1.0]),
               forward=np.array([0, 1.0, 0]), level_db=0.0)
    out = predict([s], np.array([[0, 1.0, 0]]), PredictionSettings(air_absorption=False))
    assert out.broadband[0] == pytest.approx(123.0, abs=0.01)


def test_cardioid_subwoofer_rejects_rear():
    m = next(m for m in Library().scan(BUILTIN_DATA_DIR).models.values() if m.model == "Y-SUB")
    s = Source(model=m.acoustic, position=np.zeros(3), right=np.array([1.0, 0, 0]), up=np.array([0, 0, 1.0]),
               forward=np.array([0, 1.0, 0]))
    front = predict([s], np.array([[0, 10.0, 0]]), DRY).band_spl[0, 0]
    back = predict([s], np.array([[0, -10.0, 0]]), DRY).band_spl[0, 0]
    assert front - back > 25


def test_surfaces_load_and_validate():
    lib = Library().scan(BUILTIN_DATA_DIR)
    assert len(lib.surfaces) >= 8
    carpet, concrete = lib.surface("generic.carpet_heavy"), lib.surface("generic.concrete_unpainted")
    assert carpet.absorption[BANDS.index(2000)] > concrete.absorption[BANDS.index(2000)]
    with pytest.raises(LibraryError):
        parse_surface({"schema_version": 1, "id": "x", "name": "x", "absorption": [1.5] * 8, "source": {"name": "n"}})
    with pytest.raises(LibraryError):
        parse_surface({"schema_version": 1, "id": "x", "name": "x", "absorption": [0.1] * 8})


# --- datasheet builder --------------------------------------------------------------------------

def test_datasheet_builder_roundtrips_through_the_loader():
    data = build_model_dict(manufacturer="Acme", model="Box 12", category="point_source", width_m=0.4, height_m=0.6,
                            depth_m=0.35, weight_kg=20, max_spl_db=130, freq_lo_hz=60, freq_hi_hz=18000,
                            h_deg=90, v_deg=60, control_freq_hz=1000)
    m = parse_model(json.loads(json.dumps(data)))
    assert m.id == "acme.box_12" and m.approximation
    idx = BANDS.index(8000)
    assert m.acoustic.horizontal.coverage_angle(idx) == pytest.approx(45.0, abs=0.5)
    assert m.acoustic.horizontal.coverage_angle(BANDS.index(125)) > 45.0  # wider below the control frequency


def test_datasheet_builder_rejects_bad_input():
    with pytest.raises(ValueError):
        build_model_dict(manufacturer="a", model="b", category="other", width_m=0, height_m=1, depth_m=1,
                         weight_kg=1, max_spl_db=120, freq_lo_hz=50, freq_hi_hz=100)
    with pytest.raises(ValueError):
        build_model_dict(manufacturer="a", model="b", category="other", width_m=1, height_m=1, depth_m=1,
                         weight_kg=1, max_spl_db=120, freq_lo_hz=500, freq_hi_hz=100)

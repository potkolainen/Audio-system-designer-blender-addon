import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_designer.engine import (  # noqa: E402
    BANDS, THIRD_OCTAVES, PredictionSettings, Source, air_absorption_db_per_m, evaluate, predict,
    predict_coherent, probe_response, speed_of_sound,
)
from audio_designer.engine.arrays import (  # noqa: E402
    array_summary, audience_aim_distances, auto_splay, line_array_layout, ring_levels, sub_array_layout,
)
from audio_designer.library import BUILTIN_DATA_DIR, Library  # noqa: E402

LIB = Library().scan(BUILTIN_DATA_DIR)
DRY = PredictionSettings(humidity_pct=50.0, air_absorption=False)


def src(model_id, pos=(0, 0, 0), yaw_deg=0.0, **kw):
    yaw = np.radians(yaw_deg)
    fwd = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    right = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return Source(model=LIB.get(model_id).acoustic, position=np.array(pos, float), right=right,
                  up=np.array([0, 0, 1.0]), forward=fwd, **kw)


# --- atmosphere ---------------------------------------------------------------------------------

def test_iso9613_reference_values_20c_70rh():
    f = np.array([125, 250, 500, 1000, 2000, 4000, 8000.0])
    expected_db_per_km = np.array([0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6])
    np.testing.assert_allclose(air_absorption_db_per_m(f, 20, 70) * 1000, expected_db_per_km, rtol=0.05, atol=0.1)


def test_humidity_changes_high_frequency_loss():
    dry = air_absorption_db_per_m(8000.0, 20, 20)
    mid = air_absorption_db_per_m(8000.0, 20, 60)
    assert dry > 2.0 * mid


def test_speed_of_sound_rises_with_temperature_and_humidity():
    assert speed_of_sound(20, 0) == pytest.approx(343.2, abs=0.1)
    assert speed_of_sound(30, 0) > speed_of_sound(20, 0) + 5
    assert speed_of_sound(20, 80) > speed_of_sound(20, 0)


# --- coherent engine ----------------------------------------------------------------------------

def test_coherent_matches_incoherent_for_single_source():
    s = src("generic.point_12_90x60")
    p = np.array([[0, 10.0, 0]])
    inc = predict([s], p, DRY).band_spl[0]
    coh = predict_coherent([s], p, np.array(BANDS, float), DRY).spl[0]
    np.testing.assert_allclose(coh, inc, atol=1e-6)


def test_two_in_phase_sources_add_6db_coherently():
    a, b = src("generic.sub_18"), src("generic.sub_18")
    p = np.array([[0, 20.0, 0]])
    one = predict_coherent([a], p, [63.0], DRY).spl[0, 0]
    two = predict_coherent([a, b], p, [63.0], DRY).spl[0, 0]
    assert two - one == pytest.approx(6.02, abs=0.01)


def test_inverted_polarity_cancels():
    a, b = src("generic.sub_18"), src("generic.sub_18", polarity=-1)
    out = predict_coherent([a, b], np.array([[0, 20.0, 0]]), [63.0], DRY).spl[0, 0]
    assert out < -50


def test_delay_shifts_phase_by_expected_amount():
    s = src("generic.sub_18", delay_ms=1000.0 / 100.0 / 2)  # half a period at 100 Hz
    ref = src("generic.sub_18")
    p = np.array([0, 20.0, 0])
    r0 = probe_response([ref], p, [100.0], DRY)
    r1 = probe_response([s], p, [100.0], DRY)
    dphi = (r1.total_phase_deg[0] - r0.total_phase_deg[0]) % 360
    assert dphi == pytest.approx(180.0, abs=0.5)  # delay wraps phase negative -> 180 either way


def test_evaluate_modes_return_consistent_shapes():
    s = src("generic.point_12_90x60")
    pts = np.array([[0, 5.0, 0], [0, 10.0, 0]])
    for mode in ("OCTAVE", "COHERENT"):
        v, cols = evaluate([s], pts, DRY, mode, "A", want_columns=True)
        assert v.shape == (2,)
        assert {"A", "C", "BB"} <= set(cols)
        v2, _ = evaluate([s], pts, DRY, mode, "BAND", freq=1000.0)
        assert v2.shape == (2,)
    assert len(cols) == 3 + len(THIRD_OCTAVES)


def test_large_point_sets_are_chunked_correctly():
    s = src("generic.point_12_90x60")
    pts = np.column_stack([np.zeros(45000), np.linspace(2, 50, 45000), np.zeros(45000)])
    full = predict([s], pts, DRY).band_spl
    np.testing.assert_allclose(full[:5], predict([s], pts[:5], DRY).band_spl)
    np.testing.assert_allclose(full[-5:], predict([s], pts[-5:], DRY).band_spl)


# --- line array ---------------------------------------------------------------------------------

def test_line_array_straight_column_geometry():
    lay = line_array_layout(4, 0.3, 0.4, [0, 0, 0], 0.0)
    np.testing.assert_allclose(lay.front_centres[:, 2], [-0.15, -0.45, -0.75, -1.05], atol=1e-9)
    np.testing.assert_allclose(lay.front_centres[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(lay.centres[:, 1], -0.2)


def test_line_array_splay_curves_downwards_and_keeps_hinges_connected():
    lay = line_array_layout(5, 0.3, 0.4, [3, 3, 3, 3], 2.0)
    np.testing.assert_allclose(lay.tilts_deg, [2, 5, 8, 11, 14])
    # consecutive front faces meet: bottom of i == top of i+1
    phi = np.radians(lay.tilts_deg)
    down = np.stack([np.zeros(5), -np.sin(phi), -np.cos(phi)], axis=1)
    bottoms = lay.front_centres + down * 0.15
    tops = lay.front_centres - down * 0.15
    np.testing.assert_allclose(bottoms[:-1], tops[1:], atol=1e-9)


def test_auto_splay_aims_top_far_bottom_near():
    top, splays = auto_splay(8, 0.3, 0.45, hang_height_m=8.0, near_m=10.0, far_m=40.0, max_splay_deg=10.0)
    lay = line_array_layout(8, 0.3, 0.45, splays, top)
    dist = audience_aim_distances(lay, 8.0)
    assert np.all(np.diff(dist) < 0)
    assert dist[0] == pytest.approx(40.0 - 30.0 / 16, rel=0.15)
    assert dist[-1] == pytest.approx(10.0 + 30.0 / 16, rel=0.25)
    assert np.all(splays >= 0) and np.all(splays <= 10.0)


def test_auto_splay_rejects_bad_input():
    with pytest.raises(ValueError):
        auto_splay(4, 0.3, 0.4, hang_height_m=-1, near_m=5, far_m=20)
    with pytest.raises(ValueError):
        auto_splay(4, 0.3, 0.4, hang_height_m=5, near_m=20, far_m=5)


def test_array_summary():
    lay = line_array_layout(6, 0.3, 0.4, [2] * 5, 0.0)
    s = array_summary(lay, 30.0)
    assert s["total_weight_kg"] == 180.0 and s["cabinets"] == 6 and s["height_m"] > 1.5


# --- subwoofer arrays ---------------------------------------------------------------------------

def _sub_sources(kind, count, spacing, c=343.0):
    out = []
    for p in sub_array_layout(kind, count, spacing, c):
        out.append(src("generic.sub_18", pos=(p.x, p.y, 0), yaw_deg=p.yaw_deg, delay_ms=p.delay_ms, polarity=p.polarity))
    return out


def test_cardioid_recipe_rejects_rear_at_design_frequency():
    c = PredictionSettings(humidity_pct=50.0, air_absorption=False).c
    spacing = c / (4 * 63.0)  # quarter wavelength at ~63 Hz
    subs = _sub_sources("CARDIOID", 1, spacing, c)
    ring = ring_levels(subs, (0, 0, 0), 15.0, [63.0], DRY, [0, 180], (0, 1))
    assert ring[0, 0] - ring[1, 0] > 10.0


def test_endfire_is_louder_in_front_than_behind():
    c = DRY.c
    subs = _sub_sources("ENDFIRE", 4, c / (4 * 80.0), c)
    ring = ring_levels(subs, (0, 0, 0), 20.0, [80.0], DRY, [0, 180], (0, 1))
    assert ring[0, 0] - ring[1, 0] > 10.0


def test_broadside_has_no_delay_and_is_symmetric():
    recipe = sub_array_layout("BROADSIDE", 4, 0.8)
    assert all(r.delay_ms == 0 and r.polarity == 1 for r in recipe)
    assert sum(r.x for r in recipe) == pytest.approx(0.0)


def test_sub_recipe_validation():
    with pytest.raises(ValueError):
        sub_array_layout("NOPE", 2, 1.0)
    with pytest.raises(ValueError):
        sub_array_layout("ENDFIRE", 0, 1.0)

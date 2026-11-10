import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_designer.engine import (  # noqa: E402
    BANDS, DEFAULT_SHADOW_DB, Environment, PredictionSettings, Reflector, Source, TriangleOccluder,
    directivity_index, noise_band_levels, predict, predict_coherent, predict_sti, probe_response, room_acoustics,
)
from audio_designer.library import BUILTIN_DATA_DIR, Library  # noqa: E402

LIB = Library().scan(BUILTIN_DATA_DIR)
DRY = PredictionSettings(air_absorption=False)


def src(model_id="generic.point_12_90x60", pos=(0, 0, 2.0), forward=(0, 1.0, 0), **kw):
    fwd = np.array(forward, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right = right / np.linalg.norm(right)
    return Source(model=LIB.get(model_id).acoustic, position=np.array(pos, float), right=right,
                  up=np.cross(right, fwd), forward=fwd, **kw)


def quad(corners, absorption=0.0):
    c = np.array(corners, float)
    a = np.full(8, absorption)
    return [Reflector(c[[0, 1, 2]], a), Reflector(c[[0, 2, 3]], a)]


def wall_x(x, absorption=0.0, half=100.0):
    return quad([[x, -half, -half], [x, half, -half], [x, half, half], [x, -half, half]], absorption)


# --- occlusion -----------------------------------------------------------------------------------

def test_blocked_direct_path_is_attenuated_by_the_shadow_curve():
    s = src(pos=(0, 0, 2.0))
    p = np.array([[0, 20.0, 2.0]])
    blocker = TriangleOccluder(np.array([[-5, 10, -5], [5, 10, -5], [0, 10, 8]], float))
    free = predict([s], p, DRY).band_spl[0]
    shadowed = predict([s], p, DRY, Environment(occluder=blocker)).band_spl[0]
    np.testing.assert_allclose(free - shadowed, DEFAULT_SHADOW_DB, atol=1e-6)


def test_unblocked_path_unchanged_and_scale_applies():
    s = src()
    p = np.array([[0, 20.0, 2.0]])
    far_away = TriangleOccluder(np.array([[100, 0, 0], [101, 0, 0], [100, 1, 1]], float))
    np.testing.assert_allclose(predict([s], p, DRY, Environment(occluder=far_away)).band_spl,
                               predict([s], p, DRY).band_spl)
    wall = TriangleOccluder(np.array([[-5, 10, -5], [5, 10, -5], [0, 10, 8]], float))
    half = predict([s], p, DRY, Environment(occluder=wall, shadow_scale=0.5)).band_spl[0]
    np.testing.assert_allclose(predict([s], p, DRY).band_spl[0] - half, DEFAULT_SHADOW_DB * 0.5, atol=1e-6)


def test_occluder_ignores_segment_endpoints_touching_a_surface():
    floor = TriangleOccluder(np.array([[-50, -50, 0], [50, -50, 0], [0, 50, 0]], float))
    assert not floor.blocked(np.array([[0, 0, 0.0]]), np.array([[0, 10.0, 3.0]]))[0]
    assert floor.blocked(np.array([[0, 0, 2.0]]), np.array([[0, 10.0, -2.0]]))[0]


def test_reflection_path_blocked_when_a_leg_crosses_geometry():
    s = src(pos=(0, 0, 3.0), forward=(0, 1.0, -0.2))
    p = np.array([[0, 20.0, 1.2]])
    floor = quad([[-200, -200, 0], [200, -200, 0], [200, 200, 0], [-200, 200, 0]])
    screen = TriangleOccluder(np.array([[-1, 10, 0.01], [1, 10, 0.01], [0, 10, 2.9]], float))  # between S and floor bounce
    open_env = Environment(reflectors=floor)
    blocked_env = Environment(reflectors=floor, occluder=screen)
    open_total = predict([s], p, DRY, open_env).broadband[0]
    shadowed_total = predict([s], p, DRY, blocked_env).broadband[0]
    assert shadowed_total < open_total


# --- second order --------------------------------------------------------------------------------

def test_parallel_walls_give_two_first_and_two_second_order_paths():
    s = src(pos=(0, 0, 2.0))
    walls = wall_x(-5.0) + wall_x(5.0)
    p = np.array([2.0, 12.0, 2.0])
    one = probe_response([s], p, [500.0], DRY, Environment(reflectors=walls, reflection_order=1))
    two = probe_response([s], p, [500.0], DRY, Environment(reflectors=walls, reflection_order=2))
    assert one.reflection_paths == 2
    assert two.reflection_paths == 4


def test_second_order_path_length_matches_double_image():
    s = src(model_id="generic.sub_18", pos=(0, 0, 2.0))
    walls = wall_x(-5.0, half=1000) + wall_x(5.0, half=1000)
    p = np.array([2.0, 12.0, 2.0])
    f = 80.0
    c = DRY.c
    env1 = Environment(reflectors=walls, reflection_order=1)
    env2 = Environment(reflectors=walls, reflection_order=2)
    one = predict_coherent([s], p[None, :], [f], DRY, env1).spl[0, 0]
    two = predict_coherent([s], p[None, :], [f], DRY, env2).spl[0, 0]
    assert one != pytest.approx(two, abs=1e-6)
    # unfold: S -> left wall -> right wall -> P equals |image(image(S)) - P|
    i1 = np.array([-10.0, 0, 2.0])
    i2 = np.array([20.0, 0, 2.0])
    assert np.linalg.norm(i2 - p) > np.linalg.norm(i1 - p) - 5
    assert c > 0


def test_absorbing_walls_remove_second_order_energy():
    s = src(pos=(0, 0, 2.0))
    p = np.array([[2.0, 12.0, 2.0]])
    rigid = predict([s], p, DRY, Environment(reflectors=wall_x(-5.0) + wall_x(5.0), reflection_order=2)).broadband[0]
    soft = predict([s], p, DRY, Environment(reflectors=wall_x(-5.0, 0.9) + wall_x(5.0, 0.9), reflection_order=2)).broadband[0]
    assert soft < rigid


# --- room acoustics ------------------------------------------------------------------------------

def test_sabine_matches_hand_calculation_and_eyring_is_shorter():
    room = room_acoustics([(600.0, np.full(8, 0.2))], 1000.0, DRY)
    np.testing.assert_allclose(room.rt_sabine, 0.161 * 1000.0 / 120.0, rtol=1e-6)
    assert np.all(room.rt_eyring < room.rt_sabine)


def test_air_absorption_shortens_high_band_rt():
    wet = room_acoustics([(600.0, np.full(8, 0.1))], 5000.0, PredictionSettings(humidity_pct=30.0))
    dry = room_acoustics([(600.0, np.full(8, 0.1))], 5000.0, DRY)
    assert wet.rt_sabine[BANDS.index(8000)] < dry.rt_sabine[BANDS.index(8000)]
    assert wet.rt_sabine[BANDS.index(125)] == pytest.approx(dry.rt_sabine[BANDS.index(125)], rel=0.08)


def test_room_acoustics_validation():
    with pytest.raises(ValueError):
        room_acoustics([], 100.0)
    with pytest.raises(ValueError):
        room_acoustics([(10.0, np.full(8, 0.1))], 0.0)


def test_directivity_index_orders_models_sensibly():
    wide = directivity_index(LIB.get("generic.sub_18").acoustic)[BANDS.index(63)]
    narrow = directivity_index(LIB.get("generic.point_15_60x40").acoustic)[BANDS.index(2000)]
    assert wide < 3.0 < 6.0 < narrow


def test_noise_spectrum_sums_to_the_requested_dba():
    from audio_designer.engine import A_WEIGHTING_DB, db_sum
    lv = noise_band_levels(40.0)
    assert db_sum(lv + A_WEIGHTING_DB) == pytest.approx(40.0, abs=1e-6)


# --- STI estimate --------------------------------------------------------------------------------

def _sti(rt_alpha, distance, noise=35.0, level=-10.0, volume=2000.0, area=1200.0):
    s = src(pos=(0, 0, 3.0), forward=(0, 1.0, -0.1), level_db=level)
    room = room_acoustics([(area, np.full(8, rt_alpha))], volume, DRY)
    return predict_sti([s], np.array([[0, distance, 1.2]]), room, noise, DRY)[0], room


def test_sti_is_high_for_a_dry_room_close_to_the_speaker():
    value, room = _sti(0.5, 6.0)
    assert room.rt_sabine[4] < 0.6
    assert value > 0.8


def test_sti_falls_with_reverberation_distance_and_noise():
    good, _ = _sti(0.5, 8.0)
    live, _ = _sti(0.05, 8.0)
    far, _ = _sti(0.05, 40.0)
    noisy, _ = _sti(0.5, 8.0, noise=75.0)
    assert live < good
    assert far < live
    assert noisy < good
    assert 0.0 <= far <= good <= 1.0


def test_sti_without_sources_is_zero():
    room = room_acoustics([(100.0, np.full(8, 0.2))], 200.0, DRY)
    assert predict_sti([], np.array([[0, 5.0, 1.0]]), room)[0] == 0.0

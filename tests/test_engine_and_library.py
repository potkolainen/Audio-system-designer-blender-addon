import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_designer.engine import (  # noqa: E402
    BANDS, PredictionSettings, Source, coverage_stats, db_sum, predict, source_contributions,
    speed_of_sound,
)
from audio_designer.library import BUILTIN_DATA_DIR, Library, LibraryError, parse_model  # noqa: E402

RAW = json.loads((BUILTIN_DATA_DIR / "generic" / "point_12_90x60.json").read_text())


def make_source(model_id="generic.point_12_90x60", pos=(0, 0, 0), **kw):
    lib = Library().scan(BUILTIN_DATA_DIR)
    m = lib.get(model_id)
    return Source(model=m.acoustic, position=np.array(pos, float), right=np.array([1.0, 0, 0]),
                  up=np.array([0, 0, 1.0]), forward=np.array([0, 1.0, 0]), **kw)


def test_builtin_library_loads_cleanly():
    lib = Library().scan(BUILTIN_DATA_DIR)
    assert lib.errors == []
    assert len(lib.models) >= 4
    assert all(m.approximation for m in lib.models.values())


def test_inverse_square_law_without_air():
    s = make_source()
    s_settings = PredictionSettings(air_absorption=False)
    near = predict([s], np.array([[0, 2.0, 0]]), s_settings).band_spl[0]
    far = predict([s], np.array([[0, 4.0, 0]]), s_settings).band_spl[0]
    np.testing.assert_allclose(near - far, 6.0206, atol=1e-3)


def test_on_axis_level_matches_model():
    s = make_source(level_db=-10.0)
    out = predict([s], np.array([[0, 1.0, 0]]), PredictionSettings(air_absorption=False))
    expected = s.model.max_spl_1m_db - 10.0 + s.model.response_db
    np.testing.assert_allclose(out.band_spl[0], expected, atol=1e-9)


def test_minus_6db_at_half_coverage_angle():
    s = make_source()
    idx = BANDS.index(1000)
    half = s.model.horizontal.coverage_angle(idx) 
    assert half == pytest.approx(45.0, abs=0.01)
    r = 10.0
    ang = np.radians(half)
    on = predict([s], np.array([[0, r, 0]]), PredictionSettings(air_absorption=False)).band_spl[0, idx]
    off = predict([s], np.array([[r * np.sin(ang), r * np.cos(ang), 0]]),
                  PredictionSettings(air_absorption=False)).band_spl[0, idx]
    assert on - off == pytest.approx(6.0, abs=0.2)


def test_two_identical_sources_add_3db():
    a, b = make_source(), make_source()
    p = np.array([[0, 5.0, 0]])
    one = predict([a], p).band_spl
    two = predict([a, b], p).band_spl
    np.testing.assert_allclose(two - one, 3.0103, atol=1e-6)


def test_disabled_source_ignored_and_no_sources_is_floor():
    s = make_source(enabled=False)
    out = predict([s], np.array([[0, 5.0, 0]]))
    assert np.all(out.band_spl <= -90)


def test_behind_speaker_is_much_quieter():
    s = make_source()
    front = predict([s], np.array([[0, 5.0, 0]])).a_weighted[0]
    back = predict([s], np.array([[0, -5.0, 0]])).a_weighted[0]
    assert front - back > 20


def test_arrival_time_includes_delay_and_speed_of_sound():
    s = make_source(delay_ms=10.0)
    s_set = PredictionSettings(temperature_c=20.0, humidity_pct=0.0)
    _, dist, arrival, _ = source_contributions([s], np.array([[0, 34.3, 0]]), s_set)
    c = speed_of_sound(20.0)
    assert c == pytest.approx(343.2, abs=0.2)
    assert arrival[0, 0] == pytest.approx(34.3 / c * 1000 + 10.0)
    assert dist[0, 0] == pytest.approx(34.3)


def test_air_absorption_reduces_high_bands_only_noticeably():
    s = make_source()
    p = np.array([[0, 50.0, 0]])
    dry = predict([s], p, PredictionSettings(air_absorption=False)).band_spl[0]
    wet = predict([s], p, PredictionSettings(air_absorption=True)).band_spl[0]
    loss = dry - wet
    assert loss[BANDS.index(8000)] > 5.0
    assert loss[BANDS.index(63)] < 0.5


def test_points_inside_one_metre_are_clamped():
    s = make_source()
    p = np.array([[0, 0.1, 0], [0, 1.0, 0]])
    out = predict([s], p, PredictionSettings(air_absorption=False)).band_spl
    np.testing.assert_allclose(out[0], out[1])


def test_db_sum_and_stats():
    assert db_sum(np.array([90.0, 90.0])) == pytest.approx(93.0103, abs=1e-3)
    stats = coverage_stats(np.array([80.0, 90.0, -100.0]))
    assert stats["min"] == 80.0 and stats["max"] == 90.0 and stats["spread"] == 10.0


def test_parse_rejects_bad_data():
    bad = json.loads(json.dumps(RAW))
    bad["acoustic"]["directivity"]["horizontal_deg"] = [90, 90]
    with pytest.raises(LibraryError):
        parse_model(bad)
    bad = json.loads(json.dumps(RAW))
    del bad["source"]
    with pytest.raises(LibraryError):
        parse_model(bad)
    bad = json.loads(json.dumps(RAW))
    bad["schema_version"] = 99
    with pytest.raises(LibraryError):
        parse_model(bad)


def test_visual_only_model_needs_no_acoustic_data():
    data = json.loads(json.dumps(RAW))
    data["data_quality"] = "visual_only"
    del data["acoustic"]
    m = parse_model(data)
    assert m.acoustic is None and not m.data_quality.simulatable


def test_tabulated_directivity_roundtrip():
    data = json.loads(json.dumps(RAW))
    angles = [0, 90, 180]
    row = [0.0, -6.0, -20.0]
    data["acoustic"]["directivity"] = {
        "type": "tabulated", "angles_deg": angles,
        "horizontal_db": {str(b): row for b in BANDS},
        "vertical_db": {str(b): row for b in BANDS},
    }
    m = parse_model(data)
    assert m.acoustic.horizontal.coverage_angle(0) == pytest.approx(90.0)


def test_bad_files_reported_not_fatal(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "good.json").write_text(json.dumps(RAW))
    (tmp_path / "dup.json").write_text(json.dumps(RAW))
    lib = Library().scan(tmp_path)
    assert len(lib.models) == 1
    assert len(lib.errors) == 2

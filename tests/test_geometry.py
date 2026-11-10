import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_designer.engine.geometry import subdivide_triangles, triangle_areas_normals, walkable_islands  # noqa: E402


def quad(p0, p1, p2, p3):
    p = np.array([p0, p1, p2, p3], float)
    return [p[[0, 1, 2]], p[[0, 2, 3]]]


def horizontal(x0, x1, y0, y1, z):
    return quad((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))


def riser(y, z0, z1, x0=0.0, x1=10.0):
    # normal points towards -y (towards the stage), like a real step riser facing the front
    return quad((x0, y, z0), (x0, y, z1), (x1, y, z1), (x1, y, z0))


def stairs(n=4, tread=1.0, rise=0.2, width=10.0):
    tris = []
    for k in range(n):
        tris += horizontal(0, width, k * tread, (k + 1) * tread, k * rise + rise)
        tris += riser(k * tread, k * rise, (k + 1) * rise, 0, width)
    return np.array(tris)


def test_flat_floor_is_one_island_with_the_right_area():
    ids, area = walkable_islands(np.array(horizontal(0, 10, 0, 10, 0)))
    assert set(ids) == {0}
    assert area[0] == pytest.approx(100.0)


def test_steps_become_separate_islands_even_though_risers_connect_them():
    tris = stairs(4)
    ids, area = walkable_islands(tris)
    assert len(area) == 4
    np.testing.assert_allclose(area, 10.0)
    assert (ids >= 0).sum() == 8  # four treads, two triangles each; risers excluded


def test_slope_limit_rejects_steep_ramps_and_accepts_raked_seating():
    ramp = lambda deg: np.array(quad((0, 0, 0), (10, 0, 0), (10, 5 * np.cos(np.radians(deg)), 5 * np.sin(np.radians(deg))),
                                     (0, 5 * np.cos(np.radians(deg)), 5 * np.sin(np.radians(deg)))))  # noqa: E731
    assert len(walkable_islands(ramp(30), 45)[1]) == 1
    assert len(walkable_islands(ramp(60), 45)[1]) == 0


def test_downward_faces_are_not_walkable():
    ids, area = walkable_islands(np.array(horizontal(0, 10, 0, 10, 3.0))[:, ::-1])
    assert len(area) == 0 and np.all(ids == -1)


def test_small_details_form_small_islands_that_can_be_filtered():
    big = np.array(horizontal(0, 10, 0, 10, 0))
    chair_top = np.array(horizontal(2, 2.4, 2, 2.4, 0.9))
    ids, area = walkable_islands(np.concatenate([big, chair_top]))
    assert sorted(np.round(area, 2)) == [0.16, 100.0]


def test_separate_but_touching_meshes_are_welded_into_one_island():
    a = np.array(horizontal(0, 5, 0, 5, 0))
    b = np.array(horizontal(5, 10, 0, 5, 0))
    assert len(walkable_islands(np.concatenate([a, b]))[1]) == 1


def test_subdivision_preserves_area_and_respects_cell_size():
    tris = np.array(horizontal(0, 10, 0, 6, 0))
    fine = subdivide_triangles(tris, 0.5)
    assert triangle_areas_normals(fine)[0].sum() == pytest.approx(60.0)
    edges = np.linalg.norm(fine[:, 1] - fine[:, 0], axis=1)
    assert edges.max() <= 0.5 * 1.5
    assert len(fine) >= 60 / (0.5 * 0.5 / 2) * 0.5


def test_empty_inputs():
    assert len(walkable_islands(np.empty((0, 3, 3)))[1]) == 0
    assert subdivide_triangles(np.empty((0, 3, 3)), 0.5).shape == (0, 3, 3)

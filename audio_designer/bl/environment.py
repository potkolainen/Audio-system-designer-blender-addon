"""Scene geometry for the engine: surfaces, reflectors, occluders and the room."""
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..engine import Environment, Reflector, room_acoustics
from . import scene_io

MAX_REFLECTOR_TRIANGLES = 6000
GROUND_EXTENT_M = 10000.0
LIVE_REFLECTORS = 64
SEGMENT_NUDGE_M = 0.02

FLOOR_FALLBACK = "generic.wood_floor_parquet"
CEILING_FALLBACK = "generic.plasterboard"
WALL_FALLBACK = "generic.concrete_painted"

# First match wins. Matched against Blender material names, then object names.
KEYWORDS = (
    (("glass", "window", "glazing"), "generic.glass"),
    (("curtain", "drape", "velour", "fabric", "cloth", "banner"), "generic.curtain_heavy"),
    (("carpet", "rug"), "generic.carpet_heavy"),
    (("audience", "seat", "chair", "crowd", "people"), "generic.audience_seated"),
    (("stage",), "generic.wood_stage_floor"),
    (("parquet", "wood floor", "timber floor"), "generic.wood_floor_parquet"),
    (("panel", "wood", "timber", "plywood"), "generic.wood_panelling"),
    (("plaster", "gypsum", "drywall", "ceiling"), "generic.plasterboard"),
    (("brick",), "generic.brick"),
    (("concrete", "cement", "screed"), "generic.concrete_unpainted"),
)

_cache = {"key": None, "env": None, "live": None, "notes": []}
_revision = [0]


def bump_revision():
    _revision[0] += 1


def on_depsgraph_update(scene, depsgraph):
    """Drop cached geometry when something that shapes the environment changes."""
    for u in depsgraph.updates:
        idb = u.id
        if isinstance(idb, bpy.types.Object):
            if idb.ad.kind == "NONE":
                bump_revision()
                return
        elif isinstance(idb, (bpy.types.Mesh, bpy.types.Material)):
            bump_revision()
            return


def has_surface(ob) -> bool:
    return ob.type == "MESH" and ob.ad.kind == "NONE" and bool(ob.ad.surface_id or ob.ad.surface_custom or ob.ad.surface_auto)


def is_visible(ob) -> bool:
    return not ob.hide_viewport and not ob.hide_get()


def keyword_surface(*names):
    for name in names:
        low = (name or "").lower()
        for words, surface_id in KEYWORDS:
            if any(w in low for w in words):
                return surface_id
    return None


def fallback_surface(normal_z: float) -> str:
    if normal_z >= 0.7:
        return FLOOR_FALLBACK
    if normal_z <= -0.7:
        return CEILING_FALLBACK
    return WALL_FALLBACK


def object_triangles(ob):
    """(world triangles (T, 3, 3), source polygon index per triangle (T,))."""
    me = ob.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    mw = np.array(ob.matrix_world)
    co = co.reshape(nv, 3) @ mw[:3, :3].T + mw[:3, 3]
    idx, poly = [], []
    for pi, p in enumerate(me.polygons):
        v = p.vertices
        for i in range(1, len(v) - 1):
            idx.append((v[0], v[i], v[i + 1]))
            poly.append(pi)
    if not idx:
        return np.empty((0, 3, 3)), np.empty(0, dtype=int)
    return co[np.array(idx, dtype=int)], np.array(poly, dtype=int)


def tri_areas_normals(tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(n, axis=1)
    return 0.5 * length, n / np.maximum(length, 1e-12)[:, None]


def _surface_for(library, surface_id):
    return library.surface(surface_id) or library.surface(WALL_FALLBACK)


def _object_reflectors(ob, library, notes, min_area):
    """Reflectors of one mesh object. Auto objects pick a surface per face from material names and orientation."""
    tris, poly = object_triangles(ob)
    if not len(tris):
        return []
    areas, normals = tri_areas_normals(tris)
    keep = areas >= max(min_area, 1e-9)
    ad = ob.ad
    out = []
    if ad.surface_custom or (ad.surface_id and not ad.surface_auto):
        if ad.surface_custom:
            absorption, scattering = np.array(ad.surface_absorption, dtype=float), float(ad.surface_scattering)
        else:
            material = library.surface(ad.surface_id)
            if material is None:
                notes.append(f"{ob.name}: unknown surface '{ad.surface_id}'")
                return []
            absorption, scattering = material.absorption, material.scattering
        return [Reflector(t, absorption, scattering, ob.name) for t, k in zip(tris, keep) if k]

    slot_names = [s.material.name if s.material else "" for s in ob.material_slots]
    me = ob.data
    npoly = len(me.polygons)
    mat_index = np.empty(npoly, dtype=np.int32)
    me.polygons.foreach_get("material_index", mat_index)
    local_normals = np.empty(npoly * 3, dtype=np.float32)
    me.polygons.foreach_get("normal", local_normals)
    pnormals = local_normals.reshape(npoly, 3) @ np.array(ob.matrix_world.to_3x3()).T
    pnz = pnormals[:, 2] / np.maximum(np.linalg.norm(pnormals, axis=1), 1e-9)
    cache = {}
    for t, k, pi in zip(tris, keep, poly):
        if not k:
            continue
        mi = int(mat_index[pi])
        name = slot_names[mi] if mi < len(slot_names) else ""
        nz = float(pnz[pi])
        key = (name, 0 if nz >= 0.7 else 1 if nz <= -0.7 else 2)
        if key not in cache:
            sid = keyword_surface(name, ob.name) or fallback_surface(nz)
            cache[key] = _surface_for(library, sid)
        material = cache[key]
        if material is not None:
            out.append(Reflector(t, material.absorption, material.scattering, ob.name))
    return out


def collect_reflectors(scene, library, include_ground=True, limit=MAX_REFLECTOR_TRIANGLES, min_area=0.0):
    """Return (reflectors, notes) for every mesh with a surface, plus the optional ground plane."""
    st = scene.ad_settings
    reflectors, notes = [], []
    ground = []
    if include_ground and st.ground_enabled:
        material = library.surface(st.ground_surface_id)
        if material is not None:
            e, z = GROUND_EXTENT_M, st.ground_z
            quad = np.array([[-e, -e, z], [e, -e, z], [e, e, z], [-e, e, z]], dtype=float)
            ground = [Reflector(quad[[0, 1, 2]], material.absorption, material.scattering, "Ground"),
                      Reflector(quad[[0, 2, 3]], material.absorption, material.scattering, "Ground")]
    for ob in scene.objects:
        if has_surface(ob):
            reflectors.extend(_object_reflectors(ob, library, notes, min_area))
    if limit is not None and len(reflectors) + len(ground) > limit:
        notes.append(f"{len(reflectors)} reflector triangles; the {limit} largest are used")
        reflectors.sort(key=lambda r: -np.linalg.norm(np.cross(r.tri[1] - r.tri[0], r.tri[2] - r.tri[0])))
        reflectors = reflectors[: max(limit - len(ground), 0)]
    return ground + reflectors, notes


class BVHOccluder:
    def __init__(self, tree):
        self.tree = tree

    def blocked(self, a, b):
        b = np.atleast_2d(np.asarray(b, dtype=float))
        a = np.broadcast_to(np.asarray(a, dtype=float), b.shape)
        d = b - a
        length = np.linalg.norm(d, axis=1)
        unit = d / np.maximum(length, 1e-9)[:, None]
        out = np.zeros(len(b), dtype=bool)
        cast = self.tree.ray_cast
        for i in np.nonzero(length > 4 * SEGMENT_NUDGE_M)[0]:
            hit = cast(Vector(a[i] + unit[i] * SEGMENT_NUDGE_M), Vector(unit[i]), float(length[i] - 2 * SEGMENT_NUDGE_M))
            out[i] = hit[0] is not None
        return out


def build_occluder(scene):
    st = scene.ad_settings
    verts, polys, offset = [], [], 0
    for ob in scene.objects:
        if ob.type != "MESH" or ob.ad.kind != "NONE" or not is_visible(ob):
            continue
        if st.occluder_source == "SURFACES" and not has_surface(ob):
            continue
        me = ob.data
        nv = len(me.vertices)
        co = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        mw = np.array(ob.matrix_world)
        world = co.reshape(nv, 3) @ mw[:3, :3].T + mw[:3, 3]
        verts.extend(map(tuple, world.tolist()))
        polys.extend(tuple(int(i) + offset for i in p.vertices) for p in me.polygons)
        offset += nv
    if not polys:
        return None
    return BVHOccluder(BVHTree.FromPolygons(verts, polys, all_triangles=False, epsilon=0.0))


def _settings_key(st):
    return (st.use_reflections, st.ground_enabled, st.ground_z, st.ground_surface_id, st.use_occlusion,
            st.shadow_scale, st.reflection_order, st.order2_limit, st.occluder_source, st.reflector_min_area_m2)


def active_environment(scene, library, live=False):
    """(Environment, notes) for the scene's reflection and occlusion settings, cached until the geometry changes.

    live=True returns a cheaper first-order version for per-redraw feedback such as the probe panel.
    """
    st = scene.ad_settings
    key = (scene.name, _revision[0], _settings_key(st))
    if _cache["key"] != key:
        reflectors, notes = ([], [])
        if st.use_reflections:
            reflectors, notes = collect_reflectors(scene, library, True, MAX_REFLECTOR_TRIANGLES, st.reflector_min_area_m2)
        occluder = build_occluder(scene) if st.use_occlusion else None
        if st.use_occlusion and occluder is None:
            notes.append("Occlusion is on but no mesh qualifies as an occluder")
        env = Environment(reflectors, occluder, st.shadow_scale, int(st.reflection_order), st.order2_limit)
        _cache.update(key=key, env=env, live=env.limited(LIVE_REFLECTORS, 1), notes=notes)
    return (_cache["live"] if live else _cache["env"]), _cache["notes"]


def room_for_scene(scene, library):
    """(RoomAcoustics or None, message). Uses every surface-assigned mesh; volume from the override or the bounding box."""
    st = scene.ad_settings
    reflectors, _ = collect_reflectors(scene, library, include_ground=False, limit=None, min_area=0.0)
    if not reflectors:
        return None, "Assign surfaces to the room meshes first (Surfaces panel)."
    tris = np.array([r.tri for r in reflectors], dtype=float)
    areas, _n = tri_areas_normals(tris)
    lo, hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    box = float(np.prod(np.maximum(hi - lo, 0.1)))
    volume = st.room_volume_m3 if st.room_volume_m3 > 0 else box * st.room_fill
    room = room_acoustics(((float(a), r.absorption) for a, r in zip(areas, reflectors)), volume,
                          scene_io.prediction_settings(st))
    source = "override" if st.room_volume_m3 > 0 else f"bounding box x {st.room_fill:g}"
    return room, f"{volume:.0f} m3 ({source}), {room.surface_m2:.0f} m2 of surface"


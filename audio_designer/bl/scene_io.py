"""Blender scene <-> engine conversion, speaker geometry and heatmap painting."""
import math

import bpy
import numpy as np
from mathutils import Vector

from ..engine import PredictionSettings, Source
from ..engine.render import colormap  # noqa: F401

AD_COLOR_ATTR = "AD_Color"
HEATMAP_MATERIAL = "AD_Heatmap"


def prediction_settings(st):
    return PredictionSettings(st.temperature_c, st.humidity_pct, st.pressure_kpa, st.air_absorption)


def selected_freq_hz(st):
    return float(st.band if st.mode == "OCTAVE" else st.third)


def metric_label(st):
    if st.metric == "BAND":
        return f"{selected_freq_hz(st):g} Hz"
    return {"A": "dB(A)", "C": "dB(C)", "BROADBAND": "Broadband", "STI": "STI (estimate)"}[st.metric]


def horizontal_forward(ob):
    """World-space horizontal direction of the object's local +Y axis (x, y)."""
    v = ob.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    return np.array([v.x, v.y])


def collect_sources(scene, library, parent=None):
    """Return (engine sources, names of speakers skipped because they have no acoustic data).

    If `parent` is given, only speakers parented directly to that object are included.
    """
    sources, skipped = [], []
    for ob in scene.objects:
        if ob.ad.kind != "SPEAKER" or (parent is not None and ob.parent != parent):
            continue
        model = library.get(ob.ad.model_id)
        if model is None or model.acoustic is None:
            skipped.append(ob.name)
            continue
        mw = ob.matrix_world
        q = mw.to_quaternion()
        sources.append(Source(
            model=model.acoustic,
            position=np.array(mw.translation),
            right=np.array(q @ Vector((1, 0, 0))),
            up=np.array(q @ Vector((0, 0, 1))),
            forward=np.array(q @ Vector((0, 1, 0))),
            level_db=ob.ad.level_db,
            delay_ms=ob.ad.delay_ms,
            polarity=-1 if ob.ad.polarity_inverted else 1,
            enabled=not ob.ad.mute,
            name=ob.name,
        ))
    return sources, skipped


def audience_yaw(scene, origin):
    """Z rotation (radians) that makes an object's local +Y front face the audience direction setting."""
    d = scene.ad_settings.audience_dir
    fixed = {"+Y": 0.0, "-Y": math.pi, "+X": -math.pi / 2.0, "-X": math.pi / 2.0}
    if d in fixed:
        return fixed[d]
    receivers = [o.matrix_world.translation for o in scene.objects if o.ad.kind == "RECEIVER"]
    if not receivers:
        return 0.0
    centre = sum(receivers, Vector((0.0, 0.0, 0.0))) / len(receivers)
    delta = centre - Vector(origin)
    return math.atan2(-delta.x, delta.y) if (delta.x or delta.y) else 0.0


def _material(name, color):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = color
    return mat


def speaker_mesh(model):
    """Shared box mesh, front (+Y) face in a second material. The object origin is the acoustic origin."""
    name = "ADM_" + model.id
    me = bpy.data.meshes.get(name)
    if me is not None:
        return me
    x, y, z = model.width_m / 2, model.depth_m / 2, model.height_m / 2
    # Side walls taper towards the back so that line-array cabinets can fold at their maximum splay.
    zr = max(z - y * np.tan(np.radians(model.side_angle_deg) / 2.0), z * 0.2)
    ox, oy, oz = model.acoustic_origin_m
    verts = [(-x - ox, -y - oy, -zr - oz), (x - ox, -y - oy, -zr - oz), (x - ox, y - oy, -z - oz),
             (-x - ox, y - oy, -z - oz), (-x - ox, -y - oy, zr - oz), (x - ox, -y - oy, zr - oz),
             (x - ox, y - oy, z - oz), (-x - ox, y - oy, z - oz)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.materials.append(_material("AD_Cabinet", (0.05, 0.05, 0.05, 1)))
    me.materials.append(_material("AD_Front", (0.35, 0.35, 0.38, 1)))
    me.polygons[4].material_index = 1
    return me


def grid_mesh(name, width, depth, cell):
    nx = max(1, int(np.ceil(width / cell)))
    ny = max(1, int(np.ceil(depth / cell)))
    xs = np.linspace(-width / 2, width / 2, nx + 1)
    ys = np.linspace(-depth / 2, depth / 2, ny + 1)
    verts = [(x, y, 0.0) for y in ys for x in xs]
    faces = [(j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i)
             for j in range(ny) for i in range(nx)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def face_points(ob, height):
    """World-space listening points: face centres, raised by `height` on floor-like faces.

    ob.ad.listen_mode: AUTO raises faces whose normal is within 60 degrees of vertical (so a receiver stood on its
    side, a vertical cut through the room, stays exactly where it is), ALWAYS raises all, NEVER none.
    """
    me = ob.data
    n = len(me.polygons)
    centers = np.empty(n * 3, dtype=np.float32)
    me.polygons.foreach_get("center", centers)
    normals = np.empty(n * 3, dtype=np.float32)
    me.polygons.foreach_get("normal", normals)
    mw = np.array(ob.matrix_world)
    pts = centers.reshape(n, 3) @ mw[:3, :3].T + mw[:3, 3]
    nz = np.abs((normals.reshape(n, 3) @ mw[:3, :3].T) @ np.array([0.0, 0.0, 1.0]))
    nz = nz / np.maximum(np.linalg.norm(normals.reshape(n, 3) @ mw[:3, :3].T, axis=1), 1e-9)
    mode = ob.ad.listen_mode
    lift = np.ones(n) if mode == "ALWAYS" else np.zeros(n) if mode == "NEVER" else (nz >= 0.5).astype(float)
    pts[:, 2] += height * lift
    return pts


def write_face_floats(me, name, values):
    attr = me.attributes.get(name) or me.attributes.new(name, "FLOAT", "FACE")
    attr.data.foreach_set("value", np.asarray(values, dtype=np.float32))


def paint_heatmap(ob, values, lo, hi):
    me = ob.data
    n = len(me.polygons)
    totals = np.array([len(p.vertices) for p in me.polygons], dtype=np.int32)
    loop_face = np.repeat(np.arange(n), totals)
    rgba = colormap(values, lo, hi)[loop_face]

    attr = me.color_attributes.get(AD_COLOR_ATTR)
    if attr is None:
        attr = me.color_attributes.new(AD_COLOR_ATTR, "FLOAT_COLOR", "CORNER")
    attr.data.foreach_set("color", rgba.ravel().astype(np.float32))
    me.color_attributes.active_color = attr
    me.color_attributes.render_color_index = list(me.color_attributes.keys()).index(AD_COLOR_ATTR)

    mat = bpy.data.materials.get(HEATMAP_MATERIAL) or _heatmap_material()
    if mat.name not in [m.name for m in me.materials if m]:
        me.materials.append(mat)
    me.update()


def _heatmap_material():
    mat = bpy.data.materials.new(HEATMAP_MATERIAL)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = AD_COLOR_ATTR
    links.new(attr.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs[0], out.inputs["Surface"])
    return mat


def clear_heatmap(ob):
    me = ob.data
    for name in [a.name for a in me.attributes if a.name.startswith("AD_") and a.name != AD_COLOR_ATTR]:
        me.attributes.remove(me.attributes[name])
    attr = me.color_attributes.get(AD_COLOR_ATTR)
    if attr is not None:
        me.color_attributes.remove(attr)
    ob.ad.stat_metric = ""

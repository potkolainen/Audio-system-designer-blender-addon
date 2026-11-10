"""Volumetric SPL display inside a Volume Region empty.

Primary path: the field is uploaded as a 3D GPU texture and drawn with a ray-marched fog shader and/or stacks of cut
planes whose opacity follows a Slice Focus empty. If shader or texture creation fails, a coarser billboard-blob
overlay is used instead. Both are viewport overlays, not renderable Blender volume objects.
"""
import math

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from ..engine import FLOOR_DB, evaluate
from . import environment, scene_io, state
from .ad_collections import get_collection, new_object
from .util import select_only, start_grab, tag_redraw

MAX_VOXELS = 200_000
RAY_STEPS = 160

_VERT = """
void main() {
  localPos = pos;
  vec4 wp = modelMatrix * vec4(pos, 1.0);
  worldPos = wp.xyz;
  gl_Position = viewProjectionMatrix * wp;
}
"""

_COMMON = """
vec3 colormap(float t) {
  const vec3 c0 = vec3(0.02, 0.08, 0.85);
  const vec3 c1 = vec3(0.0, 0.75, 1.0);
  const vec3 c2 = vec3(0.0, 0.85, 0.15);
  const vec3 c3 = vec3(1.0, 0.85, 0.0);
  const vec3 c4 = vec3(0.9, 0.02, 0.0);
  t = clamp(t, 0.0, 1.0) * 4.0;
  if (t < 1.0) return mix(c0, c1, t);
  if (t < 2.0) return mix(c1, c2, t - 1.0);
  if (t < 3.0) return mix(c2, c3, t - 2.0);
  return mix(c3, c4, t - 3.0);
}

float fetchT(ivec3 i, ivec3 n) {
  return texelFetch(volumeTex, clamp(i, ivec3(0), n - ivec3(1)), 0).r;
}

float sampleT(vec3 uvw) {
  ivec3 n = textureSize(volumeTex, 0);
  vec3 p = uvw * vec3(n) - 0.5;
  vec3 f = fract(p);
  ivec3 i = ivec3(floor(p));
  float x00 = mix(fetchT(i, n), fetchT(i + ivec3(1, 0, 0), n), f.x);
  float x10 = mix(fetchT(i + ivec3(0, 1, 0), n), fetchT(i + ivec3(1, 1, 0), n), f.x);
  float x01 = mix(fetchT(i + ivec3(0, 0, 1), n), fetchT(i + ivec3(1, 0, 1), n), f.x);
  float x11 = mix(fetchT(i + ivec3(0, 1, 1), n), fetchT(i + ivec3(1, 1, 1), n), f.x);
  return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}
"""

_FOG_FRAG = _COMMON + """
void main() {
  vec3 ro;
  vec3 rd;
  if (isPersp > 0.5) {
    ro = (invModelMatrix * vec4(camPos, 1.0)).xyz;
    rd = (invModelMatrix * vec4(worldPos, 1.0)).xyz - ro;
  } else {
    rd = normalize((invModelMatrix * vec4(viewDir, 0.0)).xyz);
    ro = (invModelMatrix * vec4(worldPos, 1.0)).xyz - rd * 50.0;
  }
  vec3 invd = 1.0 / (rd + vec3(1e-9));
  vec3 t0 = (vec3(-1.0) - ro) * invd;
  vec3 t1 = (vec3(1.0) - ro) * invd;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  float tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
  float tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tf <= tn) discard;

  const int steps = %d;
  float dt = (tf - tn) / float(steps);
  float stepLen = dt * length(rd);
  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  vec3 col = vec3(0.0);
  float acc = 0.0;
  for (int k = 0; k < steps; k++) {
    float t = tn + (float(k) + jitter) * dt;
    float v = sampleT((ro + rd * t) * 0.5 + 0.5);
    if (v > threshold) {
      float s = (v - threshold) / (1.0 - threshold);
      float a = 1.0 - exp(-opacity * 6.0 * s * s * stepLen);
      col += (1.0 - acc) * a * colormap(v);
      acc += (1.0 - acc) * a;
      if (acc > 0.98) break;
    }
  }
  fragColor = vec4(col, acc);
}
""" % RAY_STEPS

_SLICE_FRAG = _COMMON + """
void main() {
  float v = sampleT(localPos * 0.5 + 0.5);
  if (v <= threshold) discard;
  float s = (v - threshold) / (1.0 - threshold);
  float d = abs(dot(localPos - focusLocal, axisVec));
  float w = exp(-0.5 * d * d / (sigma * sigma));
  float a = opacity * (0.3 + 0.7 * s) * mix(1.0, mix(floorAlpha, 1.0, w), useFocus);
  fragColor = vec4(colormap(v) * a, a);
}
"""

_CUBE_VERTS = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], dtype=np.float32)
_CUBE_TRIS = np.array([(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                       (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)], dtype=np.int32)

_PLANE_OFFSETS = np.array([
    [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
    [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)],
    [(0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1)],
], dtype=np.float32).reshape(12, 3)
_FAN = np.array([[0, 1 + 4 * p + k, 1 + 4 * p + (k + 1) % 4] for p in range(3) for k in range(4)], dtype=np.int32)

_S = {
    "handle": None, "info": "",
    "tex": None, "matrix": None, "fog": None, "slice": None, "cube": None, "slice_batches": {}, "shader_failed": False,
    "fallback": None, "fallback_shader": None, "blend": "ADDITIVE", "field": None,
}


def overlay_info() -> str:
    return _S["info"]


def field_snapshot():
    """The last computed field (normalised voxels, region matrix, size, range, label) or None."""
    return _S["field"]


def _blend_mode(st):
    return "ADDITIVE_PREMULT" if st.vol_blend == "ADDITIVE" else "ALPHA_PREMULT"


def _create_shaders():
    iface = gpu.types.GPUStageInterfaceInfo("ad_volume_iface")
    iface.smooth("VEC3", "worldPos")
    iface.smooth("VEC3", "localPos")

    def make(frag, extra):
        info = gpu.types.GPUShaderCreateInfo()
        for kind, name in [("MAT4", "viewProjectionMatrix"), ("MAT4", "modelMatrix"), ("MAT4", "invModelMatrix"),
                           ("VEC3", "camPos"), ("VEC3", "viewDir"), ("FLOAT", "isPersp"), ("FLOAT", "threshold"),
                           ("FLOAT", "opacity")] + extra:
            info.push_constant(kind, name)
        info.sampler(0, "FLOAT_3D", "volumeTex")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_out(iface)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_VERT)
        info.fragment_source(frag)
        return gpu.shader.create_from_info(info)

    _S["fog"] = make(_FOG_FRAG, [])
    _S["slice"] = make(_SLICE_FRAG, [("VEC3", "axisVec"), ("VEC3", "focusLocal"), ("FLOAT", "useFocus"),
                                     ("FLOAT", "sigma"), ("FLOAT", "floorAlpha")])
    _S["cube"] = batch_for_shader(_S["fog"], "TRIS", {"pos": _CUBE_VERTS}, indices=_CUBE_TRIS)


def _slice_batch(axis: int, count: int):
    key = (axis, count)
    if key not in _S["slice_batches"]:
        s = (np.arange(count, dtype=np.float32) + 0.5) / count * 2.0 - 1.0
        corners = np.array([[(-1, -1), (1, -1), (1, 1), (-1, 1)]], dtype=np.float32)[0]
        verts = np.zeros((count, 4, 3), dtype=np.float32)
        other = [i for i in range(3) if i != axis]
        verts[:, :, axis] = s[:, None]
        verts[:, :, other[0]] = corners[None, :, 0]
        verts[:, :, other[1]] = corners[None, :, 1]
        base = (np.arange(count, dtype=np.int32) * 4)[:, None, None]
        tris = (base + np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)[None]).reshape(-1, 3)
        _S["slice_batches"][key] = batch_for_shader(_S["slice"], "TRIS", {"pos": verts.reshape(-1, 3)}, indices=tris)
    return _S["slice_batches"][key]


def _fallback_shader():
    if _S["fallback_shader"] is None:
        try:
            _S["fallback_shader"] = gpu.shader.from_builtin("SMOOTH_COLOR")
        except ValueError:
            _S["fallback_shader"] = gpu.shader.from_builtin("3D_SMOOTH_COLOR")
    return _S["fallback_shader"]


def _build_fallback(points, rgba, half):
    m = len(points)
    pos = np.empty((m, 13, 3), dtype=np.float32)
    pos[:, 0] = points
    pos[:, 1:] = points[:, None, :] + _PLANE_OFFSETS[None] * half
    col = np.empty((m, 13, 4), dtype=np.float32)
    col[:, 0] = rgba
    col[:, 1:] = rgba[:, None, :]
    col[:, 1:, 3] = 0.0
    idx = (np.arange(m, dtype=np.int32)[:, None, None] * 13 + _FAN[None]).reshape(-1, 3)
    return batch_for_shader(_fallback_shader(), "TRIS", {"pos": pos.reshape(-1, 3), "color": col.reshape(-1, 4)},
                            indices=idx)


def _set(shader, name, value):
    try:
        shader.uniform_float(name, value)
    except ValueError:
        pass  # optimised out when a shader does not use this uniform


def _set_common(shader, rv3d, st):
    model = _S["matrix"]
    _set(shader, "viewProjectionMatrix", rv3d.perspective_matrix)
    _set(shader, "modelMatrix", model)
    _set(shader, "invModelMatrix", model.inverted())
    _set(shader, "camPos", rv3d.view_matrix.inverted().translation)
    _set(shader, "viewDir", rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)))
    _set(shader, "isPersp", 1.0 if rv3d.is_perspective else 0.0)
    _set(shader, "threshold", st.vol_threshold)
    _set(shader, "opacity", st.vol_opacity)
    shader.uniform_sampler("volumeTex", _S["tex"])


def _focus_local(scene):
    focus = next((o for o in scene.objects if o.ad.kind == "FOCUS"), None)
    if focus is None:
        return Vector((0.0, 0.0, 0.0)), 0.0
    return _S["matrix"].inverted() @ focus.matrix_world.translation, 1.0


def _draw_volume(scene, rv3d, st):
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)
    gpu.state.blend_set(_blend_mode(st))
    if st.vol_display in ("FOG", "BOTH"):
        shader = _S["fog"]
        shader.bind()
        _set_common(shader, rv3d, st)
        gpu.state.face_culling_set("FRONT")
        _S["cube"].draw(shader)
    if st.vol_display in ("SLICES", "BOTH"):
        shader = _S["slice"]
        shader.bind()
        _set_common(shader, rv3d, st)
        focus, use = _focus_local(scene)
        _set(shader, "focusLocal", focus)
        _set(shader, "useFocus", use)
        _set(shader, "sigma", max(st.slice_sigma * 2.0, 1e-3))
        _set(shader, "floorAlpha", st.slice_floor)
        gpu.state.face_culling_set("NONE")
        for axis in range(3):
            if st.slice_axes[axis]:
                vec = [0.0, 0.0, 0.0]
                vec[axis] = 1.0
                _set(shader, "axisVec", vec)
                _slice_batch(axis, st.slice_count).draw(shader)
    gpu.state.face_culling_set("NONE")


def _draw_fallback():
    gpu.state.blend_set(_S["blend"])
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(False)
    shader = _fallback_shader()
    shader.bind()
    _S["fallback"].draw(shader)


def _draw():
    ctx = bpy.context
    rv3d = ctx.region_data
    if rv3d is None:
        return
    try:
        if _S["tex"] is not None:
            _draw_volume(ctx.scene, rv3d, ctx.scene.ad_settings)
        elif _S["fallback"] is not None:
            _draw_fallback()
    except Exception as e:  # a GPU error must never break the viewport
        print("[Audio Designer] volume draw failed, switching to fallback:", e)
        _S["tex"] = None
    finally:
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
        gpu.state.face_culling_set("NONE")


def register_overlay():
    _S["handle"] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")


def unregister_overlay():
    if _S["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_S["handle"], "WINDOW")
        _S["handle"] = None
    _S.update(tex=None, fallback=None, slice_batches={})


def volume_region(context):
    ob = context.object
    if ob is not None and ob.ad.kind == "VOLUME":
        return ob
    return next((o for o in context.scene.objects if o.ad.kind == "VOLUME"), None)


class AD_OT_add_volume_region(bpy.types.Operator):
    bl_idname = "audio_designer.add_volume_region"
    bl_label = "Add Volume Region"
    bl_description = "Box that bounds the volumetric field. Move, scale and rotate it; make it thin for a slice"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ob = new_object(context.scene, get_collection(context.scene, "Visualization"), "Audio Volume Region", None)
        ob.empty_display_type = "CUBE"
        ob.empty_display_size = 1.0
        ob.scale = (10.0, 10.0, 4.0)
        ob.location.z += 4.0
        ob.ad.kind = "VOLUME"
        select_only(context, ob)
        start_grab(context)
        return {"FINISHED"}


class AD_OT_add_slice_focus(bpy.types.Operator):
    bl_idname = "audio_designer.add_slice_focus"
    bl_label = "Add Slice Focus"
    bl_description = ("Empty that controls the slice planes: planes near it are opaque and the others fade. "
                      "Snap it to surfaces with Blender's snapping")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ob = new_object(context.scene, get_collection(context.scene, "Visualization"), "Audio Slice Focus", None)
        ob.empty_display_type = "PLAIN_AXES"
        ob.empty_display_size = 1.0
        ob.ad.kind = "FOCUS"
        select_only(context, ob)
        start_grab(context)
        return {"FINISHED"}


class AD_OT_calculate_volume(bpy.types.Operator):
    bl_idname = "audio_designer.calculate_volume"
    bl_label = "Calculate Volumetric Field"
    bl_description = "Evaluate the selected map on a 3D grid inside the volume region"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene, st = context.scene, context.scene.ad_settings
        region = volume_region(context)
        if region is None:
            self.report({"ERROR"}, "Add a volume region first.")
            return {"CANCELLED"}
        sources, _ = scene_io.collect_sources(scene, state.LIBRARY)
        if not sources:
            self.report({"ERROR"}, "No simulation-capable speakers in the scene.")
            return {"CANCELLED"}
        if st.metric == "STI":
            self.report({"ERROR"}, "STI is only available on receiver surfaces. Choose another map.")
            return {"CANCELLED"}
        env, _notes = environment.active_environment(scene, state.LIBRARY)

        mw = region.matrix_world.copy()
        dims = [max(2.0 * abs(s), 1e-3) for s in mw.to_scale()]
        counts = [max(2, math.ceil(d / st.vol_cell_m)) for d in dims]
        total = counts[0] * counts[1] * counts[2]
        if total > MAX_VOXELS:
            self.report({"ERROR"}, f"{total} voxels (limit {MAX_VOXELS}). Increase voxel size or shrink the region.")
            return {"CANCELLED"}

        # x varies fastest, as the 3D texture upload expects.
        ax, ay, az = [(np.arange(n) + 0.5) / n * 2.0 - 1.0 for n in counts]
        zz, yy, xx = np.meshgrid(az, ay, ax, indexing="ij")
        local = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        m = np.array(mw)
        world = local @ m[:3, :3].T + m[:3, 3]

        values, _ = evaluate(sources, world, scene_io.prediction_settings(st), st.mode, st.metric,
                             scene_io.selected_freq_hz(st), env=env)
        valid = values > FLOOR_DB
        if not valid.any():
            self.report({"WARNING"}, "No signal in the region.")
            return {"CANCELLED"}
        if st.auto_range:
            hi = math.ceil(float(np.percentile(values[valid], 99)) / 5.0) * 5.0
            lo = hi - st.vol_range_db
        else:
            lo, hi = st.min_db, st.max_db
        t = np.where(valid, np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0), 0.0).astype(np.float32)

        _S["matrix"] = mw
        _S["tex"] = None
        _S["blend"] = st.vol_blend
        _S["field"] = {"t": t.reshape(counts[2], counts[1], counts[0]), "matrix": mw.copy(), "dims": dims, "lo": lo,
                       "hi": hi, "label": scene_io.metric_label(st)}
        try:
            if _S["fog"] is None and not _S["shader_failed"]:
                _create_shaders()
            if _S["fog"] is not None:
                buf = gpu.types.Buffer("FLOAT", len(t), t.tolist())
                _S["tex"] = gpu.types.GPUTexture(size=tuple(counts), format="R32F", data=buf)
        except Exception as e:
            _S["shader_failed"] = True
            _S["fog"] = _S["slice"] = None
            print("[Audio Designer] 3D texture rendering unavailable, using billboard fallback:", e)
        if _S["tex"] is None:
            keep = t > st.vol_threshold
            if not keep.any():
                self.report({"WARNING"}, "All voxels fall below the 'Hide below' threshold.")
                return {"CANCELLED"}
            rgba = scene_io.colormap(lo + t[keep] * (hi - lo), lo, hi).astype(np.float32)
            rgba[:, 3] = st.vol_opacity * t[keep]
            cell = max(dims[i] / counts[i] for i in range(3))
            _S["fallback"] = _build_fallback(world[keep].astype(np.float32), rgba, cell * 0.75)

        mode = "3D texture" if _S["tex"] is not None else "billboard fallback"
        _S["info"] = f"{scene_io.metric_label(st)}: {counts[0]}x{counts[1]}x{counts[2]} voxels, {lo:g} to {hi:g} dB ({mode})"
        tag_redraw(context)
        self.report({"INFO"}, _S["info"])
        return {"FINISHED"}


class AD_OT_clear_volume(bpy.types.Operator):
    bl_idname = "audio_designer.clear_volume"
    bl_label = "Clear Volumetric Field"

    def execute(self, context):
        _S.update(tex=None, fallback=None, info="", field=None)
        tag_redraw(context)
        return {"FINISHED"}


classes = (AD_OT_add_volume_region, AD_OT_add_slice_focus, AD_OT_calculate_volume, AD_OT_clear_volume)

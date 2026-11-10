"""Line array and subwoofer array builders for the Blender layer."""
import math

import bpy
import numpy as np
from bpy.props import FloatProperty, IntProperty
from mathutils import Euler, Vector

from ..engine.arrays import (
    array_summary, audience_aim_distances, auto_splay, line_array_layout, line_profile, ring_levels,
    sub_array_layout,
)
from . import scene_io, state
from .ad_collections import get_collection, new_object
from .operators import selected_library_model
from .util import find_root, select_only, start_grab

SUB_REPORT_FREQS = np.array([31.5, 40, 50, 63, 80, 100, 125, 160, 200])
RING_ANGLES = np.arange(0, 360, 10)


def array_cabinets(root):
    return sorted((o for o in root.children if o.ad.kind == "SPEAKER"), key=lambda o: o.ad.array_index)


def line_layout_for(root):
    model = state.LIBRARY.get(root.ad.model_id)
    n = len(array_cabinets(root))
    if model is None or n == 0:
        return None, model
    return line_array_layout(n, model.height_m, model.depth_m, list(root.ad.splays)[: n - 1],
                             root.ad.top_tilt_deg), model


def relayout_line_array(root):
    """Position and rotate every cabinet from the root's splay settings."""
    if root is None or root.ad.kind != "LINE_ARRAY":
        return
    layout, model = line_layout_for(root)
    if layout is None:
        return
    origin = Vector(model.acoustic_origin_m)
    for i, cab in enumerate(array_cabinets(root)):
        rot = Euler((-math.radians(layout.tilts_deg[i]), 0.0, 0.0), "XYZ")
        cab.rotation_mode = "XYZ"
        cab.rotation_euler = rot
        cab.location = Vector(layout.centres[i]) + rot.to_matrix() @ origin
        cab.lock_location = cab.lock_rotation = (True, True, True)


def _new_cabinet(scene, col, root, model, index):
    cab = new_object(scene, col, f"{root.name} Cab {index + 1}", scene_io.speaker_mesh(model))
    cab.parent = root
    cab.ad.kind = "SPEAKER"
    cab.ad.model_id = model.id
    cab.ad.array_index = index
    return cab


class AD_OT_add_line_array(bpy.types.Operator):
    bl_idname = "audio_designer.add_line_array"
    bl_label = "Add Line Array"
    bl_description = "Build a line array from the selected library model (category: line array)"
    bl_options = {"REGISTER", "UNDO"}

    count: IntProperty(name="Cabinets", default=8, min=2, max=24)
    initial_splay: FloatProperty(name="Initial splay", default=2.0, min=0.0, max=30.0)

    def execute(self, context):
        model = selected_library_model(context)
        if model is None or model.category != "line_array" or model.acoustic is None:
            self.report({"ERROR"}, "Select a simulation-capable line array model in the library.")
            return {"CANCELLED"}
        n = min(self.count, model.max_cabinets)
        scene = context.scene
        root = bpy.data.objects.new("Line Array", None)
        root.location = scene.cursor.location
        root.rotation_euler.z = scene_io.audience_yaw(scene, root.location)
        col = get_collection(scene, "Arrays", root.name)
        col.objects.link(root)
        root.empty_display_type = "ARROWS"
        root.empty_display_size = 0.5
        root.ad.kind = "LINE_ARRAY"
        root.ad.model_id = model.id
        root.ad.array_count = n
        root.ad.splays = [min(self.initial_splay, model.max_splay_deg)] * (n - 1) + [0.0] * (24 - n + 1)
        for i in range(n):
            _new_cabinet(scene, col, root, model, i)
        relayout_line_array(root)
        select_only(context, root)
        start_grab(context)
        return {"FINISHED"}


class _LineArrayOp:
    @classmethod
    def poll(cls, context):
        return find_root(context.object, "LINE_ARRAY") is not None


class AD_OT_array_add_cabinet(_LineArrayOp, bpy.types.Operator):
    bl_idname = "audio_designer.array_add_cabinet"
    bl_label = "Add Cabinet"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = find_root(context.object, "LINE_ARRAY")
        model = state.LIBRARY.get(root.ad.model_id)
        cabs = array_cabinets(root)
        if model is None or len(cabs) >= model.max_cabinets:
            self.report({"WARNING"}, "Maximum cabinet count reached.")
            return {"CANCELLED"}
        col = root.users_collection[0]
        _new_cabinet(context.scene, col, root, model, len(cabs))
        root.ad.array_count = len(cabs) + 1
        relayout_line_array(root)
        return {"FINISHED"}


class AD_OT_array_remove_cabinet(_LineArrayOp, bpy.types.Operator):
    bl_idname = "audio_designer.array_remove_cabinet"
    bl_label = "Remove Bottom Cabinet"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = find_root(context.object, "LINE_ARRAY")
        cabs = array_cabinets(root)
        if len(cabs) <= 2:
            self.report({"WARNING"}, "An array needs at least two cabinets.")
            return {"CANCELLED"}
        bpy.data.objects.remove(cabs[-1], do_unlink=True)
        root.ad.array_count = len(cabs) - 1
        relayout_line_array(root)
        return {"FINISHED"}


class AD_OT_auto_splay(_LineArrayOp, bpy.types.Operator):
    bl_idname = "audio_designer.auto_splay"
    bl_label = "Auto Splay"
    bl_description = ("Suggest splay angles that aim the cabinets at evenly spaced points between the near and far "
                      "audience distances. Geometric suggestion only")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = find_root(context.object, "LINE_ARRAY")
        layout, model = line_layout_for(root)
        if layout is None:
            self.report({"ERROR"}, "Array model is missing from the library.")
            return {"CANCELLED"}
        st = context.scene.ad_settings
        hang = root.matrix_world.translation.z - (root.ad.audience_z + st.listen_height)
        try:
            top, splays = auto_splay(len(layout.centres), model.height_m, model.depth_m, hang, root.ad.aim_near_m,
                                     root.ad.aim_far_m, 0.0, model.max_splay_deg)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        vec = [0.0] * 24
        vec[: len(splays)] = [float(s) for s in splays]
        root.ad.top_tilt_deg = top
        root.ad.splays = vec
        relayout_line_array(root)
        return {"FINISHED"}


def line_aim_distances(root, context):
    layout, _ = line_layout_for(root)
    if layout is None:
        return None
    hang = root.matrix_world.translation.z - (root.ad.audience_z + context.scene.ad_settings.listen_height)
    return audience_aim_distances(layout, hang) if hang > 0 else None


class AD_OT_array_report(_LineArrayOp, bpy.types.Operator):
    bl_idname = "audio_designer.array_report"
    bl_label = "Analyse Array"
    bl_description = "Weight, size and level drop along the audience line (octave power sum, this array alone)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene, st = context.scene, context.scene.ad_settings
        root = find_root(context.object, "LINE_ARRAY")
        layout, model = line_layout_for(root)
        if layout is None:
            self.report({"ERROR"}, "Array model is missing from the library.")
            return {"CANCELLED"}
        sources, _ = scene_io.collect_sources(scene, state.LIBRARY, parent=root)
        if not sources:
            self.report({"ERROR"}, "No active speakers in this array.")
            return {"CANCELLED"}
        summary = array_summary(layout, model.weight_kg)
        fwd = scene_io.horizontal_forward(root)
        origin = root.matrix_world.translation
        z = root.ad.audience_z + st.listen_height
        dist, res = line_profile(sources, (origin.x, origin.y), fwd, z, root.ad.aim_near_m, root.ad.aim_far_m,
                                 scene_io.prediction_settings(st))
        a = res.a_weighted
        lines = [
            f"{summary['cabinets']} cabinets, {summary['total_weight_kg']:.0f} kg nominal (cabinets only)",
            f"Height {summary['height_m']:.2f} m, depth {summary['depth_m']:.2f} m, total splay {summary['total_splay_deg']:.1f} deg",
            f"Approx. CoG (local) y {summary['cog_local_m'][1]:.2f} z {summary['cog_local_m'][2]:.2f} m",
            f"dB(A) near {a[0]:.1f}  far {a[-1]:.1f}  drop {a[0] - a[-1]:.1f}",
            f"dB(A) min {a.min():.1f}  max {a.max():.1f}  spread {a.max() - a.min():.1f}",
            "Not rigging or safety data.",
        ]
        root.ad.report = "\n".join(lines)
        return {"FINISHED"}


class AD_OT_add_sub_array(bpy.types.Operator):
    bl_idname = "audio_designer.add_sub_array"
    bl_label = "Add Subwoofer Array"
    bl_description = "Build a subwoofer array with the delay and polarity recipe from the selected library subwoofer"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        model = selected_library_model(context)
        if model is None or model.category != "subwoofer" or model.acoustic is None:
            self.report({"ERROR"}, "Select a simulation-capable subwoofer model in the library.")
            return {"CANCELLED"}
        scene = context.scene
        root = bpy.data.objects.new("Sub Array", None)
        root.location = scene.cursor.location
        root.rotation_euler.z = scene_io.audience_yaw(scene, root.location)
        col = get_collection(scene, "Subwoofer Arrays", root.name)
        col.objects.link(root)
        root.empty_display_type = "ARROWS"
        root.empty_display_size = 0.5
        root.ad.kind = "SUB_ARRAY"
        root.ad.model_id = model.id
        build_sub_array(context, root)
        select_only(context, root)
        start_grab(context)
        return {"FINISHED"}


def build_sub_array(context, root):
    """(Re)create the subwoofer children from the root's configuration and recipe."""
    model = state.LIBRARY.get(root.ad.model_id)
    st = context.scene.ad_settings
    for old in list(root.children):
        if old.ad.kind == "SPEAKER":
            bpy.data.objects.remove(old, do_unlink=True)
    kind = root.ad.sub_kind
    if kind == "CUSTOM":
        recipe = sub_array_layout("BROADSIDE", root.ad.sub_count, root.ad.sub_spacing_m, scene_io.prediction_settings(st).c)
        for r in recipe:
            r.delay_ms = 0.0
    else:
        recipe = sub_array_layout(kind, root.ad.sub_count, root.ad.sub_spacing_m, scene_io.prediction_settings(st).c)
    col = root.users_collection[0]
    ox, oy, oz = model.acoustic_origin_m
    for i, r in enumerate(recipe):
        sub = new_object(context.scene, col, f"{root.name} Sub {i + 1}", scene_io.speaker_mesh(model))
        sub.parent = root
        sub.rotation_euler = (0.0, 0.0, math.radians(r.yaw_deg))
        sub.location = (r.x, r.y, model.height_m / 2.0 + oz)
        sub.ad.kind = "SPEAKER"
        sub.ad.model_id = model.id
        sub.ad.array_index = i
        sub.ad.delay_ms = r.delay_ms
        sub.ad.polarity_inverted = r.polarity < 0
        sub.ad.level_db = -10.0


class AD_OT_rebuild_sub_array(bpy.types.Operator):
    bl_idname = "audio_designer.rebuild_sub_array"
    bl_label = "Rebuild From Recipe"
    bl_description = "Recreate the subwoofers from the configuration. Manual changes to the units are lost"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return find_root(context.object, "SUB_ARRAY") is not None

    def execute(self, context):
        root = find_root(context.object, "SUB_ARRAY")
        if state.LIBRARY.get(root.ad.model_id) is None:
            self.report({"ERROR"}, "Subwoofer model is missing from the library.")
            return {"CANCELLED"}
        build_sub_array(context, root)
        select_only(context, root)
        return {"FINISHED"}


class AD_OT_sub_report(bpy.types.Operator):
    bl_idname = "audio_designer.sub_report"
    bl_label = "Analyse Subwoofer Array"
    bl_description = "Front-to-back levels on a horizontal ring around the array (phase-aware, this array alone)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return find_root(context.object, "SUB_ARRAY") is not None

    def execute(self, context):
        scene, st = context.scene, context.scene.ad_settings
        root = find_root(context.object, "SUB_ARRAY")
        sources, _ = scene_io.collect_sources(scene, state.LIBRARY, parent=root)
        if not sources:
            self.report({"ERROR"}, "No active subwoofers in this array.")
            return {"CANCELLED"}
        centre = np.array(root.matrix_world.translation) + np.array([0, 0, 0.3])
        ring = ring_levels(sources, centre, root.ad.analysis_distance_m, SUB_REPORT_FREQS,
                           scene_io.prediction_settings(st), [0, 180], scene_io.horizontal_forward(root))
        lines = [f"At {root.ad.analysis_distance_m:g} m  (Hz: front / back / F-B dB)"]
        for i, f in enumerate(SUB_REPORT_FREQS):
            lines.append(f"{f:g} Hz: {ring[0, i]:.1f} / {ring[1, i]:.1f} / {ring[0, i] - ring[1, i]:+.1f}")
        root.ad.report = "\n".join(lines)
        return {"FINISHED"}


classes = (AD_OT_add_line_array, AD_OT_array_add_cabinet, AD_OT_array_remove_cabinet, AD_OT_auto_splay,
           AD_OT_array_report, AD_OT_add_sub_array, AD_OT_rebuild_sub_array, AD_OT_sub_report)

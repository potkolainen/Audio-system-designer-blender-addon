import json
import math

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, StringProperty
from mathutils import Matrix

from ..engine import coverage_stats, evaluate, predict_sti
from ..library import CATEGORIES, LibraryError, parse_model
from ..library.datasheet import build_model_dict
from . import environment, scene_io, state
from .ad_collections import get_collection, new_object
from .util import select_only, start_grab

PREFIX_CONE = "AD_Cone_"
MAX_CONE_HALF_ANGLE_DEG = 85.0
MAX_RECEIVER_CELLS = 250_000


def selected_library_model(context):
    wm = context.window_manager
    if 0 <= wm.ad_library_index < len(wm.ad_library_items):
        return state.LIBRARY.get(wm.ad_library_items[wm.ad_library_index].model_id)
    return None


class AD_OT_reload_library(bpy.types.Operator):
    bl_idname = "audio_designer.reload_library"
    bl_label = "Reload Library"

    def execute(self, context):
        state.reload_library()
        n_err = len(state.LIBRARY.errors)
        self.report({"WARNING" if n_err else "INFO"},
                    f"{len(state.LIBRARY.models)} models loaded" + (f", {n_err} file(s) rejected" if n_err else ""))
        for err in state.LIBRARY.errors:
            print("[Audio Designer]", err)
        return {"FINISHED"}


class AD_OT_place_speaker(bpy.types.Operator):
    bl_idname = "audio_designer.place_speaker"
    bl_label = "Place Speaker"
    bl_options = {"REGISTER", "UNDO"}

    model_id: StringProperty(options={"HIDDEN"})

    def execute(self, context):
        model = state.LIBRARY.get(self.model_id) if self.model_id else selected_library_model(context)
        if model is None:
            self.report({"ERROR"}, "Select a loudspeaker model first.")
            return {"CANCELLED"}
        ob = new_object(context.scene, get_collection(context.scene, "Speakers"), model.model,
                        scene_io.speaker_mesh(model))
        ob.ad.kind = "SPEAKER"
        ob.ad.model_id = model.id
        ob.rotation_euler.z = scene_io.audience_yaw(context.scene, ob.location)
        select_only(context, ob)
        if not model.data_quality.simulatable:
            self.report({"WARNING"}, "Visual-only model: it will be ignored by simulations.")
        start_grab(context)
        return {"FINISHED"}


class AD_OT_add_receiver(bpy.types.Operator):
    bl_idname = "audio_designer.add_receiver"
    bl_label = "Add Receiver Area"
    bl_options = {"REGISTER", "UNDO"}

    width: FloatProperty(name="Width", default=20.0, min=0.5, max=500.0, unit="LENGTH")
    depth: FloatProperty(name="Depth", default=15.0, min=0.5, max=500.0, unit="LENGTH")
    cell: FloatProperty(name="Cell size", default=0.5, min=0.05, max=20.0, unit="LENGTH")

    def execute(self, context):
        if (self.width / self.cell) * (self.depth / self.cell) > MAX_RECEIVER_CELLS:
            self.report({"ERROR"}, "Grid too dense (over 250k cells). Increase cell size.")
            return {"CANCELLED"}
        ob = new_object(context.scene, get_collection(context.scene, "Receivers"), "Audio Receiver",
                        scene_io.grid_mesh("AD_ReceiverGrid", self.width, self.depth, self.cell))
        ob.ad.kind = "RECEIVER"
        select_only(context, ob)
        return {"FINISHED"}


class AD_OT_receiver_from_mesh(bpy.types.Operator):
    bl_idname = "audio_designer.receiver_from_mesh"
    bl_label = "Receiver From Selected Mesh"
    bl_description = ("Duplicate the selected mesh (raised 2 cm) as a receiver surface. "
                      "Every face centre becomes a listening point, so subdivide the source mesh for resolution")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(o.type == "MESH" and o.ad.kind == "NONE" for o in context.selected_objects)

    def execute(self, context):
        col = get_collection(context.scene, "Receivers")
        created = []
        for src in [o for o in context.selected_objects if o.type == "MESH" and o.ad.kind == "NONE"]:
            ob = src.copy()
            ob.data = src.data.copy()
            ob.name = "Audio Receiver " + src.name
            ob.data.materials.clear()
            ob.location.z += 0.02
            col.objects.link(ob)
            ob.ad.kind = "RECEIVER"
            created.append(ob)
        for o in context.selected_objects:
            o.select_set(False)
        for ob in created:
            ob.select_set(True)
        context.view_layer.objects.active = created[-1]
        return {"FINISHED"}


class AD_OT_add_probe(bpy.types.Operator):
    bl_idname = "audio_designer.add_probe"
    bl_label = "Add Audio Probe"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ob = new_object(context.scene, get_collection(context.scene, "Probes"), "Audio Probe", None)
        ob.empty_display_type = "SPHERE"
        ob.empty_display_size = 0.3
        ob.ad.kind = "PROBE"
        select_only(context, ob)
        start_grab(context)
        return {"FINISHED"}


class AD_OT_calculate(bpy.types.Operator):
    bl_idname = "audio_designer.calculate"
    bl_label = "Calculate Coverage"
    bl_description = "Free-field prediction on all receiver surfaces. Not validated"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        st = scene.ad_settings
        sources, skipped = scene_io.collect_sources(scene, state.LIBRARY)
        receivers = [o for o in scene.objects if o.ad.kind == "RECEIVER" and o.type == "MESH" and o.data.polygons]
        if not sources:
            self.report({"ERROR"}, "No simulation-capable speakers in the scene.")
            return {"CANCELLED"}
        if not receivers:
            self.report({"ERROR"}, "No receiver surfaces. Add a receiver area first.")
            return {"CANCELLED"}

        settings = scene_io.prediction_settings(st)
        freq = scene_io.selected_freq_hz(st)
        env, notes = environment.active_environment(scene, state.LIBRARY)
        room = None
        if st.metric == "STI":
            room, room_msg = environment.room_for_scene(scene, state.LIBRARY)
            if room is None:
                self.report({"ERROR"}, room_msg)
                return {"CANCELLED"}
        results = []
        for ob in receivers:
            pts = scene_io.face_points(ob, st.listen_height)
            if room is not None:
                values = predict_sti(sources, pts, room, st.noise_dba, settings, env)
                cols = {"STI": values}
            else:
                values, cols = evaluate(sources, pts, settings, st.mode, st.metric, freq, want_columns=True, env=env)
            results.append((ob, values, cols))

        if st.metric == "STI":
            lo, hi = 0.0, 1.0
        else:
            if st.auto_range:
                stats = coverage_stats(np.concatenate([v for _, v, _ in results]))
                st.min_db = math.floor(stats["min"] / 5.0) * 5.0
                st.max_db = max(math.ceil(stats["max"] / 5.0) * 5.0, st.min_db + 5.0)
            lo, hi = st.min_db, st.max_db

        label = scene_io.metric_label(st)
        for ob, values, cols in results:
            scene_io.clear_heatmap(ob)
            for name, data in cols.items():
                scene_io.write_face_floats(ob.data, "AD_" + name, data)
            scene_io.paint_heatmap(ob, values, lo, hi)
            s = coverage_stats(values)
            ob.ad.stat_metric = label
            ob.ad.stat_height = st.listen_height
            ob.ad.stat_lo, ob.ad.stat_hi = lo, hi
            ob.ad.stat_min, ob.ad.stat_max = s["min"], s["max"]
            ob.ad.stat_mean, ob.ad.stat_spread = s["mean"], s["spread"]

        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                try:
                    area.spaces.active.shading.color_type = "VERTEX"
                except (AttributeError, TypeError):
                    pass

        msg = f"{sum(len(o.data.polygons) for o in receivers)} points, {len(sources)} source(s)"
        if env.reflectors:
            msg += f", {len(env.reflectors)} reflector triangle(s), order {env.reflection_order}"
        if env.occluder is not None:
            msg += ", occlusion on"
        msg += "."
        if skipped:
            msg += f" Skipped (no acoustic data): {', '.join(skipped)}."
        if notes:
            msg += " " + "; ".join(notes) + "."
        self.report({"WARNING" if skipped or notes else "INFO"}, msg)
        return {"FINISHED"}


class AD_OT_assign_surface(bpy.types.Operator):
    bl_idname = "audio_designer.assign_surface"
    bl_label = "Assign Surface"
    bl_description = "Give the selected meshes the chosen library surface so they reflect sound"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(o.type == "MESH" and o.ad.kind == "NONE" for o in context.selected_objects)

    def execute(self, context):
        material = state.LIBRARY.surface(context.window_manager.ad_surface_choice)
        if material is None:
            self.report({"ERROR"}, "Choose a surface first.")
            return {"CANCELLED"}
        count = 0
        for ob in context.selected_objects:
            if ob.type == "MESH" and ob.ad.kind == "NONE":
                ob.ad.surface_id = material.id
                ob.ad.surface_custom = False
                ob.ad.surface_absorption = [float(v) for v in material.absorption]
                ob.ad.surface_scattering = material.scattering
                count += 1
        self.report({"INFO"}, f"{material.name} assigned to {count} object(s).")
        return {"FINISHED"}


class AD_OT_clear_surface(bpy.types.Operator):
    bl_idname = "audio_designer.clear_surface"
    bl_label = "Clear Surface"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        for ob in context.selected_objects:
            ob.ad.surface_id = ""
            ob.ad.surface_custom = False
        return {"FINISHED"}


class AD_OT_new_speaker(bpy.types.Operator):
    bl_idname = "audio_designer.new_speaker"
    bl_label = "New Speaker From Datasheet"
    bl_description = ("Create a library entry from datasheet figures. The polar shape is modelled from the nominal "
                      "coverage, so the entry is marked as an approximation")
    bl_options = {"REGISTER"}

    manufacturer: StringProperty(name="Manufacturer", default="")
    model: StringProperty(name="Model", default="")
    category: EnumProperty(name="Category", items=[(c, c.replace("_", " ").title(), "") for c in CATEGORIES],
                           default="point_source")
    width_mm: FloatProperty(name="Width (mm)", default=300.0, min=1.0)
    height_mm: FloatProperty(name="Height (mm)", default=500.0, min=1.0)
    depth_mm: FloatProperty(name="Depth (mm)", default=300.0, min=1.0)
    weight_kg: FloatProperty(name="Weight (kg)", default=10.0, min=0.1)
    max_spl_db: FloatProperty(name="Max SPL at 1 m (dB)", default=125.0, min=60.0, max=180.0)
    freq_lo_hz: FloatProperty(name="Low limit (Hz)", default=60.0, min=10.0)
    freq_hi_hz: FloatProperty(name="High limit (Hz)", default=18000.0, min=20.0)
    pattern: EnumProperty(name="Pattern", items=[
        ("coverage", "Nominal coverage", "Horizontal x vertical angle below"),
        ("omni", "Subwoofer, near omni", ""), ("cardioid", "Subwoofer, cardioid", "")], default="coverage")
    h_deg: FloatProperty(name="Horizontal (deg)", default=90.0, min=5.0, max=360.0)
    v_deg: FloatProperty(name="Vertical (deg)", default=60.0, min=5.0, max=360.0)
    control_freq_hz: FloatProperty(name="Pattern control from (Hz)", description="Coverage is held at the nominal angle above this frequency and widens below", default=1000.0, min=100.0)
    side_angle_deg: FloatProperty(name="Line array: cabinet side angle (deg)", default=0.0, min=0.0, max=59.0)
    max_splay_deg: FloatProperty(name="Line array: max splay (deg)", default=10.0, min=0.0, max=30.0)
    source_url: StringProperty(name="Datasheet URL", default="")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def execute(self, context):
        if not self.manufacturer.strip() or not self.model.strip():
            self.report({"ERROR"}, "Manufacturer and model are required.")
            return {"CANCELLED"}
        try:
            data = build_model_dict(
                manufacturer=self.manufacturer.strip(), model=self.model.strip(), category=self.category,
                width_m=self.width_mm / 1000.0, height_m=self.height_mm / 1000.0, depth_m=self.depth_mm / 1000.0,
                weight_kg=self.weight_kg, max_spl_db=self.max_spl_db, freq_lo_hz=self.freq_lo_hz,
                freq_hi_hz=self.freq_hi_hz, h_deg=self.h_deg, v_deg=self.v_deg,
                control_freq_hz=self.control_freq_hz, pattern=self.pattern, source_url=self.source_url,
                source_name=f"{self.manufacturer.strip()} {self.model.strip()} datasheet (entered by user)",
                notes="Entered by the user from datasheet figures. Polar shape is modelled, not measured.",
                side_angle_deg=self.side_angle_deg, max_splay_deg=self.max_splay_deg)
            parse_model(data)
        except (ValueError, LibraryError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        target = state.default_user_dir() / f"{data['id'].replace('.', '_')}.json"
        if target.exists() or data["id"] in state.LIBRARY.models:
            self.report({"ERROR"}, f"'{data['id']}' already exists. Use a different model name.")
            return {"CANCELLED"}
        try:
            target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError as e:
            self.report({"ERROR"}, f"Could not save: {e}")
            return {"CANCELLED"}
        state.reload_library()
        self.report({"INFO"}, f"Saved {target}")
        return {"FINISHED"}


class AD_OT_clear_heatmap(bpy.types.Operator):
    bl_idname = "audio_designer.clear_heatmap"
    bl_label = "Clear Heatmap"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        for ob in context.scene.objects:
            if ob.ad.kind == "RECEIVER" and ob.type == "MESH":
                scene_io.clear_heatmap(ob)
        return {"FINISHED"}


class AD_OT_update_cones(bpy.types.Operator):
    bl_idname = "audio_designer.update_cones"
    bl_label = "Build / Update Coverage Cones"
    bl_description = "Wire cones at the -6 dB coverage angles of the selected band, parented to each speaker"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        st = scene.ad_settings
        for old in [o for o in bpy.data.objects if o.name.startswith(PREFIX_CONE)]:
            me = old.data
            bpy.data.objects.remove(old, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)

        col = get_collection(scene, "Visualization")
        band_idx = [str(b) for b in (63, 125, 250, 500, 1000, 2000, 4000, 8000)].index(st.cone_band)
        length = st.cone_length
        count = 0
        for sp in [o for o in scene.objects if o.ad.kind == "SPEAKER"]:
            model = state.LIBRARY.get(sp.ad.model_id)
            if model is None or model.acoustic is None:
                continue
            h = min(model.acoustic.horizontal.coverage_angle(band_idx), MAX_CONE_HALF_ANGLE_DEG)
            v = min(model.acoustic.vertical.coverage_angle(band_idx), MAX_CONE_HALF_ANGLE_DEG)
            ox, y0, oz = model.acoustic_origin_m
            steps = 48
            verts = [(ox, y0, oz)]
            for j in range(steps):
                a = 2.0 * math.pi * j / steps
                verts.append((ox + length * math.tan(math.radians(h)) * math.cos(a), y0 + length,
                              oz + length * math.tan(math.radians(v)) * math.sin(a)))
            faces = [(0, j + 1, (j + 1) % steps + 1) for j in range(steps)]
            me = bpy.data.meshes.new(PREFIX_CONE + sp.name)
            me.from_pydata(verts, [], faces)
            me.update()
            cone = bpy.data.objects.new(PREFIX_CONE + sp.name, me)
            col.objects.link(cone)
            cone.parent = sp
            cone.matrix_parent_inverse = Matrix.Identity(4)
            cone.display_type = "WIRE"
            cone.hide_select = True
            cone.show_in_front = True
            count += 1
        self.report({"INFO"}, f"{count} cone(s) built. They follow speaker movement; rebuild after changing band or length.")
        return {"FINISHED"}


classes = (AD_OT_reload_library, AD_OT_place_speaker, AD_OT_add_receiver, AD_OT_receiver_from_mesh,
           AD_OT_add_probe, AD_OT_calculate, AD_OT_assign_surface, AD_OT_clear_surface, AD_OT_new_speaker,
           AD_OT_clear_heatmap, AD_OT_update_cones)

import bpy

from ..engine import (
    A_WEIGHTING_DB, BANDS, FLOOR_DB, db_sum, evaluate, predict, probe_response, source_contributions,
)
from . import arrays, environment, scene_io, state, volume
from .util import find_root

QUALITY_ICONS = {"visual_only": "MESH_CUBE", "basic_spec": "INFO",
                 "simulation_ready": "CHECKMARK", "manufacturer_validated": "CHECKMARK"}
QUALITY_LABELS = {"visual_only": "Visual only", "basic_spec": "Basic spec",
                  "simulation_ready": "Simulation-ready", "manufacturer_validated": "Mfr-validated"}
PREDICTION_NOTE = "Free-field estimate. Not validated."


class AD_UL_library(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.name)
        text = QUALITY_LABELS.get(item.quality, item.quality) + ("*" if item.approximation else "")
        row.label(text=text, icon=QUALITY_ICONS.get(item.quality, "QUESTION"))

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        wm = context.window_manager
        manufacturer, category = wm.ad_filter_manufacturer, wm.ad_filter_category
        needle = wm.ad_filter_search.lower()
        flags = []
        for it in items:
            ok = ((manufacturer == "ALL" or it.manufacturer == manufacturer)
                  and (category == "ALL" or it.category == category)
                  and (not needle or needle in it.name.lower()))
            flags.append(self.bitflag_filter_item if ok else 0)
        return flags, []


class _Panel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Audio Designer"


def _lines(layout, text):
    for line in text.splitlines():
        layout.label(text=line)


class AD_PT_library(_Panel, bpy.types.Panel):
    bl_label = "Speaker Library"

    def draw(self, context):
        layout, wm = self.layout, context.window_manager
        row = layout.row(align=True)
        row.prop(context.scene.ad_settings, "audience_dir", text="Audience")
        row.operator("audio_designer.aim_audience", text="", icon="ORIENTATION_CURSOR")
        row = layout.row(align=True)
        row.prop(wm, "ad_filter_manufacturer", text="")
        row.prop(wm, "ad_filter_category", text="")
        row.operator("audio_designer.reload_library", text="", icon="FILE_REFRESH")
        layout.prop(wm, "ad_filter_search", text="", icon="VIEWZOOM")
        layout.template_list("AD_UL_library", "", wm, "ad_library_items", wm, "ad_library_index", rows=5)

        if 0 <= wm.ad_library_index < len(wm.ad_library_items):
            model = state.LIBRARY.get(wm.ad_library_items[wm.ad_library_index].model_id)
            if model:
                box = layout.box()
                box.label(text=model.display_name)
                box.label(text=f"Data: {model.quality_label}", icon=QUALITY_ICONS[model.data_quality.value])
                box.label(text=f"{model.width_m:.2f} x {model.height_m:.2f} x {model.depth_m:.2f} m, {model.weight_kg:.1f} kg")
                box.label(text=f"Source: {model.source.get('name', '?')}")
        col = layout.column(align=True)
        col.operator("audio_designer.place_speaker", icon="ADD")
        col.operator("audio_designer.add_line_array", icon="OUTLINER_OB_LATTICE")
        col.operator("audio_designer.add_sub_array", icon="MESH_GRID")
        layout.operator("audio_designer.new_speaker", icon="TEXT")
        if state.LIBRARY.errors:
            layout.label(text=f"{len(state.LIBRARY.errors)} library file(s) rejected (see console)", icon="ERROR")


class AD_PT_speaker(_Panel, bpy.types.Panel):
    bl_label = "Selected Speaker"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.ad.kind == "SPEAKER"

    def draw(self, context):
        ad = context.object.ad
        model = state.LIBRARY.get(ad.model_id)
        layout = self.layout
        layout.label(text=model.display_name if model else f"Unknown model: {ad.model_id}",
                     icon="SPEAKER" if model else "ERROR")
        if model:
            layout.label(text=f"Data: {model.quality_label}", icon=QUALITY_ICONS[model.data_quality.value])
        layout.prop(ad, "level_db")
        layout.prop(ad, "delay_ms")
        row = layout.row()
        row.prop(ad, "polarity_inverted")
        row.prop(ad, "mute")


class AD_PT_line_array(_Panel, bpy.types.Panel):
    bl_label = "Line Array"

    @classmethod
    def poll(cls, context):
        return find_root(context.object, "LINE_ARRAY") is not None

    def draw(self, context):
        layout = self.layout
        root = find_root(context.object, "LINE_ARRAY")
        ad = root.ad
        layout.label(text=root.name, icon="OUTLINER_OB_LATTICE")
        col = layout.column(align=True)
        col.prop(ad, "aim_near_m")
        col.prop(ad, "aim_far_m")
        col.prop(ad, "audience_z")
        layout.operator("audio_designer.auto_splay", icon="AUTO")

        layout.prop(ad, "top_tilt_deg")
        layout.label(text="Splay below each cabinet (deg)")
        cabs = arrays.array_cabinets(root)
        layout_geo, _ = arrays.line_layout_for(root)
        aim = arrays.line_aim_distances(root, context)
        box = layout.box()
        for i, cab in enumerate(cabs):
            row = box.row(align=True)
            sel = "RADIOBUT_ON" if cab == context.object else "RADIOBUT_OFF"
            row.label(text=f"{i + 1}", icon=sel)
            if i < len(cabs) - 1:
                row.prop(ad, "splays", index=i, text="")
            else:
                row.label(text="")
            tilt = f"{layout_geo.tilts_deg[i]:.1f}°" if layout_geo is not None else ""
            far = "" if aim is None or aim[i] == float("inf") else f"  aims {aim[i]:.0f} m"
            row.label(text=tilt + far)
        row = layout.row(align=True)
        row.operator("audio_designer.array_add_cabinet", icon="ADD")
        row.operator("audio_designer.array_remove_cabinet", icon="REMOVE")
        layout.operator("audio_designer.array_report", icon="GRAPH")
        if ad.report:
            _lines(layout.box(), ad.report)


class AD_PT_sub_array(_Panel, bpy.types.Panel):
    bl_label = "Subwoofer Array"

    @classmethod
    def poll(cls, context):
        return find_root(context.object, "SUB_ARRAY") is not None

    def draw(self, context):
        layout = self.layout
        root = find_root(context.object, "SUB_ARRAY")
        ad = root.ad
        layout.label(text=root.name, icon="MESH_GRID")
        layout.prop(ad, "sub_kind")
        layout.prop(ad, "sub_count")
        layout.prop(ad, "sub_spacing_m")
        layout.operator("audio_designer.rebuild_sub_array", icon="FILE_REFRESH")
        layout.label(text="Delay/polarity of each sub: select it and edit under Selected Speaker.")
        layout.separator()
        layout.prop(ad, "analysis_distance_m")
        layout.operator("audio_designer.sub_report", icon="GRAPH")
        if ad.report:
            _lines(layout.box(), ad.report)
        layout.label(text="Heatmap tip: Calculation = 1/3 octave, Map = Single band, 40-100 Hz.")


class AD_PT_receivers(_Panel, bpy.types.Panel):
    bl_label = "Venue and Audience"

    def draw(self, context):
        layout = self.layout
        layout.operator("audio_designer.add_receiver", icon="MESH_GRID")
        layout.operator("audio_designer.receiver_from_mesh", icon="MOD_MESHDEFORM")
        layout.operator("audio_designer.auto_receivers", icon="AUTO")
        ob = context.object
        if ob is not None and ob.ad.kind == "RECEIVER":
            layout.prop(ob.ad, "listen_mode")
        for ob in context.scene.objects:
            if ob.ad.kind == "RECEIVER" and ob.ad.stat_metric:
                box = layout.box()
                box.label(text=f"{ob.name} ({ob.ad.stat_metric})")
                box.label(text=f"min {ob.ad.stat_min:.1f}  max {ob.ad.stat_max:.1f}  mean {ob.ad.stat_mean:.1f}")
                box.label(text=f"spread {ob.ad.stat_spread:.1f} dB")


class AD_PT_surfaces(_Panel, bpy.types.Panel):
    bl_label = "Surfaces"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout, st, wm = self.layout, context.scene.ad_settings, context.window_manager
        layout.prop(st, "use_reflections")
        if st.use_reflections:
            col = layout.column(align=True)
            col.prop(st, "reflection_order", text="")
            if st.reflection_order == "2":
                col.prop(st, "order2_limit")
            col.prop(st, "reflector_min_area_m2")
        box = layout.box()
        box.prop(st, "ground_enabled")
        if st.ground_enabled:
            box.prop(st, "ground_z")
            box.prop(st, "ground_surface_id", text="")
        layout.prop(st, "use_occlusion")
        if st.use_occlusion:
            col = layout.column(align=True)
            col.prop(st, "shadow_scale")
            col.prop(st, "occluder_source", text="")
        layout.operator("audio_designer.auto_surfaces", icon="AUTO")
        layout.separator()
        layout.prop(wm, "ad_surface_choice", text="")
        row = layout.row(align=True)
        row.operator("audio_designer.assign_surface", icon="MATERIAL")
        row.operator("audio_designer.clear_surface", text="", icon="X")

        ob = context.object
        if ob is not None and ob.type == "MESH" and ob.ad.kind == "NONE":
            ad = ob.ad
            box = layout.box()
            material = state.LIBRARY.surface(ad.surface_id) if ad.surface_id else None
            box.label(text=f"{ob.name}: {material.name if material else 'no surface'}", icon="MESH_DATA")
            box.prop(ad, "surface_custom")
            if ad.surface_auto and not ad.surface_id:
                box.label(text="Auto: surface per face from material name and orientation", icon="AUTO")
            if ad.surface_custom or material:
                col = box.column(align=True)
                col.enabled = ad.surface_custom
                for i, b in enumerate(BANDS):
                    col.prop(ad, "surface_absorption", index=i, text=f"{b} Hz")
                col.prop(ad, "surface_scattering")
            if material:
                box.label(text=f"Source: {material.source.get('name', '?')[:48]}")
        layout.label(text="Specular reflections and shadowing only. No diffraction, no diffuse field.", icon="INFO")


class AD_PT_room(_Panel, bpy.types.Panel):
    bl_label = "Room Acoustics"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout, st = self.layout, context.scene.ad_settings
        col = layout.column(align=True)
        col.prop(st, "room_volume_m3")
        col.prop(st, "room_fill")
        layout.prop(st, "noise_dba")
        layout.operator("audio_designer.room_acoustics", icon="GRAPH")
        if st.room_info and st.room_rt_sabine[4] > 0:
            box = layout.box()
            box.label(text=st.room_info)
            grid = box.grid_flow(row_major=True, columns=3, align=True)
            grid.label(text="Band")
            grid.label(text="Sabine")
            grid.label(text="Eyring")
            for i, b in enumerate(BANDS):
                grid.label(text=f"{b} Hz")
                grid.label(text=f"{st.room_rt_sabine[i]:.2f} s")
                grid.label(text=f"{st.room_rt_eyring[i]:.2f} s")
        layout.label(text="Set Map to STI (estimate) under Acoustic Simulation for a speech map.")
        layout.label(text="STI: direct + diffuse tail + noise. No masking; not IEC-compliant.", icon="INFO")


class AD_PT_simulation(_Panel, bpy.types.Panel):
    bl_label = "Acoustic Simulation"

    def draw(self, context):
        layout, st = self.layout, context.scene.ad_settings
        box = layout.box()
        box.label(text="Atmosphere")
        box.prop(st, "temperature_c")
        box.prop(st, "humidity_pct")
        box.prop(st, "pressure_kpa")
        box.prop(st, "air_absorption")
        box.label(text=f"Speed of sound {scene_io.prediction_settings(st).c:.1f} m/s")

        layout.prop(st, "mode")
        layout.prop(st, "metric")
        if st.metric == "BAND":
            layout.prop(st, "band" if st.mode == "OCTAVE" else "third")
        if st.metric == "STI":
            layout.label(text="Needs surfaces. Uses noise level and room settings from Room Acoustics.", icon="INFO")
        if st.mode == "COHERENT":
            layout.label(text="Point-source cabinets: treat results above ~1.5 kHz as indicative.", icon="INFO")
        layout.prop(st, "listen_height")
        layout.prop(st, "use_reflections")
        layout.prop(st, "auto_range")
        row = layout.row(align=True)
        row.enabled = not st.auto_range
        row.prop(st, "min_db")
        row.prop(st, "max_db")
        layout.operator("audio_designer.calculate", icon="PLAY")
        layout.operator("audio_designer.clear_heatmap", icon="X")
        layout.label(text=PREDICTION_NOTE, icon="INFO")
        layout.separator()
        layout.prop(st, "cone_band")
        layout.prop(st, "cone_length")
        layout.operator("audio_designer.update_cones", icon="CONE")


class AD_PT_volume(_Panel, bpy.types.Panel):
    bl_label = "Volumetric Field"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout, st = self.layout, context.scene.ad_settings
        row = layout.row(align=True)
        row.operator("audio_designer.add_volume_region", icon="CUBE")
        row.operator("audio_designer.add_slice_focus", icon="EMPTY_AXIS")
        col = layout.column(align=True)
        col.prop(st, "vol_cell_m")
        col.prop(st, "vol_range_db")
        layout.operator("audio_designer.calculate_volume", icon="PLAY")
        layout.operator("audio_designer.clear_volume", icon="X")
        if volume.overlay_info():
            layout.label(text=volume.overlay_info())
        layout.separator()
        layout.prop(st, "vol_display", expand=True)
        col = layout.column(align=True)
        col.prop(st, "vol_opacity")
        col.prop(st, "vol_threshold")
        col.prop(st, "vol_blend")
        if st.vol_display in ("SLICES", "BOTH"):
            box = layout.box()
            box.prop(st, "slice_count")
            box.prop(st, "slice_axes", text="Axes")
            box.prop(st, "slice_sigma")
            box.prop(st, "slice_floor")
            box.label(text="Move the Slice Focus empty (snap it to surfaces) to light up planes.")
        layout.label(text="Uses the Calculation and Map chosen under Acoustic Simulation.")
        layout.label(text="Viewport overlay; not a renderable volume.", icon="INFO")


class AD_PT_probes(_Panel, bpy.types.Panel):
    bl_label = "Audio Probes"

    def draw(self, context):
        layout, scene = self.layout, context.scene
        layout.operator("audio_designer.add_probe", icon="EMPTY_AXIS")
        probes = [o for o in scene.objects if o.ad.kind == "PROBE"]
        if not probes:
            return
        sources, _ = scene_io.collect_sources(scene, state.LIBRARY)
        active = [s for s in sources if s.enabled]
        if not active:
            layout.label(text="No active speakers.", icon="INFO")
            return
        st = scene.ad_settings
        settings = scene_io.prediction_settings(st)
        env, _notes = environment.active_environment(scene, state.LIBRARY, live=True)
        coherent = st.mode == "COHERENT"
        for probe in probes:
            pt = list(probe.matrix_world.translation)
            levels, dist, arrival, used = source_contributions(active, [pt], settings)
            if coherent:
                a_val = evaluate(active, [pt], settings, "COHERENT", "A", env=env)[0][0]
                c_val = evaluate(active, [pt], settings, "COHERENT", "C", env=env)[0][0]
                bb_val = evaluate(active, [pt], settings, "COHERENT", "BROADBAND", env=env)[0][0]
                band_vals = probe_response(active, pt, BANDS, settings, env).total_spl
            else:
                combined = predict(active, [pt], settings, env)
                a_val, c_val, bb_val = combined.a_weighted[0], combined.c_weighted[0], combined.broadband[0]
                band_vals = combined.band_spl[0]
            box = layout.box()
            head = box.row()
            head.label(text=probe.name, icon="RESTRICT_SELECT_OFF" if probe == context.object else "EMPTY_AXIS")
            head.label(text=f"{a_val:.1f} dB(A)")
            if probe != context.object:
                continue
            box.label(text=f"C {c_val:.1f} dB(C)   Broadband {bb_val:.1f} dB" + ("   (phase-aware)" if coherent else ""))
            grid = box.grid_flow(row_major=True, columns=2, align=True)
            for i, b in enumerate(BANDS):
                grid.label(text=f"{b} Hz: {band_vals[i]:.1f} dB")
            box.separator()
            for i, src in enumerate(used):
                a = db_sum(levels[i, 0] + A_WEIGHTING_DB)
                row = box.row()
                row.label(text=src.name)
                row.label(text=f"{dist[i, 0]:.1f} m  {arrival[i, 0]:.1f} ms  {max(a, FLOOR_DB):.1f} dB(A)")
            box.label(text=PREDICTION_NOTE, icon="INFO")


class AD_PT_export(_Panel, bpy.types.Panel):
    bl_label = "Report"

    def draw(self, context):
        layout = self.layout
        layout.operator("audio_designer.export_report", icon="EXPORT")
        layout.label(text="One PDF: speakers and angles, arrays, room acoustics,")
        layout.label(text="receiver and fog images, probes, surfaces, data sources.")


classes = (AD_UL_library, AD_PT_library, AD_PT_speaker, AD_PT_line_array, AD_PT_sub_array, AD_PT_receivers,
           AD_PT_surfaces, AD_PT_room, AD_PT_simulation, AD_PT_volume, AD_PT_probes, AD_PT_export)

"""Gathers everything the add-on knows about the scene into one readable PDF report."""
import math
from datetime import datetime

import bpy
import numpy as np
from bpy.props import StringProperty
from mathutils import Vector

from ..engine import (
    A_WEIGHTING_DB, BANDS, FLOOR_DB, THIRD_OCTAVES, db_sum, evaluate, predict, probe_response, source_contributions,
)
from ..engine.arrays import array_summary
from ..engine.render import render_fog, render_receivers_top
from ..engine.report import build_report
from . import arrays, environment, scene_io, state, volume

DISCLAIMER_NOTES = [
    "All levels are free-field point-source estimates from the figures in the speaker library; they are not "
    "measurements and are not validated.",
    "Entries marked 'approximation' use published figures for size, range and nominal coverage; the polar shape is "
    "modelled, not measured.",
    "Line array geometry, weight and centre of gravity are not rigging data. Use the manufacturer's own tool for "
    "rigging and safety calculations.",
    "STI values are an estimate (no auditory masking, flat spectrum) and are not IEC 60268-16 results.",
]


def _speakers(scene):
    rows = []
    for ob in scene.objects:
        if ob.ad.kind != "SPEAKER":
            continue
        model = state.LIBRARY.get(ob.ad.model_id)
        f = ob.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
        p = ob.matrix_world.translation
        rows.append({
            "name": ob.name, "model": model.display_name if model else f"unknown ({ob.ad.model_id})",
            "quality": model.data_quality.label if model else "", "x": p.x, "y": p.y, "z": p.z,
            "azimuth": math.degrees(math.atan2(f.x, f.y)), "tilt": math.degrees(math.asin(max(-1.0, min(1.0, f.z)))),
            "level": ob.ad.level_db, "delay": ob.ad.delay_ms, "inverted": ob.ad.polarity_inverted, "mute": ob.ad.mute,
            "group": ob.parent.name if ob.parent else "",
        })
    return rows


def _array_sections(context):
    scene = context.scene
    out = []
    for root in scene.objects:
        if root.ad.kind == "LINE_ARRAY":
            layout, model = arrays.line_layout_for(root)
            if layout is None:
                continue
            s = array_summary(layout, model.weight_kg)
            aim = arrays.line_aim_distances(root, context)
            cabs = arrays.array_cabinets(root)
            lines = [f"{model.display_name}: {s['cabinets']} cabinets, {s['total_weight_kg']:.0f} kg nominal (cabinets only)",
                     f"Height {s['height_m']:.2f} m, depth {s['depth_m']:.2f} m, total splay {s['total_splay_deg']:.1f} deg",
                     f"Audience: {root.ad.aim_near_m:g} m to {root.ad.aim_far_m:g} m, floor Z {root.ad.audience_z:g} m",
                     "Geometry only; not rigging data."]
            lines += [ln for ln in root.ad.report.splitlines() if ln.startswith("dB(A)")]
            rows = []
            for i, cab in enumerate(cabs):
                far = "" if aim is None or not math.isfinite(aim[i]) else f"{aim[i]:.1f}"
                rows.append([i + 1, f"{layout.tilts_deg[i]:.1f}", f"{root.ad.splays[i - 1]:.1f}" if i else "-", far,
                             f"{cab.ad.level_db:.1f}", f"{cab.ad.delay_ms:.2f}"])
            out.append({"title": f"Line array: {root.name}", "lines": lines,
                        "table": {"headers": ["Cab", "Tilt deg", "Splay deg", "Aims at m", "Level dB", "Delay ms"],
                                  "rows": rows}})
        elif root.ad.kind == "SUB_ARRAY":
            model = state.LIBRARY.get(root.ad.model_id)
            units = arrays.array_cabinets(root)
            kinds = {"BROADSIDE": "ground stack", "ENDFIRE": "end-fire", "CARDIOID": "cardioid", "CUSTOM": "custom"}
            lines = [f"{model.display_name if model else root.ad.model_id}: {len(units)} units, "
                     f"{kinds.get(root.ad.sub_kind, root.ad.sub_kind)}, spacing {root.ad.sub_spacing_m:g} m"]
            lines += root.ad.report.splitlines()
            rows = []
            for i, u in enumerate(units):
                p = u.matrix_world.translation
                rows.append([i + 1, f"{p.x:.2f}", f"{p.y:.2f}", f"{p.z:.2f}",
                             f"{math.degrees(u.matrix_world.to_euler().z):.0f}", f"{u.ad.delay_ms:.2f}",
                             "inverted" if u.ad.polarity_inverted else "normal", f"{u.ad.level_db:.1f}"])
            out.append({"title": f"Subwoofer array: {root.name}", "lines": lines,
                        "table": {"headers": ["Unit", "X", "Y", "Z", "Yaw deg", "Delay ms", "Polarity", "Level dB"],
                                  "rows": rows}})
    return out


def _room(st):
    if not st.room_info or st.room_rt_sabine[4] <= 0:
        return None
    return {"info": st.room_info, "noise": st.noise_dba,
            "rows": [[f"{b} Hz", f"{st.room_rt_sabine[i]:.2f} s", f"{st.room_rt_eyring[i]:.2f} s"]
                     for i, b in enumerate(BANDS)]}


def _markers(scene):
    marks = [(ob.matrix_world.translation, ob.name) for ob in scene.objects if ob.ad.kind in ("SPEAKER", "PROBE")]
    return marks


def _receiver_image(ob, markers):
    me = ob.data
    attr = me.color_attributes.get(scene_io.AD_COLOR_ATTR)
    if attr is None or not len(me.polygons):
        return None
    totals = np.array([len(p.vertices) for p in me.polygons], dtype=int)
    starts = np.concatenate([[0], np.cumsum(totals)[:-1]])
    colour = np.empty(len(me.loops) * 4, dtype=np.float32)
    attr.data.foreach_get("color", colour)
    rgb = np.clip(colour.reshape(-1, 4)[starts][:, :3] * 255.0, 0, 255).astype(np.uint8)
    pts = scene_io.face_points(ob, 0.0)
    areas = np.empty(len(me.polygons), dtype=np.float32)
    me.polygons.foreach_get("area", areas)
    cell = max(float(np.sqrt(np.median(areas)) * np.mean(np.abs(ob.matrix_world.to_scale()))), 0.05)
    image = render_receivers_top(pts, rgb, cell, 900)
    return image, [(p.x, p.y, name) for p, name in markers]


def _receivers(scene, markers):
    out = []
    for ob in scene.objects:
        if ob.ad.kind != "RECEIVER" or not ob.ad.stat_metric:
            continue
        item = {"name": ob.name, "metric": ob.ad.stat_metric, "min": ob.ad.stat_min, "max": ob.ad.stat_max,
                "mean": ob.ad.stat_mean, "spread": ob.ad.stat_spread, "image": None}
        rendered = _receiver_image(ob, markers)
        if rendered is not None:
            item["image"], item["markers"] = rendered
            item["legend"] = (ob.ad.stat_lo, ob.ad.stat_hi, ob.ad.stat_metric)
        out.append(item)
    return out


def _fog(scene, markers):
    snap = volume.field_snapshot()
    if snap is None:
        return None
    st = scene.ad_settings
    dx, dy, dz = snap["dims"]
    inv = snap["matrix"].inverted()
    top, side = [], []
    for p, name in markers:
        lx, ly, lz = inv @ p
        if max(abs(lx), abs(ly), abs(lz)) <= 1.05:
            top.append((lx * dx / 2, ly * dy / 2, name))
            side.append((ly * dy / 2, lz * dz / 2, name))
    return {
        "info": volume.overlay_info() or snap["label"],
        "legend": (snap["lo"], snap["hi"], snap["label"]),
        "top": render_fog(snap["t"], "top", snap["dims"], st.vol_opacity, st.vol_threshold),
        "side": render_fog(snap["t"], "side", snap["dims"], st.vol_opacity, st.vol_threshold),
        "top_markers": top, "side_markers": side,
    }


def _probes(scene):
    st = scene.ad_settings
    sources, _ = scene_io.collect_sources(scene, state.LIBRARY)
    active = [s for s in sources if s.enabled]
    probes = [o for o in scene.objects if o.ad.kind == "PROBE"]
    if not probes or not active:
        return []
    settings = scene_io.prediction_settings(st)
    env, _notes = environment.active_environment(scene, state.LIBRARY)
    coherent = st.mode == "COHERENT"
    method = ("Phase-aware 1/3 octave" if coherent else "Octave bands, power sum")
    if env.reflectors:
        method += f"; reflections order {env.reflection_order} ({len(env.reflectors)} triangles)"
    if env.occluder is not None:
        method += "; occlusion"
    out = []
    for probe in probes:
        pt = list(probe.matrix_world.translation)
        levels, dist, arrival, used = source_contributions(active, [pt], settings)
        if coherent:
            a = evaluate(active, [pt], settings, "COHERENT", "A", env=env)[0][0]
            c = evaluate(active, [pt], settings, "COHERENT", "C", env=env)[0][0]
            bb = evaluate(active, [pt], settings, "COHERENT", "BROADBAND", env=env)[0][0]
            bands = probe_response(active, pt, BANDS, settings, env).total_spl
        else:
            res = predict(active, [pt], settings, env)
            a, c, bb, bands = res.a_weighted[0], res.c_weighted[0], res.broadband[0], res.band_spl[0]
        resp = probe_response(active, pt, THIRD_OCTAVES, settings, env)
        out.append({
            "name": probe.name, "x": pt[0], "y": pt[1], "z": pt[2], "a": float(a), "c": float(c), "bb": float(bb),
            "method": method, "band_names": list(BANDS), "bands": [float(v) for v in bands],
            "speakers": [{"name": s.name, "dist": float(dist[i, 0]), "arrival": float(arrival[i, 0]),
                          "dba": float(max(db_sum(levels[i, 0] + A_WEIGHTING_DB), FLOOR_DB))}
                         for i, s in enumerate(used)],
            "response": (resp.freqs, resp.total_spl),
        })
    return out


def _surfaces(scene):
    rows = []
    st = scene.ad_settings
    if st.use_reflections and st.ground_enabled:
        g = state.LIBRARY.surface(st.ground_surface_id)
        if g:
            rows.append(["Ground plane", f"{g.name} (Z {st.ground_z:g} m)", _abs(g.absorption)])
    for ob in scene.objects:
        if not environment.has_surface(ob):
            continue
        ad = ob.ad
        if ad.surface_custom:
            rows.append([ob.name, "Custom", _abs(ad.surface_absorption)])
        elif ad.surface_id and not ad.surface_auto:
            m = state.LIBRARY.surface(ad.surface_id)
            rows.append([ob.name, m.name if m else ad.surface_id, _abs(m.absorption) if m else ""])
        else:
            rows.append([ob.name, "Automatic, per face", "from material names and orientation"])
    return rows


def _abs(values):
    return "  ".join(f"{float(v):.2f}" for v in values)


def _sources(scene):
    seen, out = set(), []
    for ob in scene.objects:
        if ob.ad.kind != "SPEAKER" or ob.ad.model_id in seen:
            continue
        seen.add(ob.ad.model_id)
        m = state.LIBRARY.get(ob.ad.model_id)
        if m:
            out.append({"model": m.display_name, "quality": m.quality_label, "source": m.source.get("name", ""),
                        "url": m.source.get("url", ""), "notes": m.source.get("notes", "")})
    return out


def _settings(scene, speakers, receivers, probes):
    st = scene.ad_settings
    s = scene_io.prediction_settings(st)
    rows = [
        ("Temperature / humidity / pressure", f"{st.temperature_c:g} C, {st.humidity_pct:g} %, {st.pressure_kpa:g} kPa"),
        ("Speed of sound", f"{s.c:.1f} m/s"),
        ("Air absorption", "on" if st.air_absorption else "off"),
        ("Calculation", "Phase-aware 1/3 octave" if st.mode == "COHERENT" else "Octave bands, power sum"),
        ("Map", scene_io.metric_label(st)),
        ("Listening height", f"{st.listen_height:g} m"),
        ("Reflections", (f"order {st.reflection_order}" if st.use_reflections else "off")
         + (f", ground at Z {st.ground_z:g} m" if st.use_reflections and st.ground_enabled else "")),
        ("Occlusion", f"on, shadow strength {st.shadow_scale:g}" if st.use_occlusion else "off"),
        ("Contents", f"{len(speakers)} speakers, {len(receivers)} calculated receiver areas, {len(probes)} probes"),
    ]
    return rows


class AD_OT_export_report(bpy.types.Operator):
    bl_idname = "audio_designer.export_report"
    bl_label = "Export Report (PDF)"
    bl_description = ("One readable PDF with the speakers and their angles, arrays, room acoustics, receiver and fog "
                      "images, audio probes, surfaces and data sources")

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.pdf", options={"HIDDEN"})

    def invoke(self, context, event):
        if not self.filepath:
            stem = bpy.path.display_name_from_filepath(bpy.data.filepath) if bpy.data.filepath else "audio"
            self.filepath = f"{stem}_report.pdf"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        scene = context.scene
        speakers = _speakers(scene)
        if not speakers:
            self.report({"ERROR"}, "No speakers in the scene.")
            return {"CANCELLED"}
        markers = _markers(scene)
        receivers = _receivers(scene, markers)
        probes = _probes(scene)
        data = {
            "title": "Audio system report",
            "scene": f"Scene {scene.name}" + (f"   {bpy.path.basename(bpy.data.filepath)}" if bpy.data.filepath else ""),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "settings": _settings(scene, speakers, receivers, probes),
            "speakers": speakers,
            "arrays": _array_sections(context),
            "room": _room(scene.ad_settings),
            "receivers": receivers,
            "fog": _fog(scene, markers),
            "probes": probes,
            "surfaces": _surfaces(scene),
            "sources": _sources(scene),
            "notes": DISCLAIMER_NOTES,
        }
        path = bpy.path.abspath(self.filepath)
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            pdf = build_report(data)
            with open(path, "wb") as f:
                f.write(pdf)
        except OSError as e:
            self.report({"ERROR"}, f"Could not write the report: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Report saved: {path}")
        return {"FINISHED"}


classes = (AD_OT_export_report,)

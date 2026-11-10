"""Automatic receiver areas, automatic surfaces, speaker aiming and room acoustics operators."""
import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty

from ..engine.geometry import subdivide_triangles, triangle_areas_normals, walkable_islands
from . import environment, scene_io, state
from .ad_collections import get_collection, new_object
from .operators import MAX_RECEIVER_CELLS
from .util import select_only

ARRAY_KINDS = ("LINE_ARRAY", "SUB_ARRAY")
MAX_RAY_REDIRECTS = 6


def _candidate_meshes(context):
    chosen = [o for o in context.selected_objects if o.type == "MESH" and o.ad.kind == "NONE"]
    if chosen:
        return chosen
    return [o for o in context.scene.objects if o.type == "MESH" and o.ad.kind == "NONE" and environment.is_visible(o)]


def _roof_above(context, point) -> bool:
    """True when something in the scene (not a helper object) lies above the point."""
    depsgraph = context.evaluated_depsgraph_get()
    origin = (point[0], point[1], point[2] + 0.05)
    for _ in range(MAX_RAY_REDIRECTS):
        hit, loc, _n, _i, ob, _m = context.scene.ray_cast(depsgraph, origin, (0.0, 0.0, 1.0))
        if not hit:
            return False
        if ob is not None and getattr(ob, "ad", None) is not None and ob.ad.kind == "NONE":
            return True
        origin = (loc[0], loc[1], loc[2] + 0.02)
    return False


class AD_OT_auto_receivers(bpy.types.Operator):
    bl_idname = "audio_designer.auto_receivers"
    bl_label = "Auto Receiver From Scene"
    bl_description = ("Find the walkable, upward-facing surfaces of the selected meshes (or all visible meshes) and "
                      "build a receiver from them. Steps and raked seating work because connected faces are grouped "
                      "into islands before small details are discarded")
    bl_options = {"REGISTER", "UNDO"}

    cell: FloatProperty(name="Cell size", default=0.5, min=0.05, max=10.0, unit="LENGTH")
    max_slope_deg: FloatProperty(name="Max slope", description="Steeper faces are not walkable", default=45.0,
                                 min=0.0, max=89.0)
    min_island_m2: FloatProperty(name="Ignore areas smaller than (m\u00b2)", description="Seat tops, props and trim",
                                 default=1.0, min=0.0, unit="NONE")
    require_roof: BoolProperty(name="Only under a roof", description="Skip surfaces with nothing above them "
                               "(outdoor tops of walls and roofs)", default=True)

    def execute(self, context):
        meshes = _candidate_meshes(context)
        if not meshes:
            self.report({"ERROR"}, "No meshes to analyse.")
            return {"CANCELLED"}
        tris = [environment.object_triangles(o)[0] for o in meshes]
        tris = np.concatenate([t for t in tris if len(t)]) if any(len(t) for t in tris) else np.empty((0, 3, 3))
        ids, island_area = walkable_islands(tris, self.max_slope_deg)
        if not len(island_area):
            self.report({"ERROR"}, "No walkable surfaces found. Check the slope limit and that faces point up.")
            return {"CANCELLED"}

        areas, _ = triangle_areas_normals(tris)
        kept_islands, too_small, no_roof = [], 0, 0
        for island in range(len(island_area)):
            if island_area[island] < self.min_island_m2:
                too_small += 1
                continue
            if self.require_roof:
                members = np.nonzero(ids == island)[0]
                rep = members[np.argmax(areas[members])]
                if not _roof_above(context, tris[rep].mean(axis=0)):
                    no_roof += 1
                    continue
            kept_islands.append(island)
        if not kept_islands:
            self.report({"ERROR"}, f"Nothing left: {too_small} area(s) too small, {no_roof} without a roof. "
                                   "Lower the minimum area or switch off 'Only under a roof'.")
            return {"CANCELLED"}

        selected = tris[np.isin(ids, kept_islands)]
        cells = subdivide_triangles(selected, self.cell)
        if len(cells) > MAX_RECEIVER_CELLS:
            self.report({"ERROR"}, f"{len(cells)} cells (limit {MAX_RECEIVER_CELLS}). Increase the cell size.")
            return {"CANCELLED"}

        verts = cells.reshape(-1, 3)
        me = bpy.data.meshes.new("AD_AutoReceiver")
        me.from_pydata(verts.tolist(), [], [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(cells))])
        me.update()
        ob = new_object(context.scene, get_collection(context.scene, "Receivers"), "Audio Receiver (auto)", me)
        ob.location = (0.0, 0.0, 0.0)
        ob.ad.kind = "RECEIVER"
        select_only(context, ob)
        self.report({"INFO"}, f"{len(kept_islands)} surface(s), {len(cells)} cells. Ignored: {too_small} too small, "
                              f"{no_roof} without a roof.")
        return {"FINISHED"}


class AD_OT_auto_surfaces(bpy.types.Operator):
    bl_idname = "audio_designer.auto_surfaces"
    bl_label = "Auto-assign Surfaces"
    bl_description = ("Mark meshes so each face gets a surface from its Blender material name (glass, curtain, carpet, "
                      "seat...) or, failing that, from its orientation (floor, ceiling, wall)")
    bl_options = {"REGISTER", "UNDO"}

    only_unassigned: BoolProperty(name="Only meshes without a surface", default=True)

    def execute(self, context):
        count = 0
        for ob in _candidate_meshes(context):
            if self.only_unassigned and has_manual_surface(ob):
                continue
            ob.ad.surface_auto = True
            ob.ad.surface_id = ""
            ob.ad.surface_custom = False
            count += 1
        environment.bump_revision()
        self.report({"INFO"}, f"Auto surfaces on {count} mesh(es).")
        return {"FINISHED"}


def has_manual_surface(ob) -> bool:
    return bool(ob.ad.surface_custom or (ob.ad.surface_id and not ob.ad.surface_auto))


class AD_OT_aim_audience(bpy.types.Operator):
    bl_idname = "audio_designer.aim_audience"
    bl_label = "Aim Selected At Audience"
    bl_description = "Turn the selected speakers and arrays to face the audience direction set in the Speaker Library"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        roots = set()
        for ob in context.selected_objects:
            if ob.ad.kind in ARRAY_KINDS:
                roots.add(ob)
            elif ob.parent is not None and ob.parent.ad.kind in ARRAY_KINDS:
                roots.add(ob.parent)
            elif ob.ad.kind == "SPEAKER":
                roots.add(ob)
        for r in roots:
            r.rotation_mode = "XYZ"
            r.rotation_euler.z = scene_io.audience_yaw(context.scene, r.matrix_world.translation)
        self.report({"INFO"}, f"{len(roots)} object(s) aimed.")
        return {"FINISHED"}


class AD_OT_room_acoustics(bpy.types.Operator):
    bl_idname = "audio_designer.room_acoustics"
    bl_label = "Calculate Room Acoustics"
    bl_description = ("RT60 per octave band (Sabine and Eyring) from the surface-assigned meshes, plus air absorption. "
                      "Volume is the override or the bounding box of the surfaces")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        st = context.scene.ad_settings
        room, message = environment.room_for_scene(context.scene, state.LIBRARY)
        if room is None:
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        st.room_rt_sabine = [float(v) for v in room.rt_sabine]
        st.room_rt_eyring = [float(v) for v in room.rt_eyring]
        st.room_info = message
        self.report({"INFO"}, f"RT60 at 1 kHz: {room.rt_sabine[4]:.2f} s Sabine, {room.rt_eyring[4]:.2f} s Eyring. {message}")
        return {"FINISHED"}


classes = (AD_OT_auto_receivers, AD_OT_auto_surfaces, AD_OT_aim_audience, AD_OT_room_acoustics)

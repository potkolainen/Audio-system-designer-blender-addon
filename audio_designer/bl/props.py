import bpy
from bpy.props import (
    BoolProperty,
    BoolVectorProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from ..engine import BANDS, THIRD_OCTAVES
from ..library import CATEGORIES
from ..library.model import MAX_ARRAY_CABINETS
from . import state

_manufacturer_items_cache = []
_surface_items_cache = []

KIND_ITEMS = [("NONE", "None", ""), ("SPEAKER", "Speaker", ""), ("PROBE", "Probe", ""),
              ("RECEIVER", "Receiver", ""), ("LINE_ARRAY", "Line array", ""),
              ("SUB_ARRAY", "Subwoofer array", ""), ("VOLUME", "Volume region", ""),
              ("FOCUS", "Slice focus", "")]
SUB_KIND_ITEMS = [
    ("BROADSIDE", "Ground stack (broadside)", "Subs side by side, no delay"),
    ("ENDFIRE", "End-fire", "Subs in depth, delayed towards the front"),
    ("CARDIOID", "Cardioid (inverted rear)", "Forward sub plus inverted, delayed rear-facing sub per column"),
    ("CUSTOM", "Custom", "Position and set delay/polarity manually; recipe is not applied"),
]
BAND_ITEMS = [(str(b), f"{b} Hz", "") for b in BANDS]
THIRD_ITEMS = [(f"{f:g}", f"{f:g} Hz", "") for f in THIRD_OCTAVES]


def _manufacturer_items(self, context):
    global _manufacturer_items_cache
    _manufacturer_items_cache = [("ALL", "All manufacturers", "")] + [
        (m, m, "") for m in state.LIBRARY.manufacturers()]
    return _manufacturer_items_cache


def _surface_items(self, context):
    global _surface_items_cache
    _surface_items_cache = [(s.id, s.name, "") for s in sorted(state.LIBRARY.surfaces.values(), key=lambda s: s.name)]
    return _surface_items_cache or [("", "(no surfaces loaded)", "")]


def _relayout(self, context):
    from . import arrays
    arrays.relayout_line_array(self.id_data)


class AD_ObjectProps(bpy.types.PropertyGroup):
    kind: EnumProperty(items=KIND_ITEMS, default="NONE")
    model_id: StringProperty(name="Model ID")
    level_db: FloatProperty(
        name="Level (dB re max)", description="Drive level relative to the model's maximum SPL at 1 m",
        default=-10.0, min=-60.0, max=0.0)
    delay_ms: FloatProperty(
        name="Delay (ms)", description="Signal delay. Shifts phase in coherent mode, arrival time everywhere",
        default=0.0, min=0.0, max=1000.0)
    polarity_inverted: BoolProperty(name="Invert polarity", default=False)
    mute: BoolProperty(name="Mute", default=False)

    stat_metric: StringProperty()
    stat_min: FloatProperty()
    stat_max: FloatProperty()
    stat_mean: FloatProperty()
    stat_spread: FloatProperty()
    stat_height: FloatProperty()
    stat_lo: FloatProperty()
    stat_hi: FloatProperty()

    # Array roots
    array_index: IntProperty(description="Position of this cabinet within its array")
    array_count: IntProperty()
    top_tilt_deg: FloatProperty(name="Top cabinet tilt", description="Downward tilt of the first cabinet",
                                default=0.0, min=-90.0, max=90.0, update=_relayout)
    splays: FloatVectorProperty(name="Splay (deg)", size=MAX_ARRAY_CABINETS, default=[0.0] * MAX_ARRAY_CABINETS,
                                min=-30.0, max=30.0, update=_relayout)
    aim_near_m: FloatProperty(name="Audience near", default=8.0, min=0.5, max=500.0, unit="LENGTH")
    aim_far_m: FloatProperty(name="Audience far", default=35.0, min=1.0, max=1000.0, unit="LENGTH")
    audience_z: FloatProperty(name="Audience floor Z", default=0.0, unit="LENGTH")
    sub_kind: EnumProperty(name="Configuration", items=SUB_KIND_ITEMS, default="CARDIOID")
    sub_count: IntProperty(name="Count", description="Subs (broadside/end-fire) or columns (cardioid)",
                           default=2, min=1, max=16)
    sub_spacing_m: FloatProperty(name="Spacing", default=1.0, min=0.1, max=10.0, unit="LENGTH")
    analysis_distance_m: FloatProperty(name="Analysis radius", default=15.0, min=2.0, max=200.0, unit="LENGTH")
    report: StringProperty(name="Report")

    surface_id: StringProperty(name="Surface", description="Library surface used for reflections")
    surface_custom: BoolProperty(name="Custom absorption", description="Use the values below instead of the library surface")
    surface_absorption: FloatVectorProperty(
        name="Absorption", size=8, min=0.0, max=1.0, default=[0.1] * 8, description="Octave bands 63 Hz to 8 kHz")
    surface_scattering: FloatProperty(name="Scattering", default=0.1, min=0.0, max=1.0,
                                      description="Share of reflected energy that is not specular")
    surface_auto: BoolProperty(
        name="Auto surfaces",
        description="Pick a surface per face from Blender material names (glass, curtain, carpet...) and orientation")
    listen_mode: EnumProperty(
        name="Listening height",
        items=[("AUTO", "Auto", "Raise floor-like faces only; vertical faces stay where they are"),
               ("ALWAYS", "Always", "Raise every face"), ("NEVER", "Never", "Use the faces exactly as they are")],
        default="AUTO")


class AD_SceneSettings(bpy.types.PropertyGroup):
    temperature_c: FloatProperty(name="Temperature (°C)", default=20.0, min=-30.0, max=50.0)
    humidity_pct: FloatProperty(name="Relative humidity (%)", default=50.0, min=0.0, max=100.0)
    pressure_kpa: FloatProperty(name="Air pressure (kPa)", default=101.325, min=70.0, max=110.0)
    air_absorption: BoolProperty(name="Air absorption", default=True)
    listen_height: FloatProperty(name="Listening height", description="Height above each receiver face",
                                 default=1.2, min=0.0, max=10.0, unit="LENGTH")
    mode: EnumProperty(
        name="Calculation",
        items=[("OCTAVE", "Octave bands, power sum", "Incoherent. Robust for wide-band coverage"),
               ("COHERENT", "1/3 octave, phase-aware",
                "Complex summation with delay and polarity. Needed for subwoofer and array interaction")],
        default="OCTAVE")
    metric: EnumProperty(
        name="Map",
        items=[("A", "dB(A)", ""), ("C", "dB(C)", ""), ("BROADBAND", "Broadband (unweighted)", ""),
               ("BAND", "Single band", ""), ("STI", "STI (estimate)", "Speech intelligibility estimate, needs surfaces")],
        default="A")
    band: EnumProperty(name="Band", items=BAND_ITEMS, default="1000")
    third: EnumProperty(name="Frequency", items=THIRD_ITEMS, default="63")
    auto_range: BoolProperty(name="Auto range", default=True)
    min_db: FloatProperty(name="Min dB", default=70.0, min=-50.0, max=160.0)
    max_db: FloatProperty(name="Max dB", default=110.0, min=-40.0, max=180.0)
    cone_length: FloatProperty(name="Cone length", default=15.0, min=1.0, max=200.0, unit="LENGTH")
    cone_band: EnumProperty(name="Cone band", items=BAND_ITEMS, default="1000")

    vol_cell_m: FloatProperty(name="Voxel size", default=1.0, min=0.1, max=20.0, unit="LENGTH")
    vol_range_db: FloatProperty(name="Dynamic range (dB)", description="Auto range: shown span below the loudest voxel",
                                default=30.0, min=5.0, max=100.0)
    vol_opacity: FloatProperty(name="Opacity", default=0.35, min=0.01, max=1.0)
    vol_threshold: FloatProperty(name="Hide below", description="Fraction of the colour range under which voxels are hidden",
                                 default=0.1, min=0.0, max=0.95)
    vol_blend: EnumProperty(name="Blend", items=[("ADDITIVE", "Additive (glow)", ""), ("ALPHA", "Alpha", "")],
                            default="ADDITIVE")
    vol_display: EnumProperty(
        name="Display",
        items=[("FOG", "Fog", "Smooth ray-marched volume"),
               ("SLICES", "Slice planes", "Stacks of cut planes whose opacity follows the Slice Focus empty"),
               ("BOTH", "Fog and slices", "")],
        default="FOG")
    slice_count: IntProperty(name="Planes per axis", default=24, min=2, max=96)
    slice_axes: BoolVectorProperty(name="Axes", size=3, default=(True, True, True))
    slice_sigma: FloatProperty(name="Focus falloff", description="Distance from the focus at which planes fade, as a fraction of the region size",
                               default=0.15, min=0.01, max=2.0)
    slice_floor: FloatProperty(name="Far plane opacity", description="Opacity of planes far from the focus",
                               default=0.05, min=0.0, max=1.0)

    use_reflections: BoolProperty(
        name="Reflections", description="Specular reflections from surfaces", default=False)
    reflection_order: EnumProperty(
        name="Order", items=[("1", "First order", ""), ("2", "First and second order", "Much slower")], default="1")
    order2_limit: IntProperty(name="Second-order reflectors", description="Only the largest triangles take part in second order",
                              default=24, min=2, max=200)
    reflector_min_area_m2: FloatProperty(name="Ignore faces smaller than", description="Tiny details barely reflect",
                                         default=0.25, min=0.0, max=10.0)
    use_occlusion: BoolProperty(
        name="Occlusion", description="Attenuate direct and reflected paths that cross geometry", default=False)
    shadow_scale: FloatProperty(name="Shadow strength", description="Scales the per-band attenuation of a blocked path",
                                default=1.0, min=0.0, max=3.0)
    occluder_source: EnumProperty(
        name="Occluders",
        items=[("SURFACES", "Meshes with a surface", "Only meshes that were given a surface block sound"),
               ("ALL", "All visible meshes", "Every visible mesh except speakers, receivers and helpers")],
        default="SURFACES")
    audience_dir: EnumProperty(
        name="Audience direction", description="Direction new speakers and arrays face",
        items=[("+Y", "+Y", ""), ("-Y", "-Y", ""), ("+X", "+X", ""), ("-X", "-X", ""),
               ("AUTO", "Towards receivers", "Face the centre of the receiver surfaces")],
        default="+Y")

    room_volume_m3: FloatProperty(name="Room volume override (m³)", description="0 = bounding box of the surfaces",
                                  default=0.0, min=0.0, unit="NONE")
    room_fill: FloatProperty(name="Bounding box fill", description="Share of the bounding box that is air",
                             default=0.85, min=0.1, max=1.0)
    noise_dba: FloatProperty(name="Background noise dB(A)", default=35.0, min=0.0, max=110.0)
    room_rt_sabine: FloatVectorProperty(size=8, default=[0.0] * 8)
    room_rt_eyring: FloatVectorProperty(size=8, default=[0.0] * 8)
    room_info: StringProperty()
    ground_enabled: BoolProperty(name="Ground plane", description="Infinite reflecting floor", default=False)
    ground_z: FloatProperty(name="Ground Z", default=0.0, unit="LENGTH")
    ground_surface_id: EnumProperty(name="Ground surface", items=_surface_items)


class AD_LibraryItem(bpy.types.PropertyGroup):
    model_id: StringProperty()
    manufacturer: StringProperty()
    category: StringProperty()
    quality: StringProperty()
    approximation: BoolProperty()


def register_types():
    bpy.types.Object.ad = PointerProperty(type=AD_ObjectProps)
    bpy.types.Scene.ad_settings = PointerProperty(type=AD_SceneSettings)
    wm = bpy.types.WindowManager
    wm.ad_library_items = CollectionProperty(type=AD_LibraryItem)
    wm.ad_library_index = IntProperty(default=0)
    wm.ad_filter_manufacturer = EnumProperty(name="Manufacturer", items=_manufacturer_items)
    wm.ad_filter_category = EnumProperty(
        name="Category",
        items=[("ALL", "All categories", "")] + [(c, c.replace("_", " ").title(), "") for c in CATEGORIES])
    wm.ad_filter_search = StringProperty(name="Search", options={"TEXTEDIT_UPDATE"})
    wm.ad_surface_choice = EnumProperty(name="Surface", items=_surface_items)


def unregister_types():
    wm = bpy.types.WindowManager
    for name in ("ad_surface_choice", "ad_filter_search", "ad_filter_category", "ad_filter_manufacturer",
                 "ad_library_index", "ad_library_items"):
        delattr(wm, name)
    del bpy.types.Scene.ad_settings
    del bpy.types.Object.ad


classes = (AD_ObjectProps, AD_SceneSettings, AD_LibraryItem)

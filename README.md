# Audio System Designer for Blender

Loudspeaker placement and SPL prediction for live sound, inside Blender. Place real loudspeaker models in a 3D venue,
build line arrays and subwoofer arrays, draw coverage maps on the audience, look at a 3D field, check probes and
reflections, and export a PDF report.

Works as a Blender extension (Blender 4.2 or newer).

> **Status: 0.2.0, early. Not validated against measurements.** Results are for comparing options, not for certifying a
> system. **This is not rigging software**: array geometry, weight and centre of gravity are approximations. Real rigging
> and safety calculations must come from the manufacturer's own tools (ArrayCalc, Soundvision).

## Features

- **Loudspeaker library** built from published datasheet figures (source URL and retrieval date in every file):
  d&b audiotechnik (E3, Y10P, V8, Y-SUB, V-GSUB) and L-Acoustics (KARA II, SB18 IIi, KS28), plus clearly marked
  generic placeholders. Add your own from a datasheet (*New Speaker From Datasheet*) or with JSON files
- **Placement**: place speakers by mouse, face an audience direction, line arrays with auto splay, subwoofer arrays
  (ground stack, end-fire, cardioid)
- **Receivers**: draw areas, use any mesh, or let *Auto Receiver From Scene* find walkable surfaces
- **Calculation**: coverage maps (octave bands power sum, or 1/3 octave phase-aware 20 Hz - 20 kHz), air absorption
  from temperature, humidity and pressure, first and second order reflections, occlusion, RT60 (Sabine and Eyring),
  STI estimate
- **Volumetric field** with fog, slice planes or both; live **probes** with per-speaker contribution and arrival time
- **Surfaces**: library with per-band absorption, auto-assign from material names or orientation
- **PDF report**: settings, speakers, arrays, room acoustics, plan view, volumetric field, probes, surfaces and data sources

## Install

1. Download `audio_designer-0.2.0.zip` from the [Releases](../../releases) page, or build it: `python3 build.py`.
2. Blender: *Edit > Preferences > Get Extensions > Install from Disk* and pick the zip.
3. Open the 3D viewport sidebar (N) and the **Audio Designer** tab.

Everything the add-on creates goes into the collection *Audio Designer*. Speaker origin is the acoustic origin (front
face centre); local +Y is front, +Z up, +X right.

## Workflow

1. **Library**: set *Audience* (the direction new speakers face), pick a model, *Place Speaker*.
2. **Line array / Subwoofer array**: pick a model, *Add Line Array* or *Add Subwoofer Array*, set audience near / far /
   floor, *Auto Splay*, *Analyse Array*.
3. **Receivers**: *Add Receiver Area*, *Receiver From Selected Mesh* or *Auto Receiver From Scene*.
4. **Surfaces**: assign library surfaces or *Auto-assign Surfaces*; optionally switch on Reflections and Occlusion.
5. **Simulation**: atmosphere, calculation mode, map, *Calculate Coverage*. *Add Volume Region* and
   *Calculate Volumetric Field* for the 3D view. Move the probe for live values.
6. **Report**: *Export Report (PDF)*. Whatever has not been calculated is left out.

## How it calculates (and what it does not)

- Directivity H(angle) + V(angle) (separable), inverse-square law (clamped inside 1 m), ISO 9613-1 air absorption.
- Cabinets are point sources with magnitude-only directivity, so line-array results above about 1.5 kHz are indicative.
  Frequency response and directivity are tabulated in octave bands; below 63 Hz the 63 Hz values are held.
- Reflections are specular image sources on the triangles of assigned meshes (and the ground plane). No diffuse or
  reverberant field in the maps, no diffraction.
- Occlusion attenuates blocked paths per band as a crude stand-in for diffraction.
- STI is an estimate for comparing options, **not** an IEC 60268-16 measurement or prediction.
- Library polar shapes are modelled from the published nominal coverage and marked *Basic spec (approximation)*. No
  manufacturer CAD or acoustic datasets (GLL / CLF) are included; meshes are boxes at the published size.
- Bundled surface values are typical textbook-style figures, not certified product data.

Known gaps: manufacturer datasets (GLL / CLF), 3D LODs, background calculation, diffraction, third and higher order
reflections, diffuse reflections, delay alignment assistant, measurement comparison.

## Your own data

- *New Speaker From Datasheet* (Library panel) saves to `<Blender user data>/audio_designer_library`.
- Or put `.json` files (same schema as `audio_designer/library/data`) in a folder set as *User library folder* in the
  add-on preferences. Each needs a `source` and a `data_quality`. Tabulated polars (`"type": "tabulated"`) and
  `"cardioid"` are supported.
- Surfaces: `audio_designer/library/data/surfaces/*.json` (`"kind": "surface_set"`), same user folders.
- Regenerate the bundled library with `python3 tools/generate_library.py`.

## Development

```
python3 -m pytest tests      engine and library, no Blender needed (numpy and pytest)
```

## Licence and data

GPL-3.0-or-later, see [LICENSE](LICENSE). Loudspeaker names and figures are taken from the manufacturers' published
datasheets for planning purposes; trademarks belong to their owners and no affiliation or endorsement is implied.

## Support

Bug reports and ideas: [Issues](../../issues).

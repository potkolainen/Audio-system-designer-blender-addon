# Audio-system-designer-blender-addon
Loudspeaker placement and SPL prediction for live sound

Install
  python3 build.py            -> audio_designer.zip
  Blender: Edit > Preferences > Get Extensions > Install from Disk > audio_designer.zip
  Sidebar (N) in the 3D viewport > "Audio Designer".

Tests (engine + library, no Blender needed)
  python3 -m pytest tests

Everything the add-on creates goes into the collection "Audio Designer"
(Speakers, Arrays/<name>, Subwoofer Arrays/<name>, Receivers, Probes, Visualization).

Speakers: the object origin is the acoustic origin (front face centre by default). Rotating a speaker pivots about
its source point. Local +Y = front, +Z = up, +X = right.

Library
  Bundled, from published datasheet figures (source URL and retrieval date in each file):
    d&b audiotechnik   E3, Y10P, V8, Y-SUB, V-GSUB
    L-Acoustics        KARA II, SB18 IIi, KS28
  Dimensions, weight, frequency range, max SPL and nominal coverage are the manufacturers' figures. The polar
  shape is modelled from the nominal coverage (widening below a control frequency), so every entry is marked
  "Basic spec (approximation)". Items that were not published are listed as assumptions in each entry's notes.
  No manufacturer CAD or acoustic datasets (GLL/CLF) are included; meshes are boxes at the published size.
  "Generic" entries are placeholders and not real products.
  Regenerate with: python3 tools/generate_library.py

  Your own speakers
    New Speaker From Datasheet (Library panel): type in the datasheet figures; it is saved under
    <Blender user data>/audio_designer_library and shows up immediately.
    Or put .json files (schema: any file in audio_designer/library/data) in a folder set as "User library
    folder" in the add-on preferences. Each needs a "source" and a "data_quality". Tabulated polars are
    supported ("type": "tabulated"), as is "cardioid". "max_spl_broadband_db" takes a published broadband SPL.
  Surfaces: audio_designer/library/data/surfaces/*.json ("kind": "surface_set"), same user folders.

Workflow
  Speakers      Library > set Audience (the direction new speakers and arrays face: +Y, -Y, +X, -X or towards the
                receivers) > pick a model > Place Speaker (attaches to the mouse; click to drop).
                The button next to Audience turns the selected speakers/arrays to that direction.
  Line array    Pick a "Line Array" model > Add Line Array. Set audience near/far/floor Z, Auto Splay, edit any
                splay, add/remove cabinets, Analyse Array.
  Sub array     Pick a "Subwoofer" model > Add Subwoofer Array: ground stack, end-fire, cardioid (inverted rear).
  Receivers     Add Receiver Area, Receiver From Selected Mesh, or Auto Receiver From Scene: finds walkable
                (upward-facing) surfaces in the selected or all visible meshes, groups connected faces into islands
                (so each step tread is its own island), drops islands smaller than a minimum area (seat tops, trim),
                optionally drops surfaces with nothing above them, and subdivides to the cell size.
                Listening height is added to floor-like faces only (per-receiver Listening height: Auto/Always/
                Never), so a receiver stood on its side for a vertical cut stays exactly on its plane.
  Surfaces      Surfaces panel: select meshes, pick a library surface, Assign Surface, or Auto-assign Surfaces:
                each face gets a surface from its Blender material name (glass, curtain, carpet, seat, stage...) or,
                failing that, from its orientation (floor, ceiling, wall). Absorption per band can be overridden per
                object. Optional infinite ground plane. Switch on Reflections (first or second order) and/or Occlusion.
  Room          Room Acoustics panel: RT60 per octave band (Sabine and Eyring) from the surfaces, air absorption
                included. Volume = override or bounding box x fill. Map = STI (estimate) draws a speech map.
  Simulation    Atmosphere (temperature, humidity, pressure), Calculation mode, Map; Calculate Coverage.
  Volume        Add Volume Region (scale/rotate; a thin box gives a slice), Calculate Volumetric Field.
                Display: Fog (ray-marched 3D texture, trilinear), Slice planes, or both. Add Slice Focus: an empty
                that lights up the planes nearest to it (snap it to surfaces); opacity, threshold, falloff and
                far-plane opacity update live. If the GPU path fails, a coarser billboard overlay is used.
  Probes        Move the probe empty: SPL per band, distance, arrival time, per-speaker contribution.
  Report        Report panel > Export Report (PDF): one readable document with the settings, every speaker with
                position, azimuth, tilt, level, delay and polarity, line and sub array tables, room acoustics (RT60),
                calculated receiver areas with a plan-view image (the colours you see in Blender), the volumetric
                field from above and from the side (if one was calculated), every audio probe (levels, per-speaker
                distance and arrival, response graph), assigned surfaces and the data source of each speaker model.
                Calculate the heatmap/volume/room first; whatever has not been calculated is left out.

Calculation modes
  Octave, power sum: incoherent, 63 Hz-8 kHz.
  1/3 octave, phase-aware: complex summation incl. delay, polarity, propagation phase and reflections, 20 Hz-20 kHz.
  Cabinets are point sources with magnitude-only directivity, so line-array results above ~1.5 kHz are indicative.
  Frequency response and directivity are tabulated in octave bands; below 63 Hz the 63 Hz values are held.

Physics
  Directivity H(angle)+V(angle) (separable), inverse-square law (clamped inside 1 m), ISO 9613-1 air absorption and
  speed of sound from temperature, humidity and pressure.
  Reflections: specular image sources on triangles of assigned meshes (and the ground plane), first order and
  optionally second order (the largest triangles only). Reflected power is scaled by (1 - absorption) and
  (1 - scattering). No diffuse or reverberant field in the maps, no diffraction.
  Occlusion: every direct and reflected path leg is ray-tested against the occluder meshes. A blocked path is not
  removed but attenuated per band (63 Hz 3 dB ... 8 kHz 28 dB, scaled by Shadow strength), a crude stand-in for
  diffraction. Doors and thin objects block as much as walls do.
  Live probes use first order and the 64 largest reflectors; maps and the report use the full settings.
  STI: modulation-transfer estimate per receiver from the direct sound of each speaker (with delays and
  shadowing), a diffuse reverberant tail (RT60 and room constant from the surfaces, source power from level and
  directivity index) and pink background noise. Flat signal spectrum, no auditory masking, no absolute threshold:
  an estimate for comparing options, NOT an IEC 60268-16 measurement or prediction.
  Not validated against measurements.
  Surface values bundled with the add-on are typical textbook-style figures, not certified product data.

Not rigging data
  Array geometry (front-hinged kinematics), weight and centre of gravity are approximations. Real rigging and safety
  calculations must come from the manufacturer's own tools (ArrayCalc, Soundvision).

Known gaps
  Manufacturer datasets (GLL/CLF), 3D LODs, drag-and-drop from the panel, background calculation, diffraction,
  third and higher order reflections, diffuse reflections, delay alignment assistant, measurement comparison.

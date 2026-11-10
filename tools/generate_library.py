"""Regenerates the manufacturer models in audio_designer/library/data from published datasheet figures.

Run from the Audio folder:  python3 tools/generate_library.py
Every figure below was read from the manufacturer's product page on the retrieval date. Anything that is an
assumption (control frequency, vertical coverage, upper limit of a subwoofer, splay limits) is stated in "notes".
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_designer.library.datasheet import build_model_dict  # noqa: E402

DB = "https://www.dbaudio.com/global/en/products/"
LA = "https://www.l-acoustics.com/products/"
DATE = "Retrieved 2026-11-08. "
POLAR = (" Polar shape is modelled from the nominal coverage (widening below the control frequency); "
         "it is not measured data.")

SPECS = {
    "dbaudio/e3": dict(
        manufacturer="d&b audiotechnik", model="E3", category="point_source",
        width_m=0.190, height_m=0.352, depth_m=0.204, weight_kg=7.2, max_spl_db=123,
        freq_lo_hz=80, freq_hi_hz=18000, h_deg=90, v_deg=60, control_freq_hz=1000,
        source_name="d&b audiotechnik E3 product page (heritage)", source_url=DB + "heritage/e3/",
        notes=DATE + "Published: 80 Hz-18 kHz (-5 dB), SPLmax 123 dB (1 m, free field, peak, pink noise CF 4), "
        "90x60 rotatable CD horn, 352x190x204 mm (h x w x d), 7.2 kg. The 1 kHz control frequency is an assumption."
        + POLAR),
    "dbaudio/y10p": dict(
        manufacturer="d&b audiotechnik", model="Y10P", category="point_source",
        width_m=0.257, height_m=0.580, depth_m=0.341, weight_kg=18.0, max_spl_db=136,
        freq_lo_hz=59, freq_hi_hz=18000, h_deg=110, v_deg=40, control_freq_hz=500,
        source_name="d&b audiotechnik Y10P product page", source_url=DB + "series/y-series/y10p/",
        notes=DATE + "Published: 59 Hz-18 kHz (-5 dB), SPLmax 136 dB (with D40/D80/D90/40D), 110x40 horn rotatable "
        "by 90 degrees, horizontal directivity controlled down to 500 Hz, 580x257x341 mm, 18 kg." + POLAR),
    "dbaudio/v8": dict(
        manufacturer="d&b audiotechnik", model="V8", category="line_array",
        width_m=0.700, height_m=0.310, depth_m=0.460, weight_kg=34.0, max_spl_db=142,
        freq_lo_hz=67, freq_hi_hz=18000, h_deg=80, v_deg=10, control_freq_hz=250, v_control_freq_hz=1000,
        side_angle_deg=14, max_splay_deg=14, max_cabinets=24,
        source_name="d&b audiotechnik V8 product page", source_url=DB + "series/v-series/v8/",
        notes=DATE + "Published: 67 Hz-18 kHz (-5 dB), SPLmax 142 dB, 80 degree horizontal (>= 250 Hz), splay "
        "0-14 degrees, up to 24 cabinets, 310x700x460 mm, 34 kg. The 10 degree vertical coverage and its 1 kHz "
        "control frequency are assumptions, not published figures." + POLAR
        + " Rigging and safety calculations must be done in d&b ArrayCalc."),
    "dbaudio/y_sub": dict(
        manufacturer="d&b audiotechnik", model="Y-SUB", category="subwoofer",
        width_m=0.630, height_m=0.500, depth_m=0.700, weight_kg=52.0, max_spl_db=134,
        freq_lo_hz=39, freq_hi_hz=140, pattern="cardioid",
        source_name="d&b audiotechnik Y-SUB product page", source_url=DB + "series/y-series/y-sub/",
        notes=DATE + "Published: 39-140 Hz (-5 dB), SPLmax 134 dB (with D40/D80/D90/40D), cardioid, 500x630x700 mm "
        "(depth is listed as 700 / 804 mm, 700 used), 52 kg. Directivity is an ideal cardioid. The engine holds "
        "the 63 Hz response below 63 Hz."),
    "dbaudio/v_gsub": dict(
        manufacturer="d&b audiotechnik", model="V-GSUB", category="subwoofer",
        width_m=0.700, height_m=0.606, depth_m=0.728, weight_kg=61.0, max_spl_db=137,
        freq_lo_hz=37, freq_hi_hz=115, pattern="cardioid",
        source_name="d&b audiotechnik V-GSUB product page", source_url=DB + "series/v-series/v-gsub/",
        notes=DATE + "Published: 37-115 Hz (-5 dB), SPLmax 137 dB (with D40/D80/D90/40D), cardioid, 606x700x728 mm "
        "(without wheels), 61 kg. Directivity is an ideal cardioid. The engine holds the 63 Hz response below "
        "63 Hz."),
    "lacoustics/kara_ii": dict(
        manufacturer="L-Acoustics", model="KARA II", category="line_array",
        width_m=0.733, height_m=0.252, depth_m=0.500, weight_kg=26.0, max_spl_db=142,
        freq_lo_hz=55, freq_hi_hz=20000, h_deg=90, v_deg=10, control_freq_hz=1000,
        side_angle_deg=10, max_splay_deg=10, max_cabinets=24,
        source_name="L-Acoustics KARA II product page", source_url=LA + "kara-ii/",
        notes=DATE + "Published: 55 Hz-20 kHz, max SPL 142 dB, horizontal 70/90/110 degrees selectable (90 used), "
        "vertical 10 degrees, 733x252x500 mm (W x H x D), 26 kg. The 1 kHz control frequency, the 10 degree "
        "maximum splay and the 24 cabinet limit are assumptions, not published figures." + POLAR
        + " Rigging and safety calculations must be done in L-Acoustics Soundvision."),
    "lacoustics/sb18_iii": dict(
        manufacturer="L-Acoustics", model="SB18 IIi", category="subwoofer",
        width_m=0.701, height_m=0.540, depth_m=0.728, weight_kg=48.0, max_spl_db=138,
        freq_lo_hz=32, freq_hi_hz=100, pattern="omni",
        source_name="L-Acoustics SB18 IIi product page", source_url=LA + "sb18-iii/",
        notes=DATE + "Published: low frequency limit 32 Hz, max SPL 138 dB, 1 x 18 inch, 701x540x728 mm (W x H x D), "
        "48 kg, standard or cardioid configurations. The standard configuration is modelled; build a cardioid "
        "set-up with a Subwoofer Array. The 100 Hz upper limit is an assumption (none is published on the page). "
        "The pattern is a wide, nearly omnidirectional approximation."),
    "lacoustics/ks28": dict(
        manufacturer="L-Acoustics", model="KS28", category="subwoofer",
        width_m=1.340, height_m=0.550, depth_m=0.719, weight_kg=79.0, max_spl_db=143,
        freq_lo_hz=25, freq_hi_hz=100, pattern="omni",
        source_name="L-Acoustics KS28 product page", source_url=LA + "ks28/",
        notes=DATE + "Published: low frequency limit 25 Hz, max SPL 143 dB, 2 x 18 inch, 1340x550x719 mm (W x H x D), "
        "79 kg, standard or cardioid configurations. The standard configuration is modelled; build a cardioid "
        "set-up with a Subwoofer Array. The 100 Hz upper limit is an assumption (none is published on the page). "
        "The pattern is a wide, nearly omnidirectional approximation."),
}


def main():
    out_root = ROOT / "audio_designer" / "library" / "data"
    for rel, spec in SPECS.items():
        path = out_root / f"{rel}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(build_model_dict(**spec), indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_designer.engine.pdf import PdfDocument  # noqa: E402
from audio_designer.engine.render import (  # noqa: E402
    colormap_rgb8, legend_strip, render_fog, render_receivers_top,
)
from audio_designer.engine.report import build_report  # noqa: E402


def check_structure(pdf: bytes):
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    start = int(re.search(rb"startxref\n(\d+)\n", pdf).group(1))
    assert pdf[start:start + 4] == b"xref"
    size = int(re.search(rb"trailer\n<< /Size (\d+)", pdf).group(1))
    entries = re.findall(rb"(\d{10}) 00000 n ", pdf[start:])
    assert len(entries) == size - 1
    for oid, off in enumerate(entries, 1):
        assert pdf[int(off):].startswith(f"{oid} 0 obj".encode()), oid
    return size


def test_empty_and_text_document_is_structurally_valid():
    doc = PdfDocument("t", compress=False)
    page = doc.new_page()
    page.text(50, 700, "Hello (world) \\ \u00b0C", 12, True)
    page.line(0, 0, 10, 10)
    page.rect(5, 5, 20, 20, (1, 0, 0), (0, 0, 0))
    page.polyline([(0, 0), (5, 5), (9, 2)])
    pdf = doc.to_bytes()
    check_structure(pdf)
    assert b"Hello \\(world\\) \\\\ \xb0C" in pdf


def test_image_is_embedded_and_decodes_to_the_same_pixels():
    img = np.random.default_rng(1).integers(0, 255, (7, 9, 3), dtype=np.uint8)
    doc = PdfDocument("t")
    doc.new_page().image(10, 10, 90, 70, img)
    pdf = doc.to_bytes()
    check_structure(pdf)
    m = re.search(rb"/Width 9 /Height 7.*?stream\n", pdf, re.S)
    body = pdf[m.end():]
    raw = zlib.decompress(body[:body.index(b"\nendstream")])
    assert raw == img.tobytes()


def test_colormap_ends_and_no_data():
    c = colormap_rgb8([-200.0, 0.0, 1.0], 0.0, 1.0)
    assert tuple(c[0]) == (38, 38, 38)           # no-data grey
    assert c[1][2] > c[1][0] and c[2][0] > c[2][2]  # blue at low end, red at high end
    assert legend_strip().shape == (1, 256, 3)


def test_receiver_top_view_places_cells_and_respects_height_order():
    xs, ys = np.meshgrid(np.arange(10) + 0.5, np.arange(6) + 0.5)
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(60)])
    colours = np.tile(np.array([[255, 0, 0]], np.uint8), (60, 1))
    img = render_receivers_top(pts, colours, 1.0, max_px=100)
    assert img.extent == (0.0, 10.0, 0.0, 6.0)
    assert img.rgb.shape[1] > img.rgb.shape[0]
    assert tuple(img.rgb[img.rgb.shape[0] // 2, img.rgb.shape[1] // 2]) == (255, 0, 0)
    # a higher cell drawn over a lower one at the same spot wins
    stacked = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 3.0]])
    two = render_receivers_top(stacked, np.array([[255, 0, 0], [0, 255, 0]], np.uint8), 1.0, max_px=50)
    assert tuple(two.rgb[two.rgb.shape[0] // 2, two.rgb.shape[1] // 2]) == (0, 255, 0)


def test_fog_renders_blob_in_the_right_place_from_top_and_side():
    t = np.zeros((8, 10, 12), np.float32)          # nz, ny, nx
    t[6:8, 2:4, 9:11] = 1.0                         # high z, low y, high x
    top = render_fog(t, "top", (12.0, 10.0, 8.0), opacity=0.8, threshold=0.1, max_px=120)
    side = render_fog(t, "side", (12.0, 10.0, 8.0), opacity=0.8, threshold=0.1, max_px=120)
    bg = np.array([30, 30, 34])

    def bright(img):
        d = np.abs(img.rgb.astype(int) - bg).sum(axis=2)
        ys, xs = np.nonzero(d > 40)
        return ys.mean() / img.rgb.shape[0], xs.mean() / img.rgb.shape[1]

    ty, tx = bright(top)
    assert tx > 0.65 and ty > 0.6          # high x -> right, low y -> bottom
    sy, sx = bright(side)
    assert sy < 0.35 and sx < 0.35         # high z -> top, low y -> left
    assert top.extent == (-6.0, 6.0, -5.0, 5.0) and side.extent == (-5.0, 5.0, -4.0, 4.0)
    with pytest.raises(ValueError):
        render_fog(t, "bottom", (1, 1, 1))


def test_fog_is_empty_below_threshold():
    img = render_fog(np.full((4, 4, 4), 0.05, np.float32), "top", (4, 4, 4), threshold=0.1, max_px=40)
    assert np.all(img.rgb == np.array([30, 30, 34], np.uint8))


def _sample_data():
    top = render_receivers_top(np.array([[0, 0, 0.0], [5, 5, 0.0]]), np.array([[255, 0, 0], [0, 0, 255]], np.uint8), 1.0, 80)
    return {
        "title": "Test report", "scene": "Hall", "date": "2026-11-08",
        "settings": [("Temperature", "20 C")],
        "speakers": [{"name": f"S{i}", "model": "Model X", "quality": "Basic spec", "x": 1, "y": 2, "z": 3,
                      "azimuth": 10, "tilt": -5, "level": -10, "delay": 0, "group": "Array"} for i in range(60)],
        "arrays": [{"title": "Line Array", "lines": ["8 cabinets"],
                    "table": {"headers": ["Cab", "Tilt"], "rows": [[1, "2.0"], [2, "4.0"]]}}],
        "room": {"info": "1000 m3", "rows": [["63 Hz", "1.20 s", "1.10 s"]], "noise": 35.0},
        "receivers": [{"name": "Audience", "metric": "dB(A)", "min": 80, "max": 100, "mean": 90, "spread": 20,
                       "image": top, "legend": (70, 110, "dB(A)"), "markers": [(1, 1, "S1")]}],
        "probes": [{"name": "Probe", "x": 1, "y": 2, "z": 3, "a": 95, "c": 97, "bb": 98, "method": "Octave",
                    "band_names": [63, 125], "bands": [90, 91],
                    "speakers": [{"name": "S1", "dist": 10, "arrival": 29, "dba": 90}],
                    "response": (np.array([20, 100, 1000, 20000.0]), np.array([80, 90, 95, 70.0]))}],
        "surfaces": [["Floor", "Carpet", "0.02 ... 0.65"]],
        "sources": [{"model": "Model X", "quality": "Basic spec (approximation)", "source": "Datasheet", "url": "https://x",
                     "notes": "n"}],
        "notes": ["An estimate."],
    }


def test_full_report_is_valid_paginates_and_contains_the_data():
    pdf = build_report(_sample_data(), compress=False)
    check_structure(pdf)
    assert pdf.count(b"/Type /Page ") >= 2
    for needle in (b"Test report", b"Line Array", b"RT60 Sabine", b"Audience", b"Audio probes", b"Probe", b"Datasheet",
                   b"Page 1 /"):
        assert needle in pdf


def test_report_with_no_sections_still_produces_a_page():
    pdf = build_report({})
    check_structure(pdf)
    assert b"/Count 1" in pdf

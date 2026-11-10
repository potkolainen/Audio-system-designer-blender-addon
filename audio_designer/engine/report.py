"""Builds the human-readable PDF report from plain data. No Blender imports.

`data` is a dict of simple values; every section is optional. See `build_report` for the keys.
"""
from __future__ import annotations

import textwrap

import numpy as np

from .pdf import A4, PdfDocument
from .render import RenderedImage, legend_strip

MARGIN = 40.0
GREY = (0.35, 0.35, 0.35)
RULE = (0.75, 0.75, 0.75)
MAX_IMAGE_HEIGHT = 470.0


class _Writer:
    def __init__(self, title: str, compress: bool = True):
        self.doc = PdfDocument(title, compress)
        self.width, self.height = A4
        self.content_w = self.width - 2 * MARGIN
        self.page = None
        self.y = 0.0
        self._new_page()

    def _new_page(self):
        self.page = self.doc.new_page()
        self.y = self.height - MARGIN

    def ensure(self, h: float):
        if self.y - h < MARGIN + 14:
            self._new_page()

    def space(self, h: float = 6.0):
        self.y -= h

    def line(self, text, size=9.0, bold=False, color=(0, 0, 0), indent=0.0):
        self.ensure(size * 1.4)
        self.y -= size * 1.25
        self.page.text(MARGIN + indent, self.y, text, size, bold, color)
        self.y -= size * 0.15

    def title(self, text, sub=""):
        self.line(text, 20, True)
        if sub:
            self.line(sub, 10, False, GREY)
        self.space(6)

    def heading(self, text, need=40.0):
        self.ensure(need)
        self.space(10)
        self.line(text, 13, True)
        self.page.line(MARGIN, self.y - 1, self.width - MARGIN, self.y - 1, 0.8, RULE)
        self.space(4)

    def subheading(self, text):
        self.ensure(28)
        self.space(4)
        self.line(text, 10, True)

    def para(self, text, size=9.0, color=(0, 0, 0)):
        for raw in str(text).splitlines() or [""]:
            for chunk in textwrap.wrap(raw, max(20, int(self.content_w / (size * 0.5)))) or [""]:
                self.line(chunk, size, False, color)

    def kv(self, rows, key_w=170.0, size=9.0):
        for k, v in rows:
            self.ensure(size * 1.4)
            self.y -= size * 1.25
            self.page.text(MARGIN, self.y, str(k), size, True)
            self.page.text(MARGIN + key_w, self.y, _fit(str(v), self.content_w - key_w, size), size)
            self.y -= size * 0.15

    def table(self, headers, rows, widths=None, size=8.0):
        n = len(headers)
        widths = widths or [1.0] * n
        total = float(sum(widths))
        cols = [self.content_w * w / total for w in widths]
        xs = [MARGIN + sum(cols[:i]) for i in range(n)]
        row_h = size * 1.55

        def header():
            self.ensure(row_h * 2)
            self.y -= row_h
            for x, c, h in zip(xs, cols, headers):
                self.page.text(x, self.y, _fit(str(h), c - 3, size), size, True)
            self.page.line(MARGIN, self.y - 2.5, self.width - MARGIN, self.y - 2.5, 0.6, RULE)
            self.y -= 2

        header()
        for r in rows:
            if self.y - row_h < MARGIN + 14:
                self._new_page()
                header()
            self.y -= row_h
            for x, c, v in zip(xs, cols, r):
                self.page.text(x, self.y, _fit(str(v), c - 3, size), size)
        self.space(4)

    def image_block(self, img: RenderedImage, caption="", legend=None, markers=(), size=7.0):
        left = 34.0
        h_px, w_px = img.rgb.shape[:2]
        w = self.content_w - left
        h = w * h_px / w_px
        if h > MAX_IMAGE_HEIGHT:
            h, w = MAX_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT * w_px / h_px
        self.ensure(h + 52)
        if caption:
            self.line(caption, 9, True)
            self.space(9)
        x, y = MARGIN + left, self.y - h - 2
        self.page.image(x, y, w, h, img.rgb)
        x0, x1, y0, y1 = img.extent
        fx = lambda v: x + (v - x0) / max(x1 - x0, 1e-9) * w  # noqa: E731
        fy = lambda v: y + (v - y0) / max(y1 - y0, 1e-9) * h  # noqa: E731
        for mx, my, label in markers:
            if x0 <= mx <= x1 and y0 <= my <= y1:
                px, py = fx(mx), fy(my)
                self.page.line(px - 3, py, px + 3, py, 1.0, (1, 1, 1))
                self.page.line(px, py - 3, px, py + 3, 1.0, (1, 1, 1))
                self.page.text(px + 4, py + 2, label, size - 1, False, (1, 1, 1))
        for v in (y0, (y0 + y1) / 2, y1):
            self.page.text(MARGIN, fy(v) - 2, _tick(v), size, False, GREY)
        for v in (x0, (x0 + x1) / 2, x1):
            self.page.text(fx(v) - 6, y - 9, _tick(v), size, False, GREY)
        self.page.text(x + w - 60, y - 9, img.x_label, size, False, GREY)
        self.page.text(MARGIN, y + h + 2, img.y_label, size, False, GREY)
        self.y = y - 12
        if legend:
            lo, hi, label = legend
            self.ensure(24)
            self.page.image(MARGIN + left, self.y - 8, 180, 7, legend_strip())
            self.page.text(MARGIN + left, self.y - 17, f"{lo:g}", size, False, GREY)
            self.page.text(MARGIN + left + 160, self.y - 17, f"{hi:g}", size, False, GREY)
            self.page.text(MARGIN + left + 190, self.y - 7, label, size)
            self.y -= 22
        self.space(6)

    def response_plot(self, freqs, spl, title="", width=None, height=110.0, size=7.0):
        width = width or self.content_w - 34
        self.ensure(height + 34)
        if title:
            self.line(title, 9, True)
            self.space(6)
        x, y = MARGIN + 34, self.y - height - 4
        freqs, spl = np.asarray(freqs, dtype=float), np.asarray(spl, dtype=float)
        ok = spl > -90
        top = float(np.ceil((spl[ok].max() if ok.any() else 0) / 10.0) * 10.0)
        bottom = top - 60.0
        fx = lambda f: x + (np.log10(f) - np.log10(20.0)) / (np.log10(20000.0) - np.log10(20.0)) * width  # noqa: E731
        fy = lambda v: y + (np.clip(v, bottom, top) - bottom) / (top - bottom) * height  # noqa: E731
        self.page.rect(x, y, width, height, None, RULE)
        for f in (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000):
            self.page.line(fx(f), y, fx(f), y + height, 0.3, RULE)
            self.page.text(fx(f) - 8, y - 9, f"{f / 1000:g}k" if f >= 1000 else f"{f:g}", size, False, GREY)
        for v in np.arange(bottom, top + 0.1, 10.0):
            self.page.line(x, fy(v), x + width, fy(v), 0.3, RULE)
            self.page.text(MARGIN, fy(v) - 2, f"{v:.0f}", size, False, GREY)
        self.page.polyline([(fx(f), fy(v)) for f, v, k in zip(freqs, spl, ok) if k], 1.2, (0.1, 0.25, 0.85))
        self.page.text(x + width - 40, y + height + 2, "dB / Hz", size, False, GREY)
        self.y = y - 14
        self.space(4)

    def finish(self) -> bytes:
        total = len(self.doc.pages)
        for i, page in enumerate(self.doc.pages, 1):
            page.text(MARGIN, 22, "Audio System Designer. Estimates only; not validated against measurements.", 7, False, GREY)
            page.text(self.width - MARGIN - 40, 22, f"Page {i} / {total}", 7, False, GREY)
        return self.doc.to_bytes()


def _tick(v: float) -> str:
    return f"{(round(v, 1) or 0.0):g}"


def _fit(text: str, width: float, size: float) -> str:
    limit = max(int(width / (size * 0.5)), 3)
    return text if len(text) <= limit else text[: limit - 2] + ".."


def _f(v, digits=1):
    return "" if v is None else f"{v:.{digits}f}"


def build_report(data: dict, compress: bool = True) -> bytes:
    """data keys (all optional): title, scene, date, settings [(k, v)], speakers [dict], arrays [dict],
    room dict, receivers [dict], fog dict, probes [dict], surfaces [list], sources [dict], notes [str]."""
    w = _Writer(data.get("title", "Audio system report"), compress)
    w.title(data.get("title", "Audio system report"), f"{data.get('scene', '')}   {data.get('date', '')}".strip())

    if data.get("settings"):
        w.heading("Prediction settings")
        w.kv(data["settings"])

    speakers = data.get("speakers") or []
    if speakers:
        w.heading("Speakers")
        w.para("Position in metres. Azimuth is the horizontal aim, 0 = +Y, positive towards +X. Tilt is the vertical "
               "aim of the speaker axis, negative = downwards.", 8, GREY)
        w.table(["Name", "Model", "Data", "X", "Y", "Z", "Azimuth", "Tilt", "Level dB", "Delay ms", "Pol", "Group"],
                [[s["name"], s["model"], s.get("quality", ""), _f(s["x"], 2), _f(s["y"], 2), _f(s["z"], 2),
                  _f(s["azimuth"]) + "\u00b0", _f(s["tilt"]) + "\u00b0", _f(s["level"]), _f(s["delay"], 2),
                  "inv" if s.get("inverted") else ("mute" if s.get("mute") else "+"), s.get("group", "")]
                 for s in speakers],
                [2.2, 2.4, 1.6, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 0.7, 1.6], 7.0)

    for arr in data.get("arrays") or []:
        w.subheading(arr["title"])
        for line in arr.get("lines", []):
            w.line(line, 8.5)
        if arr.get("table"):
            w.table(arr["table"]["headers"], arr["table"]["rows"], arr["table"].get("widths"), 7.5)

    room = data.get("room")
    if room:
        w.heading("Room acoustics")
        w.para(room["info"])
        w.table(["Band", "RT60 Sabine", "RT60 Eyring"], room["rows"], [1, 1, 1], 8.5)
        w.kv([("Background noise", f"{room['noise']:g} dB(A)")])

    receivers = data.get("receivers") or []
    if receivers:
        w.heading("Receiver areas", need=150)
        w.table(["Receiver", "Map", "Min", "Max", "Mean", "Spread"],
                [[r["name"], r["metric"], _f(r["min"], 2), _f(r["max"], 2), _f(r["mean"], 2), _f(r["spread"], 2)]
                 for r in receivers], [3, 2, 1, 1, 1, 1], 8.5)
        for r in receivers:
            if r.get("image") is not None:
                w.image_block(r["image"], f"{r['name']}: {r['metric']} seen from above", r.get("legend"),
                              r.get("markers", ()))

    fog = data.get("fog")
    if fog:
        w.heading("Volumetric field", need=330)
        w.para(fog["info"])
        w.para("Both views use the volume region's own axes; markers are speakers and probes.", 8, GREY)
        for key, caption in (("top", "Seen from above"), ("side", "Seen from the side (from +X, y to the right)")):
            if fog.get(key) is not None:
                w.image_block(fog[key], caption, fog.get("legend"), fog.get(key + "_markers", ()))

    probes = data.get("probes") or []
    if probes:
        w.heading("Audio probes")
        for p in probes:
            w.subheading(p["name"])
            w.kv([("Position (m)", f"{p['x']:.2f}, {p['y']:.2f}, {p['z']:.2f}"),
                  ("Level", f"{p['a']:.1f} dB(A)   {p['c']:.1f} dB(C)   {p['bb']:.1f} dB broadband"),
                  ("Method", p.get("method", ""))])
            w.table(["Band"] + [f"{b} Hz" for b in p["band_names"]], [["SPL dB"] + [_f(v) for v in p["bands"]]],
                    [1.2] + [1] * len(p["band_names"]), 7.5)
            if p.get("speakers"):
                w.table(["Speaker", "Distance m", "Arrival ms", "Contribution dB(A)"],
                        [[s["name"], _f(s["dist"]), _f(s["arrival"]), _f(s["dba"])] for s in p["speakers"]],
                        [3, 1, 1, 1.5], 7.5)
            if p.get("response") is not None:
                w.response_plot(p["response"][0], p["response"][1], "Combined response, phase-aware 1/3 octave")

    if data.get("surfaces"):
        w.heading("Surfaces")
        w.table(["Object", "Surface", "Absorption 63 Hz ... 8 kHz"], data["surfaces"], [2, 3, 5], 7.5)

    if data.get("sources"):
        w.heading("Data sources of the speakers used")
        for s in data["sources"]:
            w.line(f"{s['model']}  ({s['quality']})", 9, True)
            w.para(f"{s['source']}  {s.get('url', '')}", 8)
            if s.get("notes"):
                w.para(s["notes"], 7.5, GREY)

    if data.get("notes"):
        w.heading("Notes")
        for n in data["notes"]:
            w.para("- " + n, 8.5)
    return w.finish()

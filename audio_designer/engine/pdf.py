"""Minimal PDF writer: text, lines, rectangles, polylines and RGB images. Standard library and NumPy only."""
from __future__ import annotations

import zlib
from datetime import datetime

import numpy as np

A4 = (595.276, 841.89)


def _escape(text: str) -> bytes:
    raw = text.encode("cp1252", errors="replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _num(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") or "0"


def _rgb(c) -> str:
    return " ".join(_num(v) for v in c)


class PdfPage:
    def __init__(self, width: float, height: float):
        self.width, self.height = width, height
        self.ops: list[bytes] = []
        self.images: list[tuple[np.ndarray, int, int]] = []

    def text(self, x, y, s, size=10, bold=False, color=(0, 0, 0)):
        font = "F2" if bold else "F1"
        self.ops.append(f"BT /{font} {_num(size)} Tf {_rgb(color)} rg {_num(x)} {_num(y)} Td (".encode() + _escape(s)
                        + b") Tj ET")

    def line(self, x1, y1, x2, y2, width=0.5, color=(0, 0, 0)):
        self.ops.append(f"{_rgb(color)} RG {_num(width)} w {_num(x1)} {_num(y1)} m {_num(x2)} {_num(y2)} l S".encode())

    def polyline(self, points, width=1.0, color=(0, 0, 0)):
        pts = list(points)
        if len(pts) < 2:
            return
        path = f"{_num(pts[0][0])} {_num(pts[0][1])} m " + " ".join(f"{_num(x)} {_num(y)} l" for x, y in pts[1:])
        self.ops.append(f"{_rgb(color)} RG {_num(width)} w {path} S".encode())

    def rect(self, x, y, w, h, fill=None, stroke=None, width=0.5):
        parts = []
        if fill is not None:
            parts.append(f"{_rgb(fill)} rg")
        if stroke is not None:
            parts.append(f"{_rgb(stroke)} RG {_num(width)} w")
        op = "B" if fill is not None and stroke is not None else "f" if fill is not None else "S"
        parts.append(f"{_num(x)} {_num(y)} {_num(w)} {_num(h)} re {op}")
        self.ops.append(" ".join(parts).encode())

    def image(self, x, y, w, h, rgb: np.ndarray):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self.images.append((rgb, rgb.shape[1], rgb.shape[0]))
        self.ops.append(f"q {_num(w)} 0 0 {_num(h)} {_num(x)} {_num(y)} cm /Im{len(self.images)} Do Q".encode())


class PdfDocument:
    def __init__(self, title: str = "", compress: bool = True):
        self.title = title
        self.compress = compress
        self.pages: list[PdfPage] = []

    def new_page(self, width: float = A4[0], height: float = A4[1]) -> PdfPage:
        page = PdfPage(width, height)
        self.pages.append(page)
        return page

    def _stream(self, data: bytes, extra: str = "") -> bytes:
        if self.compress:
            data = zlib.compress(data, 6)
            extra += " /Filter /FlateDecode"
        return f"<< {extra} /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"

    def to_bytes(self) -> bytes:
        objects: dict[int, bytes] = {}
        n_pages = len(self.pages)
        first_page_id = 5
        ids = []
        next_id = first_page_id
        for page in self.pages:
            page_id, content_id = next_id, next_id + 1
            image_ids = [next_id + 2 + i for i in range(len(page.images))]
            ids.append((page_id, content_id, image_ids))
            next_id += 2 + len(page.images)

        objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
        kids = " ".join(f"{p[0]} 0 R" for p in ids)
        objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode()
        objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
        objects[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
        for page, (page_id, content_id, image_ids) in zip(self.pages, ids):
            xobjects = " ".join(f"/Im{i + 1} {iid} 0 R" for i, iid in enumerate(image_ids))
            objects[page_id] = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_num(page.width)} {_num(page.height)}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> /XObject << {xobjects} >> >> "
                f"/Contents {content_id} 0 R >>").encode()
            objects[content_id] = self._stream(b"\n".join(page.ops))
            for iid, (rgb, w, h) in zip(image_ids, page.images):
                objects[iid] = self._stream(
                    rgb.tobytes(), f"/Type /XObject /Subtype /Image /Width {w} /Height {h} "
                                   "/ColorSpace /DeviceRGB /BitsPerComponent 8")
        info_id = next_id
        stamp = datetime.now().strftime("D:%Y%m%d%H%M%S")
        objects[info_id] = b"<< /Title (" + _escape(self.title) + b") /Producer (Audio System Designer) /CreationDate (" \
            + stamp.encode() + b") >>"

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = {}
        for oid in sorted(objects):
            offsets[oid] = len(out)
            out += f"{oid} 0 obj\n".encode() + objects[oid] + b"\nendobj\n"
        xref = len(out)
        size = max(objects) + 1
        out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
        for oid in range(1, size):
            out += f"{offsets[oid]:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {size} /Root 1 0 R /Info {info_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode()
        return bytes(out)

    def save(self, path) -> None:
        with open(path, "wb") as f:
            f.write(self.to_bytes())

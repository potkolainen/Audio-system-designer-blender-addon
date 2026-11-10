"""Loudspeaker definitions: JSON on disk -> validated LoudspeakerModel (no Blender imports)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from ..engine import BANDS, NUM_BANDS, AcousticModel, PolarTable

SCHEMA_VERSION = 1
MAX_ARRAY_CABINETS = 24
POLAR_ANGLES_DEG = np.arange(0.0, 181.0, 5.0)
PARAMETRIC_FLOOR_DB = -30.0

CATEGORIES = ("point_source", "line_array", "subwoofer", "fill", "other")


class LibraryError(ValueError):
    pass


class DataQuality(Enum):
    VISUAL_ONLY = "visual_only"
    BASIC_SPEC = "basic_spec"
    SIMULATION_READY = "simulation_ready"
    MANUFACTURER_VALIDATED = "manufacturer_validated"

    @property
    def label(self) -> str:
        return {
            "visual_only": "Visual only",
            "basic_spec": "Basic specification",
            "simulation_ready": "Simulation-ready",
            "manufacturer_validated": "Manufacturer-validated",
        }[self.value]

    @property
    def simulatable(self) -> bool:
        return self is not DataQuality.VISUAL_ONLY


@dataclass(frozen=True)
class LoudspeakerModel:
    id: str
    manufacturer: str
    model: str
    category: str
    data_quality: DataQuality
    approximation: bool
    width_m: float
    height_m: float
    depth_m: float
    weight_kg: float
    acoustic_origin_m: tuple
    acoustic: AcousticModel | None
    source: dict
    path: str = ""
    side_angle_deg: float = 0.0
    max_splay_deg: float = 10.0
    max_cabinets: int = 24

    @property
    def display_name(self) -> str:
        return f"{self.manufacturer} {self.model}"

    @property
    def quality_label(self) -> str:
        suffix = " (approximation)" if self.approximation else ""
        return self.data_quality.label + suffix


def _require(data: dict, key: str, where: str):
    if key not in data:
        raise LibraryError(f"{where}: missing '{key}'")
    return data[key]


def _band_vector(values, where: str) -> np.ndarray:
    if isinstance(values, dict):
        try:
            values = [values[str(b)] for b in BANDS]
        except KeyError as e:
            raise LibraryError(f"{where}: missing band {e}") from None
    arr = np.asarray(values, dtype=float)
    if arr.shape != (NUM_BANDS,):
        raise LibraryError(f"{where}: expected {NUM_BANDS} band values, got {arr.shape}")
    return arr


def _parametric_table(widths_deg: np.ndarray) -> PolarTable:
    """Elliptical -6 dB beam: level = -6 * (angle / half_width)^2, floored."""
    half = np.maximum(widths_deg, 1.0)[:, None] * 0.5
    levels = np.maximum(-6.0 * (POLAR_ANGLES_DEG[None, :] / half) ** 2, PARAMETRIC_FLOOR_DB)
    return PolarTable(POLAR_ANGLES_DEG.copy(), levels)


def _tabulated_table(angles, per_band: dict, where: str) -> PolarTable:
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1 or angles.size < 2 or np.any(np.diff(angles) <= 0):
        raise LibraryError(f"{where}: angles must be strictly increasing")
    rows = []
    for b in BANDS:
        row = np.asarray(_require(per_band, str(b), where), dtype=float)
        if row.shape != angles.shape:
            raise LibraryError(f"{where}: band {b} length does not match angles")
        rows.append(row)
    return PolarTable(angles, np.vstack(rows))


def _cardioid_table() -> PolarTable:
    """Ideal cardioid, 0.5 * (1 + cos(angle)), identical in every band."""
    row = np.maximum(20.0 * np.log10(np.maximum((1.0 + np.cos(np.radians(POLAR_ANGLES_DEG))) / 2.0, 1e-3)),
                     PARAMETRIC_FLOOR_DB)
    return PolarTable(POLAR_ANGLES_DEG.copy(), np.tile(row, (NUM_BANDS, 1)))


def _parse_directivity(d: dict, where: str):
    kind = _require(d, "type", where)
    if kind == "cardioid":
        table = _cardioid_table()
        return table, table
    if kind == "parametric_coverage":
        return (_parametric_table(_band_vector(_require(d, "horizontal_deg", where), where + ".horizontal_deg")),
                _parametric_table(_band_vector(_require(d, "vertical_deg", where), where + ".vertical_deg")))
    if kind == "tabulated":
        angles = _require(d, "angles_deg", where)
        return (_tabulated_table(angles, _require(d, "horizontal_db", where), where + ".horizontal_db"),
                _tabulated_table(angles, _require(d, "vertical_db", where), where + ".vertical_db"))
    raise LibraryError(f"{where}: unknown directivity type '{kind}'")


def parse_model(data: dict, path: str = "") -> LoudspeakerModel:
    where = path or data.get("id", "<model>")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise LibraryError(f"{where}: unsupported schema_version {data.get('schema_version')!r}")

    try:
        quality = DataQuality(_require(data, "data_quality", where))
    except ValueError:
        raise LibraryError(f"{where}: invalid data_quality") from None
    category = data.get("category", "other")
    if category not in CATEGORIES:
        raise LibraryError(f"{where}: invalid category '{category}'")

    phys = _require(data, "physical", where)
    dims = [float(_require(phys, k, where + ".physical")) for k in ("width_m", "height_m", "depth_m")]
    if any(v <= 0 for v in dims):
        raise LibraryError(f"{where}: dimensions must be positive")
    origin = tuple(float(v) for v in data.get("acoustic_origin_m", (0.0, dims[2] / 2.0, 0.0)))
    if len(origin) != 3:
        raise LibraryError(f"{where}: acoustic_origin_m needs 3 values")

    arr = data.get("array", {})
    if not 0.0 <= float(phys.get("side_angle_deg", 0.0)) < 60.0:
        raise LibraryError(f"{where}: side_angle_deg must be in [0, 60)")
    if not 1 <= int(arr.get("max_cabinets", 24)) <= MAX_ARRAY_CABINETS:
        raise LibraryError(f"{where}: array.max_cabinets must be 1..{MAX_ARRAY_CABINETS}")

    source = data.get("source")
    if not isinstance(source, dict) or not source.get("name"):
        raise LibraryError(f"{where}: 'source' with a name is required (provenance)")

    acoustic = None
    if quality.simulatable:
        a = _require(data, "acoustic", where)
        horizontal, vertical = _parse_directivity(_require(a, "directivity", where + ".acoustic"),
                                                  where + ".acoustic.directivity")
        response = _band_vector(a.get("frequency_response_db", [0.0] * NUM_BANDS),
                                where + ".acoustic.frequency_response_db")
        if "max_spl_broadband_db" in a:
            # Published broadband peak SPL -> per-band level whose octave power sum reproduces it.
            max_spl = float(a["max_spl_broadband_db"]) - 10.0 * float(np.log10(np.sum(10.0 ** (response / 10.0))))
        else:
            max_spl = float(_require(a, "max_spl_1m_db", where + ".acoustic"))
        acoustic = AcousticModel(max_spl, response, horizontal, vertical)

    return LoudspeakerModel(
        id=str(_require(data, "id", where)),
        manufacturer=str(_require(data, "manufacturer", where)),
        model=str(_require(data, "model", where)),
        category=category,
        data_quality=quality,
        approximation=bool(data.get("approximation", False)),
        width_m=dims[0], height_m=dims[1], depth_m=dims[2],
        weight_kg=float(phys.get("weight_kg", 0.0)),
        acoustic_origin_m=origin,
        acoustic=acoustic,
        source=source,
        path=path,
        side_angle_deg=float(phys.get("side_angle_deg", 0.0)),
        max_splay_deg=float(arr.get("max_splay_deg", 10.0)),
        max_cabinets=int(arr.get("max_cabinets", 24)),
    )


@dataclass(frozen=True)
class SurfaceMaterial:
    id: str
    name: str
    category: str
    absorption: np.ndarray   # (8,) octave bands 63 Hz-8 kHz
    scattering: float
    source: dict
    path: str = ""


def parse_surface(data: dict, path: str = "") -> SurfaceMaterial:
    where = path or data.get("id", "<surface>")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise LibraryError(f"{where}: unsupported schema_version {data.get('schema_version')!r}")
    absorption = _band_vector(_require(data, "absorption", where), where + ".absorption")
    if np.any(absorption < 0) or np.any(absorption > 1):
        raise LibraryError(f"{where}: absorption coefficients must be within 0..1")
    scattering = float(data.get("scattering", 0.1))
    if not 0.0 <= scattering <= 1.0:
        raise LibraryError(f"{where}: scattering must be within 0..1")
    source = data.get("source")
    if not isinstance(source, dict) or not source.get("name"):
        raise LibraryError(f"{where}: 'source' with a name is required (provenance)")
    return SurfaceMaterial(str(_require(data, "id", where)), str(_require(data, "name", where)),
                           str(data.get("category", "other")), absorption, scattering, source, path)


class Library:
    def __init__(self):
        self.models: dict[str, LoudspeakerModel] = {}
        self.surfaces: dict[str, SurfaceMaterial] = {}
        self.errors: list[str] = []

    def scan(self, *roots) -> "Library":
        self.models.clear()
        self.surfaces.clear()
        self.errors.clear()
        for root in roots:
            if not root:
                continue
            root = Path(root)
            if not root.is_dir():
                continue
            for file in sorted(root.rglob("*.json")):
                try:
                    data = json.loads(file.read_text(encoding="utf-8"))
                    if not isinstance(data, dict) or "schema_version" not in data:
                        continue
                    if data.get("kind") == "surface_set":
                        for entry in _require(data, "surfaces", str(file)):
                            entry = {"schema_version": data["schema_version"], "source": data.get("source"), **entry}
                            self._add(self.surfaces, parse_surface(entry, str(file)), file)
                        continue
                    if data.get("kind") == "surface":
                        item, table = parse_surface(data, str(file)), self.surfaces
                    else:
                        item, table = parse_model(data, str(file)), self.models
                except (OSError, json.JSONDecodeError, LibraryError) as e:
                    self.errors.append(f"{file}: {e}")
                    continue
                self._add(table, item, file)
        return self

    def _add(self, table, item, file) -> None:
        if item.id in table:
            self.errors.append(f"{file}: duplicate id '{item.id}' (first definition kept)")
        else:
            table[item.id] = item

    def manufacturers(self) -> list[str]:
        return sorted({m.manufacturer for m in self.models.values()}, key=str.lower)

    def get(self, model_id: str) -> LoudspeakerModel | None:
        return self.models.get(model_id)

    def surface(self, surface_id: str) -> "SurfaceMaterial | None":
        return self.surfaces.get(surface_id)


BUILTIN_DATA_DIR = Path(__file__).parent / "data"

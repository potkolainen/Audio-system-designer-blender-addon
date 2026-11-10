"""Build audio_designer.zip with the manifest at the ZIP root (installable via Install from Disk)."""
import zipfile
from pathlib import Path

root = Path(__file__).parent
pkg = root / "audio_designer"
out = root / "audio_designer.zip"

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(pkg.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            z.write(f, f.relative_to(pkg))
print(out)

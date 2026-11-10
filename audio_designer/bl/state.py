"""Shared add-on state: the loaded library and the add-on preferences lookup."""
import bpy

from pathlib import Path

from ..library import BUILTIN_DATA_DIR, Library

ADDON_ID = __package__.rpartition(".")[0]
LIBRARY = Library()


def user_library_path() -> str:
    entry = bpy.context.preferences.addons.get(ADDON_ID)
    if entry is None:
        return ""
    return bpy.path.abspath(getattr(entry.preferences, "user_library_path", ""))


def default_user_dir() -> Path:
    """Per-user folder where 'New Speaker From Datasheet' saves its definitions."""
    return Path(bpy.utils.user_resource("DATAFILES", path="audio_designer_library", create=True))


def reload_library() -> None:
    LIBRARY.scan(BUILTIN_DATA_DIR, user_library_path(), default_user_dir())
    wm = bpy.context.window_manager
    items = wm.ad_library_items
    items.clear()
    for m in sorted(LIBRARY.models.values(), key=lambda m: (m.manufacturer.lower(), m.model.lower())):
        item = items.add()
        item.name = m.display_name
        item.model_id = m.id
        item.manufacturer = m.manufacturer
        item.category = m.category
        item.quality = m.data_quality.value
        item.approximation = m.approximation

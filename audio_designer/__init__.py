"""Deliberately Blender-free: the engine and library must import without bpy."""
__version__ = (0, 2, 0)

bl_info = {
    "name": "Audio System Designer",
    "author": "Pötkö",
    "version": __version__,
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Audio Designer",
    "description": "Loudspeaker library, placement and free-field SPL prediction for live sound design",
    "category": "3D View",
}


def register():
    from . import bl
    bl.register()


def unregister():
    from . import bl
    bl.unregister()

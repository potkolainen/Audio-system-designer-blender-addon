import bpy
from bpy.props import StringProperty

from . import state


class AD_Preferences(bpy.types.AddonPreferences):
    bl_idname = state.ADDON_ID

    user_library_path: StringProperty(
        name="User library folder",
        description="Folder scanned recursively for loudspeaker .json definitions, in addition to the built-in library",
        subtype="DIR_PATH")

    def draw(self, context):
        self.layout.prop(self, "user_library_path")
        self.layout.operator("audio_designer.reload_library")


classes = (AD_Preferences,)

import bpy

from . import arrays, auto, environment, operators, prefs, props, report, state, ui, volume

_classes = (props.classes + prefs.classes + operators.classes + auto.classes + arrays.classes + volume.classes
            + report.classes + ui.classes)


def _deferred_library_load():
    state.reload_library()
    return None


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    props.register_types()
    volume.register_overlay()
    bpy.app.handlers.depsgraph_update_post.append(environment.on_depsgraph_update)
    bpy.app.timers.register(_deferred_library_load, first_interval=0.2)


def unregister():
    if bpy.app.timers.is_registered(_deferred_library_load):
        bpy.app.timers.unregister(_deferred_library_load)
    if environment.on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(environment.on_depsgraph_update)
    volume.unregister_overlay()
    props.unregister_types()
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)

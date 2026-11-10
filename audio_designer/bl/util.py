import bpy


def start_grab(context):
    """Attach the new object to the mouse (like pressing G)."""
    area = context.area
    if area is None or area.type != "VIEW_3D":
        return
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    if region is None:
        return
    try:
        with context.temp_override(area=area, region=region):
            bpy.ops.transform.translate("INVOKE_DEFAULT")
    except RuntimeError:
        pass


def select_only(context, ob):
    for o in context.view_layer.objects:
        o.select_set(False)
    ob.select_set(True)
    context.view_layer.objects.active = ob


def find_root(ob, kind):
    """The array/region root for `ob`: itself, or its parent if that has the wanted kind."""
    if ob is None:
        return None
    if ob.ad.kind == kind:
        return ob
    if ob.parent is not None and ob.parent.ad.kind == kind:
        return ob.parent
    return None


def tag_redraw(context):
    for area in context.screen.areas:
        area.tag_redraw()

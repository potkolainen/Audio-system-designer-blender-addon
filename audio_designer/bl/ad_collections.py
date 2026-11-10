"""Add-on collections: everything the add-on creates goes under one 'Audio Designer' collection."""
import bpy

ROOT_NAME = "Audio Designer"


def get_collection(scene, *path):
    """Return (creating and linking as needed) the nested collection Audio Designer/<path...>."""
    parent = scene.collection
    for name in (ROOT_NAME, *path):
        col = bpy.data.collections.get(name)
        if col is None:
            col = bpy.data.collections.new(name)
        if name not in parent.children:
            parent.children.link(col)
        parent = col
    return parent


def new_object(scene, collection, name, data):
    ob = bpy.data.objects.new(name, data)
    collection.objects.link(ob)
    ob.location = scene.cursor.location
    return ob

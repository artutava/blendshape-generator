"""
Rendering utilities for the ARKit Blendshape Generator.

This module provides helper functions for rendering the neutral mesh
from multiple cameras into temporary image files. The function
`render_neutral_views` takes care of resetting shape keys to neutral
positions, switching cameras, and restoring the user's render settings.
"""

import bpy
import os

from .logging_utils import log


def render_neutral_views(context, obj, output_dir, resolution=512):
    """Render the mesh in its neutral state from all face cameras.

    Parameters
    ----------
    context : bpy.types.Context
        The current context containing the scene.
    obj : bpy.types.Object
        The target mesh object to render.
    output_dir : str
        Directory to write the rendered images into.
    resolution : int
        Square render resolution (pixels) for both width and height.

    Returns
    -------
    dict
        A mapping of camera names to the absolute filepaths of the
        rendered images. If no cameras exist, an empty dict is returned.
    """
    scene = context.scene
    prev_camera = scene.camera
    prev_res_x = scene.render.resolution_x
    prev_res_y = scene.render.resolution_y
    prev_filepath = scene.render.filepath

    log(
        f"Rendering neutral views for '{obj.name}' to '{output_dir}' at {resolution}x{resolution}"
    )

    # Configure render resolution
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution

    # Ensure all shapekeys (except Basis) are zeroed out
    if obj.data.shape_keys:
        for kb in obj.data.shape_keys.key_blocks:
            if kb.name != "Basis":
                kb.value = 0.0

    # Find the camera collection
    coll_name = context.scene.arkit_gen_settings.camera_collection_name
    coll = bpy.data.collections.get(coll_name)
    if not coll:
        log(f"Camera collection '{coll_name}' was not found")
        return {}

    views = {}
    for cam in coll.objects:
        if cam.type != 'CAMERA':
            continue
        # Set this camera and render
        scene.camera = cam
        # Compose output filepath
        filename = f"neutral_{cam.name}.png"
        filepath = os.path.join(output_dir, filename)
        scene.render.filepath = filepath
        log(f"Rendering camera '{cam.name}' -> '{filepath}'")
        bpy.ops.render.render(write_still=True)
        views[cam.name] = filepath

    # Restore previous camera and resolution
    scene.camera = prev_camera
    scene.render.resolution_x = prev_res_x
    scene.render.resolution_y = prev_res_y
    scene.render.filepath = prev_filepath
    log(f"Rendered {len(views)} neutral views")
    return views

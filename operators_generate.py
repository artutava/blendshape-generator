"""
Generation operators for the ARKit Blendshape Generator.

These operators handle the process of rendering neutral views, invoking
an external generative AI to obtain target images for a specific
blendshape, and applying the resulting deformation back onto the mesh.
At this stage the API client and solver are stubs; they can be
extended to integrate real services and geometry processing.
"""

import bpy
import os
from bpy.types import Operator
from bpy.props import StringProperty

from .constants import ARKIT_BLENDSHAPES
from .render_utils import render_neutral_views
from .api_client import request_targets
from .solver import solve_shape


class ARKITGEN_OT_generate_blendshape(Operator):
    """Generate a specific ARKit blendshape using the configured API."""

    bl_idname = "arkitgen.generate_blendshape"
    bl_label = "Generate Blendshape"
    bl_description = "Generate the specified blendshape"

    blendshape_name: StringProperty()

    def execute(self, context):
        settings = context.scene.arkit_gen_settings
        # Find target object
        obj = context.scene.objects.get(settings.target_object_name)
        if not obj:
            self.report({'ERROR'}, "No target object set")
            return {'CANCELLED'}
        # Render neutral views to temporary directory
        tmp_dir = bpy.app.tempdir
        views = render_neutral_views(context, obj, tmp_dir, resolution=settings.render_resolution)
        if not views:
            self.report({'ERROR'}, "No cameras found; create cameras first")
            return {'CANCELLED'}
        # Request target images from the API (stub)
        try:
            targets = request_targets(settings, self.blendshape_name, views)
        except Exception as e:
            self.report({'ERROR'}, f"API error: {e}")
            return {'CANCELLED'}
        # Solve and apply deformation (stub)
        try:
            solve_shape(obj, self.blendshape_name, targets)
        except Exception as e:
            self.report({'ERROR'}, f"Solver error: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Generated blendshape '{self.blendshape_name}'")
        return {'FINISHED'}


class ARKITGEN_OT_regenerate_blendshape(Operator):
    """Regenerate a specific ARKit blendshape, overwriting any previous result."""

    bl_idname = "arkitgen.regenerate_blendshape"
    bl_label = "Regenerate Blendshape"
    bl_description = "Regenerate the specified blendshape"

    blendshape_name: StringProperty()

    def execute(self, context):
        # For now regeneration is identical to generation, but the operator
        # exists for future differentiation (e.g. backup and restore).
        return bpy.ops.arkitgen.generate_blendshape(blendshape_name=self.blendshape_name)


class ARKITGEN_OT_generate_all(Operator):
    """Generate all ARKit blendshapes sequentially."""

    bl_idname = "arkitgen.generate_all"
    bl_label = "Generate All Blendshapes"
    bl_description = "Generate all blendshapes sequentially"

    def execute(self, context):
        # Iterate through all blendshape names in a defined order. We could
        # consider grouping by difficulty later on. Break on first error.
        for name in ARKIT_BLENDSHAPES:
            result = bpy.ops.arkitgen.generate_blendshape(blendshape_name=name)
            # The operator returns a set containing 'FINISHED' or other
            # strings; check for cancellation or failure.
            if 'CANCELLED' in result or 'ERROR' in result:
                self.report({'ERROR'}, f"Stopped at '{name}' due to error")
                return {'CANCELLED'}
        self.report({'INFO'}, "Finished generating all blendshapes")
        return {'FINISHED'}

"""
Generation operators for the ARKit Blendshape Generator.

These operators handle the process of rendering neutral views, invoking
an external generative AI to obtain target images for a specific
blendshape, and applying the resulting deformation back onto the mesh.
At this stage the API client and solver are stubs; they can be
extended to integrate real services and geometry processing.
"""

import time
import os
import bpy
from bpy.types import Operator
from bpy.props import StringProperty

from .cache_utils import prepare_generation_cache_dir
from .constants import ARKIT_BLENDSHAPES
from .render_utils import render_neutral_views
from .api_client import APIQuotaExceededError, APIGenerationError, request_targets
from .logging_utils import log, log_section, set_verbose_logging
from .solver import solve_shape


class ARKITGEN_OT_generate_blendshape(Operator):
    """Generate a specific ARKit blendshape using the configured API."""

    bl_idname = "arkitgen.generate_blendshape"
    bl_label = "Generate Blendshape"
    bl_description = "Generate the specified blendshape"

    blendshape_name: StringProperty()

    def execute(self, context):
        settings = context.scene.arkit_gen_settings
        set_verbose_logging(settings.enable_verbose_logging)
        started_at = time.perf_counter()
        log_section(f"Generate Blendshape: {self.blendshape_name}")
        # Find target object
        obj = context.scene.objects.get(settings.target_object_name)
        if not obj:
            self.report({'ERROR'}, "No target object set")
            log("Generation aborted because no target object was configured")
            return {'CANCELLED'}
        # Render neutral views into a persistent preview cache when enabled.
        if settings.cache_generated_images:
            generation_dir = prepare_generation_cache_dir(settings, self.blendshape_name)
        else:
            generation_dir = os.path.join(bpy.app.tempdir, "arkit_gen_runtime")
            os.makedirs(generation_dir, exist_ok=True)
            log(f"Using temporary generation directory '{generation_dir}'")

        views = render_neutral_views(
            context,
            obj,
            generation_dir,
            resolution=settings.render_resolution,
        )
        if not views:
            self.report({'ERROR'}, "No cameras found; create cameras first")
            log("Generation aborted because no face cameras were found")
            return {'CANCELLED'}

        targets = {}
        try:
            targets = request_targets(
                settings,
                self.blendshape_name,
                views,
                output_dir=generation_dir,
            )
        except APIQuotaExceededError as e:
            log(f"Remote quota exceeded for '{self.blendshape_name}': {e}")
            if not settings.allow_procedural_fallback:
                self.report({'ERROR'}, f"API error: {e}")
                return {'CANCELLED'}
            self.report(
                {'WARNING'},
                "Quota exceeded in remote API. Continuing with local procedural fallback.",
            )
            log("Continuing with local procedural fallback because 'Fallback Solver' is enabled")
        except APIGenerationError as e:
            log(f"Remote image generation failed for '{self.blendshape_name}': {e}")
            if not settings.allow_procedural_fallback:
                self.report({'ERROR'}, f"API error: {e}")
                return {'CANCELLED'}
            self.report(
                {'WARNING'},
                "Remote image generation failed. Continuing with local procedural fallback.",
            )
        except Exception as e:
            log(f"Unexpected API error for '{self.blendshape_name}': {e}")
            if not settings.allow_procedural_fallback:
                self.report({'ERROR'}, f"API error: {e}")
                return {'CANCELLED'}
            self.report(
                {'WARNING'},
                "Unexpected API error. Continuing with local procedural fallback.",
            )

        try:
            solver_result = solve_shape(
                obj,
                self.blendshape_name,
                targets,
                settings=settings,
                neutral_views=views,
            )
        except Exception as e:
            self.report({'ERROR'}, f"Solver error: {e}")
            log(f"Solver raised an exception for '{self.blendshape_name}': {e}")
            return {'CANCELLED'}

        elapsed = time.perf_counter() - started_at
        mode = solver_result.get("mode", "unknown")
        intensity = solver_result.get("intensity", 0.0)
        self.report(
            {'INFO'},
            f"Generated blendshape '{self.blendshape_name}' using {mode} mode (strength {intensity:.2f})",
        )
        log(
            f"Blendshape '{self.blendshape_name}' finished in {elapsed:.2f}s "
            f"using mode='{mode}' intensity={intensity:.2f} cache='{generation_dir}'"
        )
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
        total = len(ARKIT_BLENDSHAPES)
        for index, name in enumerate(ARKIT_BLENDSHAPES, start=1):
            log(f"Batch progress {index}/{total}: generating '{name}'")
            result = bpy.ops.arkitgen.generate_blendshape(blendshape_name=name)
            # The operator returns a set containing 'FINISHED' or other
            # strings; check for cancellation or failure.
            if 'CANCELLED' in result or 'ERROR' in result:
                self.report({'ERROR'}, f"Stopped at '{name}' due to error")
                return {'CANCELLED'}
        self.report({'INFO'}, "Finished generating all blendshapes")
        return {'FINISHED'}

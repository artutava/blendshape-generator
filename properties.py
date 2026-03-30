"""
Property definitions for the ARKit Blendshape Generator.

The property group defined here stores all configuration options and
runtime settings used by the add-on. These properties appear in the UI
panel and are saved with the Blender file.
"""

import bpy
from bpy.types import PropertyGroup
from bpy.props import (
    StringProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    BoolProperty,
)


class ARKITGenSettings(PropertyGroup):
    """Configuration and runtime settings for the ARKit generator."""

    target_object_name: StringProperty(
        name="Target Object",
        description="Name of the mesh object to operate on",
        default="",
    )

    head_vertex_group: StringProperty(
        name="Head Vertex Group",
        description="Vertex group name defining the head region",
        default="FACE_HEAD",
    )

    camera_collection_name: StringProperty(
        name="Camera Collection",
        description="Collection name for face cameras",
        default="FACEGEN_CAMERAS",
    )

    api_endpoint: StringProperty(
        name="API Endpoint",
        description="URL of the generative AI API",
        default="",
    )

    api_key: StringProperty(
        name="API Key",
        description="API key for authentication",
        default="",
        subtype='PASSWORD',
    )

    model_name: StringProperty(
        name="Model Name",
        description="Name of the remote AI model",
        default="",
    )

    style_preset: EnumProperty(
        name="Style",
        description="Style preset for expressions",
        items=[
            ('realistic', "Realistic", "Natural, anatomically plausible style"),
            ('pixar', "Pixar", "Stylized, animation-friendly style"),
            ('stylized', "Stylized", "Generic stylized expressions"),
        ],
        default='realistic',
    )

    default_intensity: FloatProperty(
        name="Intensity",
        description="Default expression intensity (0 to 1)",
        default=1.0,
        min=0.0,
        max=1.0,
    )

    render_resolution: IntProperty(
        name="Resolution",
        description="Square render resolution in pixels",
        default=512,
        min=128,
        max=4096,
    )

    num_candidates: IntProperty(
        name="Candidates",
        description="Number of target variations to request",
        default=1,
        min=1,
        max=5,
    )

    show_advanced: BoolProperty(
        name="Show Advanced",
        description="Show advanced API settings",
        default=False,
    )

    enable_verbose_logging: BoolProperty(
        name="Verbose Logs",
        description="Print detailed execution logs to Blender's terminal",
        default=True,
    )

    allow_procedural_fallback: BoolProperty(
        name="Fallback Solver",
        description="If the remote image API fails, still generate a procedural blendshape locally",
        default=True,
    )

    cache_generated_images: BoolProperty(
        name="Cache Images",
        description="Save neutral and generated images to a persistent folder for preview",
        default=True,
    )

    cache_directory: StringProperty(
        name="Cache Folder",
        description="Optional folder for persistent neutral/generated image previews",
        default="",
        subtype='DIR_PATH',
    )

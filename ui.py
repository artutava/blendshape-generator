"""
UI definitions for the ARKit Blendshape Generator.

This module defines a panel that appears in the 3D Viewport sidebar. It
exposes controls for selecting a target mesh, creating head groups,
camera rigs, and shapekeys, configuring API settings, and triggering
blendshape generation.
"""

import bpy

from bpy.types import Panel

from .constants import ARKIT_BLENDSHAPES


class ARKITGEN_PT_main_panel(Panel):
    """Main panel for the ARKit generator interface."""

    bl_label = "ARKit Blendshape Generator"
    bl_idname = "ARKITGEN_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ARKit Gen'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.arkit_gen_settings

        # Target selection
        box = layout.box()
        box.label(text="Target Mesh", icon='MESH_CUBE')
        if settings.target_object_name:
            box.label(text=f"Object: {settings.target_object_name}")
        else:
            box.label(text="No object selected", icon='ERROR')
        row = box.row(align=True)
        row.operator("arkitgen.use_selected_object", text="Use Selected Object")

        # Head region
        box = layout.box()
        box.label(text="Head Region", icon='GROUP_VERTEX')
        row = box.row(align=True)
        row.operator("arkitgen.create_head_group", text="Create Head Group")
        row.operator("arkitgen.capture_head_selection", text="Use Selection")
        box.label(text=f"Group: {settings.head_vertex_group}")

        # Camera rig
        box = layout.box()
        box.label(text="Camera Rig", icon='CAMERA_DATA')
        row = box.row(align=True)
        row.operator("arkitgen.create_cameras", text="Create Face Cameras")
        box.label(text=f"Collection: {settings.camera_collection_name}")

        # Shape keys
        box = layout.box()
        box.label(text="Shape Keys", icon='SHAPEKEY_DATA')
        row = box.row(align=True)
        row.operator("arkitgen.create_arkit_shapekeys", text="Create ARKit Shapekeys")

        # API settings
        box = layout.box()
        box.label(text="API Settings", icon='PREFERENCES')
        box.prop(settings, "api_endpoint")
        box.prop(settings, "api_key")
        box.prop(settings, "model_name")
        box.prop(settings, "style_preset")
        box.prop(settings, "default_intensity")
        box.prop(settings, "render_resolution")
        box.prop(settings, "num_candidates")

        # Blendshape list and generation buttons
        box = layout.box()
        box.label(text="Blendshapes", icon='SHAPEKEY_DATA')
        for name in ARKIT_BLENDSHAPES:
            row = box.row(align=True)
            row.label(text=name)
            op = row.operator("arkitgen.generate_blendshape", text="Generate")
            op.blendshape_name = name
            op = row.operator("arkitgen.regenerate_blendshape", text="Regenerate")
            op.blendshape_name = name

        # Batch operations
        row = layout.row(align=True)
        row.operator("arkitgen.generate_all", text="Generate All", icon='SEQ_SEQUENCER')

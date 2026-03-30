"""
Setup operators for the ARKit Blendshape Generator.

These operators perform initial setup tasks: assigning a mesh as the
target, creating a vertex group for the head, capturing the current
selection into that group, building a multiview camera rig, and adding
empty ARKit-compatible shapekeys.
"""

import bpy
import mathutils
import bmesh

from bpy.types import Operator

from .constants import ARKIT_BLENDSHAPES


class ARKITGEN_OT_use_selected_object(Operator):
    """Assign the currently selected mesh as the target for generation."""

    bl_idname = "arkitgen.use_selected_object"
    bl_label = "Use Selected Object"
    bl_description = "Assign the currently selected mesh as the target"

    def execute(self, context):
        settings = context.scene.arkit_gen_settings
        obj = context.object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected")
            return {'CANCELLED'}
        settings.target_object_name = obj.name
        self.report({'INFO'}, f"Target set to '{obj.name}'")
        return {'FINISHED'}


class ARKITGEN_OT_create_head_group(Operator):
    """Create an empty vertex group on the target object for the head region."""

    bl_idname = "arkitgen.create_head_group"
    bl_label = "Create Head Group"
    bl_description = "Create an empty vertex group on the target for the head region"

    def execute(self, context):
        settings = context.scene.arkit_gen_settings
        obj = context.scene.objects.get(settings.target_object_name)
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first")
            return {'CANCELLED'}
        vg_name = settings.head_vertex_group
        if vg_name not in obj.vertex_groups:
            obj.vertex_groups.new(name=vg_name)
            self.report({'INFO'}, f"Created vertex group '{vg_name}'")
        else:
            self.report({'INFO'}, f"Vertex group '{vg_name}' already exists")
        return {'FINISHED'}


class ARKITGEN_OT_capture_head_selection(Operator):
    """Assign selected vertices to the head vertex group."""

    bl_idname = "arkitgen.capture_head_selection"
    bl_label = "Use Selection as Head"
    bl_description = "Assign selected vertices to the head vertex group"

    def execute(self, context):
        settings = context.scene.arkit_gen_settings
        obj = context.scene.objects.get(settings.target_object_name)
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first")
            return {'CANCELLED'}
        vg_name = settings.head_vertex_group
        if vg_name not in obj.vertex_groups:
            self.report({'ERROR'}, f"Vertex group '{vg_name}' does not exist")
            return {'CANCELLED'}
        # Ensure edit mode
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Enter edit mode and select vertices")
            return {'CANCELLED'}
        vg = obj.vertex_groups[vg_name]
        bm = bmesh.from_edit_mesh(obj.data)
        selected_verts = [v.index for v in bm.verts if v.select]
        if not selected_verts:
            self.report({'ERROR'}, "No vertices selected")
            return {'CANCELLED'}
        for v_idx in selected_verts:
            vg.add([v_idx], 1.0, 'REPLACE')
        self.report({'INFO'}, f"Assigned {len(selected_verts)} vertices to '{vg_name}'")
        return {'FINISHED'}


class ARKITGEN_OT_create_cameras(Operator):
    """Create a rig of face cameras around the head for multiview rendering."""

    bl_idname = "arkitgen.create_cameras"
    bl_label = "Create Face Cameras"
    bl_description = "Create a camera rig for multiview renders"

    def execute(self, context):
        settings = context.scene.arkit_gen_settings
        obj = context.scene.objects.get(settings.target_object_name)
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first")
            return {'CANCELLED'}

        collection_name = settings.camera_collection_name
        # Remove existing camera collection if it exists
        if collection_name in bpy.data.collections:
            coll = bpy.data.collections[collection_name]
            # Unlink objects and remove them
            for ob in list(coll.objects):
                bpy.data.objects.remove(ob, do_unlink=True)
            bpy.data.collections.remove(coll)
        # Create new collection
        coll = bpy.data.collections.new(collection_name)
        context.scene.collection.children.link(coll)

        # Compute approximate center of the object for camera placement
        depsgraph = context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        bbox_center = sum((obj_eval.matrix_world @ mathutils.Vector(corner) for corner in obj_eval.bound_box), mathutils.Vector()) / 8.0

        # Create an empty target to act as the look-at point
        empty = bpy.data.objects.new("FACEGEN_TARGET", None)
        empty.location = bbox_center
        coll.objects.link(empty)

        # Define relative camera positions (local offsets from center)
        positions = {
            "Front": (0.0, -2.0, 0.0),
            "ThreeQuarterLeft": (-1.5, -1.5, 0.0),
            "ThreeQuarterRight": (1.5, -1.5, 0.0),
            "ProfileLeft": (-2.0, 0.0, 0.0),
            "ProfileRight": (2.0, 0.0, 0.0),
        }

        for cam_name, offset in positions.items():
            cam_data = bpy.data.cameras.new(name=cam_name)
            cam = bpy.data.objects.new(cam_name, cam_data)
            # Position the camera relative to center
            cam.location = bbox_center + mathutils.Vector(offset)
            cam.data.lens = 50.0
            cam.data.type = 'PERSP'
            # Link to collection and parent to empty
            coll.objects.link(cam)
            cam.parent = empty
            # Add a Track To constraint so the camera looks at the empty
            constraint = cam.constraints.new(type='TRACK_TO')
            constraint.target = empty
            constraint.track_axis = 'TRACK_NEGATIVE_Z'
            constraint.up_axis = 'UP_Y'

        self.report({'INFO'}, "Created face cameras")
        return {'FINISHED'}


class ARKITGEN_OT_create_arkit_shapekeys(Operator):
    """Create empty shapekeys for each ARKit blendshape on the target object."""

    bl_idname = "arkitgen.create_arkit_shapekeys"
    bl_label = "Create ARKit Shapekeys"
    bl_description = "Create empty shapekeys for ARKit blendshapes"

    def execute(self, context):
        settings = context.scene.arkit_gen_settings
        obj = context.scene.objects.get(settings.target_object_name)
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first")
            return {'CANCELLED'}
        # Ensure a Basis shape key exists
        if not obj.data.shape_keys:
            obj.shape_key_add(name="Basis")
        added = 0
        for name in ARKIT_BLENDSHAPES:
            if name not in obj.data.shape_keys.key_blocks:
                obj.shape_key_add(name=name)
                added += 1
        self.report({'INFO'}, f"Added {added} shapekeys")
        return {'FINISHED'}

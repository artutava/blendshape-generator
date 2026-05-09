"""
Geometry transfer utilities for Hunyuan-driven blendshape solving.

This module imports the neutral and expressive reconstructions generated
from the frontal renders, aligns them to the original mesh, and
transfers the reconstruction delta back to the target shape key.
"""

from __future__ import annotations

import os

import bpy
import mathutils
import numpy as np
from mathutils import kdtree

from .logging_utils import log


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _band(x: float, start: float, peak_start: float, peak_end: float, end: float) -> float:
    if x <= start or x >= end:
        return 0.0
    if peak_start <= x <= peak_end:
        return 1.0
    if x < peak_start:
        return _smoothstep(start, peak_start, x)
    return 1.0 - _smoothstep(peak_end, end, x)


def _build_masks(nx: float, ny: float, nz: float) -> dict:
    front = _smoothstep(0.10, 0.70, 1.0 - ny)
    center_x = 1.0 - min(abs(nx - 0.5) * 2.0, 1.0)
    outer_x = 1.0 - center_x
    return {
        "left": _smoothstep(0.0, 0.42, 1.0 - nx),
        "right": _smoothstep(0.0, 0.42, nx),
        "center": _smoothstep(0.15, 0.75, center_x),
        "outer": _smoothstep(0.10, 0.65, outer_x),
        "front": front,
        "lower": 1.0 - _smoothstep(0.34, 0.70, nz),
        "upper": _smoothstep(0.56, 0.92, nz),
        "mouth": _band(nz, 0.04, 0.10, 0.36, 0.50) * front,
        "upper_lip": _band(nz, 0.18, 0.24, 0.34, 0.42) * front,
        "lower_lip": _band(nz, 0.04, 0.08, 0.18, 0.28) * front,
        "brow": _band(nz, 0.74, 0.80, 0.94, 1.02) * front,
        "eye": _band(nz, 0.48, 0.56, 0.72, 0.80) * front,
        "nose": _band(nz, 0.36, 0.44, 0.62, 0.72) * front,
        "cheek": _band(nz, 0.22, 0.30, 0.56, 0.70) * front,
        "corner": _band(center_x, -0.01, 0.00, 0.22, 0.42) * front,
    }


def _profile_gate(blendshape_name: str, masks: dict) -> float:
    left = masks["left"]
    right = masks["right"]
    center = masks["center"]
    outer = masks["outer"]
    lower = masks["lower"]
    mouth = masks["mouth"]
    upper_lip = masks["upper_lip"]
    lower_lip = masks["lower_lip"]
    brow = masks["brow"]
    eye = masks["eye"]
    nose = masks["nose"]
    cheek = masks["cheek"]
    corner = masks["corner"]

    if blendshape_name == "browInnerUp":
        return brow * center * (1.0 - nose)
    if blendshape_name == "browDownLeft":
        return brow * left
    if blendshape_name == "browDownRight":
        return brow * right
    if blendshape_name == "browOuterUpLeft":
        return brow * left * outer
    if blendshape_name == "browOuterUpRight":
        return brow * right * outer
    if blendshape_name == "eyeBlinkLeft":
        return eye * left
    if blendshape_name == "eyeBlinkRight":
        return eye * right
    if blendshape_name == "eyeSquintLeft":
        return max(eye * left, cheek * left)
    if blendshape_name == "eyeSquintRight":
        return max(eye * right, cheek * right)
    if blendshape_name == "eyeWideLeft":
        return eye * left
    if blendshape_name == "eyeWideRight":
        return eye * right
    if blendshape_name == "noseSneerLeft":
        return max(nose * left, cheek * left * 0.6)
    if blendshape_name == "noseSneerRight":
        return max(nose * right, cheek * right * 0.6)
    if blendshape_name.startswith("jaw"):
        return lower * (0.55 + 0.45 * center)
    if blendshape_name.startswith("mouth") or blendshape_name == "tongueOut":
        return max(mouth, upper_lip, lower_lip, corner)
    if blendshape_name.startswith("cheek"):
        return cheek
    return max(brow, eye, nose, mouth, cheek)


def _axis_weights(blendshape_name: str) -> tuple[float, float, float]:
    if blendshape_name.startswith("brow"):
        return (0.15, 0.10, 1.00)
    if blendshape_name.startswith("eye"):
        return (0.10, 0.10, 1.00)
    if blendshape_name.startswith("nose"):
        return (0.10, 0.55, 1.00)
    if blendshape_name == "jawForward":
        return (0.10, 1.00, 0.45)
    if blendshape_name in {"jawLeft", "jawRight", "mouthLeft", "mouthRight", "mouthStretchLeft", "mouthStretchRight"}:
        return (1.00, 0.20, 0.35)
    if blendshape_name.startswith("jaw"):
        return (0.20, 0.55, 1.00)
    if blendshape_name.startswith("mouthSmile") or blendshape_name.startswith("mouthFrown"):
        return (1.00, 0.20, 1.00)
    if blendshape_name.startswith("mouth") or blendshape_name == "tongueOut":
        return (0.45, 1.00, 1.00)
    if blendshape_name.startswith("cheek"):
        return (0.35, 1.00, 0.45)
    return (0.60, 0.60, 0.60)


def _build_adjacency(obj: bpy.types.Object, head_indices: list[int]) -> dict[int, list[int]]:
    head_set = set(head_indices)
    adjacency = {index: [] for index in head_indices}
    for edge in obj.data.edges:
        v1 = edge.vertices[0]
        v2 = edge.vertices[1]
        if v1 in head_set and v2 in head_set:
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
    return adjacency


def _smooth_deltas(
    deltas: dict[int, mathutils.Vector],
    weights: dict[int, float],
    adjacency: dict[int, list[int]],
    iterations: int = 8,
    blend_factor: float = 0.72,
) -> dict[int, mathutils.Vector]:
    current = {index: value.copy() for index, value in deltas.items()}
    for _ in range(iterations):
        updated = {}
        for index, delta in current.items():
            neighbors = adjacency.get(index) or []
            if not neighbors:
                updated[index] = delta.copy()
                continue
            neighbor_values = [current[neighbor] for neighbor in neighbors if neighbor in current]
            if not neighbor_values:
                updated[index] = delta.copy()
                continue
            average = mathutils.Vector((0.0, 0.0, 0.0))
            for neighbor_delta in neighbor_values:
                average += neighbor_delta
            average /= len(neighbor_values)
            gate = _clamp(weights.get(index, 0.0), 0.0, 1.0)
            local_blend = blend_factor * (0.35 + 0.65 * gate)
            updated[index] = delta.lerp(average, local_blend)
        current = updated
    return current


def _import_mesh(filepath: str, prefix: str) -> list[bpy.types.Object]:
    if not filepath or not os.path.exists(filepath):
        raise RuntimeError(f"Reconstruction mesh does not exist: '{filepath}'")

    extension = os.path.splitext(filepath)[1].lower()
    before_names = set(bpy.data.objects.keys())

    if extension == ".glb":
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif extension == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif extension == ".ply":
        bpy.ops.wm.ply_import(filepath=filepath)
    else:
        raise RuntimeError(f"Unsupported reconstruction mesh format '{extension}'")

    imported = [bpy.data.objects[name] for name in bpy.data.objects.keys() if name not in before_names]
    mesh_objects = [obj for obj in imported if obj.type == 'MESH']
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects were imported from '{filepath}'")

    for index, obj in enumerate(mesh_objects, start=1):
        obj.name = f"{prefix}_{index}"
    return mesh_objects


def _cleanup_objects(objects: list[bpy.types.Object]) -> None:
    if not objects:
        return
    meshes = []
    for obj in objects:
        if obj.type == 'MESH':
            meshes.append(obj.data)
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _gather_world_vertices(objects: list[bpy.types.Object]) -> np.ndarray:
    points = []
    for obj in objects:
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            points.append((world.x, world.y, world.z))
    if not points:
        raise RuntimeError("Imported reconstruction mesh has no vertices")
    return np.asarray(points, dtype=np.float64)


def _bbox_diagonal(points: np.ndarray) -> float:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return float(np.linalg.norm(maximum - minimum))


def _rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    src_zero = src - src_center
    dst_zero = dst - dst_center
    covariance = src_zero.T @ dst_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = dst_center - (rotation @ src_center)
    return rotation, translation


def _apply_rigid_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (points @ rotation.T) + translation


def _build_kdtree(points: np.ndarray) -> kdtree.KDTree:
    tree = kdtree.KDTree(len(points))
    for index, point in enumerate(points):
        tree.insert(point, index)
    tree.balance()
    return tree


def _align_reconstruction(neutral_points: np.ndarray, target_points: np.ndarray, reference_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = _bbox_diagonal(reference_points) / max(_bbox_diagonal(neutral_points), 1e-8)
    neutral_aligned = neutral_points * scale
    target_aligned = target_points * scale

    neutral_center = neutral_aligned.mean(axis=0)
    reference_center = reference_points.mean(axis=0)
    offset = reference_center - neutral_center
    neutral_aligned = neutral_aligned + offset
    target_aligned = target_aligned + offset

    reference_tree = _build_kdtree(reference_points)
    sample_count = min(len(neutral_aligned), 1500)
    if sample_count < len(neutral_aligned):
        sample_indices = np.linspace(0, len(neutral_aligned) - 1, sample_count, dtype=int)
    else:
        sample_indices = np.arange(len(neutral_aligned))

    for _ in range(6):
        src = neutral_aligned[sample_indices]
        dst = np.zeros_like(src)
        for sample_index, point in enumerate(src):
            nearest_co, _, _ = reference_tree.find(point)
            dst[sample_index] = np.asarray(nearest_co, dtype=np.float64)
        rotation, translation = _rigid_fit(src, dst)
        neutral_aligned = _apply_rigid_transform(neutral_aligned, rotation, translation)
        target_aligned = _apply_rigid_transform(target_aligned, rotation, translation)

    return neutral_aligned, target_aligned


def apply_reconstruction_transfer(
    obj: bpy.types.Object,
    basis_key,
    target_key,
    head_indices: list[int],
    blendshape_name: str,
    strength: float,
    reconstruction_result: dict,
) -> dict | None:
    neutral_path = reconstruction_result.get("neutral_mesh_path")
    target_path = reconstruction_result.get("target_mesh_path")
    if reconstruction_result.get("status") != "finished" or not neutral_path or not target_path:
        return None

    neutral_objects = []
    target_objects = []
    try:
        neutral_objects = _import_mesh(neutral_path, f"HunyuanNeutral_{blendshape_name}")
        target_objects = _import_mesh(target_path, f"HunyuanTarget_{blendshape_name}")
        neutral_points = _gather_world_vertices(neutral_objects)
        target_points = _gather_world_vertices(target_objects)
    except Exception as exc:
        _cleanup_objects(neutral_objects)
        _cleanup_objects(target_objects)
        log(f"Could not import reconstruction meshes for transfer: {exc}")
        return None

    try:
        basis_world = np.asarray(
            [
                tuple(obj.matrix_world @ basis_key.data[index].co)
                for index in head_indices
            ],
            dtype=np.float64,
        )
        neutral_aligned, target_aligned = _align_reconstruction(
            neutral_points,
            target_points,
            basis_world,
        )
        reconstruction_delta = target_aligned - neutral_aligned
        neutral_tree = _build_kdtree(neutral_aligned)
        inv_world = obj.matrix_world.inverted().to_3x3()

        min_coords = basis_world.min(axis=0)
        max_coords = basis_world.max(axis=0)
        size = np.maximum(max_coords - min_coords, 1e-8)
        distance_limit = _bbox_diagonal(basis_world) * 0.08
        axis_weights = _axis_weights(blendshape_name)
        max_component = _bbox_diagonal(basis_world) * 0.018
        adjacency = _build_adjacency(obj, head_indices)

        for vertex in obj.data.vertices:
            target_key.data[vertex.index].co = basis_key.data[vertex.index].co

        candidate_deltas: dict[int, mathutils.Vector] = {}
        candidate_weights: dict[int, float] = {}
        for offset_index, vertex_index in enumerate(head_indices):
            basis_world_co = basis_world[offset_index]
            normalized = (basis_world_co - min_coords) / size
            masks = _build_masks(
                float(normalized[0]),
                float(normalized[1]),
                float(normalized[2]),
            )
            semantic_gate = _profile_gate(blendshape_name, masks)
            if semantic_gate <= 1e-4:
                continue

            nearest = neutral_tree.find_n(basis_world_co, 4)
            weighted_delta = np.zeros(3, dtype=np.float64)
            weight_sum = 0.0
            nearest_distance = None
            for nearest_co, nearest_index, distance in nearest:
                if nearest_distance is None:
                    nearest_distance = distance
                if distance > distance_limit:
                    continue
                weight = 1.0 / max(distance, 1e-5)
                weighted_delta += reconstruction_delta[nearest_index] * weight
                weight_sum += weight

            if weight_sum <= 1e-8 or nearest_distance is None:
                continue

            delta_world = weighted_delta / weight_sum
            distance_gate = 1.0 - _smoothstep(distance_limit * 0.25, distance_limit, nearest_distance)
            if distance_gate <= 1e-4:
                continue

            delta_world = delta_world * strength * semantic_gate * distance_gate
            delta_world = delta_world * np.asarray(axis_weights, dtype=np.float64)
            delta_world = np.clip(delta_world, -max_component, max_component)

            delta_local = inv_world @ mathutils.Vector(delta_world.tolist())
            if delta_local.length <= 3e-5:
                continue

            candidate_deltas[vertex_index] = delta_local
            candidate_weights[vertex_index] = semantic_gate * distance_gate

        smoothed_deltas = _smooth_deltas(candidate_deltas, candidate_weights, adjacency)

        moved_vertices = 0
        max_delta = 0.0
        for vertex_index, delta_local in smoothed_deltas.items():
            if delta_local.length <= 5e-5:
                continue
            basis_local = basis_key.data[vertex_index].co.copy()
            target_key.data[vertex_index].co = basis_local + delta_local
            moved_vertices += 1
            max_delta = max(max_delta, float(delta_local.length))

        log(
            f"Transferred Hunyuan reconstruction delta for '{blendshape_name}' "
            f"over {moved_vertices} vertices (raw_candidates={len(candidate_deltas)} "
            f"max_local_delta={max_delta:.5f})"
        )
        return {
            "mode": "hunyuan-transfer",
            "intensity": strength,
            "vertex_count": moved_vertices,
            "max_delta": max_delta,
        }
    finally:
        _cleanup_objects(neutral_objects)
        _cleanup_objects(target_objects)

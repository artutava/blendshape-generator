"""
Image-guided solver for applying blendshape deformations.

The generated Gemini images are compared against neutral renders using
OpenCV. We then project mesh vertices into each camera view, sample the
per-pixel attention maps, and apply only blendshape-specific motions in
the regions that visually changed. This is still heuristic, but it is
far more localized than a global procedural fallback.
"""

import bpy
import cv2
import mathutils
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

from .logging_utils import log
from .reconstruction_transfer import apply_reconstruction_transfer

try:
    import mediapipe as mp
except Exception:
    mp = None


FACE_MESH = None
MEDIAPIPE_BACKEND = "unavailable"
MEDIAPIPE_INIT_ERROR = None


def _init_mediapipe_backend():
    global FACE_MESH, MEDIAPIPE_BACKEND, MEDIAPIPE_INIT_ERROR
    if mp is None:
        MEDIAPIPE_BACKEND = "unavailable"
        return

    try:
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "face_mesh"):
            FACE_MESH = solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            MEDIAPIPE_BACKEND = "solutions"
            return

        MEDIAPIPE_BACKEND = "tasks-model-required"
        MEDIAPIPE_INIT_ERROR = (
            "This MediaPipe build exposes the Tasks API instead of mp.solutions. "
            "A Face Landmarker .task model is required before landmark detection can be enabled."
        )
    except Exception as exc:
        FACE_MESH = None
        MEDIAPIPE_BACKEND = "init-failed"
        MEDIAPIPE_INIT_ERROR = str(exc)


_init_mediapipe_backend()


REGION_LANDMARKS = {
    "brow_left": [46, 52, 53, 55, 63, 65, 66, 70],
    "brow_right": [276, 282, 283, 285, 293, 295, 296, 300],
    "brow_inner": [55, 65, 52, 53, 285, 295, 282, 283, 9, 107, 336],
    "eye_left": [33, 133, 159, 145, 160, 144, 158, 153, 157, 173],
    "eye_right": [362, 263, 386, 374, 387, 373, 385, 380, 384, 398],
    "nose": [1, 2, 4, 5, 6, 19, 94, 97, 98, 168, 195, 197, 327, 326],
    "mouth": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
    "cheek_left": [50, 101, 118, 119, 120, 123, 126, 142, 187, 205, 206, 207],
    "cheek_right": [280, 329, 347, 348, 349, 352, 355, 371, 411, 425, 426, 427],
}

BLENDSHAPE_REGIONS = {
    "browDownLeft": ["brow_left"],
    "browDownRight": ["brow_right"],
    "browInnerUp": ["brow_inner"],
    "browOuterUpLeft": ["brow_left"],
    "browOuterUpRight": ["brow_right"],
    "cheekPuff": ["cheek_left", "cheek_right"],
    "cheekSquintLeft": ["cheek_left", "eye_left"],
    "cheekSquintRight": ["cheek_right", "eye_right"],
    "eyeBlinkLeft": ["eye_left"],
    "eyeBlinkRight": ["eye_right"],
    "eyeLookDownLeft": ["eye_left"],
    "eyeLookDownRight": ["eye_right"],
    "eyeLookInLeft": ["eye_left"],
    "eyeLookInRight": ["eye_right"],
    "eyeLookOutLeft": ["eye_left"],
    "eyeLookOutRight": ["eye_right"],
    "eyeLookUpLeft": ["eye_left"],
    "eyeLookUpRight": ["eye_right"],
    "eyeSquintLeft": ["eye_left", "cheek_left"],
    "eyeSquintRight": ["eye_right", "cheek_right"],
    "eyeWideLeft": ["eye_left"],
    "eyeWideRight": ["eye_right"],
    "jawForward": ["mouth"],
    "jawLeft": ["mouth"],
    "jawOpen": ["mouth"],
    "jawRight": ["mouth"],
    "mouthClose": ["mouth"],
    "mouthDimpleLeft": ["mouth", "cheek_left"],
    "mouthDimpleRight": ["mouth", "cheek_right"],
    "mouthFrownLeft": ["mouth"],
    "mouthFrownRight": ["mouth"],
    "mouthFunnel": ["mouth"],
    "mouthLeft": ["mouth"],
    "mouthLowerDownLeft": ["mouth"],
    "mouthLowerDownRight": ["mouth"],
    "mouthPressLeft": ["mouth"],
    "mouthPressRight": ["mouth"],
    "mouthPucker": ["mouth"],
    "mouthRight": ["mouth"],
    "mouthRollLower": ["mouth"],
    "mouthRollUpper": ["mouth"],
    "mouthShrugLower": ["mouth"],
    "mouthShrugUpper": ["mouth"],
    "mouthSmileLeft": ["mouth", "cheek_left"],
    "mouthSmileRight": ["mouth", "cheek_right"],
    "mouthStretchLeft": ["mouth"],
    "mouthStretchRight": ["mouth"],
    "mouthUpperUpLeft": ["mouth"],
    "mouthUpperUpRight": ["mouth"],
    "noseSneerLeft": ["nose", "cheek_left"],
    "noseSneerRight": ["nose", "cheek_right"],
    "tongueOut": ["mouth"],
}


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


def _safe_percentile(values: np.ndarray, percentile: float, fallback: float) -> float:
    if values.size == 0:
        return fallback
    return float(np.percentile(values, percentile))


def _detect_face_landmarks(image_bgr: np.ndarray):
    if FACE_MESH is None:
        return None

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = FACE_MESH.process(rgb)
    if not result.multi_face_landmarks:
        return None

    face = result.multi_face_landmarks[0]
    points = np.array([[lm.x, lm.y] for lm in face.landmark], dtype=np.float32)
    return points


def _region_mask_from_landmarks(image_shape, landmarks: np.ndarray, indices: list[int]) -> np.ndarray:
    height, width = image_shape[:2]
    points = []
    for index in indices:
        if index >= len(landmarks):
            continue
        x = int(_clamp(landmarks[index, 0], 0.0, 1.0) * (width - 1))
        y = int(_clamp(landmarks[index, 1], 0.0, 1.0) * (height - 1))
        points.append((x, y))

    if len(points) < 3:
        return np.zeros((height, width), dtype=np.float32)

    contour = cv2.convexHull(np.array(points, dtype=np.int32))
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillConvexPoly(mask, contour, 1.0)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    max_value = float(mask.max())
    if max_value > 1e-6:
        mask /= max_value
    return mask


def _build_region_masks(image_shape, landmarks: np.ndarray) -> dict:
    return {
        region_name: _region_mask_from_landmarks(image_shape, landmarks, indices)
        for region_name, indices in REGION_LANDMARKS.items()
    }


def _compute_face_scale(landmarks: np.ndarray) -> float:
    min_xy = landmarks.min(axis=0)
    max_xy = landmarks.max(axis=0)
    diag = np.linalg.norm(max_xy - min_xy)
    return max(float(diag), 1e-4)


def _compute_region_scores(neutral_landmarks: np.ndarray, target_landmarks: np.ndarray) -> dict:
    face_scale = _compute_face_scale(neutral_landmarks)
    scores = {}
    for region_name, indices in REGION_LANDMARKS.items():
        valid_indices = [idx for idx in indices if idx < len(neutral_landmarks) and idx < len(target_landmarks)]
        if not valid_indices:
            scores[region_name] = 0.0
            continue
        neutral_points = neutral_landmarks[valid_indices]
        target_points = target_landmarks[valid_indices]
        distances = np.linalg.norm(target_points - neutral_points, axis=1)
        score = float(np.mean(distances) / face_scale)
        scores[region_name] = score
    return scores


def _merge_region_masks(region_masks: dict, region_names: list[str], shape) -> np.ndarray:
    height, width = shape[:2]
    merged = np.zeros((height, width), dtype=np.float32)
    for region_name in region_names:
        mask = region_masks.get(region_name)
        if mask is None:
            continue
        merged = np.maximum(merged, mask)
    return merged


def _load_view_analysis(neutral_path: str, target_path: str) -> dict:
    neutral = cv2.imread(neutral_path, cv2.IMREAD_COLOR)
    target = cv2.imread(target_path, cv2.IMREAD_COLOR)
    if neutral is None:
        raise RuntimeError(f"Could not read neutral image '{neutral_path}'")
    if target is None:
        raise RuntimeError(f"Could not read generated image '{target_path}'")

    if neutral.shape[:2] != target.shape[:2]:
        target = cv2.resize(target, (neutral.shape[1], neutral.shape[0]), interpolation=cv2.INTER_CUBIC)

    neutral_gray = cv2.cvtColor(neutral, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    neutral_blur = cv2.GaussianBlur(neutral_gray, (5, 5), 0)
    target_blur = cv2.GaussianBlur(target_gray, (5, 5), 0)

    diff = cv2.absdiff(target_blur, neutral_blur).astype(np.float32) / 255.0
    flow = cv2.calcOpticalFlowFarneback(
        neutral_blur,
        target_blur,
        None,
        0.5,
        3,
        21,
        5,
        5,
        1.2,
        0,
    )
    flow_mag = np.linalg.norm(flow, axis=2)
    diff_denom = _safe_percentile(diff, 99.0, 1.0)
    flow_denom = _safe_percentile(flow_mag, 99.0, 1.0)
    diff_norm = np.clip(diff / max(diff_denom, 1e-6), 0.0, 1.0)
    flow_norm = np.clip(flow_mag / max(flow_denom, 1e-6), 0.0, 1.0)
    attention = np.clip((diff_norm * 0.75) + (flow_norm * 0.25), 0.0, 1.0)
    neutral_landmarks = _detect_face_landmarks(neutral)
    target_landmarks = _detect_face_landmarks(target)
    region_masks = {}
    region_scores = {}
    detection_mode = "opencv-only"
    if neutral_landmarks is not None and target_landmarks is not None:
        region_masks = _build_region_masks(neutral.shape, neutral_landmarks)
        region_scores = _compute_region_scores(neutral_landmarks, target_landmarks)
        detection_mode = "mediapipe-opencv"

    return {
        "attention": attention,
        "flow": flow,
        "height": int(attention.shape[0]),
        "width": int(attention.shape[1]),
        "score": float(np.mean(np.sort(attention.reshape(-1))[-4096:])),
        "region_masks": region_masks,
        "region_scores": region_scores,
        "has_landmarks": neutral_landmarks is not None and target_landmarks is not None,
        "neutral_landmarks_detected": neutral_landmarks is not None,
        "target_landmarks_detected": target_landmarks is not None,
        "detection_mode": detection_mode,
    }


def _sample_map(map_data: np.ndarray, x_norm: float, y_norm: float) -> float:
    height, width = map_data.shape[:2]
    px = _clamp(x_norm, 0.0, 1.0) * (width - 1)
    py = (1.0 - _clamp(y_norm, 0.0, 1.0)) * (height - 1)
    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = px - x0
    ty = py - y0

    top = (map_data[y0, x0] * (1.0 - tx)) + (map_data[y0, x1] * tx)
    bottom = (map_data[y1, x0] * (1.0 - tx)) + (map_data[y1, x1] * tx)
    return float((top * (1.0 - ty)) + (bottom * ty))


def _estimate_expression_strength(analyses: dict, blendshape_name: str, default_intensity: float) -> float:
    if not analyses:
        return _clamp(default_intensity, 0.15, 1.0)

    scores = []
    region_scores = []
    relevant_regions = BLENDSHAPE_REGIONS.get(blendshape_name, [])
    for view_name, analysis in analyses.items():
        score = analysis["score"]
        scores.append(score)
        log(f"Image attention score for view '{view_name}': {score:.4f}")
        if analysis.get("has_landmarks"):
            per_view_scores = analysis.get("region_scores", {})
            if relevant_regions:
                region_score = max(per_view_scores.get(region, 0.0) for region in relevant_regions)
                region_scores.append(region_score)
                log(f"Landmark motion score for view '{view_name}': {region_score:.4f}")

    average_score = sum(scores) / len(scores)
    estimated = average_score * 1.1
    if region_scores:
        estimated += (sum(region_scores) / len(region_scores)) * 8.0
    estimated = _clamp(estimated, 0.08, 1.0)
    log(f"Estimated image-driven intensity: {estimated:.4f}")
    return estimated


def _get_basis_key(obj: bpy.types.Object):
    shape_keys = obj.data.shape_keys
    if not shape_keys:
        return None
    return shape_keys.key_blocks.get("Basis")


def _get_target_key(obj: bpy.types.Object, blendshape_name: str):
    shape_keys = obj.data.shape_keys
    if not shape_keys:
        return None
    return shape_keys.key_blocks.get(blendshape_name)


def _get_head_indices(obj: bpy.types.Object, settings) -> list[int]:
    vertex_group_name = getattr(settings, "head_vertex_group", "")
    if not vertex_group_name or vertex_group_name not in obj.vertex_groups:
        return list(range(len(obj.data.vertices)))

    group_index = obj.vertex_groups[vertex_group_name].index
    indices = []
    for vertex in obj.data.vertices:
        if any(group.group == group_index and group.weight > 0.0 for group in vertex.groups):
            indices.append(vertex.index)
    return indices or list(range(len(obj.data.vertices)))


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


def _view_weights(blendshape_name: str) -> dict:
    if blendshape_name.startswith("brow") or blendshape_name.startswith("eye") or blendshape_name.startswith("nose"):
        return {
            "Front": 1.0,
            "ThreeQuarterLeft": 0.8,
            "ThreeQuarterRight": 0.8,
        }
    if blendshape_name.startswith("jaw") or blendshape_name == "tongueOut":
        return {
            "Front": 0.7,
            "ThreeQuarterLeft": 0.9,
            "ThreeQuarterRight": 0.9,
            "ProfileLeft": 0.8,
            "ProfileRight": 0.8,
        }
    return {
        "Front": 1.0,
        "ThreeQuarterLeft": 0.8,
        "ThreeQuarterRight": 0.8,
        "ProfileLeft": 0.35,
        "ProfileRight": 0.35,
    }


def _sample_vertex_attention(scene, obj, basis_co, analyses: dict, weights: dict, blendshape_name: str) -> float:
    world_co = obj.matrix_world @ basis_co
    total = 0.0
    total_weight = 0.0

    for view_name, view_weight in weights.items():
        camera = bpy.data.objects.get(view_name)
        analysis = analyses.get(view_name)
        if camera is None or analysis is None:
            continue

        projected = world_to_camera_view(scene, camera, world_co)
        if projected.z <= 0.0:
            continue
        if projected.x < 0.0 or projected.x > 1.0 or projected.y < 0.0 or projected.y > 1.0:
            continue

        attention = _sample_map(analysis["attention"], projected.x, projected.y)
        blendshape_region_mask = analysis.get("blendshape_region_mask")
        if blendshape_region_mask is not None:
            region_weight = _sample_map(blendshape_region_mask, projected.x, projected.y)
            attention *= (0.15 + 0.85 * region_weight)
        total += attention * view_weight
        total_weight += view_weight

    if total_weight <= 1e-6:
        return 0.0
    return total / total_weight


def _profile_delta(blendshape_name: str, masks: dict) -> mathutils.Vector:
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

    delta = mathutils.Vector((0.0, 0.0, 0.0))

    if blendshape_name == "browInnerUp":
        delta.z += brow * center * (1.0 - nose)
    elif blendshape_name == "browDownLeft":
        delta.z -= brow * left * (1.0 - center * 0.25)
    elif blendshape_name == "browDownRight":
        delta.z -= brow * right * (1.0 - center * 0.25)
    elif blendshape_name == "browOuterUpLeft":
        delta.z += brow * left * outer
    elif blendshape_name == "browOuterUpRight":
        delta.z += brow * right * outer
    elif blendshape_name == "eyeBlinkLeft":
        delta.z -= eye * left * 0.85
    elif blendshape_name == "eyeBlinkRight":
        delta.z -= eye * right * 0.85
    elif blendshape_name == "eyeSquintLeft":
        delta.z -= eye * left * 0.55
        delta.y += cheek * left * 0.15
    elif blendshape_name == "eyeSquintRight":
        delta.z -= eye * right * 0.55
        delta.y += cheek * right * 0.15
    elif blendshape_name == "eyeWideLeft":
        delta.z += eye * left * 0.60
    elif blendshape_name == "eyeWideRight":
        delta.z += eye * right * 0.60
    elif blendshape_name == "noseSneerLeft":
        delta.z += nose * left * 0.35
        delta.y += nose * left * 0.15
    elif blendshape_name == "noseSneerRight":
        delta.z += nose * right * 0.35
        delta.y += nose * right * 0.15
    elif blendshape_name == "jawOpen":
        delta.z -= lower * (0.4 + 0.6 * center)
    elif blendshape_name == "jawForward":
        delta.y -= lower * (0.3 + 0.7 * center)
    elif blendshape_name == "jawLeft":
        delta.x -= lower
    elif blendshape_name == "jawRight":
        delta.x += lower
    elif blendshape_name == "mouthSmileLeft":
        delta.x -= corner * left * 0.6
        delta.z += corner * left * 0.9
    elif blendshape_name == "mouthSmileRight":
        delta.x += corner * right * 0.6
        delta.z += corner * right * 0.9
    elif blendshape_name == "mouthFrownLeft":
        delta.z -= corner * left * 0.85
    elif blendshape_name == "mouthFrownRight":
        delta.z -= corner * right * 0.85
    elif blendshape_name == "mouthLeft":
        delta.x -= mouth
    elif blendshape_name == "mouthRight":
        delta.x += mouth
    elif blendshape_name == "mouthStretchLeft":
        delta.x -= mouth * left * 0.8
    elif blendshape_name == "mouthStretchRight":
        delta.x += mouth * right * 0.8
    elif blendshape_name == "mouthPucker":
        delta.y -= mouth * center * 0.8
        delta.x += mouth * (0.5 - right) * 0.5
        delta.x -= mouth * (0.5 - left) * 0.5
    elif blendshape_name == "mouthFunnel":
        delta.y -= mouth * center * 0.7
        delta.z -= mouth * center * 0.25
    elif blendshape_name == "mouthPressLeft":
        delta.y += (upper_lip + lower_lip) * left * 0.55
    elif blendshape_name == "mouthPressRight":
        delta.y += (upper_lip + lower_lip) * right * 0.55
    elif blendshape_name == "mouthRollUpper":
        delta.y += upper_lip * center * 0.55
        delta.z -= upper_lip * center * 0.25
    elif blendshape_name == "mouthRollLower":
        delta.y += lower_lip * center * 0.55
        delta.z += lower_lip * center * 0.18
    elif blendshape_name == "mouthShrugUpper":
        delta.z += upper_lip * center * 0.7
    elif blendshape_name == "mouthShrugLower":
        delta.z -= lower_lip * center * 0.7
    elif blendshape_name == "mouthUpperUpLeft":
        delta.z += upper_lip * left * 0.9
    elif blendshape_name == "mouthUpperUpRight":
        delta.z += upper_lip * right * 0.9
    elif blendshape_name == "mouthLowerDownLeft":
        delta.z -= lower_lip * left * 0.9
    elif blendshape_name == "mouthLowerDownRight":
        delta.z -= lower_lip * right * 0.9
    elif blendshape_name == "mouthClose":
        delta.z += lower_lip * center * 0.4
        delta.z -= upper_lip * center * 0.4
    elif blendshape_name == "mouthDimpleLeft":
        delta.y += cheek * left * 0.35
    elif blendshape_name == "mouthDimpleRight":
        delta.y += cheek * right * 0.35
    elif blendshape_name == "cheekPuff":
        delta.y -= cheek * 0.7
    elif blendshape_name == "cheekSquintLeft":
        delta.z += cheek * left * 0.35
    elif blendshape_name == "cheekSquintRight":
        delta.z += cheek * right * 0.35
    elif blendshape_name == "tongueOut":
        delta.y -= lower_lip * center * 0.4
        delta.z -= lower_lip * center * 0.35

    return delta


def solve_shape(
    obj: bpy.types.Object,
    blendshape_name: str,
    targets: dict,
    settings=None,
    neutral_views: dict | None = None,
    reconstruction_result: dict | None = None,
):
    """Apply an image-guided deformation profile to the requested shape key."""
    hunyuan_only = bool(getattr(settings, "enable_hunyuan_reconstruction", False))
    basis_key = _get_basis_key(obj)
    target_key = _get_target_key(obj, blendshape_name)
    if basis_key is None:
        raise RuntimeError("Target object has no Basis shape key.")
    if target_key is None:
        raise RuntimeError(f"Shape key '{blendshape_name}' does not exist.")

    for key_block in obj.data.shape_keys.key_blocks:
        if key_block.name != blendshape_name and key_block.name != "Basis":
            key_block.value = 0.0

    neutral_views = neutral_views or {}
    reconstruction_result = reconstruction_result or {}
    reconstruction_status = reconstruction_result.get("status", "disabled")
    neutral_reconstruction_mesh = reconstruction_result.get("neutral_mesh_path")
    target_reconstruction_mesh = reconstruction_result.get("target_mesh_path")
    if reconstruction_status == "finished" and neutral_reconstruction_mesh and target_reconstruction_mesh:
        log(
            f"Reconstruction guidance is available for '{blendshape_name}' "
            f"from neutral='{neutral_reconstruction_mesh}' target='{target_reconstruction_mesh}'"
        )
    elif reconstruction_status not in {"disabled", "not_configured"}:
        log(
            f"Reconstruction guidance for '{blendshape_name}' is not ready "
            f"(status='{reconstruction_status}')"
        )
    default_intensity = getattr(settings, "default_intensity", 0.7)
    if hunyuan_only:
        analyses = {}
        strength = _clamp(default_intensity, 0.08, 1.0)
        mode = "hunyuan-only"
        log(
            f"Hunyuan-only mode is enabled for '{blendshape_name}'. "
            f"Skipping OpenCV/MediaPipe guidance and using intensity={strength:.3f}."
        )
    else:
        analyses = {}
        mediapipe_views = 0
        relevant_regions = BLENDSHAPE_REGIONS.get(blendshape_name, [])
        for view_name, neutral_path in neutral_views.items():
            target_paths = targets.get(view_name) or []
            if not target_paths:
                continue
            try:
                analyses[view_name] = _load_view_analysis(neutral_path, target_paths[0])
                analysis = analyses[view_name]
                log(
                    f"View '{view_name}' detection mode: {analysis['detection_mode']} "
                    f"(neutral_landmarks={analysis['neutral_landmarks_detected']}, "
                    f"target_landmarks={analysis['target_landmarks_detected']})"
                )
                if relevant_regions and analyses[view_name].get("region_masks"):
                    analyses[view_name]["blendshape_region_mask"] = _merge_region_masks(
                        analyses[view_name]["region_masks"],
                        relevant_regions,
                        analyses[view_name]["attention"].shape,
                    )
                    region_summary = ", ".join(
                        f"{region}={analysis['region_scores'].get(region, 0.0):.4f}"
                        for region in relevant_regions
                    )
                    log(f"View '{view_name}' relevant region scores for '{blendshape_name}': {region_summary}")
                if analyses[view_name].get("has_landmarks"):
                    mediapipe_views += 1
            except Exception as exc:
                log(f"Could not analyze view '{view_name}': {exc}")

        strength = _estimate_expression_strength(analyses, blendshape_name, default_intensity)
        if analyses and mediapipe_views > 0:
            mode = "mediapipe-opencv"
        elif analyses:
            mode = "opencv-only"
        else:
            mode = "procedural-fallback"

        if mp is None:
            log("MediaPipe is not installed in Blender's Python. Using OpenCV-only image guidance.")
        elif MEDIAPIPE_BACKEND == "tasks-model-required":
            log(
                "MediaPipe is installed, but this build only exposes the Tasks API and still needs a "
                "Face Landmarker .task model. Using OpenCV-only guidance for now."
            )
        elif MEDIAPIPE_BACKEND == "init-failed":
            log(f"MediaPipe failed to initialize: {MEDIAPIPE_INIT_ERROR}. Using OpenCV-only guidance.")
        elif analyses and mediapipe_views == 0:
            log("MediaPipe did not detect landmarks in the available views. Using OpenCV-only guidance.")
        elif analyses:
            log(
                f"MediaPipe backend '{MEDIAPIPE_BACKEND}' detected usable landmarks in "
                f"{mediapipe_views}/{len(analyses)} analyzed views."
            )

    indices = _get_head_indices(obj, settings)
    basis_coords = [basis_key.data[index].co.copy() for index in indices]

    min_x = min(co.x for co in basis_coords)
    max_x = max(co.x for co in basis_coords)
    min_y = min(co.y for co in basis_coords)
    max_y = max(co.y for co in basis_coords)
    min_z = min(co.z for co in basis_coords)
    max_z = max(co.z for co in basis_coords)

    size_x = max(max_x - min_x, 1e-5)
    size_y = max(max_y - min_y, 1e-5)
    size_z = max(max_z - min_z, 1e-5)

    amplitude = mathutils.Vector((
        size_x * (0.006 + 0.010 * strength),
        size_y * (0.005 + 0.014 * strength),
        size_z * (0.007 + 0.018 * strength),
    ))

    for vert_index in range(len(obj.data.vertices)):
        target_key.data[vert_index].co = basis_key.data[vert_index].co

    transfer_result = apply_reconstruction_transfer(
        obj,
        basis_key,
        target_key,
        indices,
        blendshape_name,
        strength,
        reconstruction_result,
    )
    if transfer_result and transfer_result.get("vertex_count", 0) > 0:
        target_key.value = 1.0
        log(
            f"Applied solver result for '{blendshape_name}' using mode='{transfer_result['mode']}' "
            f"with strength={transfer_result['intensity']:.3f} over "
            f"{transfer_result['vertex_count']} driven vertices"
        )
        return transfer_result
    if hunyuan_only:
        raise RuntimeError(
            "Hunyuan-only mode is enabled, but the reconstruction transfer did not produce a usable "
            "deformation. Check the Gemini frontal target and the Hunyuan outputs in the cache folder."
        )

    relevant_views = _view_weights(blendshape_name)
    scene = bpy.context.scene
    moved_vertices = 0

    for vert_index, basis_co in zip(indices, basis_coords):
        nx = (basis_co.x - min_x) / size_x
        ny = (basis_co.y - min_y) / size_y
        nz = (basis_co.z - min_z) / size_z
        masks = _build_masks(nx, ny, nz)
        profile_delta = _profile_delta(blendshape_name, masks)
        if profile_delta.length_squared <= 1e-10:
            continue

        if analyses:
            attention = _sample_vertex_attention(
                scene,
                obj,
                basis_co,
                analyses,
                relevant_views,
                blendshape_name,
            )
            if attention < 0.06:
                continue
            evidence = _smoothstep(0.06, 0.55, attention)
        else:
            evidence = 1.0

        delta = mathutils.Vector((
            profile_delta.x * amplitude.x * evidence,
            profile_delta.y * amplitude.y * evidence,
            profile_delta.z * amplitude.z * evidence,
        ))

        if delta.length_squared <= 1e-12:
            continue

        target_key.data[vert_index].co = basis_co + delta
        moved_vertices += 1

    target_key.value = 1.0
    log(
        f"Applied solver result for '{blendshape_name}' using mode='{mode}' "
        f"with strength={strength:.3f} over {moved_vertices} driven vertices"
    )
    return {
        "mode": mode,
        "intensity": strength,
        "vertex_count": moved_vertices,
    }

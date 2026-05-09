"""
Utilities for front-first generation and Hunyuan3D-2.1 reconstruction jobs.
"""

import json
import os
import subprocess
from dataclasses import asdict, dataclass

import bpy

from .logging_utils import log


@dataclass
class ReconstructionJob:
    blendshape_name: str
    job_dir: str
    neutral_front_path: str
    target_front_path: str
    neutral_mesh_path: str
    target_mesh_path: str
    manifest_path: str


def create_reconstruction_job(blendshape_name: str, generation_dir: str, neutral_front_path: str, target_front_path: str) -> ReconstructionJob:
    neutral_mesh_path = os.path.join(generation_dir, f"hunyuan_neutral_{blendshape_name}.glb")
    target_mesh_path = os.path.join(generation_dir, f"hunyuan_target_{blendshape_name}.glb")
    manifest_path = os.path.join(generation_dir, "reconstruction_job.json")
    return ReconstructionJob(
        blendshape_name=blendshape_name,
        job_dir=generation_dir,
        neutral_front_path=neutral_front_path,
        target_front_path=target_front_path,
        neutral_mesh_path=neutral_mesh_path,
        target_mesh_path=target_mesh_path,
        manifest_path=manifest_path,
    )


def save_reconstruction_manifest(job: ReconstructionJob, settings) -> None:
    payload = asdict(job)
    payload["hunyuan_enabled"] = bool(getattr(settings, "enable_hunyuan_reconstruction", False))
    payload["hunyuan_python_path"] = getattr(settings, "hunyuan_python_path", "")
    payload["hunyuan_workdir"] = getattr(settings, "hunyuan_workdir", "")
    payload["hunyuan_model_id"] = getattr(settings, "hunyuan_model_id", "")
    payload["hunyuan_subfolder"] = getattr(settings, "hunyuan_subfolder", "")
    with open(job.manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log(f"Saved reconstruction manifest to '{job.manifest_path}'")


def _bridge_script_path() -> str:
    return os.path.join(os.path.dirname(__file__), "tools", "hunyuan_shape_bridge.py")


def import_reconstruction_preview(job: ReconstructionJob, mesh_path: str) -> dict:
    if not mesh_path or not os.path.exists(mesh_path):
        return {"status": "missing_output", "object_name": None}

    extension = os.path.splitext(mesh_path)[1].lower()
    before_names = set(bpy.data.objects.keys())

    try:
        if extension == ".glb":
            bpy.ops.import_scene.gltf(filepath=mesh_path)
        elif extension == ".obj":
            bpy.ops.wm.obj_import(filepath=mesh_path)
        elif extension == ".ply":
            bpy.ops.wm.ply_import(filepath=mesh_path)
        else:
            log(f"Reconstruction preview import is not supported for '{extension}' files")
            return {"status": "unsupported_format", "object_name": None}
    except Exception as exc:
        log(f"Failed to import reconstruction preview '{mesh_path}': {exc}")
        return {"status": "import_failed", "object_name": None}

    imported_names = [name for name in bpy.data.objects.keys() if name not in before_names]
    imported_meshes = [name for name in imported_names if bpy.data.objects[name].type == 'MESH']
    object_name = imported_meshes[0] if imported_meshes else (imported_names[0] if imported_names else None)
    if object_name:
        obj = bpy.data.objects[object_name]
        obj.name = f"HunyuanPreview_{job.blendshape_name}"
        object_name = obj.name
        log(f"Imported reconstruction preview object '{object_name}'")
        return {"status": "imported", "object_name": object_name}

    log("Reconstruction preview import completed, but no new object could be identified")
    return {"status": "imported_unknown", "object_name": None}


def _run_single_hunyuan(
    python_path: str,
    workdir: str,
    bridge_script: str,
    timeout_seconds: int,
    model_id: str,
    subfolder: str,
    image_path: str,
    output_mesh_path: str,
    label: str,
) -> tuple[str, str | None]:
    cmd = [
        python_path,
        bridge_script,
        "--image",
        image_path,
        "--output",
        output_mesh_path,
        "--model",
        model_id,
        "--subfolder",
        subfolder,
    ]
    log(f"Launching Hunyuan reconstruction for {label} from '{workdir}'")
    log(f"Hunyuan command ({label}): {' '.join(cmd)}")

    try:
        completed = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log(f"Hunyuan reconstruction for {label} timed out after {timeout_seconds}s")
        return "timeout", None
    except Exception as exc:
        log(f"Failed to start Hunyuan reconstruction for {label}: {exc}")
        return "launch_failed", None

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if stdout:
        log(f"Hunyuan stdout ({label}):\n{stdout}")
    if stderr:
        log(f"Hunyuan stderr ({label}):\n{stderr}")

    if completed.returncode != 0:
        log(f"Hunyuan reconstruction for {label} exited with code {completed.returncode}")
        return "failed", None

    if not os.path.exists(output_mesh_path):
        log(f"Hunyuan completed for {label} but no mesh was found at '{output_mesh_path}'")
        return "missing_output", None

    log(f"Hunyuan reconstruction finished for {label}: '{output_mesh_path}'")
    return "finished", output_mesh_path


def run_hunyuan_reconstruction(job: ReconstructionJob, settings) -> dict:
    if not getattr(settings, "enable_hunyuan_reconstruction", False):
        return {"status": "disabled", "neutral_mesh_path": None, "target_mesh_path": None}

    python_path = getattr(settings, "hunyuan_python_path", "").strip()
    workdir = bpy.path.abspath(getattr(settings, "hunyuan_workdir", "").strip())
    model_id = getattr(settings, "hunyuan_model_id", "").strip() or "tencent/Hunyuan3D-2.1"
    subfolder = getattr(settings, "hunyuan_subfolder", "").strip() or "hunyuan3d-dit-v2-1"

    if not python_path:
        log("Hunyuan reconstruction is enabled, but no Python path is configured.")
        return {"status": "not_configured", "neutral_mesh_path": None, "target_mesh_path": None}
    if not os.path.exists(python_path):
        log(f"Hunyuan Python path does not exist: '{python_path}'")
        return {"status": "missing_python", "neutral_mesh_path": None, "target_mesh_path": None}
    if not workdir:
        log("Hunyuan reconstruction is enabled, but no workdir is configured.")
        return {"status": "not_configured", "neutral_mesh_path": None, "target_mesh_path": None}
    if not os.path.isdir(workdir):
        log(f"Hunyuan workdir does not exist: '{workdir}'")
        return {"status": "missing_workdir", "neutral_mesh_path": None, "target_mesh_path": None}

    bridge_script = _bridge_script_path()
    timeout_seconds = max(int(getattr(settings, "hunyuan_timeout_seconds", 1800)), 60)
    neutral_status, neutral_mesh_path = _run_single_hunyuan(
        python_path,
        workdir,
        bridge_script,
        timeout_seconds,
        model_id,
        subfolder,
        job.neutral_front_path,
        job.neutral_mesh_path,
        "neutral",
    )
    if neutral_status != "finished":
        return {
            "status": f"neutral_{neutral_status}",
            "neutral_mesh_path": neutral_mesh_path,
            "target_mesh_path": None,
        }

    target_status, target_mesh_path = _run_single_hunyuan(
        python_path,
        workdir,
        bridge_script,
        timeout_seconds,
        model_id,
        subfolder,
        job.target_front_path,
        job.target_mesh_path,
        "target",
    )
    if target_status != "finished":
        return {
            "status": f"target_{target_status}",
            "neutral_mesh_path": neutral_mesh_path,
            "target_mesh_path": target_mesh_path,
        }

    return {
        "status": "finished",
        "neutral_mesh_path": neutral_mesh_path,
        "target_mesh_path": target_mesh_path,
    }

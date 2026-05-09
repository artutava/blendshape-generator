"""
Operators for installing and configuring the external Hunyuan3D runtime.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import venv

import bpy
from bpy.types import Operator

from .logging_utils import log, set_verbose_logging


def _default_install_root() -> str:
    base = bpy.utils.user_resource('SCRIPTS', path="arkit_gen_tools", create=True)
    return os.path.join(base, "hunyuan")


def _venv_python_path(venv_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _pick_base_python() -> tuple[str, bool]:
    if os.name == "nt":
        launcher = ["py", "-3.10", "-c", "import sys; print(sys.executable)"]
        completed = subprocess.run(
            launcher,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return "3.10", True
    return sys.executable, False


def _run_command(command: list[str], cwd: str | None = None) -> None:
    log(f"Running command: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout.strip():
        log(f"Command stdout:\n{completed.stdout.strip()}")
    if completed.stderr.strip():
        log(f"Command stderr:\n{completed.stderr.strip()}")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def _write_shape_requirements(repo_dir: str) -> str:
    source_requirements = os.path.join(repo_dir, "hy3dshape", "requirements.txt")
    if not os.path.exists(source_requirements):
        raise RuntimeError(f"Missing shape requirements file: '{source_requirements}'")

    blocked_prefixes = (
        "bpy",
        "cupy",
        "deepspeed",
        "pythreejs",
    )
    blocked_exact = {
        "gradio",
        "fastapi",
        "uvicorn",
    }

    lines = []
    with open(source_requirements, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            normalized = line.lower()
            if normalized in blocked_exact:
                continue
            if normalized.startswith(blocked_prefixes):
                continue
            lines.append(line)

    lines.extend([
        "huggingface-hub",
        "safetensors",
        "scipy",
        "imageio",
        "pandas",
        "timm",
        "torchdiffeq",
        "sentencepiece",
    ])

    temp_dir = tempfile.mkdtemp(prefix="arkitgen_hunyuan_")
    output_path = os.path.join(temp_dir, "shape_requirements.txt")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return output_path


class ARKITGEN_OT_install_hunyuan(Operator):
    """Download and configure a local Hunyuan3D-2.1 runtime for this addon."""

    bl_idname = "arkitgen.install_hunyuan"
    bl_label = "Install Hunyuan3D"
    bl_description = "Clone Hunyuan3D-2.1, create a virtual environment, and configure the addon paths"

    def execute(self, context):
        settings = context.scene.arkit_gen_settings
        set_verbose_logging(settings.enable_verbose_logging)

        install_root = bpy.path.abspath(settings.hunyuan_install_root.strip()) if settings.hunyuan_install_root.strip() else _default_install_root()
        repo_url = settings.hunyuan_repo_url.strip() or "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
        repo_dir = os.path.join(install_root, "Hunyuan3D-2.1")
        venv_dir = os.path.join(repo_dir, ".venv")
        python_path = _venv_python_path(venv_dir)
        selected_python, use_py_launcher = _pick_base_python()

        try:
            os.makedirs(install_root, exist_ok=True)
            log(f"Hunyuan installer using root '{install_root}'")

            if not os.path.isdir(repo_dir):
                _run_command(
                    ["git", "-c", "core.longpaths=true", "clone", "--depth", "1", repo_url, repo_dir],
                    cwd=install_root,
                )
            else:
                log(f"Hunyuan repository already exists at '{repo_dir}', skipping clone")

            if not os.path.exists(python_path):
                log(f"Creating virtual environment at '{venv_dir}'")
                if use_py_launcher:
                    _run_command(["py", "-3.10", "-m", "venv", venv_dir], cwd=repo_dir)
                else:
                    log(
                        "Python 3.10 was not found via the Windows py launcher. "
                        f"Falling back to '{selected_python}'."
                    )
                    builder = venv.EnvBuilder(with_pip=True)
                    builder.create(venv_dir)
            else:
                log(f"Virtual environment already exists at '{venv_dir}', skipping creation")

            _run_command([python_path, "-m", "pip", "install", "--upgrade", "pip"], cwd=repo_dir)
            if use_py_launcher:
                _run_command(
                    [
                        python_path,
                        "-m",
                        "pip",
                        "install",
                        "torch==2.5.1",
                        "torchvision==0.20.1",
                        "torchaudio==2.5.1",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu124",
                    ],
                    cwd=repo_dir,
                )
            else:
                log(
                    "Skipping pinned PyTorch install because the installer did not find Python 3.10. "
                    "You may need to install a compatible torch build manually in this environment."
                )

            requirements_path = _write_shape_requirements(repo_dir)
            _run_command([python_path, "-m", "pip", "install", "-r", requirements_path], cwd=repo_dir)

            pyproject_path = os.path.join(repo_dir, "pyproject.toml")
            setup_path = os.path.join(repo_dir, "setup.py")
            if os.path.exists(pyproject_path) or os.path.exists(setup_path):
                _run_command([python_path, "-m", "pip", "install", "-e", "."], cwd=repo_dir)
            else:
                log(
                    "Skipping editable install because the Hunyuan repository does not define "
                    "a setup.py or pyproject.toml. The bridge will import modules directly from the workdir."
                )

        except Exception as exc:
            self.report({'ERROR'}, f"Hunyuan install failed: {exc}")
            log(f"Hunyuan installation failed: {exc}")
            return {'CANCELLED'}

        settings.hunyuan_install_root = install_root
        settings.hunyuan_workdir = repo_dir
        settings.hunyuan_python_path = python_path
        settings.enable_hunyuan_reconstruction = True

        self.report({'INFO'}, "Hunyuan3D-2.1 installed and configured")
        log(
            f"Hunyuan installation completed. workdir='{repo_dir}' python='{python_path}'"
        )
        return {'FINISHED'}

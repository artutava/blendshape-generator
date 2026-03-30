"""
Helpers for storing persistent preview/cache images for each generation run.
"""

import os
from datetime import datetime

import bpy

from .logging_utils import log


def _sanitize_name(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "unnamed"


def get_cache_root(settings) -> str:
    """Return the persistent cache root directory."""
    custom_dir = getattr(settings, "cache_directory", "").strip()
    if custom_dir:
        root = bpy.path.abspath(custom_dir)
    else:
        root = bpy.utils.user_resource("SCRIPTS", path="arkit_gen_cache", create=True)
    os.makedirs(root, exist_ok=True)
    return root


def prepare_generation_cache_dir(settings, blendshape_name: str) -> str:
    """Create a timestamped cache directory for a blendshape generation run."""
    root = get_cache_root(settings)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"{stamp}_{_sanitize_name(blendshape_name)}"
    cache_dir = os.path.join(root, dirname)
    os.makedirs(cache_dir, exist_ok=True)
    log(f"Using persistent cache directory '{cache_dir}'")
    return cache_dir

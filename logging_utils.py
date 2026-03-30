"""
Lightweight logging helpers for the ARKit Blendshape Generator.
"""

import os
from datetime import datetime

import bpy


LOG_PREFIX = "[ARKitGen]"
_verbose_enabled = True


def set_verbose_logging(enabled: bool) -> None:
    """Enable or disable non-critical terminal logs."""
    global _verbose_enabled
    _verbose_enabled = bool(enabled)


def log(message: str, force: bool = False) -> None:
    """Print a timestamped message to Blender's terminal."""
    if not force and not _verbose_enabled:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{LOG_PREFIX} {timestamp} | {message}")


def log_section(title: str, force: bool = False) -> None:
    """Print a visual section header to the terminal."""
    log(f"=== {title} ===", force=force)


def log_addon_source(module_file: str, package_name: str) -> None:
    """Describe where Blender loaded the addon from and whether a user-addon entry exists."""
    source_dir = os.path.realpath(os.path.dirname(module_file))
    module_name = package_name.split(".")[0]
    user_addons_dir = bpy.utils.user_resource("SCRIPTS", path="addons", create=True)
    linked_path = os.path.join(user_addons_dir, module_name)

    log_section("Addon Startup", force=True)
    log(f"Loaded package '{package_name}' from '{source_dir}'", force=True)
    log(f"User addons directory: '{user_addons_dir}'", force=True)

    if not os.path.exists(linked_path):
        log(
            "No matching entry exists in Blender's user addons directory for this module. "
            "If Blender Development Addon is expected to link it, the link was not created.",
            force=True,
        )
        return

    linked_real_path = os.path.realpath(linked_path)
    is_link = os.path.islink(linked_path)
    same_target = linked_real_path == source_dir
    kind = "symlink" if is_link else "directory/junction"

    log(f"Found user addon entry '{linked_path}' ({kind}) -> '{linked_real_path}'", force=True)
    if not same_target:
        log(
            "WARNING: Blender is seeing another addon path with the same module name. "
            "This usually means there is an old installed copy shadowing the workspace version.",
            force=True,
        )

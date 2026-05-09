"""
External bridge for running a Hunyuan3D-2.1 single-image shape reconstruction.

This script is designed to be executed by a separate Python environment
where Hunyuan3D-2.1 and its dependencies are installed. It keeps the
Blender add-on lightweight while still allowing a front-first
reconstruction workflow.
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hunyuan3D-2.1 shape reconstruction")
    parser.add_argument("--image", required=True, help="Input frontal target image")
    parser.add_argument("--output", required=True, help="Output mesh path (.glb/.obj/.ply)")
    parser.add_argument("--model", required=True, help="Model identifier or local checkpoint path")
    parser.add_argument("--subfolder", default="", help="Optional model subfolder")
    return parser.parse_args()


def _load_pipeline(model_id: str, subfolder: str):
    workdir = os.getcwd()
    for candidate in (
        workdir,
        os.path.join(workdir, "hy3dshape"),
        os.path.join(workdir, "hy3dpaint"),
    ):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)

    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    except Exception as exc:
        raise RuntimeError(
            "Could not import Hunyuan3D-2.1 pipeline. "
            "Make sure this Python environment has the project installed."
        ) from exc

    kwargs = {}
    if subfolder:
        kwargs["subfolder"] = subfolder
    try:
        return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_id, **kwargs)
    except TypeError:
        return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_id)


def _export_mesh(mesh, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if hasattr(mesh, "export"):
        mesh.export(output_path)
        return
    if hasattr(mesh, "save"):
        mesh.save(output_path)
        return

    raise RuntimeError(
        "The Hunyuan pipeline returned an unsupported mesh object. "
        "Expected an object with .export(...) or .save(...)."
    )


def _run_reconstruction(pipeline, image_path: str):
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required in the Hunyuan environment.") from exc

    image = Image.open(image_path).convert("RGBA")

    if callable(pipeline):
        result = pipeline(image)
    elif hasattr(pipeline, "predict"):
        result = pipeline.predict(image)
    else:
        raise RuntimeError("Unsupported Hunyuan pipeline interface.")

    if isinstance(result, dict):
        for key in ("mesh", "trimesh", "output", "result"):
            candidate = result.get(key)
            if candidate is not None:
                return candidate
        raise RuntimeError("Pipeline returned a dict but no mesh-like field was found.")
    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("Pipeline returned an empty sequence.")
        return result[0]

    return result


def main() -> int:
    args = _parse_args()

    if not os.path.exists(args.image):
        raise RuntimeError(f"Input image does not exist: '{args.image}'")

    print(f"[HunyuanBridge] Loading model '{args.model}'")
    if args.subfolder:
        print(f"[HunyuanBridge] Using subfolder '{args.subfolder}'")

    pipeline = _load_pipeline(args.model, args.subfolder)
    mesh = _run_reconstruction(pipeline, args.image)
    _export_mesh(mesh, args.output)

    print(f"[HunyuanBridge] Exported mesh to '{args.output}'")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[HunyuanBridge] ERROR: {exc}", file=sys.stderr)
        raise

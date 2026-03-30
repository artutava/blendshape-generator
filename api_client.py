import os
import base64
import mimetypes
import requests
import bpy

from .properties import ARKITGenSettings


MODEL_NAME = "gemini-3-pro-image-preview"


def _encode_file_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/png"


def _save_bytes(data: bytes, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(data)


def _build_prompt(blendshape_name: str, style: str, intensity: float) -> str:
    return f"""
Edit this exact rendered character.

CRITICAL REQUIREMENTS:
- Preserve identity EXACTLY
- Preserve camera angle, framing, lens, and lighting
- Preserve materials, textures, and proportions
- Keep pixel alignment as stable as possible
- Do not redesign the face
- Do not change age, ethnicity, hairstyle, or head pose
- Only modify the facial expression

Target ARKit blendshape: {blendshape_name}
Target intensity: {intensity}
Style target: {style}

Goal:
Create a natural and convincing facial expression that corresponds to the ARKit blendshape above.
The result must still look like the same exact character and same exact render, only with the requested facial deformation.

Important:
- No background change
- No body change
- No clothing change
- No added accessories
- No text
- No extra objects
- No stylization drift
""".strip()


def request_targets(settings: ARKITGenSettings, blendshape_name: str, views: dict):
    """
    Calls Gemini / Nano Banana Pro to generate target images for each input view.

    Parameters
    ----------
    settings : ARKITGenSettings
        Add-on settings containing API key, endpoint, style, etc.
    blendshape_name : str
        ARKit blendshape name.
    views : dict
        Mapping like {"Front": "/tmp/front.png", ...}

    Returns
    -------
    dict
        Mapping like {"Front": ["/tmp/generated1.png"], ...}
    """
    api_key = settings.api_key.strip()
    if not api_key:
        raise RuntimeError("API key is empty.")

    # If user filled a custom endpoint in Blender, use it.
    # Otherwise use the official Gemini generateContent endpoint.
    endpoint = settings.api_endpoint.strip() or (
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
    )

    prompt = _build_prompt(
        blendshape_name=blendshape_name,
        style=settings.style_preset,
        intensity=settings.default_intensity,
    )

    results = {}

    for view_name, image_path in views.items():
        image_b64 = _encode_file_base64(image_path)
        mime_type = _guess_mime_type(image_path)

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_b64,
                            }
                        },
                        {
                            "text": prompt
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["Image"]
            }
        }

        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates returned for view '{view_name}'.")

        saved_paths = []
        parts = candidates[0].get("content", {}).get("parts", [])

        for i, part in enumerate(parts):
            inline_data = part.get("inlineData") or part.get("inline_data")
            if not inline_data:
                continue

            img_b64 = inline_data.get("data")
            if not img_b64:
                continue

            out_path = os.path.join(
                bpy.app.tempdir,
                "arkit_gen",
                f"{blendshape_name}_{view_name}_{i}.png",
            )
            _save_bytes(base64.b64decode(img_b64), out_path)
            saved_paths.append(out_path)

        if not saved_paths:
            raise RuntimeError(f"No image output returned for view '{view_name}'.")

        results[view_name] = saved_paths

    return results
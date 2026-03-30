import os
import base64
import mimetypes
import time
import requests
import bpy

from .logging_utils import log
from .properties import ARKITGenSettings


MODEL_NAME = "gemini-3-pro-image-preview"
MAX_RETRIES = 4
DEFAULT_RETRY_DELAY_SECONDS = 5


class APIQuotaExceededError(RuntimeError):
    """Raised when the remote image API rejects requests due to quota exhaustion."""


class APIGenerationError(RuntimeError):
    """Raised when the remote image API returns an unexpected failure."""


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


def _parse_retry_delay_seconds(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    return float(DEFAULT_RETRY_DELAY_SECONDS * attempt)


def _extract_error_message(response: requests.Response) -> str:
    try:
        error_payload = response.json()
    except ValueError:
        error_payload = {}
    return (
        error_payload.get("error", {}).get("message")
        or response.text[:500]
        or f"HTTP {response.status_code}"
    )


def _is_quota_exhausted(error_message: str) -> bool:
    message = error_message.lower()
    return (
        "quota exceeded" in message
        or "billing" in message
        or "limit: 0" in message
        or "current quota" in message
    )


def _post_with_rate_limit_retry(endpoint: str, headers: dict, payload: dict) -> requests.Response:
    last_response = None

    for attempt in range(1, MAX_RETRIES + 1):
        log(f"Sending image generation request attempt {attempt}/{MAX_RETRIES} to '{endpoint}'")
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        last_response = response

        if response.status_code != 429:
            log(f"Received HTTP {response.status_code} from image generation API")
            response.raise_for_status()
            return response

        error_message = _extract_error_message(response)
        if _is_quota_exhausted(error_message):
            raise APIQuotaExceededError(
                "Gemini quota is exhausted for this project. "
                f"Server message: {error_message} "
                "Switch to a billed Gemini project, choose another endpoint/model, "
                "or keep 'Fallback Solver' enabled to continue locally without AI images."
            )

        if attempt == MAX_RETRIES:
            break

        delay = _parse_retry_delay_seconds(response, attempt)
        log(f"API returned 429 for attempt {attempt}. Waiting {delay:.1f}s before retrying")
        time.sleep(delay)

    if last_response is None:
        raise RuntimeError("No response received from the image generation API.")

    error_message = _extract_error_message(last_response)
    detailed_message = (
        "Rate limit reached while generating images. "
        f"Tried {MAX_RETRIES} times and the API still returned 429. "
        f"Server message: {error_message}"
    )
    if _is_quota_exhausted(error_message):
        raise APIQuotaExceededError(
            detailed_message
            + " Switch to a billed Gemini project, choose another endpoint/model, "
            + "or keep 'Fallback Solver' enabled to continue locally without AI images."
        )
    raise APIGenerationError(detailed_message)


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


def request_targets(
    settings: ARKITGenSettings,
    blendshape_name: str,
    views: dict,
    output_dir: str | None = None,
):
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
    model_name = settings.model_name.strip() or MODEL_NAME
    endpoint = settings.api_endpoint.strip() or (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    )
    log(f"Using model '{model_name}' with endpoint '{endpoint}'")

    prompt = _build_prompt(
        blendshape_name=blendshape_name,
        style=settings.style_preset,
        intensity=settings.default_intensity,
    )

    results = {}

    for view_name, image_path in views.items():
        log(f"Preparing API payload for view '{view_name}' from '{image_path}'")
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

        try:
            response = _post_with_rate_limit_retry(endpoint, headers, payload)
        except requests.HTTPError as exc:
            response = exc.response
            status = response.status_code if response is not None else "unknown"
            body = ""
            if response is not None:
                try:
                    body = response.text[:500]
                except Exception:
                    body = ""
            raise APIGenerationError(
                f"HTTP {status} while generating view '{view_name}'. Response: {body}"
            ) from exc
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise APIGenerationError(f"No candidates returned for view '{view_name}'.")

        saved_paths = []
        parts = candidates[0].get("content", {}).get("parts", [])

        for i, part in enumerate(parts):
            inline_data = part.get("inlineData") or part.get("inline_data")
            if not inline_data:
                continue

            img_b64 = inline_data.get("data")
            if not img_b64:
                continue

            target_dir = output_dir or os.path.join(bpy.app.tempdir, "arkit_gen")
            out_path = os.path.join(target_dir, f"generated_{blendshape_name}_{view_name}_{i}.png")
            _save_bytes(base64.b64decode(img_b64), out_path)
            saved_paths.append(out_path)
            log(f"Saved generated target for view '{view_name}' to '{out_path}'")

        if not saved_paths:
            raise APIGenerationError(f"No image output returned for view '{view_name}'.")

        results[view_name] = saved_paths

    log(f"Received generated targets for {len(results)} view(s)")
    return results

"""
Stub implementation of the external generative API client.

This module defines a function that would, in a production system,
communicate with a remote generative AI service to request images
showing a target expression on a face. For the purposes of this
prototype, the function simply returns the input views unchanged. The
`settings` argument can be extended to pass through authentication
parameters and other options.
"""

from .properties import ARKITGenSettings


def request_targets(settings: ARKITGenSettings, blendshape_name: str, views: dict):
    """Request target expressive images from a generative API.

    Parameters
    ----------
    settings : ARKITGenSettings
        User-configurable settings containing API endpoint, key, and style.
    blendshape_name : str
        Name of the ARKit blendshape being generated.
    views : dict
        Mapping of camera names to filepaths for neutral renders.

    Returns
    -------
    dict
        Mapping of camera names to lists of filepaths representing
        candidate target images. In this stub implementation, each
        camera's list contains just the neutral input image.
    """
    # TODO: Implement real API call using settings.api_endpoint and settings.api_key
    # For now we simply wrap the input neutral views into candidate lists.
    targets = {name: [path] for name, path in views.items()}
    return targets

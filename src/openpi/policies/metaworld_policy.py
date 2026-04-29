"""Input/output transforms for the MetaWorld V3 dataset.

The dataset (see ``hil_dsrl_pi0/scripts/generate_metaworld_mt50_dataset.py``)
mirrors the lerobot/metaworld_mt50 schema: single ``observation.image``
camera at 256×256 CHW, 4D ``observation.state`` (hand_xyz + gripper_norm),
4D ``action`` (xyz_delta + gripper_effort), per-frame ``task`` string from
the V3 task list.

For pi0/pi0.5 we feed the single image into the ``base_0_rgb`` slot and
zero-pad the wrist slots — there is no wrist camera on Sawyer in
MetaWorld. State and actions are passed through at their native 4D; the
model's internal padding handles the rest.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_metaworld_example() -> dict:
    """Random example for unit tests / shape inference."""
    return {
        "observation/state": np.random.rand(4),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "press the button",
    }


def _parse_image(image) -> np.ndarray:
    """Coerce a possibly-float, possibly-CHW image to uint8 HWC.

    LeRobot's image features are stored as PNG bytes but the loader
    returns float (C, H, W) tensors by default. Pi0's vision encoders
    expect uint8 (H, W, C) — this matches what ``libero_policy`` does.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MetaworldInputs(transforms.DataTransformFn):
    """Convert dataset / inference dicts to the format pi0 expects.

    Single-cam Sawyer setup: scene image goes to ``base_0_rgb``, wrist
    slots are zero-padded with ``image_mask=False`` so the model knows
    to ignore them (Pi0 uses False; Pi0-FAST uses True per its own
    masking convention — same as LiberoInputs).
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # Per LiberoInputs: pi0 uses False to mask the padding
                # image; pi0-FAST uses True. Match that convention.
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MetaworldOutputs(transforms.DataTransformFn):
    """Slice the model's 32D action back to MetaWorld's native 4D."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :4])}

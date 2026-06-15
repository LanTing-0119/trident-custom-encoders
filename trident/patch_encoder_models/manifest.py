"""Portable checkpoint layout and installation status for patch encoders."""

from pathlib import Path


PATCH_ENCODER_LAYOUT = {
    "conch_v1": "conch_v1_pytorch_model.bin",
    "conch_v15": "conch_v1_5_pytorch_model.bin",
    "uni_v1": "uni_v1_pytorch_model.bin",
    "uni_v2": "uni_v2_pytorch_model.bin",
    "ctranspath": "ctranspath.pth",
    "phikon": "phikon_v1",
    "phikon_v2": "phikon_v2",
    "resnet50": "resnet50/model.safetensors",
    "keep": "keep",
    "gigapath": "prov-gigapath_pytorch_model.bin",
    "virchow": "virchow/pytorch_model.bin",
    "virchow2": "virchow2_pytorch_model.bin",
    "hoptimus0": "hoptimus0_pytorch_model.bin",
    "hoptimus1": "hoptimus1_pytorch_model.bin",
    "h0-mini": "h0-mini/model.safetensors",
    "musk": "musk/model.safetensors",
    "openmidnight": "openmidnight/teacher_checkpoint_load.pt",
    "gpfm": "GPFM.pth",
    "digepath": "digepath.model.safetensors",
    "rui_path": "ruipath_visionfoundation_v1.0.bin",
    "hibou_l": "hibou_l",
    "kaiko-vitb8": "kaiko/vitb8/model.safetensors",
    "kaiko-vitb16": "kaiko/vitb16/model.safetensors",
    "kaiko-vits8": "kaiko/vits8/model.safetensors",
    "kaiko-vits16": "kaiko/vits16/model.safetensors",
    "kaiko-vitl14": "kaiko/vitl14/model.safetensors",
    "lunit-vits8": "lunit-vits8/model.safetensors",
    "midnight12k": "midnight12k",
    "genbio-pathfm": "genbio-pathfm/model.pth",
    "gemma4-e4b": "gemma4-e4b",
    "gemma4-26b": "gemma4-26b",
}

BLOCKED_GATED_ENCODERS = {"musk", "h0-mini", "openmidnight"}
EXCLUDED_ENCODERS = {"gemma4-e4b", "gemma4-26b"}


def checkpoint_path(root: str | Path, encoder_name: str) -> Path | None:
    relative_path = PATCH_ENCODER_LAYOUT.get(encoder_name)
    return Path(root).expanduser() / relative_path if relative_path else None


def expected_status(encoder_name: str) -> str:
    if encoder_name in BLOCKED_GATED_ENCODERS:
        return "blocked-gated"
    if encoder_name in EXCLUDED_ENCODERS:
        return "excluded-gemma4"
    return "ready"

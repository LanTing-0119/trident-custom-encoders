#!/usr/bin/env python
"""Download and audit the portable Trident patch-encoder checkpoint layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trident.patch_encoder_models.manifest import (  # noqa: E402
    BLOCKED_GATED_ENCODERS,
    EXCLUDED_ENCODERS,
    PATCH_ENCODER_LAYOUT,
    checkpoint_path,
    expected_status,
)


DOWNLOADS = {
    "conch_v1": (
        "MahmoodLab/conch",
        ".",
        [("pytorch_model.bin", "conch_v1_pytorch_model.bin")],
    ),
    "conch_v15": (
        "MahmoodLab/conchv1_5",
        ".",
        [("pytorch_model_vision.bin", "conch_v1_5_pytorch_model.bin")],
    ),
    "uni_v1": (
        "MahmoodLab/uni",
        ".",
        [("pytorch_model.bin", "uni_v1_pytorch_model.bin")],
    ),
    "uni_v2": (
        "MahmoodLab/UNI2-h",
        ".",
        [("pytorch_model.bin", "uni_v2_pytorch_model.bin")],
    ),
    "ctranspath": (
        "MahmoodLab/hest-bench",
        ".",
        [("fm_v1/ctranspath/CHIEF_CTransPath.pth", "ctranspath.pth")],
    ),
    "phikon": (
        "owkin/phikon",
        "phikon_v1",
        ["config.json", "pytorch_model.bin"],
    ),
    "phikon_v2": (
        "owkin/phikon-v2",
        "phikon_v2",
        ["config.json", "model.safetensors"],
    ),
    "keep": ("Astaxanthin/KEEP", "keep", ["config.json", "model.safetensors", "modeling_keep.py"]),
    "resnet50": ("timm/resnet50.tv_in1k", "resnet50", ["config.json", "model.safetensors"]),
    "hibou_l": (
        "histai/hibou-L",
        "hibou_l",
        [
            "config.json",
            "configuration_dinov2.py",
            "model.safetensors",
            "modeling_dinov2.py",
            "preprocessor_config.json",
        ],
    ),
    "gigapath": (
        "prov-gigapath/prov-gigapath",
        ".",
        [("pytorch_model.bin", "prov-gigapath_pytorch_model.bin")],
    ),
    "virchow": (
        "paige-ai/Virchow",
        "virchow",
        [("pytorch_model.bin", "pytorch_model.bin")],
    ),
    "virchow2": (
        "paige-ai/Virchow2",
        ".",
        [("pytorch_model.bin", "virchow2_pytorch_model.bin")],
    ),
    "hoptimus0": (
        "bioptimus/H-optimus-0",
        ".",
        [("pytorch_model.bin", "hoptimus0_pytorch_model.bin")],
    ),
    "hoptimus1": (
        "bioptimus/H-optimus-1",
        ".",
        [("pytorch_model.bin", "hoptimus1_pytorch_model.bin")],
    ),
    "gpfm": ("majiabo/GPFM", ".", ["GPFM.pth"]),
    "digepath": (
        "xtxx/Digepath",
        ".",
        [("model.safetensors", "digepath.model.safetensors")],
    ),
    "kaiko-vits8": (
        "1aurent/vit_small_patch8_224.kaiko_ai_towards_large_pathology_fms",
        "kaiko/vits8",
        ["config.json", "model.safetensors"],
    ),
    "kaiko-vits16": (
        "1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms",
        "kaiko/vits16",
        ["config.json", "model.safetensors"],
    ),
    "kaiko-vitb8": (
        "1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms",
        "kaiko/vitb8",
        ["config.json", "model.safetensors"],
    ),
    "kaiko-vitb16": (
        "1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
        "kaiko/vitb16",
        ["config.json", "model.safetensors"],
    ),
    "kaiko-vitl14": (
        "1aurent/vit_large_patch14_reg4_dinov2.kaiko_ai_towards_large_pathology_fms",
        "kaiko/vitl14",
        ["config.json", "model.safetensors"],
    ),
    "lunit-vits8": (
        "1aurent/vit_small_patch8_224.lunit_dino",
        "lunit-vits8",
        ["config.json", "model.safetensors"],
    ),
    "midnight12k": ("kaiko-ai/midnight", "midnight12k", ["config.json", "model.safetensors"]),
    "genbio-pathfm": ("genbio-ai/genbio-pathfm", "genbio-pathfm", ["model.pth"]),
    "musk": ("xiangjx/musk", "musk", ["model.safetensors"]),
    "h0-mini": ("bioptimus/H0-mini", "h0-mini", ["config.json", "model.safetensors"]),
    "openmidnight": ("SophontAI/OpenMidnight", "openmidnight", ["teacher_checkpoint_load.pt"]),
}

REPO_TYPES = {"ctranspath": "dataset"}

ALIASES = {
    "hibou_l": ["hibou-L.safetensors"],
    "virchow": ["../slide_encoders/virchow/pytorch_model.bin"],
}

REQUIRED_DIRECTORY_FILES = {
    "keep": ["config.json", "model.safetensors", "modeling_keep.py"],
    "phikon": ["config.json", "pytorch_model.bin"],
    "phikon_v2": ["config.json", "model.safetensors"],
    "hibou_l": ["config.json", "model.safetensors", "modeling_dinov2.py", "configuration_dinov2.py"],
    "midnight12k": ["config.json", "model.safetensors"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--encoders", nargs="*", default=None)
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def is_complete(root: Path, name: str) -> bool:
    path = checkpoint_path(root, name)
    if path is None or not path.exists():
        return False
    required = REQUIRED_DIRECTORY_FILES.get(name)
    return not required or all((path / filename).is_file() for filename in required)


def materialize_alias(root: Path, name: str) -> bool:
    destination = checkpoint_path(root, name)
    if destination is None or destination.exists():
        return bool(destination and destination.exists())
    for relative_source in ALIASES.get(name, []):
        source = (root / relative_source).resolve()
        if source.is_file():
            if name == "hibou_l":
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination / "model.safetensors")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            return True
    return False


def download_encoder(root: Path, name: str, token: str | None) -> None:
    repo_id, relative_dir, filenames = DOWNLOADS[name]
    target_dir = (root / relative_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_spec in filenames:
        if isinstance(file_spec, tuple):
            source_filename, target_filename = file_spec
        else:
            source_filename = target_filename = file_spec
        destination = target_dir / target_filename
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"[skip] {name}: {destination}")
            continue
        print(f"[download] {name}: {repo_id}/{source_filename}")
        downloaded = Path(hf_hub_download(
            repo_id=repo_id,
            filename=source_filename,
            repo_type=REPO_TYPES.get(name),
            token=token,
            local_dir=target_dir,
        ))
        if downloaded.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(destination)


def ensure_dinov2_source(root: Path) -> None:
    destination = root / "dinov2"
    if destination.exists():
        return
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "https://github.com/facebookresearch/dinov2.git",
            str(destination),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "7764ea0f912e53c92e82eb78a2a1631e92725fc8"],
        check=True,
    )


def status_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    for name in PATCH_ENCODER_LAYOUT:
        expected = expected_status(name)
        installed = is_complete(root, name)
        if expected == "excluded-gemma4":
            status = expected
        elif installed:
            status = "ready"
        elif expected == "ready":
            status = "missing"
        else:
            status = expected
        rows.append(
            {
                "encoder": name,
                "status": status,
                "path": str(checkpoint_path(root, name)),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime_cache = Path(os.environ.get("TRIDENT_RUNTIME_CACHE", REPO_ROOT / ".trident_runtime_cache"))
    os.environ.setdefault("HF_HOME", str(runtime_cache))
    os.environ.setdefault("HF_MODULES_CACHE", str(runtime_cache / "modules"))
    token = os.environ.get("HF_TOKEN")
    selected = args.encoders or list(DOWNLOADS)

    if not args.status_only:
        for name in PATCH_ENCODER_LAYOUT:
            materialize_alias(root, name)
        for name in selected:
            if name in EXCLUDED_ENCODERS:
                print(f"[excluded-gemma4] {name}")
                continue
            if name in BLOCKED_GATED_ENCODERS and not args.include_gated:
                print(f"[blocked-gated] {name}")
                continue
            if name not in DOWNLOADS:
                print(f"[local-only] {name}")
                continue
            try:
                download_encoder(root, name, token)
                if name == "openmidnight":
                    ensure_dinov2_source(root)
            except GatedRepoError as exc:
                print(f"[blocked-gated] {name}: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"[error] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    rows = status_rows(root)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

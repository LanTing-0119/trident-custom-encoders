#!/usr/bin/env python
"""Sequentially load patch encoders and optionally run one-image inference."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--encoders", nargs="*")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    os.environ["TRIDENT_PATCH_ENCODER_DIR"] = str(root)
    runtime_cache = Path(os.environ.get("TRIDENT_RUNTIME_CACHE", REPO_ROOT / ".trident_runtime_cache"))
    os.environ.setdefault("HF_HOME", str(runtime_cache))
    os.environ.setdefault("HF_MODULES_CACHE", str(runtime_cache / "modules"))

    from trident.patch_encoder_models import encoder_factory, encoder_registry
    from trident.patch_encoder_models.manifest import (
        BLOCKED_GATED_ENCODERS,
        EXCLUDED_ENCODERS,
        checkpoint_path,
    )

    names = args.encoders or [
        name for name in encoder_registry if name not in BLOCKED_GATED_ENCODERS | EXCLUDED_ENCODERS
    ]
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    rows = []

    for name in names:
        row = {"encoder": name, "path": str(checkpoint_path(root, name))}
        try:
            encoder = encoder_factory(name).eval()
            row["load"] = "ok"
            if args.forward:
                encoder = encoder.to(args.device)
                inputs = encoder.eval_transforms(image).unsqueeze(0).to(args.device)
                enabled = args.device.startswith("cuda") and encoder.precision in (torch.float16, torch.bfloat16)
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda" if args.device.startswith("cuda") else "cpu",
                    dtype=encoder.precision,
                    enabled=enabled,
                ):
                    output = encoder(inputs)
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"Expected Tensor, received {type(output).__name__}")
                if output.shape[0] != 1 or not torch.isfinite(output).all():
                    raise RuntimeError(f"Invalid output shape/values: {tuple(output.shape)}")
                row["forward"] = "ok"
                row["shape"] = list(output.shape)
            print(f"[ok] {name}: {row.get('shape', 'loaded')}")
        except Exception as exc:
            row["load"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[error] {name}: {row['error']}", file=sys.stderr)
        finally:
            if "encoder" in locals():
                del encoder
            if "inputs" in locals():
                del inputs
            if "output" in locals():
                del output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if any(row.get("load") == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

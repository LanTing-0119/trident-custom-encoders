# Portable Patch Encoders

This fork provides a portable, offline layout for all Trident patch encoders.
Set one environment variable on Windows or Linux:

```powershell
$env:TRIDENT_PATCH_ENCODER_DIR = "C:\models\patch_encoders"
```

```bash
export TRIDENT_PATCH_ENCODER_DIR=/path/to/patch_encoders
```

`PATCH_ENCODER_DIR` is accepted as a backwards-compatible alias. An explicit
`weights_path` argument or `<ENCODER_NAME>_WEIGHTS_PATH` environment variable
overrides the shared directory.

## Installation status

- `ready`: 28 encoders, downloaded and tested with one-image GPU inference.
- `blocked-gated`: `h0-mini`. The current Hugging Face account still reports
  that access to `bioptimus/H0-mini` is awaiting review.
- `excluded-gemma4`: `gemma4-e4b` and `gemma4-26b`. They require
  `transformers>=5` and are intentionally excluded from the shared
  `transformers==4.42.4` environment.

The machine-readable status is in
[`patch_encoder_status.json`](patch_encoder_status.json).

## Environment

```bash
conda env create -f environment_patch_encoders.yml
conda activate trident-patch-encoders
pip install -e .
```

The MUSK dependency is pinned to commit
`714b666969c1911e5efe70d991140a21030f4ef3`. OpenMidnight uses a local DINOv2
checkout pinned by the downloader.

## Download

The downloader is idempotent: completed files are skipped and only the
inference checkpoint format is downloaded.
RuiPath is the sole manual checkpoint because it is distributed through
ModelScope rather than Hugging Face.

```bash
python scripts/download_patch_encoders.py \
  --root /path/to/patch_encoders \
  --json-output patch_encoder_install_status.json
```

After access is approved for the remaining gated repository:

```bash
python scripts/download_patch_encoders.py \
  --root /path/to/patch_encoders \
  --include-gated \
  --encoders h0-mini
```

Access pages:

- https://huggingface.co/xiangjx/musk
- https://huggingface.co/bioptimus/H0-mini
- https://huggingface.co/SophontAI/OpenMidnight

## Verification

Load-only verification:

```bash
python scripts/verify_patch_encoders.py --root /path/to/patch_encoders
```

Sequential one-image GPU inference:

```bash
python scripts/verify_patch_encoders.py \
  --root /path/to/patch_encoders \
  --forward \
  --device cuda \
  --json-output patch_encoder_forward_validation.json
```

The verifier loads and releases one model at a time to limit GPU memory use.

## Custom encoders

This fork also adds:

| Encoder | Architecture | Input | Output | Checkpoint |
| --- | --- | --- | --- | --- |
| `digepath` | ViT-L/16 | 224 x 224 | 1024 | `digepath.model.safetensors` |
| `rui_path` | ViT-L/16 | 224 x 224 | 1024 | `ruipath_visionfoundation_v1.0.bin` |

Sources:

- Digepath: https://huggingface.co/xtxx/Digepath
- RuiPath: https://www.modelscope.cn/datasets/Ruijin_Hospital/RuiPath_Open_Source_Intro

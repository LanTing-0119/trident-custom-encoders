# Custom Patch Encoders

This fork adds two patch encoders to Trident:

| Encoder | Architecture | Input | Output | Default checkpoint filename |
| --- | --- | --- | --- | --- |
| `digepath` | ViT-L/16 | 224 x 224 | 1024 | `digepath.model.safetensors` |
| `rui_path` | ViT-L/16 | 224 x 224 | 1024 | `ruipath_visionfoundation_v1.0.bin` |

Model checkpoints are not stored in Git. Configure them on each machine using
one of the following methods.

## Shared checkpoint directory

Put both files in one directory:

```text
/path/to/patch_encoders/
  digepath.model.safetensors
  ruipath_visionfoundation_v1.0.bin
```

Linux:

```bash
export TRIDENT_PATCH_ENCODER_DIR=/path/to/patch_encoders
```

PowerShell:

```powershell
$env:TRIDENT_PATCH_ENCODER_DIR = "E:\Pathology_AI_Models\patch_encoders"
```

## Per-model checkpoint paths

Per-model variables override the shared directory and `local_ckpts.json`:

```bash
export DIGEPATH_WEIGHTS_PATH=/path/to/digepath.model.safetensors
export RUIPATH_WEIGHTS_PATH=/path/to/ruipath_visionfoundation_v1.0.bin
```

The batch CLI also supports an explicit path for one run:

```bash
python run_batch_of_slides.py \
  --task feat \
  --wsi_dir /path/to/wsis \
  --job_dir /path/to/output \
  --patch_encoder digepath \
  --patch_encoder_ckpt_path /path/to/digepath.model.safetensors \
  --mag 20 \
  --patch_size 256 \
  --gpus 0
```

## Installation

```bash
git clone <your-fork-url>
cd TRIDENT
conda create -n trident python=3.10 -y
conda activate trident
pip install -e .
```

Verify both models after configuring the checkpoint paths:

```bash
python - <<'PY'
import torch
from trident.patch_encoder_models import encoder_factory

for name in ("digepath", "rui_path"):
    encoder = encoder_factory(name).eval()
    output = encoder(torch.zeros(1, 3, 224, 224))
    print(name, tuple(output.shape))
PY
```

Expected output:

```text
digepath (1, 1024)
rui_path (1, 1024)
```

Sources:

- Digepath: https://huggingface.co/xtxx/Digepath
- RuiPath: https://www.modelscope.cn/datasets/Ruijin_Hospital/RuiPath_Open_Source_Intro

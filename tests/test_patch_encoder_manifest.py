import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from trident.patch_encoder_models.load import BasePatchEncoder
from trident.patch_encoder_models.manifest import (
    PATCH_ENCODER_LAYOUT,
    checkpoint_path,
    expected_status,
)


class _PathProbeEncoder(BasePatchEncoder):
    def _build(self):
        self.enc_name = "digepath"
        return torch.nn.Identity(), None, torch.float32


class TestPatchEncoderManifest(unittest.TestCase):
    def test_registry_covers_all_portable_encoders(self):
        from trident.patch_encoder_models import encoder_registry

        self.assertEqual(set(PATCH_ENCODER_LAYOUT), set(encoder_registry))

    def test_status_categories(self):
        self.assertEqual(expected_status("digepath"), "ready")
        self.assertEqual(expected_status("musk"), "blocked-gated")
        self.assertEqual(expected_status("gemma4-e4b"), "excluded-gemma4")

    def test_shared_directory_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = checkpoint_path(temp_dir, "digepath")
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.touch()
            encoder = _PathProbeEncoder()
            with patch.dict(os.environ, {"TRIDENT_PATCH_ENCODER_DIR": temp_dir}, clear=False):
                self.assertEqual(encoder._get_weights_path(), str(expected))


if __name__ == "__main__":
    unittest.main()

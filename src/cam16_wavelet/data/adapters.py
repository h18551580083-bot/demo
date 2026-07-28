from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class RGBInputAdapter:
    """Deterministic RGB conversion without fitting on test-set statistics."""

    def __init__(self, size: int = 256) -> None:
        self.size = size

    def transform(self, image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


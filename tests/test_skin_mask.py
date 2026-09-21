import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import skin_mask
from skin_mask import LIN_FaceSkinMask, LIN_SkinDeShineMaskLayer, LIN_SkinDeShineMaskEditor


class FakeAnalysis:
    def get_landmarks(self, image, extended_landmarks=True):
        face = np.array([[8, 8], [56, 8], [56, 56], [8, 56]])
        eyes = np.array([[20, 24], [28, 24], [28, 30], [20, 30], [36, 24], [44, 24], [44, 30], [36, 30]])
        brow = np.array([[18, 18], [30, 18], [30, 21], [18, 21]])
        mouth = np.array([[24, 40], [40, 40], [40, 47], [24, 47]])
        empty = np.array([[0, 0], [1, 0], [1, 1]])
        return [empty, empty, eyes, eyes[:4], eyes[4:], empty, mouth, brow, brow, face, face]


def main():
    image = torch.zeros((1, 64, 64, 3))
    skin_mask._get_insightface_model = lambda: FakeAnalysis()
    mask, preview = LIN_FaceSkinMask().create_mask(
        image, "insightface", 2, 1, 1.5,
        "add", 0, 0, 0.0, 0.0, 1.0, 0.35,
    )
    assert mask.shape == (1, 64, 64)
    assert preview.shape == image.shape
    assert mask[:, 32, 16].item() > 0.0
    assert mask[:, 26, 24].item() < 0.1
    assert mask[:, 0, 0].item() == 0.0
    layer_result = LIN_SkinDeShineMaskLayer().composite(
        image, torch.ones_like(image), mask, 0.8, 0.5,
    )
    layer_preview, edited = layer_result["result"]
    assert edited.shape == mask.shape
    assert layer_preview.shape == image.shape
    edited_image, edited_mask = LIN_SkinDeShineMaskEditor().edit(image, "", "add", mask)
    assert edited_image.shape == image.shape
    assert torch.equal(edited_mask, mask)
    print("skin mask node self-check passed")


if __name__ == "__main__":
    main()

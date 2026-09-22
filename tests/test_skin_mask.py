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


def _fake_mediapipe_points():
    points = np.full((478, 2), 32.0, dtype=np.float32)
    angles = np.linspace(-np.pi / 2.0, 3.0 * np.pi / 2.0, len(skin_mask._MEDIAPIPE_FACE_OVAL), endpoint=False)
    oval = np.stack((32.0 + np.cos(angles) * 22.0, 33.0 + np.sin(angles) * 25.0), axis=1)
    for index, point in zip(skin_mask._MEDIAPIPE_FACE_OVAL, oval):
        points[index] = point
    return points


def main():
    image = torch.zeros((1, 64, 64, 3))
    skin_mask._get_insightface_model = lambda: FakeAnalysis()

    mask, preview = LIN_FaceSkinMask().create_mask(
        image, "insightface", 2, 1, 1.5, 0.0,
        "add", 0, 0, 0.0, 0.0, 1.0, 0.35,
    )
    assert mask.shape == (1, 64, 64)
    assert preview.shape == image.shape
    assert mask[:, 32, 16].item() > 0.0
    assert mask[:, 26, 24].item() < 0.1
    assert mask[:, 0, 0].item() == 0.0

    # v1.1: forehead expansion must add coverage only above the original upper oval.
    points = _fake_mediapipe_points()
    base = skin_mask._mediapipe_face_mask(points, 64, 64, 0.0)
    expanded = skin_mask._mediapipe_face_mask(points, 64, 64, 8.0)
    assert expanded.sum() > base.sum()
    base_rows = np.where(base > 0)[0]
    expanded_rows = np.where(expanded > 0)[0]
    assert expanded_rows.min() < base_rows.min()

    # v1.1 experimental hybrid: agreement stays full strength; one-model-only
    # coverage remains partial instead of becoming a hard union.
    mp_mask = np.zeros((8, 8), dtype=np.float32)
    insight_mask = np.zeros((8, 8), dtype=np.float32)
    mp_mask[2:6, 2:5] = 1.0
    insight_mask[2:6, 3:7] = 1.0
    hybrid = skin_mask._blend_hybrid_masks(mp_mask, insight_mask, 0.55)
    assert hybrid[3, 3] == 1.0
    assert np.isclose(hybrid[3, 2], 0.55)
    assert np.isclose(hybrid[3, 6], 0.55)

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

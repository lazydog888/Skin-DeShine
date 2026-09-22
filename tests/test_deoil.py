import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deoil import LIN_DeOilSkin, _select_processing_mask


def main():
    image = torch.full((1, 64, 64, 3), 0.55)
    image[:, 20:44, 20:44] = 0.95
    image[:, 0:8] = 0.17
    mask = torch.zeros((1, 64, 64))
    mask[:, 12:52, 12:52] = 1.0

    output, shine, preview = LIN_DeOilSkin().deoil(
        image, mask, "inside_mask", 1.0, 0.78, 0.42,
        "add", 0, 0, 0.0, 0.0, 1.0, 0.35,
    )
    assert output.shape == image.shape
    assert shine.shape == mask.shape
    assert preview.shape == image.shape
    assert torch.equal(output[:, :8], image[:, :8])
    assert output.min() >= 0.0 and output.max() <= 1.0
    assert output[:, 32, 32].mean() < image[:, 32, 32].mean()

    # v1.1 processing-area modes.
    inside = _select_processing_mask(mask, "inside_mask")
    outside = _select_processing_mask(mask, "outside_mask")
    full = _select_processing_mask(mask, "full_image")
    assert torch.equal(inside, mask)
    assert torch.allclose(outside, 1.0 - mask)
    assert torch.all(full == 1.0)

    # Outside mode must leave the original inside-mask region untouched.
    outside_output, _, _ = LIN_DeOilSkin().deoil(
        image, mask, "outside_mask", 1.0, 0.78, 0.42,
        "add", 0, 0, 0.0, 0.0, 1.0, 0.35,
    )
    assert torch.equal(outside_output[:, 20:44, 20:44], image[:, 20:44, 20:44])

    print("deoil node self-check passed")


if __name__ == "__main__":
    main()

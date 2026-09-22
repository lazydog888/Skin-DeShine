import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deoil import LIN_DeOilSkin, _select_processing_mask


def run_node(image, processing_area, skin_mask=None):
    return LIN_DeOilSkin().deoil(
        image=image,
        processing_area=processing_area,
        strength=1.0,
        core_repair=0.78,
        color_repair=0.42,
        manual_mode="add",
        manual_expand=0,
        manual_shrink=0,
        manual_blur=0.0,
        manual_threshold=0.0,
        manual_strength=1.0,
        preview_opacity=0.35,
        skin_mask=skin_mask,
    )


def main():
    image = torch.full((1, 64, 64, 3), 0.55)
    image[:, 20:44, 20:44] = 0.95
    image[:, 0:8] = 0.17
    mask = torch.zeros((1, 64, 64))
    mask[:, 12:52, 12:52] = 1.0

    output, shine, preview = run_node(image, "inside_mask", mask)
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
    outside_output, _, _ = run_node(image, "outside_mask", mask)
    assert torch.equal(outside_output[:, 20:44, 20:44], image[:, 20:44, 20:44])

    # Full-image mode must work without any mask input.
    full_output, full_shine, full_preview = run_node(image, "full_image", None)
    assert full_output.shape == image.shape
    assert full_shine.shape == mask.shape
    assert full_preview.shape == image.shape

    # Masked modes must fail clearly if no mask is supplied.
    try:
        run_node(image, "inside_mask", None)
    except ValueError:
        pass
    else:
        raise AssertionError("inside_mask should require skin_mask")

    print("deoil node self-check passed")


if __name__ == "__main__":
    main()

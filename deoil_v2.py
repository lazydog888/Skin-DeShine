"""Skin DeShine v2 測試版（離線跑，不影響已安裝的節點）。

相對 v1 只改兩件事：
1. 壓暗 L 的同時，依壓暗量把 a/b 拉向周圍非高光膚色 → 不再變灰。
2. drop / repair 圖先羽化再套用 → 不再有硬邊。

用法：python deoil_v2.py 原圖.png 皮膚遮罩.png 輸出前綴
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .deoil import _blur, _lab_to_rgb, _masked_blur, _quantile, _rgb_to_lab, _smoothstep
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ComfyUI 內嵌 python 不會自動加腳本目錄
    from deoil import _blur, _lab_to_rgb, _masked_blur, _quantile, _rgb_to_lab, _smoothstep


def process_v2(image, skin_mask, strength=1.0, core_repair=0.78, color_repair=0.42, feather=0.025):
    mask = skin_mask.clamp(0.0, 1.0)
    valid = mask > 0.45
    if valid.any():
        xs = torch.where(valid)[1]
        face_width = max(1, int((xs.max() - xs.min() + 1).item()))
    else:
        face_width = max(image.shape[:2])

    lightness, a_channel, b_channel = _rgb_to_lab(image.clamp(0.0, 1.0)).unbind(-1)
    low_l = _blur(lightness, max(2.0, face_width * 0.007))
    broad_l = _blur(lightness, max(2.0, face_width * 0.075))

    q55, q72, q92 = (_quantile(low_l, mask, q) for q in (0.55, 0.72, 0.92))
    ramp = _smoothstep((low_l - q55) / (q92 - q55).clamp_min(1.0)) * mask
    local_excess = (low_l - broad_l).clamp_min(0.0)
    global_excess = (low_l - q72).clamp_min(0.0)
    drop = (local_excess * 0.48 + global_excess * 0.16).clamp(max=18.0) * ramp * strength
    # v2-2: 先擴張再羽化——邊界變柔，但高光中心的壓暗量不被攤薄
    radius = max(1, int(face_width * feather))
    drop = F.max_pool2d(drop[None, None], radius * 2 + 1, stride=1, padding=radius)[0, 0]
    drop = _blur(drop, max(1.0, radius * 0.6))
    base_l = low_l - drop

    q90, q965, q995 = (_quantile(low_l, mask, q) for q in (0.90, 0.965, 0.995))
    core = _smoothstep((low_l - q90) / (q995 - q90).clamp_min(1.0))
    core = core * _smoothstep((low_l - q72) / (q965 - q72).clamp_min(1.0)) * ramp
    repair = _blur((core * core_repair).clamp(0.0, 0.95), max(1.0, face_width * feather * 0.5))

    sigma = max(3.0, face_width * 0.035)
    source = (mask * (1.0 - repair)).clamp(0.0, 1.0)
    repaired_l = _masked_blur(low_l, source, sigma)
    target_l = base_l * (1.0 - repair) + repaired_l * repair

    # v2-1: 參考膚色取自「遮罩內、非高光」的區域；壓暗越多，顏色補得越多
    skin_ref = (mask * (1.0 - ramp)).clamp(0.0, 1.0)
    ref_a = _masked_blur(a_channel, skin_ref, sigma * 2.0)
    ref_b = _masked_blur(b_channel, skin_ref, sigma * 2.0)
    chroma = torch.maximum((drop / 8.0).clamp(0.0, 1.0), repair) * color_repair
    target_a = a_channel * (1.0 - chroma) + ref_a * chroma
    target_b = b_channel * (1.0 - chroma) + ref_b * chroma

    high_l = lightness - low_l
    # v2-4: 高光區裡只削「比周圍亮」的細節（毛孔反光），暗面紋理原樣保留
    zone = (drop / 6.0).clamp(0.0, 1.0)
    high_l = high_l - high_l.clamp_min(0.0) * zone * min(1.0, 0.7 * strength)
    restored_high = high_l * (1.0 - repair * 0.10) + _blur(high_l, max(0.8, face_width * 0.004)) * (repair * 0.10)
    output = _lab_to_rgb(torch.stack((target_l + restored_high, target_a, target_b), dim=-1))
    output = image + (output - image) * mask[..., None]
    return output.clamp(0.0, 1.0), core.clamp(0.0, 1.0), (drop / 18.0).clamp(0.0, 1.0)


def _load(path, mode):
    return torch.from_numpy(np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0)


def _save(tensor, path):
    Image.fromarray((tensor.numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(path)


if __name__ == "__main__":
    img_path, mask_path, prefix = sys.argv[1:4]
    color_repair = float(sys.argv[4]) if len(sys.argv) > 4 else 0.42
    image, mask = _load(img_path, "RGB"), _load(mask_path, "L")
    out, core, drop = process_v2(image, mask, color_repair=color_repair)
    assert out.shape == image.shape and torch.isfinite(out).all()
    assert torch.equal(out[mask == 0], image[mask == 0]), "遮罩外像素必須不變"
    _save(out, prefix + ".png")
    _save(drop, prefix + "_drop.png")

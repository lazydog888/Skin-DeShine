import torch
import torch.nn.functional as F

try:
    from .skin_mask import adjust_mask, overlay_mask
except ImportError:
    from skin_mask import adjust_mask, overlay_mask


def _smoothstep(value):
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _gaussian_kernel(sigma, device, dtype):
    radius = max(1, int(round(sigma * 3.0)))
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel[:, None] * kernel[None, :]


def _blur(channel, sigma):
    if sigma <= 0.0:
        return channel
    kernel = _gaussian_kernel(sigma, channel.device, channel.dtype)
    radius = kernel.shape[0] // 2
    padded = F.pad(channel[None, None], (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel[None, None])[0, 0]


def _masked_blur(channel, weights, sigma):
    denominator = _blur(weights, sigma)
    numerator = _blur(channel * weights, sigma)
    return torch.where(denominator > 1e-4, numerator / denominator.clamp_min(1e-4), channel)


def _quantile(values, mask, fraction):
    selected = values[mask > 0.45]
    if selected.numel() < 32:
        selected = values.reshape(-1)
    return torch.quantile(selected, fraction)


def _rgb_to_lab(rgb):
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).clamp_min(0.0).pow(2.4))
    red, green, blue = linear.unbind(-1)
    x = (red * 0.4124564 + green * 0.3575761 + blue * 0.1804375) / 0.95047
    y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    z = (red * 0.0193339 + green * 0.1191920 + blue * 0.9503041) / 1.08883
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def lab_curve(value):
        return torch.where(value > epsilon, value.clamp_min(0.0).pow(1.0 / 3.0), (kappa * value + 16.0) / 116.0)

    fx, fy, fz = lab_curve(x), lab_curve(y), lab_curve(z)
    return torch.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)), dim=-1)


def _lab_to_rgb(lab):
    lightness, a, b = lab.unbind(-1)
    fy = (lightness + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def xyz_curve(value):
        cube = value.pow(3.0)
        return torch.where(cube > epsilon, cube, (116.0 * value - 16.0) / kappa)

    x = xyz_curve(fx) * 0.95047
    y = xyz_curve(fy)
    z = xyz_curve(fz) * 1.08883
    red = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    green = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    blue = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
    linear = torch.stack((red, green, blue), dim=-1).clamp_min(0.0)
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.pow(1.0 / 2.4) - 0.055).clamp(0.0, 1.0)


def _process_image(image, skin_mask, strength, core_repair, color_repair):
    height, width = image.shape[:2]
    mask = skin_mask.clamp(0.0, 1.0)
    valid = mask > 0.45
    if valid.any():
        ys, xs = torch.where(valid)
        face_width = max(1, int((xs.max() - xs.min() + 1).item()))
    else:
        face_width = max(height, width)

    lab = _rgb_to_lab(image.clamp(0.0, 1.0))
    lightness, a_channel, b_channel = lab.unbind(-1)
    low_sigma = max(2.0, face_width * 0.007)
    low_l = _blur(lightness, low_sigma)
    broad_l = _blur(lightness, max(2.0, face_width * 0.075))

    q55, q72, q92 = (_quantile(low_l, mask, q) for q in (0.55, 0.72, 0.92))
    ramp = _smoothstep((low_l - q55) / (q92 - q55).clamp_min(1.0)) * mask
    local_excess = (low_l - broad_l).clamp_min(0.0)
    global_excess = (low_l - q72).clamp_min(0.0)
    drop = (local_excess * 0.48 + global_excess * 0.16).clamp(max=18.0) * ramp * strength
    base_l = low_l - drop

    q90, q965, q995 = (_quantile(low_l, mask, q) for q in (0.90, 0.965, 0.995))
    core = _smoothstep((low_l - q90) / (q995 - q90).clamp_min(1.0))
    core = core * _smoothstep((low_l - q72) / (q965 - q72).clamp_min(1.0)) * ramp
    repair = (core * core_repair).clamp(0.0, 0.95)

    reconstruction_sigma = max(3.0, face_width * 0.035)
    source = (mask * (1.0 - repair)).clamp(0.0, 1.0)
    repaired_l = _masked_blur(low_l, source, reconstruction_sigma)
    repaired_a = _masked_blur(a_channel, source, reconstruction_sigma)
    repaired_b = _masked_blur(b_channel, source, reconstruction_sigma)
    target_l = base_l * (1.0 - repair) + repaired_l * repair
    chroma_repair = repair * color_repair
    target_a = a_channel * (1.0 - chroma_repair) + repaired_a * chroma_repair
    target_b = b_channel * (1.0 - chroma_repair) + repaired_b * chroma_repair

    high_l = lightness - low_l
    softened_high = _blur(high_l, max(0.8, face_width * 0.004))
    restored_high = high_l * (1.0 - repair * 0.10) + softened_high * (repair * 0.10)
    output_lab = torch.stack((target_l + restored_high, target_a, target_b), dim=-1)
    output = _lab_to_rgb(output_lab)
    output = image + (output - image) * mask[..., None]
    return output.clamp(0.0, 1.0), core.clamp(0.0, 1.0)


class LIN_DeOilSkin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "輸入影像 / Input image."}),
                "skin_mask": ("MASK", {"tooltip": "皮膚遮罩 / Skin mask."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "去油光強度 / De-shine strength."}),
                "core_repair": ("FLOAT", {"default": 0.78, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "高光核心重建 / Highlight-core repair."}),
                "color_repair": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "膚色補回 / Color repair."}),
                "manual_mode": (["add", "subtract", "intersect", "replace"], {"default": "add", "tooltip": "手動遮罩合併 / Manual-mask mode."}),
                "manual_expand": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1, "tooltip": "遮罩外擴(px) / Expand."}),
                "manual_shrink": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1, "tooltip": "遮罩內縮(px) / Shrink."}),
                "manual_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 32.0, "step": 0.5, "tooltip": "再次羽化 / Extra blur."}),
                "manual_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "二值門檻；0=關 / Threshold; 0=off."}),
                "manual_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "手動遮罩強度 / Manual strength."}),
                "preview_opacity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "預覽透明度 / Preview opacity."}),
            },
            "optional": {
                "manual_mask": ("MASK", {"tooltip": "外部手動遮罩 / Optional manual mask."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "shine_mask", "mask_preview")
    FUNCTION = "deoil"
    CATEGORY = "LIN/Retouch"

    def deoil(
        self,
        image,
        skin_mask,
        strength,
        core_repair,
        color_repair,
        manual_mode,
        manual_expand,
        manual_shrink,
        manual_blur,
        manual_threshold,
        manual_strength,
        preview_opacity,
        manual_mask=None,
    ):
        skin_mask = adjust_mask(
            skin_mask,
            manual_mask=manual_mask,
            manual_mode=manual_mode,
            manual_expand=manual_expand,
            manual_shrink=manual_shrink,
            manual_blur=manual_blur,
            manual_threshold=manual_threshold,
            manual_strength=manual_strength,
            batch_size=image.shape[0],
            size=image.shape[1:3],
        )

        outputs = []
        masks = []
        for index, frame in enumerate(image):
            current_mask = skin_mask[index % skin_mask.shape[0]].to(device=frame.device, dtype=torch.float32)
            output, shine_mask = _process_image(frame.float(), current_mask, strength, core_repair, color_repair)
            outputs.append(output.to(dtype=image.dtype))
            masks.append(shine_mask)
        return torch.stack(outputs), torch.stack(masks), overlay_mask(image, skin_mask, preview_opacity).clamp(0.0, 1.0)


NODE_CLASS_MAPPINGS = {"LIN_DeOilSkin": LIN_DeOilSkin}
NODE_DISPLAY_NAME_MAPPINGS = {"LIN_DeOilSkin": "Skin DeShine"}

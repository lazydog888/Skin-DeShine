import cv2
import numpy as np
import torch
import torch.nn.functional as F
import uuid
from pathlib import Path


_INSIGHTFACE_MODEL = None
_MEDIAPIPE_FEATURES = (
    (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    (70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
    (300, 293, 334, 296, 336, 285, 295, 282, 283, 276),
    (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95),
)


class _NativeInsightFaceAdapter:
    """Use the installed InsightFace package when the optional ComfyUI wrapper is absent."""

    def __init__(self):
        from insightface.app import FaceAnalysis

        try:
            import folder_paths

            root = Path(folder_paths.models_dir) / "insightface"
        except ImportError:
            root = Path.home() / ".insightface"
        model_dir = root / "models" / "buffalo_l"
        if not model_dir.is_dir():
            raise RuntimeError(f"找不到 InsightFace buffalo_l 模型：{model_dir}")
        self.app = FaceAnalysis(name="buffalo_l", root=str(root))
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    @staticmethod
    def _ellipse(center, rx, ry):
        center = np.asarray(center, dtype=np.float32)
        angles = np.linspace(0.0, np.pi * 2.0, 24, endpoint=False)
        return np.stack((center[0] + np.cos(angles) * rx, center[1] + np.sin(angles) * ry), axis=1)

    def get_landmarks(self, image, extended_landmarks=True):
        faces = self.app.get(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not faces:
            return None
        face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        points = getattr(face, "landmark_2d_106", None)
        if points is not None:
            face_polygon = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
        else:
            x0, y0, x1, y1 = np.asarray(face.bbox, dtype=np.float32)
            face_polygon = np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=np.float32)

        keypoints = np.asarray(getattr(face, "kps", []), dtype=np.float32)
        if keypoints.shape == (5, 2):
            face_width = max(1.0, float(face.bbox[2] - face.bbox[0]))
            left_eye = self._ellipse(keypoints[0], face_width * 0.075, face_width * 0.045)
            right_eye = self._ellipse(keypoints[1], face_width * 0.075, face_width * 0.045)
            nose = self._ellipse(keypoints[2], face_width * 0.045, face_width * 0.055)
            mouth_center = (keypoints[3] + keypoints[4]) * 0.5
            mouth = self._ellipse(mouth_center, face_width * 0.13, face_width * 0.065)
        else:
            left_eye = right_eye = nose = mouth = face_polygon
        return [None, None, left_eye, None, None, None, nose, right_eye, mouth, face_polygon]


def _preview_ui(image):
    """Expose an IMAGE output as a temporary preview for ComfyUI widgets."""
    from PIL import Image as PILImage

    try:
        import folder_paths
        temp_dir = Path(folder_paths.get_temp_directory())
    except ImportError:
        import tempfile
        temp_dir = Path(tempfile.gettempdir())

    temp_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, frame in enumerate(image.detach().float().cpu().clamp(0.0, 1.0)):
        array = (frame.numpy() * 255.0 + 0.5).astype(np.uint8)
        if array.shape[-1] == 4:
            pil_image = PILImage.fromarray(array, mode="RGBA")
        else:
            pil_image = PILImage.fromarray(array[..., :3], mode="RGB")
        filename = f"lin_skin_deshine_{uuid.uuid4().hex[:10]}_{index}.png"
        try:
            pil_image.save(temp_dir / filename)
        except OSError:
            return {"images": []}
        results.append({"filename": filename, "subfolder": "", "type": "temp"})
    return {"images": results}


def _polygon_mask(points, height, width):
    points = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    return mask


def _finish_skin_mask(face, protected, feature_margin, boundary_shrink, edge_blur):
    if feature_margin > 0:
        size = feature_margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        protected = cv2.dilate(protected, kernel)

    if boundary_shrink > 0:
        size = boundary_shrink * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        face = cv2.erode(face, kernel)

    skin = np.where(protected > 0, 0, face).astype(np.float32) / 255.0
    if edge_blur > 0:
        skin = cv2.GaussianBlur(skin, (0, 0), edge_blur)
        skin *= face.astype(np.float32) / 255.0
    return np.clip(skin, 0.0, 1.0)


def _get_insightface_model():
    global _INSIGHTFACE_MODEL
    if _INSIGHTFACE_MODEL is None:
        try:
            from comfyui_faceanalysis.faceanalysis import InsightFace
            _INSIGHTFACE_MODEL = InsightFace("CPU")
        except ImportError:
            _INSIGHTFACE_MODEL = _NativeInsightFaceAdapter()
    return _INSIGHTFACE_MODEL


def _prepare_mask(mask, batch_size, size, device):
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError("MASK 必須是 [H,W] 或 [B,H,W]")
    if tuple(mask.shape[1:]) != tuple(size):
        mask = F.interpolate(mask[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1)
    elif mask.shape[0] != batch_size:
        mask = mask[:1].expand(batch_size, -1, -1)
    return mask.clamp(0.0, 1.0)


def adjust_mask(
    mask,
    manual_mask=None,
    manual_mode="add",
    manual_expand=0,
    manual_shrink=0,
    manual_blur=0.0,
    manual_threshold=0.0,
    manual_strength=1.0,
    batch_size=None,
    size=None,
):
    """Apply an optional hand-painted mask and simple edge refinements."""
    if mask.ndim == 2:
        inferred_batch = 1
        inferred_size = tuple(mask.shape)
    else:
        inferred_batch = mask.shape[0]
        inferred_size = tuple(mask.shape[-2:])
    batch_size = batch_size or inferred_batch
    size = size or inferred_size
    result = _prepare_mask(mask, batch_size, size, mask.device)

    if manual_mask is not None:
        edit = _prepare_mask(manual_mask, batch_size, size, result.device)
        edit = edit * float(manual_strength)
        if manual_mode == "subtract":
            result = result * (1.0 - edit)
        elif manual_mode == "intersect":
            result = torch.minimum(result, edit)
        elif manual_mode == "replace":
            result = edit
        else:
            result = torch.maximum(result, edit)

    frames = []
    for frame in result:
        array = frame.detach().float().cpu().numpy()
        if manual_expand > 0:
            size_px = int(manual_expand) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size_px, size_px))
            array = cv2.dilate(array, kernel)
        if manual_shrink > 0:
            size_px = int(manual_shrink) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size_px, size_px))
            array = cv2.erode(array, kernel)
        if manual_blur > 0.0:
            array = cv2.GaussianBlur(array, (0, 0), float(manual_blur))
        if manual_threshold > 0.0:
            array = (array >= float(manual_threshold)).astype(np.float32)
        frames.append(torch.from_numpy(np.clip(array, 0.0, 1.0)))
    return torch.stack(frames).to(device=result.device, dtype=torch.float32)


def overlay_mask(image, mask, opacity=0.35):
    """Return the original image with the current mask shown in green."""
    prepared = _prepare_mask(mask, image.shape[0], image.shape[1:3], image.device)
    alpha = prepared.clamp(0.0, 1.0) * float(opacity)
    color = torch.tensor((0.12, 1.0, 0.20), device=image.device, dtype=image.dtype)
    return image * (1.0 - alpha[..., None]) + color * alpha[..., None]


def _insightface_skin_mask(analysis_models, image, feature_margin, boundary_shrink, edge_blur):
    height, width = image.shape[:2]
    landmarks = None
    try:
        landmarks = analysis_models.get_landmarks(image, extended_landmarks=True)
    except Exception:
        landmarks = analysis_models.get_landmarks(image, extended_landmarks=False)
    if landmarks is None:
        return np.zeros((height, width), dtype=np.float32)

    face = _polygon_mask(landmarks[-1], height, width)
    protected = np.zeros((height, width), dtype=np.uint8)
    for feature in (landmarks[2], landmarks[7], landmarks[8], landmarks[6]):
        protected = np.maximum(protected, _polygon_mask(feature, height, width))

    return _finish_skin_mask(face, protected, feature_margin, boundary_shrink, edge_blur)


def _mediapipe_model_path():
    candidates = (
        Path(__file__).parent / "models" / "face_landmarker.task",
        Path(__file__).resolve().parents[2] / "models" / "mediapipe" / "face_landmarker.task",
        Path(r"D:\MediaPipe\face_landmarker.task"),
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("找不到 MediaPipe face_landmarker.task，請放在 ComfyUI/models/mediapipe 或 D:\\MediaPipe")


def _mediapipe_components(detector, image):
    import mediapipe as mp

    height, width = image.shape[:2]
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
    if not result.face_landmarks:
        return None, None

    faces = [
        np.array([(point.x * width, point.y * height) for point in face], dtype=np.float32)
        for face in result.face_landmarks
    ]
    points = max(faces, key=lambda value: (value[:, 0].max() - value[:, 0].min()) * (value[:, 1].max() - value[:, 1].min()))
    face = _polygon_mask(points, height, width)
    protected = np.zeros((height, width), dtype=np.uint8)
    for indices in _MEDIAPIPE_FEATURES:
        protected = np.maximum(protected, _polygon_mask(points[list(indices)], height, width))
    return face, protected


def _mediapipe_skin_mask(detector, image, feature_margin, boundary_shrink, edge_blur):
    height, width = image.shape[:2]
    face, protected = _mediapipe_components(detector, image)
    if face is None:
        return np.zeros((height, width), dtype=np.float32)
    return _finish_skin_mask(face, protected, feature_margin, boundary_shrink, edge_blur)


def _run_sam3(image_tensor, prompt, threshold):
    """沒接 sam3_mask 時，直接呼叫已安裝的 comfyui-rmbg SAM3Segment。"""
    import nodes

    node_class = nodes.NODE_CLASS_MAPPINGS.get("SAM3Segment")
    if node_class is None:
        raise RuntimeError("找不到 SAM3Segment（comfyui-rmbg）；請安裝，或手動把 SAM3 MASK 接到 sam3_mask。")
    _, masks, _ = node_class().segment(image_tensor, prompt, "Auto", confidence_threshold=threshold)
    return masks


def _forehead_region(face, forehead_extend):
    """MediaPipe 臉框往上延伸，補回額頭上半部；只用來限制 SAM3 的範圍。"""
    ys, xs = np.nonzero(face)
    if ys.size == 0:
        return face
    shift = int((ys.max() - ys.min()) * forehead_extend)
    region = face.copy()
    if shift > 0:
        region[:-shift] = np.maximum(region[:-shift], face[shift:])
    size = max(3, int((xs.max() - xs.min()) * 0.04)) * 2 + 1
    return cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def _hybrid_skin_mask(detector, image, sam3_mask, feature_margin, boundary_shrink, edge_blur, forehead_extend=0.25):
    height, width = image.shape[:2]
    mp_face, protected = _mediapipe_components(detector, image)
    if protected is None:
        return np.zeros((height, width), dtype=np.float32)

    # SAM3 負責真實的皮膚邊界（額頭到髮際線），MediaPipe 臉框限制範圍（不含脖子、耳朵），再和臉框取聯集補洞
    sam3 = (np.clip(sam3_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    face = np.maximum(np.minimum(sam3, _forehead_region(mp_face, forehead_extend)), mp_face).astype(np.float32)
    skin = _finish_skin_mask(face, protected, feature_margin, boundary_shrink, edge_blur)
    if feature_margin > 0:
        size = feature_margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        protected = cv2.dilate(protected, kernel)
    protect_alpha = protected.astype(np.float32) / 255.0
    if edge_blur > 0:
        protect_alpha = cv2.GaussianBlur(protect_alpha, (0, 0), max(0.75, edge_blur * 0.5))
    return np.clip(skin * (1.0 - protect_alpha), 0.0, 1.0)


class LIN_FaceSkinMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "要分析的人像原圖；節點會以畫面中最大的人臉為主要目標。"}),
                "algorithm": (["mediapipe", "insightface", "mediapipe_sam3"], {"default": "mediapipe", "tooltip": "選擇遮罩算法。mediapipe_sam3：SAM3 圈皮膚（額頭到髮際線），MediaPipe 限制在臉部並排除五官；沒接 sam3_mask 時會自動執行 SAM3。"}),
                "feature_margin": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1, "tooltip": "五官保護區向外擴張的像素數；數值越大，眼睛、眉毛和嘴唇周圍越不容易被納入皮膚。"}),
                "boundary_shrink": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1, "tooltip": "臉部外框向內縮的像素數，用來減少頭髮、耳朵和背景被選入。"}),
                "edge_blur": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 32.0, "step": 0.5, "tooltip": "遮罩邊緣羽化半徑；數值越大，遮罩邊界越柔和。"}),
                "manual_mode": (["add", "subtract", "intersect", "replace"], {"default": "add", "tooltip": "手動 MASK 的合併方式：add 加入、subtract 排除、intersect 取交集、replace 完全取代自動遮罩。"}),
                "manual_expand": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1, "tooltip": "對合併後遮罩向外擴張的像素數；0 代表不調整。"}),
                "manual_shrink": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1, "tooltip": "對合併後遮罩向內收縮的像素數；0 代表不調整。"}),
                "manual_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 32.0, "step": 0.5, "tooltip": "對最終遮罩再次羽化的半徑；0 代表保留原有邊緣。"}),
                "manual_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "將遮罩二值化的門檻；0 代表不二值化，適合保留柔邊。"}),
                "manual_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "手動 MASK 的影響強度；1 為完整套用，0 為不套用。"}),
                "preview_opacity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "預覽疊圖中遮罩顏色的透明度；不會改變輸出的 MASK。"}),
            },
            "optional": {
                "manual_mask": ("MASK", {"tooltip": "可接外部手工繪製或修正過的 MASK，搭配 manual_mode 與 manual_strength 使用。"}),
                "sam3_mask": ("MASK", {"tooltip": "僅供 mediapipe_sam3：可接自己的 SAM3 MASK；不接則用 sam3_prompt 自動執行。"}),
                "sam3_prompt": ("STRING", {"default": "skin", "tooltip": "mediapipe_sam3 自動執行 SAM3 時的文字提示。"}),
                "forehead_extend": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.6, "step": 0.01, "tooltip": "mediapipe_sam3：臉框往上延伸的比例（相對臉高），讓 SAM3 能圈到額頭上半部。"}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("skin_mask", "preview")
    FUNCTION = "create_mask"
    CATEGORY = "LIN/Retouch"

    def create_mask(
        self,
        image,
        algorithm,
        feature_margin,
        boundary_shrink,
        edge_blur,
        manual_mode,
        manual_expand,
        manual_shrink,
        manual_blur,
        manual_threshold,
        manual_strength,
        preview_opacity,
        manual_mask=None,
        sam3_mask=None,
        sam3_prompt="skin",
        forehead_extend=0.25,
    ):
        masks = []
        if algorithm in ("mediapipe", "mediapipe_sam3"):
            import mediapipe as mp

            sam3_masks = None
            if algorithm == "mediapipe_sam3":
                if sam3_mask is None:
                    sam3_mask = _run_sam3(image, sam3_prompt, 0.4)
                sam3_masks = _prepare_mask(sam3_mask, image.shape[0], image.shape[1:3], image.device)

            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(_mediapipe_model_path())),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=5,
            )
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                for index, frame in enumerate(image):
                    rgb = (frame[..., :3].clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
                    if algorithm == "mediapipe_sam3":
                        sam3_frame = sam3_masks[index].detach().float().cpu().numpy()
                        mask = _hybrid_skin_mask(detector, rgb, sam3_frame, feature_margin, boundary_shrink, edge_blur, forehead_extend)
                    else:
                        mask = _mediapipe_skin_mask(detector, rgb, feature_margin, boundary_shrink, edge_blur)
                    masks.append(torch.from_numpy(mask))
        else:
            analysis_models = _get_insightface_model()
            for frame in image:
                rgb = (frame[..., :3].clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
                masks.append(torch.from_numpy(_insightface_skin_mask(analysis_models, rgb, feature_margin, boundary_shrink, edge_blur)))
        skin_mask = adjust_mask(
            torch.stack(masks),
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
        preview = overlay_mask(image, skin_mask, preview_opacity).clamp(0.0, 1.0)
        # ui 預覽讓下游的 Mask Layer 畫布能直接拿「原圖＋自動遮罩」當底圖
        return {"ui": _preview_ui(preview), "result": (skin_mask, preview)}


class LIN_SkinDeShineMaskLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE", {"tooltip": "底層／背景圖，通常接原始人像。"}),
                "source": ("IMAGE", {"tooltip": "要插入的圖層，通常接 MaskToImage 的輸出，或接另一張背景／參考圖。"}),
                "mask": ("MASK", {"tooltip": "控制 source 圖層出現位置的遮罩，通常接 Skin DeShine Mask 的 skin_mask。"}),
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "遮罩透明度；1 代表完整使用 MASK，0 代表不顯示 source。輸出的 mask 也會套用這個透明度。"}),
                "background_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "source 圖層的顯示透明度；只影響合成預覽，不會削弱輸出的 mask。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "composite"
    CATEGORY = "LIN/Retouch"

    def composite(self, destination, source, mask, mask_opacity, background_opacity):
        if destination.ndim != 4 or source.ndim != 4:
            raise ValueError("destination 與 source 必須是 IMAGE")

        height, width = destination.shape[1:3]
        if source.shape[1:3] != (height, width):
            source = F.interpolate(
                source.movedim(-1, 1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
        if source.shape[0] == 1 and destination.shape[0] > 1:
            source = source.expand(destination.shape[0], -1, -1, -1)
        elif source.shape[0] != destination.shape[0]:
            source = source[:1].expand(destination.shape[0], -1, -1, -1)

        output_mask = _prepare_mask(mask, destination.shape[0], (height, width), destination.device)
        output_mask = (output_mask * float(mask_opacity)).clamp(0.0, 1.0)
        blend = output_mask * float(background_opacity)
        image = destination.float() * (1.0 - blend[..., None]) + source.to(destination.device).float() * blend[..., None]
        image = image.clamp(0.0, 1.0).to(dtype=destination.dtype)
        return {"ui": _preview_ui(image), "result": (image, output_mask)}


class LIN_SkinDeShineMaskEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "畫布底圖。paint_on_image：接原圖（Load Image）。edit_auto_mask：接 Skin DeShine Mask 的 preview（原圖＋綠色自動遮罩），先跑一次才會顯示。"}),
                "mask": ("STRING", {"default": "", "multiline": False, "widgetType": "PAINTER", "image_upload": True, "tooltip": "在底圖上直接畫。edit_auto_mask 模式：任何顏色＝加入，紅色筆刷＝擦掉自動遮罩。"}),
                "edit_mode": (["paint_on_image", "edit_auto_mask"], {"default": "edit_auto_mask", "tooltip": "paint_on_image：只用你畫的範圍當遮罩。edit_auto_mask：以自動遮罩為基礎，一般顏色加、紅色減。"}),
            },
            "optional": {
                "auto_mask": ("MASK", {"tooltip": "edit_auto_mask 用：接 Skin DeShine Mask 的 skin_mask。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "edit"
    CATEGORY = "LIN/Retouch"

    def edit(self, image, mask, edit_mode="edit_auto_mask", auto_mask=None):
        import os

        if image.ndim != 4:
            raise ValueError("image 必須是 IMAGE")
        base_image = image[:1].float()
        height, width = base_image.shape[1:3]

        strokes = np.zeros((height, width, 4), dtype=np.float32)
        if mask and mask.strip():
            import folder_paths
            import node_helpers
            from PIL import Image

            mask_path = folder_paths.get_annotated_filepath(mask)
            if not os.path.isfile(mask_path):
                raise ValueError(f"找不到 Painter MASK 檔案：{mask}")
            painter_img = node_helpers.pillow(Image.open, mask_path).convert("RGBA")
            if painter_img.size != (width, height):
                painter_img = painter_img.resize((width, height), Image.LANCZOS)
            strokes = np.asarray(painter_img).astype(np.float32) / 255.0

        output_mask = torch.from_numpy(combine_strokes(strokes, auto_mask_np(auto_mask, height, width), edit_mode))[None]
        output_image = overlay_mask(base_image, output_mask, 0.35)
        return output_image.clamp(0.0, 1.0).to(dtype=image.dtype), output_mask.clamp(0.0, 1.0)


def auto_mask_np(auto_mask, height, width):
    if auto_mask is None:
        return np.zeros((height, width), dtype=np.float32)
    return _prepare_mask(auto_mask, 1, (height, width), torch.device("cpu"))[0].numpy()


def combine_strokes(strokes, auto, edit_mode):
    """strokes: Painter 的 RGBA（0–1）。紅色筆刷在 edit_auto_mask 模式下代表擦除。"""
    rgb, alpha = strokes[..., :3], strokes[..., 3]
    if edit_mode == "paint_on_image":
        return alpha.astype(np.float32)
    red = (rgb[..., 0] > 0.6) & (rgb[..., 1] < 0.35) & (rgb[..., 2] < 0.35)
    result = np.maximum(auto, np.where(red, 0.0, alpha))
    return (result * (1.0 - np.where(red, alpha, 0.0))).astype(np.float32)

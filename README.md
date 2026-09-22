# Skin-DeShine

A ComfyUI custom node for reducing unwanted skin shine and specular highlights while preserving image structure and high-frequency skin detail.

Skin-DeShine separates **where to process** from **how to repair**:

- **Skin DeShine Mask** builds a face-skin processing mask.
- **Skin DeShine** reduces excessive low-frequency brightness and reconstructs severe highlight cores while preserving most original high-frequency texture.

> 中文摘要：本節點以臉部皮膚遮罩限制處理範圍，在 Lab 的亮度低頻上抑制相對油光，並對極亮高光核心做有限度的亮度／膚色重建。v1.1 主要改善遮罩精度，尤其是 MediaPipe 額頭覆蓋不足的問題。

## Before / After

| Before | After |
| --- | --- |
| ![Before Skin DeShine](examples/before.jpg) | ![After Skin DeShine](examples/after.jpg) |

The example images are resized/compressed for GitHub display only.

## Nodes

Available under `LIN/Retouch`:

- **Skin DeShine** (`LIN_DeOilSkin`)
- **Skin DeShine Mask** (`LIN_FaceSkinMask`)
- **Skin DeShine Mask Composite** (`LIN_SkinDeShineMaskLayer`)
- **Skin DeShine Mask Layer** (`LIN_SkinDeShineMaskEditor`)

Canonical workflow:

[`workflows/Skin-DeShine.json`](workflows/Skin-DeShine.json)

## v1.1 mask improvements

The v1.1 development branch focuses on mask quality rather than changing the de-shine algorithm.

### Better MediaPipe forehead coverage

MediaPipe Face Landmarker can produce a facial oval that is too conservative around the upper forehead. v1.1 adds an adaptive **`forehead_expand`** control.

Instead of dilating the complete face mask, Skin-DeShine extends only the upper part of the facial oval. This reduces the risk of unnecessarily expanding the cheeks, jaw, or background.

The eyebrow protection margin is also smaller than the eye/mouth protection margin, so protecting the eyebrows does not remove too much lower-forehead skin.

Default:

```text
forehead_expand = 5.5
```

The value is expressed as a percentage of detected face height.

### Experimental hybrid mask

v1.1 adds a third mask backend:

```text
mediapipe
insightface
hybrid
```

The **hybrid** mode combines MediaPipe and InsightFace masks conservatively:

- Areas detected by both backends receive full mask strength.
- Areas detected by only one backend receive partial weight (currently 0.55).
- It is intentionally not a hard union.

This makes hybrid useful for comparison and experimentation, but it should not be assumed to be more accurate in every image.

**Important:** hybrid mode requires the optional InsightFace integration described below. The default remains `mediapipe`.

## Installation

### 1. Clone into ComfyUI `custom_nodes`

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lazydog888/Skin-DeShine.git
```

For the current v1.1 development version:

```bash
git checkout v1.1-dev
```

### 2. Install Python dependencies

Use the same Python environment that launches ComfyUI:

```bash
python -m pip install -r ComfyUI/custom_nodes/Skin-DeShine/requirements.txt
```

ComfyUI Portable on Windows commonly uses:

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\Skin-DeShine\requirements.txt
```

MediaPipe is the primary package dependency. PyTorch, Pillow and NumPy are expected from the existing ComfyUI environment.

### 3. Download the MediaPipe Face Landmarker model

The default MediaPipe backend requires `face_landmarker.task`. The model is not included in this repository.

Official model:

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Recommended location:

```text
ComfyUI/models/mediapipe/face_landmarker.task
```

Also supported:

```text
ComfyUI/custom_nodes/Skin-DeShine/models/face_landmarker.task
```

### 4. Restart ComfyUI

Restart ComfyUI completely and look for the nodes under `LIN/Retouch`.

## Skin DeShine Mask

### `mediapipe` — default

Uses MediaPipe Face Landmarker to obtain dense face landmarks.

The node:

1. selects the largest detected face,
2. builds a facial skin region,
3. excludes eyes, eyebrows, and mouth,
4. optionally extends the upper forehead,
5. shrinks the outer face boundary,
6. feathers the final mask.

MediaPipe is the default because it does not require the optional `ComfyUI_FaceAnalysis` integration.

### `insightface` — optional

Skin-DeShine attempts to load InsightFace internally from:

[`ComfyUI_FaceAnalysis`](https://github.com/cubiq/ComfyUI_FaceAnalysis)

This is an optional integration, not a required dependency.

### `hybrid` — experimental

Runs both MediaPipe and InsightFace, then blends their masks using model agreement.

Use it when comparing mask coverage or when one backend misses parts of the face. It is not guaranteed to outperform either backend on every image.

## Main mask controls

- `feature_margin`: protects facial features from the skin mask.
- `boundary_shrink`: pulls the outer mask inward.
- `edge_blur`: feathers mask boundaries.
- `forehead_expand`: extends only the upper forehead region (v1.1).
- `manual_mask`: optional hand-painted correction.
- `manual_mode`: `add`, `subtract`, `intersect`, or `replace`.
- `manual_expand` / `manual_shrink`: post-adjust mask size.
- `manual_blur`: additional feathering.
- `manual_threshold`: optional hard threshold; `0` keeps soft edges.
- `manual_strength`: manual-mask influence.
- `preview_opacity`: preview overlay only.

## Processing-area modes

v1.1 adds a `processing_area` control to **Skin DeShine**:

- `inside_mask` — original behavior. Only pixels inside `skin_mask` can be modified.
- `outside_mask` — invert the supplied mask and process only the area outside it.
- `full_image` — ignore mask limits and allow processing across the whole image. In this mode, `skin_mask` is optional.

中文：

- `inside_mask`：原本模式，只處理遮罩內。
- `outside_mask`：反轉遮罩，只處理遮罩外。
- `full_image`：非遮罩模式，整張圖都可處理，而且可以不接 `skin_mask`。

> **Caution:** the de-shine algorithm was designed around skin-region statistics. `outside_mask` and `full_image` can include hair, clothes, background, or other non-skin materials, so they are experimental and should be used mainly for missed skin areas or controlled testing.

## Skin DeShine algorithm

The repair node always accepts an `IMAGE`. A skin `MASK` is required for `inside_mask` and `outside_mask`, but optional for `full_image`.

High-level pipeline:

1. Convert RGB to Lab.
2. Separate brightness (`L`) from the color channels (`a/b`).
3. Build low-frequency brightness at scales relative to the masked face size.
4. Detect relatively bright skin using local brightness excess and skin-region quantiles.
5. Reduce ordinary shine mainly in low-frequency brightness.
6. Detect severe highlight cores.
7. Reconstruct those cores using surrounding valid skin.
8. Optionally restore `a/b` skin color with `color_repair`.
9. Restore most original high-frequency detail.
10. Blend the result only inside the resolved processing area (`inside_mask`, inverted `outside_mask`, or `full_image`).

Main controls:

- `strength`: overall de-shine amount.
- `core_repair`: reconstruction strength for very bright highlight cores.
- `color_repair`: amount of surrounding skin color restored into highlight cores.

Python defaults:

```text
strength = 1.0
core_repair = 0.78
color_repair = 0.42
```

The canonical workflow intentionally uses a stronger preset:

```text
strength = 1
core_repair = 0.95
color_repair = 1
```

## Manual mask editing

The package includes helper nodes for previewing and editing masks.

**Skin DeShine Mask Composite**

Combines a source image, destination image and mask for mask inspection.

**Skin DeShine Mask Layer**

Provides a ComfyUI Painter-based mask workflow:

- `add`: preserve the automatic mask and add painted regions.
- `replace`: use the painted mask instead.

## Optional iTools dependency

The canonical workflow uses **`iToolsCompareImage`** for before/after comparison.

It comes from:

[`ComfyUI-iTools`](https://github.com/MohammadAboulEla/ComfyUI-iTools)

This is only a workflow dependency. Skin-DeShine itself does not require iTools.

## Tests

Core checks:

```bash
python tests/test_deoil.py
python tests/test_skin_mask.py
python tests/test_workflow_schema.py
```

The v1.1 mask tests include:

- forehead-expansion regression coverage,
- conservative hybrid-mask blending,
- mask shape/range checks,
- feature exclusion,
- workflow schema checks.

These tests do not replace real-image evaluation with the MediaPipe model. Mask quality should still be visually checked across different faces, hairstyles, camera angles, and lighting conditions.

## Version safety

The repository keeps the previous stable state on:

```text
v1.0-stable
```

Current experimental development is isolated on:

```text
v1.1-dev
```

The `main` branch is not modified by v1.1 development until the changes are reviewed and intentionally merged.

## License

GPL-3.0. See [LICENSE](LICENSE).

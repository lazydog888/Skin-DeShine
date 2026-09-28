# Skin-DeShine

**English** | [繁體中文](README.zh-TW.md)

A ComfyUI custom node for reducing excessive skin shine while preserving most original skin texture. It uses a skin mask and low-frequency Lab adjustments to soften highlights, with limited reconstruction of bright highlight cores.

## Before / After

| Before | After |
| --- | --- |
| ![Before](examples/before.jpg) | ![After](examples/after.jpg) |

## What's new in v2 (dev)

- **Skin DeShine `version` (default `v2`)**: restores the surrounding skin colour while lowering highlights (no grey patches), feathers the correction so there are no hard edges, and trims bright pore-level glints while keeping the darker texture. `v1` stays available for comparison.
- **`mediapipe_sam3` mask mode**: runs SAM3 (comfyui-rmbg) automatically when `sam3_mask` is not connected. SAM3 traces the real skin edge up to the hairline; the MediaPipe face region, extended upward by `forehead_extend`, keeps it off the neck and ears and removes eyes, brows and lips.
- **Mask Layer `edit_mode`**: `paint_on_image` (paint directly on the original image) or `edit_auto_mask` (connect the mask node's `preview`; any colour adds, a **red** brush erases the auto mask). Workflows saved with the old `add` / `replace` values need the mode re-selected.
- Tip: `edge_blur` around 8 avoids visible seams around the eyebrows.

## Install

Run these commands from your ComfyUI directory, using the same Python environment that runs ComfyUI:

```bash
git clone https://github.com/lazydog888/Skin-DeShine.git custom_nodes/Skin-DeShine
python -m pip install -r custom_nodes/Skin-DeShine/requirements.txt
```

Download Google's [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) into:

```text
ComfyUI/models/mediapipe/face_landmarker.task
```

Alternatively, place it in `custom_nodes/Skin-DeShine/models/`. The model is not included in this repository. Restart ComfyUI after installation.

## Get started

Load [the example workflow](workflows/Skin-DeShine.json) or connect your own `IMAGE` and skin `MASK` to **Skin DeShine**. Use **Skin DeShine Mask** to generate a facial skin mask automatically.

Four nodes are available under `LIN/Retouch`: **Skin DeShine**, **Skin DeShine Mask**, **Skin DeShine Mask Composite**, and **Skin DeShine Mask Layer**.

## Notes

- **MediaPipe** is the default mask method. **InsightFace** is optional and requires [ComfyUI_FaceAnalysis](https://github.com/cubiq/ComfyUI_FaceAnalysis); no external `analysis_models` input is needed.
- The example workflow uses **iToolsCompareImage** from [ComfyUI-iTools](https://github.com/MohammadAboulEla/ComfyUI-iTools) for its comparison view. This is optional for using the Skin-DeShine nodes.
- The workflow retains its stronger settings: `strength=1`, `core_repair=0.95`, `color_repair=1`. The node's Python defaults are `1 / 0.78 / 0.42`.
- The tool reduces bright highlights; it cannot recover image detail already lost to clipping.

Node descriptions are also available in English and Traditional Chinese via the supplied `locales/` files. Self-checks are in [`tests/`](tests/).

**License:** [GPL-3.0](LICENSE).

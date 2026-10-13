# Skin-DeShine

**English** | [繁體中文](README.zh-TW.md)

A ComfyUI custom node that reduces excessive highlights on skin while keeping the original texture. It works only inside a facial skin mask, lowers bright areas with low-frequency Lab adjustments, and puts the surrounding skin colour back so corrected areas do not turn grey.

## Before / After

| Before | After |
| --- | --- |
| ![Before](examples/before.jpg) | ![After](examples/after.jpg) |

## Features

- **Natural highlight reduction (v2)**: lowers shine on the forehead, nose and cheeks, restores the surrounding skin colour, feathers the correction so there are no hard edges, and softens bright pore-level glints while keeping the darker texture. The original algorithm is still available as `version = v1`.
- **Automatic skin mask**: MediaPipe finds the face and protects the eyes, brows and lips. The `mediapipe_sam3` mode adds SAM3 so the mask follows the real skin edge up to the hairline instead of stopping mid-forehead.
- **Manual mask editing**: paint directly on the image (`paint_on_image`), or start from the automatic mask and adjust it (`edit_auto_mask`): any colour adds, a **red** brush erases.

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

For the `mediapipe_sam3` mask mode, also install [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG), which provides SAM3.

## Get started

Load [the example workflow](workflows/Skin-DeShine.json). It runs:

**Load Image → Skin DeShine Mask → Skin DeShine Mask Layer → Skin DeShine**

- **Skin DeShine Mask** builds the skin mask (`mediapipe_sam3` in the example).
- **Skin DeShine Mask Layer** passes the mask through unchanged until you paint on it. Run the workflow once and its canvas shows the image with the mask in green; paint to add, use red to erase.
- **Skin DeShine** reduces the highlights. Start with `strength` around 0.5 and raise it if more shine should go.

You can also connect any `IMAGE` and skin `MASK` straight to **Skin DeShine**. All nodes are under `LIN/Retouch`.

## Notes

- Mask methods: **MediaPipe** (default), **MediaPipe + SAM3**, or **InsightFace** (optional, requires [ComfyUI_FaceAnalysis](https://github.com/cubiq/ComfyUI_FaceAnalysis)).
- The mask is required. It limits the correction to skin and also tells the node what counts as a highlight on this face; processing the whole frame darkens eyes, lips, hair and the background.
- Higher `strength` flattens the lit side of side-lit faces. Values around 0.5 look most natural on soft studio light.
- The example workflow uses **iToolsCompareImage** from [ComfyUI-iTools](https://github.com/MohammadAboulEla/ComfyUI-iTools) for its comparison view; it is optional.
- The node lowers bright highlights; it cannot recover detail already lost to clipping.

Node descriptions are available in English and Traditional Chinese via `locales/`. Self-checks are in [`tests/`](tests/).

**License:** [GPL-3.0](LICENSE).

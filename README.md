# Skin-DeShine

**English** | [繁體中文](README.zh-TW.md)

A ComfyUI custom node for reducing unwanted skin shine and specular highlights while preserving image structure and most high-frequency skin detail.

Skin-DeShine works on a user-supplied skin mask. It suppresses relative shine in the low-frequency luminance of Lab color space and applies limited reconstruction to extremely bright highlight cores. The final correction is blended only within the supplied mask. Its automatic mask generator is designed for faces; manual masks can be used to refine or extend the processing area.

## Before / After

| Before | After |
| --- | --- |
| ![Before Skin-DeShine](examples/before.jpg) | ![After Skin-DeShine](examples/after.jpg) |

> The example images have been resized and compressed for GitHub. This does not change the node's processing pipeline or the canonical workflow.

## Nodes

After installation, find these four nodes under **LIN/Retouch**:

- **Skin DeShine** (`LIN_DeOilSkin`) — reduce skin shine and repair bright highlight cores.
- **Skin DeShine Mask** (`LIN_FaceSkinMask`) — generate a face-skin mask and optionally combine it with a hand-painted mask.
- **Skin DeShine Mask Composite** (`LIN_SkinDeShineMaskLayer`) — preview a masked image composite and output its mask.
- **Skin DeShine Mask Layer** (`LIN_SkinDeShineMaskEditor`) — edit or replace a mask using ComfyUI Painter.

The maintained example is [`workflows/Skin-DeShine.json`](workflows/Skin-DeShine.json). It is the project's **canonical workflow**.

## Installation

### 1. Clone into ComfyUI's custom_nodes directory

~~~bash
cd ComfyUI/custom_nodes
git clone https://github.com/lazydog888/Skin-DeShine.git
~~~

The repository root is already a ComfyUI custom-node package. Do **not** create an additional `lin_deoil_node` subdirectory.

### 2. Install Python dependencies

Use the **same Python environment that runs ComfyUI**:

~~~bash
python -m pip install -r ComfyUI/custom_nodes/Skin-DeShine/requirements.txt
~~~

A common command for ComfyUI Portable on Windows is:

~~~bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\Skin-DeShine\requirements.txt
~~~

MediaPipe is the principal package-specific dependency. The existing ComfyUI environment supplies common packages such as PyTorch, Pillow, and NumPy. MediaPipe 1.x also installs its required OpenCV runtime.

### 3. Download the MediaPipe Face Landmarker model

The default `mediapipe` mask algorithm requires Google's `face_landmarker.task` model. **The model is not included in this repository and should not be committed to Git.**

Official model:

~~~text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
~~~

Recommended location:

~~~text
ComfyUI/models/mediapipe/face_landmarker.task
~~~

Alternatively, place it in this node's private model directory:

~~~text
ComfyUI/custom_nodes/Skin-DeShine/models/face_landmarker.task
~~~

The `.gitignore` excludes `face_landmarker.task` and `models/*.task`.

### 4. Restart ComfyUI

Restart ComfyUI completely, then look for the four nodes under **LIN/Retouch**.

## Skin DeShine Mask

The **Skin DeShine Mask** node supports two facial landmark methods.

### mediapipe — default

MediaPipe Face Landmarker provides a dense set of face landmarks. This is the default in the canonical workflow; it does not require an external `analysis_models` input.

The node selects the largest detected face, builds a facial skin region, and excludes the eyes, eyebrows, and lips/mouth. It then contracts the face boundary and feathers mask edges to reduce accidental inclusion of hair, facial features, or background.

### insightface — optional integration

The InsightFace mode also has **no external `analysis_models` workflow input**. Skin-DeShine attempts to load `comfyui_faceanalysis.faceanalysis.InsightFace` internally from [ComfyUI_FaceAnalysis](https://github.com/cubiq/ComfyUI_FaceAnalysis).

This integration is **optional**, not a required dependency. Install and configure ComfyUI_FaceAnalysis and its InsightFace environment/models separately if you choose this method.

## Mask controls

Both **Skin DeShine Mask** and **Skin DeShine** support manual mask refinement:

| Parameter | Description |
| --- | --- |
| `manual_mask` | Optional external, hand-painted or corrected MASK. |
| `manual_mode` | `add`: include painted areas; `subtract`: exclude them; `intersect`: keep the overlap; `replace`: use the manual mask instead of the automatic mask. |
| `manual_expand` / `manual_shrink` | Expand or contract the resulting mask, in pixels. |
| `manual_blur` | Feather the mask boundary again. |
| `manual_threshold` | Binarize the mask when a hard edge is needed; `0` preserves soft edges. |
| `manual_strength` | Amount contributed by the manual mask. |
| `preview_opacity` | Opacity of the green mask overlay in the preview only; does not alter de-shining. |

**Skin DeShine Mask** outputs `skin_mask` (the automatic mask plus manual refinements) and `preview` (the input image with a green skin-mask overlay).

## Skin DeShine algorithm

**Skin DeShine** takes an `IMAGE` and a skin `MASK`. For facial retouching, exclude the eyes, eyebrows, lips, hair, clothing, and background from the mask as far as possible.

1. Convert RGB to Lab.
2. Compute low-frequency luminance at a scale derived from the detected mask extent.
3. Use relative luminance quantiles and local luminance excess inside the masked skin region to identify shine.
4. Suppress the low-frequency luminance of ordinary shiny regions.
5. Perform limited low-frequency reconstruction of very bright highlight cores using neighboring valid skin.
6. Use `color_repair` to control recovery of the Lab a/b skin color in those cores.
7. Preserve most original high-frequency detail, and blend the correction only inside the supplied mask.

The three main controls are:

| Parameter | Effect |
| --- | --- |
| `strength` | Overall de-shining strength. `0` disables the luminance reduction. |
| `core_repair` | Amount of low-frequency reconstruction applied to white highlight cores. Lower values retain more of the original highlights. |
| `color_repair` | Amount of skin chroma restored in highlight cores. Lower values reduce chroma restoration. |

The Python node's default values are:

~~~text
strength=1.0
core_repair=0.78
color_repair=0.42
~~~

### The canonical workflow intentionally uses stronger values

The included [canonical workflow](workflows/Skin-DeShine.json) stores:

~~~text
strength=1
core_repair=0.95
color_repair=1
~~~

These values are **deliberately different from the Python class defaults**. Loading the workflow should not silently reset them to `0.78 / 0.42`. Lower `core_repair` or `color_repair` if the repair looks too strong.

**Skin DeShine** outputs:

- `image` — the corrected image.
- `shine_mask` — the highlight-core detection mask; connect to `MaskPreview` to inspect it.
- `mask_preview` — the input image overlaid with the actual processing mask.

## Canonical workflow and optional iTools dependency

The canonical workflow uses these built-in ComfyUI nodes: `MaskPreview`, `MaskToImage`, `ImageCompositeMasked`, and `PreviewImage`.

It also uses **`iToolsCompareImage`** for before/after comparison. This node comes from [ComfyUI-iTools](https://github.com/MohammadAboulEla/ComfyUI-iTools) and is **optional for the example workflow**, not a dependency of the Skin-DeShine Python package.

Without ComfyUI-iTools, the Skin-DeShine nodes still install and work. However, the comparison node will be missing when loading the canonical workflow. Install ComfyUI-iTools or replace the comparison widget with your preferred image preview/comparison nodes.

> ComfyUI-iTools currently lists Image Compare as not yet supported by its Node.2 Beta implementation. For the complete comparison UI in the included workflow, use the classic node system supported by iTools.

## Optional mask-editing helpers

In addition to ComfyUI's built-in mask/composite nodes used by the canonical workflow, this package includes two helpers.

### Skin DeShine Mask Composite

This helper provides a conventional masked-composite interface:

- `destination`: the base/original portrait.
- `source`: the layer to composite, e.g. `MaskToImage` output.
- `mask`: where the source layer appears.
- `mask_opacity`: scales the mask and also affects the returned `mask`.
- `background_opacity`: controls the source layer's visibility in the composite preview only.

Outputs: `image`, `mask`.

### Skin DeShine Mask Layer

This helper provides ComfyUI's Painter interface:

- `image`: the background shown while editing.
- `auto_mask`: an optional starting mask, usually from Skin DeShine Mask.
- `edit_mode=add`: retain the automatic mask and add painted areas.
- `edit_mode=replace`: replace the automatic mask with the Painter mask.

Connect the resulting `mask` directly to `Skin DeShine.skin_mask`.

## Language

[English](README.md) | [繁體中文](README.zh-TW.md) switches the GitHub documentation. GitHub does not provide a native in-place README language button; these are links between two documents.

For ComfyUI versions that support custom-node localizations, select **Settings → Comfy → Locale → Language** to use the supplied `locales/en/nodeDefs.json` and `locales/zh-TW/nodeDefs.json` node descriptions and parameter tooltips. Changing the UI language does not change node IDs, parameter values, existing workflows, or the image-processing algorithm. Older frontends may fall back to the backend's Chinese tooltips.

## Tests

The repository includes model-free core self-checks and workflow/mapping regression checks:

~~~bash
python tests/test_deoil.py
python tests/test_skin_mask.py
python tests/test_workflow_schema.py
~~~

- `test_deoil.py`: checks shapes, value ranges, unchanged pixels outside the mask, and reduced luminance in bright areas.
- `test_skin_mask.py`: uses fake InsightFace landmarks to check skin masks, protected facial features, and mask-helper I/O.
- `test_workflow_schema.py`: checks the canonical workflow's nodes, links, and stronger repair settings, as well as custom-node mappings and the absence of the retired `analysis_models` socket.

These tests do **not** replace a live integration test with the MediaPipe model. Download `face_landmarker.task` before running the `mediapipe` path.

## License

GPL-3.0. See [`LICENSE`](LICENSE).

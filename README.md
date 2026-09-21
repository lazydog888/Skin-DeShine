# Skin-DeShine

A ComfyUI custom node for reducing unwanted skin shine and specular highlights while preserving image structure and high-frequency skin detail.

`Skin-DeShine` 以臉部皮膚遮罩限制處理範圍，在 Lab 亮度低頻上壓制相對油亮，並對極亮高光核心做有限度的低頻／膚色重建。遮罩外像素不參與修正，原始高頻紋理會盡量保留。

## Before / After example

| Before | After |
| --- | --- |
| ![Before Skin DeShine](examples/before.jpg) | ![After Skin DeShine](examples/after.jpg) |

> README 範例影像已縮放並壓縮，以便 GitHub 顯示；不影響節點處理流程或 canonical workflow。

## Nodes

安裝後可在 `LIN/Retouch` 找到：

- **Skin DeShine** (`LIN_DeOilSkin`)
- **Skin DeShine Mask** (`LIN_FaceSkinMask`)
- **Skin DeShine Mask Composite** (`LIN_SkinDeShineMaskLayer`)
- **Skin DeShine Mask Layer** (`LIN_SkinDeShineMaskEditor`)

正式範例工作流位於 [`workflows/Skin-DeShine.json`](workflows/Skin-DeShine.json)。這是本專案的 canonical workflow。

## Installation

### 1. Clone into ComfyUI `custom_nodes`

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lazydog888/Skin-DeShine.git
```

Repo root 本身就是 ComfyUI custom node package，不需要再額外建立 `lin_deoil_node` 子目錄。

### 2. Install Python dependencies

使用**啟動 ComfyUI 的同一個 Python 環境**安裝：

```bash
python -m pip install -r ComfyUI/custom_nodes/Skin-DeShine/requirements.txt
```

ComfyUI Portable（Windows）常見做法：

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\Skin-DeShine\requirements.txt
```

主安裝依賴只有 MediaPipe。PyTorch、Pillow、NumPy 等基礎套件由既有 ComfyUI 環境提供；目前 MediaPipe 1.x 也會安裝其所需的 OpenCV runtime。

### 3. Download MediaPipe Face Landmarker model

預設 `mediapipe` 遮罩算法需要 Google MediaPipe 的 `face_landmarker.task`。模型**不包含在本 repo，也不應提交到 Git**。

官方模型：

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

建議放在：

```text
ComfyUI/models/mediapipe/face_landmarker.task
```

也支援節點私有模型路徑：

```text
ComfyUI/custom_nodes/Skin-DeShine/models/face_landmarker.task
```

`.gitignore` 已排除 `face_landmarker.task` / `models/*.task`，避免模型被誤提交。

### 4. Restart ComfyUI

完全重新啟動 ComfyUI 後，確認 `LIN/Retouch` 分類中可以看到上述節點。

## Skin DeShine Mask

`Skin DeShine Mask` 有兩種臉部定位算法：

### `mediapipe` — default

使用 MediaPipe Face Landmarker 取得高密度臉部 landmarks。這是 canonical workflow 的預設路徑，不需要外接 `analysis_models`。

節點會以畫面中最大的臉為主要目標，建立臉部皮膚區域，排除雙眼、眉毛、嘴唇／口腔，再進行臉部邊界內縮與柔邊，以降低頭髮、五官與背景被納入的機率。

### `insightface` — optional integration

InsightFace 模式同樣**不需要 workflow 外接 `analysis_models`**；Skin-DeShine 會在節點內部嘗試從 [`ComfyUI_FaceAnalysis`](https://github.com/cubiq/ComfyUI_FaceAnalysis) 載入 `comfyui_faceanalysis.faceanalysis.InsightFace`。

因此這是 optional integration，不是 Skin-DeShine 主安裝依賴。若要使用此模式，必須另外安裝並正確設定 `ComfyUI_FaceAnalysis` 及其 InsightFace 環境／模型。

## Mask controls

`Skin DeShine Mask` 與 `Skin DeShine` 都支援手動 MASK 修正：

- `manual_mask`：外部手工繪製或修正過的 MASK。
- `manual_mode`：`add` / `subtract` / `intersect` / `replace`。
- `manual_expand` / `manual_shrink`：以像素擴張或收縮最終處理範圍。
- `manual_blur`：再次羽化遮罩邊緣。
- `manual_threshold`：需要硬邊時才設定二值化門檻；`0` 保留柔邊。
- `manual_strength`：控制手動 MASK 的影響量。
- `preview_opacity`：只影響遮罩疊圖預覽，不改變去油光結果。

`Skin DeShine Mask` 輸出：

- `skin_mask`：自動遮罩加手動修正後的正式 MASK。
- `preview`：原圖加綠色皮膚遮罩疊圖。

## Skin DeShine algorithm

`Skin DeShine` 輸入 `IMAGE` 與皮膚 `MASK`。遮罩應盡量排除眼睛、眉毛、嘴唇、頭髮、衣物與背景。

主要處理流程：

1. RGB 轉換至 Lab。
2. 依臉部／遮罩尺度建立亮度低頻。
3. 以皮膚區域的相對亮度 quantile 與局部亮度 excess 找出油亮區。
4. 壓制一般油光的低頻亮度。
5. 對最亮的高光核心，以周圍有效皮膚做有限度低頻 reconstruction。
6. `color_repair` 控制高光核心的 a/b 膚色補回量。
7. 保留大部分原始高頻細節，最後只在輸入 MASK 內混合修正結果。

主要參數：

- `strength`：整體去油光強度。
- `core_repair`：白色高光核心的低頻重建量。
- `color_repair`：高光核心的膚色補回量。

Python 節點本身的平衡預設值為：

```text
strength=1.0
core_repair=0.78
color_repair=0.42
```

### Canonical workflow uses a stronger setting

`workflows/Skin-DeShine.json` **刻意保存較強的修復設定**：

```text
strength=1
core_repair=0.95
color_repair=1
```

這是 canonical workflow 的既有設定，不等同於 Python class 的 default，也不應在載入 workflow 時被默默改回 `0.78 / 0.42`。若效果過強，可由使用者自行降低 `core_repair` 或 `color_repair`。

`Skin DeShine` 輸出：

- `image`：去油光結果。
- `shine_mask`：本次偵測到的高光核心，可接 `MaskPreview` 檢查。
- `mask_preview`：實際套用皮膚 MASK 的疊圖預覽。

## Canonical workflow and optional iTools dependency

Canonical workflow 使用 ComfyUI core 的：

- `MaskPreview`
- `MaskToImage`
- `ImageCompositeMasked`
- `PreviewImage`

並使用 **`iToolsCompareImage`** 做 before/after 比較。這顆比較節點來自 [`ComfyUI-iTools`](https://github.com/MohammadAboulEla/ComfyUI-iTools)，**只屬於範例 workflow 的 optional dependency，不是 Skin-DeShine Python package 的必要依賴**。

若沒有安裝 ComfyUI-iTools，Skin-DeShine custom nodes 本身仍可安裝與使用，但載入 canonical workflow 時比較節點會顯示為缺失。可安裝 ComfyUI-iTools，或自行改用其他圖片比較／Preview 節點。

> ComfyUI-iTools 目前將 Image Compare 列為 Node.2 Beta 尚未支援的節點；若要完整使用 canonical workflow 的比較 UI，請使用其支援的 classic node system。

## Optional mask editing helpers

除了 canonical workflow 使用的 ComfyUI core mask/composite 節點，本 package 也提供兩個 helper：

### Skin DeShine Mask Composite

輸入方式對應一般 masked composite：

- `destination`：底層／原始人像。
- `source`：要插入的圖層，可使用 `MaskToImage` 的輸出。
- `mask`：控制 source 顯示範圍。
- `mask_opacity`：控制 MASK 強度，並同步影響輸出的 `mask`。
- `background_opacity`：只控制 source 在合成預覽中的透明度。

輸出：`image`, `mask`。

### Skin DeShine Mask Layer

提供 ComfyUI Painter 介面：

- `image`：編輯背景。
- `auto_mask`：可接自動生成的 `skin_mask`。
- `edit_mode=add`：保留自動 MASK，再加入 Painter 新畫區域。
- `edit_mode=replace`：完全使用 Painter MASK。

輸出的正式 `mask` 可直接接回 `Skin DeShine.skin_mask`。

## Tests

不需要模型即可執行核心 self-check 與 workflow/mapping regression check：

```bash
python tests/test_deoil.py
python tests/test_skin_mask.py
python tests/test_workflow_schema.py
```

- `test_deoil.py`：檢查 shape、值域、遮罩外像素不變，以及高亮區確實被壓低。
- `test_skin_mask.py`：以 fake InsightFace landmarks 檢查皮膚遮罩、五官排除與 mask helper I/O。
- `test_workflow_schema.py`：檢查 canonical workflow 的節點／links／強修復參數，以及 custom node mappings；也會確保淘汰的 `analysis_models` socket 不會重新出現。

這些測試不取代實際 MediaPipe 模型的整合測試；正式使用 `mediapipe` 時仍需要先下載 `face_landmarker.task`。

## License

GPL-3.0. See [`LICENSE`](LICENSE).

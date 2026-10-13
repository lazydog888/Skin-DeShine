# Skin-DeShine

[English](README.md) | **繁體中文**

適用於 ComfyUI 的皮膚高光修正節點，在減少過強高光的同時保留原始皮膚紋理。它只處理臉部皮膚遮罩內的範圍，用 Lab 低頻亮度調整壓低亮部，並補回周圍膚色，修正過的地方不會發灰。

## 效果對比

| 處理前 | 處理後 |
| --- | --- |
| ![處理前](examples/before.jpg) | ![處理後](examples/after.jpg) |

## 功能

- **自然的高光修正（v2）**：降低額頭、鼻樑、臉頰的反光，補回周圍膚色，修正範圍有羽化、沒有硬邊，毛孔上的細碎反光也會一起降低，暗面紋理保留。舊演算法可用 `version = v1` 切換。
- **自動皮膚遮罩**：MediaPipe 找出臉部並保護眼睛、眉毛與嘴唇。`mediapipe_sam3` 模式再加上 SAM3，讓遮罩沿著真實皮膚邊緣一路圈到髮際線，不會停在額頭中段。
- **手動修改遮罩**：可直接在原圖上畫（`paint_on_image`），或以自動遮罩為基礎修改（`edit_auto_mask`）：任何顏色＝加入，**紅色**筆刷＝擦除。

## 安裝

在 ComfyUI 目錄下執行，並使用**啟動 ComfyUI 的同一個 Python 環境**：

```bash
git clone https://github.com/lazydog888/Skin-DeShine.git custom_nodes/Skin-DeShine
python -m pip install -r custom_nodes/Skin-DeShine/requirements.txt
```

下載 Google 官方 [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)，放到：

```text
ComfyUI/models/mediapipe/face_landmarker.task
```

也可放在 `custom_nodes/Skin-DeShine/models/`。模型檔不包含在 repo 內。安裝後請重新啟動 ComfyUI。

要使用 `mediapipe_sam3` 遮罩模式，需另外安裝提供 SAM3 的 [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)。

## 使用方式

載入[範例工作流](workflows/Skin-DeShine.json)，流程是：

**Load Image → Skin DeShine Mask → Skin DeShine Mask Layer → Skin DeShine**

- **Skin DeShine Mask** 產生皮膚遮罩（範例用 `mediapipe_sam3`）。
- **Skin DeShine Mask Layer** 在你畫之前會直接把遮罩傳下去。先執行一次，畫布會顯示原圖加綠色遮罩；一般顏色加入範圍，紅色擦除。
- **Skin DeShine** 負責降低高光。`strength` 建議從 0.5 開始，想去掉更多反光再往上調。

也可以把任何 `IMAGE` 與皮膚 `MASK` 直接接到 **Skin DeShine**。所有節點位於 `LIN/Retouch`。

## 補充說明

- 遮罩方式：**MediaPipe**（預設）、**MediaPipe + SAM3**，或 **InsightFace**（選用，需另裝 [ComfyUI_FaceAnalysis](https://github.com/cubiq/ComfyUI_FaceAnalysis)）。
- 遮罩是必要的。它除了限制只處理皮膚，也讓節點判斷「這張臉上什麼算高光」；整張圖一起處理會把眼睛、嘴唇、頭髮和背景一起壓暗。
- 側光人像的 `strength` 調太高，受光面會變平。柔光棚拍大約 0.5 最自然。
- 範例工作流使用 [ComfyUI-iTools](https://github.com/MohammadAboulEla/ComfyUI-iTools) 的 **iToolsCompareImage** 顯示前後對比，非必要。
- 本節點能降低明顯高光，但無法還原已過曝而遺失的細節。

`locales/` 提供英文及繁體中文節點說明；自我檢查程式位於 [`tests/`](tests/)。

**授權：** [GPL-3.0](LICENSE)。

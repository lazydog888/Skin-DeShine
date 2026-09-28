# Skin-DeShine

[English](README.md) | **繁體中文**

適用於 ComfyUI 的皮膚去油光節點。透過皮膚遮罩與 Lab 低頻亮度調整，降低過強高光，並對極亮區域做有限度重建，盡量保留原始皮膚紋理。

## 效果對比

| 處理前 | 處理後 |
| --- | --- |
| ![處理前](examples/before.jpg) | ![處理後](examples/after.jpg) |

## v2 更新（開發中）

- **Skin DeShine 新增 `version`（預設 `v2`）**：壓暗高光時同步補回周圍膚色，不再發灰；修正範圍羽化，沒有硬邊；毛孔上的細碎反光會一起降低，暗面紋理保留。`v1` 保留作對照。
- **遮罩新增 `mediapipe_sam3`**：沒接 `sam3_mask` 時自動執行 SAM3（comfyui-rmbg）。SAM3 圈出真實皮膚邊界（額頭到髮際線），MediaPipe 臉框往上延伸 `forehead_extend` 後限制範圍，排除脖子、耳朵與五官。
- **Mask Layer 的 `edit_mode`**：`paint_on_image`（直接在原圖上畫）或 `edit_auto_mask`（接遮罩節點的 `preview`；任何顏色＝加入、**紅色**筆刷＝擦掉自動遮罩）。舊工作流裡的 `add`／`replace` 需要重新選擇。
- 建議：`edge_blur` 設 8 左右，眉毛周圍比較不會出現交界線。

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

## 使用方式

直接載入[範例工作流](workflows/Skin-DeShine.json)，或將 `IMAGE` 和皮膚 `MASK` 接到 **Skin DeShine**。需要自動產生臉部遮罩時，使用 **Skin DeShine Mask**。

節點位於 `LIN/Retouch`，包含 **Skin DeShine**、**Skin DeShine Mask**、**Skin DeShine Mask Composite**、**Skin DeShine Mask Layer**。

## 補充說明

- **MediaPipe** 是預設遮罩算法。**InsightFace** 為選用整合，需另裝 [ComfyUI_FaceAnalysis](https://github.com/cubiq/ComfyUI_FaceAnalysis)，不需外接 `analysis_models`。
- 範例工作流使用 [ComfyUI-iTools](https://github.com/MohammadAboulEla/ComfyUI-iTools) 的 **iToolsCompareImage** 顯示前後對比；使用 Skin-DeShine 節點本身不需要安裝 iTools。
- 工作流保留較強設定：`strength=1`、`core_repair=0.95`、`color_repair=1`。Python 節點預設為 `1 / 0.78 / 0.42`。
- 本工具可降低明顯高光，但無法還原已過曝而遺失的細節。

`locales/` 提供英文及繁體中文節點說明；自我檢查程式位於 [`tests/`](tests/)。

**授權：** [GPL-3.0](LICENSE)。

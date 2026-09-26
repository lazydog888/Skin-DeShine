# Skin-DeShine

[English](README.md) | **繁體中文**

適用於 ComfyUI 的皮膚去油光節點。透過皮膚遮罩與 Lab 低頻亮度調整，降低過強高光，並對極亮區域做有限度重建，盡量保留原始皮膚紋理。

## 效果對比

| 處理前 | 處理後 |
| --- | --- |
| ![處理前](examples/before.jpg) | ![處理後](examples/after.jpg) |

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

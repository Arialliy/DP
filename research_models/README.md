# DeepPro 的三项本地模型设计

| 模型 | 构建名称 | 实现 | 设计文档 |
| --- | --- | --- | --- |
| DeepPro-LV | `lv_full` | [lv.py](lv.py) | [LV](docs/LV.md) |
| DeepPro-LVH / LVH-dec | `lvh_full` / `lvh_dec` | [lvh.py](lvh.py) | [LVH](docs/LVH.md) |
| MRTP | `mr_lv` | [mr.py](mr.py)、[multirate.py](multirate.py)、[infer_mr.py](infer_mr.py) | [MRTP](docs/MRTP.md) |

`lvh_dec` 将空间证据分成外观和新出现两条分支，使用 [stabilize.py](stabilize.py) 提供的窗口外历史背景。MRTP 在多个时间采样间隔下复用同一个网络，将各速率输出对齐回原视频帧，再进行最大值或 CFAR 融合。

## 加载模型

在 `DeepPro-main` 根目录的 PyTorch 环境中使用同一个构建入口。

```python
from research_models.models import build

lv, lv_inputs = build('lv_full', seqlen=40)
lvh, lvh_inputs = build('lvh_dec', seqlen=40)
mrtp, mrtp_inputs = build('mr_lv', seqlen=40)
```

三个模型的主输入形状是 `[B, 1, T, H, W]`，返回 `(features, logits)`，其中 `logits` 的形状是 `[B, T, H, W]`。`build()` 的第二项列出辅助输入。`valid` 标识有效帧，形状为 `[B, T]`。`pvalid` 标识配准后有效的像素，形状与主输入相同。LVH 用 `hist` 和 `hvalid` 分别接收历史背景和有效区域，形状为 `[B, 1, H, W]`。多速率推理可以使用 `infer_mr.make_scorer()` 和 `infer_mr.run_rates()`，其依赖包括 NumPy、SciPy、OpenCV 和 Pillow。

## 实验来源与数据接口

模型代码取自本地 `Deep learning and Temporal networks/official_protocol_s46/best_checkpoint/mrtp/source`，设计文档保留原实验中的路径和命令。复制范围是模型、推理代码、RGB-T 划分和设计文档。权重和实验记录位于原实验目录。

`python -m research_models.infer_mr --help` 显示原研究的 RGB-T 推理入口。它沿用 VT-Tiny-MOT 目录结构和 RGB-T 归一化。用 `VT_ROOT` 指定数据集目录，用 `HOMOGRAPHY_DIR` 指定图像配准矩阵目录。`--clips foldK/val` 需要另行提供 `research_models/configs/folds.json`。

项目根目录的 `train.py` / `test.py` 使用官方模型接口。若要在 NUDT-MIRSDT 上训练这些研究模型，需要接入相应的数据、历史背景和采样流程。原始 MRTP 的 40 帧、间隔 4 推理至少需要 160 个源帧，NUDT-MIRSDT 的 100 帧序列在该设置下会跳过这个速率。

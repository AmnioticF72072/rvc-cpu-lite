# 出处与改动

这个仓库不是从零写的。推理流程、数值细节、ONNX 图的做法都来自下面两个上游项目，
它们都是 MIT 协议。列出具体出处和改动范围，免得后来的人对不上号。

## RVC-Project/Retrieval-based-Voice-Conversion-WebUI

MIT。用到的部分：

- `SynthesizerTrnMsNSFsidM` 模型结构（`tools/export_generator.py` 直接 import 它的
  `infer/lib/infer_pack/models_onnx.py`）
- `infer/modules/vc/pipeline.py` 的处理顺序：48 Hz 高通、f0 的 Mel 量化（50~1100 Hz 映射到
  1~255 槽）、protect 混合、RMS 包络对齐、分块切点搜索、每块裁掉上下文的做法
- `infer/lib/infer_pack/attentions.py` 的相对位置注意力（定长导出的坑就是这里面来的）

改动：

- `modules.py` 里 `x0, x1 = torch.split(x, [self.half_channels] * 2, 1)` 在 torch 2.9 上会崩
  （`'tuple' object has no attribute 'split'`），改成了 `x.split(self.half_channels, dim=1)`
- `models_onnx.py` 的 `ResidualCouplingBlock.forward` 里 flow 返回 tuple 时取 `out[0]`
- 分块参数从官方的 `3 / 10 / 50 / 50` 改成 `1 / 1 / 6 / 8`，配合定长 1024 帧。原因写在 README

仓库地址：https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

## dev6699/rvc-onnx

MIT。`rvc_cpu/mel.py`、`rvc_cpu/rmvpe.py`、`rvc_cpu/vec.py` 三个模块是照着它的实现改的，
骨架和算法一致。

改动：

- `mel.py` 重写成纯 numpy，去掉了 librosa / numba / scipy 依赖。mel 滤波器组按 HTK 的
  slaney 频率刻度和 slaney 面积归一化自己算，STFT 用 `sliding_window_view + rfft` 手搓，
  对齐原版 librosa 的 center=True 反射填充。
- 48 Hz 高通从 `scipy.signal.butter(5, 48) + filtfilt` 换成 FFT 域的
  `1 / (1 + (fc/f)^(2*order))` 响应。filtfilt 是前向+反向各滤一次，等价于幅度平方，且零相位，
  所以两者数值上是一回事，省掉一个 60MB 的 scipy。
- `rmvpe.py` 的 import 路径改成包内相对导入。
- 1 秒固定的 HuBERT 输入改成了动态长度（`tools/export_contentvec.py`），
  因为原版的 HubERT ONNX 内部 Reshape 有写死的常量，换长度就报 `Invalid input shape: {1}`。

仓库地址：https://github.com/dev6699/rvc-onnx

## 运行时依赖

| 包 | 协议 |
|---|---|
| onnxruntime | MIT |
| numpy | BSD-3-Clause |
| soundfile（含 libsndfile） | BSD-3-Clause / LGPL-2.1 |
| soxr | LGPL-2.1 |

## 不在本仓库范围

模型权重。RMVPE、ContentVec、以及任何人的音色模型，都不在本仓库里，
也不在 MIT 的覆盖范围内。获取方式见 `docs/models.md`。

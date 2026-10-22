# rvc-cpu-lite

RVC 的纯 ONNX 推理实现。不装 PyTorch，不装 fairseq，不放 CUDA，只吃 CPU。
`numpy + onnxruntime + soundfile + soxr` 四个包就够，装完 140MB，代码本身几百 KB。

我拿一台没有独显的机器（i5-10400 + 核显）跑完了整个流程，
20 秒素材进、20.00 秒出来，耗时 30 秒。换 i5-12400 大概能再快三四成。

这个仓库只做推理。训练、实时变声、UVR 分轨都不在里面。

## 为什么要重写一遍

官方 WebUI 拉起来要 3~6GB 的 torch 环境，模型还得先 load 到显存或者内存里，
一台办公本装完基本没剩什么空间，第一次启动要等半分钟。

RVC 的推理其实只有三段：ContentVec 抽特征、RMVPE 抽基频、HiFiGAN 出波形。
三段都能导成 ONNX，剩下的就是切片和拼接的活儿。做完之后体积从 3.5GB 掉到 995MB，
（那 995MB 里有 830MB 是模型权重），启动 10 秒。

## 实测

i5-10400，6 核 12 线程，UHD 630，16G 内存，Windows 10：

| 环节 | 耗时 |
|---|---|
| 生成器（1024 帧一块，10.24 秒音频） | 8.2 s |
| ContentVec（9 秒音频） | 1.18 s |
| RMVPE（9 秒音频） | 0.43 s |
| 端到端 20 秒素材，RTF | 1.51 |

RTF 1.51 的意思是 1 分钟素材要等 1.5 分钟。
瓶颈几乎全在生成器，也就是 HiFiGAN 上采样那一段，占 85% 以上。

i5-12400 是我按单核性能和内存带宽推算的（约 RTF 1.0），没实测过，别当结论。

## 怎么用

```bash
pip install -r requirements.txt

python convert.py -i 素材.mp3 -o 输出.wav -m models/你的模型.onnx
```

音高不合就加 `--key`。男声转女声大概 +12，女声转男声 -12，同性别人声留 0：

```bash
python convert.py -i 素材.wav -m models/你的模型.onnx --key 12
```

Windows 上不想敲命令，双击 `转换.bat`，把文件拖进黑窗口回车就行。

模型不随仓库走（理由见 `docs/models.md`），需要先自己准备三个文件放进 `models/`：

```
models/
  your_voice.onnx      # 音色模型，用 tools/export_generator.py 从 .pth 导
  your_voice.json      # 导出时自动生成的旁挂配置（采样率、帧数）
  vec768_dyn.onnx      # ContentVec，tools/export_contentvec.py
  rmvpe.onnx           # 基频提取
```

## 两个必须知道的坑

这两个都是我在真机上撞出来的，写在这里省得别人再花两天。

**一、生成器 ONNX 只能定长导出。**

官方 `attentions.py` 的 `_relative_position_to_absolute_position()` 里有这么一行：

```python
x_flat = F.pad(x_flat, [0, int(length) - 1, 0, 0, 0, 0])
```

`int(length)` 是个 Python int，导出时拿不到符号，直接按当时的数值写成常量塞进图里。
结果就是导出的模型只认导出时那个帧数，喂别的长度必炸：

```
Reshape input_shape_size == size was false
```

报错里的形状还会伪装成 `{1,2,2002,4001}` 这种看不出毛病的数字，很容易查错方向。
`do_constant_folding=False` 也没用，常量来自 trace，不来自图优化。
所以 `tools/export_generator.py` 老老实实按定长导，推理端再按这个长度切块。

帧数选 1024（10.24 秒）。试过 2048，导出时 trace 的峰值内存把 16G 的机器直接压死。
如果你机器有 32G 以上，可以试试 1536，块数少了重叠浪费也少。

**二、分块参数不能随便调。**

`rvc_cpu/infer.py` 顶上这几个常量互相咬得很紧：

```python
X_PAD, X_QUERY, X_CENTER, X_MAX = 1, 1, 6, 8
GEN_T = 1024
```

约束是：相邻两个切点的最大跨度（`X_CENTER + 2 * X_QUERY`）加上两边的上下文（`2 * X_PAD`），
换算成帧数必须 ≤ `GEN_T`。超了就会被截断，输出会短——我第一次跑出来 20 秒素材只出了 18.86 秒，
就是这条没对上。

另外 `X_QUERY` 必须小于 `X_CENTER`，否则切点会落在段首，切出一堆几秒钟的碎片块。

## 目录

```
convert.py            命令行入口
转换.bat              Windows 拖拽入口
rvc_cpu/
  infer.py            主流水线：分块、f0、特征、protect 混合、RMS 对齐
  hw_profile.py       硬件探测，按 CPU 型号给线程数和后端
  mel.py              mel 频谱（纯 numpy，替掉 librosa）
  rmvpe.py            RMVPE 基频
  vec.py              ContentVec 封装
tools/
  export_generator.py    .pth → 定长 ONNX
  export_contentvec.py   ContentVec → 动态 ONNX
docs/
  models.md           模型从哪来
  faq.md              报错对照表
```

## 做不到的

- 训练、微调：没有。要用官方 WebUI 或者 applio。
- 实时变声：没有。这个走的是整段文件，块之间靠重叠裁剪接缝，延迟两三秒。
- m4a / aac / wma：libsndfile 不认，先转 mp3 或 wav。
- faiss 检索索引：代码留了接口（`--index`），但 faiss 得你自己装，我没打包进去。
- OpenVINO / 核显 / NPU：`hw_profile.py` 里写了分支，但这台机器没有兼容的核显可测，
  跑不到那条路径。谁有 Lunar Lake 或者 Arc 的机器试通了，欢迎发 PR。

## 许可

代码 MIT。RVC 官方仓库和 dev6699/rvc-onnx 也是 MIT，出处和改动范围写在 `NOTICE.md`。

模型权重不在许可范围内：RVC 的预训练权重、你训练出来的音色模型，各自遵守它们的来源协议。
这个仓库不附带任何权重文件。

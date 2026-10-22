## Plus 1.2（2026-10-23）：音色模型库

界面最上面多一张「音色模型」卡片，换模型不再靠手动改名覆盖：

- **导入模型…**：拖入 `.pth`（本机有 PyTorch 就自动转 ONNX，进度条可见）或
  `.onnx`（连同名 `.json`）。`.index` 可选一起放，自动启用特征检索。
- **多模型并存**：点一下切换，`models/registry.json` 记录当前模型；删除有确认，
  且不允许删掉最后一个。
- **两类模型会被明确拒绝**：① v1（256 维 hubert 底座，本版只带 768 维 ContentVec）；
  ② 定长帧数 `gen_T` 太小的 ONNX —— 分块步长会小于最小段长，每块的 3 秒上下文裁剪
  会把真声吃光（实测 12 秒进、1 秒出）。`rvc_cpu/infer.py` 加载时也有同样的自检。
- `.pth → .onnx` 转换用随包的 `tools/_rvc_export_lib`（官方 RVC 2.2 模型代码 +
  torch 2.9 兼容补丁，约 200KB），不再需要外部仓库；命令行用法见 `docs/models.md`。

代码：`rvc_cpu/models.py`（导入 / 切换 / 删除 / 注册表）。

---

## Plus 版（2026-10-23）

加了图形界面和三个新能力，推理代码与下面描述的一致。

双击 `启动界面(Plus).bat` 就能用。

1. **去和声（Karaoke）**：整首歌里叠在主唱后面的伴唱/和声，会被变声模型一起当成主唱处理，
   出来又糊又乱，参数怎么调都没用。用 UVR 的 `5_HP-Karaoke-UVR.pth` 先把伴唱剥掉再转。
   模型 127MB，放 `models/sep/`，首次使用会自动下载。
2. **智能校准**：逐帧测出原曲音高，取中位 / 70 分位 / 95 分位，自动选一个**偶数半音**，
   让「70 分位」落进模型舒适区（默认 250~320Hz），且「95 分位」不超过上限（默认 375Hz）。
   每个模型可以用 `<模型名>.pitch.json` 单独设阈值。
3. **特征检索**：`.index` 检索把音色往训练集拉，强度可调，0.4~0.6 最稳。

踩坑记录：faiss 的 IVF 索引默认没有 direct map，直接 `reconstruct_batch(id)` 会抛
`direct map not initialized`，检索会被静默跳过（音色等于没拉）。加载后必须调 `make_direct_map()`。

界面是本机 HTTP 服务 + 浏览器应用窗（Edge/Chrome 的 `--app`），这样才能用上
`backdrop-filter` 做真毛玻璃。服务只监听 `127.0.0.1`，音频不出本机。
详细用法见 `使用说明(Plus).txt`。

---

# rvc-cpu-lite

RVC 的纯 ONNX 推理实现。不装 PyTorch，不装 fairseq，不放 CUDA，只吃 CPU。
`numpy + onnxruntime + soundfile + soxr` 四个包就够，装完 140MB，代码本身几百 KB。

我拿一台没有独显的机器（i5-10400 + 核显）跑完了整个流程，
20 秒素材进、20.00 秒出来，耗时 30 秒。换 i5-12400 大概能再快三四成。

这个仓库做推理，外加一个**可选**的分离预处理（UVR Karaoke，用来剥和声）。训练、实时变声不做。

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

## 跑得动的最低配置

八代 i5、8G 内存、没有独显的办公本就能跑，这是下限。

| | 要求 | 说明 |
|---|---|---|
| CPU | x64，4 核起步，6 核最舒服 | 需要 AVX2。2013 年以后的 Intel（Haswell 起）、2017 年以后的 AMD Zen 都有；老赛扬、Atom 这类缺指令集的会直接起不来 |
| 内存 | 8 GB 能跑，16 GB 舒服 | 峰值 2~3 GB：830MB 权重 + 激活值 + 采样缓冲 |
| 显卡 | 不需要，核显都不用 | 纯 CPU 路径 |
| 硬盘 | 1.2 GB 可用 | 代码 + 三个模型约 1 GB，剩下的给输出 wav（4 分钟 48k 无损约 40MB） |
| 系统 | Windows 10/11 x64 实测 | 源码跑的话 Linux 和 macOS 也该行，我没试 |

内存这一项是硬约束。8G 机器上同时开着游戏和浏览器，转换会掉进换页，时间翻倍还不止。

按 RTF 1.51 算，一首歌的实际耗时大致是：

| 歌长 | 6 核 i5 | 12 代 i5 |
|---|---|---|
| 3 分钟 | 4.5 分钟 | 3 分钟 |
| 4 分钟 | 6 分钟 | 4 分钟 |
| 5 分钟 | 7.5 分钟 | 5 分钟 |

RTF 基本是线性的，长音频不会额外变慢。

训练就不行了。v2 常规是 200 epoch、10~15 分钟素材，3060 上要 2~4 小时，
CPU 慢 50~100 倍而且内存吃不消。训练放 GPU 上做一次，导出 ONNX 之后这台机器只负责推理。

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
换算成帧数必须 ≤ `GEN_T`。超了就会被截断，输出会比输入短。我第一次跑 20 秒素材只出了 18.86 秒，
就是这条没对上，对着代码看了半下午。

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

# rvc-cpu-lite

RVC 的纯 ONNX 推理实现。不装 PyTorch，不装 fairseq，不放 CUDA，只吃 CPU。

`numpy + onnxruntime + soundfile + soxr` 四个包就够。24 秒素材进、24 秒出来，
一台没有独显的办公本（i5-10400 + 核显）跑完整个流程。

**两种用法：**

| 玩法 | 输入 | 适合 |
|---|---|---|
| **翻唱 / 变声** | 已有音频（说话、唱歌都行） | 换音色、做翻唱 |
| **朗读** | 一段文字 | 配音、念旁白、视频画外音 |

两条路共用同一个模型、同一条管线、同一套质检。区别只在「东西从哪来」。

```
v1.x   音频 ──────> 变声 ──────> 成品
v2.0   音频 ──┐
       文字 ──┴──> 变声 ──────> 成品
```

---

## 为什么重写一遍

官方 WebUI 拉起来要 3~6GB 的 torch 环境，模型还得先 load 到显存或内存里，
一台办公本装完基本没剩什么空间，第一次启动要等半分钟。

RVC 的推理其实只有三段：ContentVec 抽特征、RMVPE 抽基频、HiFiGAN 出波形。
三段都能导成 ONNX，剩下的就是切片和拼接的活儿。做完之后体积从 3.5GB 掉到
**916MB**（其中 830MB 是模型权重），启动 10 秒。

---

## 特点

**纯 CPU，不需要显卡**
核显都不用。八代 i5 + 8G 内存是下限。

**自动定调，不用懂乐理**
逐帧测音高，取中位 / 70 / 95 分位，自动选偶数半音，让 70 分位落进模型舒适区
（默认 250~320Hz），且 95 分位不超上限（默认 375Hz）。每个模型可用
`<模型名>.pitch.json` 单独设阈值。

**去和声**
整首歌里叠在主唱后面的伴唱，会被模型当成主唱一起处理，出来又糊又乱，参数怎么调都没用。
用 UVR 的 `5_HP-Karaoke-UVR.pth` 先把伴唱剥掉再转。模型 127MB，放 `models/sep/`，首次自动下载。

**特征检索**
`.index` 把音色往训练集拉，强度可调，0.4~0.6 最稳。

**转换后质检**
跑完给 0~100 分，列出谐波度 / 低频残留 / 电平稳定性 / 峰值，外加具体建议
（比如"谐波度偏低，检查是不是没去干净和声"）。

**朗读带情绪**
7 种情绪（平静 / 温柔 / 欢快 / 激动 / 悲伤 / 严肃 / 耳语），11 个载体语音，
按模型音域自动选男女声。

**模型库**
导入 / 切换 / 删除，支持 `.pth`（本机有 PyTorch 就自动转 ONNX）、`.onnx`、`.index`。

---

## 快速开始

### 1. 装依赖

```bash
pip install -r requirements.txt
```

四个包，装完约 140MB。

### 2. 准备三个模型文件

模型不随仓库走（理由见 [`docs/models.md`](docs/models.md)），需要自己准备放进 `models/`：

```
models/
  your_voice.onnx      # 音色模型，用 tools/export_generator.py 从 .pth 导
  your_voice.json      # 导出时自动生成的旁挂配置（采样率、帧数）
  vec768_dyn.onnx      # ContentVec，tools/export_contentvec.py
  rmvpe.onnx           # 基频提取
```

`vec768_dyn.onnx` 和 `rmvpe.onnx` 是**底座模型**，跟音色无关，所有音色模型共用：

- `rmvpe.onnx` —— 来自 [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI)，
  用 `tools/export_rmvpe.py` 转成 ONNX
- `vec768_dyn.onnx` —— ContentVec，来自 [lengyue233/content-vec-best](https://huggingface.co/lengyue233/content-vec-best)，
  用 `tools/export_contentvec.py` 转成 ONNX

### 3. 跑

```bash
# 命令行
python convert.py -i 素材.mp3 -o 输出.wav -m models/你的模型.onnx

# 或者图形界面（Windows）
双击 启动界面(Plus).bat
```

音高不合就加 `--key`。男声转女声大概 +12，女声转男声 -12，同性别人留 0：

```bash
python convert.py -i 素材.wav -m models/你的模型.onnx --key 12
```

Windows 上不想敲命令，双击 `转换.bat`，把文件拖进黑窗口回车。

---

## 实测

i5-10400，6 核 12 线程，UHD 630，16G 内存，Windows 10：

| 环节 | 耗时 | 占端到端 |
|---|---|---|
| 生成器 HiFiGAN（1024 帧 = 10.24 秒音频） | 8.2 s | **73~79%** |
| ContentVec（9 秒音频） | 1.18 s | ~10% |
| RMVPE（9 秒音频） | 0.43 s | ~5% |
| 端到端 20 秒素材 | 30 s | RTF 1.51 |

**瓶颈几乎全在生成器**（HiFiGAN 上采样）。想优化别的部分都是白费力气——下面是实测过的。

### 优化过、有结论的东西

| 改动 | 结果 | 采用 |
|---|---|---|
| ORT 图优化 `DISABLE_ALL` → `ENABLE_BASIC` | 推理快 **1.15x**，加载快 27%，单句延迟 23.5s → 9.7s | 是 |
| 线程数 4 / 6 / 8 / 12 | 6 最快；8 慢 33%、12 慢 47%（超线程反效果） | 维持 6 |
| INT8 量化生成器 MatMul/Gemm | 孤立测 1.29x / SNR 38.6dB，**端到端 0.98x** | **否** |

INT8 为什么没用：反量化开销吃掉收益，量化模型多占 122MB 内存，而 i5-10400 是
Comet Lake，**没有 AVX-512 VNNI**（11 代才引入），INT8 只能走 AVX2 模拟路径。
换到支持 VNNI 的机器可以重新评估，脚本在 `tools/quantize_gen.py`。

> 教训：孤立微基准会骗人。任何优化都必须端到端测，而且同一进程交替测。

---

## 跑得动的最低配置

八代 i5、8G 内存、没有独显的办公本就能跑，这是下限。

| | 要求 | 说明 |
|---|---|---|
| CPU | x64，4 核起步，6 核最舒服 | 需要 AVX2。2013 年后的 Intel（Haswell 起）、2017 年后的 AMD Zen 都有；老赛扬、Atom 这类缺指令集的会直接起不来 |
| 内存 | 8 GB 能跑，16 GB 舒服 | 峰值 2~3 GB：830MB 权重 + 激活值 + 采样缓冲 |
| 显卡 | 不需要，核显都不用 | 纯 CPU 路径 |
| 硬盘 | 1.3 GB 可用 | 代码 + 三个模型约 1 GB，剩下的给输出 wav（4 分钟 48k 无损约 40MB） |
| 系统 | Windows 10/11 x64 实测 | 源码跑的话 Linux 和 macOS 该也行，没试过 |

内存是硬约束。8G 机器上同时开着游戏和浏览器，转换会掉进换页，时间翻倍还不止。

按 RTF 1.5 算，一首歌大致耗时：

| 歌长 | 6 核 i5 | 12 代 i5 |
|---|---|---|
| 3 分钟 | 4.5 分钟 | 3 分钟 |
| 4 分钟 | 6 分钟 | 4 分钟 |
| 5 分钟 | 7.5 分钟 | 5 分钟 |

RTF 基本线性，长音频不会额外变慢。

---

## 目录

```
convert.py            命令行入口
转换.bat              Windows 拖拽入口
gui.py                图形界面 + HTTP 路由（薄入口）
app/
  core.py             状态与模型生命周期
  reading.py          朗读任务
rvc_cpu/
  infer.py            主流水线：分块、f0、特征、protect 混合、RMS 对齐
  audio_io.py         音频读取与校验（坏文件在这里拦住并翻译成人话）
  models.py           音色模型库：导入 / 切换 / 删除 / 注册表
  tts.py              文字转声音（载体、情绪、切句）
  read.py             朗读编排（载体 → 定调 → 变声 → 质检）
  ultra_lite.py       分段音高曲线 / 转换后质检 / 频谱匹配 EQ
  calibrate.py        自动定档：测音高、算偶数半音移调
  separate.py         分离调度（去和声 / 分人声，交给带 torch 的子进程）
  hw_profile.py       硬件探测，按 CPU 型号给线程数和后端
  mel.py              mel 频谱（纯 numpy，替掉 librosa）
  rmvpe.py            RMVPE 基频
  vec.py              ContentVec 封装
tools/
  export_generator.py    .pth → 定长 ONNX
  export_contentvec.py   ContentVec → 动态 ONNX
  quantize_gen.py        INT8 量化（实测无收益，留作换机器再评估）
  _rvc_export_lib/       .pth 转换用的官方 RVC 2.2 模型代码
docs/
  models.md           模型从哪来、怎么导入
  faq.md              报错对照表
  PITFALLS.md         踩坑记录
  RECIPES.md          参数从哪抄的
  v2.0-说明.md        2.0 的完整设计与实测数据
```

---

## 两个必须知道的坑

这两个都是真机上撞出来的，写在这里省得别人再花两天。

**一、生成器 ONNX 只能定长导出。**

官方 `attentions.py` 的 `_relative_position_to_absolute_position()` 里有这么一行：

```python
x_flat = F.pad(x_flat, [0, int(length) - 1, 0, 0, 0, 0])
```

`int(length)` 是 Python int，导出时拿不到符号，直接按当时数值写成常量塞进图里。
结果就是导出的模型只认导出时那个帧数，喂别的长度必炸：

```
Reshape input_shape_size == size was false
```

报错里的形状还会伪装成 `{1,2,2002,4001}` 这种看不出毛病的数字，很容易查错方向。
`do_constant_folding=False` 也没用，常量来自 trace，不来自图优化。
所以 `tools/export_generator.py` 老老实实按定长导，推理端再按这个长度切块。

帧数选 1024（10.24 秒）。试过 2048，导出时 trace 的峰值内存把 16G 机器直接压死。
有 32G 以上可以试 1536，块数少了重叠浪费也少。

**二、分块参数不能随便调。**

`rvc_cpu/infer.py` 顶上这几个常量互相咬得很紧：

```python
X_PAD, X_QUERY, X_CENTER, X_MAX = 1, 1, 6, 8
GEN_T = 1024
```

约束是：相邻两个切点的最大跨度（`X_CENTER + 2 * X_QUERY`）加上两边上下文（`2 * X_PAD`），
换算成帧数必须 ≤ `GEN_T`。超了会被截断，输出比输入短。第一次跑 20 秒素材只出了 18.86 秒，
就是这条没对上。

另外 `X_QUERY` 必须小于 `X_CENTER`，否则切点落在段首，切出一堆几秒钟的碎片块。

---

## 版本历史

### Plus 2.0.1（2026-10-23）：修三个真 bug

都是「一句英文报错看不出所以然」的问题，查下来是三个缺陷叠在一起。

- **损坏音频静默放行**：`sf.info()` 只解析文件头，而损坏的 mp3 头是好的、数据是坏的。
  坏文件顺利进队列，等到转换才炸成 `Unspecified internal error`（libsndfile 原文）。
  现在用 `audio_io.probe(deep=True)` 流式解码整段校验，并把错误翻成能照做的中文。
- **音频能冒充模型**：以前只检查扩展名。现在魔数嗅探 + 真建 session 验证；
  已有坏文件在列表里标「不可用」并禁止选择，`active` 也跳过它们。
- **切模型可能不生效**：`ensure_pipeline` 只判设备不判模型，一旦漏了 `invalidate()`
  就继续用旧模型，而界面已显示新模型。现在模型名也纳入缓存判断。

### Plus 2.0（2026-10-23）：加朗读，架构重构

- **朗读模式**：文字 → 载体 → 自动定调 → 变声。7 情绪 + 11 载体语音。
  关键实测：载体中位 116Hz 配 213Hz 男声模型，`+0` 移调输出仅 97Hz / 质量 65 分，
  `+10` 移调输出 136Hz / 质量 86 分。**朗读必须自动定调**，照搬翻唱的 0 移调会发闷发虚。
- **架构拆分**：gui.py 1454 行按职责拆到 `app/` + `rvc_cpu/`，
  迁移用「保留同名引用」，原 15 个路由一行没改，回归 21/21 直接过。
- **算子优化**：图优化 1.15x 采用；INT8 实测无收益弃用（见上文）。
- 详见 [`docs/v2.0-说明.md`](docs/v2.0-说明.md)

### Plus 1.3（2026-10-23）：把质检接进界面

- 转换后自动质检（0~100 分 + 四项指标 + 建议）
- 分段音高条（10 秒一段，超 375Hz 标橙、超 450Hz 标红）
- 队列单项移除、参数记忆、预计剩余时间、A/B 试听保持位置
- 修复：「分析音高」按钮不解锁、换模型后旧分析不失效、忙时上传顶掉当前文件

### Plus 1.2（2026-10-23）：音色模型库

- 导入 / 切换 / 删除模型，`.pth` 自动转 ONNX
- 两类模型被明确拒绝：v1（256 维 hubert）、非 v2 生成器

### Plus 1.1（2026-10-22）：自动校准

- 逐帧测音高，自动算偶数半音移调
- 去和声（UVR Karaoke）、特征检索（faiss）

### 1.0（2026-10-22）

- 纯 ONNX 推理，分块拼接，输出长度与输入严格对齐
- 硬件自动识别

---

## 做不到的

- **训练、微调**：没有。要用官方 WebUI 或 applio。v2 常规是 200 epoch，
  3060 上要 2~4 小时，CPU 慢 50~100 倍且内存吃不消。训练在 GPU 上做一次，
  导出 ONNX 之后这台机器只负责推理。
- **麦克风级实时变声**：没有。实测 RTF 1.96~4.87（<1 才叫实时），物理上做不到。
  做到的是「按句流水线」，体感是「写完一段，等几秒出音」。
- **m4a / aac / wma**：libsndfile 不认，先转 mp3 或 wav。
- **faiss 检索索引**：`pip install -r requirements.txt` 会装 faiss-cpu，
  但 `.index` 文件得你自己有——那是训练时生成的，仓库不附带。
- **OpenVINO / 核显 / NPU**：`hw_profile.py` 里有分支，但手上没有兼容的核显可测，
  跑不到那条路径。有 Lunar Lake 或 Arc 的机器试通了欢迎发 PR。
- **内置二次元角色模型**：没有，也不会内置。RVC 角色模型都是用真人声优声音训练的，
  打包分发属于侵权。自己训练或从合法来源获取后导入。

---

## 许可

代码 MIT。RVC 官方仓库和 dev6699/rvc-onnx 也是 MIT，出处和改动范围写在 `NOTICE.md`。

模型权重不在许可范围内：RVC 预训练权重、你训练的音色模型，各自遵守来源协议。
这个仓库不附带任何权重文件。

---

## 相关记录

- [踩坑记录](docs/PITFALLS.md) —— 定长导出、分块参数、faiss direct map 等
- [报错对照表](docs/faq.md) —— 按报错信息排
- [参数来源](docs/RECIPES.md) —— 参数从哪抄的
- [2.0 设计与实测](docs/v2.0-说明.md) —— 含失败的结论

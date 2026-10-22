# 模型从哪来

仓库不带权重。三个文件各自由来不同，这里写清楚每个怎么弄出来。

放进 `models/` 目录，文件名建议就用下面这些（`convert.py` 默认按这几个名字找）：

| 文件 | 大小 | 哪来 |
|---|---|---|
| `vec768_dyn.onnx` | ~360 MB | ContentVec 768 维，自己导 |
| `rmvpe.onnx` | ~345 MB | 基频提取，导或现成 |
| `your_voice.onnx` + `.json` | ~120 MB | 你要用的音色，从 .pth 导 |

## 一、音色模型（your_voice.onnx）

需要原始的 `.pth`。自己训的就在 `logs/<实验名>/` 下面，网上找的公开模型通常也提供 pth。
然后在一个装了 torch 的环境里（官方 WebUI 或 applio 的 venv 都行）：

```bash
python tools/export_generator.py \
  --pth E:/rvc/logs/myvoice/myvoice_16236.pth \
  --out models/your_voice.onnx \
  --rvc E:/rvc/Retrieval-based-Voice-Conversion-WebUI-2.2.231006 \
  --T 1024
```

`--rvc` 指向官方仓库根目录，脚本要 import 里面的 `models_onnx.py`。
如果直接跑报 `torch.split` 相关的错，是官方 `modules.py` 和 torch 2.9 不兼容，
把这两处改掉就行（改动内容在 `NOTICE.md` 里）：

`infer/lib/infer_pack/modules.py`
```python
# 原： x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
x0, x1 = x.split(self.half_channels, dim=1)
```

`infer/lib/infer_pack/models_onnx.py`，`ResidualCouplingBlock.forward` 的 reverse 分支
```python
out = flow(x, x_mask, g=g, reverse=reverse)
x = out[0] if isinstance(out, tuple) else out
```

导完会在 `--out` 旁边生成一个同名 `.json`，里面记了采样率、帧数、版本号。
这个 json 别删，推理端靠它知道输出是 40k 还是 48k，比去读 ONNX protobuf 快得多。

**注意 `--T`。** 模型是按这个帧数定长导出的，推理端 `GEN_T` 必须跟它一致。
改了一个不改另一个，会报 `Reshape ... size was false`。

## 二、ContentVec（vec768_dyn.onnx）

要 transformers 格式的 ContentVec 目录（`config.json` + 权重文件，768 维）。
applio 里一般躺在 `rvc/models/embedders/contentvec/`。

```bash
python tools/export_contentvec.py \
  --emb E:/applio/rvc/models/embedders/contentvec \
  --out models/vec768_dyn.onnx
```

脚本会用 16000 和 320000 两种样本数各跑一遍验证，输出帧数应该是 50 帧/秒（样本数 ÷ 320，
误差一两帧）。

如果你手上有的是 `.pt` 形式的 hubert/ContentVec，得先转成 transformers 格式，
或者用官方的 `hubert_base.pt` 加 RVC 自己的加载代码包一层，再走同样的 `torch.onnx.export`。

## 三、RMVPE（rmvpe.onnx）

两条路：

1. 用 dev6699/rvc-onnx 里的导出脚本，把官方 `rmvpe.pt` 转成 ONNX。
   官方权重在 HuggingFace 的 `lj1995/VoiceConversionWebUI` 下面（`rmvpe.pt`）。
2. 直接用那个仓库已经转好的 `rmvpe.onnx`（注意看它的输入输出名字，本仓库按单输入单输出写的）。

不管哪条路，推理时 RMVPE 固定跑 CPU。原版在 OpenVINO 的 GPU/NPU 上算 f0 会出错误的值，
这个坑上游已经踩过，`hw_profile.py` 里也写死了不让它上核显。

## 别把别人的音色模型传上来

音色模型（尤其是自己或别人声音训出来的）属于生物特征数据，传公共仓库有隐私和法律问题。
`.gitignore` 已经挡了 `models/*.onnx`，提 PR 的时候也留意别用 `git add -f` 绕过。

想在别人的机器上验证整个流程，用公开授权的模型，或者自己录一段干净人声现训一个。

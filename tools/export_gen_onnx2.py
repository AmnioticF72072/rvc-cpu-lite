#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 applio .pth 生成器导出为 ONNX（官方 models_onnx 类，torch 环境跑一次）

借鉴 dev6699/rvc-onnx 的 export.py（MIT），模型类用 RVC 官方 2.2 的
SynthesizerTrnMsNSFsidM（标准 LayerNorm，torch 导出器兼容）。

用法:
  G:/AIVoice/env/rvc/Scripts/python.exe export_gen_onnx2.py \
      --pth .../myvoice51_16236.pth --out ../models/myvoice51.onnx [--fp16]
"""
import argparse
import json
import os
import sys

import torch
torch.set_num_threads(3)   # 本机内存/CPU 紧张：限制线程，防止把整机压死

# 官方 2.2 老版仓库位置：优先用随包 vendor 的 _rvc_export_lib（便携包自带，
# 不依赖外部仓库）；其次是本地解压的完整仓库（开发机调试用）。
_HERE = os.path.dirname(os.path.abspath(__file__))
VENDORED = os.path.join(_HERE, "_rvc_export_lib")
UPSTREAM = VENDORED if os.path.isdir(os.path.join(VENDORED, "infer")) else \
    os.path.join(_HERE, "..", "Retrieval-based-Voice-Conversion-WebUI-2.2.231006")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true", help="导出 FP16（更小更快，CPU 上 ORT 自动转 FP32 算）")
    ap.add_argument("--T", type=int, default=1024, help="定长帧数（必须与推理端 GEN_T 一致）")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(UPSTREAM))
    from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

    cpt = torch.load(args.pth, map_location="cpu", weights_only=True)
    version = cpt.get("version", "v1")
    vec_channels = 256 if version == "v1" else 768
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    tgt_sr = cpt["config"][-1]
    print("模型: version=%s sr=%d vec=%d f0=%s" % (version, tgt_sr, vec_channels, cpt.get("f0")))

    T = args.T  # 定长导出（官方 _relative_position_to_absolute_position 的 int(length) 会
                # 在 trace 时烤成常量，动态导出仅 T==导出长度可用；见 attentions.py:327-352）
                # 1024 帧 = 10.24s：T=2048 在 16G 内存机上 trace 峰值过高会 OOM
    use_fp16 = args.fp16
    net_g = SynthesizerTrnMsNSFsidM(*cpt["config"], is_half=use_fp16, version=version)
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval()
    if use_fp16:
        net_g.half()

    test_phone = torch.rand(1, T, vec_channels)
    test_phone_lengths = torch.tensor([T]).long()
    test_pitch = torch.randint(size=(1, T), low=5, high=255)
    test_pitchf = torch.rand(1, T)
    test_ds = torch.LongTensor([0])
    test_rnd = torch.rand(1, 192, T)

    if use_fp16:
        test_phone, test_pitchf, test_rnd = test_phone.half(), test_pitchf.half(), test_rnd.half()

    torch.onnx.export(
        net_g,
        (test_phone, test_phone_lengths, test_pitch, test_pitchf, test_ds, test_rnd),
        args.out,
        # 注意：不能用 dynamic_axes —— 注意力 rel→abs 的 pad 量是 Python int，trace 即烤死，
        # 只有导出长度能跑。推理端按 2048 帧定长切片（见 infer.py）。
        do_constant_folding=False,
        external_data=True,   # 权重写外置 .data，降低 protobuf 峰值内存（本机内存紧张）
        opset_version=17,
        dynamo=False,  # torch 2.9 默认 dynamo 导出器与官方模型的 torch.split 不兼容，回 legacy
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
        output_names=["audio"],
    )

    # 元数据：不再用 onnx.load/save（115MB 模型 protobuf 反序列化要再吃 1G+ 内存，
    # 本机内存紧时会把整机压死）。改为写轻量 JSON 旁挂，推理端优先读它。
    side = os.path.splitext(args.out)[0] + ".json"
    info = {k: v for k, v in cpt.items() if k != "weight"}
    info["gen_T"] = T
    info["tgt_sr"] = tgt_sr
    info["vec_channels"] = vec_channels
    with open(side, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)
    print("旁挂配置: %s" % side)
    print("完成: %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1024**2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 RVC 的 .pth 生成器导成定长 ONNX。需要 torch 环境（只在导出时用一次）。

    python tools/export_generator.py \
        --pth path/to/model.pth \
        --out models/myvoice.onnx \
        --rvc path/to/Retrieval-based-Voice-Conversion-WebUI-2.2.231006 \
        --T 1024

为什么是定长，不是动态 shape
---------------------------
官方 attentions.py 的 _relative_position_to_absolute_position() 里这一行：

    x_flat = F.pad(x_flat, [0, int(length) - 1, 0, 0, 0, 0])

int(length) 是个 Python int，torch 导出时拿不到符号，直接把当时的数值当常量写进图里。
于是导出的 ONNX 只有「喂进去的帧数 == 导出时的帧数」才跑得通，换个长度就是：

    Reshape input_shape_size == size was false

关掉 do_constant_folding 也不行，常量来自 trace 而不是图优化。
所以这里干脆按定长导出，推理端再按这个长度切块（见 rvc_cpu/infer.py 的 GEN_T）。

T 怎么选
--------
T 越大，块数越少、重叠浪费越少；但 trace 和推理的峰值内存涨得很快。
实测 T=2048（20.5 秒/块）在 16G 内存的机器上导出时会把机器拖死，
T=1024（10.24 秒/块）稳定。默认就是 1024。
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="RVC 模型权重 .pth")
    ap.add_argument("--out", required=True, help="输出 .onnx")
    ap.add_argument("--rvc", required=True,
                    help="RVC 官方仓库根目录（需要其 infer/lib/infer_pack/models_onnx.py）")
    ap.add_argument("--T", type=int, default=1024, help="定长帧数，100 帧/秒，1024 = 10.24 秒")
    ap.add_argument("--fp16", action="store_true", help="导出 FP16（CPU 上没收益，只是文件小一半）")
    args = ap.parse_args()

    import torch

    sys.path.insert(0, os.path.abspath(args.rvc))
    from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

    cpt = torch.load(args.pth, map_location="cpu", weights_only=True)
    version = cpt.get("version", "v1")
    vec_channels = 256 if version == "v1" else 768
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    tgt_sr = cpt["config"][-1]
    print("模型: version=%s sr=%d vec=%d f0=%s" % (version, tgt_sr, vec_channels, cpt.get("f0")))

    T = args.T
    net_g = SynthesizerTrnMsNSFsidM(*cpt["config"], is_half=args.fp16, version=version)
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval()
    if args.fp16:
        net_g.half()

    phone = torch.rand(1, T, vec_channels)
    phone_lengths = torch.tensor([T]).long()
    pitch = torch.randint(size=(1, T), low=5, high=255)
    pitchf = torch.rand(1, T)
    ds = torch.LongTensor([0])
    rnd = torch.rand(1, 192, T)
    if args.fp16:
        phone, pitchf, rnd = phone.half(), pitchf.half(), rnd.half()

    torch.onnx.export(
        net_g,
        (phone, phone_lengths, pitch, pitchf, ds, rnd),
        args.out,
        do_constant_folding=False,
        opset_version=17,
        # torch 2.9 的 dynamo 导出器和官方模型的 torch.split 不兼容，必须回 legacy tracer
        dynamo=False,
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
        output_names=["audio"],
    )

    # 采样率和帧数写进旁挂 JSON。不用 onnx.load/save 回写元数据：
    # 100MB+ 的模型反序列化要再吃 1G 内存，小内存机器上很容易被压死。
    side = os.path.splitext(args.out)[0] + ".json"
    info = {k: v for k, v in cpt.items() if k != "weight"}
    info["gen_T"] = T
    info["tgt_sr"] = tgt_sr
    info["vec_channels"] = vec_channels
    with open(side, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)
    print("旁挂配置: %s" % side)
    print("完成: %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1024 ** 2))


if __name__ == "__main__":
    main()

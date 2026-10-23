# -*- coding: utf-8 -*-
"""把生成器量化成 INT8，放进 models/quant/ 供推理时自动启用。

只量化 MatMul / Gemm：实测 1.29x、SNR 38.6 dB（几乎无损）。
Conv 类的动态量化在 ORT CPU EP 上不支持（报 initializer 错误），不动。

跑完会在 models/quant/ 下生成 <模型名>.int8.onnx，
infer.py 启动时若发现同名文件存在就优先加载（不存在则用 FP32，行为不变）。
"""
import os

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = r"G:\AIVoice\rvc-cpu-portable"
MODELS = os.path.join(ROOT, "models")
OUTDIR = os.path.join(MODELS, "quant")

TARGETS = ["myvoice51"]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for name in TARGETS:
        src = os.path.join(MODELS, name + ".onnx")
        if not os.path.isfile(src):
            print("跳过（不存在）:", src)
            continue
        dst = os.path.join(OUTDIR, name + ".int8.onnx")
        print("量化 %s (%.0f MB) → %s"
              % (name, os.path.getsize(src) / 1048576, os.path.basename(dst)))
        quantize_dynamic(
            src, dst,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=True,
            reduce_range=False,
        )
        m = onnx.load(dst)
        onnx.checker.check_model(m)
        print("  完成 %.0f MB，校验通过"
              % (os.path.getsize(dst) / 1048576))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ContentVec / HuBERT 导成动态长度的 ONNX。需要 torch + transformers。

    python tools/export_contentvec.py \
        --emb path/to/contentvec_dir \
        --out models/vec768_dyn.onnx

--emb 是 transformers 格式的目录（config.json + pytorch_model.bin，768 维的 v2 版本）。
applio 里一般在 rvc/models/embedders/contentvec/。

这个模型是真动态的：输入 wav 的维度 1 是符号 'N'，多少样本都行。
导出后会立刻用 1 秒和 20 秒两种长度跑一遍做校验。
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="transformers 格式的 ContentVec 目录")
    ap.add_argument("--out", required=True, help="输出 .onnx")
    args = ap.parse_args()

    import torch
    from transformers import HubertModel

    class Wrap(torch.nn.Module):
        def __init__(self, hubert):
            super().__init__()
            self.hubert = hubert

        def forward(self, wav):                     # (1, N) @16k
            return self.hubert(input_values=wav).last_hidden_state   # (1, T, 768)

    model = HubertModel.from_pretrained(args.emb).eval()
    wrapper = Wrap(model).eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper, (torch.rand(1, 16000),), args.out,
            input_names=["wav"], output_names=["feats"],
            dynamic_axes={"wav": {1: "N"}, "feats": {1: "T"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    print("导出完成: %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1024 ** 2))

    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print("输入:", i.name, i.shape)
    ok = True
    for length in (16000, 320000):
        out = sess.run(None, {"wav": np.random.randn(1, length).astype(np.float32)})[0]
        print("输入 %d 样本 → 输出 %s" % (length, out.shape))
        if abs(out.shape[1] - length / 320) > 2:
            ok = False
    print("动态长度校验：" + ("通过" if ok else "失败，检查 dynamic_axes"))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""命令行入口：把一个音频文件转成 ONNX 模型里的音色。

    python convert.py -i 输入.mp3 -o 输出.wav -m models/myvoice.onnx
    python convert.py -i 输入.mp3 --key -12          # 降 12 个半音

也可以不带参数直接跑，会把文件拖进来回车即可（Windows 上双击 bat 就是这条路）。
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}

DEFAULT_MODEL_DIR = os.path.join(HERE, "models")


def interactive(args):
    print("把音频文件拖到这个窗口里，回车：")
    line = sys.stdin.readline().strip().strip('"').strip()
    if not line:
        return None
    args.input = line
    return args


def main():
    ap = argparse.ArgumentParser(description="RVC CPU/ONNX 推理（无需 torch）")
    ap.add_argument("-i", "--input", help="输入音频（wav/flac/mp3/ogg/aiff）")
    ap.add_argument("-o", "--output", help="输出 wav；不给就写到 输出/<原名>_转换.wav")
    ap.add_argument("-m", "--model", help="音色模型 .onnx 路径")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录")
    ap.add_argument("--key", type=int, default=0, help="移调半音数，男转女约 +12，女转男约 -12")
    ap.add_argument("--sid", type=int, default=0, help="多说话人模型的说话人 ID")
    ap.add_argument("--index", default="", help="faiss 索引（可选，需要额外装 faiss）")
    ap.add_argument("--index-rate", type=float, default=0.0, help="索引混合比例，0~1")
    ap.add_argument("--rms-mix", type=float, default=0.25, help="响度包络混合比例")
    ap.add_argument("--protect", type=float, default=0.33, help="清音保护，0.33 是官方默认")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.input:
        args = interactive(args)
        if args is None:
            print("没有输入文件，退出。")
            return 1

    src = args.input
    if not os.path.isfile(src):
        print("找不到文件：%s" % src)
        return 1
    if os.path.splitext(src)[1].lower() not in AUDIO_EXT:
        print("[警告] 扩展名不在白名单里，仍然试一下。m4a/aac/wma 大概率失败，先转 mp3。")

    model = args.model or os.path.join(args.model_dir, "myvoice51.onnx")
    vec = os.path.join(args.model_dir, "vec768_dyn.onnx")
    rmvpe = os.path.join(args.model_dir, "rmvpe.onnx")
    missing = [p for p in (model, vec, rmvpe) if not os.path.isfile(p)]
    if missing:
        print("缺模型文件：\n  " + "\n  ".join(missing))
        print("\n见 docs/models.md，模型不进仓库，需要自己放。")
        return 1

    out = args.output
    if not out:
        out = os.path.join(HERE, "输出",
                           os.path.splitext(os.path.basename(src))[0] + "_转换.wav")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    from rvc_cpu.infer import RvcPipeline

    t0 = time.time()
    pipe = RvcPipeline(model, vec, rmvpe, index_path=args.index,
                       index_rate=args.index_rate, rms_mix_rate=args.rms_mix,
                       protect=args.protect, verbose=not args.quiet)
    if not args.quiet:
        print("\n转换中，窗口别关…")
    elapsed, dur = pipe.convert_file(src, out, f0_up_key=args.key, sid=args.sid)
    wall = time.time() - t0

    print("\n输出：%s" % out)
    print("音频 %.2fs，总耗时 %.1fs，RTF %.2f（越小越快，<1 表示快于实时）"
          % (dur, wall, wall / max(dur, 1e-6)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断。")
        sys.exit(130)

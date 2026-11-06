# -*- coding: utf-8 -*-
"""音高精度自检：合成已知频率的音频，看 RMVPE 读出来多少 Hz。

这个脚本存在的理由：2.2.2 及以前，mel 频率刻度误用了 slaney（librosa 的默认值），
而 RMVPE 训练时用的是 HTK。刻度错位不会报错，只让所有音高低 5 个半音，
于是整整三个大版本都在"安静地输出错的音高"。

判据：每个音与真值的偏差都应 ≤ 20 音分。超了就是刻度/前端又出问题了。

    python tools/check_pitch_accuracy.py
    python tools/check_pitch_accuracy.py -m models/rmvpe.onnx
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# 取 5 个音，跨男声到女声的主要音区
TONES = (110.0, 196.0, 294.0, 392.0, 523.25, 659.25)
TOLERANCE_CENTS = 20.0


def tone(freq, sec=3.0, sr=16000):
    """带二次谐波的合成音（比纯正弦更接近人声，RMVPE 更稳）。"""
    t = np.arange(int(sec * sr)) / sr
    x = 0.7 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(4 * np.pi * freq * t)
    return (0.5 * x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="RMVPE 音高精度自检")
    ap.add_argument("-m", "--rmvpe", default=os.path.join(ROOT, "models", "rmvpe.onnx"))
    ap.add_argument("-t", "--tolerance", type=float, default=TOLERANCE_CENTS,
                    help="允许的最大偏差（音分），默认 %(default)s")
    args = ap.parse_args()

    if not os.path.isfile(args.rmvpe):
        print("找不到 RMVPE 模型：%s" % args.rmvpe)
        print("用 -m 指定路径，或把 rmvpe.onnx 放进 models/。")
        return 2

    from rvc_cpu.rmvpe import RMVPE
    from rvc_cpu.mel import mel_filterbank

    # 先确认滤波器组本身对得上 librosa（不依赖 librosa，比对两者的一致性没意义，
    # 这里只报刻度参数是否确为 HTK —— 用 1000Hz 在 HTK 里的 mel 值做锚点）
    anchor = 2595.0 * np.log10(1.0 + 1000.0 / 700.0)
    from rvc_cpu.mel import hz_to_mel_htk
    if abs(float(hz_to_mel_htk(1000.0)) - anchor) > 1e-9:
        print("FAIL  mel 刻度不是 HTK（hz_to_mel_htk 与公式不符）")
        return 1
    basis = mel_filterbank(16000, 1024, 128, 30, 8000)
    if basis.shape != (128, 513):
        print("FAIL  滤波器组形状异常：%s" % (basis.shape,))
        return 1

    r = RMVPE(args.rmvpe, providers=["CPUExecutionProvider"])
    print("%-10s %-12s %-12s %s" % ("真值", "读数", "偏差", "判定"))
    worst = 0.0
    for f in TONES:
        f0 = r.infer_from_audio(tone(f), thred=0.03)
        v = f0[f0 > 0]
        med = float(np.median(v)) if v.size else 0.0
        if med <= 0:
            print("%-10.2f %-12s %-12s 没读出音高" % (f, "-", "-"))
            worst = float("inf")
            continue
        cents = 1200.0 * np.log2(med / f)
        worst = max(worst, abs(cents))
        print("%-10.2f %-12.2f %-+12.1f %s" % (f, med, cents,
                                               "OK" if abs(cents) <= args.tolerance else "偏高/偏低"))
    print()
    if worst <= args.tolerance:
        print("PASS  最大偏差 %.1f 音分（容差 %.0f）" % (worst, args.tolerance))
        return 0
    print("FAIL  最大偏差 %.1f 音分（容差 %.0f）—— 检查 mel 刻度是不是又退回 slaney 了"
          % (worst, args.tolerance))
    return 1


if __name__ == "__main__":
    sys.exit(main())

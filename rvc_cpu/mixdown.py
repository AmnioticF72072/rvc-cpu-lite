# -*- coding: utf-8 -*-
"""混音：把换好的人声和原伴奏合回去。

为什么需要这个：
    翻唱的标准流程是「**只换主唱，伴奏保留原样**」。
    v2.0.x 之前，选了「分人声」之后只输出了人声轨，伴奏压根没被用上 ——
    用户听到的是"伴奏没了 / 伴奏异常"，其实伴奏是被丢掉了。

这里做的事很朴素：
    1. 人声轨做一次轻度清理（切掉 90Hz 以下，那是贝斯和底鼓的地盘）
    2. 逐样本对齐（长度不一致时按较短裁，避免时间轴错位）
    3. 按能量比例混合人声与伴奏，并保证不削波

为什么人声要切 90Hz：
    分离出来的"人声"轨里，低频总带着伴奏残留（贝斯、底鼓的基频）。
    不切的话合轨后低频会叠成一团，听起来"糊"。
    90Hz 以下是男声基频的下限之外（男低音极限约 82Hz），切掉基本无损。
"""
from __future__ import annotations

import os

import numpy as np

# 混音默认比例：人声略高一点，因为 RVC 输出通常比原唱电平低
DEFAULT_VOCAL_DB = 0.0
DEFAULT_ACC_DB = -1.5


def _read_mono_or_stereo(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x, sr


def _resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    n = int(round(len(x) * sr_out / sr_in))
    if n <= 0:
        return np.zeros((0, x.shape[1]), np.float32)
    idx = np.linspace(0, len(x) - 1, n)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    frac = (idx - i0)[:, None]
    return (x[i0] * (1 - frac) + x[i1] * frac).astype(np.float32)


def _highpass(x, sr, cutoff=90.0):
    """零相位高通（频域实现，不做 IIR 循环——几百万样本用 Python 循环会慢到没法用）。"""
    if len(x) < 64 or sr <= 0:
        return x
    n = len(x)
    spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    # 25Hz 过渡带，避免陡峭截止带来的振铃
    gain = np.clip((freqs - cutoff * 0.6) / (cutoff * 0.4), 0.0, 1.0)
    gain = gain[:, None] if x.ndim > 1 else gain
    return np.fft.irfft(spec * gain, n, axis=0).astype(np.float32)


def _db_to_gain(db):
    return float(10.0 ** (db / 20.0))


def mix(vocal_path, accomp_path, out_path, vocal_db=DEFAULT_VOCAL_DB,
        accomp_db=DEFAULT_ACC_DB, cut_vocal_low=True):
    """人声 + 伴奏 → 成品。返回 (out_path, 统计 dict)。"""
    import soundfile as sf

    if not os.path.isfile(vocal_path):
        raise RuntimeError("找不到人声文件：%s" % os.path.basename(vocal_path))
    if not os.path.isfile(accomp_path):
        raise RuntimeError("找不到伴奏文件：%s" % os.path.basename(accomp_path))

    v, vsr = _read_mono_or_stereo(vocal_path)
    a, asr = _read_mono_or_stereo(accomp_path)

    if vsr != asr:
        a = _resample(a, asr, vsr)

    # 声道数对齐：人声是单声道、伴奏是立体声时，把人声铺到两个声道
    if v.shape[1] != a.shape[1]:
        if v.shape[1] == 1:
            v = np.repeat(v, a.shape[1], axis=1)
        elif a.shape[1] == 1:
            a = np.repeat(a, v.shape[1], axis=1)

    # 长度对齐：分离是分块做的，末尾可能差几十毫秒
    n = min(len(v), len(a))
    v, a = v[:n], a[:n]

    if cut_vocal_low:
        v = _highpass(v, vsr, 90.0)

    gv, ga = _db_to_gain(vocal_db), _db_to_gain(accomp_db)
    mixd = v * gv + a * ga

    peak = float(np.abs(mixd).max()) if mixd.size else 0.0
    clipped = peak > 0.999
    if clipped:
        # 整体降下来而不是硬削顶——削顶会产生刺耳失真，那正是"带电"的来源
        mixd = mixd / peak * 0.98

    sf.write(out_path, mixd, vsr, subtype="PCM_24")

    def rms_db(x):
        r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        return 20 * np.log10(r + 1e-12)

    stats = {
        "seconds": n / vsr,
        "sr": vsr,
        "channels": mixd.shape[1],
        "vocal_rms_db": round(rms_db(v * gv), 1),
        "acc_rms_db": round(rms_db(a * ga), 1),
        "peak": round(float(np.abs(mixd).max()), 4),
        "had_to_reduce": clipped,
    }
    return out_path, stats

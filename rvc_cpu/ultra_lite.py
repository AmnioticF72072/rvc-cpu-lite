# -*- coding: utf-8 -*-
"""Ultra Lite —— 把 Ultra 工程验证过的三个纯 numpy 能力搬进 CPU 版。

Ultra 原版（projects/不朽的星空/ultra）靠 GPU 跑多模型融合，CPU 版搬不动；
但它的「分段音高曲线」「自动质检」「频谱匹配 EQ」全是 numpy 实现，
原样搬过来就能用，而且有两处比原版更准：

1. 分段音高：Ultra 用自相关（会被倍频骗），这里直接复用 RMVPE 逐帧 f0，
   按窗口分段统计 —— 原版 dynamic_shift.py 的能量裁决都不用上了。
2. 质检：照搬 qa_ultra.py 的四个指标，加了「问题秒段定位」，
   界面上能直接看到第几秒到第几秒不行。

对外三个入口：
    segment_curve(f0, sr_f0, win_sec)   → 分段音高曲线（给深度分析用）
    qa_scan(x, sr, material_max)        → 转换后自动质检（给结果卡片用）
    match_eq_curve(ref_x, in_x, sr)     → 频谱匹配 EQ 串（firequalizer）
"""
from __future__ import annotations

import numpy as np

SR = 48000


# ---------------------------------------------------------------- 分段音高
def segment_curve(f0, sr_f0=50.0, win_sec=10.0):
    """把逐帧 f0 按窗口切段，每段给统计量。

    f0: RMVPE 输出的逐帧基频（0 = 清音），每秒 sr_f0 帧（50）。
    返回 [{t0, t1, median, p95, chorus, over}] —— over = 该段 95 分位
    变调前就超过常见模型上限（375Hz）的标记，界面上标红用。
    """
    f0 = np.asarray(f0, dtype=np.float32)
    n = f0.size
    win = max(1, int(win_sec * sr_f0))
    segs = []
    total_voice = [v for v in (f0[f0 > 0],) if v.size][0] if (f0 > 0).any() else None
    med_all = float(np.median(total_voice)) if total_voice is not None else 0.0
    for i in range(0, n, win):
        seg = f0[i:i + win]
        v = seg[seg > 0]
        t0 = round(i / sr_f0, 1)
        t1 = round(min(i + win, n) / sr_f0, 1)
        if v.size < 10:
            segs.append({"t0": t0, "t1": t1, "median": 0.0, "p95": 0.0,
                         "chorus": False, "over": False, "silence": True})
            continue
        med = float(np.median(v))
        p95 = float(np.percentile(v, 95))
        # 副歌判定沿用 Ultra 的能量思路的简化版：段中位高于全曲中位 1.15 倍
        chorus = bool(med_all > 0 and med > med_all * 1.15)
        segs.append({"t0": t0, "t1": t1,
                     "median": round(med, 1), "p95": round(p95, 1),
                     "chorus": chorus, "over": bool(p95 > 375.0),
                     "silence": False})
    return segs


# ---------------------------------------------------------------- 质检
def _ac(s):
    n = len(s)
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    S = np.fft.rfft(s, nfft)
    ac = np.fft.irfft(S * np.conj(S), nfft)[:n]
    return ac / (ac[0] + 1e-12)


def _harmonicity(x, sr=SR):
    """逐秒谐波度均值（越高音高越明确、越不刺耳）。"""
    W = sr
    vals = []
    for i in range(0, len(x) - W, W):
        s = x[i:i + W]
        if np.sqrt((s ** 2).mean()) < 3e-4:
            continue
        sp = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        fr = np.fft.rfftfreq(len(s), 1 / sr)
        tot = sp[(fr > 80) & (fr < 7900)].sum() + 1e-12
        ac = _ac(s - s.mean())
        lo, hi = int(sr / 1200), min(int(sr / 60), len(ac) - 1)
        if hi <= lo:
            continue
        seg = ac[lo:hi]
        f0 = sr / (lo + int(np.argmax(seg)))
        h = sum(sp[(fr > f0 * k * 0.965) & (fr < f0 * k * 1.035)].sum()
                for k in range(1, 9) if f0 * k < 7900)
        vals.append(min(h / tot, 1.0))
    return float(np.mean(vals)) if vals else 0.0


def _low_residual(x, sr=SR):
    """<100Hz 相对能量（dB），越低伴奏/隆隆声泄漏越少。"""
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    lo = S[(fr > 20) & (fr < 100)].sum()
    return float(10 * np.log10(lo / (S.sum() + 1e-12) + 1e-12))


def _level_consistency(x, sr=SR):
    """逐秒 RMS 的标准差（dB），越小越稳。"""
    W = sr
    n = len(x) // W
    if n < 2:
        return 0.0
    rms = np.sqrt((x[:n * W].reshape(n, W) ** 2).mean(1))
    rms = rms[rms > 1e-4]
    return float(np.std(20 * np.log10(rms + 1e-12))) if rms.size else 99.0


def _bad_spans(x, sr=SR):
    """找出「有问题」的秒段：谐波度骤降或电平突跳的连续区间。"""
    W = sr
    n = len(x) // W
    if n < 4:
        return []
    hs = np.zeros(n)
    for i in range(n):
        s = x[i * W:(i + 1) * W]
        if np.sqrt((s ** 2).mean()) < 3e-4:
            hs[i] = -1          # 静音不算问题
            continue
        sp = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        fr = np.fft.rfftfreq(len(s), 1 / sr)
        tot = sp[(fr > 80) & (fr < 7900)].sum() + 1e-12
        ac = _ac(s - s.mean())
        lo, hi = int(sr / 1200), min(int(sr / 60), len(ac) - 1)
        if hi <= lo:
            hs[i] = -1
            continue
        seg = ac[lo:hi]
        f0 = sr / (lo + int(np.argmax(seg)))
        h = sum(sp[(fr > f0 * k * 0.965) & (fr < f0 * k * 1.035)].sum()
                for k in range(1, 9) if f0 * k < 7900)
        hs[i] = min(h / tot, 1.0)
    good = hs >= 0
    if good.sum() < 2:
        return []
    med = float(np.median(hs[good]))
    bad = good & (hs < med * 0.55)      # 谐波度明显低于全曲水平的秒
    spans = []
    i = 0
    while i < n:
        if bad[i]:
            j = i
            while j + 1 < n and bad[j + 1]:
                j += 1
            if j - i + 1 >= 2:          # 至少连续 2 秒才算
                spans.append({"t0": i, "t1": j + 1,
                              "harmonicity": round(float(np.mean(hs[i:j + 1])), 3)})
            i = j + 1
        else:
            i += 1
    return spans[:6]                     # 最多报 6 段，免得刷屏


def qa_scan(x, sr, src_x=None):
    """转换结果质检。x = 输出人声（会重采样到 48k 单声道）。

    返回 {harmonicity, low_residual_db, level_std_db, peak, score, spans, tips}
    评分口径与 Ultra qa_ultra.py 一致（0~100）。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        try:
            import soxr
            x = soxr.resample(x, sr, SR, quality="HQ")
        except Exception:
            pass

    h = _harmonicity(x)
    lo = _low_residual(x)
    lc = _level_consistency(x)
    pk = float(np.abs(x).max())
    spans = _bad_spans(x)

    score = 100.0
    score -= max(0.0, (0.55 - h)) * 120       # 谐波度低于经验下限
    score -= max(0.0, (lo + 32.0)) * 1.5      # 低频残留超过 -32dB
    score -= max(0.0, (lc - 6.0)) * 5         # 电平不稳
    score -= max(0.0, (pk - 0.97)) * 200      # 削波
    score -= len(spans) * 4                    # 每个问题段扣 4 分
    score = max(0.0, min(100.0, score))

    tips = []
    if h < 0.45:
        tips.append("谐波度偏低：人声可能发糊或有电音，检查是不是没去干净和声")
    if lo > -26.0:
        tips.append("低频残留偏多：伴奏或隆隆声漏进来了，试试「去和声」或把音色接近度调低一点")
    if lc > 8.0:
        tips.append("电平忽大忽小：可能混进了不同响度的段落")
    if pk > 0.97:
        tips.append("峰值接近削波：输出可能会爆音")
    if spans:
        tips.append("第 %s 秒附近有明显质量下降" % "、".join(
            "%d~%d" % (s["t0"], s["t1"]) for s in spans[:3]))

    return {"harmonicity": round(h, 3),
            "low_residual_db": round(lo, 1),
            "level_std_db": round(lc, 1),
            "peak": round(pk, 3),
            "score": int(round(score)),
            "spans": spans,
            "tips": tips}


# ---------------------------------------------------------------- 频谱匹配 EQ
FMIN, FMAX = 85.0, 16000.0
NPOINT = 48
ANCHOR = (500.0, 1600.0)


def _power_spectrum(x, sr, n_fft=8192):
    hop = 512
    nf = 1 + (len(x) - n_fft) // hop
    if nf < 8:
        return None, None
    frames = np.lib.stride_tricks.sliding_window_view(
        np.pad(x, (n_fft // 2, n_fft // 2)), n_fft)[::hop][:nf]
    S = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1))
    r = np.sqrt((frames ** 2).mean(1))
    keep = r > np.percentile(r, 55)
    if keep.sum() < 4:
        keep = np.ones(nf, bool)
    P = (S[keep] ** 2).mean(0)
    fr = np.fft.rfftfreq(n_fft, 1 / sr)
    return fr, P


def _band_db(fr, P, grid, frac=1 / 6, floor_db=-95.0):
    """按 1/6 倍频程取**平均功率密度**（不是 bin 求和）。

    为什么必须取平均：1/6 倍频程在高频包含的 FFT bin 数远多于低频，
    直接求和会让高频系统性偏高（实测能差 45 dB），EQ 就完全跑偏了。
    """
    out = np.zeros(len(grid))
    for i, f in enumerate(grid):
        k = (fr >= f / 2 ** frac) & (fr < f * 2 ** frac)
        if not k.any():
            out[i] = floor_db
            continue
        p = float(P[k].mean())
        db = 10 * np.log10(max(p, 1e-18))
        out[i] = max(db, floor_db)       # 无能量区统一压到地板，避免拿残差当信号
    return out


def _norm_curve(fr, P, grid, floor_db=-95.0):
    b = _band_db(fr, P, grid, floor_db=floor_db)
    idx = (grid >= ANCHOR[0]) & (grid < ANCHOR[1])
    if idx.sum() == 0:
        idx = np.ones(len(grid), bool)
    return b - float(b[idx].mean())


def match_eq_curve(ref_x, in_x, sr, strength=0.7, limit=8.0):
    """算「原唱参考 → 模型输出」的频谱匹配 EQ，返回 firequalizer gain_entry 串。

    ref_x: 原唱人声（目标频谱）。in_x: 模型输出人声。
    strength/limit 与 Ultra adaptive_eq.py 同义，默认更保守（0.7 / 8dB），
    避免匹配过头把音色拉歪。

    关键细节：算完差值要**减去中位数**再限幅。否则当整体电平不同时
    （比如输入比参考轻 6dB），所有频点的差值都是负的，限幅后变成
    「全频段 -8dB」——那不是修音色，是把整首歌压小。
    """
    ref_x = np.asarray(ref_x, np.float32)
    in_x = np.asarray(in_x, np.float32)
    if ref_x.ndim > 1:
        ref_x = ref_x.mean(1)
    if in_x.ndim > 1:
        in_x = in_x.mean(1)
    if sr != SR:
        try:
            import soxr
            ref_x = soxr.resample(ref_x, sr, SR, quality="HQ")
            in_x = soxr.resample(in_x, sr, SR, quality="HQ")
        except Exception:
            pass

    grid = np.geomspace(FMIN, FMAX, NPOINT)
    fr_t, Pt = _power_spectrum(ref_x, SR)
    fr_i, Pi = _power_spectrum(in_x, SR)
    if Pt is None or Pi is None:
        return ""

    # 噪声地板按各自频谱峰值下浮 80 dB，两个信号用同一口径
    peak_t = 10 * np.log10(max(float(Pt.max()), 1e-18))
    peak_i = 10 * np.log10(max(float(Pi.max()), 1e-18))
    floor = max(peak_t, peak_i) - 80.0

    t = _norm_curve(fr_t, Pt, grid, floor_db=floor)
    i = _norm_curve(fr_i, Pi, grid, floor_db=floor)
    gain = (t - i) * strength
    gain -= float(np.median(gain))        # 只修形状，不改变整体电平
    gain = np.clip(gain, -limit, limit)

    k = 6
    ker = np.hanning(2 * k + 1)
    ker /= ker.sum()
    gain = np.convolve(np.pad(gain, k, mode="edge"), ker, mode="valid")
    entries = ["entry(%g,%.2f)" % (f, g) for f, g in zip(grid, gain)]
    return "firequalizer=gain_entry='%s'" % ";".join(entries)

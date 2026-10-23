# -*- coding: utf-8 -*-
"""自动定档：按输入音频的音高分布 + 模型音域画像，建议移调多少半音。

为什么要自动：移调是这个工具里唯一需要"凭感觉"的参数。移多了声音发紧、发虚，
移少了模型唱不上去会破。已经量化的做法是看三个统计量：

    中位（median）  —— 主体音区
    70 分位（p70）  —— 副歌/高音主体，定档主要看这个
    95 分位（p95）  —— 偶尔的转音、假声，决定会不会破

判据（用户实测文档 docs/PITFALLS.md 里验证过的）：
    变调后 p70 落在 [250, 320] Hz，且变调后 p95 ≤ 模型上限（默认 375 Hz）。
在这两个约束里取"离舒适区中心最近"的偶数半音。
"""
from __future__ import annotations

import json
import os

import numpy as np

# 默认音域画像：男声模型的舒适区。放了 <模型名>.pitch.json 就用文件里的覆盖。
DEFAULT_PROFILE = {
    "name": "默认（男声）",
    "target_lo": 250.0,      # 变调后 70 分位下限
    "target_hi": 320.0,      # 变调后 70 分位上限
    "p95_max": 375.0,        # 变调后 95 分位上限（超过容易紧/破）
    "step": 2,               # 只取偶数半音（奇数半音听感偏"不准"）
    "range": 12,             # 允许 ±12
}


def load_profile(models_dir, model_stem):
    """优先读 <模型>.pitch.json，没有就用默认画像。"""
    prof = dict(DEFAULT_PROFILE)
    if not models_dir or not model_stem:
        return prof
    p = os.path.join(models_dir, model_stem + ".pitch.json")
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                prof.update({k: v for k, v in json.load(f).items() if v is not None})
            prof["from_file"] = True
        except Exception:
            pass
    return prof


def stats_from_f0(f0):
    """从逐帧 f0 算统计量。全是 0（清音）的帧要剔除。"""
    f0 = np.asarray(f0, dtype=np.float32)
    v = f0[f0 > 0]
    if v.size < 10:
        return None
    return {
        "frames": int(f0.size),
        "voiced": int(v.size),
        "voiced_ratio": round(float(v.size) / max(f0.size, 1), 3),
        "median": round(float(np.median(v)), 1),
        "p70": round(float(np.percentile(v, 70)), 1),
        "p95": round(float(np.percentile(v, 95)), 1),
        "max": round(float(v.max()), 1),
    }


def suggest(stats, profile=None):
    """给出建议移调。返回 {key, reason, shifted:{...}, profile}。"""
    prof = dict(DEFAULT_PROFILE)
    if profile:
        prof.update(profile)
    if not stats:
        return {"key": 0, "reason": "没检测到人声（这段可能是纯伴奏或静音）", "profile": prof}

    lo, hi = float(prof["target_lo"]), float(prof["target_hi"])
    center = (lo + hi) / 2.0
    p95_max = float(prof["p95_max"])
    step = max(1, int(prof.get("step", 2)))
    rng = int(prof.get("range", 12))

    cands = list(range(-rng, rng + 1, step))
    if 0 not in cands:
        cands.append(0)
        cands.sort()

    best = None
    for k in cands:
        ratio = 2.0 ** (k / 12.0)
        p70 = stats["p70"] * ratio
        p95 = stats["p95"] * ratio
        # 主判据：p70 离舒适区中心的距离（用对数，听觉上才线性）
        cost = abs(np.log2(max(p70, 1.0) / center))
        # 惩罚：超出舒适区上下限
        if p70 < lo:
            cost += 0.6 * abs(np.log2(max(p70, 1.0) / lo))
        elif p70 > hi:
            cost += 0.6 * abs(np.log2(p70 / hi))
        # 硬约束：95 分位别超过模型上限
        if p95 > p95_max:
            cost += 2.0 * (p95 / p95_max - 1.0) + 1.0
        if best is None or cost < best[0]:
            best = (cost, k)

    key = best[1] if best else 0
    ratio = 2.0 ** (key / 12.0)
    shifted = {
        "median": round(stats["median"] * ratio, 1),
        "p70": round(stats["p70"] * ratio, 1),
        "p95": round(stats["p95"] * ratio, 1),
    }
    ok_low = shifted["p70"] >= lo
    ok_high = shifted["p70"] <= hi
    ok_p95 = shifted["p95"] <= p95_max

    if key == 0:
        reason = "原曲音区已经合适，不用移调"
    else:
        reason = "建议 %+d 半音：70 分位 %.0f → %.0f Hz（舒适区 %.0f~%.0f）" % (
            key, stats["p70"], shifted["p70"], lo, hi)
    if not ok_low:
        reason += "；仍略低于舒适区，但再降会更闷"
    elif not ok_high:
        reason += "；仍略高于舒适区，再升容易破"
    if not ok_p95:
        reason += "；95 分位 %.0f Hz 超出模型上限 %.0f Hz，个别高音会发紧" % (
            shifted["p95"], p95_max)

    return {"key": int(key), "reason": reason, "shifted": shifted, "profile": prof}


def analyze_file(rmvpe_path, wav_path, progress=None, return_f0=False):
    """独立分析一个音频文件的音高分布（只加载 RMVPE，比加载整套快得多）。

    return_f0=True 时返回 (stats, f0逐帧数组)，给 Ultra Lite 的分段曲线用。
    """
    import soundfile as sf
    import soxr

    from .rmvpe import RMVPE

    def _p(f, m):
        if progress:
            try:
                progress(f, m)
            except Exception:
                pass

    _p(0.1, "读取音频…")
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if sr != 16000:
        audio = soxr.resample(audio, sr, 16000, quality="HQ")
    _p(0.35, "提取音高…")
    model = RMVPE(rmvpe_path)
    f0 = model.infer_from_audio(audio.astype(np.float32), thred=0.03)
    _p(0.9, "统计…")
    stats = stats_from_f0(f0)
    if return_f0:
        return stats, f0
    return stats

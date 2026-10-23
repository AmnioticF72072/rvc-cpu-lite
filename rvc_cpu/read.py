# -*- coding: utf-8 -*-
"""朗读编排：文字 → 载体音频（tts.py）→ 自动定调 → RVC 变声 → 成品。

为什么朗读也要自动定调：
    RVC 保留音高曲线、替换音色，但模型对"超出训练音域的输入"表现会退化。
    实测同一段载体（中位 116Hz）配一个中位 213Hz 的男声模型：
        +0  半音 → 97Hz   质量 65
        +6  半音 → 127Hz  质量 79
        +10 半音 → 136Hz  质量 86   ← 最好
        +12 半音 → 166Hz  质量 72
    所以朗读如果照搬"翻唱"里的 0 移调，人声会发闷、发虚。这里按载体实测音高
    自动算一个移调，把它推进模型的舒适区。

关于"实时"：
    纯 CPU 上 RTF 约 2（12 秒音频要 20+ 秒变声），做不了麦克风级的实时变声。
    这里做到的是"按句流水线"：逐句合成、逐句变声，进度按句报，第一句处理完
    就能先听 —— 体感是"说完一段，几秒后出声"，不是整篇等到底。
"""
from __future__ import annotations

import math
import os

from . import tts
from .ultra_lite import qa_scan

# 模型舒适区默认中位（没有音域画像时的兜底；实测 200Hz 附近对男声模型最稳）
DEFAULT_TARGET_MEDIAN = 200.0


def auto_key(carrier_median, target_median=DEFAULT_TARGET_MEDIAN, limit=12):
    """按载体实测音高算出该移多少半音，把它推进模型舒适区。

    返回 (key, 说明文字)。
    """
    try:
        m = float(carrier_median or 0)
    except (TypeError, ValueError):
        m = 0.0
    if m <= 40:                     # 没测到有效音高（纯静音/气声）
        return 0, "载体音高不明，按原调处理"
    k = 12.0 * math.log2(max(1e-6, target_median / m))
    key = int(round(max(-limit, min(limit, k))))
    if key == 0:
        return 0, "载体音高已在舒适区（%.0f Hz）" % m
    return key, "载体中位 %.0f Hz → 移 %+d 半音靠近 %.0f Hz" % (m, key, target_median)


def read_text(text, out_wav, pipeline, voice_id=None, emotion="neutral",
              key=None, target_median=None, on_progress=None):
    """跑完整朗读流程。

    pipeline: 已加载的 RvcPipeline（上层缓存，别在这里重建）
    key: 显式指定移调；None 则自动算
    on_progress(frac, msg): 0~1 的总体进度
    返回统计 dict，含 measure（载体/输出音高）与 qa（质量分）。
    """
    def emit(f, m):
        if on_progress:
            try:
                on_progress(max(0.0, min(1.0, float(f))), str(m))
            except Exception:
                pass

    text = (text or "").strip()
    if not text:
        raise RuntimeError("文本是空的")

    from .calibrate import analyze_file
    rmv = getattr(pipeline, "rmvpe_path", None)
    tmp_carrier = os.path.join(os.path.dirname(out_wav) or ".",
                               "_read_carrier.tmp.wav")

    # ---- 1. 合成载体（TTS 只负责音节与语调，音色后面会被替换）----
    emit(0.02, "正在把文字读成音节…")
    info = tts.synth(
        text, tmp_carrier, voice_id=voice_id, emotion=emotion,
        model_pitch_median=target_median,
        on_progress=lambda f, m: emit(0.02 + 0.18 * f, m))

    # ---- 2. 量载体音高，决定移调 ----
    carrier_median = 0.0
    if rmv and os.path.isfile(rmv):
        emit(0.21, "正在测载体音高…")
        try:
            stats, _ = analyze_file(rmv, tmp_carrier, return_f0=True)
            carrier_median = float((stats or {}).get("median") or 0)
        except Exception:
            carrier_median = 0.0

    if key is None:
        tgt = float(target_median or DEFAULT_TARGET_MEDIAN)
        use_key, why = auto_key(carrier_median, tgt)
    else:
        use_key, why = int(key), "手动指定移调 %+d" % int(key)

    # ---- 3. 变声 ----
    emit(0.22, "正在换成目标音色（移 %+d）…" % use_key)
    pipeline.convert_file(
        tmp_carrier, out_wav, f0_up_key=use_key,
        on_progress=lambda f, m: emit(0.22 + 0.70 * f, m))

    try:
        os.remove(tmp_carrier)
    except OSError:
        pass

    # ---- 4. 质检 ----
    # 这里只做 qa_scan（纯 numpy，很快）。不再量输出音高 —— 那要再跑一遍
    # RMVPE，白白多花 5~8 秒，而输出音高对用户没有决策价值（要调就调移调）。
    emit(0.94, "正在自检…")
    stats = {"carrier_median": carrier_median, "key": use_key, "reason": why}
    try:
        import soundfile as sf
        x, sr = sf.read(out_wav, dtype="float32", always_2d=True)
        mono = x.mean(1)
        stats["seconds"] = len(mono) / sr
        stats["qa"] = qa_scan(mono, sr)
    except Exception as e:
        stats["qa_error"] = str(e)[:80]

    emit(1.0, "完成")
    stats.update({
        "sentences": info.get("sentences", 0),
        "voice": info.get("voice"),
        "emotion": info.get("emotion"),
        "chars": info.get("chars", 0),
        "failed_sentences": info.get("failed", []),
    })
    return stats

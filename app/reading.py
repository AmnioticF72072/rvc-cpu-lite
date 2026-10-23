# -*- coding: utf-8 -*-
"""v2.0 朗读任务层：文字 → 载体 → 变声 → 成品。

与「翻唱」的区别只在于**输入源**：
  翻唱：已有音频 → 变声
  朗读：文字 → TTS 载体 → 变声

所以两者共用同一个管线、同一个状态机、同一套质检，只有取输入那一步不同。

**朗读的参数跟翻唱不一样，这不是随便调的**：

| 参数 | 唱歌 | 说话 | 为什么 |
|---|---|---|---|
| index_rate | 0.5 | **0.75** | 说话音色要贴，越靠训练集越像本人 |
| protect | 0.10 | **0.33** | 唱歌少保护以保住音色变化；说话要多保护，否则咬字糊、喘气变噪音 |
| rms_mix_rate | 0.40 | **0.25** | 说话音量本身平稳，跟源太多会把载体的机械起伏带进来 |

RVC 官方 pipeline 的默认 protect 就是 0.33（面向说话/朗读），
0.10 是社区为唱歌调低的。之前朗读错误地继承了唱歌的 0.10。
"""
from __future__ import annotations

import os
import traceback

from .core import OUT, WORK, ensure_pipeline, get_state, log, set_state

# 说话场景的默认参数（翻唱那边是 0.5 / 0.10 / 0.40）
READ_INDEX = 0.75
READ_PROTECT = 0.33
READ_RMS = 0.25


def run_read(text, emotion="neutral", voice_id=None, key=None,
             index_rate=None, device="auto"):
    """跑一次朗读。异常统一落到 state.error，不往上抛（后台线程里抛也没人接）。"""
    try:
        rate = READ_INDEX if index_rate is None else float(index_rate)
        set_state(stage="reading", frac=0.01, phase="朗读", error="",
                  output="", qa=None, read=None, started_at=__import__("time").time(),
                  msg="正在把文字读成音节…")
        # 说话专用参数：protect/rms 跟翻唱不同
        pipe = ensure_pipeline(rate, device, protect=READ_PROTECT,
                               rms_mix_rate=READ_RMS)

        from rvc_cpu.read import read_text
        os.makedirs(OUT, exist_ok=True)

        # 文件名用时间戳，避免覆盖
        import time as _t
        name = "朗读_%s.wav" % _t.strftime("%m%d_%H%M%S")
        out_path = os.path.join(OUT, name)

        def prog(f, m):
            set_state(frac=round(0.03 + 0.95 * float(f), 4), msg=str(m))

        stats = read_text(text, out_path, pipe, voice_id=voice_id,
                          emotion=emotion, key=key, on_progress=prog)

        dur = stats.get("seconds") or 0
        qa = stats.get("qa")
        set_state(stage="done", frac=1.0, output=out_path, phase="",
                  qa=qa, read=stats,
                  msg="朗读完成 · %s（%.1f 秒，%d 句，移 %+d）"
                      % (name, dur, stats.get("sentences", 0), stats.get("key", 0)))
        log("朗读完成：%s｜载体 %s / %s｜%s"
            % (name, (stats.get("voice") or {}).get("name", "?"),
               stats.get("emotion", "?"), stats.get("reason", "")))
        if qa:
            log("自检 %d 分%s" % (qa["score"],
                                 "｜" + qa["tips"][0] if qa.get("tips") else ""))

    except Exception as e:
        m = str(e)
        hint = ""
        if "edge-tts" in m or "朗读合成失败" in m:
            hint = "（朗读功能需要联网，检查网络或代理设置）"
        set_state(stage="error", error="%s%s" % (m, hint), msg="朗读失败",
                  frac=0)
        log("朗读失败：" + m)
        traceback.print_exc()

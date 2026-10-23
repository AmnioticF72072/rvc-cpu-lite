# -*- coding: utf-8 -*-
"""朗读：文字 → 音频（内容载体），随后交给 RVC 换成目标音色。

**这一层的音色不重要**。它的唯一任务是把文字读成"有音节、有语调、有停顿"的
音频，好让 RVC 有东西可转。最终听到的声音 100% 来自 RVC 模型。

两个关键设计：

1. **载体性别要跟模型匹配**。RVC 是"保留语调、替换音色"，但如果载体音频的
   音高跟模型音域差太远（比如女声载体去驱动一个男声模型），输出的音高曲线
   会被整体搬移，听起来发虚或破。所以这里读模型的音域画像自动选男/女载体。

2. **按句切段**。整段一次合成有两个坏处：句子间停顿生硬、出错重来成本高。
   切成句子后每段独立合成，也方便上层做「第一句先出音」的流水线。

Edge TTS 本身不支持 SSML 的 style 标签（旧版有，7.x 移除了），所以情绪靠
rate / pitch / volume 三个参数组合表达 —— 对"内容载体"来说这足够了。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# runtime 里 edge_tts 装在独立子目录，避免污染主 site-packages
for _p in (os.path.join(ROOT, "runtime", "_tts_libs"),
           os.path.join(os.path.dirname(sys.executable), "_tts_libs")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------- 载体语音
# 中文载体语音。性别标记用于跟 RVC 模型自动配对。
VOICES = [
    {"id": "zh-CN-YunxiNeural", "name": "云希", "gender": "male", "note": "年轻阳光"},
    {"id": "zh-CN-YunjianNeural", "name": "云健", "gender": "male", "note": "浑厚有劲"},
    {"id": "zh-CN-YunyangNeural", "name": "云扬", "gender": "male", "note": "播音腔"},
    {"id": "zh-CN-YunxiaNeural", "name": "云夏", "gender": "male", "note": "少年感"},
    {"id": "zh-CN-XiaoxiaoNeural", "name": "晓晓", "gender": "female", "note": "温柔"},
    {"id": "zh-CN-XiaoyiNeural", "name": "晓伊", "gender": "female", "note": "活泼"},
    {"id": "zh-CN-liaoning-XiaobeiNeural", "name": "晓北", "gender": "female", "note": "东北味"},
    {"id": "zh-CN-shaanxi-XiaoniNeural", "name": "晓妮", "gender": "female", "note": "陕西味"},
    {"id": "zh-HK-HiuGaaiNeural", "name": "曉佳", "gender": "female", "note": "粤语"},
    {"id": "zh-HK-WanLungNeural", "name": "雲龍", "gender": "male", "note": "粤语"},
    {"id": "zh-TW-HsiaoChenNeural", "name": "曉臻", "gender": "female", "note": "台湾腔"},
]

# ---------------------------------------------------------------- 情绪
# rate/pitch/volume 三元组。这是"载体"的情绪，RVC 会保留语调轮廓，
# 所以情绪变化能活下来，只是音色被换掉。
EMOTIONS = {
    "neutral":  {"label": "平静", "rate": "+0%",   "pitch": "+0Hz",  "volume": "+0%"},
    "gentle":   {"label": "温柔", "rate": "-10%",  "pitch": "-3Hz",  "volume": "-2%"},
    "cheerful": {"label": "欢快", "rate": "+14%",  "pitch": "+10Hz", "volume": "+2%"},
    "excited":  {"label": "激动", "rate": "+22%",  "pitch": "+15Hz", "volume": "+4%"},
    "sad":      {"label": "悲伤", "rate": "-18%",  "pitch": "-12Hz", "volume": "-4%"},
    "serious":  {"label": "严肃", "rate": "-8%",   "pitch": "-5Hz",  "volume": "+0%"},
    "whisper":  {"label": "耳语", "rate": "-6%",   "pitch": "-2Hz",  "volume": "-45%"},
}

_SENT_END = re.compile(r"(?<=[。！？!?；;\n])")
_COMMA = re.compile(r"(?<=[，,、：:])")


def split_text(text, max_len=90):
    """按句子切分；单句过长再按逗号切；仍超长就硬切。

    切分不是为了省事，是为了让上层可以"第一句先出音"，而不是整篇等完。
    """
    text = (text or "").strip()
    if not text:
        return []
    out = []
    for chunk in _SENT_END.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) <= max_len:
            out.append(chunk)
            continue
        # 太长：按逗号再切
        buf = ""
        for piece in _COMMA.split(chunk):
            if len(buf) + len(piece) <= max_len:
                buf += piece
            else:
                if buf:
                    out.append(buf.strip())
                # 单块仍超长：硬切
                while len(piece) > max_len:
                    out.append(piece[:max_len])
                    piece = piece[max_len:]
                buf = piece
        if buf.strip():
            out.append(buf.strip())
    return [x for x in out if x.strip()]


def pick_voice(model_pitch_median=None, prefer_gender=None):
    """选载体语音。

    model_pitch_median: RVC 模型的音高中位（Hz）。有就按它判性别，
        没有就默认男声（本项目主力模型是男声）。
    prefer_gender: "male"/"female"/None，显式指定优先。
    """
    if prefer_gender in ("male", "female"):
        for v in VOICES:
            if v["gender"] == prefer_gender and v["id"].startswith("zh-CN"):
                return v
    gender = "male"
    if model_pitch_median:
        gender = "female" if float(model_pitch_median) >= 200 else "male"
    for v in VOICES:
        if v["gender"] == gender and v["id"].startswith("zh-CN"):
            return v
    return VOICES[0]


def _voice_by_id(vid):
    for v in VOICES:
        if v["id"] == vid:
            return v
    return None


async def _synth_one(text, voice_id, rate, pitch, volume, out_path):
    import edge_tts
    comm = edge_tts.Communicate(text, voice_id, rate=rate, pitch=pitch,
                                volume=volume)
    await comm.save(out_path)


def synth(text, out_wav, voice_id=None, emotion="neutral", prefer_gender=None,
          model_pitch_median=None, on_progress=None):
    """把文字合成到 out_wav（48k 单声道 wav，直接给 RVC 用）。

    返回 {"sentences": [...], "voice": {...}, "emotion": "..."}。
    失败抛 RuntimeError（带可读原因，上层直接展示）。
    """
    try:
        import edge_tts  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "朗读功能缺少 edge-tts 组件（%s）。\n"
            "检查 runtime\\_tts_libs 目录是否存在；或改用「翻唱」模式。" % e)

    text = (text or "").strip()
    if not text:
        raise RuntimeError("文本是空的")

    emo = EMOTIONS.get(emotion) or EMOTIONS["neutral"]
    voice = _voice_by_id(voice_id) if voice_id else pick_voice(
        model_pitch_median, prefer_gender)
    sents = split_text(text)
    if not sents:
        raise RuntimeError("文本里没有可读内容")

    tmpdir = tempfile.mkdtemp(prefix="tts_")
    parts, done, failed = [], 0, []
    try:
        for i, sent in enumerate(sents):
            mp3 = os.path.join(tmpdir, "s%03d.mp3" % i)
            try:
                asyncio.run(_synth_one(sent, voice["id"], emo["rate"],
                                       emo["pitch"], emo["volume"], mp3))
                if not os.path.isfile(mp3) or os.path.getsize(mp3) < 200:
                    raise RuntimeError("合成结果是空的")
                parts.append(mp3)
            except Exception as e:
                failed.append((sent[:12], str(e)[:60]))
            done += 1
            if on_progress:
                on_progress(done / len(sents), "朗读第 %d/%d 句" % (done, len(sents)))

        if not parts:
            reason = failed[0][1] if failed else "未知原因"
            raise RuntimeError("朗读合成失败：%s\n（这个功能要联网，检查网络或代理）" % reason)

        _concat_to_wav(parts, out_wav)

        return {
            "sentences": len(parts),
            "failed": failed,
            "voice": voice,
            "emotion": emo["label"],
            "chars": len(text),
        }
    finally:
        for f in os.listdir(tmpdir):
            try:
                os.remove(os.path.join(tmpdir, f))
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def _concat_to_wav(mp3_list, out_wav, gap_ms=120):
    """把若干 mp3 拼成一个 48k 单声道 wav，句间留一点停顿。"""
    import numpy as np
    import soundfile as sf

    clip = None
    for p in mp3_list:
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(1)
        if clip is None:
            clip = (float(sr), x, sr)
        else:
            src_sr = float(sr)
            if abs(src_sr - clip[0]) > 1:
                x = _resample(x, int(src_sr), int(clip[0]))
            clip = (clip[0], np.concatenate([clip[1], x]), clip[2])

    if clip is None:
        raise RuntimeError("没有可拼接的音频")
    return _write_48k(clip[1], int(clip[0]), out_wav, gap_ms)


def _resample(x, sr_in, sr_out):
    import numpy as np
    if sr_in == sr_out:
        return x
    n = int(round(len(x) * sr_out / sr_in))
    if n <= 0:
        return np.zeros(0, np.float32)
    xp = np.linspace(0, 1, len(x), endpoint=False)
    xq = np.linspace(0, 1, n, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


def _write_48k(x, sr, out_wav, gap_ms=0):
    import numpy as np
    import soundfile as sf
    y = _resample(x, sr, 48000) if sr != 48000 else x
    if gap_ms:
        pass  # 句间停顿已在合成时体现，这里不再插空白（避免句首吞字）
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak > 0.995:
        y = y / peak * 0.98
    sf.write(out_wav, y.astype(np.float32), 48000, subtype="PCM_16")
    return out_wav


def list_voices():
    return VOICES


def list_emotions():
    return [{"id": k, "label": v["label"]} for k, v in EMOTIONS.items()]


if __name__ == "__main__":
    # 自测：直接命令行跑一段
    import time
    txt = sys.argv[1] if len(sys.argv) > 1 else "你好，这是朗读功能的自测。今天天气不错。"
    out = sys.argv[2] if len(sys.argv) > 2 else "tts_test.wav"
    t0 = time.time()
    r = synth(txt, out, emotion=sys.argv[3] if len(sys.argv) > 3 else "cheerful")
    import soundfile as sf
    d, sr = sf.read(out)
    print("OK  %.1f 秒合成  %d 句  载体=%s(%s)  情绪=%s  音频=%.2fs@%d"
          % (time.time() - t0, r["sentences"], r["voice"]["name"],
             r["voice"]["gender"], r["emotion"], len(d) / sr, sr))
    if r["failed"]:
        print("失败句：", r["failed"])

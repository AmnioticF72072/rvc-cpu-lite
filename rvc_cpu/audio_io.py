# -*- coding: utf-8 -*-
"""音频读取与校验（统一入口）。

为什么单独抽这一层：
    损坏的 mp3 有个很坑的性质 —— **文件头是好的，数据是坏的**。
    `soundfile.info()` 只解析头部，所以对损坏文件会"顺利通过"，
    直到真正解码时才抛错。v1.x 的上传校验正是只调了 info，
    于是坏文件一路进到转换阶段才炸，报的还是 libsndfile 的原文
    "Unspecified internal error" —— 用户完全不知道该怎么办。

    所以这里的原则是：
      1. 校验必须真解码（流式，内存恒定），不能只看头
      2. 报错必须说人话 + 给出能照做的下一步
"""
from __future__ import annotations

import os

# libsndfile 的报错 → 用户能照做的建议。
# 键取错误消息里的特征串，做包含匹配（不同版本措辞略有差异）。
_HINTS = [
    ("Unspecified internal error",
     "这个文件能读出开头，但内部数据是坏的（常见于下载中断、压缩包损坏、"
     "或音乐软件的加密缓存文件）。请重新获取原始文件，或用剪映/格式工厂"
     "另存为 wav 再试。"),
    ("Format not recognised",
     "识别不出这个音频格式。可能是加密格式或后缀名与实际内容不符"
     "（比如把 m4a 改名成 mp3）。请用剪映/格式工厂转成 wav 或 mp3。"),
    ("Malformed file",
     "文件结构损坏，无法解码。请重新获取原始文件。"),
    ("Unsupported encoding",
     "文件用的编码不支持（少数专业格式会这样）。转成 wav 再试。"),
    ("File does not exist",
     "找不到这个文件，可能已被移动或删除。"),
    ("System error",
     "读取文件时系统出错，可能是磁盘或权限问题。换个目录再试。"),
    ("Permission denied",
     "没有读取权限。把文件复制到桌面再试。"),
]

# 这些是音频文件的"魔数"开头，用来识别"把音频当模型导入"这类错误
_AUDIO_MAGIC = (
    (b"RIFF", "WAV"),
    (b"ID3", "MP3"),
    (b"OggS", "OGG"),
    (b"fLaC", "FLAC"),
    (b"FORM", "AIFF"),
    (b"\xff\xfb", "MP3"),
    (b"\xff\xf3", "MP3"),
    (b"\xff\xf2", "MP3"),
)


def friendly_error(exc, path=""):
    """把底层异常翻译成中文 + 可操作建议。"""
    msg = str(exc)
    for key, hint in _HINTS:
        if key.lower() in msg.lower():
            name = os.path.basename(path) if path else "这个文件"
            return "%s：%s" % (name, hint)
    # 没命中已知模式：保留原文但加一句引导
    return "%s 读取失败（%s）。如果反复失败，试试转成 wav。" % (
        os.path.basename(path) if path else "文件", msg[:120])


def sniff_audio(path):
    """看文件头是不是已知音频格式。返回格式名或 ""。"""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return ""
    for magic, name in _AUDIO_MAGIC:
        if head.startswith(magic):
            return name
    # ADTS AAC
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xF0) == 0xF0:
        return "AAC"
    return ""


def probe(path, deep=True):
    """校验音频并返回 (samplerate, channels, seconds)。

    deep=True 时**真正解码一遍**（流式读块，不占内存），能抓出"头好数据坏"
    的文件；deep=False 只读头（快，但会漏掉损坏文件，仅用于内部快速判断）。
    """
    import soundfile as sf

    if not os.path.isfile(path):
        raise RuntimeError("找不到文件：%s" % os.path.basename(path))

    try:
        info = sf.info(path)
    except Exception as e:
        raise RuntimeError(friendly_error(e, path))

    if info.samplerate <= 0 or info.frames <= 0:
        raise RuntimeError("%s 是空音频（时长 0）。" % os.path.basename(path))

    if deep:
        # 流式解码，块大小 1 秒左右 —— 内存恒定，几十 MB 的文件也不吃内存
        try:
            with sf.SoundFile(path) as f:
                block = max(4096, int(f.samplerate))
                read = 0
                while read < f.frames:
                    got = f.read(block, dtype="float32", always_2d=True)
                    if got.size == 0:
                        break
                    read += len(got)
        except Exception as e:
            raise RuntimeError(friendly_error(e, path))

    return info.samplerate, info.channels, info.frames / float(info.samplerate)


def read_audio(path):
    """读成单声道 float32（48k/原采样率的原始数据由调用方处理）。

    统一在这里翻译错误，避免每个调用点各写一套 try/except。
    """
    import soundfile as sf

    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        raise RuntimeError(friendly_error(e, path))
    if x.size == 0:
        raise RuntimeError("%s 读出来是空的。" % os.path.basename(path))
    return x.mean(1), sr


def looks_like_audio_file(path):
    """判断某个文件其实是音频（用于拦住"把 mp3 当模型导入"）。"""
    return sniff_audio(path) != ""

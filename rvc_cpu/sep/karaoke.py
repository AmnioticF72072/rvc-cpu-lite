# -*- coding: utf-8 -*-
"""去和声（Karaoke）模型探测。

去和声和"分人声"是两件事：
  · 分人声：把伴奏去掉，得到干净人声 —— demucs 就能做
  · 去和声：保留主唱、剥掉主唱后面的伴唱 —— 需要专门训练的 UVR 模型

UVR 模型都是 .pth，必须走 audio-separator（依赖 torch）。所以这里只负责
"找模型"，真正跑在 sep_worker 里。
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 包内优先，其次常见的缓存位置（用户以前下过就不用再下）
_SEARCH_DIRS = [
    os.path.join(ROOT, "models", "sep"),
]

# 文件名关键词：UVR 的 Karaoke / HP 系列
_KARAOKE_KEYS = ("karaoke", "5_hp", "6_hp", "uvr")


def find_model():
    """返回可用的 Karaoke 模型路径；没有返回空串。"""
    cands = []
    for d in _SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith((".pth", ".onnx")):
                continue
            low = fn.lower()
            if any(k in low for k in _KARAOKE_KEYS):
                cands.append(os.path.join(d, fn))
    # 优先 5_HP（质量与速度平衡最好），其次其它
    for c in cands:
        if "5_hp" in os.path.basename(c).lower():
            return c
    return cands[0] if cands else ""


def has_separator_backend():
    """audio-separator 是否可用（跑 Karaoke 的前提）。"""
    try:
        import audio_separator  # noqa: F401
        return True
    except Exception:
        return False

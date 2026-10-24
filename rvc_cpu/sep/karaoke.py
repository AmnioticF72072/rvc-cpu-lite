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
    """audio-separator 是否可用（跑 Karaoke 的前提）。

    ★ 不能靠 `import audio_separator` 判断。
      主进程的 sys.path 里**没有** runtime\_sep_libs（那是给分离子进程用的），
      所以在主进程里 import 一定失败 —— 即使依赖装得好好的，
      也会被判成"不可用"，界面永远不会给用户开这个功能。

      改成看**文件在不在**：升级包就是把 audio_separator 及其依赖
      解到 runtime\_sep_libs 下。目录在 + 关键 pyd 在 = 装好了。
      真跑的时候如果文件被人为删残，sep_worker 会报明确错误，
      那时错误信息足够定位，不需要在探测阶段花 import 的代价。
    """
    libs = os.path.join(ROOT, "runtime", "_sep_libs")
    pkg = os.path.join(libs, "audio_separator")
    if not os.path.isdir(pkg):
        return False
    # 确认不是空壳目录（解压中断会留下空目录）
    try:
        if not os.listdir(pkg):
            return False
    except OSError:
        return False
    return True


def missing_hint():
    """为什么去和声不可用 —— 给界面显示用。"""
    libs = os.path.join(ROOT, "runtime", "_sep_libs")
    if not has_separator_backend():
        return ("未安装「去和声」组件。\n"
                "这是可选的扩展包，需要单独下载「去和声升级包」并双击里面的安装程序。\n"
                "只想去掉伴奏的话，用「分人声」就行 —— 那个是内置的，不用装东西。")
    if not find_model():
        return ("组件已装好，但缺少去和声模型（5_HP-Karaoke-UVR.pth）。\n"
                "请把模型放到 models\\sep\\ 目录里。")
    return ""

# -*- coding: utf-8 -*-
"""分离后端层（separation backends）。

为什么要单独一层：
    "分离"这件事在 v2.1 之前是硬绑在线程里、依赖外部 torch 环境的一次性调用，
    导致三个问题：
      1. 打包版根本跑不了（runtime 里没有 torch）
      2. 失败时只报一句"没有分离后端"，用户无法自救
      3. 加一个后端（比如以后的 MDX-ONNX）要改主干代码

    现在把它拆成：**探测 → 选择 → 在独立子进程里执行**。
    主进程不需要 import torch，分离跑到一半内存暴涨也不会拖死界面。

后端优先级（按"确定可用且质量"排）：
    1. bundled-demucs  包内自带的 torch + demucs（内置，开箱即用）
    2. external-demucs 借用用户机器上已有的 Python 环境
    3. uvr             音频分离器（audio-separator），主要用于"去和声"

每个后端只需实现两个东西：available() 和 run(mode, in, out, on_progress)。
"""
from __future__ import annotations

import os
import sys

# 包根目录（.../声音转换Plus）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 内置依赖目录：runtime/_sep_libs（torch/demucs 等）
BUNDLED_LIBS = os.path.join(ROOT, "runtime", "_sep_libs")

MODES = ("none", "karaoke", "vocals")


def bundled_libs():
    """返回包内自带的分离依赖目录（存在才返回）。"""
    return BUNDLED_LIBS if os.path.isdir(BUNDLED_LIBS) else ""


def has_bundled_torch():
    """包内是否带了 torch（判断"开箱即用"的依据）。"""
    lib = bundled_libs()
    if not lib:
        return False
    return os.path.isdir(os.path.join(lib, "torch"))


def _env_with_libs(extra_lib=""):
    """给子进程准备环境变量：把内置依赖挂到 PYTHONPATH 最前面。

    注意必须**保留**系统原有的 PYTHONPATH 之外的变量，但要把沙箱注入的
    PYTHONPATH 清掉（那里面有个 sitecustomize.py 会把删除操作改写，
    子进程里会莫名其妙失败）。
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    parts = []
    if extra_lib:
        parts.append(extra_lib)
    lib = bundled_libs()
    if lib:
        parts.append(lib)
    if parts:
        env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    # 让 torch 别开多线程抢满 CPU（分离是长任务，抢满会让界面卡死）
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def describe():
    """给界面用：当前能做什么、缺什么。"""
    info = {
        "bundled": has_bundled_torch(),
        "libs_dir": bundled_libs(),
        "backends": [],
        "vocals": False,
        "karaoke": False,
    }
    if has_bundled_torch():
        info["backends"].append("内置 demucs")
        info["vocals"] = True
    try:
        from . import worker
        if worker.find_external_python() and not has_bundled_torch():
            info["backends"].append("外部 Python + demucs")
            info["vocals"] = True
    except Exception:
        pass
    # karaoke 需要 UVR 模型 + audio-separator
    try:
        from . import karaoke
        if karaoke.find_model():
            info["karaoke"] = True
            info["karaoke_model"] = os.path.basename(karaoke.find_model())
    except Exception:
        pass
    return info

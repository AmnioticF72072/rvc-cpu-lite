# -*- coding: utf-8 -*-
"""分离运行时：决定用哪个 Python、权重放哪、环境变量怎么设。

关键设计：**分离跑在独立子进程里**。
    理由有三条，都是踩过的：
      1. torch 光 import 就要几百 MB 常驻内存；本机 16G 常年吃紧，
         主进程装下它之后跑推理必然掉进换页。
      2. 分离耗时几分钟，放在主进程里会把界面一起卡住（HTTP 无法响应）。
      3. torch 崩了只带走子进程，界面还在，用户能看到报错并重试。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLED_LIBS = os.path.join(ROOT, "runtime", "_sep_libs")
# 内置的 demucs 权重目录（预置 htdemucs，避免首次使用时联网下载）
BUNDLED_TORCH_HOME = os.path.join(ROOT, "models", "sep", "torch_home")


def bundled_torch():
    return os.path.isdir(os.path.join(BUNDLED_LIBS, "torch"))


def find_external_python():
    """找一个装了 torch 的外部 Python（包内没带 torch 时兜底）。"""
    cands = []
    try:
        import shutil
        for exe in ("python", "python3"):
            w = shutil.which(exe)
            if w:
                cands.append(w)
    except Exception:
        pass
    # 训练 RVC 常见留下的环境
    for base in (r"G:\AIVoice\env", os.path.expanduser("~\\AppData\\Local\\Programs")):
        try:
            if os.path.isdir(base):
                for name in sorted(os.listdir(base)):
                    p = os.path.join(base, name, "Scripts", "python.exe")
                    if os.path.isfile(p):
                        cands.append(p)
        except OSError:
            pass
    for py in cands:
        if not os.path.isfile(py):
            continue
        try:
            import subprocess
            r = subprocess.run([py, "-c", "import torch, demucs"],
                               capture_output=True, timeout=90)
            if r.returncode == 0:
                return py
        except Exception:
            continue
    return ""


def worker_python():
    """跑分离该用哪个解释器。

    包内带了 torch 就用**当前进程自己的解释器**（runtime/python.exe）——
    配合 PYTHONPATH 指向 _sep_libs 即可，不需要用户装任何东西。
    """
    if bundled_torch():
        return sys.executable
    return find_external_python()


def build_env():
    """子进程环境。

    ★ 必须清掉 PYTHONPATH 再重设：沙箱会注入一个带 sitecustomize.py 的目录，
      那个 shim 把所有删除改写成"移入回收站"，回收站不可用时 fail-closed，
      torch 加载动态库时会莫名其妙失败。
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    paths = []
    if bundled_torch():
        paths.append(BUNDLED_LIBS)
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths)

    # 权重目录：优先用内置的（离线可用），没有才回落到用户缓存
    if os.path.isdir(BUNDLED_TORCH_HOME):
        env["TORCH_HOME"] = BUNDLED_TORCH_HOME
    else:
        env.setdefault("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))

    env["PYTHONIOENCODING"] = "utf-8"
    # 别让 torch 抢满 CPU，否则界面会卡（分离是分钟级长任务）
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    # 明确关掉 GPU：本机显卡是矿卡且无 CUDA 运行时，探测它只会浪费时间
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return env

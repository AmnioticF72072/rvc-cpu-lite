# -*- coding: utf-8 -*-
"""分离（去和声 / 分人声）的调度层。

CPU 版本体不装 torch，所以这一步交给"装了 torch 的 python"以子进程方式跑
（tools/sep_worker.py）。找不到就明确告诉用户缺什么，不静默失败。
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "tools", "sep_worker.py")

# 可能装了 torch 的 python，按优先级
PY_CANDIDATES = [
    os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
    os.path.join(ROOT, "..", "env", "rvc", "Scripts", "python.exe"),
    r"G:\AIVoice\env\rvc\Scripts\python.exe",
]


def _has_module(py, mod):
    """不真的 import（torch 太慢），直接看 site-packages 里有没有目录。"""
    if not os.path.isfile(py):
        return False
    base = os.path.dirname(os.path.dirname(py))          # 去掉 Scripts\python.exe
    for sp in (os.path.join(base, "Lib", "site-packages"),
               os.path.join(base, "lib", "python3.10", "site-packages")):
        if not os.path.isdir(sp):
            continue
        if glob.glob(os.path.join(sp, mod + "*")):
            return True
    return False


def probe():
    """返回 {python, backends, karaoke_models, ready_karaoke, ready_vocals}"""
    info = {"python": "", "backends": [], "karaoke_models": [], "dirs": []}
    for py in PY_CANDIDATES:
        if not os.path.isfile(py):
            continue
        mods = [m for m in ("demucs", "audio_separator") if _has_module(py, m)]
        if mods:
            info["python"] = py
            info["backends"] = mods
            break
    # Karaoke 模型候选：models\sep 优先，其次各种缓存
    dirs = [os.path.join(ROOT, "models", "sep"),
            os.path.expanduser("~/.cache/audio_separator"),
            os.path.expanduser("~/.cache/audio_separator/models"),
            os.path.expanduser("~/.cache/torch/hub/checkpoints")]
    info["dirs"] = [d for d in dirs if os.path.isdir(d)]
    for d in info["dirs"]:
        for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            b = os.path.basename(p).lower()
            if os.path.isfile(p) and "karaoke" in b and b.endswith((".pth", ".onnx")):
                info["karaoke_models"].append(p)
    info["ready_karaoke"] = bool(info["python"] and info["karaoke_models"])
    info["ready_vocals"] = bool(info["python"])
    return info


def run(mode, in_path, out_dir, on_progress=None, model=""):
    """跑分离。mode: karaoke | vocals。返回人声文件路径。"""
    info = probe()
    py = info["python"]
    if not py:
        raise RuntimeError(
            "这台机器没有分离后端（需要 audio-separator 或 demucs + torch）。\n"
            "只做变声、不去和声的话，把「先去和声」选成「不做」就能继续用。")

    def emit(frac, msg):
        if on_progress:
            try:
                on_progress(float(frac), str(msg))
            except Exception:
                pass

    os.makedirs(out_dir, exist_ok=True)
    if mode == "karaoke" and not model and info["karaoke_models"]:
        model = info["karaoke_models"][0]      # 探测到的模型直接传给 worker
    cmd = [py, WORKER, "--in", in_path, "--out", out_dir, "--mode", mode]
    if model:
        cmd += ["--model", model]

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)          # 沙箱注入的 shim 会污染子进程
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("HF_HOME", os.path.expanduser("~/.cache"))

    emit(0.02, "启动分离进程…")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="ignore", env=env, bufsize=1)
    out_path, err, last = "", "", ""
    for line in p.stdout:
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            last = line[-120:]
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        stage = d.get("stage")
        if stage == "error":
            err = d.get("msg", "分离失败")
        elif stage == "done":
            out_path = d.get("out", "")
            emit(0.98, d.get("msg", "分离完成"))
        elif stage == "warn":
            emit(0.5, d.get("msg", ""))
        else:
            emit(d.get("frac", 0.3) * 0.98, d.get("msg", "分离中…"))
    p.wait()

    if p.returncode != 0 or not out_path or not os.path.isfile(out_path):
        if p.returncode == 2:
            raise RuntimeError(
                "没有 Karaoke 分离模型。\n"
                "把 5_HP-Karaoke-UVR.pth 放到 models\\sep\\ 里就能用；\n"
                "或者把「先去和声」改成「只分人声和伴奏」（用 demucs，不需要额外模型）。")
        raise RuntimeError(err or ("分离进程失败（退出码 %s）%s" % (p.returncode, last)))
    return out_path

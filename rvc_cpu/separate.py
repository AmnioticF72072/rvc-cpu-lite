# -*- coding: utf-8 -*-
"""分离调度（对外唯一入口）。

2.1 起这里只做调度，真正的能力在 rvc_cpu/sep/ 里：
    sep/worker.py        选解释器、设环境变量（内置 torch / 外部 Python）
    sep/karaoke.py       找 UVR 去和声模型
    tools/sep_worker.py  子进程里的实际执行

对外接口保持兼容：probe() / run() / LAST_ACCOMP。
"""
from __future__ import annotations

import json
import os
import subprocess

from .sep import karaoke as _karaoke
from .sep import worker as _worker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "tools", "sep_worker.py")

LAST_ACCOMP = ""      # 最近一次分离产出的伴奏轨（vocals 模式才有）


def probe():
    """返回分离能力清单（界面据此显示"能不能用/缺什么"）。"""
    py = _worker.worker_python()
    info = {
        "python": py,
        "bundled": _worker.bundled_torch(),
        "backends": [],
        "karaoke_models": [],
        "ready_karaoke": False,
        "ready_vocals": False,
        "dirs": [],
    }
    if py:
        info["backends"].append("demucs")
        info["ready_vocals"] = True
    m = _karaoke.find_model()
    has_sep = _karaoke.has_separator_backend()
    info["has_separator"] = has_sep
    info["karaoke_hint"] = "" if has_sep else _karaoke.missing_hint()
    if m:
        info["karaoke_models"].append(m)
        info["ready_karaoke"] = bool(py and has_sep and m)
    return info


def describe():
    """给界面用的一句话说明。"""
    p = probe()
    if p["bundled"]:
        who = "内置分离引擎"
    elif p["python"]:
        who = "外部 Python"
    else:
        who = ""
    return {
        "vocals": p["ready_vocals"],
        "karaoke": p["ready_karaoke"],
        "engine": who,
        "model": os.path.basename(p["karaoke_models"][0]) if p["karaoke_models"] else "",
    }


def run(mode, in_path, out_dir, on_progress=None, model=""):
    """跑分离。mode: karaoke | vocals。返回人声文件路径。

    vocals 模式会把伴奏轨记在模块变量 LAST_ACCOMP —— 上层用它做合轨。
    """
    global LAST_ACCOMP
    LAST_ACCOMP = ""

    py = _worker.worker_python()
    if not py:
        raise RuntimeError(
            "没有可用的分离后端。\n"
            "这个安装包本应自带分离引擎（runtime\\_sep_libs）。"
            "如果缺失说明压缩包不完整，请重新完整解压。\n"
            "只想变声、不做分离的话，把「先去和声」选成「不做」也能正常用。")

    def emit(frac, msg):
        if on_progress:
            try:
                on_progress(float(frac), str(msg))
            except Exception:
                pass

    os.makedirs(out_dir, exist_ok=True)
    if mode == "karaoke" and not model:
        model = _karaoke.find_model()
        if not _karaoke.has_separator_backend():
            raise RuntimeError(_karaoke.missing_hint())
        if not model:
            raise RuntimeError(_karaoke.missing_hint())

    cmd = [py, WORKER, "--in", in_path, "--out", out_dir, "--mode", mode]
    if model:
        cmd += ["--model", model]

    env = _worker.build_env()
    emit(0.02, "启动分离进程…")

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="ignore",
                         env=env, bufsize=1, cwd=ROOT)
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
            acc = d.get("accomp") or ""
            if acc and os.path.isfile(acc):
                LAST_ACCOMP = acc
            emit(0.98, d.get("msg", "分离完成"))
        elif stage == "warn":
            emit(0.5, d.get("msg", ""))
        else:
            emit(d.get("frac", 0.3) * 0.98, d.get("msg", "分离中…"))
    p.wait()

    if p.returncode != 0 or not out_path or not os.path.isfile(out_path):
        if p.returncode == 2:
            raise RuntimeError(
                "没有可用的去和声模型。\n"
                "把 5_HP-Karaoke-UVR.pth 放到 models\\sep\\ 里；\n"
                "或者把「先去和声」改成「分人声」——内置引擎就能去掉伴奏。")
        raise RuntimeError(err or ("分离进程失败（退出码 %s）%s" % (p.returncode, last)))
    return out_path

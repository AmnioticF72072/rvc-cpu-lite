# -*- coding: utf-8 -*-
"""v2.0 状态与会话层。

拆出来的动因：v1.x 把"全局状态 + 管线缓存 + 任务"全塞在 gui.py 里，
1454 行导致任何新功能都要往同一个文件里加，改一处得通读全文。

这里只做两件事，边界很清楚：
  1. 一个加锁的 state 字典（谁都能读，改动必须走 set_state）
  2. 管线生命周期（加载、按设备重建、索引强度热更新）

**不**负责：路由、任务编排、前端。那些在上层。
"""
from __future__ import annotations

import os
import threading
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
WORK = os.path.join(ROOT, "_工作区")
OUT = os.path.join(ROOT, "输出")

_lock = threading.Lock()
_pipe_lock = threading.Lock()      # 保护管线重建（避免并发各建一份）

_state = {
    "stage": "idle",      # idle|analyzing|separating|loading|running|importing|reading|done|error
    "frac": 0.0,
    "msg": "准备好了",
    "phase": "",
    "input": "",
    "output": "",
    "error": "",
    "log": [],
    "model_ready": False,
    "analysis": None,
    "qa": None,
    "started_at": 0.0,
    "imported": "",
    "mode": "sing",       # sing（翻唱）| read（朗读）
    "read": None,         # 朗读结果统计
    "preview_read": "",   # 朗读试听（只合成载体，不变声）
}

_pipe = None
_pipe_device = None
_meta_cache = None


# ---------------------------------------------------------------- 状态
def log(line):
    """往环形日志里加一行（前端只显示最近 200 行）。"""
    ts = time.strftime("%H:%M:%S")
    with _lock:
        _state["log"] = (_state["log"] + ["%s  %s" % (ts, line)])[-200:]


def set_state(**kw):
    with _lock:
        _state.update(kw)


def get_state():
    with _lock:
        s = dict(_state)
        s["log"] = list(_state["log"])
    return s


def snapshot_log():
    with _lock:
        return list(_state["log"])


# ---------------------------------------------------------------- 模型路径
def model_paths():
    """当前激活模型的 (生成器, ContentVec, RMVPE, 索引或 "")。

    返回 tuple 而不是 dict —— 调用方都是 `gen, vec, rmv, idx = model_paths()`
    这种解包写法。改成 dict 会让解包静默拿到键名，进而把 "gen" 当路径传下去，
    报 "Load model from gen failed: File doesn't exist"。
    """
    from rvc_cpu import models as mm
    p = mm.paths_for()
    return p["gen"], p["vec"], p["rmv"], p["idx"]


def meta():
    """前端要的元信息（版本、模型列表、索引、设备、分离后端）。"""
    global _meta_cache
    from rvc_cpu import models as mm
    from rvc_cpu.infer import available_accel
    from rvc_cpu.separate import probe

    lst, active = mm.list_models()
    idx = ""
    try:
        p = mm.paths_for(active)
        idx = p.get("idx") or ""
    except Exception:
        pass
    sep = probe()
    return {
        "version": VERSION,
        "models": lst,
        "active": active,
        "index": {"name": os.path.basename(idx) if idx else "",
                  "vectors": 0},
        "accel": available_accel(),
        "separate": {"ready_karaoke": sep.get("ready_karaoke", False),
                     "ready_vocals": sep.get("ready_vocals", False)},
        "modes": ["sing", "read"],
        "tts": True,
    }


VERSION = "Plus 2.0"


# ---------------------------------------------------------------- 管线
def invalidate():
    """模型切换 / 导入 / 删除后调用。

    同时作废已有音高分析 —— 那个建议是按旧模型音域算的，换模型就没意义了。
    """
    global _pipe, _pipe_device, _meta_cache
    _pipe = None
    _pipe_device = None
    _meta_cache = None
    set_state(analysis=None)


def ensure_pipeline(index_rate=0.0, device="auto"):
    """模型只加载一次；设备变了重建 session。

    用 _pipe_lock 保护重建路径：管线常驻约 1 GB，两个请求同时进来各建一份，
    16G 机器直接贴上限。
    """
    global _pipe, _pipe_device
    if _pipe is not None and _pipe_device == device:
        _pipe.index_rate = float(index_rate)
        return _pipe
    with _pipe_lock:
        if _pipe is not None and _pipe_device == device:
            _pipe.index_rate = float(index_rate)
            return _pipe
        if _pipe is not None:
            try:
                _pipe = None
                import gc
                gc.collect()
            except Exception:
                pass
        gen, vec, rmv, idx = model_paths()
        missing = [n for n, p in (("生成器", gen), ("ContentVec", vec),
                                  ("RMVPE", rmv)) if not p]
        if missing:
            raise RuntimeError("models 目录里缺少模型文件：%s。请整个文件夹一起解压，"
                               "不要把 bat 单独拷出来。" % "、".join(missing))
        set_state(stage="loading", frac=0.02,
                  msg="正在加载模型（第一次要 10~30 秒）…")
        log("加载模型，设备偏好：" + device)
        from rvc_cpu.infer import RvcPipeline
        _pipe = RvcPipeline(gen, vec, rmv, index_path=idx,
                            index_rate=float(index_rate) if idx else 0.0,
                            protect=0.10, verbose=True, device=device)
        _pipe_device = device
        set_state(model_ready=True,
                  device_used=getattr(_pipe, "device_used", "CPU"))
        log("模型就绪｜设备：" + getattr(_pipe, "device_used", "CPU")
            + ("｜特征索引：" + os.path.basename(idx) if idx else "")
            + ("｜INT8 生成器" if getattr(_pipe, "gen_int8", False) else ""))
    return _pipe


def pipe_loaded():
    return _pipe is not None

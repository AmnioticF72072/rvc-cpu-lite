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
    "active_model": "",
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
_pipe_model = None      # 当前管线对应的模型名（防串模型）
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


VERSION = "Plus 2.2.2"


# ---------------------------------------------------------------- 管线
def invalidate():
    """模型切换 / 导入 / 删除后调用。

    同时作废已有音高分析 —— 那个建议是按旧模型音域算的，换模型就没意义了。
    """
    global _pipe, _pipe_device, _pipe_model, _meta_cache
    _pipe = None
    _pipe_device = None
    _pipe_model = None
    _meta_cache = None
    set_state(analysis=None)


def ensure_pipeline(index_rate=0.0, device="auto", protect=None,
                    rms_mix_rate=None):
    """按「模型 + 设备」两个条件复用管线。

    以前只判设备 —— 只要设备没变就直接复用旧管线。这意味着切换模型后，
    如果哪条路径没走到 invalidate()，就会继续用上一个模型，而界面上显示的
    已经是新模型了（"选了却不生效"）。现在把模型名也纳入判断，
    哪怕漏了 invalidate 也不会串模型。

    protect / rms_mix_rate：翻唱和朗读要的值不一样。
      · 唱歌（protect 0.10）：要保住音色变化，少保护
      · 说话（protect 0.33）：咬字和喘气要清楚，多保护
    这两个值不同就重建管线（或就地更新，因为不影响 session 结构）。
    """
    global _pipe, _pipe_device, _pipe_model
    from rvc_cpu import models as mm
    want = mm.active_name() or ""

    if _pipe is not None and _pipe_device == device and _pipe_model == want:
        # ensure_index：如果第一次是按 0 加载的（索引没读进内存），
        # 现在要 0.75，必须补加载，否则检索被静默跳过
        _pipe.ensure_index(float(index_rate))
        if protect is not None:
            _pipe.protect = float(protect)
        if rms_mix_rate is not None:
            _pipe.rms_mix_rate = float(rms_mix_rate)
        return _pipe
    with _pipe_lock:
        if _pipe is not None and _pipe_device == device and _pipe_model == want:
            _pipe.ensure_index(float(index_rate))
            if protect is not None:
                _pipe.protect = float(protect)
            if rms_mix_rate is not None:
                _pipe.rms_mix_rate = float(rms_mix_rate)
            return _pipe
        if _pipe is not None:
            log("模型有变化，重新加载：" + (want or "?"))
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
                  msg="正在加载模型「%s」（第一次要 10~30 秒）…" % (want or "?"))
        log("加载模型「%s」，设备偏好：%s" % (want or "?", device))
        from rvc_cpu.infer import RvcPipeline
        _pipe = RvcPipeline(gen, vec, rmv, index_path=idx,
                            index_rate=float(index_rate) if idx else 0.0,
                            protect=float(protect) if protect is not None else 0.10,
                            rms_mix_rate=(float(rms_mix_rate)
                                          if rms_mix_rate is not None else 0.40),
                            verbose=True, device=device)
        _pipe_device = device
        _pipe_model = want
        set_state(model_ready=True,
                  device_used=getattr(_pipe, "device_used", "CPU"),
                  active_model=want)
        log("模型就绪「%s」｜设备：%s" % (want or "?",
                                        getattr(_pipe, "device_used", "CPU"))
            + ("｜特征索引：" + os.path.basename(idx) if idx else "")
            + ("｜INT8 生成器" if getattr(_pipe, "gen_int8", False) else ""))
    return _pipe


def pipe_loaded():
    return _pipe is not None

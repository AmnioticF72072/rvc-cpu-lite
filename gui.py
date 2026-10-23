# -*- coding: utf-8 -*-
"""
声音转换 Plus —— RVC 纯 CPU 版图形界面（白色玻璃风）

Plus 相比基础版多的三件事：
  1. Karaoke 去和声：整首歌里叠的和声/伴唱会让人声"糊"，先用 UVR Karaoke 模型剥掉
  2. 智能校准：分析原曲音高（中位 / 70 分位 / 95 分位），按模型音域自动给出移调半音
  3. 特征检索：用 .index 索引把音色往训练集拉，检索强度可调

技术上：本地 HTTP 服务 + Edge 应用窗（才有 backdrop-filter 真毛玻璃）。
不经手任何外部网络，音频只在本机处理。
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "_工作区")
OUT = os.path.join(ROOT, "输出")
MODELS = os.path.join(ROOT, "models")
MODEL_STAGE = os.path.join(WORK, "_导入暂存")
PORT = int(os.environ.get("RVC_GUI_PORT", "8790"))
VERSION = "Plus 1.2"

sys.path.insert(0, ROOT)

_lock = threading.Lock()
_pipe_lock = threading.Lock()   # 保护管线重建（避免并发各建一份）
_state = {
    "stage": "idle",          # idle | analyzing | separating | loading | running | importing | done | error
    "frac": 0.0,
    "msg": "准备好了",
    "phase": "",
    "input": "",
    "output": "",
    "error": "",
    "log": [],
    "model_ready": False,
    "analysis": None,
    "started_at": 0.0,
    "imported": "",
}
_pipe = None
_pipe_device = None
_abort = False
_meta_cache = None


def log(line):
    with _lock:
        _state["log"].append("%s  %s" % (time.strftime("%H:%M:%S"), line))
        if len(_state["log"]) > 240:
            del _state["log"][:100]
    print(line, flush=True)


def set_state(**kw):
    with _lock:
        _state.update(kw)


def get_state():
    with _lock:
        d = dict(_state)
        d["log"] = list(_state["log"])
        return d


# ---------------------------------------------------------------- 资源发现
def model_paths():
    """当前激活模型的 (生成器, ContentVec, RMVPE, 索引)。"""
    from rvc_cpu import models as mm
    p = mm.paths_for()
    return p["gen"], p["vec"], p["rmv"], p["idx"]


def invalidate():
    """模型切换 / 导入 / 删除后调用：丢掉已加载流水线与元数据缓存。"""
    global _pipe, _pipe_device, _meta_cache
    _pipe = None
    _pipe_device = None
    _meta_cache = None


def meta():
    """给界面用的资源清单（缓存一次，避免每次轮询都扫盘）。"""
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache
    from rvc_cpu import models as mm
    gen, vec, rmv, idx = model_paths()
    from rvc_cpu import separate as sep
    probe = sep.probe()
    index_info = None
    if idx:
        index_info = {"name": os.path.basename(idx),
                      "mb": round(os.path.getsize(idx) / 1024 ** 2, 1)}
        try:
            import faiss
            ix = faiss.read_index(idx)
            index_info["vectors"] = int(ix.ntotal)
        except Exception as e:
            index_info["error"] = str(e)[:80]
    models, active = mm.list_models()
    _meta_cache = {
        "version": VERSION,
        "active_model": active,
        "model_list": models,
        "models": {
            "gen": os.path.basename(gen) if gen else "",
            "vec": os.path.basename(vec) if vec else "",
            "rmvpe": os.path.basename(rmv) if rmv else "",
        },
        "index": index_info,
        "sep": {
            "ready_karaoke": probe["ready_karaoke"],
            "ready_vocals": probe["ready_vocals"],
            "karaoke_models": [os.path.basename(p) for p in probe["karaoke_models"]],
            "backend": os.path.basename(os.path.dirname(os.path.dirname(probe["python"])))
            if probe["python"] else "",
            "backends": probe["backends"],
        },
    }
    return _meta_cache


def ensure_pipeline(index_rate=0.0, device="auto"):
    """模型只加载一次；但设备变了必须重建 session，否则切换不生效。

    用 _pipe_lock 保护重建路径：管线常驻约 1 GB，两个请求同时进来
    各建一份，16G 机器直接贴着上限，很容易触发换页甚至 OOM。
    """
    global _pipe, _pipe_device
    if _pipe is not None and _pipe_device == device:
        _pipe.index_rate = float(index_rate)
        return _pipe
    with _pipe_lock:
        # 等锁期间可能已经被别的线程建好了
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
        missing = [n for n, p in (("生成器", gen), ("ContentVec", vec), ("RMVPE", rmv)) if not p]
        if missing:
            raise RuntimeError("models 目录里缺少模型文件：%s。请整个文件夹一起解压，"
                               "不要把 bat 单独拷出来。" % "、".join(missing))
        set_state(stage="loading", frac=0.02, msg="正在加载模型（第一次要 10~30 秒）…")
        log("加载模型，设备偏好：" + device)
        from rvc_cpu.infer import RvcPipeline
        _pipe = RvcPipeline(gen, vec, rmv, index_path=idx,
                            index_rate=float(index_rate) if idx else 0.0,
                            protect=0.10, verbose=True, device=device)
        _pipe_device = device
        set_state(model_ready=True, device_used=getattr(_pipe, "device_used", "CPU"))
        log("模型就绪｜设备：" + getattr(_pipe, "device_used", "CPU")
            + ("｜特征索引：" + os.path.basename(idx) if idx else ""))
    return _pipe


# ---------------------------------------------------------------- 任务
def run_analyze(path):
    try:
        from rvc_cpu.calibrate import analyze_file, load_profile, suggest
        from rvc_cpu import models as mm

        _, _, rmv, _ = model_paths()
        set_state(stage="analyzing", frac=0.05, msg="分析音高（第一次约 5~10 秒）…")
        stats = analyze_file(rmv, path, progress=lambda f, m: set_state(
            stage="analyzing", frac=0.05 + 0.9 * f, msg=m))
        if not stats:
            set_state(stage="idle", frac=0, analysis=None,
                      msg="没检测到人声，这段可能是静音或纯伴奏")
            return
        prof = load_profile(MODELS, mm.active_name() or "myvoice51")
        r = suggest(stats, prof)
        set_state(stage="idle", frac=0,
                  analysis={"stats": stats, "key": r["key"], "reason": r["reason"],
                            "shifted": r.get("shifted", {}),
                            "profile": {"name": prof.get("name", ""),
                                        "lo": prof["target_lo"], "hi": prof["target_hi"],
                                        "p95": prof["p95_max"]}},
                  msg="分析完成：建议 %+d 半音" % r["key"])
        log("音高分析：中位 %.0f / 70 分位 %.0f / 95 分位 %.0f → 建议 %+d" % (
            stats["median"], stats["p70"], stats["p95"], r["key"]))
    except Exception as e:
        set_state(stage="error", error="%s: %s" % (type(e).__name__, e), msg="分析失败")
        log("分析失败：" + str(e))
        traceback.print_exc()


def run_job(files, key, index_rate, sep_mode, device="auto"):
    """批量转换：逐个文件走 分离 → 转换，总进度按文件数摊分。

    单个文件失败只跳过它并记下来，不拖垮整批 —— 批量跑一专辑时，
    一首坏文件把剩下九首全废掉是说不通的。
    """
    global _abort
    total = len(files)
    outs = []
    failures = []          # [(文件名, 失败原因)]
    try:
        for i, in_path in enumerate(files):
            if _abort:
                raise RuntimeError("已取消")
            span = 1.0 / total
            f0p = i * span

            def sub(frac, msg, _b=f0p, _s=span, _n=i + 1):
                set_state(frac=round(_b + _s * float(frac), 4),
                          msg=("[%d/%d] " % (_n, total)) + str(msg))

            try:
                src = in_path
                if sep_mode and sep_mode != "none":
                    from rvc_cpu import separate as sep

                    set_state(stage="separating", phase="分离",
                              msg="正在分离和声/伴奏（这一步最慢，耐心等）…")
                    log("分离：%s（%s）" % (os.path.basename(in_path), sep_mode))
                    src = sep.run(sep_mode, in_path, os.path.join(WORK, "sep"),
                                  on_progress=lambda f, m, _sub=sub: _sub(0.05 + 0.4 * f, m))
                    log("分离得到：" + os.path.basename(src))

                pipe = ensure_pipeline(index_rate, device)

                def prog(frac, msg, _sub=sub, _sep=(sep_mode and sep_mode != "none")):
                    if _abort:
                        raise RuntimeError("已取消")
                    b = 0.5 if _sep else 0.05
                    _sub(b + (0.95 - b) * float(frac), msg)

                base = os.path.splitext(os.path.basename(in_path))[0]
                base = re.sub(r"[\\/:*?\"<>|]", "_", base)[:60]
                os.makedirs(OUT, exist_ok=True)
                out_path = os.path.join(OUT, "%s_Plus.wav" % base)
                n = 2
                while os.path.isfile(out_path):
                    out_path = os.path.join(OUT, "%s_Plus(%d).wav" % (base, n))
                    n += 1

                set_state(stage="running", phase="转换", error="",
                          started_at=time.time())
                log("转换：%s（移调 %+d，检索 %.2f，设备 %s）" %
                    (os.path.basename(src), int(key), index_rate, device))
                pipe.convert_file(src, out_path, f0_up_key=int(key), on_progress=prog)
                outs.append(out_path)
                log("完成：" + out_path)
            except Exception as one:
                m = str(one)
                if "已取消" in m or _abort:
                    raise RuntimeError("已取消")
                failures.append((os.path.basename(in_path), m))
                log("跳过「%s」：%s" % (os.path.basename(in_path), m))
                continue

        # ---- 汇总 ----
        if not outs:
            first = failures[0][1] if failures else "没有可转换的文件"
            set_state(stage="error", frac=0, phase="", error=first, msg="全部失败")
            log("全部失败：%s" % first)
            return

        last = outs[-1]
        if failures:
            msg = "完成 %d 个，%d 个失败（详情见下方日志）" % (len(outs), len(failures))
        elif len(outs) > 1:
            msg = "全部完成（%d 个文件）" % len(outs)
        else:
            try:
                sz = os.path.getsize(last) / 1024 ** 2
            except OSError:
                sz = 0.0
            msg = "完成 · %s（%.1f MB）" % (os.path.basename(last), sz)
        set_state(stage="done", frac=1.0, output=last, phase="", outputs=outs,
                  msg=msg)
    except Exception as e:
        m = str(e)
        if "已取消" in m:
            set_state(stage="idle", frac=0, phase="", msg="已取消（已完成的文件保留）")
            log("任务已取消")
            return
        hint = ""
        if "AVX" in m:
            hint = "（这台机器的 CPU 缺 AVX2，跑不了）"
        elif "Invalid input shape" in m:
            hint = "（音频太短，至少给 2 秒）"
        set_state(stage="error", error="%s%s" % (m, hint), msg="出错了")
        log("失败：" + m)
        traceback.print_exc()


# ---------------------------------------------------------------- 导入模型
def run_import(name):
    """把暂存目录里的模型文件导入 models 并切换过去（.pth 会自动转 ONNX）。"""
    try:
        from rvc_cpu import models as mm
        set_state(stage="importing", frac=0.03, msg="准备导入…", error="", imported="")
        info = mm.import_from_stage(
            MODEL_STAGE, name,
            progress=lambda f, m: set_state(stage="importing",
                                            frac=round(0.03 + 0.94 * float(f), 3), msg=m))
        invalidate()
        set_state(stage="idle", frac=0, msg="已导入并切换到「%s」" % info["name"],
                  imported=info["name"], analysis=None)
        log("导入模型：%s（来源 %s）" % (info["name"], info.get("source") or "onnx"))
    except Exception as e:
        set_state(stage="error", error=str(e), msg="导入失败")
        log("导入失败：" + str(e))
        traceback.print_exc()


# ---------------------------------------------------------------- 页面
PAGE = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>声音转换 Plus</title>
<style>
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  body{margin:0;font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",
    "Segoe UI",sans-serif;color:#0d1526;min-height:100vh;
    background:
      radial-gradient(900px 620px at 6% -10%,#e2edff 0%,rgba(226,237,255,0) 62%),
      radial-gradient(780px 560px at 104% 2%,#efe9ff 0%,rgba(239,233,255,0) 60%),
      radial-gradient(760px 700px at 46% 118%,#e2f6ff 0%,rgba(226,246,255,0) 58%),
      #f5f8fc;}
  body::before{content:"";position:fixed;inset:-12%;z-index:0;pointer-events:none;
    background:
      radial-gradient(340px 340px at 16% 20%,rgba(10,132,255,.16),transparent 70%),
      radial-gradient(380px 380px at 84% 14%,rgba(150,110,255,.15),transparent 70%),
      radial-gradient(430px 430px at 62% 92%,rgba(0,190,180,.12),transparent 70%);}
  .wrap{position:relative;z-index:1;max-width:1080px;margin:0 auto;padding:26px 22px 60px}

  .top{display:flex;align-items:center;gap:12px;margin-bottom:20px}
  .logo{width:40px;height:40px;border-radius:13px;background:linear-gradient(150deg,#0a84ff,#5ac8fa);
    box-shadow:0 7px 20px rgba(10,132,255,.34);position:relative;flex:none}
  .logo i{position:absolute;left:9px;right:9px;height:3px;border-radius:3px;background:#fff}
  .logo i:nth-child(1){top:12px}.logo i:nth-child(2){top:19px;left:13px;right:13px}
  .logo i:nth-child(3){top:26px}
  h1{font-size:20.5px;margin:0;letter-spacing:-.3px;display:flex;align-items:center;gap:8px}
  h1 em{font-style:normal;font-size:11.5px;font-weight:700;color:#0a84ff;background:rgba(10,132,255,.1);
    border:1px solid rgba(10,132,255,.2);padding:2px 8px;border-radius:999px;letter-spacing:.5px}
  .sub{font-size:12.5px;color:#6b7a90;margin-top:3px}
  .badge{margin-left:auto;font-size:11.5px;color:#3a4a63;background:rgba(255,255,255,.62);
    border:1px solid rgba(255,255,255,.9);padding:7px 13px;border-radius:999px;
    backdrop-filter:blur(16px) saturate(160%);box-shadow:0 4px 14px rgba(20,40,80,.06);white-space:nowrap}

  .card{background:rgba(255,255,255,.63);border:1px solid rgba(255,255,255,.92);border-radius:26px;
    box-shadow:0 18px 48px rgba(20,40,80,.09),inset 0 1px 0 rgba(255,255,255,.9);
    backdrop-filter:blur(26px) saturate(180%);-webkit-backdrop-filter:blur(26px) saturate(180%);
    padding:22px 24px;margin-bottom:16px}
  .grid{display:grid;grid-template-columns:1fr 330px;gap:16px;align-items:start}
  @media (max-width:900px){.grid{grid-template-columns:1fr}}

  .step{display:flex;align-items:center;gap:9px;margin-bottom:10px}
  .num{width:22px;height:22px;border-radius:50%;background:#0a84ff;color:#fff;font-size:12px;
    display:grid;place-items:center;font-weight:600;flex:none;box-shadow:0 3px 9px rgba(10,132,255,.32)}
  .step h2{font-size:14.5px;margin:0;font-weight:650}
  .opt{font-size:11.5px;color:#8b98aa;font-weight:500;margin-left:2px}
  .hint{font-size:12.5px;color:#7a8899;margin:2px 0 12px}

  .drop{border:1.5px dashed rgba(10,132,255,.34);border-radius:20px;padding:22px 16px;text-align:center;
    background:rgba(255,255,255,.5);cursor:pointer;transition:.18s}
  .drop:hover,.drop.over{border-color:#0a84ff;background:rgba(240,247,255,.92);transform:translateY(-1px);
    box-shadow:0 10px 24px rgba(10,132,255,.12)}
  .drop b{display:block;font-size:14.5px;margin-bottom:3px}
  .drop span{font-size:12.3px;color:#7a8899}
  .file{display:none;align-items:center;gap:10px;margin-top:10px;padding:11px 13px;border-radius:14px;
    background:rgba(10,132,255,.07);border:1px solid rgba(10,132,255,.15)}
  .file.on{display:flex}
  .file em{font-style:normal;font-size:13px;color:#0a5bd0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

  .seg{display:flex;gap:8px;flex-wrap:wrap}
  .seg label{flex:1;min-width:150px;position:relative;cursor:pointer}
  .seg input{position:absolute;opacity:0;pointer-events:none}
  .seg span{display:block;text-align:center;padding:11px 10px;border-radius:15px;font-size:13px;
    color:#41526b;background:rgba(255,255,255,.72);border:1px solid rgba(13,21,38,.08);transition:.15s}
  .seg input:checked+span{background:linear-gradient(180deg,#3aa2ff,#0a84ff);color:#fff;border-color:transparent;
    box-shadow:0 8px 18px rgba(10,132,255,.28)}
  .seg input:disabled+span{opacity:.42;cursor:not-allowed}
  .seg small{display:block;font-size:10.8px;opacity:.78;margin-top:2px}

  input[type=range]{-webkit-appearance:none;width:100%;height:6px;border-radius:6px;outline:none;
    background:linear-gradient(90deg,#5ac8fa,#0a84ff);margin:12px 0 4px}
  input[type=range]:disabled{background:#dfe6f0}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:24px;height:24px;border-radius:50%;
    background:#fff;border:1px solid rgba(0,0,0,.06);box-shadow:0 4px 12px rgba(10,60,140,.24);cursor:pointer}
  .keyscale,.row{display:flex;justify-content:space-between;align-items:center}
  .keyscale{font-size:11.3px;color:#8494a8}
  .keyval{font-size:25px;font-weight:700;letter-spacing:-.5px}
  .keyval small{font-size:11.5px;font-weight:500;color:#7a8899;margin-left:5px}
  .keyhint{font-size:12.3px;color:#7a8899;text-align:right;max-width:62%}

  .btn{border:none;border-radius:999px;padding:15px 22px;font-size:15.5px;font-weight:650;cursor:pointer;
    background:linear-gradient(180deg,#339cff,#0a84ff);color:#fff;width:100%;font-family:inherit;
    box-shadow:0 12px 26px rgba(10,132,255,.3),inset 0 1px 0 rgba(255,255,255,.4);transition:.16s}
  .btn:hover:not(:disabled){filter:brightness(1.04);transform:translateY(-1px)}
  .btn:disabled{background:#c8d3e2;box-shadow:none;cursor:not-allowed;color:#f3f7fc}
  .btn.mini{width:auto;padding:9px 15px;font-size:13px;box-shadow:0 6px 16px rgba(10,132,255,.22)}
  .btn.ghost{background:rgba(255,255,255,.76);color:#0a5bd0;border:1px solid rgba(10,132,255,.22);
    box-shadow:0 6px 16px rgba(20,40,80,.08)}

  .prog{margin-top:14px}
  .bar{height:8px;border-radius:8px;background:rgba(13,21,38,.07);overflow:hidden}
  .bar>i{display:block;height:100%;width:0;border-radius:8px;
    background:linear-gradient(90deg,#5ac8fa,#0a84ff);transition:width .35s ease}
  .pmsg{display:flex;justify-content:space-between;gap:12px;font-size:12.3px;color:#5d6c82;margin-top:8px}
  .pmsg span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

  .ana{display:none;margin-top:12px;padding:14px 15px;border-radius:16px;
    background:rgba(10,132,255,.06);border:1px solid rgba(10,132,255,.14)}
  .ana.on{display:block}
  .ana .k{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px}
  .ana .k div{font-size:12px;color:#5d6c82}
  .ana .k b{display:block;font-size:16.5px;color:#0e2a4d;font-weight:700}
  .ana .why{font-size:12.5px;color:#41526b;line-height:1.55}

  .out{display:none;margin-top:14px;padding:14px 15px;border-radius:16px;
    background:rgba(10,150,95,.07);border:1px solid rgba(10,150,95,.18)}
  .out.on{display:block}
  .out b{font-size:13.5px;color:#0a7a45}
  .out code{display:block;font-size:11.6px;color:#4c5c72;word-break:break-all;margin:6px 0 10px;
    font-family:ui-monospace,Consolas,monospace}

  .side h3{font-size:12.6px;margin:0 0 9px;color:#40506a;letter-spacing:.3px}
  .how{list-style:none;padding:0;margin:0 0 14px}
  .how li{display:flex;gap:8px;font-size:12.9px;color:#3b4a60;margin-bottom:8px}
  .how b{flex:none;width:17px;height:17px;border-radius:50%;background:rgba(10,132,255,.12);
    color:#0a5bd0;font-size:10.5px;display:grid;place-items:center;margin-top:3px}
  details{border-top:1px solid rgba(13,21,38,.07);padding:9px 0}
  details:first-of-type{border-top:none}
  summary{cursor:pointer;font-size:12.8px;color:#33445c;font-weight:600;list-style:none}
  summary::-webkit-details-marker{display:none}
  summary::before{content:"＋";color:#0a84ff;margin-right:6px;font-weight:700}
  details[open] summary::before{content:"－"}
  details p{font-size:12.3px;color:#5b6a80;margin:7px 0 0 18px}

  .logbox{margin-top:12px;max-height:140px;overflow:auto;background:rgba(13,21,38,.045);
    border-radius:13px;padding:9px 11px;font:11.6px/1.7 ui-monospace,Consolas,monospace;color:#48566b;
    white-space:pre-wrap}
  .warn{font-size:12px;color:#9a6b00;background:rgba(255,190,0,.1);border:1px solid rgba(255,190,0,.22);
    padding:8px 11px;border-radius:12px;margin-top:10px;display:none}
  .warn.on{display:block}
</style></head><body>
<div class="wrap">
  <div class="top">
    <div class="logo"><i></i><i></i><i></i></div>
    <div>
      <h1>声音转换 <em>PLUS</em></h1>
      <div class="sub">RVC 纯 CPU · 自动校准 · 去和声 · 特征检索 · 全程离线</div>
    </div>
    <div class="badge" id="badge">正在读取模型…</div>
  </div>

  <div class="grid">
    <div>
      <!-- 0 音色模型 -->
      <div class="card">
        <div class="step"><div class="num">0</div><h2>音色模型</h2>
          <span class="opt">可导入别人的模型</span></div>
        <p class="hint">当前用的是「<b id="curModel">—</b>」。可以导入其他人分享的 RVC 模型：
          拖入 <b>.pth</b>（本机有 PyTorch 就自动转换）或 <b>.onnx</b> 都行，
          有配套 .index 一起放，音色会更像。</p>
        <div class="seg" id="segModel"></div>
        <div class="row" style="margin-top:10px;gap:10px">
          <button class="btn mini ghost" id="btnImport">导入模型…</button>
          <button class="btn mini ghost" id="btnDelModel">删除这个</button>
          <span id="modelInfo" style="font-size:12px;color:#7a8899;text-align:right;flex:1"></span>
        </div>
        <input type="file" id="modelPicker" accept=".pth,.onnx,.json,.index" multiple hidden>
        <div class="warn" id="modelWarn"></div>
      </div>

      <!-- 1 选文件 -->
      <div class="card">
        <div class="step"><div class="num">1</div><h2>选一个音频</h2></div>
        <p class="hint">wav / flac / ogg，也可以直接拖进来。至少 2 秒。</p>
        <div class="drop" id="drop">
          <b>点这里选文件，或把文件拖进来</b>
          <span>第一次先用 10 秒左右的短音频试，确认能用再转整首</span>
        </div>
        <div class="file" id="filebox"><em id="fname">—</em></div>
        <input type="file" id="picker" accept=".wav,.flac,.ogg,.aif,.aiff,audio/*" multiple hidden>
        <div class="row" style="margin-top:8px">
          <span id="qinfo" style="font-size:12.3px;color:#7a8899">还没选文件</span>
          <button class="btn mini ghost" id="btnClear" style="display:none">清空</button>
        </div>
      </div>

      <!-- 2 去和声 -->
      <div class="card">
        <div class="step"><div class="num">2</div><h2>先去和声 / 伴奏</h2>
          <span class="opt">整首歌才需要</span></div>
        <p class="hint">自己一个人唱的录音选「不做」；整首歌（有伴奏或叠了和声）先分离再转，效果差很多。</p>
        <div class="seg" id="segSep">
          <label><input type="radio" name="sep" value="none" checked><span>不做<small>纯人声、单人清唱</small></span></label>
          <label><input type="radio" name="sep" value="karaoke"><span>去和声<small>剥掉伴唱，留主唱</small></span></label>
          <label><input type="radio" name="sep" value="vocals"><span>分人声<small>顺带去掉伴奏</small></span></label>
        </div>
        <div class="warn" id="sepWarn"></div>
      </div>

      <!-- 3 智能校准 -->
      <div class="card">
        <div class="step"><div class="num">3</div><h2>智能校准</h2>
          <span class="opt">按模型音域自动定移调</span></div>
        <p class="hint">先分析原曲音高，程序据此给出建议移调；不分析也能手调。</p>
        <button class="btn mini ghost" id="btnAna">分析音高</button>
        <div class="ana" id="anaBox">
          <div class="k">
            <div>中位<b id="aMed">—</b></div>
            <div>70 分位<b id="aP70">—</b></div>
            <div>95 分位<b id="aP95">—</b></div>
            <div>建议移调<b id="aKey">—</b></div>
          </div>
          <div class="why" id="aWhy"></div>
        </div>
        <div class="row" style="margin-top:14px">
          <div class="keyval" id="keyval">0<small>半音</small></div>
          <div class="keyhint" id="keyhint">不变调</div>
        </div>
        <input type="range" id="key" min="-12" max="12" step="1" value="0">
        <div class="keyscale"><span>-12 低</span><span>0</span><span>+12 高</span></div>
      </div>

      <!-- 4 检索 -->
      <div class="card">
        <div class="step"><div class="num">4</div><h2>音色接近度</h2>
          <span class="opt">特征检索</span></div>
        <p class="hint">往训练集里的真实音色靠。调太高会有金属感，0.4~0.6 最稳。</p>
        <div class="row">
          <div class="keyval" id="idxval">0.50<small>强度</small></div>
          <div class="keyhint" id="idxhint">—</div>
        </div>
        <input type="range" id="idxrate" min="0" max="80" step="5" value="50">
      </div>

      <!-- 5 开始 -->
      <div class="card">
        <div class="step"><div class="num">5</div><h2>开始</h2></div>
        <button class="btn" id="go" disabled>开始转换</button>
        <div class="seg" id="segDev" style="margin-top:14px">
          <label><input type="radio" name="dev" value="auto" checked><span>自动<small>有显卡就用</small></span></label>
          <label><input type="radio" name="dev" value="cpu"><span>只用 CPU<small>最稳</small></span></label>
          <label><input type="radio" name="dev" value="gpu"><span>用 GPU<small>需加速包</small></span></label>
        </div>
        <div class="row" style="margin-top:10px">
          <button class="btn mini ghost" id="btnAbort">取消</button>
          <span id="devInfo" style="font-size:12px;color:#7a8899;text-align:right"></span>
        </div>
        <div class="prog">
          <div class="bar"><i id="bar"></i></div>
          <div class="pmsg"><span id="pmsg">选好文件后点上面按钮</span><span id="pct"></span></div>
        </div>
        <div class="out" id="outbox">
          <b>转换好了</b>
          <code id="outpath"></code>
          <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px">
            <button class="btn mini ghost" id="playIn">试听原声</button>
            <button class="btn mini ghost" id="playOut">试听结果</button>
          </div>
          <audio id="player" controls style="width:100%;display:none"></audio>
          <button class="btn mini ghost" id="reveal">打开输出文件夹</button>
        </div>
        <div class="logbox" id="log" hidden></div>
      </div>
    </div>

    <div class="side">
      <div class="card">
        <h3>三步就能用</h3>
        <ul class="how">
          <li><b>1</b><span>选音频。整首歌就把第 2 步设成「去和声」或「分人声」。</span></li>
          <li><b>2</b><span>点「分析音高」，程序会自己算出该移多少半音。</span></li>
          <li><b>3</b><span>点「开始转换」，等进度走完，去「输出」文件夹拿文件。</span></li>
        </ul>
        <h3>常见问题</h3>
        <details><summary>「去和声」和「分人声」差在哪？</summary>
          <p>去和声用的是 UVR Karaoke 模型，专门剥掉叠在主唱后面的伴唱/和声，
             主唱保留得最完整。<br>
             分人声用的是 demucs，把人声和伴奏整体分开，伴奏去得干净，
             但叠在主唱上的和声会被当成"人声"一起留下。</p></details>
        <details><summary>智能校准是怎么算的？</summary>
          <p>先用模型把整段的音高逐帧测出来，取三个统计量：中位（主体音区）、
             70 分位（副歌主体）、95 分位（偶尔的高音）。<br>
             然后找一个偶数半音，让「70 分位」落进模型的舒适区（默认 250~320 Hz），
             同时保证「95 分位」不超过上限（默认 375 Hz，超了就会发紧、破音）。</p></details>
        <details><summary>移调该往哪边调？</summary>
          <p>模型是男声、原唱是女声 → 往低了调（负数）。<br>
             反过来就往高了调。不确定就点「分析音高」让它算。</p></details>
        <details><summary>要多久？</summary>
          <p>这台机器 1 分钟音频约 1.7 分钟。整首歌如果开了分离，先多花十几秒到一分钟分离。</p></details>
        <details><summary>支持 mp3 吗？</summary>
          <p>读 mp3 要看系统解码库，有些机器读不了。最稳是先转成 wav 再拖进来。</p></details>
        <details><summary>转出来有杂音 / 发闷？</summary>
          <p>先确认输入是干净人声；再检查移调是不是拉太狠（超过 ±8 半音容易失真）；
             音色接近度调到 0.5 试试。</p></details>
        <details><summary>能换别人的音色模型吗？</summary>
          <p>点最上面「音色模型」卡片里的「导入模型…」，把下到的模型文件拖进去就行：<br>
             · <b>.pth</b>（最常见的分享格式）：本机装有 PyTorch 时会自动转换成 ONNX 再用；
               没有 PyTorch 会提示你先弄一个，或者直接要 .onnx 格式。<br>
             · <b>.onnx</b>：直接可用，建议连同名 .json（采样率等配置）一起导入。<br>
             · <b>.index</b>：可选，放进后自动启用特征检索。<br>
             注意只支持 v2 模型（768 维）。导入后会自动切换成新模型，原来的模型还在，随时切回来。</p></details>
        <details><summary>这一步会联网吗？</summary>
          <p>不会。界面是本机网页（127.0.0.1），音频不上传。
             唯一可能联网的是第一次用「去和声」时下载 Karaoke 模型。</p></details>
      </div>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let picked = null, timer = null, meta = null, autoKey = null, lastImported = '';

function setPicked(name){
  picked = name;
  $('#fname').textContent = name || '—';
  $('#filebox').classList.toggle('on', !!name);
  $('#go').disabled = !name;
  $('#btnAna').disabled = !name;
}

async function loadMeta(){
  meta = await (await fetch('/api/meta')).json();
  renderModels(meta);
  const s = meta.sep || {};
  const parts = [];
  parts.push(meta.version);
  if (meta.active_model) parts.push(meta.active_model);
  parts.push(meta.index ? '检索已启用' : '无检索索引');
  $('#badge').textContent = parts.join(' · ');

  // 索引
  if (meta.index){
    $('#idxhint').textContent = (meta.index.vectors ? meta.index.vectors + ' 条 · ' : '') + meta.index.name;
  } else {
    $('#idxrate').disabled = true;
    $('#idxhint').textContent = 'models 里没有 .index，已关闭';
    $('#idxval').innerHTML = '—<small>无索引</small>';
  }
  // 分离
  const warn = $('#sepWarn');
  if (!s.ready_vocals){
    warn.textContent = '这台机器没有分离后端（需要 torch + demucs/audio-separator）。第 2 步保持「不做」即可，其余功能不受影响。';
    warn.classList.add('on');
    document.querySelectorAll('input[name=sep]').forEach(r=>{ if(r.value!=='none') r.disabled = true; });
  } else if (!s.ready_karaoke){
    const k = document.querySelector('input[name=sep][value=karaoke]');
    if (k) k.disabled = true;
    warn.textContent = '没找到 Karaoke 模型（' + (s.karaoke_models||[]).join(', ') + '）。「分人声」可用；要用「去和声」，把 5_HP-Karaoke-UVR.pth 放进 models\\sep。';
    warn.classList.add('on');
  }
}

// ---------------------------------------------------------------- 模型管理
function renderModels(m){
  const box = $('#segModel');
  const list = (m && m.model_list) || [];
  box.innerHTML = '';
  if (!list.length){
    box.innerHTML = '<span style="font-size:12.5px;color:#9a6b00">models 目录里没有音色模型。' +
      '点「导入模型…」导入一个，或确认整个文件夹是一起解压的。</span>';
    $('#curModel').textContent = '无';
    $('#modelInfo').textContent = '';
    $('#btnDelModel').style.display = 'none';
    return;
  }
  list.forEach(mm => {
    const lb = document.createElement('label');
    const inp = document.createElement('input');
    inp.type = 'radio'; inp.name = 'mdl'; inp.value = mm.name;
    if (mm.active) inp.checked = true;
    const sp = document.createElement('span');
    sp.textContent = mm.name;
    if (mm.index){
      const sm = document.createElement('small');
      sm.textContent = '带索引';
      sp.appendChild(sm);
    }
    lb.appendChild(inp); lb.appendChild(sp);
    box.appendChild(lb);
  });
  const cur = list.find(x => x.active) || list[0];
  $('#curModel').textContent = cur.name;
  const bits = [cur.mb + ' MB'];
  if (cur.tgt_sr) bits.push(cur.tgt_sr + ' Hz');
  if (cur.version) bits.push(cur.version);
  if (cur.source) bits.push(cur.source === 'pth' ? '.pth 导入' : '.onnx 导入');
  if (cur.imported_at) bits.push(cur.imported_at.slice(5, 10) + ' 导入');
  $('#modelInfo').textContent = bits.join(' · ');
  $('#btnDelModel').style.display = list.length > 1 ? 'inline-flex' : 'none';
  box.querySelectorAll('input[name=mdl]').forEach(r => {
    r.onchange = async () => {
      if (r.value === (meta.active_model || '')) return;
      if (!confirm('切换到模型「' + r.value + '」？\n（当前模型不会删掉，随时可以切回来）')){
        renderModels(meta); return;
      }
      const j = await (await fetch('/api/model/select', {method:'POST',
        headers:{'Content-Type':'application/json'}, body: JSON.stringify({name:r.value})})).json();
      if (j.ok){
        meta = await (await fetch('/api/meta')).json();
        renderModels(meta);
        $('#pmsg').textContent = '已切换到「' + r.value + '」，下次转换用新模型';
      } else {
        $('#pmsg').textContent = '切换失败：' + j.error;
        renderModels(meta);
      }
    };
  });
}

$('#btnImport').onclick = () => $('#modelPicker').click();
$('#modelPicker').onchange = async e => {
  const files = [...e.target.files];
  e.target.value = '';
  if (!files.length) return;
  const base = (files.find(f => /\.(pth|onnx)$/i.test(f.name)) || files[0]).name.replace(/\.[^.]+$/, '');
  for (const f of files){
    $('#pmsg').textContent = '正在读入 ' + f.name + ' …';
    try {
      const r = await fetch('/api/model/stage?name=' + encodeURIComponent(f.name),
                            {method:'POST', body:f});
      const j = await r.json();
      if (!j.ok){ $('#pmsg').textContent = '读入失败：' + j.error; return; }
    } catch (err) {
      $('#pmsg').textContent = '读入失败：' + err; return;
    }
  }
  $('#pmsg').textContent = '开始导入模型…';
  await fetch('/api/model/import', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name: base})});
  poll();
};

$('#btnDelModel').onclick = async () => {
  const cur = $('#curModel').textContent;
  if (!cur || cur === '无') return;
  if (!confirm('确定删除模型「' + cur + '」？\n会删掉 models 目录里它的 .onnx / .json / .index 文件，不可恢复。')) return;
  const j = await (await fetch('/api/model/delete', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({name:cur})})).json();
  if (j.ok){
    meta = await (await fetch('/api/meta')).json();
    renderModels(meta);
    $('#pmsg').textContent = '已删除「' + cur + '」';
  } else $('#pmsg').textContent = '删除失败：' + j.error;
};

$('#drop').onclick = () => $('#picker').click();
$('#picker').onchange = async e => {
  for (const f of e.target.files) { await upload(f); }
  e.target.value = '';
};
['dragenter','dragover'].forEach(t=>$('#drop').addEventListener(t,e=>{
  e.preventDefault(); $('#drop').classList.add('over');}));
['dragleave','drop'].forEach(t=>$('#drop').addEventListener(t,e=>{
  e.preventDefault(); $('#drop').classList.remove('over');}));
$('#drop').addEventListener('drop', e=>{ const f = e.dataTransfer.files[0]; if(f) upload(f); });

async function upload(file){
  setPicked(''); $('#pmsg').textContent = '正在读入文件…';
  const r = await fetch('/api/upload?name=' + encodeURIComponent(file.name), {method:'POST', body:file});
  const j = await r.json();
  if(j.ok){
    setPicked(j.name);
    const n = j.count || 1;
    $('#qinfo').textContent = n + ' 个文件已加入队列';
    $('#btnClear').style.display = 'inline-flex';
    $('#pmsg').textContent = '已读入，可以开始转换';
  } else { $('#pmsg').textContent = '读入失败：' + j.error; }
}

function setKey(v, fromAuto){
  $('#key').value = v;
  $('#keyval').innerHTML = (v>0?'+':'') + v + '<small>半音</small>';
  $('#keyhint').textContent = v===0 ? '不变调' :
    (v>0 ? '调高 '+v+' 个半音（男声模型唱女声素材常用 +12）'
         : '调低 '+(-v)+' 个半音（女声素材配男声模型常用 -12）');
}
$('#key').oninput = e => { autoKey = null; setKey(+e.target.value, false); };

$('#btnAna').onclick = async () => {
  if(!picked) return;
  $('#btnAna').disabled = true; $('#anaBox').classList.remove('on');
  $('#pmsg').textContent = '正在分析音高…';
  await fetch('/api/analyze', {method:'POST'});
  poll();
};

$('#idxrate').oninput = e => {
  const v = +e.target.value / 100;
  $('#idxval').innerHTML = v.toFixed(2) + '<small>强度</small>';
};

$('#go').onclick = async () => {
  if(!picked) return;
  $('#go').disabled = true; $('#outbox').classList.remove('on'); $('#log').hidden = true;
  $('#bar').style.width='0%'; $('#pct').textContent='';
  const sep = (document.querySelector('input[name=sep]:checked')||{}).value || 'none';
  await fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({key:+$('#key').value, index_rate:+$('#idxrate').value/100, sep,
      device: (document.querySelector('input[name=dev]:checked')||{}).value || 'auto'})});
  poll();
};

$('#reveal').onclick = () => fetch('/api/reveal', {method:'POST'});
$('#btnAbort').onclick = () => fetch('/api/abort', {method:'POST'});
$('#btnClear').onclick = async () => {
  await fetch('/api/clear', {method:'POST'});
  setPicked(''); $('#qinfo').textContent = '还没选文件';
  $('#btnClear').style.display = 'none'; $('#outbox').classList.remove('on');
  $('#pmsg').textContent = '选好文件后点上面按钮';
};
function play(which){
  const a = $('#player');
  a.style.display = 'block';
  a.src = '/api/audio?which=' + which + '&t=' + Date.now();
  a.play().catch(()=>{});
}
$('#playIn').onclick = () => play('in');
$('#playOut').onclick = () => play('out');

async function poll(){
  clearTimeout(timer);
  const s = await (await fetch('/api/state')).json();
  if (s.imported && s.imported !== lastImported){
    lastImported = s.imported;
    meta = await (await fetch('/api/meta')).json();
    renderModels(meta);
    $('#pmsg').textContent = '已导入并切换到「' + s.imported + '」';
  }
  const p = Math.round((s.frac||0)*100);
  $('#bar').style.width = p+'%';
  $('#pct').textContent = p+'%';
  $('#pmsg').textContent = (s.phase ? '['+s.phase+'] ' : '') + (s.msg||'');
  if(s.device_used) $('#devInfo').textContent = '实际设备：' + s.device_used;
  if(s.queue && s.queue.length){
    $('#qinfo').textContent = s.queue.length + ' 个文件已加入队列';
    $('#btnClear').style.display = 'inline-flex';
  }

  if(s.analysis){
    const a = s.analysis, st = a.stats;
    $('#anaBox').classList.add('on');
    $('#aMed').textContent = Math.round(st.median) + ' Hz';
    $('#aP70').textContent = Math.round(st.p70) + ' Hz';
    $('#aP95').textContent = Math.round(st.p95) + ' Hz';
    $('#aKey').textContent = (a.key>0?'+':'') + a.key;
    $('#aWhy').textContent = a.reason;
    if(autoKey !== a.key){ autoKey = a.key; setKey(a.key, true); }
  }

  if(['analyzing','separating','loading','running','importing'].includes(s.stage)){
    timer = setTimeout(poll, 400);
    return;
  }
  if(s.stage==='done'){
    $('#outbox').classList.add('on');
    $('#outpath').textContent = s.output;
    $('#go').disabled = !picked;
  } else if(s.stage==='error'){
    $('#pmsg').textContent = s.error;
    $('#go').disabled = !picked;
  } else {
    $('#go').disabled = !picked;
  }
  $('#btnAna').disabled = !picked;
  if(s.log && s.log.length){
    const box = $('#log'); box.hidden = false;
    box.textContent = s.log.slice(-40).join('\n');
    box.scrollTop = box.scrollHeight;
  }
}

loadMeta().then(poll);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RVCCpuPlus/1.0"

    def log_message(self, fmt, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            data = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif u.path == "/api/state":
            self._json(get_state())
        elif u.path == "/api/audio":
            q = parse_qs(u.query)
            which = (q.get("which") or ["out"])[0]
            idx = int((q.get("i") or ["0"])[0] or 0)
            st = get_state()
            if which == "in":
                p = st.get("input") or ""
            else:
                outs = st.get("outputs") or ([st["output"]] if st.get("output") else [])
                p = outs[idx] if 0 <= idx < len(outs) else (outs[-1] if outs else "")
            if not p or not os.path.isfile(p):
                return self._json({"ok": False, "error": "还没有可试听的文件"}, 404)
            data = open(p, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        elif u.path == "/api/meta":
            try:
                self._json(meta())
            except Exception as e:
                self._json({"version": VERSION, "error": str(e)})
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)

        if u.path == "/api/upload":
            name = unquote(parse_qs(u.query).get("name", ["in.wav"])[0])
            try:
                if any(0x80 <= ord(c) <= 0xFF for c in name):
                    name = name.encode("latin-1").decode("utf-8")
            except Exception:
                pass
            if n <= 0:
                return self._json({"ok": False, "error": "文件是空的"})
            if n > 400 * 1024 * 1024:
                return self._json({"ok": False, "error": "文件超过 400MB"})
            os.makedirs(WORK, exist_ok=True)
            safe = re.sub(r"[\\/:*?\"<>|]", "_", os.path.basename(name))
            dst = os.path.join(WORK, safe)
            left = n
            with open(dst, "wb") as f:
                while left > 0:
                    buf = self.rfile.read(min(1 << 20, left))
                    if not buf:
                        break
                    f.write(buf)
                    left -= len(buf)
            q = [x for x in (get_state().get("queue") or []) if os.path.isfile(x)]
            q.append(dst)

            # 入库前先确认是能读的音频：否则要等到转换时才炸，
            # 一个坏文件就能把整批队列拖垮（提示也很晚）。
            try:
                import soundfile as _sf
                _info = _sf.info(dst)
                if not _info.frames or _info.samplerate <= 0:
                    raise ValueError("音频是空的")
            except Exception as _e:
                try:
                    q.remove(dst)
                    os.remove(dst)
                except Exception:
                    pass
                return self._json({"ok": False,
                                   "error": "不是能识别的音频文件（%s）。"
                                            "支持 wav / flac / ogg / aiff / mp3，"
                                            "m4a 请先转成 wav"
                                            % str(_e).split(":")[-1].strip()[:60]})

            set_state(input=dst, queue=q, output="", error="", analysis=None,
                      outputs=[])
            return self._json({"ok": True, "name": safe, "count": len(q)})

        # 模型文件暂存上传：必须放在下面通用读 body 之前，自己读流。
        # （放在后面的复选框里再读一次 rfile 会永久阻塞：keep-alive 连接不会 EOF）
        if u.path == "/api/model/stage":
            name = unquote(parse_qs(u.query).get("name", ["model.bin"])[0])
            try:
                if any(0x80 <= ord(c) <= 0xFF for c in name):
                    name = name.encode("latin-1").decode("utf-8")
            except Exception:
                pass
            if n <= 0:
                return self._json({"ok": False, "error": "文件是空的"})
            if n > 3 * 1024 * 1024 * 1024:
                return self._json({"ok": False, "error": "文件超过 3GB"})
            os.makedirs(MODEL_STAGE, exist_ok=True)
            safe = re.sub(r"[\\/:*?\"<>|]", "_", os.path.basename(name))
            dst = os.path.join(MODEL_STAGE, safe)
            left = n
            with open(dst, "wb") as f:
                while left > 0:
                    buf = self.rfile.read(min(1 << 20, left))
                    if not buf:
                        break
                    f.write(buf)
                    left -= len(buf)
            staged = [x for x in os.listdir(MODEL_STAGE)
                      if os.path.isfile(os.path.join(MODEL_STAGE, x))]
            log("暂存模型文件：%s（%.1f MB）" % (safe, n / 1024 ** 2))
            return self._json({"ok": True, "name": safe, "staged": staged})

        body = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(body or b"{}")
        except Exception:
            payload = {}

        if u.path == "/api/analyze":
            p = get_state()["input"]
            if not p or not os.path.isfile(p):
                return self._json({"ok": False, "error": "还没有选文件"})
            if get_state()["stage"] in ("running", "loading", "separating", "analyzing", "importing"):
                return self._json({"ok": False, "error": "正在忙，等一下"})
            set_state(stage="analyzing", frac=0.05, msg="开始分析…", error="")
            threading.Thread(target=run_analyze, args=(p,), daemon=True).start()
            return self._json({"ok": True})

        if u.path == "/api/start":
            st = get_state()
            p = st["input"]
            if not p or not os.path.isfile(p):
                return self._json({"ok": False, "error": "还没有选文件"})
            if st["stage"] in ("running", "loading", "separating", "analyzing", "importing"):
                return self._json({"ok": False, "error": "正在忙，等一下"})
            key = int(payload.get("key", 0))
            rate = float(payload.get("index_rate", 0.0))
            sep_mode = str(payload.get("sep", "none"))
            device = str(payload.get("device", "auto"))
            global _abort
            _abort = False
            files = [x for x in (st.get("queue") or [p]) if os.path.isfile(x)]
            if not files:
                files = [p]
            set_state(stage="loading", frac=0.01, msg="准备中…", error="", output="",
                      outputs=[], phase="", total=len(files))
            threading.Thread(target=run_job, args=(files, key, rate, sep_mode, device),
                             daemon=True).start()
            return self._json({"ok": True, "files": len(files)})

        if u.path == "/api/abort":
            _abort = True
            log("收到取消请求，当前块跑完就停")
            return self._json({"ok": True})

        if u.path == "/api/clear":
            set_state(queue=[], input="", analysis=None, outputs=[], output="")
            return self._json({"ok": True})

        if u.path == "/api/reveal":
            p = get_state().get("output") or OUT
            try:
                if os.path.isfile(p):
                    subprocess.Popen(["explorer", "/select,", os.path.normpath(p)])
                else:
                    os.makedirs(OUT, exist_ok=True)
                    os.startfile(OUT)
                return self._json({"ok": True})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)})

        # ---------------------------------------------------------------- 模型管理
        if u.path == "/api/model/import":
            st = get_state()
            if st["stage"] in ("running", "loading", "separating", "analyzing", "importing"):
                return self._json({"ok": False, "error": "正在忙，等一下"})
            if not os.path.isdir(MODEL_STAGE) or not os.listdir(MODEL_STAGE):
                return self._json({"ok": False, "error": "先选择要导入的模型文件"})
            nm = str(payload.get("name") or "").strip()
            set_state(stage="importing", frac=0.02, msg="准备导入…",
                      error="", imported="")
            threading.Thread(target=run_import, args=(nm,), daemon=True).start()
            return self._json({"ok": True})

        if u.path == "/api/model/select":
            # 参数优先取 JSON body，其次取 query（两种写法都接受，方便脚本调用）
            nm = str(payload.get("name")
                     or parse_qs(u.query).get("name", [""])[0] or "").strip()
            try:
                from rvc_cpu import models as mm
                r = mm.set_active(nm)
                invalidate()
                log("切换模型：" + nm)
                return self._json({"ok": True, "active": r["active"]})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)})

        if u.path == "/api/model/delete":
            nm = str(payload.get("name")
                     or parse_qs(u.query).get("name", [""])[0] or "").strip()
            try:
                from rvc_cpu import models as mm
                r = mm.delete_model(nm)
                invalidate()
                log("删除模型：%s（%s）" % (nm, ", ".join(r["removed"])))
                return self._json({"ok": True, "removed": r["removed"]})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)})

        self._json({"ok": False, "error": "not found"}, 404)


# ---------------------------------------------------------------- 启动
def port_in_use(port):
    import socket
    s = socket.socket()
    s.settimeout(0.4)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def find_edge():
    cands = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Microsoft\Edge\Application\msedge.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return ""


def open_window(url):
    exe = find_edge()
    if exe:
        profile = os.path.join(WORK, "_窗口")
        os.makedirs(profile, exist_ok=True)
        try:
            subprocess.Popen([exe, "--app=" + url, "--window-size=1160,900",
                              "--user-data-dir=" + profile,
                              "--no-first-run", "--no-default-browser-check"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass
    webbrowser.open(url)


def main():
    url = "http://127.0.0.1:%d/" % PORT
    if port_in_use(PORT):
        print("界面已在运行，直接打开窗口…")
        open_window(url)
        time.sleep(1.5)
        return

    gen, vec, rmv, idx = model_paths()
    from rvc_cpu import models as mm
    ms, active = mm.list_models()
    print("=" * 60)
    print(" 声音转换 %s" % VERSION)
    print(" 模型目录：" + MODELS)
    for n, p in (("生成器", gen), ("ContentVec", vec), ("RMVPE", rmv), ("检索索引", idx)):
        print("   %-10s %s" % (n, os.path.basename(p) if p else "无"))
    if len(ms) > 1:
        print(" 音色模型 %d 个（当前：%s）：%s" %
              (len(ms), active, "、".join(m["name"] for m in ms)))
    print(" 输出目录：" + OUT)
    print(" 界面地址：" + url)
    print("=" * 60)
    print(" 关掉这个黑窗口 = 退出。转换过程中别关。")
    sys.stdout.flush()

    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    if os.environ.get("RVC_GUI_NOOPEN") != "1":
        threading.Timer(0.8, lambda: open_window(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

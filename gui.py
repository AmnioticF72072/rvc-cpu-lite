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
sys.path.insert(0, ROOT)

# v2.0：状态与管线生命周期移到 app/core.py。
# 这里保留同名引用，是为了让原有路由/任务代码一行都不用改（零回归迁移）。
from app import core as _core                                  # noqa: E402
from app.core import (VERSION, ensure_pipeline, get_state, log,      # noqa: E402,F401
                      model_paths, set_state)

_lock = _core._lock
_abort = False
_meta_cache = None      # gui 自己的 meta 缓存（字段名与 app.core 不同，不能共用）
_meta_cache_at = 0.0


def invalidate():                                              # noqa: F811
    """兼容旧调用点：模型变了要同时丢掉管线与音高分析。"""
    global _meta_cache, _meta_cache_at
    _meta_cache = None
    _meta_cache_at = 0.0
    _core.invalidate()


# ---------------------------------------------------------------- 资源发现
def _has_torch():
    try:
        from rvc_cpu import models as mm
        return bool(mm.find_torch_python())
    except Exception:
        return False


def _has_faiss():
    try:
        from rvc_cpu import has_index_backend
        return bool(has_index_backend())
    except Exception:
        return False


def meta():
    """给界面用的资源清单。

    缓存 3 秒——纯粹为了挡住轮询的重复扫盘。不设更长是因为：
    用户可能自己往 models 目录里放模型文件（不走导入流程），
    缓存太久会导致"文件明明放进去了，列表里没有"。
    """
    global _meta_cache, _meta_cache_at
    if _meta_cache is not None and (time.time() - _meta_cache_at) < 3.0:
        return _meta_cache
    from rvc_cpu import models as mm
    gen, vec, rmv, idx = model_paths()
    # 附加信息（分离引擎 / 硬件探测）失败绝不能拖垮整个 meta ——
    # 它们只是给界面做提示用的，缺了顶多少显示一行，
    # 但 meta 挂掉会让模型列表整个消失，那才是致命的。
    try:
        from rvc_cpu import separate as sep
        probe = sep.probe()
    except Exception as _e:
        probe = {"ready_karaoke": False, "ready_vocals": False,
                 "karaoke_models": [], "python": "", "backends": [],
                 "bundled": False, "error": str(_e)[:80]}
    try:
        from rvc_cpu.hw_profile import pick_profile as _pp
        _hw = _pp()
    except Exception:
        _hw = {}
    index_info = None
    # 双重保险：即使上游给了不存在的路径，也不能让整个 meta 挂掉。
    # meta 挂了 = 界面拿不到模型列表 = 用户看到"什么都选不了"。
    if idx and os.path.isfile(idx):
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
        # 特征检索后端（faiss）是否可用 —— 这个不做的话 index_rate 完全无效，
        # 而 index 正是决定"像不像目标角色"的头号因素，必须让用户看见
        "index_backend": _has_faiss(),
        # .pth 导入需要 torch；没有的话界面要提前说，而不是等用户点了才失败
        "torch_backend": bool(_has_torch()),
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
            # 是不是内置引擎：界面要明确告诉用户"不用另外装东西"
            "bundled": probe.get("bundled", False),
            "engine": "内置分离引擎" if probe.get("bundled") else "",
            # 去和声组件装没装 —— 界面据此区分"缺组件"和"缺模型"
            "has_separator": bool(probe.get("has_separator")),
        },
        # 机器能力：低端机要提前告知，不能让用户干等半小时还以为卡死了
        "hw": {
            "cpu": _hw.get("cpu", ""),
            "logical": _hw.get("logical", 0),
            "threads": _hw.get("intra_threads", 0),
            "profile": _hw.get("profile", ""),
            "lowend": bool(_hw.get("lowend")),
            "avx2_risk": _hw.get("avx2_risk", ""),
            "avx2_note": _hw.get("avx2_note", ""),
            "speed_note": _hw.get("speed_note", ""),
        },
    }
    _meta_cache_at = time.time()
    return _meta_cache


# ---------------------------------------------------------------- 任务
def run_analyze(path):
    try:
        from rvc_cpu.calibrate import analyze_file, load_profile, suggest
        from rvc_cpu import models as mm
        from rvc_cpu.ultra_lite import segment_curve

        _, _, rmv, _ = model_paths()
        set_state(stage="analyzing", frac=0.05, msg="分析音高（第一次约 5~10 秒）…")
        stats, f0 = analyze_file(rmv, path, return_f0=True,
                                 progress=lambda f, m: set_state(
                                     stage="analyzing", frac=0.05 + 0.9 * f, msg=m))
        if not stats:
            set_state(stage="idle", frac=0, analysis=None,
                      msg="没检测到人声，这段可能是静音或纯伴奏")
            return
        prof = load_profile(MODELS, mm.active_name() or "myvoice51")
        r = suggest(stats, prof)
        # 分段曲线：直接复用刚才那份 RMVPE 结果，不再跑第二遍
        segs = segment_curve(f0, sr_f0=50.0, win_sec=10.0)
        segs = [s for s in segs if not s.get("silence")]
        set_state(stage="idle", frac=0,
                  analysis={"stats": stats, "key": r["key"], "reason": r["reason"],
                            "shifted": r.get("shifted", {}),
                            "segments": segs,
                            "profile": {"name": prof.get("name", ""),
                                        "lo": prof["target_lo"], "hi": prof["target_hi"],
                                        "p95": prof["p95_max"]}},
                  msg="分析完成：建议 %+d 半音" % r["key"])
        log("音高分析：中位 %.0f / 70 分位 %.0f / 95 分位 %.0f → 建议 %+d（%d 段）" % (
            stats["median"], stats["p70"], stats["p95"], r["key"], len(segs)))
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
                log("完成人声：" + os.path.basename(out_path))

                # 分人声模式下，把换好的人声和原伴奏合回去 ——
                # 这才是翻唱：伴奏保留原样，只换主唱。
                # v2.0.x 之前只输出人声、把伴奏丢了，听起来就是"伴奏没了"。
                if sep_mode == "vocals":
                    acc = getattr(sep, "LAST_ACCOMP", "") if sep else ""
                    if acc and os.path.isfile(acc):
                        try:
                            from rvc_cpu.mixdown import mix
                            set_state(msg="正在和原伴奏合轨…", frac=0.97)
                            mixed = out_path[:-4] + "_混音.wav"
                            _, mst = mix(out_path, acc, mixed)
                            log("合轨完成：%s（人声 %.1f dB / 伴奏 %.1f dB，峰值 %.2f）"
                                % (os.path.basename(mixed), mst["vocal_rms_db"],
                                   mst["acc_rms_db"], mst["peak"]))
                            out_path = mixed
                        except Exception as me:
                            log("合轨失败（人声已单独保存）：%s" % me)
                    else:
                        log("没找到伴奏轨，只输出人声。"
                            "（伴奏由 demucs 产出；若这台机器没有 demucs 就没有伴奏可合）")

                outs.append(out_path)
                log("输出：" + os.path.basename(out_path))

                # 质量自检：单文件必跑；批量只跑最后一个（全跑要多等一分多钟）
                if total == 1 or i == total - 1:
                    set_state(msg="正在自检输出质量…")
                    try:
                        import soundfile as _sf
                        from rvc_cpu.ultra_lite import qa_scan
                        _x, _sr = _sf.read(out_path, dtype="float32", always_2d=True)
                        rep = qa_scan(_x.mean(1), _sr)
                        rep["file"] = os.path.basename(out_path)
                        set_state(qa=rep)
                        log("自检 %d 分｜谐波度 %.2f｜低频残留 %.1f dB%s"
                            % (rep["score"], rep["harmonicity"], rep["low_residual_db"],
                               "｜" + rep["tips"][0] if rep["tips"] else ""))
                    except Exception as _e:
                        set_state(qa=None)
                        log("自检跳过：" + str(_e)[:70])
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
  /* 队列列表：拖错了能单独删掉一个 */
  .qlist{display:none;flex-direction:column;gap:5px;margin-top:9px}
  .qlist.on{display:flex}
  .qlist .qi{display:flex;align-items:center;gap:8px;padding:6px 10px;
    border-radius:9px;background:rgba(10,132,255,.045);
    border:1px solid rgba(10,132,255,.1);font-size:12px;color:#3a4c66}
  .qlist .qi span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .qlist .qi b{font-weight:700;color:#0e2a4d;flex:none;font-size:11px}
  .qlist .qi button{border:none;background:transparent;color:#9aa8ba;cursor:pointer;
    font-size:15px;line-height:1;padding:2px 4px;border-radius:5px;flex:none}
  .qlist .qi button:hover{background:rgba(224,90,104,.12);color:#c9434f}
  .qlist .more{font-size:11.5px;color:#9aa8ba;padding-left:10px}
  /* 分段音高条：一眼看出哪段偏高会破 */
  .segs{display:flex;align-items:flex-end;gap:2px;height:44px;margin-top:12px;
    padding:0 1px;border-bottom:1px solid rgba(10,132,255,.18)}
  .segs i{flex:1;min-width:2px;border-radius:2px 2px 0 0;background:#7fb4f0;
    transition:background .2s}
  .segs i.hi{background:#f0a04b}
  .segs i.over{background:#e05a68}
  .segs-legend{display:flex;gap:12px;margin-top:7px;font-size:11px;color:#7a8899}
  .segs-legend em{font-style:normal;display:inline-flex;align-items:center;gap:5px}
  .segs-legend em::before{content:"";width:8px;height:8px;border-radius:2px;
    background:#7fb4f0}
  .segs-legend em.hi::before{background:#f0a04b}
  .segs-legend em.over::before{background:#e05a68}

  /* 输出质量自检 */
  .qa{display:none;margin-top:14px;padding:14px 15px;border-radius:16px;
    background:rgba(248,250,253,.9);border:1px solid rgba(10,132,255,.13)}
  .qa.on{display:block}
  .qa .top{display:flex;align-items:baseline;gap:10px;margin-bottom:9px}
  .qa .top b{font:800 26px/1 Consolas,monospace;color:#0e2a4d}
  .qa .top span{font-size:12px;color:#5d6c82}
  .qa .top em{font-style:normal;margin-left:auto;font-size:11.5px;font-weight:700;
    padding:4px 9px;border-radius:7px;background:rgba(14,158,110,.12);color:#0e9e6e}
  .qa .top em.warn{background:rgba(224,90,104,.12);color:#c9434f}
  .qa .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:7px 14px}
  .qa .grid div{font-size:11.5px;color:#5d6c82;display:flex;justify-content:space-between;
    padding:5px 0;border-bottom:1px dashed rgba(10,132,255,.1)}
  .qa .grid b{font:700 12.5px Consolas,monospace;color:#0e2a4d}
  .qa .tips{margin-top:9px;font-size:12px;color:#a56a1e;line-height:1.6}
  .qa .tips div{padding-left:14px;position:relative}
  .qa .tips div::before{content:"·";position:absolute;left:4px}

  .out{display:none;margin-top:14px;padding:14px 15px;border-radius:16px;
    background:rgba(10,150,95,.07);border:1px solid rgba(10,150,95,.18)}
  .out.on{display:block}
  .out b{font-size:13.5px;color:#0a7a45}
  .out code{display:block;font-size:11.6px;color:#4c5c72;word-break:break-all;margin:6px 0 10px;
    font-family:ui-monospace,Consolas,monospace}

  /* ---------- 自定义播放器 ----------
     浏览器原生 <audio> 在这个卡片布局里只有一条细线，很多人根本看不出
     那是能点的播放器，以为只有"打开文件夹"。所以自绘一套：
     波形可点跳转、大播放键、时间、音量、循环、下载。 */
  .pl{margin:12px 0 4px;padding:14px 15px 12px;border-radius:15px;
    background:linear-gradient(180deg,#FFFFFF,#F7FAFF);
    border:1px solid #DCE6F3;
    box-shadow:0 2px 4px rgba(16,38,72,.05),0 16px 34px -20px rgba(16,38,72,.22)}
  .pl-main{display:flex;align-items:center;gap:14px}
  .pl-btn{flex:none;width:52px;height:52px;border-radius:50%;border:none;cursor:pointer;
    background:linear-gradient(150deg,#0B63F6,#4E9BFF);color:#fff;
    display:grid;place-items:center;transition:transform .15s,box-shadow .15s;
    box-shadow:0 6px 16px rgba(11,99,246,.32)}
  .pl-btn:hover{transform:translateY(-1px);box-shadow:0 9px 22px rgba(11,99,246,.4)}
  .pl-btn:active{transform:scale(.96)}
  .pl-btn svg{width:21px;height:21px;fill:#fff}
  .pl-btn.playing{background:linear-gradient(150deg,#0E9E6E,#38C795);
    box-shadow:0 6px 16px rgba(14,158,110,.3)}
  .pl-wave{flex:1;min-width:0;position:relative;height:56px;cursor:pointer;
    display:block;width:100%;border-radius:10px}
  .pl-wave:hover{background:rgba(11,99,246,.04)}
  .pl-meta{display:flex;justify-content:space-between;align-items:center;
    margin-top:7px;font-size:12.2px;color:#6B7A8D;
    font-family:ui-monospace,Consolas,monospace}
  .pl-meta .st{color:#0B63F6;font-weight:700}
  .pl-tools{display:flex;align-items:center;gap:12px;margin-top:11px;
    padding-top:11px;border-top:1px dashed #E3EAF4;flex-wrap:wrap}
  .pl-tools button{background:none;border:1px solid #DCE6F3;border-radius:9px;
    padding:6px 11px;font-size:12.3px;color:#3B4A60;cursor:pointer;
    display:inline-flex;align-items:center;gap:5px;font-family:inherit}
  .pl-tools button:hover{border-color:#9FC2F5;color:#0B63F6;background:#F4F9FF}
  .pl-tools button.on{border-color:#0E9E6E;color:#0E9E6E;background:rgba(14,158,110,.07)}
  .pl-vol{display:flex;align-items:center;gap:7px;font-size:12.3px;color:#6B7A8D}
  .pl-vol input{width:78px;accent-color:#0B63F6}
  .pl-empty{font-size:12.6px;color:#8b97a8;text-align:center;padding:16px 0}

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
  /* ---------- v2.0 双模式与朗读面板 ---------- */
  .modes{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0 14px}
  .mode{display:flex;flex-direction:column;gap:3px;align-items:flex-start;
    padding:15px 18px;border-radius:18px;cursor:pointer;text-align:left;
    background:rgba(255,255,255,.62);border:1px solid rgba(255,255,255,.9);
    box-shadow:0 8px 22px -14px rgba(20,40,80,.28);
    backdrop-filter:blur(18px);transition:.2s}
  .mode b{font-size:15.5px;color:#0e2a4d;font-weight:700}
  .mode span{font-size:12px;color:#7a8899}
  .mode:hover{background:rgba(255,255,255,.82);transform:translateY(-1px)}
  .mode.on{background:linear-gradient(180deg,#2b93ff,#0a84ff);
    border-color:transparent;box-shadow:0 12px 28px -10px rgba(10,132,255,.55)}
  .mode.on b{color:#fff}
  .mode.on span{color:rgba(255,255,255,.82)}

  .rdtext{width:100%;padding:14px 15px;border-radius:14px;resize:vertical;
    font-family:inherit;font-size:14.5px;line-height:1.75;color:#0e2a4d;
    background:rgba(255,255,255,.85);border:1px solid rgba(10,132,255,.18);
    outline:none;transition:.18s;min-height:120px}
  .rdtext:focus{border-color:#0a84ff;background:#fff;
    box-shadow:0 0 0 3px rgba(10,132,255,.1)}
  .rdtext::placeholder{color:#a9b6c6}

  .rdselect{width:100%;padding:11px 13px;border-radius:12px;font-size:14px;
    font-family:inherit;color:#0e2a4d;background:rgba(255,255,255,.9);
    border:1px solid rgba(10,132,255,.18);outline:none;cursor:pointer}
  .rdselect:focus{border-color:#0a84ff}

  .rdauto{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;
    color:#41526b;cursor:pointer;user-select:none}
  .rdauto input{width:15px;height:15px;accent-color:#0a84ff;cursor:pointer}

  /* 情绪九宫格 */
  #segEmo{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
  #segEmo label{margin:0}
  #segEmo label span{font-size:13px;padding:9px 4px}
  /* 流程卡左侧竖条：一眼看出这是第几步 */
  .card{position:relative}
  .card:has(.step)::before{content:"";position:absolute;left:0;top:20px;bottom:20px;
    width:3px;border-radius:0 3px 3px 0;
    background:linear-gradient(180deg,#0a84ff,#6fb0ff 52%,rgba(111,176,255,0))}
  /* 序号：加一层内高光，不再是纯色块 */
  .num{box-shadow:0 2px 6px rgba(10,132,255,.26),inset 0 1px 0 rgba(255,255,255,.45)}
  .step h2{letter-spacing:-.01em}
  /* 说明文字再压一档，与标题拉开层级 */
  .hint{color:#8a97a9}
  /* 主操作按钮：渐变 + 悬停微上浮；禁用态保留形状只降饱和 */
  #go{background:linear-gradient(180deg,#2b93ff,#0a84ff);
    box-shadow:0 8px 22px -6px rgba(10,132,255,.5),inset 0 1px 0 rgba(255,255,255,.32);
    transition:transform .18s ease,box-shadow .18s ease,background .18s ease}
  #go:hover:not(:disabled){transform:translateY(-1px);
    box-shadow:0 12px 28px -8px rgba(10,132,255,.58),inset 0 1px 0 rgba(255,255,255,.4)}
  #go:active:not(:disabled){transform:translateY(0) scale(.997)}
  #go:disabled{background:linear-gradient(180deg,#dae3ef,#cbd6e4);
    box-shadow:0 4px 12px -8px rgba(20,40,80,.2),inset 0 1px 0 rgba(255,255,255,.55)}
  /* 侧栏标题与正文层级 */
  .side .card h3{letter-spacing:-.01em}
  /* 结果区标题与路径更整齐 */
  .out b{letter-spacing:-.01em}
  .out code{letter-spacing:0}
</style></head><body>
<div class="wrap">
  <div class="top">
    <div class="logo"><i></i><i></i><i></i></div>
    <div>
      <h1>声音转换 <em>PLUS</em></h1>
      <div class="sub">RVC 纯 CPU · 自动校准 · 去和声 · 特征检索 · 全程离线</div>
    </div>
    <div class="badge" id="badge">正在读取模型…</div>
    <!-- 机器能力提示：低端双核 / 缺 AVX2 时才会出现。
         不给提示的话，用户会以为程序卡死了。 -->
    <div id="hwNote" style="display:none;margin-top:10px;padding:11px 14px;
      border-radius:12px;font-size:12.8px;line-height:1.6;
      background:rgba(217,138,43,.09);border:1px solid rgba(217,138,43,.28);
      color:#8a5a12"></div>
  </div>

  <!-- v2.0 双模式：翻唱用现成音频，朗读从文字开始 -->
  <div class="modes" id="modes">
    <button class="mode on" data-m="sing">
      <b>翻唱 / 变声</b><span>已有音频，换成目标音色</span>
    </button>
    <button class="mode" data-m="read">
      <b>朗读（文字）</b><span>打一段字，直接念出来</span>
    </button>
  </div>

  <div class="grid" id="gridSing">
    <div>
      <!-- 0 音色模型 -->
      <div class="card">
        <div class="step"><div class="num">0</div><h2>音色模型</h2>
          <span class="opt">可导入别人的模型</span></div>
        <p class="hint">当前用的是「<b id="curModel">—</b>」。可以导入其他人分享的 RVC 模型：
          拖入 <b>.pth</b>（本机有 PyTorch 就自动转换）或 <b>.onnx</b> 都行，
          有配套 .index 一起放，音色会更像。<br>
          <span style="color:#9aa8ba">也可以直接把 <b>.onnx</b> 拷进程序目录的 <b>models</b>
          文件夹，然后点「刷新列表」——这一步不需要转换，最省事。</span></p>
        <div class="seg" id="segModel"></div>
        <div class="row" style="margin-top:10px;gap:10px">
          <button class="btn mini ghost" id="btnImport">导入模型…</button>
          <button class="btn mini ghost" id="btnRefreshModel">刷新列表</button>
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
        <div class="qlist" id="qlist"></div>
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
          <div class="segs" id="aSegs"></div>
          <div class="segs-legend" id="aSegLegend" style="display:none">
            <em>正常</em><em class="hi">偏高</em><em class="over">会破音</em>
          </div>
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

          <div class="pl" id="sPl">
            <div class="pl-main">
              <button class="pl-btn" id="sPlBtn" title="播放 / 暂停">
                <svg viewBox="0 0 24 24" id="sPlIcon"><path d="M8 5v14l11-7z"/></svg>
              </button>
              <canvas class="pl-wave" id="sPlWave" height="56"></canvas>
            </div>
            <div class="pl-meta">
              <span id="sPlTime">0:00 / 0:00</span>
              <span id="sPlSt" class="st"></span>
            </div>
            <div class="pl-tools">
              <button id="sPlA">A 原声</button>
              <button id="sPlB" class="on">B 结果</button>
              <div class="pl-vol">
                <span>音量</span>
                <input type="range" id="sPlVol" min="0" max="100" value="100">
              </div>
              <button id="sPlLoop">🔁 循环</button>
              <button id="sPlDl">⤓ 下载</button>
            </div>
          </div>
          <audio id="player" preload="auto"></audio>
          <div class="qa" id="qaBox">
            <div class="top"><b id="qaScore">—</b><span>质量自检</span>
              <em id="qaTag">—</em></div>
            <div class="grid" id="qaGrid"></div>
            <div class="tips" id="qaTips"></div>
          </div>
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

  <!-- ============ v2.0 朗读面板（模式切到「朗读」时显示）============ -->
  <div class="grid" id="gridRead" style="display:none">
    <div>
      <div class="card">
        <div class="step"><div class="num">1</div><h2>要念的文字</h2>
          <span class="opt">打字就行，不用录音</span></div>
        <p class="hint">按句自动切分，句号/问号/感叹号都算一句。单次上限 5000 字。</p>
        <textarea id="rdText" class="rdtext" rows="7"
          placeholder="在这里打一段话，比如：&#10;大家好，我是用这个声音模型说话的。&#10;现在你听到的每一个字，都是从文字直接生成的。"></textarea>
        <div class="row" style="margin-top:8px">
          <span id="rdCount" style="font-size:12.3px;color:#7a8899">0 字</span>
          <span id="rdEst" style="font-size:12.3px;color:#9aa8ba"></span>
        </div>
      </div>

      <div class="card">
        <div class="step"><div class="num">2</div><h2>念成什么情绪</h2>
          <span class="opt">影响语调，不影响音色</span></div>
        <p class="hint">情绪作用在「念」的语调上，之后会换成你选的音色。音色本身由第 4 步决定。</p>
        <div class="seg" id="segEmo"></div>
      </div>

      <div class="card">
        <div class="step"><div class="num">3</div><h2>载体声音</h2>
          <span class="opt">只决定节奏与语调</span></div>
        <p class="hint">这一层的声音最后会被替换掉。选同性别的载体，语调更自然。</p>
        <select id="rdVoice" class="rdselect"></select>
        <div class="row" style="margin-top:10px">
          <button class="btn mini ghost" id="rdPreview">试听语调（不变声）</button>
          <span id="rdPrevMsg" style="font-size:12px;color:#7a8899;text-align:right"></span>
        </div>
      </div>

      <div class="card">
        <div class="step"><div class="num">4</div><h2>念成谁的声音</h2>
          <span class="opt">音色由模型决定</span></div>
        <p class="hint">用上面「音色模型」里选中的模型。要像某个角色说话，就导入那个角色的模型。</p>
        <div class="row">
          <div class="keyval" id="rdKeyVal">自动<small>移调</small></div>
          <label class="rdauto"><input type="checkbox" id="rdKeyAuto" checked>
            <span>自动定移调</span></label>
        </div>
        <input type="range" id="rdKey" min="-12" max="12" step="1" value="0" disabled>
        <div class="keyscale"><span>-12 低</span><span>0</span><span>+12 高</span></div>

        <div style="margin-top:18px;padding-top:16px;border-top:1px solid rgba(10,132,255,.1)">
          <div class="row">
            <div class="keyval" id="rdIdxVal">0.75<small>贴合度</small></div>
            <span id="rdIdxHint" style="font-size:12px;color:#7a8899;text-align:right">
              有 .index 才生效</span>
          </div>
          <input type="range" id="rdIdx" min="0" max="100" step="5" value="75">
          <div class="keyscale"><span>0 像载体的音</span><span>0.75 推荐</span><span>1.0 最贴模型</span></div>
        </div>
      </div>

      <div class="card">
        <div class="step"><div class="num">5</div><h2>开始朗读</h2></div>
        <button class="btn" id="rdGo">念出来</button>
        <div class="prog">
          <div class="bar"><i id="rdBar"></i></div>
          <div class="pmsg"><span id="rdMsg">填好文字后点上面按钮</span><span id="rdPct"></span></div>
        </div>
        <div class="out" id="rdOut">
          <b>念好了</b>
          <code id="rdOutPath"></code>

          <!-- 自定义播放器：波形 + 播放键 + 时间 + 音量 + 循环 + 下载 -->
          <div class="pl" id="rdPl">
            <div class="pl-main">
              <button class="pl-btn" id="rdPlBtn" title="播放 / 暂停">
                <svg viewBox="0 0 24 24" id="rdPlIcon"><path d="M8 5v14l11-7z"/></svg>
              </button>
              <canvas class="pl-wave" id="rdPlWave" height="56"></canvas>
            </div>
            <div class="pl-meta">
              <span id="rdPlTime">0:00 / 0:00</span>
              <span id="rdPlSt" class="st"></span>
            </div>
            <div class="pl-tools">
              <div class="pl-vol">
                <span>音量</span>
                <input type="range" id="rdPlVol" min="0" max="100" value="100">
              </div>
              <button id="rdPlLoop">🔁 循环</button>
              <button id="rdPlDl">⤓ 下载</button>
            </div>
          </div>
          <audio id="rdResult" preload="auto"></audio>
          <div class="qa" id="rdQa">
            <div class="top"><b id="rdQaScore">—</b><span>质量自检</span>
              <em id="rdQaTag">—</em></div>
            <div class="grid" id="rdQaGrid"></div>
          </div>
          <button class="btn mini ghost" id="rdReveal">打开输出文件夹</button>
        </div>
      </div>
    </div>

    <div class="side">
      <div class="card">
        <h3>朗读和翻唱的区别</h3>
        <ul class="how">
          <li><b>1</b><span>翻唱是「已经有一段录音」，把它换成目标音色。</span></li>
          <li><b>2</b><span>朗读是「还没有录音」，从文字直接生成，再换音色。</span></li>
          <li><b>3</b><span>两者共用同一个模型、同一套质检，只是输入来源不同。</span></li>
        </ul>
      </div>
      <div class="card">
        <h3>速度说明</h3>
        <p class="hint" style="margin-bottom:0">
          纯 CPU 处理，不是麦克风级实时。<br>
          实测：3 秒的话约 12 秒出结果，15 秒约 30 秒。<br>
          所以适合「写完再念」，不适合边说边出。
        </p>
      </div>
      <div class="card">
        <h3>常见问题</h3>
        <details><summary>为什么要联网？</summary>
          <p>文字转音节这一步用的是在线语音合成（免费、不要密钥）。
             合成完立刻变声，音频不上传到别处。断网时这一步会失败，翻唱功能不受影响。</p></details>
        <details><summary>念出来语气平怎么办？</summary>
          <p>换个情绪（欢快/激动/悲伤差别很明显），或者在文字里加标点控制停顿。
             换更口语化的写法也比书面语自然。</p></details>
      </div>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let picked = null, timer = null, meta = null, autoKey = null, lastImported = '', _wasImporting = false;

function setPicked(name){
  picked = name;
  $('#fname').textContent = name || '—';
  $('#filebox').classList.toggle('on', !!name);
  $('#go').disabled = !name;
  $('#btnAna').disabled = !name;
}

async function loadMeta(){
  meta = await (await fetch('/api/meta')).json();
  _hasIdxBackend = (meta.index_backend !== false);
  renderModels(meta);
  const s = meta.sep || {};
  const parts = [];
  parts.push(meta.version);
  if (meta.active_model) parts.push(meta.active_model);
  parts.push(meta.index ? '检索已启用' : '无检索索引');
  $('#badge').textContent = parts.join(' · ');

  // 机器能力提示
  const hw = meta.hw || {};
  const notes = [];
  if (hw.avx2_note) notes.push('⚠ ' + hw.avx2_note);
  if (hw.speed_note) notes.push('⏱ ' + hw.speed_note);
  if (hw.cpu) notes.push('CPU：' + hw.cpu + '（' + (hw.logical || '?') + ' 线程，'
                         + '推理用 ' + (hw.threads || '?') + ' 线程）');
  const nb = $('#hwNote');
  if (notes.length){
    nb.innerHTML = notes.join('<br>');
    nb.style.display = 'block';
    nb.style.background = (hw.avx2_risk === 'high')
      ? 'rgba(201,67,79,.10)' : 'rgba(217,138,43,.09)';
    nb.style.borderColor = (hw.avx2_risk === 'high')
      ? 'rgba(201,67,79,.3)' : 'rgba(217,138,43,.28)';
    nb.style.color = (hw.avx2_risk === 'high') ? '#a8323e' : '#8a5a12';
  } else {
    nb.style.display = 'none';
  }

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
    // 分两种情况：缺组件（要装升级包）vs 组件在但缺模型（放文件就行）。
    // 原来只报了"没找到模型"，但绝大多数人是根本没装组件 ——
    // 让他们去 models\sep 放文件是白费功夫。
    if (!s.has_separator){
      warn.textContent = '「去和声」需要单独安装扩展包（约 410 MB），'
        + '这个整合包里没带。「分人声」是内置的，可以直接用 —— '
        + '它会把伴奏整个剥掉，效果通常够用。';
    } else {
      warn.textContent = '「去和声」组件已装好，但缺模型文件。'
        + '把 5_HP-Karaoke-UVR.pth 放进 models\\sep 就能用。';
    }
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
    // 坏模型：显示原因并禁止选择 —— 不然选中后点转换才炸，很难查
    if (mm.broken){
      inp.disabled = true;
      const sb = document.createElement('small');
      sb.textContent = '不可用';
      sb.style.background = 'rgba(224,90,104,.14)';
      sb.style.color = '#c9434f';
      sp.appendChild(sb);
      lb.title = mm.broken;
      lb.style.opacity = '.55';
    } else if (mm.active){
      const sc = document.createElement('small');
      sc.textContent = '使用中';
      sc.style.background = 'rgba(14,158,110,.15)';
      sc.style.color = '#0e9e6e';
      sp.appendChild(sc);
    }
    lb.appendChild(inp); lb.appendChild(sp);
    box.appendChild(lb);
  });
  const cur = list.find(x => x.active) || list.find(x => !x.broken) || list[0];
  $('#curModel').textContent = cur ? cur.name : '无';
  const bits = [cur.mb + ' MB'];
  if (cur.tgt_sr) bits.push(cur.tgt_sr + ' Hz');
  if (cur.version) bits.push(cur.version);
  if (cur.source) bits.push(cur.source === 'pth' ? '.pth 导入' : '.onnx 导入');
  if (cur.imported_at) bits.push(cur.imported_at.slice(5, 10) + ' 导入');
  $('#modelInfo').textContent = bits.join(' · ');
  $('#btnDelModel').style.display = list.length > 1 ? 'inline-flex' : 'none';
  // 朗读页的贴合度滑块：模型没 .index 就没意义，直接禁用并说明
  try{ rdIdxAvail(!!(cur && cur.index)); }catch(e){}
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

$('#btnRefreshModel').onclick = async () => {
  $('#pmsg').textContent = '正在刷新模型列表…';
  const j = await (await fetch('/api/model/refresh', {method:'POST'})).json();
  if (j.ok){
    meta = await (await fetch('/api/meta')).json();
    if (meta.index_backend !== undefined) _hasIdxBackend = (meta.index_backend !== false);
    renderModels(meta);
    $('#pmsg').textContent = '已刷新：' + j.count + ' 个模型（当前「' + (j.active || '无') + '」）'
      + (j.torch ? '｜.pth 可导入' : '｜无 PyTorch，只能用 .onnx');
  } else {
    $('#pmsg').textContent = '刷新失败：' + j.error;
  }
};

$('#btnDelModel').onclick = async () => {
  const cur = $('#curModel').textContent;
  if (!cur || cur === '无') return;
  if (!confirm('确定删除模型「' + cur + '」？\n会删掉 models 目录里它的 .onnx / .json / .index 文件，不可恢复。')) return;
  const j = await (await fetch('/api/model/delete', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({name:cur})})).json();
  if (j.ok){
    meta = await (await fetch('/api/meta')).json();
    _hasIdxBackend = (meta.index_backend !== false);
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
$('#key').oninput = e => { autoKey = null; setKey(+e.target.value, false); savePrefs(); };

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
  savePrefs();
};

/* 参数记忆：移调 / 检索强度 / 去和声 / 设备，下次打开保持上次的选择 */
const PREF_KEY = 'rvc_prefs_v1';
function savePrefs(){
  try{
    localStorage.setItem(PREF_KEY, JSON.stringify({
      key: +$('#key').value,
      idx: +$('#idxrate').value,
      sep: (document.querySelector('input[name=sep]:checked')||{}).value || 'none',
      dev: (document.querySelector('input[name=dev]:checked')||{}).value || 'auto'
    }));
  }catch(e){}
}
function loadPrefs(){
  let p;
  try{ p = JSON.parse(localStorage.getItem(PREF_KEY) || '{}'); }catch(e){ return; }
  if(typeof p.idx === 'number' && p.idx >= 0 && p.idx <= 80){
    $('#idxrate').value = p.idx;
    $('#idxval').innerHTML = (p.idx / 100).toFixed(2) + '<small>强度</small>';
  }
  if(typeof p.key === 'number' && p.key >= -12 && p.key <= 12){
    autoKey = null; setKey(p.key, false);
  }
  if(p.sep){
    const r = document.querySelector('input[name=sep][value="' + p.sep + '"]');
    if(r && !r.disabled) r.checked = true;
  }
  if(p.dev){
    const r = document.querySelector('input[name=dev][value="' + p.dev + '"]');
    if(r) r.checked = true;
  }
}
document.querySelectorAll('input[name=sep],input[name=dev]').forEach(
  r => r.addEventListener('change', savePrefs));

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
  renderQueue([], '');
  $('#btnClear').style.display = 'none'; $('#outbox').classList.remove('on');
  $('#qaBox').classList.remove('on');
  $('#pmsg').textContent = '选好文件后点上面按钮';
};
/* ================= 自绘播放器 =================
   为什么不用浏览器原生 <audio controls>：
   它在卡片布局里只有一条细线，用户根本看不出那是播放器，
   以为只有"打开文件夹"这一条路 —— 也就听不到自己刚做出来的东西。

   这里自己画：波形（可点跳转）、大播放键、时间、音量、循环、下载。
   波形不是装饰：一眼看出"这里确实有声音、哪里是静音"，比单纯进度条有用。
*/
function fmtTime(t){
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60), sec = Math.floor(t % 60);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}

function makePlayer(ids){
  const a = document.getElementById(ids.audio);
  const cv = document.getElementById(ids.wave);
  const btn = document.getElementById(ids.btn);
  const icon = document.getElementById(ids.icon);
  const timeEl = document.getElementById(ids.time);
  const stEl = document.getElementById(ids.state);
  const volEl = document.getElementById(ids.vol);
  const loopBtn = document.getElementById(ids.loop);
  const dlBtn = document.getElementById(ids.download);
  if (!a || !cv || !btn) return null;

  let peaks = null;      // 每列的峰值（0~1）
  let src = '';          // 当前音频地址
  let hovering = -1;
  let raf = 0;

  function paint(){
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 400, h = 56;
    if (cv.width !== Math.round(w * dpr)){
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const dur = a.duration || 0;
    const cur = a.currentTime || 0;
    const prog = dur > 0 ? cur / dur : 0;

    if (!peaks || !peaks.length){
      // 还在解码：画一条待机细线
      ctx.strokeStyle = '#D8E2F0';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
      ctx.stroke();
      return;
    }

    const n = peaks.length;
    const bw = w / n;
    for (let i = 0; i < n; i++){
      const played = (i / n) < prog;
      const hh = Math.max(2, peaks[i] * (h - 10));
      const x = i * bw;
      const y = (h - hh) / 2;
      if (played){
        ctx.fillStyle = '#0B63F6';
      } else if (i === hovering){
        ctx.fillStyle = '#B9D4FB';
      } else {
        ctx.fillStyle = '#D5E2F2';
      }
      ctx.fillRect(x + bw * 0.18, y, Math.max(1, bw * 0.64), hh);
    }

    // 播放头
    const px = prog * w;
    ctx.fillStyle = 'rgba(11,99,246,.9)';
    ctx.fillRect(Math.min(w - 2, px), 0, 2, h);
  }

  function loopPaint(){
    paint();
    if (!a.paused && !a.ended){
      raf = requestAnimationFrame(loopPaint);
    } else {
      raf = 0;
    }
  }

  async function decodeWave(url){
    peaks = null;
    paint();
    try{
      const r = await fetch(url, {cache: 'no-store'});
      if (!r.ok) return;
      const buf = await r.arrayBuffer();
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      const ac = new AC();
      const ab = await new Promise((ok, no) => {
        // 老回调式签名兼容
        const p = ac.decodeAudioData(buf, ok, no);
        if (p && p.then) p.then(ok).catch(no);
      });
      const ch = ab.getChannelData(0);
      const cols = Math.max(60, Math.min(320, Math.round((cv.clientWidth || 400) / 3)));
      const step = Math.floor(ch.length / cols) || 1;
      const out = new Float32Array(cols);
      let max = 1e-9;
      for (let i = 0; i < cols; i++){
        let m = 0;
        const s0 = i * step;
        const s1 = Math.min(ch.length, s0 + step);
        // 每列抽样 200 个点足够，避免长音频卡住
        const skip = Math.max(1, Math.floor((s1 - s0) / 200));
        for (let j = s0; j < s1; j += skip){
          const v = Math.abs(ch[j]);
          if (v > m) m = v;
        }
        out[i] = m;
        if (m > max) max = m;
      }
      // 归一化，但保留一点相对差异，别把轻声段落放大成一样高
      for (let i = 0; i < cols; i++) out[i] = Math.min(1, out[i] / Math.max(max * 0.92, 1e-9));
      peaks = out;
      try{ ac.close(); }catch(e){}
      paint();
    }catch(e){
      // 解码失败不影响播放，只是没波形
      peaks = null;
      paint();
    }
  }

  function setIcon(playing){
    if (!icon) return;
    icon.innerHTML = playing
      ? '<path d="M7 5h4v14H7zM13 5h4v14h-4z"/>'
      : '<path d="M8 5v14l11-7z"/>';
    btn.classList.toggle('playing', playing);
  }

  a.addEventListener('play', () => { setIcon(true); if (!raf) loopPaint(); });
  a.addEventListener('pause', () => { setIcon(false); paint(); });
  a.addEventListener('ended', () => { setIcon(false); paint(); });
  a.addEventListener('timeupdate', () => {
    timeEl.textContent = fmtTime(a.currentTime) + ' / ' + fmtTime(a.duration);
  });
  a.addEventListener('loadedmetadata', () => {
    timeEl.textContent = fmtTime(0) + ' / ' + fmtTime(a.duration);
    paint();
  });

  btn.onclick = () => {
    if (!src) return;
    if (a.paused) a.play().catch(()=>{});
    else a.pause();
  };

  // 点波形跳转。
  // 坐标一律在**事件触发的那一刻**重新取 rect：页面滚动、布局变化之后
  // 之前取的 rect 就失效了（实测因为这个原因跳转位置偏得离谱）。
  cv.addEventListener('click', e => {
    if (!a.duration) return;
    // 优先用 offsetX：它是相对元素自身的坐标，不受页面滚动/缩放影响。
    // 实测在 headless 环境下 clientX - rect.left 会算出偏差（0.45s vs 期望 1.32s），
    // 而 offsetX 精确等于 306/510 = 0.6。
    const w = cv.clientWidth || cv.getBoundingClientRect().width || 1;
    let x = (typeof e.offsetX === 'number' && isFinite(e.offsetX))
      ? e.offsetX
      : (e.clientX - cv.getBoundingClientRect().left);
    if (!isFinite(x)) x = 0;
    const rel = Math.max(0, Math.min(1, x / w));
    // 留个可观测标记：出问题时能直接看出"算出来的比例是多少"，
    // 不用靠猜（这个字段也方便自动化测试）。
    cv.dataset.rel = rel.toFixed(3);
    try{ a.currentTime = rel * a.duration; }catch(err){}
    paint();
  });

  // 悬停高亮（帮用户看出可以点）
  cv.addEventListener('mousemove', e => {
    const r = cv.getBoundingClientRect();
    const rel = (e.clientX - r.left) / r.width;
    hovering = peaks ? Math.floor(rel * peaks.length) : -1;
    if (a.paused) paint();
  });
  cv.addEventListener('mouseleave', () => {
    hovering = -1;
    if (a.paused) paint();
  });

  if (volEl){
    volEl.oninput = () => { a.volume = +volEl.value / 100; };
    a.volume = +volEl.value / 100;
  }
  if (loopBtn){
    loopBtn.onclick = () => {
      a.loop = !a.loop;
      loopBtn.classList.toggle('on', a.loop);
      stEl.textContent = a.loop ? '循环播放' : '';
    };
  }
  if (dlBtn){
    dlBtn.onclick = () => {
      if (!src) return;
      const ext = '.wav';
      const base = (ids.dlname || '输出').replace(/[\\/:*?"<>|]/g, '_');
      const el = document.createElement('a');
      el.href = src;
      el.download = base + ext;
      document.body.appendChild(el); el.click(); el.remove();
    };
  }

  let loadedKey = '';     // 已装入内容的指纹（不是 URL —— URL 带时间戳每次都不同）

  return {
    /* 装入新音频。autoplay=true 时立刻播。
       key 用来判断"是不是同一份内容"：
       界面每 1 秒轮询一次状态，如果每次都重新装 src，
       播放位置就会被清零 —— 表现为"播几秒自己回到开头，根本没法听"。
       所以同一个 key 只装一次。 */
    load(url, opt){
      const o = opt || {};
      // 默认指纹：去掉 URL 里的 t=时间戳参数。
      // 这样即使某个调用点忘了传 key，同一份音频也不会被重复装载
      // （之前就是因为漏传 key 导致播放位置被反复清零）。
      const defKey = String(url).replace(/([?&])t=\d+&?/g, '$1')
                                 .replace(/[?&]$/, '');
      const key = (o.key !== undefined && o.key !== null) ? String(o.key) : defKey;
      if (key === loadedKey && a.getAttribute('src')){
        // 同一份内容，什么都不做。
        // ★ 这里绝不能"如果暂停就自动 play()"：
        //   播完（ended）之后 paused 也是 true，再 play() 会**从头重播**，
        //   而界面每秒轮询一次 —— 表现就是"放完自己又从头开始，循环不止"。
        return;
      }
      loadedKey = key;
      window.__plReloads = (window.__plReloads || 0) + 1;   // 观测用：实际装载次数
      src = url;
      stEl.textContent = '';
      a.src = url;
      a.load();
      decodeWave(url);
      timeEl.textContent = '0:00 / 0:00';
      if (o.autoplay){
        a.play().catch(()=>{});
      }
    },
    stop(){
      try{ a.pause(); a.currentTime = 0; }catch(e){}
      setIcon(false); paint();
    },
    get audio(){ return a; },
    get src(){ return src; },
    setLabel(t){ if (stEl) stEl.textContent = t || ''; },
    repaint(){ paint(); }
  };
}

/* 翻唱侧：A/B 对比共用同一个波形控件，切换时保持播放位置 */
let lastPos = 0;
const sPlayer = makePlayer({
  audio: 'player', wave: 'sPlWave', btn: 'sPlBtn', icon: 'sPlIcon',
  time: 'sPlTime', state: 'sPlSt', vol: 'sPlVol',
  loop: 'sPlLoop', download: 'sPlDl', dlname: '转换结果'
});
let abWhich = 'out';
function playAB(which){
  if (!sPlayer) return;
  const a = sPlayer.audio;
  // A/B 对比要落在同一个位置，否则切过去从头播，根本比不出来
  const pos = (a.currentTime > 0.3 && !a.ended) ? a.currentTime : lastPos;
  abWhich = which;
  // 只通过播放器装一次 src —— 之前这里同时设了 a.src 又调 load，
  // 等于加载两遍，会把自己打断
  sPlayer.load('/api/audio?which=' + which + '&t=' + Date.now(), {autoplay: true});
  a.addEventListener('loadedmetadata', () => {
    try{ a.currentTime = Math.min(pos, (a.duration || pos)); }catch(e){}
  }, {once: true});
  sPlayer.setLabel(which === 'in' ? '正在试听原声' : '正在试听结果');
  $('#sPlA').classList.toggle('on', which === 'in');
  $('#sPlB').classList.toggle('on', which === 'out');
}
$('#player').addEventListener('timeupdate', e => {
  if (!e.target.paused) lastPos = e.target.currentTime;
});
$('#sPlA').onclick = () => playAB('in');
$('#sPlB').onclick = () => playAB('out');

/* ================= v2.0 朗读模式 ================= */
let rdMode = 'sing', rdOpts = null, rdBusy = false;

// 朗读结果的播放器（和翻唱那个同一套组件，各自的 audio 元素）
const rdPlayer = makePlayer({
  audio: 'rdResult', wave: 'rdPlWave', btn: 'rdPlBtn', icon: 'rdPlIcon',
  time: 'rdPlTime', state: 'rdPlSt', vol: 'rdPlVol',
  loop: 'rdPlLoop', download: 'rdPlDl', dlname: '朗读'
});

function setMode(m){
  rdMode = m;
  document.querySelectorAll('.mode').forEach(b => {
    b.classList.toggle('on', b.dataset.m === m);
  });
  $('#gridSing').style.display = (m === 'sing') ? '' : 'none';
  $('#gridRead').style.display = (m === 'read') ? '' : 'none';
  if (m === 'read' && !rdOpts) loadReadOpts();
  try{ localStorage.setItem('rvc_mode', m); }catch(e){}
  window.scrollTo({top: 0, behavior: 'smooth'});
}
document.querySelectorAll('.mode').forEach(b => {
  b.onclick = () => setMode(b.dataset.m);
});

async function loadReadOpts(){
  try{
    rdOpts = await (await fetch('/api/read/options')).json();
  }catch(e){ return; }
  // 情绪九宫格
  $('#segEmo').innerHTML = (rdOpts.emotions || []).map((e, i) =>
    '<label><input type="radio" name="emo" value="' + e.id + '"' +
    (e.id === 'neutral' ? ' checked' : '') + '><span>' + e.label + '</span></label>'
  ).join('');
  // 载体语音下拉
  $('#rdVoice').innerHTML = '<option value="">自动（按模型音域选）</option>' +
    (rdOpts.voices || []).map(v =>
      '<option value="' + v.id + '">' + v.name + ' · ' + v.note +
      (v.gender === 'male' ? ' · 男' : ' · 女') + '</option>'
    ).join('');
}

// 字数与预估
function rdStats(){
  const t = $('#rdText').value.trim();
  const n = t.length;
  $('#rdCount').textContent = n + ' 字';
  // 中文约 4.5 字/秒；变声 RTF 按实测 2.8 估
  if (n >= 10){
    const sec = n / 4.5;
    const need = Math.round(sec * 2.8 + 12);
    $('#rdEst').textContent = '预计约 ' + sec.toFixed(0) + ' 秒音频 · 处理约 ' +
      (need >= 60 ? Math.round(need/60) + ' 分钟' : need + ' 秒');
  } else {
    $('#rdEst').textContent = '';
  }
  $('#rdGo').disabled = (n < 2);
}
$('#rdText').addEventListener('input', rdStats);

// 自动移调开关
$('#rdKeyAuto').addEventListener('change', e => {
  $('#rdKey').disabled = e.target.checked;
  $('#rdKeyVal').innerHTML = e.target.checked
    ? '自动<small>移调</small>'
    : '<span id="rdKeyNum">0</span><small>半音</small>';
  if (!e.target.checked){
    $('#rdKey').oninput = ev => {
      const v = +ev.target.value;
      $('#rdKeyVal').innerHTML = (v > 0 ? '+' : '') + v + '<small>半音</small>';
    };
  }
});

// 试听语调（只合成载体）
$('#rdPreview').onclick = async () => {
  const btn = $('#rdPreview');
  btn.disabled = true;
  $('#rdPrevMsg').textContent = '合成中…';
  const text = ($('#rdText').value.trim() || '你好，这是我的声音。').slice(0, 120);
  const emo = (document.querySelector('input[name=emo]:checked') || {}).value || 'neutral';
  try{
    const j = await (await fetch('/api/read/preview', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text, emotion: emo,
                            voice: $('#rdVoice').value || null})})).json();
    if (j.ok){
      // 走同一个播放器，别再退回原生 audio（那个在这个布局里看不见）
      if (rdPlayer) rdPlayer.load('/api/audio?which=preview&t=' + Date.now(), {autoplay: true});
      if (rdPlayer) rdPlayer.setLabel('试听载体（未变声）');
      $('#rdPrevMsg').textContent = j.voice + ' / ' + j.emotion;
    } else {
      $('#rdPrevMsg').textContent = j.error || '失败';
    }
  }catch(e){
    $('#rdPrevMsg').textContent = '网络错误';
  }
  btn.disabled = false;
};

// 开始朗读
$('#rdGo').onclick = async () => {
  const text = $('#rdText').value.trim();
  if (text.length < 2) return;
  rdBusy = true;
  $('#rdGo').disabled = true;
  $('#rdOut').classList.remove('on');
  $('#rdBar').style.width = '0%';
  $('#rdPct').textContent = '';
  $('#rdMsg').textContent = '提交中…';
  const emo = (document.querySelector('input[name=emo]:checked') || {}).value || 'neutral';
  const body = {
    text: text,
    emotion: emo,
    voice: $('#rdVoice').value || null,
    key: $('#rdKeyAuto').checked ? null : +$('#rdKey').value,
    // 用朗读页自己的贴合度，不再借用翻唱页的滑块 ——
    // 之前如果翻唱页设成 0，朗读的音色就完全不贴，还查不出原因
    index_rate: +($('#rdIdx').value || 75) / 100
  };
  try{
    const j = await (await fetch('/api/read', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)})).json();
    if (!j.ok){
      $('#rdMsg').textContent = j.error || '提交失败';
      rdBusy = false;
      $('#rdGo').disabled = false;
    }
  }catch(e){
    $('#rdMsg').textContent = '网络错误';
    rdBusy = false;
    $('#rdGo').disabled = false;
  }
};

$('#rdReveal').onclick = async () => {
  await fetch('/api/reveal', {method: 'POST'});
};

// 音色贴合度滑块
$('#rdIdx').addEventListener('input', e => {
  const v = +e.target.value / 100;
  $('#rdIdxVal').innerHTML = v.toFixed(2) + '<small>贴合度</small>';
  $('#rdIdxHint').textContent = v === 0 ? '只靠模型本身'
    : (v >= 0.9 ? '最贴，可能有金属感' : (v >= 0.6 ? '推荐范围' : '偏松'));
  try{ localStorage.setItem('rvc_read_idx', String(e.target.value)); }catch(err){}
});
// 有 .index 才谈得上"贴合度"，没有就说明白。
// 还要区分两种情况：模型没带 .index， vs 程序缺 faiss 后端（那有 .index 也用不了）
let _hasIdxBackend = true;
function rdIdxAvail(hasIndex){
  const s = $('#rdIdx');
  if (!_hasIdxBackend){
    s.disabled = true;
    $('#rdIdxHint').innerHTML =
      '<span style="color:#c9434f">特征检索组件缺失，贴合度不起作用</span>';
    return;
  }
  if (hasIndex){
    s.disabled = false;
    $('#rdIdxHint').textContent = '有索引，音色调这里';
  } else {
    s.disabled = true;
    $('#rdIdxHint').innerHTML =
      '<span style="color:#d98a2b">当前模型没有 .index，音色只能靠模型本身</span>';
  }
}

function rdRender(s){
  // 进度与提示（朗读进行中由 poll 调用）
  if (s.stage === 'reading'){
    const p = Math.round((s.frac || 0) * 100);
    $('#rdBar').style.width = p + '%';
    $('#rdPct').textContent = p + '%';
    $('#rdMsg').textContent = s.msg || '';
  }
  if (s.stage === 'done' && s.read){
    rdBusy = false;
    $('#rdGo').disabled = ($('#rdText').value.trim().length < 2);
    $('#rdBar').style.width = '100%';
    $('#rdPct').textContent = '100%';
    $('#rdMsg').textContent = s.msg || '完成';
    $('#rdOut').classList.add('on');
    $('#rdOutPath').textContent = s.output || '';
    // 装进播放器并自动播一次。key 用输出路径：轮询时同一份结果不会重装，
    // 否则每次轮询都把播放位置清零（实测就是这个原因导致"听不了"）
    if (rdPlayer){
      rdPlayer.load('/api/audio?which=out&t=' + Date.now(),
                    {autoplay: true, key: s.output || 'read_out'});
    }
    const q = s.qa;
    if (q){
      $('#rdQa').classList.add('on');
      $('#rdQaScore').textContent = q.score;
      $('#rdQaTag').textContent = q.score >= 75 ? '可用' : '建议复查';
      $('#rdQaTag').className = q.score >= 75 ? '' : 'warn';
      $('#rdQaGrid').innerHTML =
        '<div><span>谐波度</span><b>' + q.harmonicity.toFixed(2) + '</b></div>' +
        '<div><span>低频残留</span><b>' + q.low_residual_db.toFixed(1) + ' dB</b></div>';
    } else {
      $('#rdQa').classList.remove('on');
    }
  }
  if (s.stage === 'error' && rdBusy){
    rdBusy = false;
    $('#rdGo').disabled = ($('#rdText').value.trim().length < 2);
    $('#rdMsg').textContent = s.error || '出错了';
  }
}

function renderQueue(q, cur){
  const box = $('#qlist');
  if(!q || !q.length){ box.classList.remove('on'); box.innerHTML = ''; return; }
  box.classList.add('on');
  const show = q.slice(0, 5);
  box.innerHTML = show.map(p => {
    const nm = p.split(/[\\/]/).pop();
    const mark = (p === cur) ? '<b>当前</b>' : '';
    return '<div class="qi">' + mark + '<span title="' + nm + '">' + nm +
      '</span><button data-p="' + encodeURIComponent(p) + '" title="从队列移除">×</button></div>';
  }).join('') + (q.length > 5 ? '<div class="more">还有 ' + (q.length - 5) + ' 个…</div>' : '');
  box.querySelectorAll('button[data-p]').forEach(b => {
    b.onclick = async () => {
      await fetch('/api/queue/remove', {method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path: decodeURIComponent(b.dataset.p)})});
      poll();
    };
  });
}

async function poll(){
  clearTimeout(timer);
  const s = await (await fetch('/api/state')).json();
  try{ rdRender(s); }catch(e){}
  // 导入刚结束（importing → 非 importing）→ 一定要刷新列表。
  // 只靠 imported 变化来判断不够：同一个模型重复导入（内容相同会直接复用）
  // 时 imported 的值没变，列表就不会刷，用户会以为"导入没成功"。
  const wasImp = _wasImporting;
  _wasImporting = (s.stage === 'importing');
  if (s.imported && (s.imported !== lastImported || (wasImp && !_wasImporting))){
    lastImported = s.imported;
    meta = await (await fetch('/api/meta')).json();
    renderModels(meta);
    $('#pmsg').textContent = '已导入并切换到「' + s.imported + '」';
  } else if (wasImp && !_wasImporting && s.stage === 'error'){
    // 导入失败：把列表也刷一遍，避免界面停在旧状态
    meta = await (await fetch('/api/meta')).json();
    renderModels(meta);
  }
  const p = Math.round((s.frac||0)*100);
  $('#bar').style.width = p+'%';
  $('#pct').textContent = p+'%';
  let msg = (s.phase ? '['+s.phase+'] ' : '') + (s.msg||'');
  // 出错时必须把**真实原因**显示出来。
  // 原来只显示 s.msg（往往是"导入失败"这种笼统文案），
  // 真正的原因在 s.error 里 —— 用户看到的就是"点了没反应/导入失败"，
  // 完全无从下手，只能来问。
  if (s.stage === 'error' && s.error){
    msg = s.error;
  }
  // 预计剩余：按"已用时间 / 已完成进度"反推，进度太小时不显示（会乱跳）
  if(s.started_at && s.frac > 0.06 && s.frac < 0.995 &&
     ['running','separating'].includes(s.stage)){
    const used = Date.now()/1000 - s.started_at;
    const left = Math.max(0, used / s.frac - used);
    if(left >= 5) msg += ' · 剩余约 ' +
      (left >= 60 ? Math.round(left/60) + ' 分' : Math.round(left) + ' 秒');
  }
  $('#pmsg').textContent = msg;
  if(s.device_used) $('#devInfo').textContent = '实际设备：' + s.device_used;
  renderQueue(s.queue, s.input);
  if(s.queue && s.queue.length){
    $('#qinfo').textContent = s.queue.length + ' 个文件已加入队列';
    $('#btnClear').style.display = 'inline-flex';
  } else {
    $('#qinfo').textContent = '还没选文件';
    $('#btnClear').style.display = 'none';
  }

  if(s.analysis){
    const a = s.analysis, st = a.stats;
    $('#anaBox').classList.add('on');
    $('#aMed').textContent = Math.round(st.median) + ' Hz';
    $('#aP70').textContent = Math.round(st.p70) + ' Hz';
    $('#aP95').textContent = Math.round(st.p95) + ' Hz';
    $('#aKey').textContent = (a.key>0?'+':'') + a.key;
    $('#aWhy').textContent = a.reason;
    // 分段音高条：竖向高度按 95 分位映射（以 500Hz 为顶），超过 375 标橙、超 450 标红
    const segs = a.segments || [];
    const box = $('#aSegs');
    if (segs.length){
      box.innerHTML = segs.map(sg => {
        const h = Math.max(6, Math.min(100, sg.p95 / 5));
        const cls = sg.p95 > 450 ? 'over' : (sg.p95 > 375 ? 'hi' : '');
        return '<i class="' + cls + '" style="height:' + h + '%" title="' +
               sg.t0 + '~' + sg.t1 + 's · 95分位 ' + Math.round(sg.p95) + 'Hz"></i>';
      }).join('');
      $('#aSegLegend').style.display = 'flex';
    } else {
      box.innerHTML = ''; $('#aSegLegend').style.display = 'none';
    }
    if(autoKey !== a.key){ autoKey = a.key; setKey(a.key, true); }
  }

  // 输出质量自检
  if(s.qa){
    const q = s.qa;
    $('#qaBox').classList.add('on');
    $('#qaScore').textContent = q.score;
    const good = q.score >= 75;
    const tag = $('#qaTag');
    tag.textContent = good ? '可用' : '建议复查';
    tag.className = good ? '' : 'warn';
    $('#qaGrid').innerHTML =
      '<div><span>谐波度</span><b>' + q.harmonicity.toFixed(2) + '</b></div>' +
      '<div><span>低频残留</span><b>' + q.low_residual_db.toFixed(1) + ' dB</b></div>' +
      '<div><span>电平稳定</span><b>' + q.level_std_db.toFixed(1) + ' dB</b></div>' +
      '<div><span>峰值</span><b>' + q.peak.toFixed(2) + '</b></div>';
    $('#qaTips').innerHTML = (q.tips || []).map(t => '<div>' + t + '</div>').join('');
  } else {
    $('#qaBox').classList.remove('on');
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

loadMeta().then(() => {
  loadPrefs();
  try{
    const m = localStorage.getItem('rvc_mode');
    if (m === 'read') setMode('read');
  }catch(e){}
  rdStats();
  try{
    const v = localStorage.getItem('rvc_read_idx');
    if (v !== null){
      $('#rdIdx').value = v;
      $('#rdIdxVal').innerHTML = (+v/100).toFixed(2) + '<small>贴合度</small>';
    }
  }catch(e){}
  poll();
});
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RVCCpuPlus/1.0"

    def log_message(self, fmt, *a):
        pass

    def _serve_file(self, path, ctype, download_name=None):
        """带 Range 支持的文件响应。

        播放器拖动进度、A/B 对比保持位置都依赖这个：没有 Range，
        浏览器就只能从头顺序播，seek 全部失效。
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            return self._json({"ok": False, "error": "文件读不到"}, 404)
        rng = self.headers.get("Range") or ""
        start, end = 0, size - 1
        partial = False
        if rng.startswith("bytes="):
            spec = rng[6:].split(",")[0].strip()
            try:
                a, _, b = spec.partition("-")
                if a == "":                      # bytes=-500 取末尾
                    start = max(0, size - int(b))
                else:
                    start = int(a)
                    if b:
                        end = int(b)
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                partial = True
            except Exception:
                start, end, partial = 0, size - 1, False

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        if download_name:
            from urllib.parse import quote as _q
            self.send_header("Content-Disposition",
                             "attachment; filename*=UTF-8''%s" % _q(download_name))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    buf = f.read(min(262144, left))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass       # 播放器跳到别处时会主动断连，正常现象

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/read/options":
            from rvc_cpu import tts as _tts
            return self._json({"ok": True,
                               "voices": _tts.list_voices(),
                               "emotions": _tts.list_emotions()})

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
            elif which == "preview":
                p = st.get("preview_read") or ""
            else:
                outs = st.get("outputs") or ([st["output"]] if st.get("output") else [])
                p = outs[idx] if 0 <= idx < len(outs) else (outs[-1] if outs else "")
            if not p or not os.path.isfile(p):
                return self._json({"ok": False, "error": "还没有可试听的文件"}, 404)

            # ★ 必须支持 HTTP Range，否则浏览器**无法拖动进度条**：
            #   媒体元素靠 byte range 请求做 seek，服务器不认 Range 时
            #   audio.seekable 为空，给 currentTime 赋值会被静默忽略
            #   （实测：波形点击算出的比例是对的 0.600，但播放位置纹丝不动）。
            self._serve_file(p, "audio/wav")
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

            # 入库前确认是"真能解码"的音频。
            # 注意必须深校验（真解码），不能只 sf.info —— 损坏的 mp3 头是好的、
            # 数据是坏的，只读头会放过它，然后到转换阶段才炸成
            # "Unspecified internal error"（用户看不懂）。
            try:
                from rvc_cpu.audio_io import probe
                sr_u, ch_u, sec_u = probe(dst, deep=True)
                # 门槛跟界面提示、命令行提示保持一致（都是 2 秒）。
                # ContentVec 的卷积栈对超短输入会报错，太短的素材本来也转不出东西。
                if sec_u < 2.0:
                    raise RuntimeError(
                        "%s 只有 %.1f 秒，至少要 2 秒。"
                        "（太短的话模型没有足够上下文，转出来会断断续续）"
                        % (os.path.basename(dst), sec_u))
            except Exception as _e:
                try:
                    q.remove(dst)
                    os.remove(dst)
                except Exception:
                    pass
                return self._json({"ok": False, "error": str(_e)})

            # 转换中上传：只追加队列，不动当前输入和结果区，
            # 否则界面上"正在转换的文件"会突然变成刚落进来的那个。
            busy = get_state()["stage"] in ("running", "loading", "separating",
                                            "analyzing", "importing")
            if busy:
                set_state(queue=q)
            else:
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
            # 没做过音高分析就直接转换，是最常见的"声音发闷、咬字不清"来源：
            # 移调停在 0，输出音高会掉到模型舒适区之外。这里在转换前
            # 自动跑一次分析并采用建议值（用户手动设过就不动）。
            if not payload.get("key_locked") and not st.get("analysis"):
                try:
                    from rvc_cpu.calibrate import analyze_file, load_profile, suggest
                    from rvc_cpu import models as _mm
                    rmv = os.path.join(MODELS, "rmvpe.onnx")
                    if os.path.isfile(rmv):
                        set_state(stage="analyzing", frac=0.01,
                                  msg="先测一下音高，避免移调错了白跑…")
                        stats, f0 = analyze_file(rmv, p, return_f0=True)
                        prof = None
                        try:
                            prof = load_profile(MODELS, _mm.active_name())
                        except Exception:
                            prof = None
                        r = suggest(stats, prof)
                        key = int(r["key"])
                        set_state(analysis={"stats": stats, "key": key,
                                            "reason": r["reason"], "auto": True})
                        log("自动定调：移 %+d 半音（%s）" % (key, r["reason"]))
                except Exception as _e:
                    log("自动定调跳过（%s），按原移调 %+d 继续" % (str(_e)[:60], key))
            rate = float(payload.get("index_rate", 0.0))
            sep_mode = str(payload.get("sep", "none"))
            device = str(payload.get("device", "auto"))
            global _abort
            _abort = False
            files = [x for x in (st.get("queue") or [p]) if os.path.isfile(x)]
            if not files:
                files = [p]
            set_state(stage="loading", frac=0.01, msg="准备中…", error="", output="",
                      outputs=[], phase="", total=len(files), qa=None)
            threading.Thread(target=run_job, args=(files, key, rate, sep_mode, device),
                             daemon=True).start()
            return self._json({"ok": True, "files": len(files)})

        if u.path == "/api/queue/remove":
            path_to_drop = str(payload.get("path") or "").strip()
            st = get_state()
            q = [x for x in (st.get("queue") or []) if os.path.isfile(x)]
            if path_to_drop not in q:
                return self._json({"ok": False, "error": "队列里没有这个文件"})
            q.remove(path_to_drop)
            busy = st["stage"] in ("running", "loading", "separating",
                                   "analyzing", "importing")
            if busy:
                # 任务进行中：只动队列，别碰正在处理的那个
                set_state(queue=q)
            else:
                new_input = st["input"] if st["input"] in q else (q[0] if q else "")
                set_state(queue=q, input=new_input, output="", qa=None,
                          analysis=None if not new_input else st.get("analysis"))
            log("从队列移除：" + os.path.basename(path_to_drop))
            return self._json({"ok": True, "count": len(q)})

        if u.path == "/api/read":
            st = get_state()
            if st["stage"] in ("running", "loading", "separating", "analyzing",
                               "importing", "reading"):
                return self._json({"ok": False, "error": "正在忙，等一下"})
            text = str(payload.get("text") or "").strip()
            if not text:
                return self._json({"ok": False, "error": "文本是空的"})
            if len(text) > 5000:
                return self._json({"ok": False,
                                   "error": "文本太长了（%d 字），请分几次，单次上限 5000 字"
                                            % len(text)})
            from app.reading import run_read
            threading.Thread(
                target=run_read,
                args=(text,
                      str(payload.get("emotion") or "neutral"),
                      payload.get("voice") or None,
                      payload.get("key"),
                      float(payload.get("index_rate", 0.5)),
                      str(payload.get("device") or "auto")),
                daemon=True).start()
            return self._json({"ok": True, "chars": len(text)})

        if u.path == "/api/read/preview":
            # 只合成载体（不变声），用来挑声音/情绪，很快
            st = get_state()
            if st["stage"] in ("running", "loading", "separating", "analyzing",
                               "importing", "reading"):
                return self._json({"ok": False, "error": "正在忙，等一下"})
            text = str(payload.get("text") or "").strip() or "你好，这是我的声音。"
            from rvc_cpu import tts as _tts
            pv = os.path.join(WORK, "_朗读试听.wav")
            try:
                info = _tts.synth(text[:120], pv,
                                  voice_id=payload.get("voice") or None,
                                  emotion=str(payload.get("emotion") or "neutral"))
            except Exception as e:
                return self._json({"ok": False, "error": str(e)})
            set_state(preview_read=pv)
            log("试听载体：%s / %s" % (info["voice"]["name"], info["emotion"]))
            return self._json({"ok": True, "voice": info["voice"]["name"],
                               "emotion": info["emotion"]})

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

        if u.path == "/api/model/refresh":
            # 强制丢掉 meta 缓存：用户可能自己把 .onnx 拷进了 models 目录，
            # 不走导入流程也想让它出现（这是最省事的用法，得支持）
            global _meta_cache, _meta_cache_at
            _meta_cache = None
            _meta_cache_at = 0.0
            try:
                from rvc_cpu import models as _mm
                lst, act = _mm.list_models()
                hint = _mm.torch_hint()
                log("刷新模型列表：%d 个（当前 %s）" % (len(lst), act or "无"))
                return self._json({"ok": True, "count": len(lst), "active": act,
                                   "torch": bool(_mm.find_torch_python()),
                                   "torch_hint": hint})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)})

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

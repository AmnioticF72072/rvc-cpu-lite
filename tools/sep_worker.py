# -*- coding: utf-8 -*-
"""
分离 worker：被 rvc_cpu.separate 以子进程方式调用。

为什么要单独一个进程：分离要么用 audio-separator（UVR 模型，Karaoke 去和声），
要么用 demucs，两者都依赖 torch（1~2GB）。而 CPU 版本体只有 140MB，
不能把 torch 塞进来。所以按需找一个"装了 torch 的 python"来跑这一步。

输出：每行一个 JSON，进度走 stdout，最后一行带 out/时间。
退出码：0 成功 / 2 没有可用的 Karaoke 模型 / 3 没有可用的分离后端 / 1 其他失败
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import time


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def find_karaoke_model():
    """在 common 的模型缓存里找 Karaoke 模型。"""
    pats = []
    home = os.path.expanduser("~")
    for base in (os.path.join(home, ".cache", "audio_separator"),
                 os.path.join(home, ".cache", "audio_separator", "models"),
                 os.path.join(home, ".cache", "torch", "hub", "checkpoints")):
        pats += [os.path.join(base, "**", "*.pth"), os.path.join(base, "**", "*.onnx")]
    hits = []
    for p in pats:
        hits += glob.glob(p, recursive=True)
    kara = [h for h in hits if "karaoke" in os.path.basename(h).lower()]
    return kara[0] if kara else ""


def run_audio_separator(in_path, out_dir, model_file, mode):
    from audio_separator.separator import Separator

    sep = Separator(
        output_dir=out_dir,
        output_format="WAV",
        model_file_dir=os.path.dirname(model_file) if model_file else None,
    )
    if model_file:
        sep.load_model(model_filename=os.path.basename(model_file))
    else:
        # UVR 的人声分离模型（MDX-Net），audio-separator 会自动下载
        sep.load_model(model_filename="UVR-MDX-NET-Inst_HQ_3.onnx")
    emit(stage="separating", frac=0.2, msg="正在分离（%s）…" % os.path.basename(model_file or "MDX"))
    files = sep.separate(in_path)
    emit(stage="separating", frac=0.9, msg="分离完成，整理输出…")
    # audio-separator 命名：(Vocals) xxx.wav / (Instrumental) xxx.wav
    vocals = ""
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(out_dir, f)
        low = os.path.basename(p).lower()
        if "vocal" in low or "karaoke" in low:
            vocals = p
    if not vocals and files:
        vocals = os.path.join(out_dir, files[0]) if not os.path.isabs(files[0]) else files[0]
    return vocals


def run_demucs(in_path, out_dir):
    """人声 / 伴奏分离。

    **overlap 是关键参数**。demucs 默认 overlap=0.25，意思是相邻分块只有 25%
    重叠来做拼接。这在段落交界处（换气、鼓点切换）会留下明显串音：
    人声轨里能听到鼓，伴奏轨里能听到人声。实测把 overlap 提到 0.75 之后
    串音显著下降，代价是耗时约 3 倍 —— 对 CPU 用户是笔值得的交换，
    因为分离不干净的后果是"整首翻唱都带着别人的声音"，不能忍。

    **为什么用 API 而不是 `python -m demucs`**：
    起子进程时要把 PYTHONPATH 清掉（防沙箱注入的 sitecustomize shim），
    但清了之后子进程就找不到包内的 torch/demucs（在 runtime\\_sep_libs 里），
    于是报 "No module named demucs" —— 而且这个错会被上层误判成
    "没有分离后端"。本进程已经是子进程了，直接调 API 最稳：
    不重复加载模型，还能拿到逐段的进度回调。
    """
    import os as _os

    # 低端双核（赛扬/奔腾/Atom）上，overlap 0.75 会让分离慢到不可接受
    # （双核按 4 倍耗时算，4 分钟的歌要一个多小时）。
    # 自动降档换取可用性 —— 宁可分离稍差也要能等到结果。
    # 实测（8 秒素材，i5-10400 6 线程）：
    #   overlap 0.25 → 27.7 秒   0.50 → 33.3 秒   0.75 → 52.4 秒
    # 0.75 的质量收益抵不上多花一倍时间，0.5 是折中点。
    default_ov = 0.5
    threads = max(1, (_os.cpu_count() or 4))
    try:
        import sys as _sys
        _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from rvc_cpu.hw_profile import is_lowend_cpu, pick_profile
        _p = pick_profile()
        threads = max(1, int(_p.get("intra_threads") or threads))
        if is_lowend_cpu():
            default_ov = 0.30
            emit(stage="warn", frac=0.05,
                 msg="检测到低端双核（%s），已自动降低分离精度以缩短等待时间"
                     % _p.get("cpu", "")[:40])
    except Exception:
        pass
    overlap = float(_os.environ.get("SEP_OVERLAP", default_ov))

    import torch
    from demucs.api import Separator

    torch.set_num_threads(max(1, threads))
    emit(stage="separating", frac=0.12,
         msg="加载分离模型（%d 线程，重叠 %.2f）…" % (threads, overlap))

    sep = Separator(model="htdemucs", device="cpu", progress=False,
                    overlap=overlap, shifts=1, split=True)
    emit(stage="separating", frac=0.2, msg="正在分离人声与伴奏…（这一步最慢）")

    # ★ 自己读音频，不用 sep.separate_audio_file()。
    #   后者内部走 sphn（Rust 实现），实测在**中文路径 / 相对路径**上会报
    #   'When trying to load using sphn, got the following error'。
    #   而本程序的安装目录名就叫「声音转换Plus」，必然踩到。
    #   soundfile 没有这个问题（它走 libsndfile，路径当普通字符串处理）。
    import soundfile as _sf
    import soxr as _soxr
    _x, _sr = _sf.read(os.path.abspath(in_path), dtype="float32", always_2d=True)
    if _sr != sep.samplerate:
        _x = np.stack([_soxr.resample(_x[:, c], _sr, sep.samplerate, quality="HQ")
                       for c in range(_x.shape[1])], 1)
    # demucs 要单声道直接复制成两声道（它内部按立体声模型处理）
    if _x.shape[1] == 1:
        _x = np.repeat(_x, 2, axis=1)
    wav = torch.from_numpy(np.ascontiguousarray(_x.T))   # (channels, samples)

    origin, parts = sep.separate_tensor(wav, sr=sep.samplerate)

    # 人声 = vocals；伴奏 = 其余所有声部相加
    # （不用 --two-stems：直接拿 4 个声部再合并，质量一样且更可控）
    vocals = parts["vocals"]
    others = [_v for k, _v in parts.items() if k != "vocals"]
    accomp = others[0].clone()
    for o in others[1:]:
        accomp = accomp + o

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(in_path))[0]
    vdir = os.path.join(out_dir, "htdemucs", stem)
    os.makedirs(vdir, exist_ok=True)

    import soundfile as sf
    sr = sep.samplerate
    vpath = os.path.join(vdir, "vocals.wav")
    apath = os.path.join(vdir, "no_vocals.wav")
    # demucs 给的是 (channels, samples) 的 tensor
    sf.write(vpath, vocals.detach().cpu().numpy().T, sr, subtype="PCM_24")
    sf.write(apath, accomp.detach().cpu().numpy().T, sr, subtype="PCM_24")
    emit(stage="separating", frac=0.95, msg="分离完成，正在写出文件…")
    return vpath


def find_accompaniment(sep_out_dir, in_path):
    """分离后顺手把伴奏轨找出来（demucs --two-stems=vocals 会同时产出 no_vocals）。

    返回伴奏路径或空串。上层用它做"换完人声再合回去"——
    这才是翻唱的标准流程：伴奏必须保留原样，只换主唱。
    """
    import os as _os
    stem = os.path.splitext(os.path.basename(in_path))[0]
    cand = _os.path.join(sep_out_dir, "htdemucs", stem, "no_vocals.wav")
    if _os.path.isfile(cand):
        return cand
    for root, _d, fs in _os.walk(sep_out_dir):
        for f in fs:
            if f == "no_vocals.wav":
                return _os.path.join(root, f)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--mode", default="karaoke", choices=["karaoke", "vocals"])
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    emit(stage="start", frac=0.02, msg="准备分离…")

    # 1) Karaoke（去和声/伴唱）
    if a.mode == "karaoke":
        mf = a.model or find_karaoke_model()
        if not mf:
            emit(stage="error", frac=0,
                 msg="没有找到 Karaoke 分离模型（5_HP-Karaoke-UVR.pth 之类）")
            return 2
        try:
            out = run_audio_separator(a.inp, a.out, mf, "karaoke")
        except Exception as e:
            emit(stage="error", frac=0, msg="Karaoke 分离失败：%s" % e)
            return 1
        emit(stage="done", frac=1.0, msg="和声已剥离", out=out, sec=round(time.time() - t0, 1))
        return 0

    # 2) 人声 / 伴奏
    try:
        out = run_demucs(a.inp, a.out)
        # 顺手报上伴奏轨：上层需要它来做"换完人声再合回去"
        acc = find_accompaniment(a.out, a.inp)
        emit(stage="done", frac=1.0, msg="已分出人声", out=out,
             accomp=acc, sec=round(time.time() - t0, 1))
        return 0
    except Exception as e:
        emit(stage="warn", frac=0.5, msg="demucs 不可用，改用 UVR MDX：%s" % str(e)[:80])
    try:
        out = run_audio_separator(a.inp, a.out, "", "vocals")
        emit(stage="done", frac=1.0, msg="已分出人声（MDX）", out=out, sec=round(time.time() - t0, 1))
        return 0
    except Exception as e:
        emit(stage="error", frac=0, msg="没有可用的分离后端：%s" % e)
        return 3


if __name__ == "__main__":
    sys.exit(main())

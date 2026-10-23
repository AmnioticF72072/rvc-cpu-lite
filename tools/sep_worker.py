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

    shifts=1 保持默认：提高它会成倍增加耗时，收益远小于 overlap。
    cpu 上再叠加 shifts 会让 4 分钟歌从 3 分钟变成十几分钟，不划算。
    """
    import subprocess
    import os as _os
    overlap = _os.environ.get("SEP_OVERLAP", "0.75")
    cmd = [sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems=vocals",
           "--overlap", str(overlap), "-o", out_dir, in_path]
    if _os.environ.get("SEP_DEVICE"):
        cmd[3:3] = ["-d", _os.environ["SEP_DEVICE"]]
    env = dict(_os.environ)
    env.setdefault("HF_HOME", _os.path.join(_os.path.expanduser("~"), ".cache"))
    env.pop("PYTHONPATH", None)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="ignore", env=env)
    buf = []
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        buf.append(line)
        if len(buf) > 40:
            del buf[:20]
        if "%" in line:
            emit(stage="separating", frac=0.5, msg="正在分离：" + line.strip()[:60])
    p.wait()
    stem = _os.path.splitext(_os.path.basename(in_path))[0]
    cand = _os.path.join(out_dir, "htdemucs", stem, "vocals.wav")
    if not _os.path.isfile(cand):
        for root, _d, fs in _os.walk(out_dir):
            for f in fs:
                if f == "vocals.wav":
                    cand = _os.path.join(root, f)
    if p.returncode != 0 or not _os.path.isfile(cand):
        raise RuntimeError("demucs 失败：" + "\n".join(buf[-6:]))
    return cand


def find_accompaniment(sep_out_dir, in_path):
    """分离后顺手把伴奏轨找出来（demucs --two-stems=vocals 会同时产出 no_vocals）。

    返回伴奏路径或空串。上层用它做"换完人声再合回去"——
    这才是翻唱的标准流程：伴奏必须保留原样，只换主唱。
    """
    stem = _os.path.splitext(_os.path.basename(in_path))[0]
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

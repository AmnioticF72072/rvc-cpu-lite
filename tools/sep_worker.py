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
    import subprocess
    cmd = [sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems=vocals",
           "-o", out_dir, in_path]
    if os.environ.get("SEP_DEVICE"):
        cmd[3:3] = ["-d", os.environ["SEP_DEVICE"]]
    env = dict(os.environ)
    env.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
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
        low = line.lower()
        if "%" in line:
            emit(stage="separating", frac=0.5, msg="正在分离：" + line.strip()[:60])
    p.wait()
    stem = os.path.splitext(os.path.basename(in_path))[0]
    cand = os.path.join(out_dir, "htdemucs", stem, "vocals.wav")
    if not os.path.isfile(cand):
        for root, _d, fs in os.walk(out_dir):
            for f in fs:
                if f == "vocals.wav":
                    cand = os.path.join(root, f)
    if p.returncode != 0 or not os.path.isfile(cand):
        raise RuntimeError("demucs 失败：" + "\n".join(buf[-6:]))
    return cand


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
        emit(stage="done", frac=1.0, msg="已分出人声", out=out, sec=round(time.time() - t0, 1))
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

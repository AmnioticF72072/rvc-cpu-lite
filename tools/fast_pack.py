# -*- coding: utf-8 -*-
"""快速打包：zip 纯拷贝（ZIP_STORED），跳过重复压缩。

为什么换 zip：
    rar 逐文件走 WinRAR 引擎，1.3GB 内容要几分钟；而 .onnx 模型本身
    已经是压缩过的二进制，再压几乎压不动 —— 压缩时间几乎全是白花的。
    ZIP_STORED 等于"带目录结构的拷贝"，速度接近磁盘复制。

中文文件名：
    zipfile 默认写 UTF-8 文件名并置 UTF-8 flag，Windows 10+ 资源管理器
    与 7-Zip / Bandizip 都能正确显示；解压后文件名不会乱码。
"""
import os
import shutil
import sys
import time
import zipfile

SRC_STAGE = r"G:\AIVoice\dist\Plus2.1\声音转换Plus"
CODE = r"G:\AIVoice\rvc-cpu-portable"
OUT_DIR = r"G:\AIVoice\dist\Plus2.2.1"
OUT_ZIP = os.path.join(OUT_DIR, "声音转换Plus_v2.2.1.zip")
TOP = "声音转换Plus"

# 需要同步的代码（相对 CODE）
FILES = [
    "gui.py", "convert.py", "requirements.txt",
    "app/core.py", "app/reading.py", "app/__init__.py",
    "rvc_cpu/__init__.py", "rvc_cpu/infer.py", "rvc_cpu/models.py",
    "rvc_cpu/separate.py", "rvc_cpu/mixdown.py", "rvc_cpu/hw_profile.py",
    "rvc_cpu/audio_io.py", "rvc_cpu/calibrate.py", "rvc_cpu/tts.py",
    "rvc_cpu/read.py", "rvc_cpu/ultra_lite.py", "rvc_cpu/mel.py",
    "rvc_cpu/rmvpe.py", "rvc_cpu/vec.py",
    "rvc_cpu/sep/__init__.py", "rvc_cpu/sep/worker.py", "rvc_cpu/sep/karaoke.py",
    "tools/sep_worker.py",
]
DOCS = ["docs/2.2-说明.md", "docs/2.2.1-修复说明.md", "docs/角色音色朗读.md", "docs/v2.0-说明.md"]

# 打包时跳过（垃圾/缓存/用户数据）
SKIP_DIRS = {"_工作区", "_测试残留", "__pycache__", ".git", "_old"}
SKIP_EXT = {".pyc", ".pyo", ".log", ".bak"}


def log(m):
    print(m, flush=True)


def sync_code():
    n = 0
    for rel in FILES + DOCS:
        src = os.path.join(CODE, rel.replace("/", os.sep))
        dst = os.path.join(SRC_STAGE, rel.replace("/", os.sep))
        if not os.path.isfile(src):
            log("  跳过（不存在）：" + rel)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        n += 1
    log("  同步 %d 个代码/文档文件" % n)


def clean():
    """清掉不该进包的东西。"""
    removed = 0
    for name in ("_工作区", "_测试残留"):
        p = os.path.join(SRC_STAGE, name)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    # 输出目录只留空壳
    out = os.path.join(SRC_STAGE, "输出")
    if os.path.isdir(out):
        for fn in os.listdir(out):
            fp = os.path.join(out, fn)
            try:
                shutil.rmtree(fp) if os.path.isdir(fp) else os.remove(fp)
                removed += 1
            except Exception:
                pass
    # pycache
    for root, dirs, _files in os.walk(SRC_STAGE):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
                removed += 1
    log("  清理 %d 处" % removed)


def build_zip():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.isfile(OUT_ZIP):
        os.remove(OUT_ZIP)

    files = []
    total = 0
    for root, dirs, fs in os.walk(SRC_STAGE):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in fs:
            if os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, SRC_STAGE)
            files.append((full, TOP + "\\" + rel))
            try:
                total += os.path.getsize(full)
            except OSError:
                pass

    log("  待打包 %d 个文件，%.0f MB" % (len(files), total / 1048576))

    t0 = time.time()
    CHUNK = 4 * 1024 * 1024
    done = 0
    last = 0
    # ZIP_STORED：不压缩，直接拷贝（模型压不动，压缩纯浪费时间）
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED,
                         allowZip64=True) as z:
        for full, arc in files:
            try:
                # 自己写 stream，避免 zipfile 用较小缓冲
                zi = zipfile.ZipInfo.from_file(full, arc)
                zi.compress_type = zipfile.ZIP_STORED
                with open(full, "rb") as src, z.open(zi, "w") as dst:
                    while True:
                        buf = src.read(CHUNK)
                        if not buf:
                            break
                        dst.write(buf)
                done += os.path.getsize(full)
            except OSError as e:
                log("  跳过（读不到）：%s  %s" % (arc, e))
            pct = int(done / max(total, 1) * 100)
            if pct >= last + 10:
                last = pct
                log("    %d%%  （%.0f MB，%.0f 秒）"
                    % (pct, done / 1048576, time.time() - t0))

    mb = os.path.getsize(OUT_ZIP) / 1048576
    log("\n完成：%s" % OUT_ZIP)
    log("  %.0f MB，耗时 %.0f 秒" % (mb, time.time() - t0))
    return OUT_ZIP


def verify():
    """抽检：条目数 + 关键文件是否在包里。"""
    with zipfile.ZipFile(OUT_ZIP) as z:
        names = z.namelist()
    log("\n校验：")
    log("  条目数 %d" % len(names))
    keys = ["启动界面(Plus).bat", "转换.bat", "gui.py", "app/core.py",
            "rvc_cpu/sep/worker.py", "rvc_cpu/mixdown.py",
            "runtime/_sep_libs/", "runtime/_idx_libs/", "runtime/_tts_libs/",
            "models/myvoice51.onnx", "models/rmvpe.onnx",
            "models/sep/torch_home/"]
    bad = []
    for k in keys:
        # zip 内部路径统一是正斜杠 —— 这里两种都接受，别只认反斜杠
        kk = k.replace("\\", "/")
        hit = any((kk in n) or (kk.replace("/", "\\") in n) for n in names)
        log("  %-34s %s" % (k, "OK" if hit else "缺失!"))
        if not hit:
            bad.append(k)
    return bad


if __name__ == "__main__":
    log("=" * 60)
    log("Plus 2.2 快速打包（zip 纯拷贝）")
    log("=" * 60)
    log("[1/4] 同步代码")
    sync_code()
    log("[2/4] 清理垃圾")
    clean()
    log("[3/4] 打包")
    build_zip()
    log("[4/4] 校验")
    bad = verify()
    if bad:
        log("\n有缺失项：" + "、".join(bad))
        sys.exit(1)
    log("\n全部正常")

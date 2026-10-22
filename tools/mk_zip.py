# -*- coding: utf-8 -*-
"""把 rvc-cpu-portable 打成 zip，放桌面。模型不压缩（压不动），代码/文档压缩。"""
import os
import zipfile
import time

SRC = r"G:\AIVoice\rvc-cpu-portable"
DST = r"C:\Users\Administrator\Desktop\声音转换Plus_v1.1.zip"
TOP = "声音转换Plus"          # zip 里的顶层文件夹名

# 不进包的目录/文件
EXCLUDE_DIRS = {"_工作区", "输出", "文档备份", "__pycache__", ".git", "归档"}
EXCLUDE_FILES = {"_ui_preview.html"}
# 这些后缀的模型文件直接存储（已是压缩数据）
STORE_EXT = (".onnx", ".index", ".pth", ".bin", ".png", ".wav", ".flac", ".zip")


def main():
    t0 = time.time()
    total = 0
    n = 0
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for root, dirs, files in os.walk(SRC):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                if f in EXCLUDE_FILES or f.endswith((".pyc", ".log")):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, SRC)
                arc = TOP + "/" + rel.replace("\\", "/")
                ext = os.path.splitext(f)[1].lower()
                if ext in STORE_EXT:
                    zi = zipfile.ZipInfo(arc)
                    zi.compress_type = zipfile.ZIP_STORED
                    with open(full, "rb") as fh:
                        z.writestr(zi, fh.read())
                else:
                    z.write(full, arc)
                total += os.path.getsize(full)
                n += 1
                if n % 20 == 0:
                    print("  %4d 个文件，已 %.1f MB" % (n, total / 1048576), flush=True)
    dt = time.time() - t0
    print("完成：%d 个文件，%.1f MB → 包 %.1f MB，耗时 %.1fs" %
          (n, total / 1048576, os.path.getsize(DST) / 1048576, dt))


if __name__ == "__main__":
    main()

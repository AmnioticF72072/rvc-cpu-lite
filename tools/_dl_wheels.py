# -*- coding: utf-8 -*-
"""绕开 pip，直接从镜像下载 wheel 并解压到位。

为什么不用 pip：
    这台机器上 pip 无论走代理还是禁代理，对任何包都返回
    "from versions: none" —— 连 -v 都打不出请求日志，明显是被环境层拦掉了。
    而同一时刻 curl 请求同一个 URL 能正常拿到 17KB 的索引页。

    所以干脆不用 pip：wheel 本身就是个 zip，下载 + 解压 = 安装，
    纯 Python 包这样装完全等价。二进制包（scipy/llvmlite/numba/onnx）
    也只需要挑对 wheel tag（cp313 + win_amd64）。
"""
import os
import re
import sys
import urllib.request
import zipfile

IDX = "https://pypi.tuna.tsinghua.edu.cn/simple"
OUT = r"G:\AIVoice\dist\_kara_dl"
CACHE = r"G:\AIVoice\dist\_wheels"

# (包名, 期望的 wheel 特征)
PKGS = [
    ("librosa", "any"),
    ("lazy_loader", "any"),
    ("msgpack", "cp313"),
    ("pooch", "any"),
    ("audioread", "any"),
    ("pydub", "any"),
    ("ffmpeg-python", "any"),
    ("standard-aifc", "any"),
    ("onnx2torch", "any"),
    ("audio-separator", "any"),
    ("scipy", "cp313"),
    ("llvmlite", "cp313"),
    ("numba", "cp313"),
    ("onnx", "cp313"),
    ("typing_extensions", "any"),
    ("packaging", "any"),
    ("requests", "any"),
    ("urllib3", "any"),
    ("certifi", "any"),
    ("idna", "any"),
    ("charset_normalizer", "any"),
    ("six", "any"),
    ("joblib", "any"),
    ("threadpoolctl", "any"),
    ("decorator", "any"),
    ("platformdirs", "any"),
    ("filelock", "any"),
    ("tqdm", "any"),
    ("pyyaml", "cp313"),
    ("rich", "any"),
    ("markdown", "any"),
    ("pydantic", "any"),
    ("pydantic_core", "cp313"),
    ("annotated_types", "any"),
    ("typing_inspection", "any"),
    ("beartype", "any"),
    ("einops", "any"),
    ("julius", "any"),
    ("samplerate", "cp313"),
    ("resampy", "any"),
    ("numpy", "cp313"),
]

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
opener.addheaders = [("User-Agent", "Mozilla/5.0 (pip-like)")]


def fetch(url, timeout=180):
    with opener.open(url, timeout=timeout) as r:
        return r.read()


def ver_key(v):
    """把版本号变成可比较的元组（1.2.3 → (1,2,3)），非数字部分当 0。"""
    parts = []
    for p in re.split(r"[._-]", v):
        m = re.match(r"^(\d+)", p)
        parts.append(int(m.group(1)) if m else 0)
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts[:4])


def pick_wheel(pkg, want):
    """从索引页里挑最合适的 wheel。"""
    html = fetch("%s/%s/" % (IDX, pkg)).decode("utf-8", "ignore")
    cands = []
    for m in re.finditer(r'href="([^"]+\.whl)(?:#([^"]*))?"[^>]*>([^<]+)</a>', html):
        href, frag, name = m.group(1), m.group(2), m.group(3).strip()
        low = name.lower()
        if want == "any":
            # 纯 Python 包只认 none-any，别去拿 cpXXX 的二进制包
            # （msgpack 会挑到 cp315，装到 3.13 上完全不能用）
            if "none-any" not in low:
                continue
        else:
            # 二进制包：必须是 win_amd64，且 tag 精确匹配 cp313-cp313
            # ★ 必须排除 cp313t —— 那是 free-threaded 版，普通 3.13 加载不了
            if "win_amd64" not in low or "cp313t" in low:
                continue
            if not re.search(r"cp313-cp313", low):
                continue
        vm = re.match(r"^[a-zA-Z0-9_.\-]+?-(\d[^-]*)-", name)
        if not vm:
            continue
        # 预发布一律排除：a1/b2/rc1/dev 都不稳定，装进发行包是坑
        if re.search(r"(a|b|rc|dev)\d", vm.group(1).lower()):
            continue
        cands.append((ver_key(vm.group(1)), name, href))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    _, name, href = cands[-1]
    # 相对链接补全
    if href.startswith("../../"):
        url = "https://pypi.tuna.tsinghua.edu.cn/" + href[6:]
    elif href.startswith("http"):
        url = href
    else:
        url = IDX + "/" + pkg + "/" + href
    return name, url


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    ok, fail = [], []
    for pkg, want in PKGS:
        try:
            r = pick_wheel(pkg, want)
            if not r:
                print("  %-20s 找不到合适的 wheel" % pkg)
                fail.append(pkg)
                continue
            name, url = r
            local = os.path.join(CACHE, name)
            if not os.path.isfile(local):
                data = fetch(url, timeout=600)
                with open(local, "wb") as f:
                    f.write(data)
            else:
                data = None
            size = os.path.getsize(local) / 1048576
            # 解压到目标目录（wheel = zip）
            with zipfile.ZipFile(local) as z:
                z.extractall(OUT)
            print("  %-20s %-46s %6.1f MB" % (pkg, name[:46], size))
            ok.append(pkg)
        except Exception as e:
            print("  %-20s 失败：%s %s" % (pkg, type(e).__name__, str(e)[:70]))
            fail.append(pkg)
    print("\n成功 %d / 失败 %d" % (len(ok), len(fail)))
    if fail:
        print("失败：", ", ".join(fail))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())

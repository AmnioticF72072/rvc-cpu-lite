# -*- coding: utf-8 -*-
"""模型管理中心：列出 / 切换 / 导入音色模型（Plus 1.2 新增）

背景：Plus 1.0/1.1 只认死文件名 myvoice51.onnx，换模型要手动改名覆盖，
      想用别人训练的模型很别扭。这个模块把 models 目录变成"模型库"：

  - 每个音色模型 = <名字>.onnx（生成器）+ 同名 .json（旁挂配置：tgt_sr/gen_T）
    + 可选同名 .index（特征检索）+ 可选同名 .pitch.json（音域画像，给自动校准用）
  - vec768_dyn.onnx / rmvpe.onnx 是全库共享的底座，不随模型切换
  - models/registry.json 记录"当前用哪个"和每个模型的导入来源

导入支持两种形态（GUI 上就是一个"导入模型"按钮）：
  ① .onnx（+ .json / .index）—— 别的导出工具产出的，直接复制进 models
  ② .pth —— RVC 官方/applio 的训练产物，调用 tools/export_gen_onnx2.py
     转成 ONNX 再注册（需要本机有装了 torch 的 python，没有就明确报错，
     不静默失败）。.pth 转换走 vendor 在 tools/_rvc_export_lib 里的官方
     SynthesizerTrnMsNSFsidM（RVC 2.2.231006，MIT），不依赖外部仓库。

只支持 v2（768 维 ContentVec）模型：v1 老模型用 256 维 hubert soft 底座，
本版没打包那个底座，导入时会明确拒绝并说明原因。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
REGISTRY = os.path.join(MODELS, "registry.json")
EXPORT_SCRIPT = os.path.join(ROOT, "tools", "export_gen_onnx2.py")
VENDORED_LIB = os.path.join(ROOT, "tools", "_rvc_export_lib")

# 底座模型（共享，不算音色模型）
_BASES = ("vec768_dyn", "hubert_v2_768_dyn", "vec768", "contentvec", "rmvpe")

_UNSAFE = re.compile(r"[\\/:*?\"<>|\x00-\x1f]")


def _sanitize(name):
    """模型名 safe 化：去掉路径字符，限长 40。允许中文。"""
    name = _UNSAFE.sub("_", (name or "").strip())
    name = re.sub(r"\s+", " ", name)[:40].strip(" .")
    return name


# ---------------------------------------------------------------- 注册表
def _load():
    reg = {"active": "", "models": {}}
    try:
        with open(REGISTRY, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            reg.update(d)
    except Exception:
        pass
    return reg


def _save(reg):
    os.makedirs(MODELS, exist_ok=True)
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=1)
    os.replace(tmp, REGISTRY)


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------- 查询
def _clean_stage(stage_dir):
    """清空导入暂存目录。**不用 os.remove**。

    沙箱注入的 sitecustomize 会把删除改写成"移入回收站"，
    回收站不可用时 fail-closed 直接抛异常 —— 结果就是"以为清了，
    其实一个都没删"。所以统一改用 shutil.move 挪到 _废件 里，
    挪失败也不影响主流程（顶多占点磁盘，下次覆盖）。
    """
    if not os.path.isdir(stage_dir):
        return
    trash = os.path.join(os.path.dirname(stage_dir), "_导入废件")
    for fn in os.listdir(stage_dir):
        src = os.path.join(stage_dir, fn)
        if not os.path.isfile(src):
            continue
        try:
            os.makedirs(trash, exist_ok=True)
            # 加时间戳避免同名覆盖
            dst = os.path.join(trash, "%s_%s" % (time.strftime("%H%M%S"), fn))
            shutil.move(src, dst)
        except Exception:
            try:
                os.remove(src)
            except Exception:
                pass


def _discard_partial(stem):
    """导入失败时清掉 models 里的半成品（同样只挪不删）。"""
    trash = os.path.join(MODELS, "_导入失败")
    for ext in (".onnx", ".json", ".index", ".pitch.json"):
        p = os.path.join(MODELS, stem + ext)
        if not os.path.isfile(p):
            continue
        try:
            os.makedirs(trash, exist_ok=True)
            shutil.move(p, os.path.join(trash, stem + ext))
        except Exception:
            try:
                os.remove(p)
            except Exception:
                pass


def list_models():
    """扫 models 目录，合并注册表元数据。返回列表（不含底座）。"""
    reg = _load()
    stems = []
    if os.path.isdir(MODELS):
        for fn in sorted(os.listdir(MODELS)):
            if not fn.lower().endswith(".onnx"):
                continue
            stem = fn[:-5]
            if stem.lower() in _BASES:
                continue
            stems.append(stem)

    out = []
    for stem in stems:
        j = os.path.join(MODELS, stem + ".json")
        ix = os.path.join(MODELS, stem + ".index")
        pitch = os.path.join(MODELS, stem + ".pitch.json")
        cfg = _read_json(j) if os.path.isfile(j) else {}
        meta = dict(reg["models"].get(stem) or {})
        mb = 0.0
        try:
            mb = round(os.path.getsize(os.path.join(MODELS, stem + ".onnx")) / 1024 ** 2, 1)
        except Exception:
            pass
        _ok, _why = _looks_like_onnx(os.path.join(MODELS, stem + ".onnx"))
        out.append({
            "name": stem,
            "mb": mb,
            "tgt_sr": cfg.get("tgt_sr") or cfg.get("sr") or "",
            "version": cfg.get("version") or "",
            "f0": bool(cfg.get("f0", True)),
            "index": os.path.basename(ix) if os.path.isfile(ix) else "",
            "index_mb": round(os.path.getsize(ix) / 1024 ** 2, 1) if os.path.isfile(ix) else 0,
            "pitch_profile": os.path.isfile(pitch),
            "source": meta.get("source") or "",
            "imported_at": meta.get("imported_at") or "",
            "origin": meta.get("origin") or "",
            # 坏文件要在列表里就暴露，而不是等用户选中、点转换时才炸
            "broken": (not _ok) or "",
        })

    good = [m for m in out if not m["broken"]]
    active = reg.get("active") or ""
    # 登记的那个已经坏了（文件被替换/损坏）→ 不能继续当默认
    if active and not any(m["name"] == active for m in good):
        active = ""
    if not active:
        # 优先内置的 myvoice51，否则第一个能用的
        for m in good:
            if m["name"] == "myvoice51":
                active = m["name"]
                break
        if not active and good:
            active = good[0]["name"]
    for m in out:
        m["active"] = (m["name"] == active)
    return out, active


def active_name():
    return list_models()[1]


def paths_for(name=""):
    """返回 {gen, vec, rmv, idx}。name 为空用当前激活模型。"""
    models, active = list_models()
    name = name or active
    pick = None
    for m in models:
        if m["name"] == name:
            pick = m
            break
    if pick is None and models:
        pick = models[0]
        name = pick["name"]

    def shared(*cands):
        for c in cands:
            p = os.path.join(MODELS, c)
            if os.path.isfile(p):
                return p
        return ""

    gen = os.path.join(MODELS, name + ".onnx") if name else ""
    if not os.path.isfile(gen):
        gen = ""
    # ★ index 只在**文件真的存在**时才返回。
    #   原来无条件拼路径，于是"激活一个没有 .index 的模型"（角色模型基本都
    #   不带 index）时，上层拿到一个不存在的路径去 os.path.getsize，
    #   直接抛 FileNotFoundError，把整个 /api/meta 打挂 ——
    #   界面拿不到模型列表，表现就是"导入后不显示、列表里点不了"。
    _idx = os.path.join(MODELS, name + ".index") if name else ""
    return {
        "gen": gen,
        "vec": shared("vec768_dyn.onnx", "hubert_v2_768_dyn.onnx", "vec768.onnx", "contentvec.onnx"),
        "rmv": shared("rmvpe.onnx"),
        "idx": _idx if (_idx and os.path.isfile(_idx)) else "",
    }


def set_active(name):
    models, active = list_models()
    if not any(m["name"] == name for m in models):
        raise RuntimeError("模型「%s」不存在" % name)
    if name == active:
        return {"ok": True, "active": name}
    reg = _load()
    reg["active"] = name
    _save(reg)
    return {"ok": True, "active": name}


def delete_model(name):
    """删除一个音色模型的 onnx/json/index/pitch.json（单个、显式调用）。"""
    models, active = list_models()
    if not any(m["name"] == name for m in models):
        raise RuntimeError("模型「%s」不存在" % name)
    if len(models) <= 1:
        raise RuntimeError("这是最后一个模型，删了就没了")
    removed = []
    for ext in (".onnx", ".json", ".index", ".pitch.json"):
        p = os.path.join(MODELS, name + ext)
        if os.path.isfile(p):
            os.remove(p)
            removed.append(os.path.basename(p))
    reg = _load()
    reg["models"].pop(name, None)
    if reg.get("active") == name:
        reg["active"] = ""
    _save(reg)
    return {"ok": True, "removed": removed}


# ---------------------------------------------------------------- 导入
def _has_module(py, mod):
    """不真的 import（torch 太慢），看 site-packages 里有没有目录。"""
    if not py or not os.path.isfile(py):
        return False
    base = os.path.dirname(os.path.dirname(py))
    for sp in (os.path.join(base, "Lib", "site-packages"),
               os.path.join(base, "lib", "python3.10", "site-packages"),
               os.path.join(base, "lib", "python3.11", "site-packages"),
               os.path.join(base, "lib", "python3.12", "site-packages")):
        if not os.path.isdir(sp):
            continue
        import glob
        if glob.glob(os.path.join(sp, mod + "*")):
            return True
    return False


def find_torch_python():
    """找一个装了 torch 的 python（转换 .pth 用）。找不到返回空串。

    为什么要找这么多地方：用户导入的角色模型绝大多数是 `.pth`，转 ONNX
    必须有 torch。原先只认三个硬编码路径，换台机器就找不到 ——
    表现是"导入失败"，用户完全不知道该装什么。

    注意：**打包版自带的 runtime 里没有 torch**（torch 有 2GB+，
    为了体积不内置）。所以这里是在"借"用户机器上已有的 Python 环境。
    """
    cands = [
        os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
        os.path.join(ROOT, "..", "env", "rvc", "Scripts", "python.exe"),
    ]
    # 常见的本机 Python（打包/训练工具链留下的）
    for base in (r"G:\AIVoice\env", os.path.expanduser(r"~\AppData\Local\Programs")):
        try:
            if os.path.isdir(base):
                for name in sorted(os.listdir(base)):
                    p = os.path.join(base, name, "Scripts", "python.exe")
                    if os.path.isfile(p):
                        cands.append(p)
        except OSError:
            pass
    # 系统 PATH 里的 python / python3
    try:
        import shutil
        for exe in ("python", "python3", "py"):
            w = shutil.which(exe)
            if w:
                cands.insert(0, w)
    except Exception:
        pass

    seen = set()
    for py in cands:
        if not py or py.lower() in seen:
            continue
        seen.add(py.lower())
        if os.path.isfile(py) and _has_module(py, "torch"):
            return py
    return ""


def torch_hint():
    """给界面用的：找不到 torch 时该告诉用户什么。"""
    if find_torch_python():
        return ""
    return ("这台电脑上没有找到装 PyTorch 的 Python，所以无法把 .pth 转成 ONNX。\n\n"
            "三个办法（挑一个）：\n"
            "① 找找分享包里有没有同名的 .onnx 文件 —— 很多模型两种格式都提供，"
            "直接导入 .onnx 不用转换，速度也快。\n"
            "② 装一个 PyTorch：pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu\n"
            "   装完重启本程序，再导入 .pth。\n"
            "③ 在别的有 torch 的电脑上转好 .onnx 再拿过来。")


# 导入的 .pth 统一按 T=1024 导出（与内置模型一致、经过完整验证）。
# T 太小会让分块步长 < 1.5s 最小段长，每块的 3s 上下文裁剪会把真声吃光
# （表现为"12 秒进去 1 秒出来"）。推理端也有 gen_T 自检兜底。
EXPORT_T = "1024"
# 直接导入 .onnx 时的 gen_T 下限：保证 max_step = (T-202)*160 > 1.5s 最小段长
GEN_T_MIN = 512


def _export_pth(pth_path, out_onnx, progress=None):
    """调 tools/export_gen_onnx2.py 把 .pth 转成 ONNX（需要 torch python）。"""
    def _p(f, m):
        if progress:
            try:
                progress(f, m)
            except Exception:
                pass

    py = find_torch_python()
    if not py:
        raise RuntimeError(
            "转 .pth 需要本机装有 PyTorch 的 Python，但没找到。\n"
            "两个办法：① 直接导入 .onnx 格式的模型（不用转换）；\n"
            "② 在有 torch 的电脑上跑 tools/export_gen_onnx2.py 得到 .onnx 再拿过来。")
    _p(0.15, "找到 PyTorch 环境，开始转换（大模型要几分钟）…")
    env = dict(os.environ)
    env["PYTHONPATH"] = VENDORED_LIB + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [py, EXPORT_SCRIPT, "--pth", pth_path, "--out", out_onnx, "--T", EXPORT_T]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           env=env, timeout=3600, cwd=ROOT)
    except subprocess.TimeoutExpired:
        raise RuntimeError("转换超时（1 小时）。模型太大或机器太慢")
    if r.returncode != 0 or not os.path.isfile(out_onnx):
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        raise RuntimeError("转换失败：%s" % (tail[-1] if tail else "未知错误"))
    # v1 老模型检查：v1 用 256 维 hubert 底座，本版只有 768 维 ContentVec
    side = _read_json(os.path.splitext(out_onnx)[0] + ".json")
    if side.get("version") == "v1":
        try:
            os.remove(out_onnx)
        except Exception:
            pass
        raise RuntimeError(
            "这是 v1 老模型（256 维特征）。本版只支持 v2 模型（768 维 ContentVec）。\n"
            "请下载标注 v2 的模型，或用完整版 RVC  WebUI 转换后再导出。")
    _p(0.55, "转换完成，正在登记…")


def _introspect_onnx(path):
    """不加载权重，从 ONNX 图里读 gen_T / vec_channels / tgt_sr（尽力而为）。"""
    info = {"gen_T": 1024, "vec_channels": 768, "tgt_sr": 48000}
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        for i in sess.get_inputs():
            shp = i.shape
            if i.name == "phone" and len(shp) == 3:
                try:
                    info["gen_T"] = int(shp[1])
                except Exception:
                    pass
                try:
                    info["vec_channels"] = int(shp[2])
                except Exception:
                    pass
        meta = sess.get_modelmeta().custom_metadata_map or {}
        if "config" in meta:
            try:
                info["tgt_sr"] = int(json.loads(meta["config"])[-1])
            except Exception:
                pass
        del sess
    except Exception:
        pass
    return info


def _looks_like_onnx(path):
    """廉价判断：文件像不像 ONNX（读前 16 字节）。

    ONNX 是 protobuf 序列化，合法模型开头基本都是 0x08（field1=ir_version, varint）。
    音频文件开头是 RIFF / ID3 / OggS 等 —— 一眼就能区分。
    这里不做完整校验（那要建 session，太重），只用来把明显不对的文件标记出来。
    """
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return False, "读不到文件"
    if not head:
        return False, "文件是空的"
    if head.startswith((b"RIFF", b"ID3", b"OggS", b"fLaC", b"FORM")):
        return False, "这其实是音频文件，不是模型"
    if head[0] == 0xFF and len(head) > 1 and (head[1] & 0xF0) == 0xF0:
        return False, "这其实是音频文件，不是模型"
    if head[:2] == b"PK":
        return False, "这是压缩包，请先解压出 .onnx / .pth"
    if head[0] != 0x08:
        return False, "文件内容不像 ONNX（可能已损坏）"
    return True, ""


def _verify_onnx(path, stem=""):
    """试加载一次 ONNX，确认它真的能当生成器用。

    只判扩展名是不够的 —— 把音频/压缩包改个名字叫 .onnx 也能通过。
    这里用最小 SessionOptions 建一次 session（不跑推理），失败就抛可读错误。
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 3
    try:
        sess = ort.InferenceSession(path, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    except Exception as e:
        msg = str(e)
        if "INVALID_PROTOBUF" in msg or "Protobuf parsing failed" in msg:
            raise RuntimeError(
                "「%s」不是有效的模型文件（解析失败）。\n"
                "可能是文件下载不完整、或者把音频/其他文件改名成了 .onnx。"
                "请重新获取模型，或直接导入 .pth 让它自动转换。" % (stem or "这个文件"))
        if "NO_SUCHFILE" in msg:
            raise RuntimeError("模型文件不存在或无法读取：%s" % (stem or path))
        raise RuntimeError("这个模型加载不了：%s" % msg[:140])

    # 生成器必须能接受 phone / pitch 类输入；名字不同但形状对也算通过
    names = [i.name.lower() for i in sess.get_inputs()]
    if not any(n in names for n in ("phone", "phone_lengths", "pitch", "pitchf")):
        raise RuntimeError(
            "「%s」看起来不是 RVC 生成器模型（输入层对不上）。\n"
            "请确认导入的是 .pth 权重或 RVC 导出的 .onnx，不是音色索引或其他模型。"
            % (stem or "这个文件"))
    return True


def _write_minimal_json(onnx_path, stem):
    """没有旁挂 .json 时，根据 ONNX 图推断着写一个（推理端优先读它）。"""
    info = _introspect_onnx(onnx_path)
    side = os.path.splitext(onnx_path)[0] + ".json"
    cfg = {"model_name": stem, "gen_T": info["gen_T"], "tgt_sr": info["tgt_sr"],
           "vec_channels": info["vec_channels"],
           "version": "v2" if info["vec_channels"] == 768 else "v1",
           "f0": True, "note": "由 Plus 导入时自动生成，tgt_sr 为推断值"}
    with open(side, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    return cfg


def import_from_stage(stage_dir, name="", progress=None):
    """从暂存目录导入模型。stage_dir 里放 .pth 或 .onnx（.json/.index/.pitch.json 可选）。

    返回导入成功的模型信息字典；失败抛 RuntimeError（消息直接给用户看）。
    """
    def _p(f, m):
        if progress:
            try:
                progress(float(f), str(m))
            except Exception:
                pass

    files = sorted(os.listdir(stage_dir)) if os.path.isdir(stage_dir) else []
    if not files:
        raise RuntimeError("没有待导入的文件")
    pth = [f for f in files if f.lower().endswith(".pth")]
    onnx = [f for f in files if f.lower().endswith(".onnx")]
    js = [f for f in files if f.lower().endswith(".json")]
    idx = [f for f in files if f.lower().endswith(".index")]
    if not pth and not onnx:
        raise RuntimeError("导入需要一个 .onnx 或 .pth 模型文件（.json / .index 可选）")

    base = name or (pth[0] if pth else onnx[0])
    stem = _sanitize(os.path.splitext(base)[0])
    if not stem:
        raise RuntimeError("模型名无效：%s" % base)

    # ★ 挑源文件必须按名字匹配，不能取"目录里第一个"。
    #   暂存目录可能残留上次的东西（导入失败、或用户连点两次），
    #   取 onnx[0] 会拿到旧文件，把 A 模型当成 B 模型导入，
    #   而且还会因为"目标已存在"直接失败 —— 用户看到的就是
    #   "导入没反应/导入失败"，完全不知道发生了什么。
    def _pick(exts, want):
        """在候选里找与 want 同名的文件；找不到就退回第一个（单文件场景）。"""
        want_low = want.lower()
        exact = [f for f in exts if f.lower() == want_low]
        if exact:
            return exact[0]
        same = [f for f in exts if os.path.splitext(f)[0].lower() ==
                os.path.splitext(want_low)[0]]
        if same:
            return same[0]
        return exts[0] if exts else ""

    want = os.path.basename(base)
    src_name = _pick(pth + onnx, want)
    src_file = os.path.join(stage_dir, src_name)

    # 校验来源文件真的是模型，不是把音频/图片改个后缀塞进来。
    # 这类文件扩展名对得上、但内容是音频，装进 models 后会出现在模型列表里，
    # 一选中就报 onnx 解析失败 —— 必须挡在门口。
    from .audio_io import sniff_audio
    kind = sniff_audio(src_file)
    if kind:
        raise RuntimeError(
            "这个文件其实是 %s 音频，不是音色模型。\n"
            "音色模型是 .pth 或 .onnx 文件。如果你想转换这段音频，"
            "请用「翻唱」上传音频，或到 models 目录之外存放。" % kind)

    is_pth = src_name.lower().endswith(".pth")
    dst_onnx = os.path.join(MODELS, stem + ".onnx")

    # ---- 同名冲突：分三种情况处理，别一律报错 ----
    # 用户重复导入同一个模型是最常见的场景（重装、换机器、连点两次），
    # 一律"已存在，请删掉再来"会让人很挫败，而且他根本不知道该怎么办。
    if os.path.exists(dst_onnx):
        same = False
        try:
            if os.path.getsize(dst_onnx) == os.path.getsize(src_file):
                # 大小一样再比内容（115MB 全比要 1~2 秒，可接受；
                # 只比大小会因为不同模型恰好同大小而误判）
                import hashlib

                def _h(p, chunk=1 << 20):
                    h = hashlib.sha256()
                    with open(p, "rb") as f:
                        while True:
                            b = f.read(chunk)
                            if not b:
                                break
                            h.update(b)
                    return h.digest()
                same = _h(dst_onnx) == _h(src_file)
        except Exception:
            same = False

        if same:
            # 完全相同的模型：不重复占磁盘，直接切成"使用中"
            _p(0.9, "模型已存在（内容相同），直接切换…")
            reg = _load()
            if stem not in reg["models"]:
                reg["models"][stem] = {"source": "onnx", "origin": src_name,
                                       "imported_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            reg["active"] = stem
            _save(reg)
            _clean_stage(stage_dir)
            _p(1.0, "完成")
            models, active = list_models()
            info = next((m for m in models if m["name"] == stem), {"name": stem})
            info["reused"] = True
            return info

        # 内容不同 → 自动换个名字，绝不覆盖别人已有的模型
        n = 2
        while os.path.exists(os.path.join(MODELS, "%s(%d).onnx" % (stem, n))):
            n += 1
        old_stem = stem
        stem = "%s(%d)" % (stem, n)
        dst_onnx = os.path.join(MODELS, stem + ".onnx")
        _p(0.05, "已有同名的不同模型，本次导入为「%s」…" % stem)

    os.makedirs(MODELS, exist_ok=True)
    try:
        if is_pth:
            _p(0.08, "开始转换 .pth → ONNX…")
            _export_pth(src_file, dst_onnx, _p)
        else:
            _p(0.25, "正在复制模型文件（%s）…" % src_name)
            shutil.copyfile(src_file, dst_onnx)

        # 旁挂配置：pitch 画像单独放，其余当 side config
        side_done = False
        for f in js:
            if "pitch" in f.lower():
                shutil.copyfile(os.path.join(stage_dir, f),
                                os.path.join(MODELS, stem + ".pitch.json"))
            else:
                shutil.copyfile(os.path.join(stage_dir, f),
                                os.path.join(MODELS, stem + ".json"))
                side_done = True
        if not side_done and not os.path.isfile(os.path.join(MODELS, stem + ".json")):
            _p(0.65, "生成旁挂配置…")
            _write_minimal_json(dst_onnx, stem)

        # gen_T 自检：太小会把真声裁光（详见 EXPORT_T 注释）
        side = _read_json(os.path.join(MODELS, stem + ".json"))
        try:
            gen_t = int(side.get("gen_T") or 0)
        except Exception:
            gen_t = 0
        if gen_t and gen_t < GEN_T_MIN:
            raise RuntimeError(
                "这个 ONNX 的定长帧数只有 %d（需要 ≥%d）。它是按 --T %d 导出的，"
                "请按 --T 1024 重新导出；或者直接导入 .pth，让本工具自动转换。"
                % (gen_t, GEN_T_MIN, gen_t))

        if idx:
            _p(0.75, "复制特征索引…")
            shutil.copyfile(os.path.join(stage_dir, idx[0]),
                            os.path.join(MODELS, stem + ".index"))

        # 真加载一次：扩展名对不代表内容对。
        # 这里花 1~3 秒换来的是"装进去的模型一定能用"，不然用户要等到
        # 点转换时才看到 onnx 解析失败，而且那时很难判断是模型的问题。
        _p(0.84, "验证模型可用性…")
        _verify_onnx(dst_onnx, stem)

        _p(0.9, "登记并切换…")
        reg = _load()
        reg["models"][stem] = {
            "source": "pth" if is_pth else "onnx",
            "imported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "origin": src_name,
        }
        reg["active"] = stem
        _save(reg)
    except Exception:
        # 失败清掉半成品，别留垃圾 —— 但**不能真删**：
        # 沙箱的 sitecustomize 会把删除改写成"移入回收站"，
        # 回收站不可用时 fail-closed 直接抛异常，结果文件留在原地，
        # 下次导入又报"已存在"。改成一律挪进 _废件 目录。
        _discard_partial(stem)
        raise
    finally:
        # 暂存区**无论成败都清空**。
        # 原来只在成功路径清，于是失败一次之后暂存里就一直留着旧文件，
        # 下次导入会拿它当源（把 A 当 B 装进去），或者直接报"已存在"，
        # 表现就是"导入点了没反应 / 一直失败"。
        _clean_stage(stage_dir)

    _p(1.0, "完成")
    models, active = list_models()
    info = next((m for m in models if m["name"] == stem), {"name": stem})
    return info

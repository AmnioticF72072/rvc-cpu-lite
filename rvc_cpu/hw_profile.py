# -*- coding: utf-8 -*-
"""硬件探测与推理 profile 自动选择（自研核心模块）

三种 profile：
  alder-lake-cpu   i5-12400 等 Alder Lake 纯 P 核：ORT 线程绑物理核 + VNNI 量化
  lunar-lake-ovino U5 226V 等 Lunar Lake：OpenVINO 多设备（ContentVec→NPU/GPU，
                   生成器→GPU，RMVPE 固定 CPU——核显跑 f0 会出错的已知坑）
  fallback-cpu     其他环境：通用 CPU 配置
"""
import os
import platform
import re


def _cpu_brand():
    """读 CPU 全名（Windows 注册表 / Linux proc，双平台）。"""
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0]
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def _physical_cores_hint(logical):
    """从 CPU 型号推断物理核数（Windows 拿不到就按 HT 系数 2 估）。

    ★ 低端处理器（赛扬/奔腾/Atom）不能按 logical//2 估：
      它们大多是 **2 核 2 线程**（无超线程），logical//2 会算成 1 核，
      然后线程数被压到 1 —— 白白浪费掉一半算力，速度直接砍半。
    """
    brand = _cpu_brand().lower()

    # 赛扬 / 奔腾 / Atom：看型号里的真实核数，估不出来就按逻辑核算
    if any(k in brand for k in ("celeron", "pentium", "atom", "athlon")):
        # 桌面级 Celeron G 系列多为 2C/2T
        if re.search(r"(g\d{3,4}|j\d{3,4}|n\d{3,4})", brand):
            return max(1, min(logical, 2))
        return max(1, logical)

    m = re.search(r"i[3579]-(\d{4,5})", brand)
    if m:
        num = m.group(1)
        # Intel 12/13/14 代：12400=6C、12600K=6P+4E、13600K=6P+8E…
        if num.startswith(("12", "13", "14")):
            suffix = num[-2:] if len(num) == 5 else num[-1]
            # 4xx/5xx/6xx 开头通常 6 P 核
            if suffix[0] in "456":
                return 6
    return max(1, logical // 2)


# ---- 低端处理器识别（赛扬 / 奔腾 / Atom 等 2 核机器）----
_LOWEND_KEYS = ("celeron", "pentium", "atom", "athlon", "e1-", "e2-", "n3050",
                "n3060", "n3160", "n3350", "n3450", "n4000", "n4020", "n4120",
                "n4500", "n5000", "n5095", "n5100", "n6000", "j1900", "j3455",
                "j4125", "j5005", "j6412")


def is_lowend_cpu():
    """是不是低端小核处理器（赛扬/奔腾/Atom）。

    判断依据两条，任一命中即算：
      1. 型号名里带 celeron / pentium / atom / 具体低端型号
      2. 物理核数 ≤ 2

    为什么要单独识别：这类机器不是"慢一点"，是**慢一个数量级**，
    而且内存和缓存都很小。默认配置（6 线程、大 overlap 分离）在上面
    会跑到几十分钟甚至 OOM，必须换一套参数。
    """
    # 允许手工强制：有些 OEM 机器的 CPU 型号字符串很怪，自动判断会漏。
    # 设环境变量 RVC_LOWEND=1 就按低端处理（写进 启动界面.bat 也行）。
    force = os.environ.get("RVC_LOWEND", "").strip().lower()
    if force in ("1", "true", "yes", "on"):
        return True
    if force in ("0", "false", "no", "off"):
        return False

    brand = _cpu_brand().lower()
    if any(k in brand for k in _LOWEND_KEYS):
        return True
    logical = os.cpu_count() or 4
    return logical <= 2


def detect_avx2_risk():
    """估计"这台机器可能连 AVX2 都没有"的风险。

    为什么要估：onnxruntime 官方 build 要求 AVX2。Atom 血统的赛扬
    （N4000/N4020/N4120/N5100 这类）**只有 SSE4.2，没有 AVX/AVX2**，
    上去会直接崩在 "Illegal instruction" 或加载 DLL 失败，
    而且报错完全看不出是 CPU 不支持。

    返回 (风险等级, 说明)。等级："" 无风险、"warn" 可能有问题、"high" 很可能不行。
    """
    brand = _cpu_brand().lower()
    # Atom 血统：N/J 系列 + 老的 Atom
    if re.search(r"\b(n\d{3,4}|j\d{3,4}|atom)\b", brand) or "atom" in brand:
        return "high", ("这台 CPU 是 Atom 血统（%s），只有 SSE4.2、没有 AVX2。"
                        "本推理引擎依赖 AVX2，很可能直接跑不起来。" % _cpu_brand())
    if "celeron" in brand or "pentium" in brand:
        return "warn", ("这台 CPU 是 %s。如果属于 N/J 系列（Atom 血统），"
                        "可能缺少 AVX2 指令集，遇到崩溃请换用支持 AVX2 的机器。"
                        % _cpu_brand())
    return "", ""


def detect_avx_vnni():
    """AVX-VNNI 支持（Alder Lake+）。Windows 上用 PowerShell 查不算快，
    用特征启发：型号带 12/13/14 代 Intel Core 即认为支持。"""
    brand = _cpu_brand().lower()
    if "intel" not in brand:
        return False
    m = re.search(r"(?:i[3579]|[3579])-(\d{4,5})|core\s+(?:ultra\s+)?([579])", brand)
    if m:
        gen = m.group(1) or ""
        if gen.startswith(("12", "13", "14")):
            return True
        if m.group(2):  # Core Ultra（Meteor/Lunar Lake）也支持
            return True
    return False


def detect_openvino_devices():
    """枚举 OpenVINO 设备。返回 (openvino可用, 设备列表)。"""
    try:
        import openvino as ov
        core = ov.Core()
        return True, list(core.available_devices)
    except Exception:
        return False, []


def pick_profile():
    """自动选 profile。返回 dict（含策略描述）。"""
    logical = os.cpu_count() or 4
    physical = _physical_cores_hint(logical)
    brand = _cpu_brand()
    ov_ok, ov_devices = detect_openvino_devices()
    vnni = detect_avx_vnni()
    lowend = is_lowend_cpu()
    avx2_level, avx2_note = detect_avx2_risk()

    has_gpu = any("GPU" in d for d in ov_devices)
    has_npu = any("NPU" in d for d in ov_devices)

    if ov_ok and (has_gpu or has_npu):
        profile = "lunar-lake-ovino"
        detail = {
            "engine": "openvino",
            "contentvec_device": "NPU" if has_npu else "GPU",
            "generator_device": "GPU",
            "f0_device": "CPU",  # 已知坑：RMVPE 在 OpenVINO GPU 上输出错误
            "intra_threads": max(2, logical // 2),
        }
    elif lowend:
        # 低端小核（赛扬/奔腾/2 核）专用配置。
        # 这类机器坑最多，逐条说明：
        #   线程数 = 逻辑核数本身。它没有超线程，按"物理核"算会算成 1 核，
        #             白白浪费一半算力；但也不能超配，超过核数只会互相抢。
        #   切块大小不变（GEN_T 由 ONNX 模型定死，改不了），
        #     但把并发压到 1，避免任务之间抢内存。
        #   分离的 overlap 会由上层降到 0.3 左右（见 separate 相关逻辑）。
        profile = "lowend-2core"
        detail = {
            "engine": "onnxruntime-cpu",
            "intra_threads": max(1, min(logical, 2)),
            "inter_threads": 1,
            "vnni_hint": False,
            "lowend": True,
            # 给界面用的预期速度（实测外推，宁可说慢也别让人等崩）
            "speed_note": "这台机器是低端双核，处理速度约为 6 核 i5 的 1/3~1/4，"
                          "4 分钟的歌大概要 15~25 分钟。建议关闭其他程序腾出内存。",
        }
    else:
        profile = "alder-lake-cpu" if vnni else "fallback-cpu"
        detail = {
            "engine": "onnxruntime-cpu",
            "intra_threads": physical,   # 绑物理核，避免 HT 同步屏障反效应
            "inter_threads": 1,
            "vnni_hint": vnni,
        }
    return {
        "profile": profile,
        "cpu": brand,
        "logical": logical,
        "physical": physical,
        "openvino": ov_ok,
        "ov_devices": ov_devices,
        "lowend": lowend,
        "avx2_risk": avx2_level,
        "avx2_note": avx2_note,
        **detail,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(pick_profile(), ensure_ascii=False, indent=2))

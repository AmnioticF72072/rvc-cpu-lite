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
    """从 CPU 型号推断物理核数（Windows 拿不到就按 HT 系数 2 估）。"""
    brand = _cpu_brand().lower()
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
        **detail,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(pick_profile(), ensure_ascii=False, indent=2))

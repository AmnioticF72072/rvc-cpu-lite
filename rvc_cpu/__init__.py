# -*- coding: utf-8 -*-
"""rvc_cpu：纯 ONNX 的 RVC 推理实现。

这里做一件事：把「按需依赖」所在的目录挂进 sys.path。

为什么需要这个：
    打包版内置了便携 Python（runtime/），但 faiss（特征检索，约 120MB）
    和 edge-tts（朗读载体）不是人人都会用到的东西，塞进主 site-packages
    会拖慢启动、也不利于按功能取舍。所以它们放在 runtime 下的独立子目录，
    用到时再挂进来。

    这么做顺手修掉了一个隐蔽 bug：以前打包版**根本没有 faiss**，
    `.index` 特征检索一直静默失败（只打一行警告），
    而 index 恰恰是决定「音色像不像目标角色」的最关键一步。
"""
from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PKG_DIR)

# runtime 下的按需依赖目录（打包版才有；源码跑的时候不存在，跳过即可）
_EXTRA_LIBS = [
    "_idx_libs",    # faiss-cpu：.index 特征检索
    "_tts_libs",    # edge-tts + aiohttp：朗读载体合成
]


def _mount_extra_libs():
    for name in _EXTRA_LIBS:
        for base in (os.path.join(_ROOT, "runtime"),
                     os.path.dirname(sys.executable)):
            p = os.path.join(base, name)
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)


_mount_extra_libs()


def has_index_backend():
    """特征检索后端是否可用。界面用它决定要不要提示用户。"""
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False

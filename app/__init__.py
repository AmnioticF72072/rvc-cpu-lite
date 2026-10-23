# -*- coding: utf-8 -*-
"""v2.0 应用层：状态、管线生命周期、任务编排按职责分文件。

分层的意义在于「加功能不必动别人的代码」：
  core.py     状态 + 管线（谁都要用，但不能被业务逻辑污染）
  reading.py  朗读任务（唯一需要 TTS 的地方）
  features.py 能力的自述与探测（前端靠它渲染菜单）
"""
from .core import (OUT, MODELS, VERSION, WORK, ensure_pipeline, get_state,
                   invalidate, log, meta, model_paths, pipe_loaded, set_state)

__all__ = ["OUT", "MODELS", "VERSION", "WORK", "ensure_pipeline", "get_state",
           "invalidate", "log", "meta", "model_paths", "pipe_loaded", "set_state"]

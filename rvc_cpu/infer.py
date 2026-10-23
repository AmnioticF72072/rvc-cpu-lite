# -*- coding: utf-8 -*-
"""RVC 纯 CPU/核显 ONNX 推理（深度定制版）

流水线: 输入→16k→ContentVec(2x nearest 帧对齐)→RMVPE f0→生成器→模型采样率输出
- 数值流程遵循 RVC 官方 pipeline.py（f0 Mel 量化、protect 混合、RMS 包络混合）
- 骨架借鉴 dev6699/rvc-onnx（MIT，mel/rmvpe/vec 三模块直接引用，见 NOTICE）
- 硬件 profile 见 hw_profile.py（i5-12400 / Lunar Lake 分别定制）

依赖: numpy onnxruntime soundfile soxr scipy librosa（无 torch）
"""
import json
import os
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr

from .hw_profile import pick_profile
from .rmvpe import RMVPE
from .vec import ContentVec

SR16K = 16000
WINDOW = 160              # 16k 下每帧样本数（HuBERT hop）
F0_MIN, F0_MAX = 50.0, 1100.0
X_PAD, X_QUERY, X_CENTER, X_MAX = 1, 1, 6, 8     # 定长友好参数（官方 3/10/50/50 的 CPU 定制版）：
                                                  # 相邻切点最大跨度 = X_CENTER + 2*X_QUERY = 8s，
                                                  # 加 2*t_pad 上下文 = 10s = 1000 帧 ≤ GEN_T(1024)；
                                                  # 上下文只要 1s（HiFiGAN 感受野足够），重叠计算浪费更少
GEN_T = 1024                # 生成器 ONNX 定长帧数（必须与导出 T 一致）


def _highpass_zero_phase(audio, fc=48.0, order=5, sr=SR16K):
    """FFT 域零相位高通（等效官方 butter(5,48)+filtfilt 的效果，纯 numpy）。

    butterworth 高通幅度平方响应：|H|² = 1/(1+(fc/f)^(2*order))，
    filtfilt = 前向+反向滤波 = |H|² 且零相位，与 FFT 域直接乘 |H|² 等效。
    """
    n = len(audio)
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    resp = np.ones_like(freqs)
    nz = freqs > 0
    resp[nz] = 1.0 / (1.0 + (fc / freqs[nz]) ** (2 * order))
    resp[~nz] = 0.0  # DC 直接归零
    return np.fft.irfft(spec * resp, n=n).astype(np.float32)


# ---------------------------------------------------------------- session 工厂
def _cpu_session(model_path, intra_threads):
    opts = ort.SessionOptions()
    # BASIC：实测比 DISABLE_ALL 快 1.15x、加载快 27%，且不碰 Where 算子坑
    # （EXTENDED / ALL 收益反而更小，还更容易踩算子兼容问题，所以停在 BASIC）
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


# 可选加速后端：按"覆盖面 → 速度"排序。
# 这三个都是 onnxruntime 的额外运行时包，装了才会出现在 get_available_providers() 里，
# 没装就自然落到 CPU —— 所以同一份代码在两种机器上都能跑，不需要分支版本。
_ACCEL = [
    ("DmlExecutionProvider", "DirectML(GPU)"),     # onnxruntime-directml，任意 DX12 显卡
    ("CUDAExecutionProvider", "CUDA(GPU)"),        # onnxruntime-gpu，NVIDIA
    ("OpenVINOExecutionProvider", "OpenVINO"),     # onnxruntime-openvino，Intel 核显/NPU
]


def available_accel():
    """列出当前 onnxruntime 真正注册了的加速后端。"""
    have = set(ort.get_available_providers())
    return [(ep, label) for ep, label in _ACCEL if ep in have]


def _ep_session(model_path, intra_threads, eps):
    """带加速后端的 session。CPU 永远排在最后兜底。"""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    providers = list(eps) + ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def _ov_compile(model_path, device, num_threads=4):
    """OpenVINO：首次自动转 IR 并缓存（同目录 .xml/.bin），之后秒加载。"""
    import openvino as ov
    ir = os.path.splitext(model_path)[0] + ".xml"
    if not os.path.exists(ir):
        core = ov.Core()
        ov.save_model(core.read_model(model_path), ir)
    core = ov.Core()
    cfg = {"PERFORMANCE_HINT": "THROUGHPUT"}
    if device == "CPU":
        cfg["INFERENCE_NUM_THREADS"] = str(num_threads)
    return core.compile_model(ir, device, cfg)


def _side_info(gen_path):
    """读导出时写的旁挂 JSON（tgt_sr / gen_T / version）。比 onnx.load 省 1G+ 内存。"""
    side = os.path.splitext(gen_path)[0] + ".json"
    if os.path.isfile(side):
        try:
            with open(side, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _meta_sr(gen_path):
    """目标采样率：优先旁挂 JSON，其次 ONNX 元数据，最后按 48k 兜底。"""
    info = _side_info(gen_path)
    if "tgt_sr" in info:
        try:
            return int(info["tgt_sr"])
        except Exception:
            pass
    try:
        meta = ort.InferenceSession(gen_path, providers=["CPUExecutionProvider"]) \
            .get_modelmeta().custom_metadata_map
        if "config" in meta:
            return int(json.loads(meta["config"])[-1])
    except Exception:
        pass
    return 48000


# ---------------------------------------------------------------- 主流水线
class RvcPipeline:
    def __init__(self, gen_path, vec_path, rmvpe_path,
                 index_path="", index_rate=0.0,
                 rms_mix_rate=0.40, protect=0.33, verbose=True, device="auto"):
        self.verbose = verbose
        self.profile = pick_profile()
        p = self.profile

        # ---- 设备选择：auto / cpu / gpu ----
        # auto：有加速后端就用，没有就 CPU（对用户零心智负担）
        # cpu ：强制纯 CPU（怀疑显卡驱动有问题时用）
        # gpu ：优先加速，但没有就退回 CPU 并明确告诉用户，绝不报错退出
        accel = available_accel()
        self.accel = accel
        used = []
        if device == "cpu" or not accel:
            self._eps = []
            self.device_used = "CPU"
            if device == "gpu" and not accel:
                self.device_used = "CPU（未安装 GPU 加速包，已自动退回）"
        else:
            self._eps = [ep for ep, _ in accel]
            used = [label for _, label in accel]
            self.device_used = " + ".join(used)
        if self._eps:
            p["engine"] = "onnxruntime-ep"      # 走加速后端，就不再走 OpenVINO 分支
        elif device == "cpu":
            p["engine"] = "onnxruntime-cpu"

        if verbose:
            print("[硬件] %s | 逻辑核 %d(估物理 %d) | profile=%s" %
                  (p["cpu"], p["logical"], p["physical"], p["profile"]))
            print("[设备] %s%s" % (self.device_used,
                                  "" if self._eps else "（纯 CPU 路径，功能完全一致）"))
            if p.get("openvino") and not self._eps:
                print("[硬件] OpenVINO 设备: %s" % p["ov_devices"])

        self.tgt_sr = _meta_sr(gen_path)
        self.gen_T = int(_side_info(gen_path).get("gen_T", GEN_T))   # 定长帧数（与导出一致）
        self.window = WINDOW
        self.t_pad = SR16K * X_PAD
        self.t_pad_tgt = self.tgt_sr * X_PAD
        self.t_pad2 = self.t_pad * 2
        self.t_query = SR16K * X_QUERY
        self.t_center = SR16K * X_CENTER
        self.t_max = SR16K * X_MAX
        # gen_T 自检：convert_file 的分块步长 max_step = (gen_T - t_pad2/window - 2)*window
        # 必须大于最小段长 1.5s，否则每块的 3s 上下文裁剪会把真声吃光
        # （"12 秒进去 1 秒出来"）。gen_T 太小（如 256）的模型在这里直接拒绝。
        _max_step = (self.gen_T - self.t_pad2 // self.window - 2) * self.window
        if _max_step < int(1.5 * SR16K):
            raise RuntimeError(
                "模型定长帧数 gen_T=%d 太小（分块步长只有 %.2f 秒，小于最小段长 1.5 秒）。"
                "这个 ONNX 是按 --T %d 导出的，请按 --T 1024 重新导出，"
                "或直接导入 .pth 让本工具自动转换。" %
                (self.gen_T, _max_step / SR16K, self.gen_T))
        # 默认值对齐 docs/RECIPES.md：
        #   protect 0.33→调用方传 0.10（唱歌），rms_mix_rate 0.25→0.40
        #   rms_mix_rate 就是 volume_envelope 的实现：越大越跟随源音量包络。
        #   0.4 = 主要跟源，1.0 = 完全听模型（句间会自我衰减，忽大忽小）。
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect
        self.index = None
        self.index_rate = 0.0
        self._index_path = index_path or ""   # ensure_index() 按需加载时要用

        # --- 生成器 ---
        if p["engine"] == "openvino":
            dev = p.get("generator_device", "GPU")
            if verbose:
                print("[加载] 生成器 → OpenVINO %s : %s" % (dev, os.path.basename(gen_path)))
            self.gen = _ov_compile(gen_path, dev, p["intra_threads"])
            self._gen_names = [i.get_any_name() for i in self.gen.inputs]
            self._gen_run = self._run_ov
        else:
            # INT8 量化版优先：models/quant/<名>.int8.onnx 存在就用它。
            # 实测 MatMul/Gemm 动态量化 1.29x、SNR 38.6dB（几乎不可闻）。
            # 找不到就退回 FP32，行为跟以前完全一致。
            use_gen = gen_path
            q = os.path.join(os.path.dirname(gen_path), "quant",
                             os.path.splitext(os.path.basename(gen_path))[0] + ".int8.onnx")
            if os.path.isfile(q):
                use_gen = q
            if verbose:
                print("[加载] 生成器 → %s (intra=%d) : %s%s" %
                      (self.device_used, p["intra_threads"],
                       os.path.basename(use_gen),
                       "  [INT8]" if use_gen != gen_path else ""))
            self.gen = self._session(use_gen)
            self._gen_names = [i.name for i in self.gen.get_inputs()]
            self._gen_run = self._run_ort
            self.gen_int8 = (use_gen != gen_path)

        # --- ContentVec ---
        if p["engine"] == "openvino":
            dev = p.get("contentvec_device", "GPU")
            if verbose:
                print("[加载] ContentVec → OpenVINO %s" % dev)
            self.vec = _ov_compile(vec_path, dev, p["intra_threads"])
            self._vec_names = [i.get_any_name() for i in self.vec.inputs]
            self._vec_run = self._run_ov
        else:
            if verbose:
                print("[加载] ContentVec → %s" % self.device_used)
            self.vec = self._session(vec_path)
            self._vec_names = [i.name for i in self.vec.get_inputs()]
            self._vec_run = self._run_ort

        # --- RMVPE（任何 profile 固定 CPU：OpenVINO GPU 上 f0 会出错的已知坑）---
        if verbose:
            print("[加载] RMVPE → ORT CPU（f0 永不跑核显）")
        self.rmvpe = RMVPE(rmvpe_path, providers=["CPUExecutionProvider"])
        self.rmvpe_path = rmvpe_path      # 朗读流程复用它量载体/输出音高

        # --- 特征索引（可选）---
        if index_path and os.path.isfile(index_path) and index_rate > 0:
            try:
                import faiss
                self.index = faiss.read_index(index_path)
                # ★ 关键：IVF 索引默认没有 direct map，reconstruct_batch(id) 会报
                #   "direct map not initialized"，检索就被静默跳过了（音色等于没拉）。
                #   打开 direct map 之后才取得到向量。
                try:
                    if hasattr(self.index, "make_direct_map"):
                        self.index.make_direct_map()
                except Exception as e:
                    print("[警告] make_direct_map 失败: %s" % e)
                try:
                    # nprobe 越小越准但越慢；RVC 官方用 1，这里给 8 兼顾
                    self.index.nprobe = int(os.environ.get("RVC_NPROBE", "8"))
                except Exception:
                    pass
                self.index_rate = index_rate
                if verbose:
                    print("[加载] 特征索引 %s（%d 条）" % (index_path, getattr(self.index, "ntotal", 0)))
            except Exception as e:
                print("[警告] index 加载失败，跳过检索: %s" % e)

    def _session(self, path):
        """按当前设备建 session：有加速后端就用，否则纯 CPU。"""
        if self._eps:
            try:
                return _ep_session(path, self.profile["intra_threads"], self._eps)
            except Exception as e:
                if self.verbose:
                    print("[警告] %s 加载失败，退回 CPU: %s" % (self.device_used, str(e)[:80]))
                self._eps = []
                self.device_used = "CPU（加速后端加载失败，已退回）"
        return _cpu_session(path, self.profile["intra_threads"])

    # ---- 运行时适配 ----
    @staticmethod
    def _run_ort(session, names, feeds):
        return session.run(None, {n: feeds[n] for n in names})[0]

    @staticmethod
    def _run_ov(compiled, names, feeds):
        res = compiled(feeds)
        return res[next(iter(res.values()))]

    # ---- 预处理（官方流程：归一化 + 48Hz 高通，不做补边）----
    def _preprocess(self, audio):
        m = np.abs(audio).max() / 0.95
        if m > 1:
            audio = audio / m
        return _highpass_zero_phase(audio, fc=48.0, order=5)

    def _opt_cut_points(self, audio, audio_pad):
        opt = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i:i + audio.shape[0]])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                w = audio_sum[t - self.t_query: t + self.t_query]
                opt.append(t - self.t_query + int(np.argmin(w)))
        return opt

    def _extract_f0(self, audio_pad, f0_up_key):
        f0 = self.rmvpe.infer_from_audio(audio_pad.astype(np.float32), thred=0.03)
        f0 = f0 * (2 ** (f0_up_key / 12.0))
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        lo = 1127.0 * np.log(1 + F0_MIN / 700)
        hi = 1127.0 * np.log(1 + F0_MAX / 700)
        f0_mel = (f0_mel - lo) * 254.0 / (hi - lo) + 1.0
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch = np.rint(f0_mel).astype(np.int64)
        p_len = audio_pad.shape[0] // self.window
        pitchf = f0[:p_len].reshape(1, -1).astype(np.float32)   # 连续 Hz (1,P)
        pitch = pitch[:p_len].reshape(1, -1)                     # 量化槽位 (1,P)
        return pitchf, pitch

    # ---- ContentVec 特征（整段喂入，官方 2x nearest 对齐；HuBERT 输出帧数=len/160，与 p_len 严格等长）----
    def _vec_feats(self, audio_pad, p_len):
        # 不同导出版本的 ContentVec 输入名不同：
        #   - applio transformers 导出：单个 "wav" [1, N]
        #   - 官方 hubert 导出："input_values" + "padding_mask"
        #   - dev6699 导出版本：任意两个输入
        # → 按实际输入名动态构造 feeds，不再硬编码下标。
        wav = audio_pad.astype(np.float32)[np.newaxis, :]
        feeds = {}
        for n in self._vec_names:
            low = n.lower()
            if "mask" in low:
                feeds[n] = np.zeros((1, len(audio_pad)), dtype=bool)
            elif "length" in low:
                feeds[n] = np.array([len(audio_pad)], dtype=np.int64)
            else:
                feeds[n] = wav
        feats = self._vec_run(self.vec, self._vec_names, feeds)       # (1, T, 768) @50fps
        feats = np.repeat(feats, 2, axis=1)                        # → 100fps（官方 nearest 2x）
        feats = feats.transpose(0, 2, 1)[:, :, :p_len]             # (1, 768, P) 截断到 p_len
        if feats.shape[2] < p_len:                                  # 极端长音频末尾补齐
            feats = np.pad(feats, ((0, 0), (0, 0), (0, p_len - feats.shape[2])))
        return feats.astype(np.float32)

    # ---- index 检索（可选）----
    def ensure_index(self, rate):
        """按需加载 / 更新特征检索。

        为什么要单独一个方法：构造时如果 index_rate=0，索引根本不会加载
        （省内存）。但翻唱用 0.5、朗读用 0.75，同一进程里切换时如果只改
        self.index_rate，self.index 仍然是 None，检索就被静默跳过 ——
        表现是"参数调了但音色没变"。这里在需要时补加载。
        """
        self.index_rate = float(rate)
        if self.index_rate <= 0 or self.index is not None:
            return self.index is not None
        idx_path = getattr(self, "_index_path", "")
        if not idx_path or not os.path.isfile(idx_path):
            return False
        try:
            import faiss
            self.index = faiss.read_index(idx_path)
            try:
                if hasattr(self.index, "make_direct_map"):
                    self.index.make_direct_map()
            except Exception:
                pass
            try:
                self.index.nprobe = int(os.environ.get("RVC_NPROBE", "8"))
            except Exception:
                pass
            if self.verbose:
                print("[加载] 特征索引（按需）%s（%d 条）"
                      % (os.path.basename(idx_path),
                         getattr(self.index, "ntotal", 0)))
            return True
        except Exception as e:
            print("[警告] index 加载失败，跳过检索: %s" % e)
            self.index = None
            return False

    def _index_search(self, feats, p_len):
        if self.index is None or self.index_rate <= 0:
            return feats
        try:
            feat = feats[0].T                                 # (P, 768)
            xo = self.index.search(feat.astype(np.float32), k=2)[1][0]
            xs = self.index.reconstruct_batch(xo[0]) if hasattr(self.index, "reconstruct_batch") \
                else np.vstack([self.index.reconstruct(int(i)) for i in xo[0]])
            # 官方 n_shallow_search 简化版：大权重取第一名
            blend = np.zeros_like(feat)
            blend[:] = xs[:1]
            out = feat * (1 - self.index_rate) + blend * self.index_rate
            return out.T[np.newaxis].astype(np.float32)
        except Exception as e:
            if self.verbose:
                print("[警告] index 检索失败，跳过: %s" % e)
            return feats

    # ---- 单块推理 ----
    def _run_chunk(self, audio_pad, pitch, pitchf, sid):
        p_len = min(pitch.shape[1], audio_pad.shape[0] // self.window)
        pitch = pitch[:, :p_len]
        pitchf = pitchf[:, :p_len]

        feats = self._vec_feats(audio_pad, p_len)             # (1,768,P)
        feats = self._index_search(feats, p_len)

        # protect：清音区（f0<1）混回原特征，防音色过拟合（官方公式）
        if self.protect < 0.5:
            feats0 = feats.copy()
            pf = pitchf.copy()                                 # (1,P)
            pf[pitchf > 0] = 1.0
            pf[pitchf < 1] = self.protect
            pf = pf[:, np.newaxis, :]                          # (1,1,P) 广播到 (1,768,P)
            feats = feats * pf + feats0 * (1.0 - pf)

        # 定长拟合：生成器 ONNX 是 gen_T 帧定长导出（官方 rel→abs 注意力的 pad 量
        # 在 trace 时烤成常量，无法动态；见 tools/export_gen_onnx2.py 注释）。
        # 不足补零、超出截断；块长由 convert_file 保证 ≤ gen_T。
        G = self.gen_T
        P = feats.shape[2]
        P_real = min(P, G)                    # 真实帧数（填充前），裁剪区间按它算
        if P > G:
            feats = feats[:, :, :G]
            pitch, pitchf, p_len = pitch[:, :G], pitchf[:, :G], G
        elif P < G:
            pad_n = G - P
            feats = np.pad(feats, ((0, 0), (0, 0), (0, pad_n)))
            pitch = np.pad(pitch, ((0, 0), (0, pad_n)), constant_values=1)
            pitchf = np.pad(pitchf, ((0, 0), (0, pad_n)))
            p_len = G

        feeds = {
            "phone": np.transpose(feats, (0, 2, 1)).astype(np.float32),   # (1,T,768)
            "phone_lengths": np.array([p_len], np.int64),
            "pitch": pitch.astype(np.int64),
            "pitchf": pitchf.astype(np.float32),
            "ds": np.array([sid], np.int64),   # 模型输入名是 ds（说话人ID），不是 g
            "rnd": np.random.randn(1, 192, p_len).astype(np.float32),
        }
        feeds = {n: feeds[n] for n in self._gen_names if n in feeds}
        out = self._gen_run(self.gen, self._gen_names, feeds)
        out = np.asarray(out).squeeze()
        # 官方：每块首尾各裁 3s 上下文（真声帧数对应 [300帧, P_real-300帧) 区间）
        valid = P_real * self.tgt_sr // 100   # 帧数 → 48k 样本数（100 帧/秒）
        lo, hi = self.t_pad_tgt, valid - self.t_pad_tgt
        if hi <= lo:                                       # 极短块：只裁头部
            return out[lo:]
        return out[lo:hi]

    # ---- 去直流 ----
    @staticmethod
    def _dc_block(x, sr=0, cutoff=25.0):
        """去掉直流偏置。

        实测输出残留约 +0.001 的 DC。24-bit 下这点偏移本身不致命，
        但它会让整段波形整体上移，某些播放器/功放会因此提前触顶削波，
        听起来"发闷"。减均值是零风险的（不改任何频率成分）。

        没有做更狠的高通：男声基频能低到 80Hz，任何高于这个的切点都会
        吃掉胸腔共鸣让声音变薄，收益不值这个代价。
        """
        import numpy as np
        x = np.asarray(x, np.float32)
        if x.size == 0:
            return x
        m = float(np.mean(x))
        if abs(m) < 1e-6:
            return x
        return (x - m).astype(np.float32)

    # ---- RMS 包络混合（官方公式）----
    def _change_rms(self, ref16k, out, rate):
        hop1, hop2 = SR16K // 2, self.tgt_sr // 2

        def rms(x, hop):
            n = len(x) // (hop * 2)
            if n == 0:
                return np.array([np.sqrt((x ** 2).mean() + 1e-12)])
            fr = x[:n * hop * 2].reshape(n, hop * 2)
            return np.sqrt((fr ** 2).mean(1))

        r1 = rms(ref16k, hop1)
        r2 = rms(out, hop2)
        x1 = np.linspace(0, len(out), len(r1))
        x2 = np.linspace(0, len(out), len(r2))
        r1 = np.interp(np.arange(len(out)), x1, r1)
        r2 = np.maximum(np.interp(np.arange(len(out)), x2, r2), 1e-6)
        gain = np.power(r1, 1 - rate) * np.power(r2, rate - 1)
        return out * gain

    # ---- 主入口 ----
    def convert_file(self, in_path, out_path, f0_up_key=0, sid=0, on_progress=None):
        t0 = time.time()

        def _p(frac, msg):
            # 给图形界面用的进度回调；命令行不传就是 None，行为完全不变
            if on_progress is not None:
                try:
                    on_progress(float(frac), str(msg))
                except Exception:
                    pass

        _p(0.01, "读取音频…")
        # 统一走 audio_io：坏文件在这里就翻成中文可操作提示，
        # 不再把 libsndfile 的 "Unspecified internal error" 抛给用户
        from .audio_io import read_audio
        audio, sr = read_audio(in_path)
        if sr != SR16K:
            audio = soxr.resample(audio, sr, SR16K, quality="VHQ")
        _p(0.05, "重采样到 16k（%.1f 秒音频）" % (len(audio) / SR16K))

        orig = audio
        audio = self._preprocess(audio)                        # 归一化 + 高通（不补边）
        # 切点分析用的补边缓冲（官方 pipeline.py:319）
        audio_an = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = self._opt_cut_points(audio, audio_an)
        # 工作缓冲：整段前后各补 t_pad（官方 pipeline.py:341）——首尾块的真实上下文靠它，
        # 每块再裁掉 t_pad，长度才对得上；缺这一步会把首尾真声裁掉。
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        pitchf, pitch = self._extract_f0(audio_pad, f0_up_key)
        _p(0.12, "音高提取完成，开始分块推理…")

        # 切点过滤：段太短或离结尾太近的切点直接丢弃（否则尾部会剩几十毫秒的碎片，
        # ContentVec 的 Conv 栈在超短输入上会直接报 Invalid input shape）
        min_seg = int(1.5 * SR16K)
        max_step = (GEN_T - self.t_pad2 // self.window - 2) * self.window   # 非末块最大前进步长
        cuts, prev = [], 0
        for t in opt_ts:
            ta = (t // self.window) * self.window
            if ta - prev < min_seg or len(audio) - ta < min_seg:
                continue
            # 安全网：段太长就强制补切点（块超过 GEN_T 会被截断 → 输出变短）
            while ta - prev > max_step:
                prev += (max_step // self.window) * self.window
                cuts.append(prev)
            cuts.append(ta)
            prev = ta
        # 末段同样受长度约束
        while len(audio) - prev > max_step:
            prev += (max_step // self.window) * self.window
            cuts.append(prev)

        results = []
        s = 0
        total_chunks = len(cuts) + 1
        for ci, ta in enumerate(cuts):
            sw, ew = s // self.window, (ta + self.t_pad2) // self.window
            _p(0.12 + 0.83 * ci / total_chunks,
               "推理中：第 %d / %d 块" % (ci + 1, total_chunks))
            results.append(self._run_chunk(audio_pad[s: ta + self.t_pad2 + self.window],
                                           pitch[:, sw:ew], pitchf[:, sw:ew], sid))
            s = ta
        if s < len(audio_pad):
            sw = s // self.window
            _p(0.12 + 0.83 * len(cuts) / total_chunks,
               "推理中：第 %d / %d 块（末块）" % (total_chunks, total_chunks))
            results.append(self._run_chunk(audio_pad[s:], pitch[:, sw:], pitchf[:, sw:], sid))

        _p(0.96, "拼接与音量对齐…")

        out = np.concatenate(results).astype(np.float32)
        out = self._change_rms(orig, out, self.rms_mix_rate)

        # 去 DC + 次声高通：实测输出残留约 +0.001 的直流偏置，
        # 播放时会白白吃掉动态范围，听起来"发闷"，某些设备还会有轻微电流感。
        out = self._dc_block(out, self.tgt_sr)

        peak = np.abs(out).max() / 0.99
        if peak > 1:
            out = out / peak
        # 每块已在 _run_chunk 内裁掉 3s 上下文（对齐官方 pipeline.py:388），不再全局裁

        # 输出 24-bit PCM，不用 32-bit 浮点。
        # 浮点 WAV（WAVE_FORMAT_IEEE_FLOAT）在不少播放器和剪辑软件里兼容性差，
        # 会被当成整数采样解析 → 直接变成刺耳噪音，听感就是"带电/电流声"。
        # 24-bit 已经是无损级别（144dB 动态范围），兼容性远好于浮点。
        sf.write(out_path, out, self.tgt_sr, subtype="PCM_24")
        dur = len(out) / self.tgt_sr
        elapsed = time.time() - t0
        _p(1.0, "完成：%.1f 秒音频，耗时 %.1f 秒（RTF %.2f）" % (dur, elapsed, elapsed / max(dur, 1e-6)))
        if self.verbose:
            print("[完成] %s | 音频 %.1fs | 耗时 %.1fs | RTF %.2f" %
                  (out_path, dur, elapsed, elapsed / dur))
        return elapsed, dur


def main():
    import argparse
    ap = argparse.ArgumentParser(description="RVC 纯 CPU/核显推理（深度定制版）")
    ap.add_argument("--input", required=True, help="输入音频（wav/mp3/flac）")
    ap.add_argument("--output", required=True, help="输出 wav 路径")
    ap.add_argument("--model", required=True, help="生成器 ONNX 路径")
    ap.add_argument("--vec", required=True, help="ContentVec ONNX 路径")
    ap.add_argument("--rmvpe", required=True, help="RMVPE ONNX 路径")
    ap.add_argument("--key", type=int, default=0, help="移调半音数（男转女常用 -12）")
    ap.add_argument("--sid", type=int, default=0, help="多说话人 ID")
    ap.add_argument("--index", default="", help="特征索引 .index（可选）")
    ap.add_argument("--index-rate", type=float, default=0.0)
    ap.add_argument("--rms-mix", type=float, default=0.25)
    ap.add_argument("--protect", type=float, default=0.33)
    args = ap.parse_args()

    pipe = RvcPipeline(args.model, args.vec, args.rmvpe,
                       args.index, args.index_rate, args.rms_mix, args.protect)
    pipe.convert_file(args.input, args.output, args.key, args.sid)


if __name__ == "__main__":
    main()

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
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # 避开 Where 等算子的图优化坑
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


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
                 rms_mix_rate=0.25, protect=0.33, verbose=True):
        self.verbose = verbose
        self.profile = pick_profile()
        p = self.profile
        if verbose:
            print("[硬件] %s | 逻辑核 %d(估物理 %d) | profile=%s" %
                  (p["cpu"], p["logical"], p["physical"], p["profile"]))
            if p["openvino"]:
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
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect
        self.index = None
        self.index_rate = 0.0

        # --- 生成器 ---
        if p["engine"] == "openvino":
            dev = p.get("generator_device", "GPU")
            if verbose:
                print("[加载] 生成器 → OpenVINO %s : %s" % (dev, os.path.basename(gen_path)))
            self.gen = _ov_compile(gen_path, dev, p["intra_threads"])
            self._gen_names = [i.get_any_name() for i in self.gen.inputs]
            self._gen_run = self._run_ov
        else:
            if verbose:
                print("[加载] 生成器 → ORT CPU (intra=%d) : %s" %
                      (p["intra_threads"], os.path.basename(gen_path)))
            self.gen = _cpu_session(gen_path, p["intra_threads"])
            self._gen_names = [i.name for i in self.gen.get_inputs()]
            self._gen_run = self._run_ort

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
                print("[加载] ContentVec → ORT CPU")
            self.vec = _cpu_session(vec_path, p["intra_threads"])
            self._vec_names = [i.name for i in self.vec.get_inputs()]
            self._vec_run = self._run_ort

        # --- RMVPE（任何 profile 固定 CPU：OpenVINO GPU 上 f0 会出错的已知坑）---
        if verbose:
            print("[加载] RMVPE → ORT CPU（f0 永不跑核显）")
        self.rmvpe = RMVPE(rmvpe_path, providers=["CPUExecutionProvider"])

        # --- 特征索引（可选）---
        if index_path and os.path.isfile(index_path) and index_rate > 0:
            try:
                import faiss
                self.index = faiss.read_index(index_path)
                self.index_rate = index_rate
                if verbose:
                    print("[加载] 特征索引 %s" % index_path)
            except Exception as e:
                print("[警告] index 加载失败，跳过检索: %s" % e)

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
    def convert_file(self, in_path, out_path, f0_up_key=0, sid=0):
        t0 = time.time()
        audio, sr = sf.read(in_path, dtype="float32", always_2d=True)
        audio = audio.mean(1)
        if sr != SR16K:
            audio = soxr.resample(audio, sr, SR16K, quality="VHQ")

        orig = audio
        audio = self._preprocess(audio)                        # 归一化 + 高通（不补边）
        # 切点分析用的补边缓冲（官方 pipeline.py:319）
        audio_an = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = self._opt_cut_points(audio, audio_an)
        # 工作缓冲：整段前后各补 t_pad（官方 pipeline.py:341）——首尾块的真实上下文靠它，
        # 每块再裁掉 t_pad，长度才对得上；缺这一步会把首尾真声裁掉。
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        pitchf, pitch = self._extract_f0(audio_pad, f0_up_key)

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
        for ta in cuts:
            sw, ew = s // self.window, (ta + self.t_pad2) // self.window
            results.append(self._run_chunk(audio_pad[s: ta + self.t_pad2 + self.window],
                                           pitch[:, sw:ew], pitchf[:, sw:ew], sid))
            s = ta
        if s < len(audio_pad):
            sw = s // self.window
            results.append(self._run_chunk(audio_pad[s:], pitch[:, sw:], pitchf[:, sw:], sid))

        out = np.concatenate(results).astype(np.float32)
        out = self._change_rms(orig, out, self.rms_mix_rate)

        peak = np.abs(out).max() / 0.99
        if peak > 1:
            out = out / peak
        # 每块已在 _run_chunk 内裁掉 3s 上下文（对齐官方 pipeline.py:388），不再全局裁

        sf.write(out_path, out, self.tgt_sr, subtype="FLOAT")
        dur = len(out) / self.tgt_sr
        elapsed = time.time() - t0
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

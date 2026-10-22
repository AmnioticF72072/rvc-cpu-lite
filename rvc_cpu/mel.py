# -*- coding: utf-8 -*-
"""Mel 频谱（纯 numpy，数值对齐 librosa slaney 规范）

- slaney 频率尺度 + slaney 面积归一化（与 librosa.filters.mel 等价）
- STFT 纯 numpy（center=True 反射填充、hann 窗，与 librosa.stft 一致）
- 零依赖（numpy 之外），便携包体积最小
"""
import numpy as np


def _stft_magnitude(audio, n_fft, hop, win):
    """librosa.stft(center=True) 的幅值谱：(1+n_fft/2, T)。"""
    pad = n_fft // 2
    ap = np.pad(np.asarray(audio, dtype=np.float64), (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(ap, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * win, n=n_fft, axis=1)).T


def hz_to_mel_slaney(freq):
    f = np.asarray(freq, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (f - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if f.ndim == 0:
        if f >= min_log_hz:
            return min_log_mel + np.log(f / min_log_hz) / logstep
        return mels
    log_t = f >= min_log_hz
    mels = np.array(mels, dtype=np.float64)
    mels[log_t] = min_log_mel + np.log(f[log_t] / min_log_hz) / logstep
    return mels


def mel_to_hz_slaney(mels):
    m = np.asarray(mels, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * m
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if m.ndim == 0:
        if m >= min_log_mel:
            return min_log_hz * np.exp(logstep * (m - min_log_mel))
        return freqs
    log_t = m >= min_log_mel
    freqs = np.array(freqs, dtype=np.float64)
    freqs[log_t] = min_log_hz * np.exp(logstep * (m[log_t] - min_log_mel))
    return freqs


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None, dtype=np.float32):
    """等价 librosa.filters.mel(..., htk=False)（slaney 规范）。"""
    fmax = fmax if fmax is not None else sr / 2.0
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
    fftfreqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = mel_to_hz_slaney(np.linspace(hz_to_mel_slaney(fmin),
                                         hz_to_mel_slaney(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    # slaney 面积归一化
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


class MelSpectrogram:
    """log-Mel 频谱（接口对齐 dev6699/rvc-onnx 的 mel.py，供 RMVPE 使用）。"""

    def __init__(self, n_mel_channels=128, sampling_rate=16000, win_length=1024,
                 hop_length=160, n_fft=None, mel_fmin=0.0, mel_fmax=None, clamp=1e-5):
        n_fft = win_length if n_fft is None else n_fft
        self.mel_basis = mel_filterbank(sampling_rate, n_fft, n_mel_channels,
                                        mel_fmin, mel_fmax)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1.0, center=True):
        factor = 2 ** (keyshift / 12.0)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        win = np.hanning(win_length_new + 1)[:win_length_new]  # periodic hann（等价 sym=False）
        magnitude = _stft_magnitude(audio, n_fft_new, hop_length_new, win)
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            if magnitude.shape[0] < size:
                magnitude = np.pad(magnitude, ((0, size - magnitude.shape[0]), (0, 0)), mode="constant")
            magnitude = magnitude[:size, :] * (self.win_length / win_length_new)
        mel_out = np.dot(self.mel_basis, magnitude)
        return np.log(np.maximum(mel_out, self.clamp))

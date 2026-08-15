"""Reference NumPy re-implementation of NeMo AudioToMelSpectrogramPreprocessor
for nvidia/stt_en_citrinet_256_gamma_0_25.

Authoritative source of every constant:
  - nemo_x/model_config.yaml  (extracted from the NGC .nemo checkpoint)
  - NeMo main: nemo/collections/asr/parts/preprocessing/features.py
                 class FilterbankFeatures + normalize_batch()

Verified: feeding the output of nemo_mel()+normalize_per_feature() into the
exported ONNX reproduces the reference transcription of the HF librispeech
sample1.flac clip.

This is the spec the M55 C frontend must reproduce.
"""
import numpy as np
import librosa
import scipy.signal

SR = 16000
N_FFT = 512          # model_config.yaml: preprocessor.n_fft
WIN = 400            # window_size 0.025 s * 16000
HOP = 160            # window_stride 0.01 s * 16000
N_MELS = 80          # features: 80
PREEMPH = 0.97       # FilterbankFeatures default (config does not override)
LOG_GUARD = 2.0 ** -24   # log_zero_guard_type "add", log_zero_guard_value 2**-24
MAG_POWER = 2.0      # power spectrum
FMIN, FMAX = 0, SR / 2
MEL_NORM = "slaney"  # librosa.filters.mel(norm="slaney"), htk=False
STD_EPS = 1e-5       # CONSTANT added to per-feature std
# dither = 1e-5 but FilterbankFeatures only applies it when self.training -> OFF at inference
# pad_to = 16 is a pure efficiency pad; drop it and use T = number of valid frames


def nemo_mel(x: np.ndarray) -> np.ndarray:
    """float32 mono 16 kHz waveform -> (80, T) log-mel, T = 1 + len(x)//HOP (center=True)."""
    x = np.concatenate([x[:1], x[1:] - PREEMPH * x[:-1]])          # pre-emphasis, x[0] kept
    w = scipy.signal.get_window("hann", WIN, fftbins=False)         # torch.hann_window(400, periodic=False)
    S = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                     window=w, center=True, pad_mode="constant")    # torch.stft(center=True, pad_mode="constant")
    P = (np.abs(S) ** MAG_POWER).astype(np.float32)
    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                             fmin=FMIN, fmax=FMAX, norm=MEL_NORM).astype(np.float32)
    return np.log(fb @ P + LOG_GUARD)


def normalize_per_feature(m: np.ndarray, seq_len: int) -> np.ndarray:
    """normalize_batch(..., 'per_feature'): per-mel-bin mean/std over the valid frames.
    NOTE: std uses ddof=1 (NeMo divides by N-1), then += 1e-5."""
    v = m[:, :seq_len]
    mu = v.mean(axis=1, keepdims=True)
    sd = v.std(axis=1, ddof=1, keepdims=True) + STD_EPS
    out = (m - mu) / sd
    out[:, seq_len:] = 0.0
    return out


def features_for_fixed_window(wav16k: np.ndarray, T: int) -> np.ndarray:
    """Deployment contract: exactly T valid frames, no NeMo pad_to=16 padding,
    graph `length` baked to T. wav16k must hold >= T*HOP samples."""
    m = nemo_mel(wav16k)[:, :T]
    return normalize_per_feature(m, T)[None].astype(np.float32)


# greedy CTC decode needs only vocab.txt (1024 SentencePiece pieces + <blk> at 1024).
# No SentencePiece runtime required on the MCU: concatenate pieces, map U+2581 -> ' '.
def greedy_ctc(logits_TxV: np.ndarray, vocab: list[str], blank: int = 1024) -> str:
    out, prev = [], -1
    for i in logits_TxV.argmax(-1):
        if i != prev and i != blank:
            out.append(vocab[i])
        prev = i
    return "".join(out).replace("▁", " ").strip()

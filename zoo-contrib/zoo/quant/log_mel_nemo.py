"""NeMo's log-mel front end, as a `zoo.quant.calib` preprocessor.

Paste this class into `zoo/quant/calib.py` beside `WhisperLogMel` — it uses
only `register_preprocessor`, `CalibrationSpec` and numpy/librosa, all of
which that module already has. It is a separate file here only so the
contribution can be reviewed on its own.

Every constant is load-bearing and none is a default. The authority is
`model/fe.py` in the stm32n6-stt repo, which is simultaneously the NumPy
reference the C front end is written against and the oracle its unit tests
compare to; this class was checked against it and reproduces it exactly
(max |difference| = 0.0 over an 80x800 feature block — see
`zoo-contrib/README.md`).

Two of these constants are not interchangeable with a neighbouring value:

* **`ln(mel + 2**-24)`.** The log floor is the one front-end choice that
  moves word error rate. Measured on LibriSpeech dev-clean through this exact
  graph: swapping the guard for `1e-2` takes WER from 5.83 % to 30.80 %,
  while dropping pre-emphasis, using a periodic window, HTK mel spacing,
  magnitude instead of power, log10 instead of ln, or a single global
  mean/std each move it by less than a point
  (`docs/FEASIBILITY.md` section 2(c), `eval/results/fe.log`).
* **per-feature normalisation over time, `ddof=1`.** The model was trained on
  features standardised per mel bin across the utterance. Feeding it
  un-normalised log-mel calibrates every activation range from a
  distribution the board will never produce.

`center=True` with `pad_mode="constant"` means `T` frames need exactly
`(T - 1) * hop + 1` samples, which is what `samples_needed` returns less any
declared lead-in.
"""

from __future__ import annotations

import numpy as np

# In calib.py these two come from the module itself; they are imported here
# only so this file is readable and testable standalone.
from zoo.quant.calib import CalibrationSpec, register_preprocessor


@register_preprocessor("log_mel_nemo")
class NeMoLogMel:
    """NVIDIA NeMo's `AudioToMelSpectrogramPreprocessor`, as Citrinet uses it.

    `options` this preprocessor reads, all optional:

        lead_silence_samples   zeros prepended before the utterance, so a clip
                               starts where the evaluation harness starts it
                               (4800 = 300 ms for stm32n6-stt). The window is
                               still `(frames - 1) * hop + 1` samples total,
                               so this trades real audio for leading silence
                               rather than lengthening the buffer.
        preemphasis            default 0.97; `0` disables it.
        log_guard              default 2**-24. Do not raise it.
    """

    sample_rate = 16_000
    n_fft = 512
    win_length = 400
    hop = 160
    fmin = 0.0
    fmax = 8000.0
    std_eps = 1e-5

    def _frames(self, shape: list[int]) -> tuple[int, int]:
        if len(shape) < 2:
            raise ValueError(f"log_mel_nemo needs a (…, mels, frames) input; got {shape}")
        return int(shape[-2]), int(shape[-1])

    def samples_needed(self, shape: list[int], spec: CalibrationSpec) -> int:
        _, frames = self._frames(shape)
        lead = int(spec.opt("lead_silence_samples", 0))
        window = (frames - 1) * self.hop + 1
        if lead >= window:
            raise ValueError(
                f"lead_silence_samples={lead} leaves no room in a {window}-sample window"
            )
        return window - lead

    def __call__(self, audio: np.ndarray, shape: list[int], spec: CalibrationSpec) -> np.ndarray:
        import librosa
        import scipy.signal

        mels, frames = self._frames(shape)
        lead = int(spec.opt("lead_silence_samples", 0))
        preemph = float(spec.opt("preemphasis", 0.97))
        guard = float(spec.opt("log_guard", 2.0**-24))

        window_samples = (frames - 1) * self.hop + 1
        buffer = np.zeros(window_samples, dtype=np.float32)
        take = min(len(audio), window_samples - lead)
        buffer[lead : lead + take] = np.asarray(audio, dtype=np.float32)[:take]

        # Pre-emphasis, NeMo's form: sample 0 passes through untouched.
        if preemph:
            buffer = np.concatenate([buffer[:1], buffer[1:] - preemph * buffer[:-1]])

        # Symmetric Hann (`fftbins=False`), not the periodic default.
        taper = scipy.signal.get_window("hann", self.win_length, fftbins=False)
        stft = librosa.stft(
            buffer,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=taper,
            center=True,
            pad_mode="constant",
        )
        power = (np.abs(stft) ** 2.0).astype(np.float32)

        filters = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=mels,
            fmin=self.fmin,
            fmax=self.fmax,
            norm="slaney",
        ).astype(np.float32)
        feature = np.log(filters @ power + guard)

        # Per-feature (per mel bin) standardisation over time. ddof=1 matches
        # NeMo; ddof=0 is a different number on an 800-frame window.
        mean = feature.mean(axis=1, keepdims=True)
        std = feature.std(axis=1, ddof=1, keepdims=True) + self.std_eps
        feature = (feature - mean) / std

        return feature[:, :frames].reshape(shape).astype(np.float32)

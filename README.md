# stm32n6-tts — an on-device speech captioner for the STM32N6570-DK

Push-to-talk English speech recognition running entirely on an STM32N6570-DK:
microphone → log-mel on the Cortex-M55 → **Citrinet-256 CTC encoder on the
Neural-ART NPU** → greedy CTC decode → text on the 800×480 LCD.

> **Naming.** The repository is called `stm32n6-tts`, but this is
> speech-**to**-text (ASR/STT), not text-to-speech. Nothing here synthesises
> audio. Renaming the repo to `stm32n6-stt` or `stm32n6-asr` would remove a
> standing source of confusion.

## Status

**Feasibility: settled, GO.** The model has been exported, shape-frozen,
quantised to int8 on real speech, and compiled against the STM32N6 audio
application's real memory geometry. Nothing has run on the board yet.

The result that decides the project:

| window | T | epochs | **SW epochs** | activations | % of audio pool | weights | sched. cycles @1 GHz |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 s | 400 | 626 | **0** | 306 KB | 20.8 % | 9.677 MB | 73.7 ms |
| **8 s** | **800** | **628** | **0** | **625 KB** | **42.5 %** | **9.728 MB** | **91.2 ms** |
| 12 s | 1200 | 628 | **0** | 1,017 KB | 69.1 % | 9.731 MB | 124.1 ms |

**Zero software epochs.** Every operator in a full ASR encoder — 503 nodes of
exactly seven types: 282 Conv (107 of them grouped/depthwise), 130 Relu,
23 ReduceMean, 23 Sigmoid, 23 Mul, 21 Add, 1 Transpose — maps to Neural-ART
hardware, at rank 3 (`[1, 80, T]` in, `[1, T/8, 1025]` out). For contrast, the
Whisper-tiny encoder measured on this same board at **10,935 ms** with 184 of
its 391 epochs in software.

Evidence: `compile/reports/*/summary.txt`, ST Edge AI Core v4.0.1-20581.
See [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) for the full assessment,
including what the original plan got wrong.

## The model

[`OpenVoiceOS/stt_en_citrinet_256_gamma_0_25_onnx`](https://huggingface.co/OpenVoiceOS/stt_en_citrinet_256_gamma_0_25_onnx)
— a pre-exported ONNX of NVIDIA NeMo's
[`stt_en_citrinet_256_gamma_0_25`](https://catalog.ngc.nvidia.com/orgs/nvidia/models/stt_en_citrinet_256_gamma_0_25),
CC-BY-4.0. **NeMo and PyTorch are not required.**

- ~10 M parameters, 9.7 MB int8, streams from octoFlash
- 80-bin log-mel in, 1025-class CTC logits out (1024 SentencePiece unigram + blank)
- subsampling factor 8 → one output frame per 80 ms
- `kernel_size_factor: 0.25` (that is what `gamma_0_25` means — scaled temporal
  kernels, verified at `docs/nemo_model_config.yaml:30`)

## Deployment contract

Read off the generated `stai_network.h` (`compile/reports/g800_real/io_contract.h`):

| | format | shape | bytes | scale | offset |
|---|---|---|---:|---|---:|
| input | `STAI_FORMAT_S8`, `CHANNEL_FIRST` | `{80, 800, 1}` | 64,000 | 0.120522417128086 | 0 |
| output | `STAI_FORMAT_S8`, `CHANNEL_FIRST` | `{100, 1025, 1}` | 102,500 | 0.265415638685226 | 0 |

Both scales are **per-tensor**, so greedy CTC runs directly on the int8 logits —
argmax needs no dequantisation. Vocabulary is the fast axis. Read the scale at
runtime from `stai_network_get_inputs()[0]` rather than hardcoding it.

## Layout

```
model/      graph surgery, quantisation, and the NumPy reference frontend
            (fe_reference.py is the C implementation's spec AND its test oracle)
eval/       WER harness — window, int8, SNR, reverberation, gain, frontend ablation
eval/results/  measured outputs of the above
compile/    the audio-pool mpool + profile, and per-window compile evidence
tokenizer/  1025-piece vocabulary and the SentencePiece model
docs/       the feasibility assessment and upstream provenance
artifacts/  (gitignored) rescued ONNX graphs, weights, compile reports
```

## Accuracy, measured on the host

LibriSpeech dev-clean, ONNX Runtime, through the verified NeMo-exact frontend.
These are **not** board measurements.

- int8 vs fp32 at 4 s: 5.60 % → **6.09 %** WER (+0.49 points) — `eval/results/int8.json`
- window vs full spoken reference, 150 natural-length utterances:
  4 s **47.7 %**, 6 s 30.4 %, 8 s **20.0 %**, 12 s 9.0 % — truncation, not
  misrecognition. Word coverage 0.56 / 0.73 / 0.84 / 0.95.
- babble noise is the one that hurts: 60.1 % WER at 5 dB SNR — `eval/results/snr.json`

**The gain-staging landmine.** At −54 dBFS input — which is where ordinary
desk speech lands on the DK's IMP34DT05 microphone — 97.9 % of mel bins fall
below NeMo's `log_zero_guard` and WER goes **5.83 % → 35.28 %** while every log
reports a clean NPU run. Peak-normalising the captured buffer to 0.9 restores
5.83 %. Applying that gain *after* int16 truncation only recovers to 10.45 %,
so it has to happen in the PDM/MDF decimator. See `eval/results/gain.log`.

## Next

Gates 0–2 are host-side and cost hours; the first board contact is Gate 4.
Full sequence in [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md#work-plan).

## Related

[`stm32n6-deployment-zoo`](https://github.com/LarocheC/stm32n6-deployment-zoo) —
the screening funnel this model was validated against. The zoo answers "will an
arbitrary graph run on this part"; this repo answers "does this product work for
a person."

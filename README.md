# stm32n6-stt — an on-device speech captioner for the STM32N6570-DK

Push-to-talk English speech recognition running entirely on an STM32N6570-DK:
microphone → log-mel on the Cortex-M55 → **Citrinet-256 CTC encoder on the
Neural-ART NPU** → greedy CTC decode → text on the 800×480 LCD.

## Status

**Feasibility settled, GO. Gates 0–3 are closed.** The model has been exported,
shape-frozen, quantised to int8 on real speech, compiled against the STM32N6
audio application's real memory geometry, and scored for accuracy at the window
it will actually ship at. **The board now runs firmware built from source on
this machine** — the full path from compile through signing, flashing and boot
is proven, with ST's own model in the loop. The Citrinet network has not been
invoked on silicon yet; that is Gate 4.

| gate | verdict | the number that decides it |
|---|---|---|
| 1 — int8 vs fp32 WER at 8 s | **PASS** | int8 costs **+0.50 points** (4.91 % → 5.41 %, n=373, 95 % CI [+0.07, +0.94]) against a ~1.0-point pass band |
| 2 — recompile on ST's own mpool + option string | **PASS** | **0 SW / 0 hybrid epochs**, 947,200 B activations, **0 B in hyperRAM**, weights at **0x70180000** |
| 3 — build, sign, flash and boot ST's stock app | **PASS** | our own build runs from external flash: `\| 22 \| 2.07% \| 0.88 \| 1.20 \| 0.00 \|` |

Gates 1 and 2 were re-run from scratch by an adversarial verifier and reproduced
exactly — Gate 1 with an independent harness (zero per-utterance disagreements
over 1,200 model-utterance pairs), Gate 2 bit-for-bit from a clean compile. Full
report: [`docs/GATES-1-2.md`](docs/GATES-1-2.md). Gate 3 write-up, including the
working build recipe and the access matrix for this board's two boot modes:
[`board/GATE3.md`](board/GATE3.md).

**The one irreversible step turned out to be already spent — checked, not
assumed.** `fuse_vddio()` is *not* compiled out on the Makefile path
(`stm32n6570_discovery.h:59-61` self-defines `USE_STM32N6570_DK`), so running
even the unmodified ST app permanently programs OTP word 124 bits 15/16 on a
fresh board. A read-only dump of *this* board returns
**word 124 = `0x00018000`** — both bits already set, so the program branch never
runs and Gate 3 carried no irreversible action here. Evidence and the caveats
that remain: [`board/OTP.md`](board/OTP.md).

**Signing needs `-align`, and ST's Makefile omits it.** Without it the signed
header's entry point lands in the middle of `.text`, the FSBL jumps into a
function body, and the part dies before UART init — a perfectly silent board
with correct-looking flash. It cost most of Gate 3 to find. The build recipe and
a two-command pre-flash check are in [`board/GATE3.md`](board/GATE3.md).

The compile result that decides the project:

| window | T | epochs | **SW epochs** | activations | % of audio pool | weights | sched. cycles @1 GHz |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 s | 400 | 626 | **0** | 306 KB | 20.8 % | 9.677 MB | 73.7 ms |
| **8 s** | **800** | **628** | **0** | **625 KB** | **42.5 %** | **9.728 MB** | **91.2 ms** |
| 12 s | 1200 | 628 | **0** | 1,017 KB | 69.1 % | 9.731 MB | 124.1 ms |

> **These are screening numbers, and Gate 2 moved them.** The table above uses the
> zoo's option string, which includes `--Oauto-sched`. **ST's audio application
> ships without it.** Under ST's own options the 8 s row is **618 epochs, 925 KB,
> 62.8 % of the pool, 91.89 ms** — still 0 SW, still 0 hybrid, still entirely
> on-chip — and the 12 s row **spills 150 KB to PSRAM** while the epoch table
> still reads 0 SW / 0 hybrid. Adding `--Oauto-sched` back reproduces this table
> exactly. Note also that the pool is not fungible: npuRAM6 is at **94.87 %**
> (~23 KB spare) while cpuRAM2 sits at 48.83 %. See [`compile/GATE2.md`](compile/GATE2.md).

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

- int8 vs fp32 at **8 s**, the shipped window: 4.91 % → **5.41 %** WER
  (+0.50 points, 95 % CI [+0.07, +0.94]) on 373 utterance-disjoint utterances
  that fit the window — `eval/results/gate1_8s.json`, [`eval/GATE1.md`](eval/GATE1.md)
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

Gates 0–3 are closed. The next step is **Gate 4** — drop in the Citrinet network,
feed it a host-computed feature vector, and compare on-device argmax token IDs
against host ONNX Runtime. Remaining gates are planned to file level in
[`firmware/WORKLIST.md`](firmware/WORKLIST.md) (6.5 developer-days; Gate 6's
tokenizer is already built and byte-verified at 8,222 B). Gate definitions in
[`docs/FEASIBILITY.md`](docs/FEASIBILITY.md#work-plan); what has changed since it
was written is in [`docs/GATES-1-2.md`](docs/GATES-1-2.md) §2.

## Related

[`stm32n6-deployment-zoo`](https://github.com/LarocheC/stm32n6-deployment-zoo) —
the screening funnel this model was validated against. The zoo answers "will an
arbitrary graph run on this part"; this repo answers "does this product work for
a person."

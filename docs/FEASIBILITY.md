# Citrinet-256 on STM32N6 — feasibility assessment

**Verdict: GO, at an 8-second window rather than 4.**

This document records what was established before any firmware was written, what
the original plan got wrong, and what remains genuinely unknown. Every number
here was produced on this machine with ST Edge AI Core v4.0.1-20581 unless
marked otherwise. **No number in the body of this document was measured on the
board** — it is a pre-firmware assessment and is kept as one.

> **Status, 2026-08-19: gates 0-4 and 6 are closed; gate 5 is not.** The §3 work
> plan below is the plan of record and its gate *definitions* still stand, but
> the verdicts have moved and three of its numbers have been superseded on
> silicon. The corrections are marked inline and collected in §3; the board
> record is [`../board/GATE4.md`](../board/GATE4.md), and what changed since this
> document was written is in [`GATES-1-2.md`](GATES-1-2.md) §2 and
> [`../firmware/WORKLIST.md`](../firmware/WORKLIST.md) §0.

---

## 1. The question that mattered, and its answer

The plan's load-bearing assumption was that Citrinet's depthwise/grouped 1-D
convolution stack would map to Neural-ART hardware. Neural-ART is fundamentally
a 2-D convolution engine; rank-3 tensors, `groups=256` depthwise convolutions,
and squeeze-excitation blocks are all places a 2-D engine plausibly falls back
to the Cortex-M55. Each fallback is a software epoch, and on this part latency
is epoch-bound.

It maps. Completely.

| window | T | epochs | pure SW | hybrid | activations | % of 1,507,328 B | weights (octoFlash) | MACs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 s | 400 | 626 | **0** | **0** | 306.25 kB (npuRAM6) | 20.8 % | 9.677 MB | 1.117 G |
| **8 s** | **800** | **628** | **0** | **0** | **200 kB cpuRAM2 + 425 kB npuRAM6** | **42.5 %** | **9.728 MB** | 2.234 G |
| 12 s | 1200 | 628 | **0** | **0** | 600 kB cpuRAM2 + 417.19 kB npuRAM6 | 69.1 % | 9.731 MB | 3.351 G |

Source: `compile/reports/{g400,g800_real,g1200}/summary.txt`, lines 3091–3117 of
the original `network_generate_report.txt` (full reports in `artifacts/compile/`).

The graph is 503 nodes of exactly seven operator types — Conv 282 (107 grouped),
Relu 130, ReduceMean 23, Sigmoid 23, Mul 23, Add 21, Transpose 1 — at rank 3
throughout. No control flow, no attention, no Softmax, no dynamic MatMul.

**Context for how good this is.** The `stm32n6-deployment-zoo` leaderboard
records the Whisper-tiny encoder measured on this same board at **10,935 ms**,
with 184 of its 391 epochs in software and 16.4 MB of activations spilled to
PSRAM. Citrinet at 8 s is 628 epochs, none in software, entirely on-chip.

### The scheduler's own latency estimate

| window | cycles | ideal cycles | efficiency | ms @ 1 GHz | ms per second of audio |
|---|---:|---:|---:|---:|---:|
| 4 s | 73,741,608 | — | 26.6 % | 73.7 | 18.4 |
| 8 s | 91,212,624 | 39,259,512 | 43.0 % | 91.2 | 11.4 |
| 12 s | 124,080,680 | 58,811,940 | 47.4 % | 124.1 | 10.3 |

Source: `compile/reports/ws*/cycles.json`.

Doubling the window from 4 s to 8 s costs **+17.5 ms (+23.7 %), not 2×**,
because the ~9.7 MB octoFlash weight sweep is a fixed cost paid once per
utterance. This is the structural argument for the utterance-based design: it
converts the 10 MB weight stream — the thing that would sink any streaming
formulation on this part — into an amortised constant.

---

## 2. What the draft plan got wrong

### (a) Four seconds is the worst decision in the plan

Measured on 150 natural-length dev-clean utterances (median 5.55 s), scoring the
window's output against the **full** spoken reference:

| window | WER | fraction of spoken words returned |
|---|---:|---:|
| 4 s | **47.7 %** | 0.56 |
| 6 s | 30.4 % | 0.73 |
| **8 s** | **20.0 %** | **0.84** |
| 12 s | 9.0 % | 0.95 |

This is truncation, not misrecognition — but a user cannot tell the difference,
and reads it as "it cannot understand me." Only 28.3 % of dev-clean utterances
are ≤ 4 s.

### (b) But 12 s is not simply better

On 30 short utterances (≤ 3 s, mean 2.45 s, 195 words), padding to a long window
costs accuracy, because per-feature normalisation statistics computed over a
mostly-silent window drift:

| window | WER on short utterances |
|---|---:|
| exact length | 6.15 % |
| 4 s | 5.64 % |
| **8 s** | **6.15 %** |
| 12 s | 8.21 % |

Small sample (12 vs 16 word errors) — directional, not significant. But 8 s is
the only window with no measurable padding penalty *and* 84 % word coverage, at
42.5 % pool occupancy. **8 s is the choice.** The 12 s graph is compiled and
kept if long-sentence coverage later matters more.

### (c) The mel frontend was the wrong thing to be afraid of

The plan flagged NeMo frontend parity as the top risk. It is not. Against a
5.83 % reference, ablating individual frontend choices barely moves WER:

| deviation from NeMo's spec | WER |
|---|---:|
| reference (NeMo spec) | 5.83 % |
| no pre-emphasis | 5.29 % |
| periodic instead of symmetric Hann | 5.83 % |
| HTK instead of Slaney mel scale | 6.24 % |
| mel norm = None | 5.56 % |
| magnitude instead of power spectrum | 5.97 % |
| log10 instead of ln | 5.83 % |
| `center=False` | 5.43 % |
| **single global mean/std instead of per-feature** | **5.83 %** |
| **log guard 1e-2 instead of 2⁻²⁴** | **30.80 %** |

Source: `eval/results/fe.log`. Per-bin mean subtraction in the log domain cancels
any per-bin multiplicative constant exactly, which is why so much of the spec
turns out not to matter. **Only the absolute log floor matters.** The firmware
gate should be the log-floor occupancy statistic, not a bit-exactness comparison.

### (d) The plan has no gain stage, and that is the silent-failure landmine

The DK's IMP34DT05 is −26 dBFS at 94 dBSPL; conversational speech at 30–50 cm
lands at −54 to −48 dBFS. At −54 dBFS, **97.9 %** of mel-filter outputs fall
below the log guard, the log saturates, per-feature normalisation amplifies the
residue, and:

| condition | WER |
|---|---:|
| 0 dB (LibriSpeech native level) | 5.83 % |
| −54 dBFS, raw | **35.28 %** |
| −54 dBFS, then RMS-normalise to −25 dBFS | 5.97 % |
| −54 dBFS, then peak-normalise to 0.9 | **5.83 %** |
| −54 dBFS **truncated to int16**, then RMS-normalise | 10.45 % |

Source: `eval/results/gain.log`. Every one of these runs reports a clean NPU
execution. **Critical detail:** the gain must be applied in the MDF/PDM decimator
or the capture must be wider than int16 — gaining up *after* int16 truncation at
that level only recovers to 10.45 %.

### (e) Touch is not available; the LCD is a separate graft

The GT911 touch controller driver ships in neither ST Getting-Started package
(`Drivers/BSP/Components` contains only Common, aps256xx, mx25um51245g /
mx66uw1g45g, rk050hr18). "Hold to speak on the touchscreen" is not buildable as
scoped — use `BUTTON_USER1` (PC13/EXTI13), which is already in the audio
package's BSP.

Separately, STM32N6-GettingStarted-Audio is **strictly headless** — no LCD
driver, no fonts, results over UART. The display stack lifts from the
ObjectDetection package (`rk050hr18/`, `Utilities/lcd/`, `Utilities/Fonts/`,
`UTIL_LCDEx_PrintfAt`, 47-character line width already derived by ST), plus a
PSRAM region added to the linker script for the 768,000-byte framebuffer.

### (f) The named fallback is dead

**QuartzNet-15x5 does not fit.** Its k=75 depthwise layer at C=512/T=201 needs
2,366,976 B of workspace for *one layer* — 157 % of the entire audio pool — and
fails to compile on-chip (`Oauto did not find valid compile options` → E103). It
has fifteen such layers plus one k=87 at 1,705,120 B, and 3,788 MMAC against
Citrinet's 1,071 at 4 s. Also struck: Conformer-CTC-small (48 dynamic×dynamic
MatMuls, against Whisper-tiny's 8), all RNN-CTC families (LSTM/GRU/RNN appear in
zero rows of ST's operator mapping table), Zipformer-CTC (no English model),
Silero STT (CC-NC-BY licence).

**The real fallback is the same checkpoint at T=400** — already compiled,
already validated, 73.7 ms. It is a memory/latency escape hatch, not an accuracy one.

### (g) Smaller corrections

- **NeMo and PyTorch are not needed.** A pre-exported ONNX of the exact
  checkpoint exists (`OpenVoiceOS/stt_en_citrinet_256_gamma_0_25_onnx`,
  CC-BY-4.0). This removes an entire dependency and risk class — NeMo does not
  support this machine's Python 3.13 anyway.
- **`gamma_0_25` is `kernel_size_factor: 0.25`** — a temporal-kernel scaling
  factor, verified at `docs/nemo_model_config.yaml:30` and repeated on all 22
  blocks. Not a quantisation variant.
- **Vocabulary is 1025** — 1024 SentencePiece *unigram* pieces (not WordPiece)
  plus blank at id 1024. Verified: `tokenizer/vocab.txt` is exactly 1025 lines.
- **SentencePiece the library is not needed.** The only non-ASCII character in
  the entire vocabulary is `▁` (U+2581, in 812 pieces). Detokenisation is
  concatenate-then-replace-`▁`-with-space. Ship a flat NUL-separated `kPieces[]`
  blob plus a `uint16_t kOffset[1026]` — roughly 8–12 KB of flash.
- **"Screen 2 s, 4 s and 8 s graphs" is done**, and 2 s is pointless. Skip it.
- **Greedy CTC needs no dequantisation.** Both input and output scales are
  per-tensor, so argmax over the raw int8 logits is exact.

---

## 3. Work plan

Ordered so that the cheapest question is asked first and the first thing that
can contradict desk research is reached in about one developer-day. Steps 0–2
need neither the board nor a new compile.

**Where the gates stand, 2026-08-19.** The definitions below are unchanged; the
verdicts are not.

| gate | verdict | evidence |
|---|---|---|
| 0 — freeze the artifact set | **closed** | this repository |
| 1 — WER at 8 s, fp32 vs int8 | **PASS**, +0.50 points | `eval/GATE1.md`, `eval/results/gate1_8s.json` |
| 2 — recompile on ST's mpool | **PASS**, 618/0/0 at `0x70180000` | `compile/GATE2.md` |
| 3 — build/sign/flash the stock app | **PASS** | `board/GATE3.md` |
| 4 — Citrinet on silicon, canned features | **PASS**, 448 epochs, **124.0 ms measured** | `board/GATE4.md` |
| 5 — log-mel front end on the M55 | **open** — host-exact, never executed on the M55 | `firmware/FRONTEND.md`, `docs/GATES-5-6.md` |
| 6 — greedy CTC + detokeniser | **PASS**, 0 text disagreements / 9,226 chars | `firmware/test/results/gate6_ctc.json` |
| 7 / 7b — LCD graft, button | not started | `firmware/WORKLIST.md` |

Remaining effort, from `firmware/WORKLIST.md`'s roll-up with gates 4 and 6
struck: **4.5 developer-days** (Gate 5 3 d, Gate 7 1.5 d, 7b folded in).

**Gate 0 — freeze the artifact set (1 h).** Already done by this commit: the
scripts, graphs, tokenizer and compile evidence are out of `/tmp` and under
version control. *Stop if* the NGC `.nemo` SentencePiece vocabulary does not
match `tokenizer/vocab.txt[0:1024]` byte-for-byte.

**Gate 1 — WER at the shipped window, fp32 and int8 (3 h, no board).** Retarget
`eval/run_8s.py` and `eval/run_int8.py` at `q800_real.onnx` over ≥300 held-out
dev-clean utterances. *Pass:* int8 within ~1.0 point of fp32 (current 4 s
evidence: 5.60 % → 6.09 %, frame-argmax agreement 0.9649). *Stop if* int8 costs
more than 2 points — re-quantise before touching firmware. This is the last
cheap accuracy gate.

> **First, verify the calibration/evaluation split is actually disjoint.**
> `model/quant_real.py` selects its calibration utterances with one RNG seed and
> the evaluation scripts use others; the script's own comment defers the overlap
> check ("check overlap later") and it was never done. Overlap is unlikely but
> unproven, and calibrating on evaluation audio is precisely what makes a
> quantisation look free when it is not. See `model/README.md`.
>
> **Correction — this paragraph named the wrong script.** `quant_real.py`
> calibrates the **4 s** graph. The shipped 8 s graph `q800_real.onnx` is
> calibrated by **`model/q800.py`**, with a different filter (`4.0 <= d <= 7.5`),
> a different slice (`[:48]`) and therefore a different 48 utterances; the
> canonical reconstruction is `eval/sets.py:cal_800()`. Gate 1 ran the check on
> all three sets and the answer did not change — `cal_800` overlaps nothing at
> all, while `cal_400` overlaps six evaluation sets and no published number is
> contaminated. But anyone reasoning about overlap from this paragraph would have
> excluded the wrong utterances. `eval/GATE1.md` §1; `board/GATE4.md` Round 20.

**Gate 2 — recompile against the *firmware's* mpool, not the zoo's (30 min).**
ST's audio application places octoFlash at **0x70180000**; `compile/audio_strict.mpool`
places it at 0x71000000, and `sign-and-flash-model.sh` objcopies the weight blob
to ST's offset. Compile against a copy of ST's own
`Projects/X-CUBE-AI/models/stm32n6.mpool` with ST's option string — note ST's
profile has **no** `--enable-virtual-mem-pools` and **no** `--Oauto-sched`,
unlike every zoo profile. *Pass:* 0 SW epochs, 0 hybrid, ≤1,507,328 B, and the
flash base in the report reads 0x70180000. *Stop if* SW epochs appear under ST's
option set. **This mismatch would otherwise brick the first integration** — the
weights land at the wrong offset and the network reads garbage while the epoch
table looks perfect.

**Gate 3 — build and flash the *unmodified* ST audio app (0.5 d, board).**
Check OTP fuse state with `STM32_Programmer_CLI` + `OTP_FUSES_STM32N6xx.stldr`
**before** blowing anything — VDDIO2_HSLV/VDDIO3_HSLV are permanent. *Pass:* the
stock app prints AED JSON over UART. *Stop if* this fails — debug the toolchain
path with ST's known-good binary, not with your model in the loop.

**Gate 4 — swap in the Citrinet network, feed it a canned feature vector, ignore
the mic (0.5 d, board).** Embed one host-computed int8 `[80,800]` tensor, invoke
through the existing `ai_dpu.c` wrapper, keep `mcu_cache_clean_invalidate_range()`
exactly where `preproc_dpu.c` puts it, dump argmax token ids over UART. *Pass:*
on-device token ids match host ONNX Runtime argmax, and wall time is within ~2×
of 91.2 ms. **This is the first point where silicon can contradict desk research.**

> **CLOSED, and it did contradict desk research — twice.** It took twenty rounds,
> not half a day. Two Neural-ART defects had to be found on the board and worked
> around in the ONNX graph: a **stride-2 depthwise** convolution stalls the NPU
> forever (`model/fold_stride2.py`), and an **activation accelerator driving a
> convolution accelerator's data port** through the stream switch stalls forever
> at 36 sites (`model/break_relu_chain.py`). Both workarounds are bit-exact
> (`max|diff| = 0` over 3,075,000 elements) and the second is *faster* than the
> graph that stalled. The deployed build is **448 epochs, 0 SW, 0 hybrid, 300 kB
> cpuRAM2 + 425 kB npuRAM6, no PSRAM, 9.726 MB of weights at `0x70400000`**, and
> it measures **124.035 ms** — inside the ~100–250 ms band predicted in §5, and
> 2.1 % under the compiler's own cycle estimate.
>
> The pass criterion as written was not met and should not have been: on the
> canned tensor the device disagrees with host onnxruntime on 5 of 100 frames.
> That was resolved by measuring rather than by arguing — 64 utterances, device
> WER **5.81 %** against host **5.92 %**, paired difference −0.118 points,
> bootstrap 95 % [−1.290, +1.144], p = 0.897. "No difference" is *not*
> established; the interval is 2.43 points wide. `board/GATE4.md` Rounds 19-20.

**Gate 5 — log-mel frontend on the M55, with the level self-test built in from
the first line (2–3 d).** `LogMelSpectrogramColumn` (float32 variant, not
`_q15_Q8`); n_fft 512, win 400 symmetric Hann, hop 160, 80 mels, Slaney norm,
power spectrum, `ln(x + 2⁻²⁴)`. Add the gain stage (peak-normalise to 0.9 before
the STFT) and print the fraction of mel bins sitting at the guard. *Pass:*
features match `model/fe_reference.py` to within a few LSB of the int8 grid, and
guard occupancy < 20 % on live speech. *Stop if* guard occupancy stays high
after gain — the gain is in the wrong place in the chain.

> **Correction: the name in that first sentence is inverted.**
> `LogMelSpectrogramColumn_q15_Q8` **is** the float32 variant
> (`feature_extraction.c:250-324`); `_q15_f16_Q8` is the float16 one
> (`feature_extraction_f16.c:264`). The `q15` refers to the *input* PCM, not the
> arithmetic. The instruction was right and the name was wrong.
>
> **And ST's function turned out not to be usable at all.** It clamps
> non-positive mel energies to `FLT_MIN` and then takes `logf`, where NeMo adds
> `2⁻²⁴` before the log. Swapping the clamp in and changing nothing else moves
> **737,335 of 768,000 int8 values (96.01 %), max |Δ| 46 LSB**, floor −16.64 →
> −87.34. `firmware/src/citrinet_fe.c` is therefore a self-contained float32
> front end that does not use the ST library, and it reproduces `model/fe.py`
> exactly: **0 of 768,000 int8 values differ** over 12 utterances in the shipping
> CMSIS configuration. `firmware/FRONTEND.md` §2 and §5.
>
> **Still open, and this is the gate's real content:** none of it has run on the
> M55, so its cost is unmeasured; the stock capture path does not fit in RAM
> (1,026,240 B of buffers against a 1,047,552 B region, `firmware/WORKLIST.md`
> §5.4); and the gain staging of §2(d) is unsolved.

> **Corrected.** 1,026,240 B is `firmware/WORKLIST.md` §5.4's figure and it is a
> **2.6x undercount**, because it counts `proc_buff` + `audio_out` + the ring buffer
> and stops. `AudioBM_proc_t` also holds **two** processing contexts — `audioPreCtx`
> and `audioPostCtx` (`Projects/GS/Inc/audio_bm.h:61-62`) — and each carries
> `pCplxSpectrum[(NFFT/2+1)*2*COL]` of `float16_t` (`Projects/Dpu/audio_proc.h:64`,
> NFFT 512). At COL 800 that is 257x2x800x2 = **822,400 B each, 1,644,800 B for the
> pair**, giving ~**2,679,312 B** against a 1,047,552 B region.
> The derivation is checked against the stock geometry: at COL 96 it yields
> **268,048 B**, which is exactly the number `audio_bm.h:50` already records.
> Two consequences: the overrun is 2.6x rather than marginal, and it fails at
> **link** time, not in a runtime `malloc` as previously stated — `AudioBM_proc_t`
> alone exceeds the region by 2.07x. Gate 5 must therefore **replace** ST's
> pre/post-processing contexts, not resize the capture buffers; `citrinet_fe.c` is
> already self-contained and uses neither. Note the current build links at 62 %
> only because `ai_model_config.h:47` still carries the AED model's COL 96.

**Gate 6 — greedy CTC + detokeniser (0.5 d).** Argmax over 1025 int8 values ×
100 frames, collapse repeats, drop blank 1024, concatenate, `▁` → space. ~50 lines.

> **CLOSED on the host.** `firmware/src/citrinet_ctc.c` against
> `model/fe.py:greedy()` on 100 calibration-disjoint dev-clean utterances:
> **0 text disagreements over 9,226 characters, 0 argmax disagreements over
> 10,000 frames**, plus 480 synthetic logit matrices (39,972 of 48,000 frames
> carrying a tied argmax) with 0 disagreements. Tokenizer tables are 8,222 B,
> 8,757 B including code. The C decoder has **not** run on the M55.
> `firmware/test/results/gate6_ctc.json`, `docs/GATES-5-6.md`.

**Gate 7 — LCD graft (1–1.5 d).** Last, deliberately: it cannot fail in a way
that invalidates the model.

Total 8–11 developer-days, with the risk front-loaded into gates 0–4 (~1.5 days).

---

## 4. Top risks

1. **Gain staging × the log-zero guard.** Silent and catastrophic; not in the
   original plan. Detect at Gate 5, or earlier by instrumenting guard occupancy
   on the first frontend commit. Mitigate with pre-STFT peak normalisation
   applied *before* int16 truncation, and refuse to invoke when >20 % of bins
   sit at the guard.
2. ~~**ST int8 ≠ ONNX Runtime int8.**~~ **Detected at Gate 4, and quantified.**
   They do differ — 2.41 % of frames pick a different argmax — but the difference
   is concentrated at near-ties (17 host frames of 6,400 are *exact* argmax ties)
   and does not move WER measurably at n = 64: device 5.81 % vs host 5.92 %,
   paired 95 % CI [−1.29, +1.14] points. It is also **deterministic and
   schedule-independent**: two builds 616 epochs apart give one distinct
   100-token output across 23 captured runs. `board/GATE4.md` Rounds 19-20.
3. **Babble noise — i.e. the room the demo is given in.** WER at 5 dB SNR:
   white 22.3 %, pink 15.3 %, **babble 60.1 %**. Competing speech is the one
   noise type that destroys it, and demos happen in rooms full of talking
   people. Reverberation is free at arm's length (RT60 0.3 s, DRR +10 → 5.6 %)
   and costs 2× at ~1 m (RT60 0.6 s, DRR 0 → 12.0 %). "Hold the board near your
   mouth" is a legitimate and free mitigation.
4. **Grouped-convolution mapping is undocumented.** The word "group" appears
   **zero times** in `stneuralart_operator_support.html` r1.3. All 107 grouped
   convolutions reaching hardware is an empirical property of compiler
   4.0.1-20581 with no vendor commitment. Pin the toolchain; make the compile a
   regression gate on any bump. **The constant to pin is now 448 epochs / 0 SW /
   0 hybrid / 0 `ACTIV→CONVACC` links**, on `q800_relu4d_all.onnx` — and both NPU
   defects Gate 4 found are also properties of this compiler build, so a bump has
   to be re-checked against `compile/score_build.py`, not just against the epoch
   table.
5. **OTP fuses, and two flash workflows that overwrite each other.** Fuse
   blowing is permanent and this board's state is unverified. Separately,
   flashing the demo overwrites the external-flash weights that `zoo measure`
   also uses. Adopt a strict "measure, then flash demo, never interleave" rule,
   or give each workflow its own offset. **Both halves have since been settled.**
   The OTP was read before anything was flashed and word 124 already reads
   `0x00018000`, so Gate 3 carried no irreversible action (`board/OTP.md`). And
   the offsets *were* separated: `compile/stt_audio.mpool` moves the weight blob
   to `0x70400000`, which gives the application 3 MB instead of the 512 kB ST's
   own mpool leaves between `0x70100000` and the weights — and the Citrinet
   signed image does not fit 512 kB (`compile/gen_model.sh:59-68`).
   `board/flash_and_verify.sh:40-48` now refuses an image that would run into the
   weight blob; before that check existed, it silently overwrote it.

---

## 5. Genuinely unknown — only silicon settles these

- ~~**Real inference latency.**~~ **SETTLED at 124.035 ms.** The honest band
  below was **~100–250 ms**, and the measurement lands in it. The graph that runs
  is not the 628-epoch one this document costed, though: it is the 448-epoch
  rewrite of Gate 4, whose own scheduler estimate is 76,000,592 cycles, and the
  board comes in **2.1 % under** that. The epoch-transition-overhead worry did
  not materialise — measured is *below* estimate, not above. Run-to-run spread is
  0.005 % over 15 invokes. `board/GATE4.md` Round 19.
  > *Original text:* "91.2 ms is a *scheduler* estimate. It models the octoFlash
  > weight stream but excludes epoch-transition overhead; ~15 µs/epoch from prior
  > board work implies roughly +9.4 ms at 628 epochs. Nothing on this machine has
  > ever streamed 10 MB of weights, and the 1.6× cost figure is extrapolated well
  > beyond its 1.4 MB of evidence. Honest band: ~100–250 ms."
- ~~**On-device int8 fidelity.**~~ **MEASURED, and it is not a problem at this
  sample size.** Over 64 utterances the device's WER is 5.81 % against the host's
  5.92 %; the paired 95 % interval [−1.290, +1.144] contains zero. Per-frame
  argmax disagreement is 2.41 %, 6.25× enriched in the tightest decile of host
  logit margin, zero in the widest 20 %, and 70.8 % of it is blank-placement
  shifts CTC collapses away. **Not** established: "no difference" — the interval
  is 2.43 points wide. See risk 2, and `board/GATE4.md` Round 20.
- **M55 log-mel cost.** Nobody on this machine has measured *any* on-device
  audio frontend — every prior deployment fed the NPU from the host over serial.
  Estimates range 6–15 ms with no local measurement behind them. DWT-time it.
  A ~40× win is available: the stored filterbank is 2.4 % non-zero, so sparse
  triangular filters cost ~500 MAC/frame against 20,560 for a dense 80×257 matmul.
- **The DK microphone's real acoustic behaviour.** Every degradation above is
  simulated — synthetic RIRs, one real 0.1 s RIR, LibriSpeech-derived babble.
- ~~**Whether the audio app enables npuRAM3/4/5.**~~ **It does** —
  `Int_Mem_Config()` at `Projects/GS/Src/audio_bm.c:741-759` clocks SRAM2..SRAM6
  — but `stm32n6.mpool` does not declare them as pools, so the compiler cannot
  place activations there. Declaring them would widen 1,507,328 B → 2,883,584 B (`docs/MEMORY-MAP.md:44`; the 2,884,608 in `WORKLIST.md:77` is 1,024 B out)
  and invalidate the 618/0/0 evidence. At 8 s it does not matter: the deployed
  build uses 725 kB of the existing pool. `firmware/WORKLIST.md` §0.

---

## 6. Method, and its limits

Six parallel investigations (operator mapping, prior on-board audio deployments,
model export, alternative architectures, memory budget, firmware path), then
three adversarial challenges against the load-bearing claims, then synthesis.
The investigating agents did not merely read documentation — they downloaded the
checkpoint, performed the graph surgery, quantised on real LibriSpeech audio,
and ran the compiler.

Two caveats on that process, recorded rather than smoothed over:

- **One adversarial challenge (the memory gate) failed to return**, exhausting
  its retry budget. Its question was nonetheless answered more strongly than it
  could have answered it — by three real compiles against ST's actual audio-pool
  geometry, reproduced independently above from the raw reports.
- **The demo-viability challenge succeeded.** It refuted the claim that the plan
  as drafted would produce a compelling demo, on the strength of the 4-second
  truncation data and the gain-staging measurement. The plan in this document is
  the revised one; §2(a) and §2(d) are that challenge's findings.

**Every WER figure here is LibriSpeech dev-clean read speech through a simulated
channel.** The checkpoint's training set is broad — NGC lists 7,000+ hours
including Fisher, Switchboard, WSJ, NSC and Common Voice, so spontaneous desk
speech is less out-of-distribution than one might assume. But no number in this
document was produced by a human talking to this board.

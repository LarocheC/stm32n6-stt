# Gates 1 and 2 — verdicts, corrections, and the state of the board work

Scope: everything host-side. **The board was not touched.** No flash, no OTP, no
`stedgeai validate --mode target`.

Both gates were executed and then independently re-run by an adversarial verifier
who did not trust the executing agent's scripts. Where the verifier corrected a
track, **the verifier's number is the one used below** and the correction is named.

Full write-ups: [`eval/GATE1.md`](../eval/GATE1.md), [`compile/GATE2.md`](../compile/GATE2.md).
Firmware plan: [`firmware/WORKLIST.md`](../firmware/WORKLIST.md).
Pool arithmetic: [`docs/MEMORY-MAP.md`](MEMORY-MAP.md).

> **This document stops at Gate 2 and is not updated past it.** Gates 3, 4 and 6
> have since closed; the board record is [`board/GATE4.md`](../board/GATE4.md) and
> the current state of every gate is in [`../README.md`](../README.md). Two Gate 2
> numbers below were superseded on silicon: the deployed graph is **448 epochs**,
> not 618, and its weight blob is based at **`0x70400000`**, not `0x70180000`.
> Neither changes any verdict here — both are consequences of the two NPU defects
> Gate 4 found, and Gate 2's pass criterion (the report's flash base matches the
> mpool) is still the check that catches a mismatch.

---

## 1. Verdicts

### Gate 1 — int8 vs fp32 WER at the shipped 8 s window: **PASS**

**The deciding number: int8 costs +0.50 points of WER.** 4.91 % → 5.41 % on 373
utterance-disjoint dev-clean utterances that fit the 8 s window (227 → 250 errors
over 4,622 reference words). Paired bootstrap 95 % CI **[+0.07, +0.94]**,
P(cost > 1.0 pt) = 0.013.

Pre-registered pass band was ~1.0 point, stop at 2.0, ≥300 utterances
(`docs/FEASIBILITY.md:198–203`, committed in `5254fc8` and unmodified — the
threshold was set before the measurement, and n = 373 clears the floor).

```
python eval/run_gate1_8s.py 600     # -> eval/results/gate1_8s.json
```

Supporting: frame-level argmax agreement fp32 vs int8 **0.9716** (58,297/60,000),
against 0.9649 measured at 4 s. Full-reference WER (all 600 utterances, truncation
counted as deletions) 24.27 % → 24.65 %, i.e. int8 costs +0.38 points there.
The gate is judged on the *larger*, less flattering of the two deltas.

**Verifier outcome: not refuted, reproduced exactly.** An independent harness —
own decoder, own Levenshtein, own set reconstruction from the original calibration
scripts' code against the surviving scratchpad `recs.json` — re-ran all 600
utterances through both graphs and found **zero** per-utterance disagreements
across 1,200 model-utterance pairs. Frame agreement came out bit-identical.
An independent paired bootstrap (different seed, 20,000 resamples) gives
[+0.086, +0.939] against the reported [+0.067, +0.936]. Three further attacks —
seed-shopping (redraws at seeds 1/2/777 give +0.57/+0.41/+0.55, bracketing +0.50),
normalisation flattery (the `[^A-Z' ]` filter is provably inert on this
corpus/vocabulary pair; scoring with no filtering at all is bit-identical), and
baseline fairness (the fp32 graph is output-identical to the graph that was
actually quantised, 0/500 argmax disagreements) — all failed to move the number.

### Gate 2 — recompile against ST's own memory map and option string: **PASS**

**The deciding numbers: 0 pure-software epochs, 0 hybrid epochs, 947,200 B of
activations with 0 B in hyperRAM, weights based at 0x70180000.**

```
cpuRAM2    [0x34100000 - 0x34200000]:  500.000 kB /   1.000 MB ( 48.83 %)  activations
npuRAM6    [0x34350000 - 0x343C0000]:  425.000 kB / 448.000 kB ( 94.87 %)  activations
octoFlash  [0x70180000 - 0x74080000]:    9.728 MB /  63.000 MB ( 15.44 %)  weights
hyperRAM   [0x90000000 - 0x91000000]:        0  B /  16.000 MB (  0.00 %)
Total number of epochs                               618
>> pure software (SW) epochs                           0
>> hybrid epochs (using both software and hardware)    0
```

All four pass criteria met. The IO deployment contract is byte-identical to the
screening compile over every `STAI_NETWORK_IN*`/`OUT*` define, so nothing
downstream of the model changes.

**Verifier outcome: not refuted, reproduced bit-for-bit.** The verifier ran its
own compile from scratch, pointing directly at
`vendor/…/Projects/X-CUBE-AI/models/stm32n6.mpool` and `user_neural_art.json`
rather than at the gate's copies. `network_generate_report.txt` is identical
modulo `/tmp` paths and the created-date; `network.c` has zero diff lines modulo
paths; the weight blob md5 is identical (`248bed980b5497c5e2e687df14540d6a`).
Provenance closes on the compiler's own `atonn_options.ini`, which is written by
the atonn backend and not by the claimant: it records
`load-mpool-file=…/st_audio` (md5 `3962913c702e781e24ba6bbe431ee10c`, matching
ST's file) and exactly ST's eight options with no zoo-only flags. That evidence is
now archived in-repo at **`compile/reports/g800_st_verify/`**.

---

## 2. What changed relative to `docs/FEASIBILITY.md`

`docs/FEASIBILITY.md` has **not** been edited. Everything below is a delta against
it that a reader of that document needs.

### 2.1 Established figures that moved

| | FEASIBILITY says | Now measured | Why |
|---|---|---|---|
| 8 s epochs | 628 | **618** | ST's option set omits `--Oauto-sched` |
| 8 s activations | 625 kB, **42.5 %** of pool | **925 kB, 62.8 %** | same |
| 8 s scheduler latency | 91.2 ms @ 1 GHz | **91.89 ms** | same |
| 12 s placement | on-chip (600 kB cpuRAM2 + 417.19 kB npuRAM6) | **150 kB spills to hyperRAM** at 0x90000000 under ST's shipped options | same |
| 8 s full-reference WER | 20.0 %, coverage 0.84 (n=150) | **24.27 %, coverage 0.797 (n=600)** | a different, larger, longer draw — **not** a regression |
| npuRAM3/4/5 — "unknown" (§5) | open question | **enabled** by `Int_Mem_Config()`, `audio_bm.c:741-760`, but declared to nobody | source read |
| "~40× sparse filterbank win is available" (§5) | available upside | **already implemented** by ST's `MelFilterbank()`; 41.1× measured (500 vs 20,560 MAC/frame) | source read |
| Gate 5: "float32 variant, not `_q15_Q8`" | — | the naming is **inverted**: `LogMelSpectrogramColumn_q15_Q8` *is* the float32 one | source read |
| LCD line width "47 chars, already derived by ST" (§2(e)) | 47 | 47 is `N_PRINTABLE_CHARS`, sized for Font24; the OD app renders Font20 → **57** fit | measured from the font tables |
| Effort: Gate 5 / total | 2–3 d / 8–11 d | **3 d / 6.5 d** | gates 0–2 closed, Gate 6 half-built, Gate 5 grew |

Two of these deserve a sentence each.

**The 42.5 % → 62.8 % move is an option-string effect, not a memory-map effect.**
Isolated by five further compiles on ST's own mpool: `--Oauto-sched` accounts for
100 % of the delta on its own (925 → 625 kB, 618 → 628 epochs, −680,600 cycles).
`--enable-virtual-mem-pools`, the other flag the gate brief flagged as a suspect,
is a **byte- and cycle-exact no-op** on this graph and this mpool. Once
`--Oauto-sched` is added, ST's mpool reproduces the screening numbers *exactly*,
so the octoFlash rebase, the 16 MB hyperRAM row and the absent npuRAM3/4/5 rows
cost nothing at 8 s. They only move the flash base — which was the point of the gate.

**The 20.0 % → 24.27 % move is not a regression.** Replaying `run_8s.py`'s exact
150-utterance selection (seed 1, no lead-in) through the new harness after corpus
retargeting gives **19.98 % / 0.838** against the published 20.0 % / 0.84 — and
the verifier reproduced that at 19.9793 %. The 600-utterance draw is simply longer
(median 6.32 s vs 5.55 s), so more of it is truncated. The 600-utterance number is
the better estimate.

### 2.2 New figures FEASIBILITY does not carry

- **8 s fp32/int8 WER row: 4.91 % / 5.41 %** (fits-window), alongside the existing
  4 s row of 5.60 % / 6.09 %. §2 has no 8 s int8 row today.
- **The 0.3 s capture lead-in costs 1.42 points of full-reference WER** and buys
  nothing. Inherited from `model/q800.py`'s calibration placement: 4,800 samples of
  a fixed 8 s buffer spent on silence. Removing it: 24.27 % → 22.85 %, coverage
  0.797 → 0.812, fits-window WER unchanged (4.91 % → 4.84 %). Verifier reproduced
  the 1.416-point difference independently.
- **The 1,507,328 B pool is a policy choice, not a ceiling.** npuRAM3/4/5 are
  powered and clocked at runtime but not declared in ST's mpool; declaring them
  would widen the pool to 2,883,584 B. Upside only — nothing in the plan needs it,
  and doing it speculatively invalidates the 618/0 evidence.
- **There is a 512 KiB application flash slot** at `0x70100000..0x70180000`,
  between the signed app and the weight blob. Stock `aed_bm` occupies 238.8 KiB,
  so ~273 KiB of headroom for everything gates 4–7 add. Overflowing it silently
  overwrites the weights — the same failure mode Gate 2 was written to prevent,
  reached from the other side. Not documented by ST anywhere.

### 2.3 The calibration set is **not** contaminated — but the reason FEASIBILITY gives is wrong

This is the headline of the deferred Part A check, and it is more subtle than
either outcome the gate brief anticipated.

- The gate brief, `model/README.md` and `docs/FEASIBILITY.md:205–210` all point the
  disjointness check at `model/quant_real.py`. **That script built the 4 s model.**
  The shipped 8 s graph `q800_real.onnx` was calibrated by `model/q800.py` with a
  different filter (`4.0 ≤ d ≤ 7.5`, not `d ≤ 3.5`) and a different slice
  (`perm[:48]`, not `perm[300:364]`). All three calibration sets were checked.
- **`cal_800` — the shipped model's calibration set — overlaps nothing.** Zero
  intersection with all eight evaluation sets in the repository.
- **`quant_real.py`'s disjointness comment is nevertheless false.** Its "perm seed
  differs" argument is not a guarantee: 11 of `run_int8.py`'s 120 draws land in
  `cal_400`. 13 of `run_ab.py`'s 120, 7 of 100 in `run_fe`/`run_gain`. Different
  seeds over the same 591-element pool overlap at these sample sizes.
- **No published number is contaminated anyway**, for two independent reasons:
  `eval/run_int8.py` defensively rebuilds the `cal_400` key set at lines 9–13 and
  filters it out before scoring (its 109 surviving utterances carry exactly the 804
  reference words recorded in `eval/results/int8.json`); and every other overlapping
  script evaluates the fp32 `model.onnx`, which calibration data cannot influence.
  **The safeguard was in the evaluation script, not the calibration script.**
- The Gate 1 Part B set additionally excludes the union of all three calibration
  sets (158 unique keys) by construction, asserted in-script and confirmed
  independently at 0 intersection.

**Correction the verifier forced, and it stands:** "held-out" means
**utterance-disjoint, not speaker-disjoint**. All 26 `cal_800` speakers and 34 of
its 35 chapters reappear in the Gate 1 evaluation set. dev-clean contains only 40
speakers, so speaker-disjoint evaluation is impossible within this corpus. This is
immaterial for MinMax PTQ, which fits activation ranges rather than learning
content — but the unqualified phrase overstates the isolation, and `eval/GATE1.md`
§5 and the gate summary should both say *utterance-disjoint*.

### 2.4 Smaller corrections the verifier logged (none change a verdict)

- `eval/GATE1.md` §1's overlap table lists `run_occ part 1` and `run_pad` as two
  sets. They are literally the same 38 utterances (the `d ≤ 2.0` pool has only 38
  records). The table's 8 rows are 7 distinct sets, and the headline "total overlap
  67" double-counts those 7 keys.
- `eval/GATE1.md` §1 attributes `cal_800`'s zero overlap to the duration filter.
  True for 7 of 8 evaluation sets; `run_8s` is unfiltered, drawn from all 2,703
  records, where the expected overlap is 2.7. Its observed 0 is chance, not
  structure. The existing "does most of the work" hedge is load-bearing.
- `eval/results/gate1_8s.json` →
  `auxiliary.lead_in_sensitivity_fp32.OFF_4800.full_reference_wer_pct` reads
  24.267698…, inconsistent with its own 2948/12148 = 24.267369…. Cosmetic (3e-4
  points) but indicates one hand-entered value in an otherwise computed block.
- Gate 1 deviated from the pre-registered method: the brief said retarget
  `run_8s.py`/`run_int8.py`, the gate wrote a new harness `eval/run_gate1_8s.py`.
  Better execution, disclosed, and validated against the published figure — but the
  deviation belongs on the record next to the pre-registered threshold.
- `compile/GATE2.md` §4 says "three further compiles"; the evidence directory shows
  five (`out_vmp`, `out_sched`, `out_both`, `out_1200`, `out_1200s`).
- **62.8 % of the pool reads as more headroom than exists.** The pool is not
  fungible: **npuRAM6 is at 94.87 % — about 23 kB spare** — while cpuRAM2 sits at
  48.83 %. Quote the per-pool occupancies alongside the aggregate. Any future change
  that grows an npuRAM6-resident tensor has ~23 kB, not 37 % of 1.44 MB.
- `docs/MEMORY-MAP.md`'s closing line ("8 s already fits the narrow pool at 42.5 %")
  is now stale for the same reason as the README table.

---

## 3. State of the firmware work list — is Gate 3 ready?

`firmware/WORKLIST.md` is a file-level plan for gates 3–7 written from the actual
vendor trees rather than from documentation. **Gate 6 is half-built and verified**:
`firmware/tools/gen_tokenizer.py` and `firmware/inc/citrinet_vocab.h` exist,
measure **8,222 B** of `.rodata` on Cortex-M55, and round-trip 1025/1025 pieces
against the Python detokeniser.

**Gate 3 is ready to execute in every respect except one, and that one is
irreversible.** See §4.

Tooling is present, just not on `PATH`:

```
/home/claroche/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI   # v2.22.0
.../bin/STM32_SigningTool_CLI
.../bin/ExternalLoader/OTP_FUSES_STM32N6xx.stldr
.../bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr
```

### The first command

Not `make`. Read the OTP state, which is the only read-only observation that
becomes impossible after any later step:

```bash
export STM32CP=/home/claroche/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin
$STM32CP/STM32_Programmer_CLI -c port=SWD mode=HOTPLUG \
    -el $STM32CP/ExternalLoader/OTP_FUSES_STM32N6xx.stldr -otp displ
```

`-otp displ` is read-only (`-otp write` / `-otp lock` / `-otp fwrite` are the
mutating verbs; do not use them). Record the output in the repo before anything
else touches the board.

**UNVERIFIED:** the CLI's generic help documents `-otp displ [word=<id>]` as
covering "up to 96 OTP words [0 to 95]", but the fuse this project cares about is
**word 124 (HCONF1), bits 15 and 16**. Whether the N6 external loader extends that
range, or whether the full-structure dump shows word 124, has not been checked —
it cannot be checked without the board. If `word=124` is rejected, take the full
dump and locate HCONF1 in it.

Everything after that (3.3 `make bm -j8`, 3.4 sign + flash, 3.5 UART banner at
14400 8N1) is mechanical and unblocked. Two things worth pre-loading:

- Local toolchain is `arm-none-eabi-gcc 10.3.1`; ST's banner says the reference
  binaries were built with GCC 13.3.1. First difference to eliminate if anything smells.
- Add an `arm-none-eabi-size` check of the signed binary against **0x80000** to the
  build and fail loudly (§2.2, the 512 KiB slot). Nothing in ST's tooling checks it.

---

## 4. Decisions that must be made by a human before the board is touched

### 4.1 OTP fuses — and a correction to the work list

`firmware/WORKLIST.md` §3.2 says `USE_STM32N6570_DK` "is not in the Makefile's
`C_DEFS`, so on the Makefile path this compiles out — but the CubeIDE `.cproject`
may define it." **Both halves are wrong, and the conclusion inverts.**

```
$ grep -rn "USE_STM32N6570_DK" vendor/STM32N6-GettingStarted-Audio/
Projects/GS/Src/audio_bm.c:107:#if (defined(USE_STM32N6xx_NUCLEO) || defined(USE_STM32N6570_DK))
Drivers/BSP/STM32N6570-DK/stm32n6570_discovery.h:59:#if !defined (USE_STM32N6570_DK)
Drivers/BSP/STM32N6570-DK/stm32n6570_discovery.h:60:#define USE_STM32N6570_DK
Drivers/BSP/STM32N6570-DK/stm32n6570_discovery.h:61:#endif
```

The BSP header **self-defines the macro**, and `audio_bm.c:22` includes that header
85 lines above the guard. `.cproject` does not define it and does not need to
(its define list is `APP_BARE_METAL`, `ARM_MATH_CM55`, `STM32N657xx`, `DEBUG`,
`USE_FULL_ASSERT`, `VECT_TAB_SRAM`, …). `HAL_BSEC_MODULE_ENABLED` is live at
`Projects/GS/Inc/stm32n6xx_hal_conf.h:38`, so `fuse_vddio()` is compiled in.

**Therefore: `make bm` + flash + run of the *unmodified* ST app will program OTP
word 124 bits 16 (HSLV_VDDIO2) and 15 (HSLV_VDDIO3) if they are not already set.**
Not "may". `fuse_hardware_conf()` (`Projects/Common/misc_toolbox.c:69-107`) reads
first and skips if the bit is already 1, so it is idempotent — but on a virgin
board the first boot blows them, and per ST's own README *"when OTP fuses are set,
they can not be reset."*

This is verified by source read only. **UNVERIFIED: the actual fuse state of this
board.** It may well already be set, in which case the whole question is moot.

The decision, which is the engineer's and not an agent's:

1. Read the state first (§3). If both bits are already 1 — likely, if this board
   has ever run an ST N6 demo — proceed with no further thought.
2. If they are 0: decide deliberately whether to blow them, or comment out the
   `fuse_vddio()` call in a `firmware/` fork of `audio_bm.c` for Gate 3 and revisit.
   VDDIO2/3_HSLV must be 1 for the octoFlash and PSRAM to run at speed, so they
   almost certainly need blowing eventually — the objection is to blowing them as
   an unremarked side effect of `make`.

### 4.2 Do not let a demo flash disturb the zoo measurement setup

`docs/FEASIBILITY.md` risk 5 still holds unchanged and is now concrete: the weight
blob goes to **0x70180000**, the same external flash region `zoo measure` writes.
Adopt "measure, then flash demo, never interleave", or give each workflow its own
offset, **before** the first Gate 4 flash rather than after the first confusing result.

### 4.3 Two choices worth making explicitly rather than inheriting

- **The 0.3 s capture lead-in.** Worth 1.42 points of full-reference WER for zero
  measured benefit. If push-to-talk button debounce needs a lead-in, budget the
  accuracy against it; if not, delete it. A Gate 5 decision that should not be
  inherited from a quantisation script.
- **The compiler option string for shipped firmware.** Recommend ST's string **plus
  `--Oauto-sched`**:
  `--Ocache-opt -O3 --Os --native-float --Omax-ca-pipe 4 --cache-maintenance --csv-file network --all-buffers-info --Oauto-sched`.
  It costs nothing, saves 300 kB of cpuRAM2 and 0.68 ms at 8 s, and is the
  difference between on-chip and PSRAM-spilled at 12 s. Leave
  `--enable-virtual-mem-pools` off. Adding it changes the regression constant from
  618 epochs back to 628 — pick one and pin it.

### 4.4 Two hard blockers that will look like board failures if not pre-empted

Neither is a decision so much as a thing to do before the first Citrinet flash,
but both fail *silently* and will be misread as silicon problems.

- **`AiDPUCheckModel()` rejects our model.** `Projects/Dpu/ai_dpu.c:99-107` tests
  `if (STAI_FORMAT_FLOAT32 != outputs[i].format && 2 != outputs[i].shape.size)`.
  `stai_shape` is `stai_array_s32`, so `.size` is the **rank**. Our output is S8,
  rank 3, `{100,1025,1}` — both conjuncts true → `DPU_ERROR` → `Error_Handler()` at
  `ai_dpu.c:148-152`, which is `__disable_irq(); while(1);`. **The app hangs at
  startup with no UART output and looks like a boot failure.** ST's AED model
  survives it only because its output is float32. Patch before flashing.
- **No post-inference cache invalidate exists anywhere in the app.**
  `preproc_dpu.c:144` cleans the *input* before the run; nothing invalidates the
  102,500 B output after it. Add
  `mcu_cache_invalidate_range()` immediately after `stai_network_run()`. First
  suspect if Gate 4's argmax is plausible-but-wrong.

---

## 5. Updated risk list

Against `docs/FEASIBILITY.md` §4.

| # | risk | direction | why |
|---|---|:--:|---|
| 1 | Gain staging × the log-zero guard | **worse** | still the top risk, and a second mechanism found |
| 2 | ST int8 ≠ ONNX Runtime int8 | **unchanged** | Gate 1 explicitly does not bound it |
| 3 | Babble noise | **unchanged** | nothing new measured |
| 4 | Grouped-convolution mapping undocumented | **better** | reproduced under a second option set, mpool and operator |
| 5 | OTP fuses + two flash workflows | **worse** | fusing is unconditional, and a third overwrite path found |

**1 — Gain staging (worse, and now two problems).** The original risk stands
unaltered: at −54 dBFS, 97.9 % of mel bins sit at the guard and WER goes
5.83 % → 35.28 % with every log reporting a clean run. What is new is that the app
is wired to **float16** preprocessing (`app_config.h:51`, `preproc_dpu.h:39-46`,
`audio_bm.h:28-33` `#error`s if you change it), and float16 cannot represent
NeMo's mel dynamic range: at LibriSpeech's native level 47.40 % of mel energies
fall below fp16's smallest normal and 5.54 % are hard zeros; scaled to −54 dBFS
that becomes 99.77 % and **76.22 %**. Separately
`feature_extraction_f16.c:310` assigns `FLT_MIN` (1.175e-38) to a `float16_t`,
which underflows to 0, and the next line takes `logf(0)` = −∞. So the level problem
now has an arithmetic twin that a gain stage alone does not fix. Mitigations are
unchanged and adequate — the MDF gain register (`MDF_GAIN` = 2 today, range −16..+24
in ~3 dB steps, settable during acquisition) and the `/256` shift at
`Patch/stm32n6570_discovery_audio.c:3172,3197`, both upstream of int16 truncation —
plus a mandatory float32 migration. Gate 5 grew from 2–3 d to 3 d largely for this.
The guard-occupancy self-test goes in with the *first* frontend commit.

**2 — ST int8 ≠ ORT int8 (unchanged, and Gate 1 says so explicitly).** Gate 1
bounds quantisation *as a concept*, under ONNX Runtime QDQ semantics, at 0.5
points. It says nothing about ST's Neural-ART int8, which the zoo has already
recorded diverging at cosine 0.996 with a systematic +0.23 mean bias on another
model. **Gate 4 remains the first measurement that can contradict desk research.**
Reading Gate 1's PASS as protection against risk 2 would be the single most
expensive misreading of this document.

**3 — Babble noise (unchanged).** 60.1 % WER at 5 dB SNR, still the one noise type
that destroys it, still simulated, still unmeasured on the real microphone.
"Hold the board near your mouth" remains free and legitimate.

**4 — Grouped-convolution mapping (better, modestly).** The 107 grouped
convolutions reaching hardware is now reproduced across a second, independent
option set (ST's, without `--Oauto-sched`), a second mpool geometry, and a third
verifier-run compile — all still 0 SW / 0 hybrid. That is broader empirical support
than §4 had, though it remains a property of build 4.0.1-20581 with no vendor
commitment. **Pin the toolchain.** Update the regression constant: the gate is
`618 epochs / 0 SW / 0 hybrid` under ST's shipped options, `628 / 0 / 0` with
`--Oauto-sched`.

**5 — OTP and overwriting flash (worse, on both halves).** The fuse half moved from
"the `.cproject` may define it" to "the BSP header defines it and the stock app
will program the fuses on first run" (§4.1). The flash half gained a third
overwrite path nobody had named: the **512 KiB application slot** at
0x70100000..0x70180000, which the signed binary silently runs into the weight blob
if it exceeds — producing garbage inference with a clean-looking log, exactly the
failure Gate 2 was built to prevent, approached from the opposite direction.

### A sixth risk, not in the original five: silent compiles that pass every check

The 12 s finding is the concrete instance and it generalises. Under ST's shipped
option set, `q1200_real.onnx` puts 150 kB of activations in PSRAM at 0x90000000 —
**and the epoch table still reads 618 / 0 SW / 0 hybrid.** Worse, the scheduler's
cycle total does not penalise it: the spilled build reports 123,775,024 cycles
against 124,080,680 for the fully on-chip build, i.e. the spilled version looks
*faster*. Neither of the two summary numbers this project has been quoting would
catch it.

Make the **placement line** a build-time regression gate, not just the epoch table.
Two greps over `network_generate_report.txt`:

```
grep -q 'hyperRAM .* 0  B'      network_generate_report.txt
grep -q 'octoFlash  \[0x70180000' network_generate_report.txt
```

---

## 6. Documentation debt this pass created

Listed so it is decided rather than left to diverge.

1. `README.md` and `docs/FEASIBILITY.md` publish 42.5 % / 625 kB / 628 epochs /
   91.2 ms for 8 s. Those hold only with `--Oauto-sched`, which ST does not ship.
   Under ST's options it is 62.8 % / 925 kB / 618 / 91.89 ms. Same for the 12 s row,
   which is on-chip only with `--Oauto-sched`. (README's Status section is updated;
   its table is footnoted rather than rewritten.)
2. `docs/FEASIBILITY.md`'s Gate 1 blockquote points the disjointness check at
   `quant_real.py`, which is the 4 s model's script. Also consider adding the 8 s
   fp32/int8 row (4.91 % / 5.41 %) beside the existing 4 s 5.60 % / 6.09 %.
3. `model/quant_real.py:10` carries a false comment. The code is fine; only the
   claim is wrong. Gate 1 left the script byte-identical as the record of what
   produced `q400_real.onnx` and corrected `model/README.md` instead. Pick one.
4. `eval/GATE1.md` §5 and the gate summary should say **utterance-disjoint**, not
   held-out. The overlap table double-counts one 38-utterance set.
5. `compile/reports/g800_st/user_neural_art.used.json` records `memory_pool` as a
   `/tmp` scratchpad path and is not self-evidencing. The verifier's archive at
   `compile/reports/g800_st_verify/` closes the gap for the 8 s compile; `g1200_st`
   and the three ablation runs still have primary artifacts only in a scratchpad.
6. Eight `eval/run_*.py` still carry dead `/tmp` paths. Retarget them as a batch
   when they are next rerun — retargeting without rerunning breaks the
   correspondence between each script and the `results/` file it produced.
7. **`eval/results/recs.json` record order is load-bearing.** Every calibration and
   evaluation set in the repository is an RNG permutation of indices into a filtered
   slice of that list. Sorting or regenerating it silently redefines every set and
   makes the disjointness result unreproducible. Flagged in `eval/README.md`.
8. `docs/MEMORY-MAP.md`'s closing "42.5 %" is stale (§2.4).
9. ST's `generate-n6-model.sh`, `build-firmware.sh` and `sign-and-flash-model.sh`
   ship with `<path_to_stedge>` / `<PathtoCube IDE>` placeholders and Windows `.exe`
   invocations. None runs as shipped; retarget before Gate 3/4.
10. `compile/audio_profile.json`'s `memory_pool` points at a scratchpad that may no
    longer exist — same class of problem as the `eval/*.py` paths.

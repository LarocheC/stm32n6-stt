# Gate 5 — the log-mel front end in C

`firmware/src/citrinet_fe.c` + `firmware/inc/citrinet_fe.h` is a self-contained
float32 NeMo log-mel front end for `stt_en_citrinet_256_gamma_0_25`. It does not
use ST's `STM32_AI_AudioPreprocessing_Library`, because that library cannot
express this transform (§2, with the cost measured).

**It has been proven on a workstation, not on silicon.** Compiled natively with
`gcc` and run against `model/fe.py` on 12 LibriSpeech dev-clean utterances, the
shipping configuration reproduces **all 768,000 int8 feature values exactly** —
zero disagreements, not "within a few LSB". §5 has the table, §7 has the list of
things that remain unproven until the M55 runs it.

Reproduce everything below with

```bash
source /home/claroche/stm32n6-deployment-zoo/.venv/bin/activate
bash firmware/test/run_fe_parity.sh 12          # ~3 min, no board, no ARM toolchain
```

which writes `firmware/test/results/fe_parity.json`.

---

## 1. The spec

`model/fe.py` is both the authority and the test oracle. `model/fe_reference.py`
carries the provenance of each constant back to the NeMo checkpoint.

| stage | value |
|---|---|
| sample rate | 16000 |
| `n_fft` / `win_length` / `hop` | 512 / 400 / 160 |
| `n_mels` | 80 |
| window | **symmetric** Hann, `scipy.signal.get_window("hann", 400, fftbins=False)` |
| pre-emphasis | `y[0] = x[0]`, `y[n] = x[n] − 0.97·x[n−1]` |
| framing | librosa `center=True`, `pad_mode="constant"` → the signal is zero-padded by `n_fft/2 = 256` on both sides, and the 400-sample window sits at offset `(512−400)/2 = 56` inside each 512-point frame |
| spectrum | power, `|S|²` |
| mel | librosa Slaney scale + Slaney norm, `htk=False`, fmin 0, fmax 8000 |
| log | **`ln(mel + 2⁻²⁴)`** — additive, not a clamp |
| normalise | per mel bin over time, `(x − mean) / (std(ddof=1) + 1e-5)` |
| quantise | `clip(rint(x / 0.120522417128086), −128, 127)` |
| window length | 128,000 samples = 8.000 s → 801 whole frames, of which the graph takes the first **800** |
| output | 64,000 int8, mel-major, matching `{80,800,1}` `CHANNEL_FIRST` |

Three of these are places where a plausible implementation silently differs, and
each was measured rather than assumed:

**(a) The log guard.** `2⁻²⁴ = 5.9604644775390625e-8`, added. `ln(2⁻²⁴) = −16.635532`
is the floor. See §2 for what happens if you clamp instead.

**(b) `ddof = 1`.** NeMo divides the sum of squared deviations by `N−1`, not `N`.
At `T = 800` that is a 0.06 % difference in the variance — not enough to move an
int8 value on its own, but it is free to get right and expensive to debug later.
The implementation uses the two-pass form (mean, then sum of squared deviations)
in double, which is also what NumPy's `std()` does. §6 measures the alternatives.

**(c) Round-half-to-even.** `np.round` is round-half-to-even; C's `roundf` is
round-half-away-from-zero. `citrinet_fe.c` uses **`rintf`**, which under the
default rounding mode is round-half-to-even and therefore matches the oracle.
Measured on 768,000 values: **0 sit exactly on a `.5` boundary**, 107 sit within
1e-4 LSB of one, so on this corpus the two would have agreed anyway — but the
one place the portable-FFT build disagrees with the oracle at all is exactly such
a near-tie (oracle value −8.50000 LSB, §5).

---

## 2. What ST's library could not do, and what it would have cost

`Middlewares/ST/STM32_AI_AudioPreprocessing_Library/`, entry point
`LogMelSpectrogramColumn_q15_Q8()` (`feature_extraction.c:250-324`).

| # | NeMo needs | ST does | verdict |
|---|---|---|---|
| 1 | `ln(x + 2⁻²⁴)` | `if (x <= 0) x = FLT_MIN; logf(x);` (`feature_extraction.c:293-298, 315-317`) | **fatal** |
| 2 | per-mel-bin mean/std over the window, ddof=1 | no normalisation of any kind exists in the library | **fatal** |
| 3 | float32 dynamic range | the app is wired to float16 (`app_config.h:51`, `audio_bm.h:28-33` `#error`s if you change it) | **fatal** |
| 4 | quantise after the statistics are known | quantises inline, per column (`feature_extraction.c:320-323`) | structural |
| 5 | symmetric Hann | `Window_Init()` builds the periodic one (`window.c:78`) | LUT, free to fix |

**Item 1, measured.** `firmware/test/fe_parity.py:st_library_ablation()` runs the
identical pipeline twice on the same audio — same window, same filterbank, same
per-feature normalisation — changing only that one line. Over the same 12
utterances:

```
ST clamp changes 737,335 of 768,000 int8 values (96.01 %), max |Δ| 46 LSB
log-mel floor:  NeMo −16.64      ST −87.34
```

It is not a corner case that bites in quiet rooms. The clamp moves 96 % of the
feature matrix, because it pushes the floor 70 log units down and the per-feature
z-score is computed over that floor.

**Item 3, measured.** `2⁻²⁴` is numerically identical to float16's smallest
subnormal (5.9604645e-8). So `FLT_MIN` (1.18e-38) assigned into a `float16_t` in
`feature_extraction_f16.c:310` underflows to exactly 0, and the next line takes
`logf(0)` = −∞. The parity harness counts, per utterance, how many mel energies
fall below that threshold: between **3,098 and 37,721 of 64,000** on ordinary
dev-clean speech at native level. The f16 path is not marginal, it is broken for
this model at any input level.

**Item 4 is why the shape had to change.** The mean and standard deviation are
not known until the last column exists, so the int8 conversion cannot live inside
the per-column call. That forces the two-pass structure of §3 and is the reason
`LogMelSpectrogramColumn_q15_os8_batch` (`feature_extraction.c:355`) is also
unusable — it quantises per column and transposes.

---

## 3. Structure

```
pass 1, per column t in 0..799
    build_frame()   int16 -> float, /32768, pre-emphasis, centre zero-padding
    column()        symmetric Hann at offset 56 -> 512-pt real FFT -> power[257]
                    -> 500 sparse MACs -> 80 mel energies
                    -> count energies below 2^-24            <-- telemetry, always
                    -> ln(e + 2^-24) into scratch[80][800]
pass 2, per mel bin b
    mu  = mean over the 800 frames
    sd  = sqrt(sum((x-mu)^2)/(T-1)) + 1e-5                   <-- ddof = 1
    out[b*800 + t] = clip(rint(((x - mu)/sd) / scale) + zp, -128, 127)
```

Two entry points, same code:

* `citrinet_fe_run()` — one shot over a whole int16 buffer. What the host harness
  and Gate 5's first bring-up use.
* `citrinet_fe_reset()` / `citrinet_fe_column()` / `citrinet_fe_finish()` — the
  streaming decomposition, so the capture-driven path of `WORKLIST §5.7` can run
  pass 1 out of the MDF callback without buffering the utterance. Pre-emphasis is
  a one-sample-state filter and the framing overlap is 512−160 = 352 samples, so
  nothing about pass 1 needs the whole utterance to be resident.
  `citrinet_fe_build_frame()` is exposed so the streaming caller uses the same
  framing and padding arithmetic rather than reimplementing it.

The FFT is behind one switch. `CITRINET_FE_USE_CMSIS` defaults to 1 on ARM
(`arm_rfft_fast_f32`) and 0 elsewhere, where a portable radix-2 FFT in the same
file takes over. That is not a convenience — it is what makes the host parity test
possible at all, and §5 measures both.

---

## 4. Tables

`firmware/tools/gen_mel_tables.py` → `firmware/inc/citrinet_fe_tables.h`
(`--check` re-derives and diffs; the runner calls it). Every literal is `%.9g` of
the float32 value, and `FLT_DECIMAL_DIG` is 9, so the C tables are bit-for-bit the
float32 arrays `model/fe.py` uses.

The filterbank is **500 non-zero of 20,560 = 2.432 %**, every row one contiguous
run of 2..18 bins, highest FFT bin touched **255** — so the Nyquist bin is never
read and CMSIS's 256-bin filterbank-generator bug is moot here.

| table | form | bytes |
|---|---|---:|
| `kCitrinetMelW[500]` | float32, concatenated runs | 2,000 |
| `kCitrinetMelStart[80]` | uint16 | 160 |
| `kCitrinetMelLen[80]` | uint8 (max run 18) | 80 |
| **filterbank, sparse** | | **2,240** |
| filterbank, dense `[80][257]` float32 | *not used* | 82,240 |
| `kCitrinetHann[400]` | float32 | 1,600 |
| **total** | | **3,840** |

**36.7× smaller than dense**, and 500 multiply-accumulates per frame instead of
20,560 — **41.1× fewer**. Measured `.rodata` of `citrinet_fe.o` for
`arm-none-eabi-gcc -mcpu=cortex-m55 -Os` is exactly **3,840 B**, confirming
nothing else leaked into rodata.

---

## 5. Measured host parity

`bash firmware/test/run_fe_parity.sh 12`, gcc 11.4.0, 12 dev-clean utterances
picked deterministically (`np.random.default_rng(0)`), each placed in the real
128,000-sample deployment window with 4,800 samples of lead-in silence, written
as 16-bit PCM. The C binary dumps **the exact int16 buffer it was fed**, and the
oracle is run on that, so the two sides cannot disagree about their input.

64,000 int8 values × 12 utterances = **768,000 values per configuration**.

| build | int8 differing | max &#124;Δ&#124; | >1 LSB | log-mel max &#124;Δ&#124; | scratch |
|---|---:|---:|---:|---:|---:|
| **CMSIS `arm_rfft_fast_f32`, f32 scratch — ships** | **0** (0.00000 %) | **0** | 0 | 1.688e-04 | 256,000 B |
| portable FFT, f32 scratch | 1 (0.00013 %) | 1 | 0 | 9.632e-05 | 256,000 B |
| CMSIS, **fp16** scratch | 3,982 (0.51849 %) | 1 | 0 | 7.812e-03 | 128,000 B |
| CMSIS, sum/sumsq variance, **double** | 0 (0.00000 %) | 0 | 0 | 1.688e-04 | 256,000 B |
| CMSIS, sum/sumsq variance, **float32** | 236 (0.03073 %) | 1 | 0 | 1.688e-04 | 256,000 B |

Also measured, every run:

* **Log-floor telemetry agrees exactly.** The C counter and an independent NumPy
  count of `mel_energy < 2⁻²⁴` matched in all 60 runs, on counts ranging from
  3,098 to 37,721 per utterance. The telemetry is not approximate.
* **Deterministic.** Two runs of the same binary on the same input produce
  byte-identical int8.
* **End to end.** Feeding the C front end's own int8 tensor to
  `artifacts/onnx/q800_real.onnx` and greedy-decoding reproduces the LibriSpeech
  reference text, truncated at the 8 s window — e.g. `5895-34622-0009` →
  *"unknown people had worked upon his face he on the other hand had worked on his
  mind and behind this well exx"* against the reference *"UNKNOWN PEOPLE HAD
  WORKED UPON HIS FACE HE ON THE OTHER HAND HAD WORKED ON HIS MIND AND BEHIND THIS
  WELL EXECUTED MASK…"*. The front end is not merely numerically close; it
  transcribes.

On the log-mel `1.7e-04` figure: that is larger than plain float32 rounding
because of the guard itself. Where the mel energy is far below `2⁻²⁴`,
`d(ln(e+g))/de = 1/(e+g) ≈ 1.7e7`, so a 1e-11 absolute difference in a
near-silent bin becomes 1e-4 in the log domain. One int8 LSB is `0.1205 × sd` ≈
0.3 log units, so there is still a factor of ~2,000 of margin. The additive guard
is what keeps this bounded; a clamp would not.

### The gain hook and the refusal

`citrinet_fe_peak_normalize()` peak-normalises the int16 buffer in place before
the STFT, with a caller-supplied ceiling on the gain. Measured on the same
utterance (`firmware/test/results/fe_parity.json` → `guard_cases`):

| condition | input peak | guard occupancy | `citrinet_fe_run()` | log-mel range |
|---|---:|---:|---|---|
| native | 0.4088 | 4.13 % | `OK` | [−16.64, 0.04] |
| −54 dBFS, no gain | 0.0008 | **95.85 %** | **`E_GUARD`, refuses** | [−16.64, −12.37] |
| −54 dBFS, peak-normalised to 0.9 | 0.9000 | 0.00 % | `OK` | [−16.64, 1.68] |

**One utterance does not set a threshold.** Over 80 dev-clean utterances that
fill the 8 s window at correct gain, occupancy is: median **4.5 %**, p90 14.7 %,
p95 22.4 %, **max 33.1 %**. The failure mode this exists to catch sits at
**91–97 %**. The gap between 33 % and 91 % is where the threshold belongs, which
is why it is **0.50** and not the 0.20 first chosen — 0.20 sits inside the speech
distribution and refused 5 of those 80 utterances despite each transcribing
essentially perfectly. A guard that rejects good audio is worse than no guard,
because it teaches you to ignore it.

**Occupancy is computed over non-silent bins only.** `citrinet_fe_run()`
zero-fills the tail of a capture shorter than the window, and every mel energy
over that pad is identically zero — below the guard by construction, and carrying
no information about gain staging. Counting it made the statistic a function of
utterance length: 30 of 60 arbitrary dev-clean utterances were refused, and a
2.77 s utterance reported 77.4 % of which 65.1 points were pure padding.
`citrinet_fe_guard_fraction()` subtracts the exact-zero count from both numerator
and denominator.

−54 dBFS is where `eval/results/gain.log` says ordinary desk speech lands on this
board's IMP34DT05, and where WER goes 5.83 % → 35.28 % while every log reports a
clean NPU run. The front end now refuses that capture instead of transcribing
noise.

Two honest caveats. First, this is a *parity* measurement, not a WER measurement.
The native and peak-normalised rows are bit-exact against the oracle computed on
the same gained int16 — 0 of 64,000 values differ. The −54 dBFS row differs in 23
of 64,000, because with 96 % of the matrix pinned to the floor the per-bin
standard deviation collapses and amplifies float noise; that row is refused
anyway, which is the point. Second, gaining up here
is *after* the MDF's int16 truncation, which `eval/results/gain.log` shows only
recovers WER to 10.45 %. The real fix is `MDF_GAIN` and the `/256` shift in the
acquisition callbacks (`WORKLIST §5.8`); this hook is a safety net.

The refusal is deliberately hard to skip: `citrinet_fe_run()` returns
`CITRINET_FE_E_GUARD` (−4) when occupancy exceeds `CITRINET_FE_GUARD_MAX_FRAC`
(0.50), having still filled the output buffer. A caller that only tests
`!= CITRINET_FE_OK` refuses by default. `citrinet_fe_features_usable()` is the
explicit predicate; `citrinet_fe_report()` formats a one-line level banner with
integer-only `snprintf`, because newlib-nano drops `%f` unless the link line
carries `-u _printf_float` and this string must never be the reason a level
warning fails to appear.

---

## 6. The two rigour questions, answered with numbers

**`WORKLIST §5.3(b)`: "naive sumsq over 800 values in [−32,5] is fine in float32,
but check it against `model/fe_reference.py` rather than assuming."**

Checked. It is not fine, and it is not fatal either:

| variance form | int8 differing / 768,000 | max &#124;Δ&#124; | normalised max &#124;Δ&#124; |
|---|---:|---:|---:|
| two-pass, double (**ships**) | 0 | 0 | 7.7e-05 |
| running sum/sumsq, double | 0 | 0 | 7.7e-05 |
| running sum/sumsq, float32 | 236 (0.031 %) | 1 | 8.7e-03 |

`ss` reaches ~2e5 while `T·mu²` reaches nearly the same value, so the subtraction
cancels most of a 24-bit mantissa. In double it does not matter; in float32 it
costs 236 values, all by exactly 1 LSB. The Cortex-M55 has a double-precision FPU
(`-mfpu=fpv5-d16`) and this is 128,000 accumulate operations, so double costs
essentially nothing — but if a future streaming design wants running accumulators,
row two says they must be `double`, not `float`.

**`WORKLIST §5.3(a)`: "float16 for the scratch is safe after the logarithm."**

Mostly true, and now priced. fp16 halves the pass-1 buffer from 256,000 B to
128,000 B and costs **3,982 of 768,000 int8 values (0.518 %), every one of them by
exactly 1 LSB**, never more. Whether 0.5 % of the feature matrix moving by one
int8 step is acceptable is a WER question this gate did not answer; the code
supports it behind `-DCITRINET_FE_SCRATCH_F16=1`, using a software binary16
conversion so the host measures exactly what the M55 would store.

---

## 7. Memory and flash

### RAM

`sizeof(citrinet_fe_t)` = **5,808 B** on Cortex-M55 (`arm-none-eabi-nm` on a test
link), of which 4,096 B is the FFT work buffer, 1,028 B the 257-bin power
spectrum and 640 B the per-bin `mu`/`sd`. The pass-1 scratch is caller-owned:
`CITRINET_FE_SCRATCH_BYTES` = **256,000 B** float32, or 128,000 B under
`-DCITRINET_FE_SCRATCH_F16=1`.

Stack, `arm-none-eabi-gcc -Os -fstack-usage`:

| function | bytes |
|---|---:|
| `citrinet_fe_run` | 2,088 (a 512-float frame buffer) |
| `citrinet_fe_column` | 88 |
| `citrinet_fe_finish` | 64 |
| everything else | ≤ 72 |

Against the linker region `RAM (xrw) : ORIGIN = 0x34000400, LENGTH = 1023K` =
1,047,552 B (`STM32N657XX_LRUN.ld:49`):

| plan | buffers | total | % of 1023K |
|---|---|---:|---:|
| A — one-shot, whole utterance resident | int16 pcm 256,000 + scratch 256,000 + ctx 5,808 | **517,808** | 49.4 % |
| B — streaming pass 1 (`WORKLIST §5.7`) | ring ~1,344 + scratch 256,000 + ctx 5,808 | **263,152** | 25.1 % |
| C — streaming + fp16 scratch | ring ~1,344 + scratch 128,000 + ctx 5,808 | **135,152** | 12.9 % |

Plan C reproduces `WORKLIST §5.4`'s "~135 KB" estimate; the new information is
that plan B is also comfortable, so the fp16 scratch is a choice rather than a
necessity. **Plan A is the one to start with** — it fits, it is the simplest thing
that can be made to work, and it keeps the capture path out of the first
bring-up. What none of these plans can coexist with is the stock geometry
`WORKLIST §5.4` costs out at 1,026,240 B of `proc_buff` + `audio_out` + ring
buffer; that restructuring is still required.

The 64,000 B int8 output is *not* in this region — it is the NPU runtime's
`p_stai_inputs[0]`.

### Flash

`citrinet_fe.o`, `arm-none-eabi-gcc 14.3.1 -mcpu=cortex-m55 -mthumb -mfpu=fpv5-d16
-mfloat-abi=hard -Os`:

| section | CMSIS FFT | portable FFT |
|---|---:|---:|
| `.text` | 1,812 | 2,180 |
| `.rodata` (the tables) | 3,840 | 3,840 |
| `.rodata.str1.1` | 118 | 118 |
| `.bss` | 52 | 2,052 (twiddles) |

**But the FFT backend is not free, and this is the one budget surprise.** Linking a
minimal whole program (`--gc-sections`, `-ffunction-sections -fdata-sections`)
that calls `citrinet_fe_run` once:

| backend | `.text` | `.data` |
|---|---:|---:|
| CMSIS via `arm_rfft_fast_init_f32(&S, 512)` | 127,284 | 43,784 |
| `citrinet_fe.c`'s portable FFT | **14,084** | **80** |

**158,052 B of that is twiddle and bit-reversal tables.** `arm_cfft_init_f32()` is
one function with a `switch` over every transform length CMSIS supports, so
asking for a 512-point real FFT drags in the tables for all of them.
`--gc-sections` cannot help: they are all reachable. The app slot is
`0x70100000..0x70180000` = 512 KiB and stock is already 238.8 KiB
(`board/GATE3.md`), so 171 KB is not something to discover at Gate 7.

Three ways out, in order of how well they are established:

1. **Build with `-DCITRINET_FE_USE_CMSIS=0`.** 14,084 B, no CMSIS dependency, and
   it is the configuration the harness measures at 1 disagreement in 768,000.
   Costs FFT speed (a straight radix-2 complex 512 rather than a radix-8 real 256)
   — unmeasured on the M55.
2. **`ARM_DSP_CONFIG_TABLES`**, CMSIS's own table-trimming mechanism. A build with
   `-DARM_DSP_CONFIG_TABLES -DARM_FFT_ALLOW_TABLES -DARM_TABLE_TWIDDLECOEF_F32_256
   -DARM_TABLE_BITREVIDX_FLT_256 -DARM_TABLE_TWIDDLECOEF_RFFT_F32_512` links at
   12,172 B — but inspection shows the resulting image contains **no twiddle
   tables at all**, i.e. that macro set is wrong and `arm_rfft_fast_init_f32()`
   would return `ARM_MATH_ARGUMENT_ERROR` at runtime. `citrinet_fe_init()` checks
   that status and fails loudly rather than computing garbage, but **treat 12,172 B
   as unvalidated** until someone finds the correct macro names.
   `arm_cfft_sR_f32_len256` is not an escape either: `arm_const_structs.c:95` puts
   every `arm_cfft_sR_f32_*` behind `#if !defined(ARM_MATH_MVEF)`, so the const
   structs do not exist in a Helium build.
3. Accept the 171 KB and check the signed binary against 0x80000 in CI, which
   `board/GATE3.md` already recommends doing anyway.

---

## 8. What remains unproven until it runs on the M55

1. **Helium.** With the vendor Makefile's exact flags
   (`-mcpu=cortex-m55 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard`) the preprocessor
   defines `__ARM_FEATURE_MVE 3` — verified with `arm-none-eabi-gcc -dM -E`. So
   CMSIS-DSP compiles its **Helium** FFT (`_arm_radix4_butterfly_f32_mve`,
   confirmed present by `nm` on a test link), while the host harness exercises the
   **scalar** path. The two are the same algorithm but not the same summation
   order, so the bit-exact result of §5 does not automatically transfer. Two ways
   to close this: build the FE's CMSIS translation units with
   `-DARM_MATH_AUTOVECTORIZE` (forces the scalar path, `arm_math_types.h:110`), or
   use `-DCITRINET_FE_USE_CMSIS=0`, which removes the question entirely and is the
   configuration the harness proves. **Recommendation: bring up with
   `CITRINET_FE_USE_CMSIS=0`, then switch to CMSIS and re-diff on-device against a
   canned tensor before trusting Helium.**
2. **libm.** `logf`, `sqrt` and `rintf` come from glibc on the host and newlib-nano
   on the target. Both are sub-ulp, but "sub-ulp" is not "identical", and the log
   is applied 64,000 times per utterance.
3. **Timing.** Nobody has measured an audio front end on this part. The work is
   800 × (512-point FFT + 500 MAC + 80 `logf`) plus a second pass of 64,000
   divides and 64,000 `rintf`. Bracket `citrinet_fe_run()` with
   `port_dwt_reset()` / `port_dwt_get_cycles()` (`Projects/Common/cpu_stats.h`) the
   first time it runs, and note that `citrinet_fe_finish()` uses two float
   divisions per output value — if it dominates, reciprocals are the obvious fix,
   at the cost of the last-ulp agreement measured in §5.
4. **The capture path.** Everything here starts from an int16 buffer. The MDF
   decimator, the `/256` truncation, `MDF_GAIN`, and the ring-buffer restructuring
   of `WORKLIST §5.7`/`§5.8` are untouched by this gate.
5. **Guard occupancy on live speech.** §5 measures it on LibriSpeech played
   through the deployment window: 7.6 % at native level, 96 % at −54 dBFS. What
   the DK's own microphone delivers into `citrinet_fe_run()` is exactly the number
   the 50 % threshold exists to expose, and it is unknown.

---

## 9. Files

| path | what |
|---|---|
| `firmware/inc/citrinet_fe.h` | API, geometry, constants, return codes |
| `firmware/src/citrinet_fe.c` | the front end; `CITRINET_FE_USE_CMSIS`, `CITRINET_FE_SCRATCH_F16`, `CITRINET_FE_VAR_MODE` |
| `firmware/inc/citrinet_fe_tables.h` | generated Hann + sparse Slaney mel, 3,840 B |
| `firmware/tools/gen_mel_tables.py` | the generator; `--check` re-derives and diffs |
| `firmware/test/test_fe_host.c` | native driver: WAV in, int8 + both float planes + telemetry out |
| `firmware/test/fe_parity.py` | the comparison against `model/fe.py`, the ST-clamp ablation, the guard cases |
| `firmware/test/run_fe_parity.sh` | builds five configurations and runs them all |
| `firmware/test/results/fe_parity.json` | the measurements quoted above |

`citrinet_fe_tables.h` declares `static const` arrays; include it from exactly one
translation unit, as `citrinet_vocab.h` already requires. `citrinet_fe.c` holds
one file-static FFT instance and is therefore not reentrant — one front end at a
time, which is all a push-to-talk captioner needs.

`citrinet_fe.c` and `test_fe_host.c` compile clean under
`-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`, on both `gcc 11.4.0`
(host) and `arm-none-eabi-gcc 14.3.1 -mcpu=cortex-m55`, in all five build
configurations. The only diagnostics that appear at that level come from CMSIS's
own `dsp/utils.h` and `dsp/none.h`, so the strict build is available whenever
`CITRINET_FE_USE_CMSIS=0`.

---

## 9. First silicon — 2026-08-19

`citrinet_fe.c` has now run on the M55. Image `artifacts/images/gate5_wav/`,
16 canned waveforms replayed from external flash at `0x72000000` through the
front end, then the NPU, in one build. Trace:
[`board/traces/round21_wav_replay.log`](../board/traces/round21_wav_replay.log).

### Cost, which is what this run existed to measure

| stage | cycles | at 600 MHz |
|---|---:|---:|
| **log-mel front end (M55)** | 81,535,764 – 81,744,426 | **135.9 – 136.2 ms** |
| NPU encoder | 84,037,302 – 84,041,070 | 140.1 ms |
| **total per 8 s utterance** | | **~276 ms** |

**The front end costs as much as the whole NPU encoder.** `WORKLIST §5` called the
M55 cost "Unverified"; it is now measured, and it is the dominant half of a
push-to-talk latency budget rather than a rounding error on the NPU's.

The front-end figure varies with `guard_below` (81.54 M at 8,260 bins below the
guard, 81.74 M at 48,374) — `logf` on very small arguments is not constant-time.
Spread across the 16 is 0.26 %.

### An instrument trap, and it cost a run

**`AiDPUProcess()` resets the PMU cycle counter internally.** The first build
reset once, timed the front end, then timed the invoke from the front end's end
stamp — and reported the NPU at 1.7-1.9 M cycles while the two stages summed to a
suspiciously constant 83.45 M. That constant was the runtime's own measurement of
the invoke; the "NPU" figure was `84 M − 81.6 M`. Resetting before each stage and
printing the raw reads (`t0=0 t1=81535764 | n0=0 n1=84038343`) settled it.

Anything timing around `AiDPUProcess()` must call `port_dwt_reset()` itself. The
tell was that the split tracked `guard` while the sum did not move: two
fixed-cost stages cannot trade time.

### Parity against the host: 11 of 16 bit-exact

| | |
|---|---|
| feature tensors byte-identical to the host (FNV-1a 32 over 64,000 B) | **11 / 16** |
| differing | u0, u9, u11, u13, u15 |
| **`guard_below` agreeing exactly** | **16 / 16** |
| front-end return code | `rc 0` on 15; `rc -4` on u2 |

**Every guard count matches**, which bounds the disagreement: the two
implementations are seeing the same levels and the same near-zero mel energies,
so the differences are small numerical ones, not a structural mistake.

**u11 was predicted.** `1272-141231-0012` is the one truncated utterance; the
device sees a 128,000-sample buffer where `gen_corpus.py` used 127,841, so
pre-emphasis at sample 127,841 computes `0 − 0.97·x[127840]`, a value the host
never saw. `wav_ref.json` carries both hashes for it.

**u0, u9, u13 and u15 were not predicted, and are the open question.** On the
host, this configuration (CMSIS `arm_rfft_fast_f32`, float32 scratch) reproduces
`model/fe.py` exactly — 0 of 768,000 values over 12 utterances. Something the
M55 does differs on 4 of 16 here. Candidates, none yet tested:
CMSIS-DSP's `arm_rfft_fast_f32` taking a different code path on Helium than on
x86; `rintf` rounding mode; FMA contraction changing the mel dot products; the
`logf` implementation differing between newlib-nano and glibc.

**u2 reporting `rc -4` is correct, not a fault.** `5694-64025-0000` is 1.67 s of
speech in an 8 s window, so 62.5 % of its mel bins fall below the log guard
against a 50 % threshold derived from utterances that fill the window. It is one
of the loudest at −4.38 dBFS. The threshold, not the audio, is what needs
revisiting for short utterances.

### Next

Localise the four. `firmware/test/score_wav.py` already implements the host side
of a per-mel-row FNV-1a (80 rows of 800 bytes); a rebuild that prints 80 row
hashes turns "4 utterances differ" into "these mel bins, these frames" without
further host work.

---

## 10. First light on the microphone — 2026-08-19

`board/traces/round22_mic_firstlight.log`. `-DGATE5_MIC`, on-board MP23DB01HP
through MDF1 filter 0, 16 kHz mono, 8 s per utterance straight into AXISRAM3.
One speaker, one sentence — *"the birch canoe slid on the smooth planks"* —
eleven times, then fifteen utterances of an empty room.

**It transcribes.** Best two utterances, both raw, one word error each:

```
u7  "the birch cano slid on the smooth planks"      12.5 %
u9  "the birch canoe slid on the smooth blanks"     12.5 %
```

Over the eleven spoken utterances: **29.5 % WER raw, 34.1 % after
`citrinet_fe_peak_normalize()`** (26 and 30 errors over 88 reference words).

### The −54 dBFS assumption is wrong by about 50 dB

`docs/FEASIBILITY.md` §2(d) assumed the DK's microphone delivers −54 dBFS and
built a 35 % WER prediction on it; `firmware/AUDIO-INPUT.md` §5 called that the
silent failure mode of the on-board-mic option, and `WORKLIST.md` §5.8 planned
two knobs to fix it. At the stock `MDF_GAIN(16000) = 2` this board delivered a
**−3.8 dBFS peak, −23.5 dBFS RMS, 0 clipped samples**. There was no gain deficit
to correct.

### But the loudest capture was not the best one

Sorted by guard occupancy rather than by level, over the eleven spoken
utterances, raw pass:

| guard occupancy | utterances | mean word errors |
|---|---|---:|
| 0–25 % | u0, u1 | 3.5 |
| 26–50 % | u4, u7, u8, u9, u10 | **1.8** |
| 51–70 % | u2, u3, u5, u6 | 2.5 |

The evaluation corpus's own median guard occupancy is **35.6 %**
(`artifacts/corpus/wav_ref.json`, `host_guard_below` median 22,788 of 64,000).
The two best utterances sat at 34 % and 50 %; the two worst sat at 0 % and 20 %,
and those were the *loudest*.

The mechanism is distribution match, not headroom. LibriSpeech carries near-silent
pauses whose mel energies fall below the 2⁻²⁴ guard; a hot close-mic capture has
a noise floor that never does. Per-feature normalisation then computes its mean
and standard deviation from a different distribution than the one the model was
trained on. **So the controller should target guard occupancy, not a level** —
and guard occupancy is something the front end already computes for free.

**This is n = 1 per condition**: one speaker, one sentence, one room, eleven
utterances. It is a trend with a mechanism behind it, not a calibrated number.
What it is strong enough to do is redirect the controller, because the previous
target had no evidence behind it at all.

### What the run validated on the way

- **The guard instrument works on live audio.** `citrinet_fe_run()` returned
  `CITRINET_FE_E_GUARD` (−4) at 66 %, 57 %, 69 %, 56 %, 50 % and 90 %, and `OK`
  everywhere below `CITRINET_FE_GUARD_MAX_FRAC`. This is the telemetry
  `WORKLIST.md` §5.9 required from the first commit, now exercised.
- **`peak_normalize` behaves exactly as `FEASIBILITY.md` §2(d) predicts.** At u2
  (peak −25.7 dBFS, guard 66 %) it lifted ×9.65, cut guard to 3 % and halved the
  errors. It never beat a correctly-gained capture. Digital gain applied *after*
  the int16 truncation recovers some of the loss, not all of it.
- **The capture path is sound.** 128,000 samples arrive every time; `# <<<
  captured 128000 samples` on all 26 utterances, no ring buffer, no `malloc`.
- **The front end costs the same as in replay**: 83.2–83.9 M cycles, against
  81.5–81.7 M for the same code fed from flash in round 21. The 2 % is code
  layout, not the microphone.

### An AGC needs a speech gate

Utterances 11–25 were an empty room. A guard-seeking loop with no speech
detector winds the gain to a rail chasing a target silence cannot reach. The CTC
decoder already answers the question: **no tokens decoded, no adaptation.**


### Round 21's four disagreeing utterances, localised

The same image printed 80 per-mel-row FNV-1a hashes during its replay pass.
Against the host oracle (`board/traces/round21_wav_replay.log.host_row_hashes.json`):

| utterance | rows differing of 80 | which |
|---|---:|---|
| u7, u8, u10, u12, u14 | 0 | — |
| u9 `7850-281318-0006` | 2 | mel bins 2, 26 |
| u13 `84-121123-0006` | 1 | mel bin 0 |
| u15 `6345-93302-0017` | 2 | mel bins 2, 74 |
| u11 `1272-141231-0012` | 56 | the truncated utterance, predicted |

**One or two isolated, non-adjacent mel bins.** That eliminates every structural
candidate named in §9. A differing FFT would move all 80 rows, because every mel
bin is a dot product over shared FFT bins; so would a differing pre-emphasis or a
differing input buffer. A differing mel filterbank table would move a *contiguous*
run of bins, since the filters overlap. What remains is a single int8 value
crossing a rounding boundary — and bins 0 and 2 are the narrowest filters, with
the fewest taps and therefore the least averaging, which is where a 1-LSB
difference is most likely to survive to the output.

u11 is the truncated utterance and was predicted in §9: the device sees 128,000
samples where the host used 127,841, so pre-emphasis at sample 127,841 sees a
value the host never did. That it moves 56 rows rather than all 80 is
per-mel-bin normalisation spreading one frame's difference unevenly.

The next build prints each row's signed sum and sum of absolute values alongside
its hash. Sum differing by ±1 *and* sumabs by ±1 means exactly one value moved by
one LSB — which turns the paragraph above from an argument into a measurement.
Signed sum alone would not do it: two values moving +1 and −1 leave it unchanged.

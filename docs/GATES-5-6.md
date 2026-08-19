# Gates 5 & 6 — the C front end and the C decoder

Status: **both host-complete, neither has executed on the M55.** One adversarial
finding is a **must-fix before integration** (§2.1). Board untouched throughout.

> **Update, 2026-08-19.** Gate 6 is **closed** on that evidence
> (`firmware/test/results/gate6_ctc.json`: 0 text disagreements over 9,226
> characters). Gate 5 is **not** — everything below still stands, and §4's
> integration list is now actionable because **Gate 4 has reported**: the graph
> executes on the NPU in 124.035 ms and the device is not measurably less
> accurate than the host over 64 utterances (`board/GATE4.md` Rounds 19-20).
> Two things in §4 changed as a result. §4.3's runtime scale is confirmed on
> silicon — the board reads back `scale=8.297212 = 1 / 0.120522417128086`. And
> §4.8's "512 KiB app slot" is **no longer the geometry**: the Citrinet weight
> blob moved to `0x70400000` (`compile/stt_audio.mpool`) precisely because the
> app does not fit 512 KiB, so the CI assertion is against 3 MB, with
> `board/flash_and_verify.sh:40-48` as the enforcing check.

Deliverables: `firmware/src/citrinet_fe.c` + `firmware/inc/citrinet_fe.h` +
`firmware/inc/citrinet_fe_tables.h`; `firmware/src/citrinet_ctc.c` +
`firmware/inc/citrinet_ctc.h` + `firmware/inc/citrinet_vocab.h`.
Generators: `firmware/tools/gen_mel_tables.py`, `firmware/tools/gen_tokenizer.py`.
Detail: `firmware/FRONTEND.md`. Raw records: `firmware/test/results/fe_parity.json`,
`firmware/test/results/gate6_ctc.json`.

---

## 1. What is built and proven on the host

### 1.1 Front end — bit-exact against `model/fe.py`

```
source /home/claroche/stm32n6-deployment-zoo/.venv/bin/activate
bash firmware/test/run_fe_parity.sh 12
```

| build | int8 disagreements | max \|d\| |
|---|---:|---:|
| `fe_cmsis` (arm_rfft_fast_f32, f32 scratch) — the shipping config | **0 / 768,000** | 0 |
| `fe_portable` (built-in radix-2 FFT) | 1 / 768,000 | 1 |
| `fe_cmsis_f16` (`-DCITRINET_FE_SCRATCH_F16=1`) | 3,982 / 768,000 (0.518 %) | 1, never more |
| `fe_var_sumsq_f64` (`-DCITRINET_FE_VAR_MODE=1`) | 0 / 768,000 | 0 |
| `fe_var_sumsq_f32` (`-DCITRINET_FE_VAR_MODE=2`) | 236 / 768,000 (0.031 %) | 1 |

Log-mel plane max \|Δ\| 1.688e-04; normalised max \|Δ\| 7.668e-05. The C binary dumps
the exact int16 buffer it consumed and the oracle re-runs on that dump, so both
sides provably see identical input. Deterministic across runs; identical at `-O0`
and `-O2`.

Answers to `WORKLIST §5.3`: **(a)** fp16 scratch buys 256,000 → 128,000 B for
3,982 one-LSB flips — a choice, not a necessity. **(b)** naive sum/sumsq `ddof=1`
is safe in double, not in float32; the shipped code uses the two-pass form in
double, which is what NumPy's `std()` does and which the M55's `fpv5-d16` FPU
executes natively.

### 1.2 The log guard, quantified

`ln(x + 2^-24)`, additive, never a clamp (`citrinet_fe.c:276` is the only `log` in
the file; no `FLT_MIN`/`fmaxf` anywhere). Swapping in ST's clamp-to-`FLT_MIN`
and changing nothing else moves **737,335 of 768,000 int8 values (96.01 %),
max \|Δ\| 46 LSB**, floor −16.64 → −87.34
(`firmware/test/fe_parity.py:st_library_ablation()`). `2^-24` is also fp16's
smallest subnormal, so ST's f16 path takes `logf(0) = -inf` on 3,098–37,721 of
64,000 bins per utterance at native level. This is why the ST library is not
used, and it is the single reason the gate exists.

### 1.3 Tables

Sparse mel filterbank: 500 non-zero of 20,560 (2.432 %), contiguous runs 2–18,
max FFT bin touched 255 (Nyquist bin 256 provably never read, which is what makes
CMSIS's packing of bin 256 into `sp[1]` harmless). Tables cost **3,840 B**
`.rodata` (2,240 sparse mel + 1,600 Hann) versus 82,240 B dense — 36.7× smaller,
41.1× fewer MACs/frame. `arm-none-eabi-size -A citrinet_fe.o` reads `.rodata`
exactly 3840. Regenerate/verify with `python firmware/tools/gen_mel_tables.py --check`.

### 1.4 Decoder — exact against `model/fe.py:greedy()`

```
python firmware/test/run_gate6.py 100      ->  GATE 6 DECODER: PASS
```

100 calibration-disjoint dev-clean utterances (seed 20260816, drawn from the
2,545 records left after excluding all 158 calibration keys): **0 text
disagreements over 9,226 characters, 0 per-frame argmax disagreements over
10,000 frames**. Plus 480 synthetic `[100,1025]` int8 matrices — 39,972 of 48,000
frames carrying a tied argmax, 13,218 collapsed-repeat frames, 4,841 blank frames
— also 0/0.

The int8 fed to C is the graph's own quantised tensor
(`/Transpose_output_0_QuantizeLinear_Output`), not a re-quantisation;
`round(dequant/0.265415638685226)` reproduces it element-for-element, which is
the identity that makes argmax-on-int8 and argmax-on-float the same operation
(`fp32_vs_int8_text_differences: 0`).

Semantics matched deliberately: `prev` updates on every frame including blanks;
ties resolve to the lowest class id (strict `>` = `numpy.argmax`); `.strip()`
reproduced by trimming `' '`, exact because the piece alphabet is audited to be
`[a-z'<>]` plus U+2581 with no control characters.

### 1.5 Cost and hygiene

| unit | flash |
|---|---:|
| `citrinet_fe.o` (portable FFT) | 2,180 `.text` + 3,840 `.rodata` + 118 str + 2,052 `.bss` |
| `citrinet_fe.o` (CMSIS) | 1,812 `.text` + 3,840 `.rodata` + 118 str + 52 `.bss` |
| Gate 6 total | **8,757 B** (code 534 + `kPieces` 6,170 + `kOffset` 2,052 + 1 rodata), `.data` = `.bss` = 0 |

The decoder's only include is `<stdint.h>`; `arm-none-eabi-nm -u ctc.o` prints
nothing, so it links against neither libc nor the HAL.

RAM: `sizeof(citrinet_fe_t)` = 5,808 B, scratch 256,000 B (128,000 fp16), peak
stack 2,088 B in `citrinet_fe_run`. Against `RAM … LENGTH = 1023K` = 1,047,552 B
(`STM32N657XX_LRUN.ld:49`): plan A (one-shot) 517,808 B = 49.4 %; plan B
(streaming) 263,152 B = 25.1 %; plan C (streaming + fp16) 135,152 B = 12.9 %.

**Flash surprise:** `arm_rfft_fast_init_f32(&S,512)` drags 158,052 B of twiddle
and bit-reversal tables for every transform size CMSIS supports
(`arm_cfft_init_f32()` is one `switch`, so `--gc-sections` cannot help). A
minimal whole-program link is 127,284 B `.text` + 43,784 B `.data` with CMSIS
versus **14,084 B + 80 B** with the built-in portable FFT. The app slot is 512 KiB
and stock is already 238.8 KiB.

Both units compile clean under `-Wall -Wextra -Wpedantic -Wconversion -Wshadow
-Werror` on gcc 11.4.0 and arm-none-eabi-gcc 14.3.1 `-mcpu=cortex-m55`, in all
five FE configurations.

### 1.6 End to end

The C front end's own int8 tensor, pushed through `artifacts/onnx/q800_real.onnx`
and greedy-decoded, reproduces the LibriSpeech reference truncated at the 8 s
window — verified on 5 utterances by the Gate 5 harness and independently on 6
more by the verifier, where C/CMSIS, C/portable, Python-quantised and Python-float
all gave character-identical text 6 of 6.

---

## 2. What adversarial verification corrected

Overall verdict: **SOUND_WITH_FIXES**. Checks 1–6 (int8 parity on the verifier's
own 14-utterance selection at `-O0` and `-O2`; additive-guard-not-clamp; `ddof=1`
including at forced `T=50` and `T=12`; symmetric Hann; librosa slaney filterbank;
end-to-end) all **held**, re-derived with independently written harnesses. One
unasked-for claim was **REFUTED**.

### 2.1 REFUTED — `CITRINET_FE_GUARD_MAX_FRAC = 0.20` is a false-positive generator (MUST FIX)

`citrinet_fe.h:69` is still `0.20f` as of this writing. Measured over 80 dev-clean
utterances that **fill** the 8 s window (≥ 8.06 s, zero padding contributes
nothing):

| statistic | guard fraction |
|---|---:|
| median | 0.0450 |
| p90 | 0.1470 |
| p95 | 0.2235 |
| max | 0.3311 |

**5 of 80 (6.2 %) are refused with `CITRINET_FE_E_GUARD` despite being correctly
gained and transcribing essentially perfectly.** Worst case `2086-149220-0041`:
input peak 0.5637 full scale, guard 0.2224, rc −4, and its C features decode to
"she was indistinctly aware however that the gaunt figure of the old gentlewoman
was sitting in one of the straight back chairs a little withdraw". Two further
refused utterances (`1993-147149-0030` at 0.3311, `1993-147149-0001` at 0.2721)
also transcribe correctly. Root cause: `guard_below` in clean LibriSpeech is
really "fraction of inter-word silence" — the per-bin occupancy for
`2086-149220-0041` is a flat ~0.20 across all 80 mel bins, not a high-frequency
band artefact.

**Fix:** re-derive the threshold from the measured native-level distribution.
0.45–0.60 separates the −48 dBFS failure mode (91–97 % guard) from normal speech
(max observed 33 %) with margin. 0.20 sits inside the speech distribution.

### 2.2 MUST FIX — `guard_below` counts the zero pad

`citrinet_fe_run()` zero-fills the tail of a short capture; every such mel energy
is exactly 0, hence < 2^-24, hence counted. Over 60 arbitrary dev-clean
utterances placed in the 8 s window, **30 of 60 (50 %) are refused**. The context
already carries the fix: `guard_zero` holds the pure-zero count (e.g.
`2086-149220-0043`, 2.77 s: `guard_below` 77.4 %, of which `guard_zero` 65.1 % is
padding). Either exclude columns beyond `n_samples` from `guard_total`, or
evaluate `(guard_below − guard_zero) / guard_total`. Untouched, the Gate-4
canned-feature path and any file-driven bring-up test refuse half their inputs.

### 2.3 Documentation corrections the verifier won

- The Gate 5 report and `firmware/FRONTEND.md` state "native 7.61 % guard
  occupancy, usable=1" as if it characterised native level. **It is one
  utterance.** Replace with the distribution in §2.1 (n=80, full-window) so the
  threshold choice is auditable.
- The single portable-FFT disagreement does **not** "sit on a rounding tie". On
  the verifier's utterance (`2086-149220-0002`, cell `[79,481]`) the oracle value
  is −1.5000737 and the portable value −1.4999163 — 7.4e-05 LSB **past** the
  boundary, and 0 of 64,000 cells in that utterance are within 1e-6 of a
  boundary. Correct wording: "within 1e-4 LSB of a boundary", which the harness
  already measures. The 1-LSB magnitude and the `USE_CMSIS=0` recommendation are
  unaffected.
- **New limit on the parity claim:** C and `model/fe.py` stop agreeing below
  roughly −48 dBFS (9 of 64,000 int8 at −48 dB, 1,591 at −72 dB, max \|Δ\| 2). Not a
  front-end defect — the log-mel plane still agrees to 1.9e-06, but per-bin std
  collapses to 2.2e-05 where the 1e-5 eps is 45 % of it, so 1/sd amplifies a
  1e-6 difference by ~4.5e4. Every such case is already refused. State the claim
  as **"bit-exact at every level the guard accepts"**, not unconditionally.

---

## 3. What is still unproven until it runs on the M55

Settled on host: numerical parity, table correctness, decoder semantics, flash
and RAM footprints, warning-clean cross-compilation. Everything below is
**UNVERIFIED**.

1. **Helium.** With the vendor Makefile's flags (`-mcpu=cortex-m55 -mthumb
   -mfpu=fpv5-d16 -mfloat-abi=hard`) the preprocessor defines `__ARM_FEATURE_MVE 3`
   (checked with `-dM -E`), so CMSIS-DSP compiles its Helium FFT
   (`_arm_radix4_butterfly_f32_mve` confirmed by `nm` on a test link) while the
   host harness exercises the **scalar** path. Same algorithm, different summation
   order — the 0/768,000 result does **not** automatically transfer.
   `-DARM_MATH_AUTOVECTORIZE` forces scalar; `-DCITRINET_FE_USE_CMSIS=0` removes
   the question.
2. **CMSIS table trimming — UNVALIDATED.** `-DARM_DSP_CONFIG_TABLES
   -DARM_FFT_ALLOW_TABLES -DARM_TABLE_TWIDDLECOEF_F32_256
   -DARM_TABLE_BITREVIDX_FLT_256 -DARM_TABLE_TWIDDLECOEF_RFFT_F32_512` links at
   12,172 B, but `nm` shows **no twiddle tables in the image at all** — the macro
   names are wrong and `arm_rfft_fast_init_f32()` would return
   `ARM_MATH_ARGUMENT_ERROR`. `citrinet_fe_init()` checks that status and fails
   loudly, so this is safe, not silent. `arm_cfft_sR_f32_len256` is not an escape:
   `arm_const_structs.c:95` hides every `arm_cfft_sR_f32_*` behind
   `#if !defined(ARM_MATH_MVEF)`.
3. ~~**Cycle cost — UNMEASURED.**~~ **MEASURED, 2026-08-19: 136.0 ms**
   (81.5–81.7 M cycles at 600 MHz), against 140.1 ms for the whole NPU encoder.
   The front end is half the latency budget. The expectation that the decoder is
   negligible held. One instrument trap cost a run and is worth knowing:
   `AiDPUProcess()` **resets the cycle counter internally**, so a stamp taken
   before it without its own reset reports the runtime's measurement, not yours —
   the tell was two fixed-cost stages summing to a constant while their split
   tracked the guard count. `firmware/FRONTEND.md` §9.
4. ~~**libm.**~~ **Diffed on device.** Over 15 non-truncated utterances,
   960,000 int8 values, host and target disagree on **six — each by exactly one
   LSB**, localised to single mel bins by per-row sums. Any libm or FFT
   difference large enough to matter would have moved all 80 rows of a tensor,
   since every mel bin is a dot product over shared FFT bins. `logf`, `rintf`,
   `arm_rfft_fast_f32` on Helium and FMA contraction are all ruled out as
   *material* differences. `firmware/FRONTEND.md` §11.
5. ~~**The capture path — NOT TOUCHED.**~~ **Built and run.** `gate5_mic()`
   keeps `BSP_AUDIO_IN_Record` and the two BSP transfer callbacks and **drops the
   ring buffer entirely** — the utterance design never needed it, which is what
   makes the 1,026,240 B (really 2,679,312 B) problem disappear rather than get
   solved. The callbacks append 160 samples at a time straight into AXISRAM3.
   Note the live `/256` is at `:3234,3259`, not `:3172,3197` — the lines named
   here are inside `#if (USE_HAL_MDF_REGISTER_CALLBACKS == 1)`, which
   `stm32n6xx_hal_conf.h:191` sets to 0 (`firmware/AUDIO-INPUT.md` §7).
6. ~~**Guard occupancy on the DK's own microphone.**~~ **0.1 % at the stock
   gain** — 73 of 64,000 bins, at a −3.8 dBFS peak. Not 96 %, and not 4.5 %
   either: a live capture's noise floor rarely dips below 2⁻²⁴ at all, where
   LibriSpeech's near-silent pauses do. That difference is why the deployed AGC
   targets guard occupancy: it is the one number that measures how far the input
   distribution has drifted from the one per-feature normalisation was calibrated
   on. The threshold did exactly what it exists for — `citrinet_fe_run()` returned
   `CITRINET_FE_E_GUARD` at 66–90 % and `OK` below.
7. **The U+2581 → space branch in the decoder is dead code on the shipped
   header** (no piece contains a 0xE2 byte), so it is reasoned about, not measured.

---

## 4. Integration, once Gate 4 reports

**Gate 4 has reported** (`board/GATE4.md`), so this list is live. Do §4.0 before
anything else.

**4.0 — apply the two guard fixes (§2.1, §2.2).** Change
`CITRINET_FE_GUARD_MAX_FRAC` from `0.20f` to a value re-derived from the n=80
distribution (0.45–0.60), and evaluate the fraction over
`guard_below − guard_zero` (or exclude padded columns from `guard_total`).
Re-run `bash firmware/test/run_fe_parity.sh 12` and confirm the 80-utterance
refusal rate goes to 0. Without this, half of every file-driven bring-up test is
refused.

**4.1 — build config for first silicon.** `-DCITRINET_FE_USE_CMSIS=0`. It dodges
both open CMSIS questions (Helium summation order, 158 KB of tables) and it is
the configuration the harness proves at 1/768,000. Add the two `.c` files to the
vendor Makefile through the `EXTRA_CFLAGS` extension point that
`firmware/apply_vendor_mods.sh` step 4 installs — never set `OPT=` on the command
line, `make` discards every in-file `OPT +=` including `-flax-vector-conversions`.

**4.2 — memory plan A.** int16 PCM 256,000 + scratch 256,000 + ctx 5,808 =
517,808 B, 49.4 % of the 1023K region. This requires the stock
`proc_buff`/`audio_out`/ring geometry to be gone first (§3.5); plan A is the
simplest thing that fits and keeps the capture path out of first bring-up.

**4.3 — wire the tensor.** `citrinet_fe_run()` writes 64,000 B of
`{80,800,1}` CHANNEL_FIRST int8 directly into the NPU's `p_stai_inputs[0]` — not
into the 1023K region. Read `quant_scale` at runtime from
`stai_network_get_info()->inputs[0].scale.data[0]` (0.120522417128086 today);
`quant_zero_point` is 0. Do not hardcode.

**4.4 — replace Gate 4's canned features with the live ones.** Keep
`firmware/inc/canned_features.h` in the build behind a flag: it is the only way
to bisect a front-end regression from an inference regression on-device.

**4.5 — decode.** `citrinet_ctc_argmax()` takes no vocabulary and no text buffer
and is exactly what Gate 4 needs to diff on-device token ids against host ORT.
`citrinet_ctc_decode()` gives text; `CITRINET_CTC_TEXT_CAP` = 1,201 B; overflow
returns `E_TRUNC` with a valid NUL-terminated prefix. **`citrinet_vocab.h` must be
included from exactly one TU** (its tables are file-static) — `citrinet_ctc.c` is
that TU; a second includer silently duplicates 8,222 B of `.rodata`. LCD code
should call `citrinet_ctc_piece()`, not include the header.

**4.6 — first-run instrumentation.** Bracket `citrinet_fe_run()` with
`port_dwt_reset()` / `port_dwt_get_cycles()` (`Projects/Common/cpu_stats.h`).
Log `guard_below`, `guard_zero`, `input_peak`, `logmel_min/max` and `clipped`
from `citrinet_fe_t` on every run — §3.6 is answered by exactly these counters.

**4.7 — on-device re-diff before trusting anything faster.** Feed the canned
tensor's source audio through the on-device FE and diff the int8 against the host
result. Only then consider switching to CMSIS/Helium (§3.1) or reciprocal
division in `finish()` (§3.3).

**4.8 — CI check.** Assert the signed binary against 0x80000 (512 KiB app slot);
stock is already 238.8 KiB.

Housekeeping: `firmware/WORKLIST.md` items 5.3, 5.5, 5.6, 5.9 are done and 5.4's
budget is measured; that table has not been reconciled. `firmware/test/Makefile`
carries a generic name and was created by the Gate 6 track (targets `selftest`,
`oracle`, `arm`, `vocab`) — merge rather than overwrite if Gate 5 wants its own.

---

## 5. Zoo contributions — ready to apply, nothing applied

`/home/claroche/stm32n6-deployment-zoo` was **not modified**
(`git -C … status --porcelain` returns 0 lines). Everything is staged under
`/home/claroche/stm32n6-stt/zoo-contrib/`; all of it was validated against a
scratch copy using the zoo's own loaders.

| file | what it is | human action |
|---|---|---|
| `zoo/graph/budget.patch` | fixes `budget.py` double-counting weights | `git apply` (checks clean); `pytest tests/ -q` = 146 passed before and after |
| `tests/test_budget_weight_dq.py` | regression test for the above | copy in with the patch — it **fails** on the unpatched tree (73,216 ≠ 66,304) and passes after |
| `models/audio/citrinet-256-gamma025.toml` | the recipe | `zoo.recipe.load()` validates: `is_compilable` True, unpinned `[]`, anonymous_axes `[]`, 4 patches. **Not yet reproducible — see blockers below** |
| `known_issues.additions.toml` | 7 fault-atlas entries, incl. a **correction** to the existing `ll-aton-middleware-version-mismatch` | merge; merged copy loads 89 unique ids, all 4 signatures route correctly, no new duplicates |
| `zoo/quant/log_mel_nemo.py` | calibration preprocessor | drop in; verified element-wise against `model/fe.py` (`np.array_equal` True, max \|diff\| 0.0) |
| `budget-bug.md`, `policy-corrections.md`, `README.md` | the write-ups | read before applying |

The budget bug in one line: `peak_activation_fused` = 11,250,052 B on
`q800_real.onnx` where the compiler allocates 640,000 B, and `placement()` says
activations-in-PSRAM for a graph that is entirely on-chip. **87.3 % of the peak is
weights counted twice** — 405 live weight-`DequantizeLinear` outputs contributing
9,816,452 B at the peak node. Excluding them gives 1,433,600 B.

Note the `ll_aton` entry already in the atlas is **wrong**, not missing: it calls
the mismatch a warning and prescribes "same major/minor (4.0.x was accepted)",
but `artifacts/model_c/network.c:56` guards with `!=` on all four components
including DEV, and 1.1.3-262 vs 1.1.3-275 is a hard `#error`. Hence a correction
rather than an addition.

### Blockers the human must resolve before the recipe reproduces its artifact

1. **The four named graph patches do not exist as zoo patches**
   (`drop_dead_where_mask`, `reducesum_div_to_reducemean`, `se_matmul_to_conv1x1`,
   `drop_trailing_logsoftmax`). `model/clean.py` is the reference implementation,
   one numbered block each. Without them the graph does not compile at all.
2. **`qdq.py` hardcodes `ActivationSymmetric=False`** (`zoo/quant/qdq.py:386`) and
   no policy key controls it. Correction 4 in `policy-corrections.md` proposes the
   key. All 1,419 Q/DQ zero-point tensors in `q800_real.onnx` are exactly 0, which
   is what the "offset 0" contract and the offset-free `q = round(x/scale)` in the
   front end depend on.
3. **`[postconditions]` in the recipe is inert** until `zoo/recipe.py` learns to
   load it and the compile stage compares against `CompileInfo.metrics()`. The
   README sketches the change; it is not written.
4. **`notes = "lab/citrinet-256-gamma025.md"` points at a file that does not
   exist** in the zoo. The prose lives in `docs/FEASIBILITY.md`, `compile/GATE2.md`
   and `board/GATE3.md`; condensing it was out of scope.
5. `budget-bug.md` flags but does not resolve that after the patch,
   `peak_activation` is still 2.2× **above** the compiler's allocation at T=800
   (1,433,600 vs 640,000), so the module's "lower bound" framing is wrong in both
   directions. Called out as out of scope.

Two atlas entries carry provenance caveats and are marked as such in the TOML:
`ai-dpu-model-check-rejects-int8-rank3-output`'s signature is transcribed from the
source literal and log-level path, **not read off a UART** (the check was patched
before the model was flashed); and the app-slot-overflow entry's
corrupted-weights consequence is **reasoning**, not an observed run.

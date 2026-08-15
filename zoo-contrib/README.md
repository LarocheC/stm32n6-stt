# `zoo-contrib/` — what this project owes the deployment zoo

`stm32n6-stt` took one model from a Hugging Face id to a signed image running
on an STM32N6570-DK. Roughly half of what that cost was **not** about Citrinet:
it was about the compiler, the memory map, ST's Makefile and ST's signing tool,
and it will be paid again by the next model unless it goes back into
[`stm32n6-deployment-zoo`](https://github.com/LarocheC/stm32n6-deployment-zoo),
whose stated primary product is the failure atlas.

Everything here is a **ready-to-apply file for that repo**, prepared outside it.
The zoo working tree was never modified — the one patch below was verified with
`git apply --check` and against a scratch copy of `zoo/`, `tests/` and
`config/`.

The dividing line: a fact about *this model* stays in this repo; a fact about
*this part, this toolchain or this board* belongs in the zoo, because the zoo is
where the next model will look for it. Every claim below is either cited to a
file in this repo or was re-measured while writing the contribution; where
something is unverified it says so.

---

## The files, in the order to apply them

| # | file | goes to | why it belongs there |
|---|---|---|---|
| 1 | `known_issues.additions.toml` | `zoo/faults/known_issues.toml` | seven new constraints + two corrections to existing entries |
| 2 | `policy-corrections.md` | `config/policy.toml` (comments; one new key) | two budget constants get their provenance; one policy/code divergence |
| 3 | `zoo/graph/budget.patch` + `tests/test_budget_weight_dq.py` | `zoo/graph/budget.py`, `tests/test_budget.py` | a reproduced accounting bug that produces a wrong placement verdict |
| 4 | `zoo/quant/log_mel_nemo.py` | `zoo/quant/calib.py` | the calibration front end recipe #5 names |
| 5 | `models/audio/citrinet-256-gamma025.toml` | `models/audio/` | the recipe, which depends on 3 and 4 |

The order matters in one place only: the recipe (5) names a preprocessor that
does not exist until (4) is applied, and its `[postconditions]` are only
meaningful once the budget stage stops reporting an 11 MB activation peak (3).
1 and 2 are independent and can go in on their own.

---

### 1. `known_issues.additions.toml` — the atlas entries

Seven new `[[issue]]` blocks and two `[[amend]]` blocks. The additions file
carries its own placement instructions: each block names the section of
`known_issues.toml` it belongs in, because that file is ordered by what a
reader can *do* about an entry rather than by topic.

What is new:

| id | silent | signature |
|---|:--:|:--:|
| `psram-spill-with-zero-sw-epochs` | yes | — |
| `signing-without-align-unbootable-image` | **yes** | — |
| `st-makefile-passes-fcyclomatic-complexity` | no | yes |
| `make-command-line-opt-override-discards-appends` | no | yes |
| `ai-dpu-model-check-rejects-int8-rank3-output` | no | yes |
| `app-slot-512k-overflow-into-weights` | yes | — |
| `hotplug-fails-after-first-boot-from-flash` | no | deliberately none |

Two entries in the file are **corrections**, delivered as `[[amend]]` because
the loader reads only `issue` — so a blind `cat >>` of this file is safe but
incomplete, adding the seven and silently ignoring the two. Apply those by hand:

- **`ll-aton-middleware-version-mismatch` is wrong as written.** It calls the
  failure a warning (it is a `#error`, and the build stops) and prescribes
  "keep generator and app on the same major/minor (4.0.x was accepted here)".
  The guard is `!=` on all four version components including the dev number:
  the mismatch here was 1.1.3-**262** against 1.1.3-**275** — same major, minor
  *and* micro, and still fatal. Anyone following the current remedy would
  confirm 1.1.3 == 1.1.3 and go looking somewhere else.
- **`no-stm32-target-found-on-swd`** keeps its id, class and signature; only
  its `workaround` grows, because this project reproduced the same error text
  from a different cause (boot configuration latched at reset) whose remedy is
  `mode=UR` rather than the RESET button. `classify()` routes the text there,
  so that is where the second remedy has to live — which is also why the new
  `mode=HOTPLUG` entry carries no signature of its own: two entries matching
  the same text would make which remedy you get an accident of string length.

Verified with the zoo's own loader and classifier (against a merged copy, not
the repo): the additions parse, all 89 ids are unique, every new entry carries
every field `KnownIssue` reads, each of the four signatures routes to exactly
the intended entry, and no *new* duplicate signature is introduced.

Counts move 82 → 89 constraints, 30 → 33 with a verbatim signature, 32 → 35
silent. All three are quoted in the zoo's `README.md` under "The failure atlas".

The one to read first is `signing-without-align-unbootable-image`. It is the
most expensive silent failure either project has recorded — a completely dead
board, at any baud, in both boot-switch positions, with flash that reads back
byte-identical to the signed binary — and the zoo *already had the fact*, in
`config/toolchain.toml` ("2.21 introduced the mandatory `-align` on signing").
It simply was not connected to a symptom that looks like dead hardware. That is
the argument for keeping the atlas keyed on symptoms.

### 2. `policy-corrections.md` — provenance for two constants, one real divergence

1. **`onchip_bytes = 2883576`** vs the 2,883,584 the mpool arithmetic gives. The
   value is right and stays; what was missing is that the compiler emits *both*
   numbers for the same pool in the same run — `network.c`'s pool table says
   `size=2883576` (a per-pool 8-byte reservation, `alignment: 8`) while
   `network_c_info.json` says `size_bytes: 2883584`. `zoo/st/cinfo.py` parses
   the latter, so a check comparing a parsed pool size to this constant is off
   by eight.
2. **`audio_app_onchip_bytes = 1507328`** now has a file behind it: ST's own
   `stm32n6.mpool` declares exactly two non-zero internal pools, cpuRAM2
   (1024 KB) and npuRAM6 (448 KB). Plus the npuRAM3/4/5 finding — the audio app
   powers and clocks those three banks and its linker script claims none of
   them, so 1,376,256 B sit idle and the 1.47 MB figure is ST's conservative
   default rather than a hardware ceiling.
3. **`[quantize]` documents symmetric activations and `qdq.py` hardcodes
   asymmetric.** This one is a genuine divergence, and it blocks the recipe:
   see below.

### 3. `zoo/graph/budget.patch` — the bug reproduces

`docs/GATES-1-2.md` reported that `peak_activation_fused` counts hoisted
weight-`DequantizeLinear` outputs as live activations. **Re-verified from
scratch, and it reproduces.** On `artifacts/onnx/q800_real.onnx` the zoo reports
11,250,052 B of peak activation; 9,816,452 B of that — 87.3 % — is 405
weight-DQ outputs held live, which is the model's own int8 weight payload
counted a second time on top of `weight_bytes()`. `placement()` therefore
returns `activations-in-psram` for a graph the compiler places entirely
on-chip at 625 kB.

Full diagnosis, the measured breakdown, the diff and its effect on all three
window sizes: **`budget-bug.md`**. The patch is not applied. It restores the
`is_weight` half of the recipe the fault atlas already prescribes under
`qdq-graph-charged-one-byte-per-element` — the zoo's reimplementation kept the
element-width half and dropped this one.

`tests/test_budget_weight_dq.py` is the regression test, verified to fail
against the zoo as it stands (73,216 != 66,304, the difference being exactly the
weight) and to pass with the patch. The existing suite is unaffected: 146
passed before, 146 passed after.

### 4. `zoo/quant/log_mel_nemo.py` — the calibration front end

A `register_preprocessor("log_mel_nemo")` class in the shape of the existing
`WhisperLogMel`, to be pasted into `zoo/quant/calib.py`. It is a separate file
here only so it can be reviewed on its own; it imports nothing that module does
not already have.

The zoo's position is that the front end is a property of the model, not of the
corpus, and this model is the strongest case for it: calibrating Citrinet
through Whisper's front end would be as wrong as calibrating it on noise, and
the log floor in particular is not negotiable — `2**-24` versus `1e-2` is
5.83 % WER versus 30.80 % on the same graph and corpus (`eval/results/fe.log`).

Verified against `model/fe.py`, which is simultaneously the spec the C front end
is written against and the oracle its tests compare to: **bit-identical**, max
|difference| 0.0 over an 80x800 feature block from a real dev-clean utterance.

### 5. `models/audio/citrinet-256-gamma025.toml` — the recipe

The zoo's leaderboard currently has one ASR row and it is a bad one:
whisper-tiny's encoder, 10,935 ms, 184 of 391 epochs in software, 16.4 MB
spilled to PSRAM. This model is the counter-example on the same board with the
same compiler — 628 epochs, **zero** in software, everything on-chip — and
"a convolutional CTC encoder is what this part is for" deserves to be a row
rather than a paragraph.

Resolved from this repo's evidence, with nothing left as a bare `TODO`:

- **`revision` pinned and checked, not assumed.** The LFS oid of `model.onnx`
  at `3362d8d46028368e02434a3fc7655e4f32117ef6` is
  `3e66724e…d7753d`, which is the sha256 of the local
  `artifacts/onnx/citrinet256_g025.fp32.onnx` every number in the recipe came
  from.
- **T=800 pinned by name.** All three symbolic axes carry names in the export,
  so `--fix-parametric-shapes` can address them; `is_compilable` is `True` with
  no anonymous axes. The 8 s choice is measured, not default — 4 s returns 0.56
  of the spoken words, 12 s costs accuracy on short utterances.
- **`length` folded as a `constant` role**, which is what makes the length-mask
  `Where` statically dead and removes the rank-1-graph-input rejection.
- **the real calibration provider**: `audio_folder` over LibriSpeech dev-clean
  through `log_mel_nemo`, n=48 seed=7, with the 4800-sample lead-in the shipped
  artifact was built with.
- **postconditions as a regression gate**: 628 epochs, 0 SW, 0 hybrid,
  ≤ 640,000 B activations, plus per-pool ceilings and the two report greps —
  because at T=1200 the epoch table reads 0 SW / 0 hybrid *while* 150 KB sits in
  PSRAM, and the cycle total goes *down*. Risk 4 of `docs/FEASIBILITY.md` is
  that none of this is promised: the word "group" appears zero times in
  `stneuralart_operator_support.html` r1.3, and all 107 grouped convolutions
  reaching hardware is an empirical property of compiler 4.0.1-20581 with no
  vendor commitment. That is what the gate is for.

The recipe loads cleanly through `zoo.recipe.load` today. **Three things it
needs before `zoo screen` can run it end to end**, each named in the file at the
point where it occurs:

1. **The four `patches` do not exist yet.** `drop_dead_where_mask`,
   `reducesum_div_to_reducemean`, `se_matmul_to_conv1x1` and
   `drop_trailing_logsoftmax` are described in the recipe and implemented in
   `model/clean.py`, one numbered block each, but they are not zoo patches and
   each still owes the parity gate. Without them the graph does not compile at
   all: the rank-3 SE MatMul hits a compiler internal error.
2. **`[postconditions]` is a table the loader does not read.** `recipe.load`
   ignores unknown top-level keys, so the file is valid and the gate is inert.
   Wiring it is small — a `Postconditions` dataclass, a `postconditions=` field
   on `Recipe`, and a comparison in the compile stage against
   `CompileInfo.metrics()`, whose keys the table is deliberately written in
   (`epochs_total`, `epochs_sw`, `activations_bytes`, `pool_placement`,
   `pools`). Two of the checks — hybrid epochs and the placement line — have no
   `network_c_info.json` equivalent and have to be grepped out of
   `network_generate_report.txt`, which is why they sit in their own
   `[postconditions.report_grep]` sub-table.
3. **The zoo cannot yet reproduce this artifact's quantisation.**
   `qdq.py` hardcodes `ActivationSymmetric: False`; the shipped graph was built
   with it `True`, which is what makes all 1,419 zero-points exactly 0 and lets
   the deployment contract read `offset 0` at both ends. The M55 front end
   quantises with `q = round(x/scale)` and no offset term, so a re-quantised
   graph would bias every feature by a constant. Correction 4 in
   `policy-corrections.md` proposes the key and the pass-through.

---

## What was checked, and how

Everything in this directory was verified against a real file or a real run.
The board was not touched — it belongs to another process — so nothing here
claims an on-silicon measurement.

| claim | how it was checked |
|---|---|
| budget bug reproduces | `zoo.graph.budget.analyse()` on `q800_real.onnx`, then a liveness re-implementation that separates weight-rooted tensors and prints the live set at the peak |
| the patch fixes it and breaks nothing | scratch copy of `zoo/`+`tests/`+`config/`, patch applied, `pytest tests/ -q` → 146 passed (same as baseline); new test fails before / passes after; `git apply --check` clean against the real tree |
| `log_mel_nemo` is correct | compared element-wise against `model/fe.py` on a dev-clean flac: bit-identical |
| recipe is well-formed | `zoo.recipe.load()` → validates, `is_compilable=True`, `unpinned=[]`, `anonymous_axes=[]` |
| revision pin | HF API `revision/3362d8d4…?blobs=true` LFS oid vs local `sha256sum` |
| fault entries are well-formed and route | merged copy of `known_issues.toml`, `known_issues(refresh=True)` + `classify()` on the recorded error text of each |
| `-flax-vector-conversions` signature | re-ran the recorded compile line for every CMSIS-DSP / audio-preprocessing translation unit with the flag removed; first hard failure is `StatisticsFunctions.c`, `incompatible types when initializing type 'uint16x8_t' using type 'int16x8_t'` |
| make discards `OPT +=` on override | three-line makefile: `OPT=base` / `OPT+=added` gives `base added`, and `make OPT=cmdline` gives `cmdline` |
| the 512 KiB overflow | `ls -l BuildGCC/BM/GS_Audio_N6.bin` = 714,432 B against a 524,288 B slot; `arm-none-eabi-size -A network.o` = 520,957 B, 437,705 of it `.rodata` |
| ll_aton version lock | the `#if` at `artifacts/model_c/network.c:56`, and both `ll_aton_version.h` files (262 vs 275) |
| `ai_dpu` rejection prints before hanging | `LogError` at `ai_dpu.c:109` → `LogPrintf` with `LogLevel <= LOG_LEVEL`, and `LOG_LEVEL = LOG_INFO` in `app_config.h:28`; `UART_Config()` at `audio_bm.c:128` runs before `AiDPULoadModel` |
| pool arithmetic and the 8-byte reservation | pool tables in six zoo `network.c` files and in this project's, against `c_info.json` |

One correction to this repo's own notes fell out of that last row.
`firmware/WORKLIST.md` says the `ai_dpu.c` rejection makes the app "hang
silently at startup with no output". It does not: `LogError` is above the
default log level and UART is already up, so the app prints
`AI_DPU: Output format not supported` and `AI_DPU: Check model Failed` and
*then* spins in `Error_Handler()`. The atlas entry records the accurate version,
and marks that the line was read out of the source rather than off a UART here —
the check was patched before the model was ever flashed.

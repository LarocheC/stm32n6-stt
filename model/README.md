# Model pipeline

From the upstream ONNX export to a compiled Neural-ART network.

> **These scripts carry hardcoded paths into the scratchpad they were written
> in** (`/tmp/claude-1000/.../scratchpad/...`). They are checked in as the
> record of exactly what produced the artifacts in `../artifacts/`, not as a
> turnkey pipeline. Retarget the paths before rerunning — that is Gate 0/1 work.

## Order

```
1. fetch      OpenVoiceOS/stt_en_citrinet_256_gamma_0_25_onnx -> model.onnx
2. mkstatic.py T      pin length + audio_signal to [1, 80, T]   -> static_T.onnx
3. clean.py in out    four graph rewrites (below)               -> clean_T.onnx
4. quant_real.py      quant_pre_process + static QDQ int8       -> qT_real.onnx
5. fold_stride2.py    move stride 2 off the depthwise convs     -> qT_fold.onnx
6. stedgeai generate  --target stm32n6 --st-neural-art          -> network.c + weights
```

> **`fold_stride2.py` is the exception to the path warning above.** It takes its
> paths from its own location and defaults to the repository's `artifacts/onnx/`,
> so it runs as checked in.

## `fold_stride2.py` — Gate 4 blocker 1

The NPU stalls forever on a **stride-2 depthwise** convolution
(`board/GATE4.md` Rounds 10-12; the stall follows the operator across two
compiler schedules, so it is not a scheduling artefact). It does not stall on a
stride-2 `group=1` pointwise convolution — `/encoder/encoder.{1,7,14}/res.0.0/conv/Conv`
are exactly that and run correctly. The fix moves the decimation one operator
downstream, from the depthwise convolution to the pointwise one that follows it:

```
Conv group=256 k=[K] s=[2]        Conv group=256 k=[K] s=[1]
Quantize / Dequantize        ->   Quantize / Dequantize
Conv group=1   k=[1] s=[1]        Conv group=1   k=[1] s=[2]
```

It is a selection, not an approximation: with dilation 1 the stride-2 output at
index *i* is the stride-1 output at index *2i*, the quantise/dequantise pair
between them is elementwise and commutes with decimation, and the pointwise
convolution has `k=1` with no padding so it decimates from index 0. It adds no
node and changes no shape downstream.

Fold sites are **discovered, not hardcoded** — every `Conv` with `group>1` and a
stride other than 1, walked to its pointwise partner — so the script still works
if the graph is requantised or re-exported. Every assumption is asserted and any
failure aborts with exit 2 and writes nothing; a stride-2 depthwise convolution
that cannot be folded is one that will stall the board, so it must not pass
silently. Note that 46 of the 282 `Conv` nodes carry only `kernel_shape` — they
are the squeeze-excitation 1x1 convolutions built at `clean.py:65,67` — so the
script fills in the ONNX attribute defaults rather than requiring explicit ones.

```
python model/fold_stride2.py                 # q800_real.onnx -> q800_fold.onnx
python model/fold_stride2.py --runs 30       # more random inputs in the check
```

Measured on `artifacts/onnx/q800_real.onnx` (1922 nodes, 282 Conv), 2026-08-19,
onnx 1.21.0 / onnxruntime 1.25.1:

| | |
|---|---|
| fold sites found | 3 — `/encoder/encoder.{1,7,14}/mconv.20/conv/Conv`, k=[3], [3], [7] |
| `Conv` attributes changed | 6 of 282; every initializer byte-identical |
| nodes before / after | 1922 / 1922 |
| output shape | `[1, 100, 1025]`, unchanged |
| tensors whose shape changed | 9 — the three depthwise outputs and their Q/DQ outputs, each doubled in length |
| **random-input check** | **0 of 3,075,000 output elements differ over 30 inputs, max\|diff\| = 0** |
| **real-speech check** | **0 of 102,500 differ** on `artifacts/sample1.flac` through `fe.py`; argmax and transcript identical |

The random inputs cycle three families — exactly on the int8 quantisation grid,
uniform off-grid over the same range, and N(0,1), the distribution of normalised
log-mel. The check is not insensitive: changing onnxruntime's graph optimisation
level between `ORT_DISABLE_ALL` and `ORT_ENABLE_ALL` *does* change the decoded
transcript of `sample1.flac` ("for fortnight" vs "for a fortnight"), while the
fold changes not one of 102,500 float32 values at either level.

**Blocker 2 is not addressed by this script.** `Conv2D_853` — an ordinary
`group=1, k=[1,1], s=[1,1]` pointwise convolution, one of 126 identical ones —
still stalls. See `board/GATE4.md` Round 17.

## What `clean.py` does, and why each rewrite is needed

Four rewrites take the export from "compiler crashes" to 503 nodes of seven
hardware-mapped operator types:

1. **`Where` with an all-False constant condition → identity.** The export's
   length-masking branch. With `length` pinned it is statically dead, but it
   survives as a node the NPU cannot map.
2. **`ReduceSum(axes=[-1])` then `Div` by a constant → `ReduceMean`.** This is
   the squeeze-excitation masked pooling. As exported it leaves an fp32 island
   that ONNX Runtime's QDQ pass will not quantise through.
3. **SE fully-connected `Transpose → MatMul(const) → Relu → MatMul(const) →
   Transpose` → `Conv1×1 → Relu → Conv1×1`.** The important one. Rank-3 MatMul
   hits a compiler internal error; the 1×1 convolution is numerically identical
   (weights are transposed and given a trailing singleton axis) and also deletes
   46 `Transpose` nodes.
4. **Drop the trailing `LogSoftmax`.** CTC greedy decoding is argmax, which
   LogSoftmax does not change — it is a monotone transform applied per frame. It
   costs an epoch and, being unbounded below, quantises badly.

## The frontend is the spec

`fe.py` / `fe_reference.py` is the authoritative definition of what the M55 must
compute. The C implementation must match **this**, and `fe.py` doubles as the
test oracle for it.

```
sample rate 16000    n_fft 512    win_length 400    hop 160    n_mels 80
window      symmetric Hann (fftbins=False)
pre-emphasis 0.97    power spectrum (|S|^2)    mel norm Slaney    fmin 0 fmax 8000
log         ln(mel + 2^-24)          <-- the log guard is the one value that matters
normalise   per-feature: (x - mean_t) / (std_t(ddof=1) + 1e-5), zeroed past seq_len
```

`docs/FEASIBILITY.md` §2(c) shows that nearly every one of these choices can be
varied by a point of WER or less — except the log guard, which costs 25 points
if set to 1e-2. Build the guard-occupancy telemetry in from the first commit.

## Known caveat in `quant_real.py`

Its own comment says it:

```python
cal=[cal[i] for i in rng.permutation(len(cal))[300:364]]   # disjoint from eval
                                                           # set (perm seed differs;
                                                           # check overlap later)
```

**Checked at Gate 1 — and the comment is wrong.** `eval/check_disjoint.py`
reconstructs all three calibration sets and all eight evaluation sets and
intersects them by utterance key. "perm seed differs" was not a valid argument:
`run_int8.py` draws from the *same* `d <= 3.5` pool with seed 0, and **11 of its
120 draws land in `cal_400`**. Six evaluation sets overlap `cal_400`; two
overlap `cal_1200`.

No published number is contaminated, for two separate reasons: `run_int8.py`
removes the overlap at runtime before scoring (its 109 held-out utterances carry
exactly the 804 reference words in `results/int8.json`), and every other
overlapping script evaluates the fp32 `model.onnx`, which calibration cannot
touch. **`cal_800` — the calibration set of the shipped 8 s model, built by
`q800.py`, not by this script — overlaps nothing at all.**

Note also that `quant_real.py` builds the **4 s** model. The 8 s and 12 s graphs
use `q800.py` / `q1200.py`, whose calibration sets use different duration
filters and a different slice. See `eval/GATE1.md` §1.

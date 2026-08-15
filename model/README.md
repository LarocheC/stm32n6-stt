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
5. stedgeai generate  --target stm32n6 --st-neural-art          -> network.c + weights
```

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

**The calibration/evaluation disjointness was never actually verified.** The
calibration set is drawn with seed 7 and the evaluation sets with different
seeds, so overlap is unlikely but unproven. Verify it explicitly at Gate 1
before quoting any int8 WER number — calibrating and evaluating on the same
utterances is exactly the kind of thing that makes a quantisation look free.

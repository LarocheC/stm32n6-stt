#!/usr/bin/env python3
"""Bit-exactness check for an ONNX rewrite: run both graphs on the same random
inputs under onnxruntime and report max|diff| and the number of differing
elements.  A rewrite that changes even one output value is not acceptable.

    model/verify_rewrite.py <ref.onnx> <new.onnx> [n_inputs] [--opt]

Default is ORT_DISABLE_ALL so that onnxruntime's own graph optimiser cannot
mask or manufacture a difference (board/GATE4.md Round 18c: the ENABLE_ALL and
DISABLE_ALL references disagree with each other on 7 frames of 100).
"""
import sys
import numpy as np
import onnxruntime as ort

ref, new = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 8
lvl = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if "--opt" in sys.argv
       else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

so = ort.SessionOptions()
so.graph_optimization_level = lvl
sa = ort.InferenceSession(ref, so, providers=["CPUExecutionProvider"])
sb = ort.InferenceSession(new, so, providers=["CPUExecutionProvider"])
iname = sa.get_inputs()[0].name
ishape = [d if isinstance(d, int) else 1 for d in sa.get_inputs()[0].shape]
oa = [o.name for o in sa.get_outputs()]
ob = [o.name for o in sb.get_outputs()]
print("ref outputs", oa)
print("new outputs", ob)

rng = np.random.default_rng(20260819)
worst = 0.0
ndiff = 0
ntot = 0
for i in range(n):
    x = rng.standard_normal(ishape).astype(np.float32) * 3.0
    ya = sa.run(None, {iname: x})[0]
    yb = sb.run([oa[0]] if oa[0] in ob else None, {iname: x})[0]
    d = np.abs(ya.astype(np.float64) - yb.astype(np.float64))
    worst = max(worst, float(d.max()))
    ndiff += int((ya != yb).sum())
    ntot += ya.size
    print("  input %d  shape %s  max|diff| %.6g  differing %d / %d"
          % (i, ishape, float(d.max()), int((ya != yb).sum()), ya.size))
print("TOTAL: max|diff| = %.6g   differing elements = %d / %d" % (worst, ndiff, ntot))
sys.exit(0 if ndiff == 0 else 1)

#!/usr/bin/env python3
"""Fold the stride-2 decimation of each depthwise convolution into the pointwise
convolution that follows it.

This is the fix for Gate 4 blocker 1, described in board/GATE4.md Round 12. The
Neural-ART NPU stalls forever on a stride-2 DEPTHWISE convolution (Round 10
proves the stall follows the operator across two different compiler schedules).
It does not stall on a stride-2 group=1 pointwise convolution -- the residual
downsamples /encoder/encoder.{1,7,14}/res.0.0/conv/Conv are exactly that form and
execute correctly.

In every Citrinet block the depthwise convolution reaches its pointwise partner
through nothing but quantise/dequantise:

    Conv  group=G>1  k=[K]  stride=[2]      <- stalls the NPU
    QuantizeLinear / DequantizeLinear       <- elementwise, commutes with decimation
    Conv  group=1    k=[1]  stride=[1]

Moving the stride one operator downstream selects the identical elements:

    Conv  group=G>1  k=[K]  stride=[1]
    QuantizeLinear / DequantizeLinear
    Conv  group=1    k=[1]  stride=[2]

Why it is a selection and not an approximation. With input length L, pads
(pl, pr), kernel K, dilation 1, the stride-2 depthwise output at index i is the
stride-1 depthwise output at index 2i, because both accumulate over the same
input window. The stride-1 output has length N = L + pl + pr - K + 1 and the
stride-2 output has length floor((L + pl + pr - K)/2) + 1 = floor((N-1)/2) + 1,
which is exactly the length of N decimated by 2 from index 0 -- the pointwise
convolution's own output length, since it has k=1 and no padding. The
quantise/dequantise pair in between is elementwise, so it commutes with the
decimation. The fold therefore adds no node, changes no shape downstream, and
is bit-exact.

Usage:
    python model/fold_stride2.py                     # default paths, verify on
    python model/fold_stride2.py --in A --out B
    python model/fold_stride2.py --runs 20 --seed 3
    python model/fold_stride2.py --no-verify

Fold sites are DISCOVERED, never hardcoded, so this still works if the graph is
requantised or re-exported. Every assumption is asserted; any failure aborts.
"""

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import shape_inference

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(REPO, "artifacts", "onnx", "q800_real.onnx")
DEFAULT_OUT = os.path.join(REPO, "artifacts", "onnx", "q800_fold.onnx")

# Ops permitted between the depthwise conv and its pointwise partner. They must
# be elementwise for the decimation to commute past them.
ELEMENTWISE_BETWEEN = {"QuantizeLinear", "DequantizeLinear"}

MAX_CHAIN = 8  # walking further than this means the graph is not what we think


class FoldError(RuntimeError):
    """An assumption the fold depends on did not hold."""


def attrs(node):
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode()
        else:
            out[a.name] = a
    return out


def set_ints(node, name, values):
    """Set an INTS attribute, adding it if the node relies on the ONNX default."""
    for a in node.attribute:
        if a.name == name:
            del a.ints[:]
            a.ints.extend(values)
            return
    node.attribute.append(onnx.helper.make_attribute(name, list(values)))


def conv_geometry(node):
    """Conv attributes with the ONNX defaults filled in.

    46 of the 282 Convs in q800_real.onnx carry only `kernel_shape` -- they are
    the squeeze-excitation 1x1 convolutions synthesised by model/clean.py:65,67,
    which relies on the spec defaults (strides 1, pads 0, dilations 1, group 1).
    Treating a missing attribute as an error would reject them wrongly; treating
    it as anything other than the spec default would be a silent misread.
    """
    a = attrs(node)
    kernel = a.get("kernel_shape")
    if kernel is None:
        raise FoldError("Conv %s has no kernel_shape, so its spatial rank cannot "
                        "be read from the node alone" % node.name)
    r = len(kernel)
    return dict(
        kernel_shape=list(kernel),
        group=a.get("group", 1),
        strides=list(a.get("strides", [1] * r)),
        pads=list(a.get("pads", [0] * 2 * r)),
        dilations=list(a.get("dilations", [1] * r)),
        auto_pad=a.get("auto_pad", "NOTSET"),
        rank=r,
    )


def build_maps(graph):
    producer, consumers = {}, {}
    for n in graph.node:
        for o in n.output:
            if o in producer:
                raise FoldError("tensor %s produced twice" % o)
            producer[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    return producer, consumers


def find_fold_sites(graph):
    """Every (depthwise stride>1 Conv, pointwise Conv) pair eligible for the fold.

    Aborts rather than skipping when a stride>1 depthwise conv is found whose
    surroundings do not match the pattern -- a site we cannot fold is a site
    that will stall the NPU, so it must not pass silently.
    """
    _, consumers = build_maps(graph)
    graph_outputs = {o.name for o in graph.output}
    sites = []

    for dw in graph.node:
        if dw.op_type != "Conv":
            continue
        a = conv_geometry(dw)
        group = a["group"]
        strides = a["strides"]
        if group <= 1 or all(s == 1 for s in strides):
            continue

        # --- the depthwise conv itself -------------------------------------
        kernel = a["kernel_shape"]
        pads = a["pads"]
        if any(d != 1 for d in a["dilations"]):
            raise FoldError(
                "Conv %s has dilations %s; the index identity "
                "'stride-2 out[i] == stride-1 out[2i]' is only derived for "
                "dilation 1" % (dw.name, a["dilations"]))
        if a["auto_pad"] not in ("NOTSET", ""):
            raise FoldError("Conv %s uses auto_pad=%s; pads must be explicit"
                            % (dw.name, a["auto_pad"]))
        if len(dw.output) != 1:
            raise FoldError("Conv %s has %d outputs" % (dw.name, len(dw.output)))
        if dw.output[0] in graph_outputs:
            raise FoldError("Conv %s output is a graph output; shape would change"
                            % dw.name)

        # --- walk downstream to the pointwise partner -----------------------
        chain = []
        cur = dw
        pw = None
        for _ in range(MAX_CHAIN):
            t = cur.output[0]
            cs = consumers.get(t, [])
            if len(cs) != 1:
                raise FoldError(
                    "tensor %s (from %s) has %d consumers; the fold needs exactly "
                    "one consumer chain so the decimation reaches every reader"
                    % (t, cur.name, len(cs)))
            if t in graph_outputs:
                raise FoldError("tensor %s is a graph output" % t)
            nxt = cs[0]
            if nxt.op_type == "Conv":
                pw = nxt
                break
            if nxt.op_type not in ELEMENTWISE_BETWEEN:
                raise FoldError(
                    "%s sits between depthwise %s and its pointwise conv; only "
                    "%s commute with decimation"
                    % (nxt.op_type, dw.name, sorted(ELEMENTWISE_BETWEEN)))
            if len(nxt.output) != 1:
                raise FoldError("%s has %d outputs" % (nxt.name, len(nxt.output)))
            chain.append(nxt)
            cur = nxt
        if pw is None:
            raise FoldError("no pointwise Conv found within %d hops of %s"
                            % (MAX_CHAIN, dw.name))
        if not chain:
            raise FoldError("depthwise %s feeds a Conv directly; expected a "
                            "Quantize/Dequantize pair in between" % dw.name)

        # --- the pointwise conv must be able to carry the stride -------------
        pa = conv_geometry(pw)
        if pa["group"] != 1:
            raise FoldError("consumer %s has group=%d, not a pointwise conv"
                            % (pw.name, pa["group"]))
        if any(k != 1 for k in pa["kernel_shape"]):
            raise FoldError("consumer %s has kernel_shape %s, not 1x1"
                            % (pw.name, pa["kernel_shape"]))
        if any(s != 1 for s in pa["strides"]):
            raise FoldError("consumer %s already has strides %s"
                            % (pw.name, pa["strides"]))
        if any(p != 0 for p in pa["pads"]):
            raise FoldError("consumer %s has pads %s; a padded pointwise conv "
                            "does not decimate from index 0"
                            % (pw.name, pa["pads"]))
        if any(d != 1 for d in pa["dilations"]):
            raise FoldError("consumer %s has dilations %s"
                            % (pw.name, pa["dilations"]))
        if pa["auto_pad"] not in ("NOTSET", ""):
            raise FoldError("consumer %s uses auto_pad=%s"
                            % (pw.name, pa["auto_pad"]))
        if pa["rank"] != a["rank"]:
            raise FoldError("spatial rank mismatch: %s rank %d vs %s rank %d"
                            % (dw.name, a["rank"], pw.name, pa["rank"]))
        # The depthwise input must reach the depthwise conv only; if the same
        # tensor also feeds something else that is fine -- we do not change it.
        sites.append(dict(dw=dw, pw=pw, chain=chain, strides=list(strides),
                          group=group, kernel=list(kernel), pads=list(pads)))

    return sites


def apply_fold(sites):
    for s in sites:
        set_ints(s["dw"], "strides", [1] * len(s["strides"]))
        set_ints(s["pw"], "strides", s["strides"])


def describe(sites):
    lines = []
    for s in sites:
        lines.append(
            "  %-42s grp %-4d k %-5s s %-5s pads %s\n"
            "    -> %-39s grp 1    k [1]   s %s"
            % (s["dw"].name, s["group"], s["kernel"], s["strides"], s["pads"],
               s["pw"].name, s["strides"]))
        lines.append("       via " + " -> ".join(n.op_type for n in s["chain"]))
    return "\n".join(lines)


def graph_signature(model):
    g = model.graph
    return dict(
        nodes=len(g.node),
        initializers=len(g.initializer),
        op_counts=_op_counts(g),
        outputs=[(o.name, _shape(o)) for o in g.output],
        inputs=[(i.name, _shape(i), i.type.tensor_type.elem_type) for i in g.input],
    )


def _op_counts(graph):
    c = {}
    for n in graph.node:
        c[n.op_type] = c.get(n.op_type, 0) + 1
    return c


def _shape(vi):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in vi.type.tensor_type.shape.dim]


def make_inputs(model, runs, seed):
    """Random inputs on and around the int8 grid the graph was quantised for.

    The graph's first node quantises `audio_signal` with a per-tensor scale; an
    int8-range input is that scale times [-128, 127]. Three families are used:
    values exactly on the quantisation grid, uniform values off the grid, and
    a realistic N(0,1) feature distribution (log-mel after per-feature
    normalisation), which is what the model actually sees.
    """
    g = model.graph
    if len(g.input) != 1:
        raise FoldError("expected 1 graph input, found %d" % len(g.input))
    vi = g.input[0]
    name = vi.name
    elem = vi.type.tensor_type.elem_type
    if elem != onnx.TensorProto.FLOAT:
        raise FoldError("graph input %s has elem_type %d, expected FLOAT"
                        % (name, elem))
    shape = _shape(vi)
    if any(not isinstance(d, int) for d in shape):
        raise FoldError("graph input %s has a dynamic shape %s" % (name, shape))

    # per-tensor scale of the QuantizeLinear that consumes the graph input
    scale = None
    inits = {i.name: i for i in g.initializer}
    for n in g.node:
        if n.op_type == "QuantizeLinear" and n.input[0] == name:
            t = inits.get(n.input[1])
            if t is not None:
                scale = float(onnx.numpy_helper.to_array(t).reshape(-1)[0])
            break
    if scale is None:
        raise FoldError("could not read the input QuantizeLinear scale")

    rng = np.random.default_rng(seed)
    batch = []
    for k in range(runs):
        kind = ("grid", "uniform", "gaussian")[k % 3]
        if kind == "grid":
            x = rng.integers(-128, 128, size=shape).astype(np.float32) * scale
        elif kind == "uniform":
            x = rng.uniform(-128.0, 127.0, size=shape).astype(np.float32) * scale
        else:
            x = rng.standard_normal(size=shape).astype(np.float32)
        batch.append((kind, x))
    return name, scale, shape, batch


def verify(orig_path, fold_path, runs, seed):
    import onnxruntime as ort

    model = onnx.load(orig_path, load_external_data=False)
    name, scale, shape, batch = make_inputs(model, runs, seed)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    # Keep ORT from rewriting the graphs differently on either side.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sa = ort.InferenceSession(orig_path, so, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(fold_path, so, providers=["CPUExecutionProvider"])

    oshape = sa.get_outputs()[0].shape
    if list(sb.get_outputs()[0].shape) != list(oshape):
        raise FoldError("output shape changed: %s -> %s"
                        % (oshape, sb.get_outputs()[0].shape))

    print("verification input  %s %s float32, quant scale %.9g (int8 range "
          "%.4f .. %.4f)" % (name, shape, scale, -128 * scale, 127 * scale))
    print("verification output %s %s" % (sa.get_outputs()[0].name, oshape))
    print()

    total_elems = 0
    total_diff = 0
    worst = 0.0
    for k, (kind, x) in enumerate(batch):
        ya = sa.run(None, {name: x})[0]
        yb = sb.run(None, {name: x})[0]
        if ya.shape != yb.shape:
            raise FoldError("run %d: shapes %s vs %s" % (k, ya.shape, yb.shape))
        d = np.abs(ya.astype(np.float64) - yb.astype(np.float64))
        ndiff = int(np.count_nonzero(ya != yb))
        mx = float(d.max())
        total_elems += ya.size
        total_diff += ndiff
        worst = max(worst, mx)
        print("  run %d  %-8s  elements %7d   differing %d   max|diff| %.17g"
              % (k, kind, ya.size, ndiff, mx))

    print()
    print("  TOTAL   runs %d   elements %d   differing %d   max|diff| %.17g"
          % (len(batch), total_elems, total_diff, worst))
    return dict(runs=len(batch), elements=total_elems, differing=total_diff,
                max_abs_diff=worst, output_shape=list(oshape))


def verify_real_speech(orig_path, fold_path):
    """Second, independent check: real speech through the real frontend.

    Random tensors exercise the arithmetic but not the distribution the model
    ships against. This runs artifacts/sample1.flac through model/fe.py -- the
    frontend that is the M55's spec -- and compares logits, argmax and the
    decoded transcript. Skips (does not fail) if the audio or librosa/soundfile
    are unavailable, since it is supplementary to the random-input check.
    """
    audio = os.path.join(REPO, "artifacts", "sample1.flac")
    if not os.path.exists(audio):
        print("real-speech check SKIPPED: %s not present" % audio)
        return None
    sys.path.insert(0, os.path.join(REPO, "model"))
    try:
        import soundfile as sf
        import onnxruntime as ort
        import fe
    except Exception as e:  # librosa / soundfile / vocab not available
        print("real-speech check SKIPPED: %s" % e)
        return None

    w, sr = sf.read(audio)
    w = np.asarray(w, dtype=np.float32)
    nw = 799 * 160 + 1
    buf = np.zeros(nw, dtype=np.float32)
    n = min(len(w), nw)
    buf[:n] = w[:n]
    x = fe.norm_pf(fe.nemo_mel(buf))[:, :800][None].astype(np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ya = ort.InferenceSession(orig_path, so, providers=["CPUExecutionProvider"]
                              ).run(None, {"audio_signal": x})[0]
    yb = ort.InferenceSession(fold_path, so, providers=["CPUExecutionProvider"]
                              ).run(None, {"audio_signal": x})[0]
    ndiff = int(np.count_nonzero(ya != yb))
    mx = float(np.abs(ya.astype(np.float64) - yb.astype(np.float64)).max())
    ta, tb = fe.greedy(ya[0]), fe.greedy(yb[0])
    print("real speech (%s, %.2f s): elements %d  differing %d  max|diff| %.17g"
          % (os.path.basename(audio), len(w) / sr, ya.size, ndiff, mx))
    print("  argmax identical: %s   transcript identical: %s"
          % (bool((ya.argmax(-1) == yb.argmax(-1)).all()), ta == tb))
    print("  transcript: %r" % ta)
    return dict(elements=int(ya.size), differing=ndiff, max_abs_diff=mx,
                transcript_equal=ta == tb)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", default=DEFAULT_IN)
    ap.add_argument("--out", dest="dst", default=DEFAULT_OUT)
    ap.add_argument("--runs", type=int, default=9,
                    help="random inputs for the bit-exactness check (default 9)")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    print("in   %s" % args.src)
    print("out  %s" % args.dst)
    print()

    model = onnx.load(args.src)
    before = graph_signature(model)
    print("before: %d nodes, %d initializers" % (before["nodes"],
                                                 before["initializers"]))

    sites = find_fold_sites(model.graph)
    if not sites:
        raise FoldError("no stride>1 depthwise convolution found in %s -- either "
                        "the graph is already folded or it is not the graph this "
                        "script was written for" % args.src)
    print("\nfold sites discovered: %d\n%s\n" % (len(sites), describe(sites)))

    apply_fold(sites)

    # Stale value_info would survive inference and hide a shape change, so drop
    # it and let strict inference rebuild and cross-check against graph.output.
    del model.graph.value_info[:]
    model = shape_inference.infer_shapes(model, strict_mode=True,
                                         data_prop=False)

    after = graph_signature(model)
    problems = []
    if after["nodes"] != before["nodes"]:
        problems.append("node count %d -> %d" % (before["nodes"], after["nodes"]))
    if after["initializers"] != before["initializers"]:
        problems.append("initializer count %d -> %d"
                        % (before["initializers"], after["initializers"]))
    if after["op_counts"] != before["op_counts"]:
        problems.append("op mix changed")
    if after["outputs"] != before["outputs"]:
        problems.append("output %s -> %s" % (before["outputs"], after["outputs"]))
    if after["inputs"] != before["inputs"]:
        problems.append("input %s -> %s" % (before["inputs"], after["inputs"]))
    if problems:
        raise FoldError("structure changed: " + "; ".join(problems))

    print("after:  %d nodes, %d initializers  (unchanged)"
          % (after["nodes"], after["initializers"]))
    print("output: %s %s  (unchanged)" % (after["outputs"][0][0],
                                          after["outputs"][0][1]))

    onnx.checker.check_model(model, full_check=False)
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    onnx.save(model, args.dst)
    print("wrote   %s (%d B)\n" % (args.dst, os.path.getsize(args.dst)))

    if args.no_verify:
        print("verification skipped (--no-verify)")
        return 0

    r = verify(args.src, args.dst, args.runs, args.seed)
    print()
    rs = verify_real_speech(args.src, args.dst)
    if rs is not None and (rs["differing"] or rs["max_abs_diff"]):
        print("NOT BIT-EXACT on real speech")
        return 1
    print()
    if r["max_abs_diff"] == 0.0 and r["differing"] == 0:
        print("BIT-EXACT: 0 of %d output elements differ over %d random inputs."
              % (r["elements"], r["runs"]))
        return 0
    print("NOT BIT-EXACT: %d of %d output elements differ over %d random inputs, "
          "max|diff| = %.17g" % (r["differing"], r["elements"], r["runs"],
                                 r["max_abs_diff"]))
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FoldError as e:
        print("ABORT: %s" % e, file=sys.stderr)
        sys.exit(2)

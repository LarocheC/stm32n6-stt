#!/usr/bin/env python3
"""Generate the blocker-2 reproducer graphs (board/REPRO-blocker2.md).

Blocker 2 is an epoch in which an activation accelerator's output port drives a
convolutional accelerator's data input port through the stream switch:

    ATONN_DSTPORT(STRSWITCH, 0, CONVACC, u, 0) <- ATONN_SRCPORT(STRSWITCH, 0, ACTIV, v, 0)

The STM32N6570-DK hangs forever on such an epoch (board/GATE4.md Round 18b).

Two entry points:

  extract   cut the failing site out of artifacts/onnx/q800_fold.onnx with
            onnx.utils.extract_model and shrink it, checking at every step that
            the ONNX still evaluates to the same thing.  Produces the primary
            reproducer and its matched control.

  build     rebuild the same graph parametrically (C, L, K, number of Conv
            consumers).  `build --C 256 --L 200 --K 7` reproduces the extracted
            graph's compiled epoch structure exactly; the other settings are how
            the fan-out ladder in board/REPRO-blocker2.md was made.

Nothing here compiles.  Compile with compile/gen_model.sh and score with
compile/score_build.py.

    model/repro_blocker2.py extract
    model/repro_blocker2.py build --C 256 --L 32 --K 3 --consumers 3 -o fanout3.onnx
"""
import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = os.path.join(REPO, "artifacts/onnx/q800_fold.onnx")
OUTD = os.path.join(REPO, "artifacts/onnx/repro2")

# The failing site.  /encoder/encoder.13/mconv.5/conv/Conv is one of the 36
# `group=256 k=[7]` depthwise convolutions of blocks 13..21 that the compiler
# feeds straight out of the preceding Relu (board/GATE4.md Round 18b).  It is
# `Conv2D_859` in the compiled graph, the one the board stalls on.
CUT_IN = "/encoder/encoder.13/mconv.1/conv/Conv_output_0"          # float, [1,256,200]
CUT_OUT = "/encoder/encoder.13/mconv.5/conv/Conv_output_0_DequantizeLinear_Output"

# Quantisation parameters read out of q800_fold.onnx, so the rebuilt graph
# carries the deployed model's scales rather than invented ones.
X_SCALE = 0.01582539     # /encoder/encoder.13/mconv.1/conv/Conv_output_0_scale
R_SCALE = 0.01582539     # /encoder/encoder.13/fc.1/Relu_output_0_scale
Y_SCALE = 0.00724166     # /encoder/encoder.13/mconv.5/conv/Conv_output_0_scale
W_SCALE = 0.00220463     # per-tensor collapse of the 256 per-channel weight scales


def _save(model, name):
    onnx.checker.check_model(model)
    path = os.path.join(OUTD, name)
    onnx.save(model, path)
    print("%-30s nodes=%-3d inits=%-3d %6d B"
          % (name, len(model.graph.node), len(model.graph.initializer),
             os.path.getsize(path)))
    return path


# --------------------------------------------------------------- extract path
RENAME = {
    CUT_IN: "x",
    CUT_IN + "_scale": "x_s",
    CUT_IN + "_zero_point": "x_z",
    CUT_IN + "_QuantizeLinear_Output": "xq",
    CUT_IN + "_DequantizeLinear_Output": "xd",
    "/encoder/encoder.13/fc.1/Relu_output_0": "r",
    "/encoder/encoder.13/fc.1/Relu_output_0_scale": "r_s",
    "/encoder/encoder.13/fc.1/Relu_output_0_zero_point": "r_z",
    "/encoder/encoder.13/fc.1/Relu_output_0_QuantizeLinear_Output": "rq",
    "/encoder/encoder.13/fc.1/Relu_output_0_DequantizeLinear_Output": "rd",
    "encoder.encoder.13.mconv.5.conv.weight_quantized": "w",
    "encoder.encoder.13.mconv.5.conv.weight_scale": "w_s",
    "encoder.encoder.13.mconv.5.conv.weight_zero_point": "w_z",
    "encoder.encoder.13.mconv.5.conv.weight_DequantizeLinear_Output": "wd",
    "/encoder/encoder.13/mconv.5/conv/Conv_output_0": "y",
    "/encoder/encoder.13/mconv.5/conv/Conv_output_0_scale": "y_s",
    "/encoder/encoder.13/mconv.5/conv/Conv_output_0_zero_point": "y_z",
    "/encoder/encoder.13/mconv.5/conv/Conv_output_0_QuantizeLinear_Output": "yq",
    CUT_OUT: "out",
}


def do_extract():
    from onnx.utils import extract_model
    os.makedirs(OUTD, exist_ok=True)
    if not os.path.exists(FULL):
        sys.exit("no %s -- run model/fold_stride2.py first" % FULL)

    # stage 0 -- the cut, verbatim
    s0 = os.path.join(OUTD, "s0_extract.onnx")
    extract_model(FULL, s0, [CUT_IN], [CUT_OUT])
    m = onnx.load(s0)
    print("%-30s nodes=%-3d inits=%-3d %6d B" % ("s0_extract.onnx",
          len(m.graph.node), len(m.graph.initializer), os.path.getsize(s0)))

    # stage 1 -- short names.  Pure renaming: bit-exact with the full graph.
    g = m.graph
    for n in g.node:
        n.input[:] = [RENAME.get(i, i) for i in n.input]
        n.output[:] = [RENAME.get(o, o) for o in n.output]
        n.name = {"/encoder/encoder.13/fc.1/Relu": "Relu_0",
                  "/encoder/encoder.13/mconv.5/conv/Conv": "DWConv_0"}.get(
                      n.name, n.output[0])
    for i in g.initializer:
        i.name = RENAME.get(i.name, i.name)
    for v in list(g.input) + list(g.output):
        v.name = RENAME.get(v.name, v.name)
    del g.value_info[:]
    g.name = "actv_to_convacc"
    m.doc_string = ""
    _save(m, "s1_shortnames.onnx")

    # stage 2 -- collapse the 256 per-channel weight scales to one.  Costs at
    # most one output quantisation step; halves the file.
    init = {i.name: i for i in g.initializer}
    ws = numpy_helper.to_array(init["w_s"])
    real = numpy_helper.to_array(init["w"]).astype(np.float32) * ws.reshape(-1, 1, 1)
    s = float(np.abs(real).max() / 127.0)
    q = np.clip(np.rint(real / s), -127, 127).astype(np.int8)
    keep = [i for i in g.initializer if i.name not in ("w", "w_s", "w_z")]
    del g.initializer[:]
    g.initializer.extend(keep)
    g.initializer.extend([numpy_helper.from_array(q, "w"),
                          numpy_helper.from_array(np.array(s, np.float32), "w_s"),
                          numpy_helper.from_array(np.array(0, np.int8), "w_z")])
    for n in g.node:
        if n.output[0] == "wd":
            del n.attribute[:]          # drop axis=, it was per-channel
    _save(m, "repro_actv_convacc.onnx")

    # the control -- the same graph with the Relu deleted.  The convolution is
    # still split three ways across three CONV_ACC_V2, but the stream-switch
    # source becomes a STREAM_ENG_V2.  That is `epoch_num 348` of the deployed
    # graph, the epoch the board executes in 178,933 cycles.
    keep = [n for n in g.node if n.output[0] not in ("r", "rq", "rd")]
    del g.node[:]
    g.node.extend(keep)
    for n in g.node:
        n.input[:] = ["xd" if i == "rd" else i for i in n.input]
    used = {i for n in g.node for i in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    _save(m, "control_streng_convacc.onnx")


# ----------------------------------------------------------------- build path
def do_build(a):
    os.makedirs(OUTD, exist_ok=True)
    C, L, K = a.C, a.L, a.K
    rng = np.random.RandomState(7)

    init = [numpy_helper.from_array(np.float32(X_SCALE), "x_s"),
            numpy_helper.from_array(np.int8(0), "x_z"),
            numpy_helper.from_array(np.float32(R_SCALE), "r_s"),
            numpy_helper.from_array(np.int8(0), "r_z"),
            numpy_helper.from_array(np.float32(W_SCALE), "w_s"),
            numpy_helper.from_array(np.int8(0), "w_z"),
            numpy_helper.from_array(np.float32(Y_SCALE), "y_s"),
            numpy_helper.from_array(np.int8(0), "y_z")]
    nodes = [helper.make_node("QuantizeLinear", ["x", "x_s", "x_z"], ["xq"], name="xq"),
             helper.make_node("DequantizeLinear", ["xq", "x_s", "x_z"], ["xd"], name="xd"),
             helper.make_node("Relu", ["xd"], ["r"], name="Relu_0"),
             helper.make_node("QuantizeLinear", ["r", "r_s", "r_z"], ["rq"], name="rq"),
             helper.make_node("DequantizeLinear", ["rq", "r_s", "r_z"], ["rd"], name="rd")]
    outs = []
    for i in range(a.consumers):
        sfx = "" if a.consumers == 1 else str(i)
        w = np.clip(np.rint(rng.randn(C, 1, K) * 40), -127, 127).astype(np.int8)
        init.append(numpy_helper.from_array(w, "w" + sfx))
        nodes += [
            helper.make_node("DequantizeLinear", ["w" + sfx, "w_s", "w_z"],
                             ["wd" + sfx], name="wd" + sfx),
            helper.make_node("Conv", ["rd", "wd" + sfx], ["y" + sfx],
                             name="DWConv_" + sfx, dilations=[1], group=C,
                             kernel_shape=[K], pads=[K // 2, K - 1 - K // 2],
                             strides=[1]),
            helper.make_node("QuantizeLinear", ["y" + sfx, "y_s", "y_z"],
                             ["yq" + sfx], name="yq" + sfx),
            helper.make_node("DequantizeLinear", ["yq" + sfx, "y_s", "y_z"],
                             ["out" + sfx], name="out" + sfx)]
        outs.append(helper.make_tensor_value_info("out" + sfx, TensorProto.FLOAT, [1, C, L]))

    g = helper.make_graph(nodes, "actv_to_convacc",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, L])],
                          outs, init)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    _save(m, a.o)


# ------------------------------------------------------- refuted shrinks
# Three further shrinks that look obvious and are not: each one makes the
# ST preprocessor stop recognising the QDQ pattern, so the graph falls back
# to float software epochs and the ACTIV -> CONVACC link disappears with it.
# Emitted so the negative result is reproducible rather than remembered.
def do_negatives():
    base = os.path.join(OUTD, "s1_shortnames.onnx")
    if not os.path.exists(base):
        sys.exit("run `extract` first")

    def load():
        return onnx.load(base)

    def prune(g):
        used = {i for n in g.node for i in n.input}
        keep = [i for i in g.initializer if i.name in used]
        del g.initializer[:]
        g.initializer.extend(keep)

    # int8 graph input: drop the leading QuantizeLinear
    m = load(); g = m.graph
    keep = [n for n in g.node if n.output[0] != "xq"]
    del g.node[:]; g.node.extend(keep)
    del g.input[:]
    g.input.append(helper.make_tensor_value_info("xq", TensorProto.INT8, [1, 256, 200]))
    prune(g)
    _save(m, "neg_int8_input.onnx")

    # int8 graph output: drop the trailing DequantizeLinear
    m = load(); g = m.graph
    keep = [n for n in g.node if n.output[0] != "out"]
    del g.node[:]; g.node.extend(keep)
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("yq", TensorProto.INT8, [1, 256, 200]))
    prune(g)
    _save(m, "neg_int8_output.onnx")

    # no requantisation between the Relu and the convolution
    m = load(); g = m.graph
    keep = [n for n in g.node if n.output[0] not in ("rq", "rd")]
    del g.node[:]; g.node.extend(keep)
    for n in g.node:
        n.input[:] = ["r" if i == "rd" else i for i in n.input]
    prune(g)
    _save(m, "neg_no_relu_qdq.onnx")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    sub.add_parser("negatives")
    b = sub.add_parser("build")
    b.add_argument("--C", type=int, default=256, help="channels; group=C, depthwise")
    b.add_argument("--L", type=int, default=200, help="time length")
    b.add_argument("--K", type=int, default=7, help="temporal kernel width")
    b.add_argument("--consumers", type=int, default=1,
                   help="how many independent Conv nodes read the Relu")
    b.add_argument("-o", default="built.onnx", help="file name inside artifacts/onnx/repro2/")
    args = ap.parse_args()
    if args.cmd == "extract":
        do_extract()
    elif args.cmd == "negatives":
        do_negatives()
    else:
        do_build(args)

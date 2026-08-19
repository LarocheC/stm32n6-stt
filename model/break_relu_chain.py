#!/usr/bin/env python3
"""Break the 36 ACTIV -> CONVACC stream-switch chains of board/GATE4.md Round 18b.

The Neural-ART part hangs forever on any epoch in which an activation
accelerator's output port drives a convolutional accelerator's *data* input
port through the stream switch (GATE4.md Round 18b, one positive case and one
negative control on silicon).  In the folded 800-frame Citrinet graph exactly
36 epochs do that, all of them

    /encoder/encoder.{13..21}/mconv.{5,10,15,20}/conv/Conv

each `group=256, kernel_shape=[7]` depthwise and each immediately preceded by a
Relu.  The global workaround `--force-all-in-out-to-mem` costs 618 -> 1064
epochs.

WHY THE REWRITE WORKS.  atonn canonicalises every convolution to 2-D, so a 1-D
graph gets Reshape 3D<->4D pairs inserted around every non-conv operator.  At
these 36 sites that yields

    Conv2D_853(4D) -> Q/DQ -> Reshape(4D->3D) -> Relu(3D) -> Reshape(3D->4D) -> Conv2D_859(4D)

and the compiler puts the leading Reshape in an epoch of its own (a real
STRENG->STRENG copy) while chaining Relu -> Conv2D_859 through the switch.
Where no Reshape separates a convolution from its Relu the compiler chains the
*other* way -- CONVACC -> ARITH -> ACTIV -> STRENG, e.g. epoch_block 345 of
artifacts/compile/g800_noauto -- which lands the Relu's result in memory and
makes the next convolution read it back through a streaming engine.

So: run the Relu, and its own Q/DQ pair, on the 4-D tensor.  atonn's
fuse_consecutive_reshapes/eliminate_nop_reshape then cancel the inserted pairs
against ours and the Relu ends up adjacent to its producing convolution.

    DQ -> Reshape([0,0,-1,1]) -> Relu -> Q -> DQ -> Reshape([0,0,-1]) -> Conv

Reshape does not change values, so the rewrite is exact by construction; verify
with model/verify_rewrite.py regardless.

    model/break_relu_chain.py <in.onnx> <out.onnx> [--all]

  --all            also rewrite Relus whose output already has two consumers
                   and therefore already materialises to memory.
  --min-kernel=N   only depthwise convolutions with a temporal kernel >= N
                   (default 7: the 36 sites that split into three submasks and
                   are the ones that stall).
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def kernel(node):
    for a in node.attribute:
        if a.name == "kernel_shape":
            return list(a.ints)
    return []


def group(node):
    for a in node.attribute:
        if a.name == "group":
            return a.i
    return 1


def rewrite(path_in, path_out, do_all=False, min_kernel=7):
    m = onnx.load(path_in)
    g = m.graph
    producer = {}
    consumers = {}
    for n in g.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    # sites: Relu -> QuantizeLinear -> DequantizeLinear -> Conv(group=256, k=[7])
    sites = []
    for conv in g.node:
        if conv.op_type != "Conv":
            continue
        k = kernel(conv)
        if group(conv) < 2 or len(k) != 1 or k[0] < min_kernel:
            continue
        dq = producer.get(conv.input[0])
        if dq is None or dq.op_type != "DequantizeLinear":
            continue
        q = producer.get(dq.input[0])
        if q is None or q.op_type != "QuantizeLinear":
            continue
        relu = producer.get(q.input[0])
        if relu is None or relu.op_type != "Relu":
            continue
        fanout = len(consumers.get(dq.output[0], []))
        if fanout != 1 and not do_all:
            continue
        # The rewrite only helps when the Relu's own producer is a convolution:
        # it works by letting the compiler chain CONVACC -> ARITH -> ACTIV ->
        # STRENG, which needs a CONVACC at the head.  Applied where the producer
        # is an Add instead (/encoder/encoder.22/mconv.0, whose Relu follows the
        # residual add) it *creates* a chain the baseline did not have: measured,
        # artifacts/compile/r19_relu4d epoch_block 582.
        pdq = producer.get(relu.input[0])
        pq = producer.get(pdq.input[0]) if pdq is not None and pdq.op_type == "DequantizeLinear" else None
        pconv = producer.get(pq.input[0]) if pq is not None and pq.op_type == "QuantizeLinear" else None
        if pconv is None or pconv.op_type != "Conv":
            print("   skip %s: Relu producer is %s, not Conv"
                  % (conv.name, "none" if pconv is None else pconv.op_type))
            continue
        sites.append((relu, q, dq, conv, fanout))

    print("sites: %d" % len(sites))
    for _, _, _, conv, fo in sites[:3]:
        print("   e.g. %s (dq fan-out %d)" % (conv.name, fo))

    to4 = numpy_helper.from_array(np.array([0, 0, -1, 1], dtype=np.int64), "brc_shape4")
    to3 = numpy_helper.from_array(np.array([0, 0, -1], dtype=np.int64), "brc_shape3")
    g.initializer.extend([to4, to3])

    new_nodes = []
    patched = {}          # node name -> list of nodes replacing it
    for relu, q, dq, conv, _ in sites:
        base = relu.name.replace("/", "_").strip("_")
        t_in4 = base + "_brc_in4"
        t_out3 = base + "_brc_out3"
        pre = helper.make_node("Reshape", [relu.input[0], "brc_shape4"], [t_in4],
                               name=base + "_brc_pre", allowzero=0)
        post = helper.make_node("Reshape", [dq.output[0], "brc_shape3"], [t_out3],
                                name=base + "_brc_post", allowzero=0)
        relu.input[0] = t_in4
        conv.input[0] = t_out3
        patched[relu.name] = (pre, post, dq.name)
        new_nodes.append((relu.name, pre, dq.name, post))

    # splice: pre goes immediately before the Relu, post immediately after the DQ
    out = []
    pre_by_relu = {r: p for r, p, _, _ in new_nodes}
    post_by_dq = {d: q for _, _, d, q in new_nodes}
    for n in g.node:
        if n.name in pre_by_relu:
            out.append(pre_by_relu[n.name])
        out.append(n)
        if n.name in post_by_dq:
            out.append(post_by_dq[n.name])
    del g.node[:]
    g.node.extend(out)

    # shapes of intermediate value_info are now stale for the rewritten tensors
    del g.value_info[:]
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, path_out)
    print("wrote %s (%d nodes)" % (path_out, len(m.graph.node)))
    return len(sites)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mk = 7
    for a in sys.argv[1:]:
        if a.startswith("--min-kernel="):
            mk = int(a.split("=")[1])
    rewrite(args[0], args[1], "--all" in sys.argv, mk)

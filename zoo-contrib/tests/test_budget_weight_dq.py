"""Regression test for the weight-DequantizeLinear activation double-count.

Belongs at the end of `tests/test_budget.py`, next to
`test_macs_are_not_silently_zero_on_a_quantised_graph` — same graph shape, and
the two failures have the same root: in a QDQ graph a Conv's weight is a node
output rather than an initializer, so any pass that keys on `graph.initializer`
misclassifies it. One direction undercounts MACs to zero; this one overcounts
the activation peak by the whole weight set.

Verified: FAILS against the zoo as it stands (73,216 != 66,304) and PASSES with
`zoo-contrib/zoo/graph/budget.patch` applied. See `zoo-contrib/budget-bug.md`.
"""
from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from zoo.graph import budget as bmod


def test_hoisted_weight_dequantize_is_not_a_live_activation() -> None:
    """A QDQ weight is a weight, wherever the exporter put its DequantizeLinear.

    ORT emits every weight-DequantizeLinear at the top of the topological
    order, so a liveness pass that treats node outputs as activations holds
    the whole weight set live from node 0. On Citrinet-256 that inflated the
    reported peak from 1,433,600 B to 11,250,052 B and turned an on-chip fit
    into `activations-in-psram`.
    """
    w = numpy_helper.from_array(np.ones((256, 3, 3, 3), dtype=np.int8), name="Wq")
    scale = numpy_helper.from_array(np.float32(0.02), name="s")
    zp = numpy_helper.from_array(np.int8(0), name="z")
    graph = helper.make_graph(
        [
            helper.make_node("DequantizeLinear", ["Wq", "s", "z"], ["W"]),
            helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        ],
        "hoisted_weight_dq",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 256, 8, 8])],
        initializer=[w, scale, zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    b = bmod.analyse(model)
    x_bytes = 1 * 3 * 8 * 8 * 4
    y_bytes = 1 * 256 * 8 * 8 * 4
    # The weight's 6,912 int8 bytes belong to weight_bytes, not to the peak.
    assert b.peak_activation_fused == x_bytes + y_bytes
    assert b.quantised_weight_bytes >= 256 * 3 * 3 * 3

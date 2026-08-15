# `peak_activation` counts hoisted weight-DequantizeLinear outputs as activations

**It reproduces.** On `artifacts/onnx/q800_real.onnx` the zoo reports a fused
peak activation of **11,250,052 B** where the honest figure is **1,433,600 B**.
**87.3 % of the reported peak is the model's own weights, counted a second
time**, and the consequence is a wrong verdict, not a wrong number: the zoo
calls this graph `activations-in-psram` and `fits(audio_app_onchip_bytes)`
returns `False`, while the compiler puts **625.000 kB entirely in cpuRAM2 +
npuRAM6 with 0 B in hyperRAM**.

Everything below was produced against the zoo working tree as it stands
(`zoo/graph/budget.py`, 373 lines, `peak_activation` at line 257). Nothing in
`~/stm32n6-deployment-zoo` was modified.

---

## 1. Reproduction

```
$ source ~/stm32n6-deployment-zoo/.venv/bin/activate
$ python -c '
import sys; sys.path.insert(0, "/home/claroche/stm32n6-deployment-zoo")
from zoo.graph import budget as B
b = B.analyse("/home/claroche/stm32n6-tts/artifacts/onnx/q800_real.onnx")
print(b.metrics())
print(b.placement({"budget": {"onchip_bytes": 2883576, "hyperram_bytes": 33554432,
                              "octoflash_bytes": 117440512}}))
print("fits(1507328):", b.fits(1507328))'
```

```
weight_bytes            10,473,684
quantised_weight_bytes   9,924,460
macs                 2,123,214,848
peak_activation_raw     40,906,756
peak_activation_fused   11,250,052      <-- the graph's activations are 625 kB
io_bytes                   666,000
unresolved_tensors               0
is_quantised                  True
placement            activations-in-psram
fits(1507328): False
```

Against the compiler, same graph, same window
(`compile/reports/g800_stt_0x70400000/summary.txt`, ST Edge AI Core
4.0.1-20581):

```
cpuRAM2   [0x34100000 - 0x34200000]:  200.000 kB / 1.000 MB   activations: 200.000 kB
npuRAM6   [0x34350000 - 0x343C0000]:  425.000 kB / 448.000 kB activations: 425.000 kB
hyperRAM  [0x90000000 - 0x91000000]:        0  B / 16.000 MB  activations:       0  B
octoFlash [0x70400000 - 0x74000000]:    9.728 MB / 60.000 MB  weights:     9.728 MB
```

The module's docstring is explicit that the analytic peak is a **lower bound**
and that this stage "may reject, it must never certify". Here it rejects, and
it rejects a graph the allocator places on-chip with room to spare. A lower
bound that exceeds the truth by 17.6x is not a conservative estimate; it is a
false negative in the direction the module is least defended against.

## 2. Diagnosis

`peak_activation` walks node order and treats every node output as a live
buffer:

```python
for index, node in enumerate(model.graph.node):
    for name in node.output:
        if not name or name in initializers:      # <-- outputs are never initializers
            continue
```

In a QDQ graph a Conv's weight does not arrive as an initializer. It arrives as
the **output of a DequantizeLinear** whose input is the int8 initializer, so
the `name in initializers` guard never fires for it. `macs()`, thirty lines
above, already knows this — it resolves one hop back through Q/DQ precisely
because "a lookup that only checks initializers finds nothing". `peak_activation`
was not given the same knowledge.

That alone would cost each weight's bytes for the span between its
DequantizeLinear and its consumer. What makes it 87 % rather than a rounding
error is **where ORT puts those nodes**: at the top of the topological order,
all of them, before the first Conv. Measured on the graph:

| | |
|---|---:|
| nodes | 1,922 |
| DequantizeLinear nodes | 915 |
| — of which weight-rooted | **411** |
| — of which activation-rooted | 504 |
| weight-DQ node positions | **0 – 411** |
| distance from a weight-DQ to its consumer (min / median / max) | 259 / **934** / 1,873 |

So by node 411 every weight in the network is "live", and stays live for a
median of 934 further nodes. The peak lands at node 441 (a Conv), where the
live set is:

```
   819,200 B  activation  /encoder/encoder.1/mconv.0/conv/Conv_output_0
   656,000 B  WEIGHT-DQ   decoder.decoder_layers.0.weight_DequantizeLinear_Output
   409,600 B  activation  /encoder/encoder.1/res.0.0/conv/Conv_output_0
   204,800 B  act-DQ      /encoder/encoder.0/mout/fc.1/Relu_output_0_DequantizeLinear_Output
   163,840 B  WEIGHT-DQ   onnx::Conv_3014_DequantizeLinear_Output
    65,536 B  WEIGHT-DQ   onnx::Conv_2636_DequantizeLinear_Output
    ...
```

405 of the live tensors at that instant are weight-DQ outputs, totalling
**9,816,452 B of the 11,250,052 B peak (87.3 %)**. That figure is recognisable:
it is the model's int8 weight payload, which `weight_bytes()` reports
separately as 9,924,460 B. `Budget.placement()` then computes `weights + peak`,
so the same ten megabytes are charged twice in the one comparison that decides
the pool.

Note the `fused_qdq=True` path makes this *worse-looking-but-more-honest*: it
charges those outputs at the quantised element size, which is exactly why the
inflated peak comes out at ~1x the weight payload rather than ~4x. Under
`fused_qdq=False` the same defect gives `peak_activation_raw = 40,906,756 B`,
against a corrected 2,048,000 B.

**This is a distinct defect from the two the fault atlas already records.**
`qdq-graph-charged-one-byte-per-element` is about element *width*; this is
about which tensors are counted at all. But that entry's workaround already
prescribes the fix, in the form the dnsmos implementation used:

> `is_weight(name)` follows producers up to 4 hops (no producer -> name in
> inits; non-passthrough producer -> False; hop exhaustion -> False)

The zoo's reimplementation kept the element-width half of that recipe and
dropped the `is_weight` half. The patch below restores it.

## 3. Suggested patch — **not applied**

`git apply` it from the zoo root; the same diff is at
`zoo-contrib/zoo/graph/budget.patch` and has been checked with
`git apply --check` against the current tree.

```diff
--- a/zoo/graph/budget.py
+++ b/zoo/graph/budget.py
@@ -56,6 +56,11 @@
 #: new buffer the accelerator must materialise.
 _QDQ = ("QuantizeLinear", "DequantizeLinear")
 
+#: Ops that relabel or re-scale their input rather than computing from it. A
+#: tensor reached from an initializer through only these is still a weight,
+#: however many hops away it is.
+_PASSTHROUGH = ("DequantizeLinear", "QuantizeLinear", "Cast", "Transpose", "Identity")
+
 
 @dataclass
 class Budget:
@@ -254,6 +259,45 @@
     return total
 
 
+def _weight_rooted(model: Any, *, max_hops: int = 4) -> set[str]:
+    """Tensor names that are weights wearing an activation's clothes.
+
+    In a QDQ graph a Conv's weight arrives as the output of a
+    DequantizeLinear, so it is a node output rather than an initializer. ORT
+    emits those nodes at the top of the topological order — every one of them,
+    before the first Conv — and a liveness pass that treats node outputs as
+    activations therefore holds the entire weight set live from node 0 until
+    each weight's consumer runs.
+
+    Measured on `artifacts/onnx/q800_real.onnx` (Citrinet-256, T=800): 411
+    weight-DequantizeLinear nodes occupy positions 0-411 of 1,922, their median
+    distance to their consumer is 934 nodes, and at the reported peak 405 of
+    them are live and contribute 9,816,452 B of 11,250,052 B — 87 %. The same
+    bytes are already counted by `weight_bytes()`, and `placement()` adds the
+    two together, so they are charged twice.
+
+    Follows producers up to `max_hops` through `_PASSTHROUGH` only: no
+    producer means the root is an initializer (a weight), anything else means
+    a real computation produced it (an activation).
+    """
+    producers = {out: node for node in model.graph.node for out in node.output if out}
+    initializers = {init.name for init in model.graph.initializer}
+
+    rooted: set[str] = set()
+    for name in producers:
+        cursor = name
+        for _ in range(max_hops):
+            node = producers.get(cursor)
+            if node is None:
+                if cursor in initializers:
+                    rooted.add(name)
+                break
+            if node.op_type not in _PASSTHROUGH or not node.input or not node.input[0]:
+                break
+            cursor = node.input[0]
+    return rooted
+
+
 def peak_activation(model: Any, *, fused_qdq: bool = False) -> tuple[int, int]:
     """`(peak_bytes, unresolved_tensor_count)` by liveness over the node order.
 
@@ -274,6 +318,9 @@
     initializers = {init.name for init in model.graph.initializer}
     graph_inputs = {vi.name for vi in model.graph.input}
     graph_outputs = {vi.name for vi in model.graph.output}
+    # Weights that reach their consumer through Q/DQ are still weights; they
+    # are counted by weight_bytes() and must not be counted again here.
+    weight_rooted = _weight_rooted(model) - graph_outputs
 
     # Element size the accelerator would actually hold each tensor at.
     elem_override: dict[str, int] = {}
@@ -317,7 +364,7 @@
 
     for index, node in enumerate(model.graph.node):
         for name in node.output:
-            if not name or name in initializers:
+            if not name or name in initializers or name in weight_rooted:
                 continue
             size = size_of(name)
             if size is None:
```

Three choices in it worth defending:

- **`- graph_outputs`.** A graph whose output is literally a dequantised
  constant is pathological, but if one exists its output buffer is real and
  must stay counted. Cheap insurance against a class of graph nobody has met
  yet.
- **`max_hops=4`, matching the atlas entry.** Hop exhaustion returns
  "activation", so the failure direction is toward over-counting — the same
  direction the module already errs in, and the safe one for a stage that may
  reject but must never certify.
- **applied to both `fused_qdq` paths.** A weight buffer is a weight in either
  accounting, and `weight_bytes()` already counts it in both.

## 4. Effect

| graph | reported now | with the patch | compiler's own figure |
|---|---:|---:|---:|
| `q400_real.onnx` (T=400) | 10,533,252 | **716,800** | 313,600 |
| `q800_real.onnx` (T=800) | 11,250,052 | **1,433,600** | 640,000 |
| `q1200_real.onnx` (T=1200) | 11,966,852 | **2,150,400** | 1,041,600 |

The right-hand column is the `activations` total from
`compile/reports/{g400,g800_stt_0x70400000,g1200}/summary.txt` — 306.250 kB,
625.000 kB and 1,017.188 kB respectively, all with `--Oauto-sched` and all
entirely on-chip.

Placement verdicts move with them:

| | now | with the patch |
|---|---|---|
| `placement(policy)` | `activations-in-psram` | `weights-in-flash` |
| `fits(1,507,328)` at T=800 | `False` | `True` |
| `fits(1,507,328)` at T=1200 | `False` | `False` |

`weights-in-flash` is exactly where this model ships, and the T=1200 row still
correctly refuses the narrow audio-application pool — the patch is not a
blanket "everything fits now".

The corrected analytic peak remains **above** the compiler's allocation
(1,433,600 vs 640,000 at T=800), which is the module's stated posture inverted
and worth flagging: `peak_activation` is documented as a *lower* bound on what
the allocator must find room for, and on this graph it is 2.2x *higher*,
because a fused liveness walk cannot see the buffer splitting and re-scheduling
`--Oauto-sched` performs. The safe reading after this patch is that the number
is a screening estimate with error in both directions, not a bound in either.
That is a docstring correction, not a code one, and it is out of scope here.

## 5. Regression test

`zoo-contrib/tests/test_budget_weight_dq.py`, to be appended to
`tests/test_budget.py`. It builds the minimal instance — one weight
DequantizeLinear emitted before its Conv — and asserts the peak is exactly the
input plus the output, with no weight bytes in it.

Verified both ways:

```
$ cd ~/stm32n6-deployment-zoo && pytest .../test_budget_weight_dq.py -q
E   assert 73216 == (768 + 65536)          # 73,216 - 66,304 = 6,912 = the weight
1 failed

$ cd <copy with the patch applied> && pytest .../test_budget_weight_dq.py -q
1 passed
```

The zoo's existing suite is unaffected: **146 passed** before the patch and
**146 passed** after, run against a scratch copy of `zoo/`, `tests/` and
`config/` with the patch applied.

"""Independent liveness / peak-activation recompute. Written from scratch.

Not reusing zoo/graph/budget.py. Reports:
  - per-node live set size in int8 element terms
  - the top-20 peak moments with the identity of every live tensor
  - stride / channel audit
"""
import sys, collections
import numpy as np
import onnx
from onnx import numpy_helper

path = sys.argv[1]
ELEM = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # bytes per activation element

m = onnx.load(path)
m = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
g = m.graph
inits = {i.name for i in g.initializer}

shp = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    d = []
    ok = True
    for dim in vi.type.tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            d.append(dim.dim_value)
        else:
            ok = False
    shp[vi.name] = tuple(d) if ok else None

def nbytes(n):
    s = shp.get(n)
    if s is None:
        return None
    return int(np.prod(s)) * ELEM

nodes = list(g.node)
last = {}
for i, nd in enumerate(nodes):
    for x in nd.input:
        if x and x not in inits:
            last[x] = i
gout = {o.name for o in g.output}
for o in gout:
    last[o] = len(nodes)

live = {}
unres = 0
for vi in g.input:
    if vi.name in inits:
        continue
    b = nbytes(vi.name)
    if b is None: unres += 1
    else: live[vi.name] = b

events = []
for i, nd in enumerate(nodes):
    for o in nd.output:
        if not o or o in inits:
            continue
        b = nbytes(o)
        if b is None:
            unres += 1
        else:
            live[o] = b
    tot = sum(live.values())
    events.append((tot, i, nd.op_type, nd.name, dict(live)))
    for x in list(live):
        if last.get(x, -1) <= i:
            del live[x]

events.sort(key=lambda e: -e[0])
print("path", path, "elem", ELEM, "unresolved", unres, "nodes", len(nodes))
print("PEAK bytes =", events[0][0])
for tot, i, op, name, ls in events[:6]:
    print(f"--- node#{i} {op} {name}  live={tot}")
    for k, v in sorted(ls.items(), key=lambda kv: -kv[1])[:8]:
        print(f"      {v:>10}  {k}  {shp.get(k)}")

# --- stride / channel audit -------------------------------------------------
strides = []
maxch = 0
for nd in nodes:
    if nd.op_type == "Conv":
        st = 1
        for a in nd.attribute:
            if a.name == "strides":
                st = list(a.ints)[0]
        strides.append(st)
prod = 1
for s in strides:
    prod *= s
print("conv count", len(strides), "stride product", prod, "stride hist", collections.Counter(strides))
for n, s in shp.items():
    if s and len(s) == 3:
        maxch = max(maxch, s[1])
print("max channel dim over rank-3 tensors:", maxch)
print("graph outputs:", [(o.name, shp.get(o.name)) for o in g.output])
# largest single tensor
big = sorted([(v, k) for k, v in ((k, nbytes(k)) for k in shp) if v], reverse=True)[:6]
print("largest tensors (bytes @elem):", [(b, n, shp[n]) for b, n in big])

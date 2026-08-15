import sys, onnx, numpy as np
from onnx import numpy_helper, helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
init = {i.name: i for i in g.initializer}
initA = {k: numpy_helper.to_array(v) for k, v in init.items()}
nodes = list(g.node)
prod = {o: n for n in nodes for o in n.output}
cons = {}
for n in nodes:
    for i in n.input: cons.setdefault(i, []).append(n)

rename = {}
def resolve(x):
    while x in rename: x = rename[x]
    return x

drop = set()
new_nodes = []
new_inits = []

# 1) Where with all-False const cond -> identity (rewire)
for n in nodes:
    if n.op_type == "Where" and n.input[0] in initA and not initA[n.input[0]].any():
        rename[n.output[0]] = n.input[2]; drop.add(id(n))

# 2) ReduceSum(x,axes=[-1],keepdims) then Div by const -> ReduceMean
rs_by_out = {n.output[0]: n for n in nodes if n.op_type == "ReduceSum"}
for n in nodes:
    if n.op_type == "Div" and n.input[0] in rs_by_out and n.input[1] in initA:
        rs = rs_by_out[n.input[0]]
        axes = initA[rs.input[1]] if len(rs.input) > 1 else None
        assert list(axes) == [-1], axes
        drop.add(id(rs)); drop.add(id(n))
        new_nodes.append((rs, helper.make_node("ReduceMean", [rs.input[0]], [n.output[0]],
                                               name=rs.name + "_mean", axes=[-1], keepdims=1)))

# 3) SE fc: Transpose -> MatMul(const) -> Relu -> MatMul(const) -> Transpose  ==> Conv1x1, Relu, Conv1x1
tp = {n.output[0]: n for n in nodes if n.op_type == "Transpose"}
for n in nodes:
    if n.op_type != "Transpose" or id(n) in drop: continue
    # find pattern start: Transpose whose consumer is MatMul with const 2nd input
    c = cons.get(n.output[0], [])
    if len(c) != 1 or c[0].op_type != "MatMul": continue
    mm1 = c[0]
    if mm1.input[1] not in initA: continue
    c2 = cons.get(mm1.output[0], [])
    if len(c2) != 1 or c2[0].op_type != "Relu": continue
    rl = c2[0]
    c3 = cons.get(rl.output[0], [])
    if len(c3) != 1 or c3[0].op_type != "MatMul": continue
    mm2 = c3[0]
    if mm2.input[1] not in initA: continue
    c4 = cons.get(mm2.output[0], [])
    if len(c4) != 1 or c4[0].op_type != "Transpose": continue
    tp2 = c4[0]
    W1 = initA[mm1.input[1]]   # (Cin, H)
    W2 = initA[mm2.input[1]]   # (H, Cout)
    w1 = W1.T[:, :, None].astype(np.float32)   # (H,Cin,1)
    w2 = W2.T[:, :, None].astype(np.float32)   # (Cout,H,1)
    n1 = n.name + "_w1"; n2 = n.name + "_w2"
    new_inits += [numpy_helper.from_array(w1, n1), numpy_helper.from_array(w2, n2)]
    mid1 = n.name + "_c1out"; mid2 = n.name + "_c2out"
    new_nodes.append((n, helper.make_node("Conv", [n.input[0], n1], [mid1], name=n.name + "_conv1", kernel_shape=[1])))
    new_nodes.append((n, helper.make_node("Relu", [mid1], [mid2], name=n.name + "_relu")))
    new_nodes.append((n, helper.make_node("Conv", [mid2, n2], [tp2.output[0]], name=n.name + "_conv2", kernel_shape=[1])))
    for x in (n, mm1, rl, mm2, tp2): drop.add(id(x))

# assemble
out_nodes = []
ins_map = {}
for anchor, nn in new_nodes: ins_map.setdefault(id(anchor), []).append(nn)
for n in nodes:
    for nn in ins_map.get(id(n), []): out_nodes.append(nn)
    if id(n) in drop: continue
    out_nodes.append(n)
for n in out_nodes:
    for k, i in enumerate(n.input): n.input[k] = resolve(i)
g.initializer.extend(new_inits)
# topological sort
avail = set(i.name for i in g.initializer) | set(i.name for i in g.input)
ordered=[]; pending=list(out_nodes)
while pending:
    progress=False; rest=[]
    for n in pending:
        if all((i in avail or i=='') for i in n.input):
            ordered.append(n); avail.update(n.output); progress=True
        else: rest.append(n)
    pending=rest
    if not progress: raise RuntimeError("cycle/unresolved: %s" % [(n.op_type,list(n.input)) for n in pending[:3]])
del g.node[:]; g.node.extend(ordered)

# 4) drop LogSoftmax -> raw logits output
ls = [n for n in g.node if n.op_type == "LogSoftmax"]
if ls:
    n = ls[0]
    g.node.remove(n)
    for o in g.output:
        if o.name == n.output[0]: o.name = n.input[0]
    print("dropped LogSoftmax, output renamed to", g.output[0].name)

onnx.checker.check_model(m, full_check=False)
onnx.save(m, dst)
print("saved", dst)

import sys, onnx, numpy as np
from onnx import numpy_helper
T=int(sys.argv[1])
m=onnx.load("model.onnx"); g=m.graph
inp=[i for i in g.input if i.name=='length'][0]; g.input.remove(inp)
g.initializer.append(numpy_helper.from_array(np.array([T],dtype=np.int64),'length'))
a=[i for i in g.input if i.name=='audio_signal'][0]; d=a.type.tensor_type.shape.dim
for k,v in ((0,1),(1,80),(2,T)): d[k].Clear(); d[k].dim_value=v
od=g.output[0].type.tensor_type.shape.dim
od[0].Clear(); od[0].dim_value=1; od[1].Clear(); od[1].dim_param="Tout"; od[2].Clear(); od[2].dim_value=1025
onnx.save(m, f"static_{T}.onnx")

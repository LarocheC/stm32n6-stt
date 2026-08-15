import numpy as np, onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
import sys
src,dst,T = sys.argv[1], sys.argv[2], int(sys.argv[3])
class R(CalibrationDataReader):
    def __init__(s):
        rng=np.random.default_rng(0)
        s.d=iter([{"audio_signal": rng.standard_normal((1,80,T)).astype(np.float32)} for _ in range(8)])
    def get_next(s): return next(s.d, None)
quantize_static(src, dst, R(), quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
    per_channel=True, reduce_range=False, calibrate_method=CalibrationMethod.MinMax,
    extra_options={"ActivationSymmetric":True,"WeightSymmetric":True})
print("ok")

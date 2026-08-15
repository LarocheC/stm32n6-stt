import sys, json, numpy as np, soundfile as sf
sys.path.insert(0,'/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/adv')
import fe
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
D='/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/'
recs=json.load(open(D+'adv/recs.json'))
rng=np.random.default_rng(7)
cal=[r for r in recs if r['d']<=3.5]
cal=[cal[i] for i in rng.permutation(len(cal))[300:364]]   # disjoint from eval set (perm seed differs; check overlap later)
NW=399*160+1
feats=[]
for r in cal:
    w,_=sf.read(r['f']); w=w.astype(np.float32)
    buf=np.zeros(NW,dtype=np.float32); off=4800
    L=min(len(w),NW-off); buf[off:off+L]=w[:L]
    feats.append(fe.norm_pf(fe.nemo_mel(buf))[None].astype(np.float32))
print('cal feats', len(feats), feats[0].shape, flush=True)
class R(CalibrationDataReader):
    def __init__(s): s.d=iter([{"audio_signal":f} for f in feats])
    def get_next(s): return next(s.d,None)
src=D+'citrinet/clean_400.onnx'; pre=D+'adv/clean_400.pre.onnx'; dst=D+'adv/q400_real.onnx'
quant_pre_process(src, pre, skip_optimization=True)
quantize_static(pre, dst, R(), quant_format=QuantFormat.QDQ,
  activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
  reduce_range=False, calibrate_method=CalibrationMethod.MinMax,
  extra_options={"ActivationSymmetric":True,"WeightSymmetric":True})
print('ok')

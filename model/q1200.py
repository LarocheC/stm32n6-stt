import sys, json, numpy as np, soundfile as sf
S='/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/'
sys.path.insert(0,S+'adv')
import fe
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
recs=json.load(open(S+'adv/recs.json')); rng=np.random.default_rng(7)
pool=[r for r in recs if 5.0<=r['d']<=11.5]
cal=[pool[i] for i in rng.permutation(len(pool))[:48]]
T=1200; NW=(T-1)*160+1; feats=[]
for r in cal:
    w,_=sf.read(r['f']); w=w.astype(np.float32)
    b=np.zeros(NW,dtype=np.float32); L=min(len(w),NW-4800); b[4800:4800+L]=w[:L]
    feats.append(fe.norm_pf(fe.nemo_mel(b))[:,:T][None].astype(np.float32))
print('cal',len(feats),feats[0].shape,flush=True)
class R(CalibrationDataReader):
    def __init__(s): s.d=iter([{"audio_signal":f} for f in feats])
    def get_next(s): return next(s.d,None)
quant_pre_process(S+'citrinet/clean_1200.onnx', S+'adv/clean_1200.pre.onnx', skip_optimization=True)
quantize_static(S+'adv/clean_1200.pre.onnx', S+'adv/q1200_real.onnx', R(), quant_format=QuantFormat.QDQ,
  activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True, reduce_range=False,
  calibrate_method=CalibrationMethod.MinMax, extra_options={"ActivationSymmetric":True,"WeightSymmetric":True})
print('quantized ok',flush=True)

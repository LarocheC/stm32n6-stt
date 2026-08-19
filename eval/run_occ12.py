# NOT RUNNABLE AS CHECKED IN. This script belongs to the Gate 1 host evaluation
# and reads a helper module and a data directory from the scratch tree it was
# written in, which no longer exists. It is kept as the PROVENANCE of the numbers
# in eval/results/ and eval/GATE1.md -- so the published figures have visible
# code behind them -- not as a tool to run. Retargeting the two paths at the top
# is Gate 0/1 work; nothing in QUICKSTART.md depends on it.
import sys, json, numpy as np, soundfile as sf, onnxruntime as ort
D='/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/'
sys.path.insert(0,D+'adv'); import fe
recs=json.load(open(D+'adv/recs.json')); rng=np.random.default_rng(3)
so=ort.SessionOptions(); so.intra_op_num_threads=8
s=ort.InferenceSession(D+'citrinet/model.onnx',so,providers=['CPUExecutionProvider'])
def infer(f):
    o=s.run(None,{'audio_signal':f[None].astype(np.float32),'length':np.array([f.shape[1]],dtype=np.int64)})[0]
    return fe.greedy(o[0])
sh=[r for r in recs if r['d']<=3.0]
sel=[sh[i] for i in rng.permutation(len(sh))[:30]]
print('n',len(sel),'mean dur',np.mean([r['d'] for r in sel]),flush=True)
Ts=[None,400,800,1200]
acc={T:[0,0] for T in Ts}
for r in sel:
    w,_=sf.read(r['f']); w=w.astype(np.float32); ref=r['ref'].lower()
    for T in Ts:
        if T is None: b=w
        else:
            NW=(T-1)*160+1; b=np.zeros(NW,dtype=np.float32); b[:min(len(w),NW)]=w[:NW]
        h=infer(fe.norm_pf(fe.nemo_mel(b))); e,n=fe.wer(ref,h.lower())
        acc[T][0]+=e; acc[T][1]+=n
for T in Ts:
    a=acc[T]; print(f'{"exact" if T is None else str(T//100)+"s window":>12s}  WER {100*a[0]/a[1]:6.2f}%  ({a[0]}/{a[1]})',flush=True)

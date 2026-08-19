# NOT RUNNABLE AS CHECKED IN. This script belongs to the Gate 1 host evaluation
# and reads a helper module and a data directory from the scratch tree it was
# written in, which no longer exists. It is kept as the PROVENANCE of the numbers
# in eval/results/ and eval/GATE1.md -- so the published figures have visible
# code behind them -- not as a tool to run. Retargeting the two paths at the top
# is Gate 0/1 work; nothing in QUICKSTART.md depends on it.
import sys, json, numpy as np, soundfile as sf, onnxruntime as ort
sys.path.insert(0,'/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/adv')
import fe
D='/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/'
recs=json.load(open(D+'adv/recs.json'))
rng=np.random.default_rng(1)
so=ort.SessionOptions(); so.intra_op_num_threads=4
s=ort.InferenceSession(D+'citrinet/model.onnx',so,providers=['CPUExecutionProvider'])
def infer(f):
    o=s.run(None,{'audio_signal':f[None].astype(np.float32),'length':np.array([f.shape[1]],dtype=np.int64)})[0]
    return fe.greedy(o[0])
NW=399*160+1
sh=[r for r in recs if r['d']<=2.0]
sel=[sh[i] for i in rng.permutation(len(sh))[:38]]
print('n',len(sel))
# NOISE ONLY IN THE PAD REGION (speech region untouched) -> isolates the normalization-statistics effect
lvls=[None,-60,-50,-40,-30]
acc={}
for L in lvls:
    for occ in ['speech_only_exact','in_4s_window']:
        acc[(L,occ)]=[0,0]
for r in sel:
    w,_=sf.read(r['f']); w=w.astype(np.float32); ref=r['ref'].lower(); n=len(w)
    for L in lvls:
        b=np.zeros(NW,dtype=np.float32); b[:n]=w
        if L is not None: b[n:]=rng.normal(0,10**(L/20.),NW-n)
        h=infer(fe.norm_pf(fe.nemo_mel(b))); e,nn=fe.wer(ref,h.lower())
        acc[(L,'in_4s_window')][0]+=e; acc[(L,'in_4s_window')][1]+=nn
    h=infer(fe.norm_pf(fe.nemo_mel(w))); e,nn=fe.wer(ref,h.lower())
    acc[(None,'speech_only_exact')][0]+=e; acc[(None,'speech_only_exact')][1]+=nn
a=acc[(None,'speech_only_exact')]; print(f'exact-length (no window)      WER {100*a[0]/a[1]:6.2f}%')
for L in lvls:
    a=acc[(L,'in_4s_window')]
    print(f'4s window, pad floor {str(L)+" dBFS":>12s}  WER {100*a[0]/a[1]:6.2f}%  ({a[0]}/{a[1]})')

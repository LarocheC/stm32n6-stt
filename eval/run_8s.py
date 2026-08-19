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
recs=json.load(open(D+'adv/recs.json')); rng=np.random.default_rng(1)
so=ort.SessionOptions(); so.intra_op_num_threads=4
s=ort.InferenceSession(D+'citrinet/model.onnx',so,providers=['CPUExecutionProvider'])
def infer(f):
    o=s.run(None,{'audio_signal':f[None].astype(np.float32),'length':np.array([f.shape[1]],dtype=np.int64)})[0]
    return fe.greedy(o[0])
def win(w,secs):
    N=(secs*100-1)*160+1
    b=np.zeros(N,dtype=np.float32); L=min(len(w),N); b[:L]=w[:L]; return b
# ALL utterances, unrestricted length -> "what a user who just talks gets"
sel=[recs[i] for i in rng.permutation(len(recs))[:150]]
print('n',len(sel),'median dur',round(float(np.median([r['d'] for r in sel])),2))
res={}
for secs in [4,6,8,12]:
    e=n=0; got=0; tot=0
    for r in sel:
        w,_=sf.read(r['f']); w=w.astype(np.float32); ref=r['ref'].lower()
        h=infer(fe.norm_pf(fe.nemo_mel(win(w,secs))))
        a,b=fe.wer(ref,h.lower()); e+=a; n+=b; got+=len(h.split()); tot+=b
    res[secs]=(e,n,got,tot)
    print(f'{secs}s window: WER vs full reference {100*e/n:5.1f}%   words returned/spoken {got/tot:.2f}',flush=True)

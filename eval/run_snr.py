# NOT RUNNABLE AS CHECKED IN. This script belongs to the Gate 1 host evaluation
# and reads a helper module and a data directory from the scratch tree it was
# written in, which no longer exists. It is kept as the PROVENANCE of the numbers
# in eval/results/ and eval/GATE1.md -- so the published figures have visible
# code behind them -- not as a tool to run. Retargeting the two paths at the top
# is Gate 0/1 work; nothing in QUICKSTART.md depends on it.
import sys, json, time, numpy as np, soundfile as sf, onnxruntime as ort, glob
sys.path.insert(0,'/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/adv')
import fe
D='/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/'
recs=json.load(open(D+'adv/recs.json'))
short=[r for r in recs if r['d']<=3.5]
rng=np.random.default_rng(0)
sel=[short[i] for i in rng.permutation(len(short))[:120]]
so=ort.SessionOptions(); so.intra_op_num_threads=8
s=ort.InferenceSession(D+'citrinet/model.onnx',so,providers=['CPUExecutionProvider'])
def infer(feat):
    o=s.run(None,{'audio_signal':feat[None].astype(np.float32),'length':np.array([feat.shape[1]],dtype=np.int64)})[0]
    return fe.greedy(o[0])
NW=399*160+1
# babble: sum 6 random long utterances
longs=[r for r in recs if r['d']>10]
bab=np.zeros(16000*40,dtype=np.float32)
for i in rng.permutation(len(longs))[:6]:
    w,_=sf.read(longs[i]['f']); w=w.astype(np.float32)
    w=np.tile(w,int(np.ceil(len(bab)/len(w))))[:len(bab)]
    bab+=w
bab/=np.sqrt((bab**2).mean())
def pink(n):
    X=np.fft.rfft(rng.normal(0,1,n)); f=np.arange(len(X)); f[0]=1
    X=X/np.sqrt(f); y=np.fft.irfft(X,n).astype(np.float32); return y/np.sqrt((y**2).mean())
def mixed(w, snr, kind):
    buf=np.zeros(NW,dtype=np.float32); off=4800
    L=min(len(w),NW-off); buf[off:off+L]=w[:L]
    act=buf[off:off+L]; ps=(act**2).mean()
    if snr is None: return buf
    if kind=='white': n=rng.normal(0,1,NW).astype(np.float32)
    elif kind=='pink': n=pink(NW)
    else:
        st=rng.integers(0,len(bab)-NW); n=bab[st:st+NW].copy()
    n=n/np.sqrt((n**2).mean())
    return buf + n*np.sqrt(ps/10**(snr/10.0))
conds=[('clean',None,'white')]+[(f'{k}{snr}',snr,k) for k in ['white','pink','babble'] for snr in [30,20,15,10,5]]
acc={c[0]:[0,0] for c in conds}; ex={}
t0=time.time()
for n,r in enumerate(sel):
    w,_=sf.read(r['f']); w=w.astype(np.float32); ref=r['ref'].lower()
    for name,snr,kind in conds:
        h=infer(fe.norm_pf(fe.nemo_mel(mixed(w,snr,kind))))
        e,nn=fe.wer(ref,h.lower()); acc[name][0]+=e; acc[name][1]+=nn
        if n==0: ex[name]=h
    if n%20==0: print(n, round(time.time()-t0,1), flush=True)
print('REF:', sel[0]['ref'].lower())
for name,_,_ in conds:
    print(f'{name:10s} WER {100*acc[name][0]/acc[name][1]:6.2f}%   ex: {ex[name][:90]}')
json.dump({k:[int(v[0]),int(v[1])] for k,v in acc.items()},open(D+'adv/snr.json','w'))

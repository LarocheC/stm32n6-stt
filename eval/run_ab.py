import sys, json, time, numpy as np, soundfile as sf, onnxruntime as ort
sys.path.insert(0,'/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/adv')
import fe
D='/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/'
recs=json.load(open(D+'adv/recs.json'))
short=[r for r in recs if r['d']<=3.8]
rng=np.random.default_rng(0)
idx=rng.permutation(len(short))[:120]
sel=[short[i] for i in idx]
so=ort.SessionOptions(); so.intra_op_num_threads=8
s=ort.InferenceSession(D+'citrinet/model.onnx',so,providers=['CPUExecutionProvider'])
def infer(feat):
    T=feat.shape[1]
    o=s.run(None,{'audio_signal':feat[None].astype(np.float32),'length':np.array([T],dtype=np.int64)})[0]
    return fe.greedy(o[0])
def norm_txt(t): return t.lower().replace("'"," '").replace(" '","'") if False else t.lower()
NOISE_DB=-45.0
res={k:[0,0] for k in 'ABCD'}
hyps={k:[] for k in 'ABCD'}
t0=time.time()
for n,r in enumerate(sel):
    w,sr=sf.read(r['f']); w=w.astype(np.float32)
    ref=r['ref'].lower()
    # A: native full length
    m=fe.nemo_mel(w); hA=infer(fe.norm_pf(m))
    # window buffers
    N=400*160  # 64000 samples -> center=True gives 401 frames; use 63840 for 400
    buf=np.zeros(399*160+1,dtype=np.float32)   # -> 400 frames
    L=min(len(w),len(buf)); buf[:L]=w[:L]
    mB=fe.nemo_mel(buf); Tv=m.shape[1]
    hB=infer(fe.norm_pf(mB))                       # norm over all 400
    hC=infer(fe.norm_pf(mB,seq_len=min(Tv,mB.shape[1])))  # norm over valid only
    # D: mic noise floor across whole window
    amp=10**(NOISE_DB/20.0)
    bufD=buf+rng.normal(0,amp,size=buf.shape).astype(np.float32)
    hD=infer(fe.norm_pf(fe.nemo_mel(bufD)))
    for k,h in zip('ABCD',[hA,hB,hC,hD]):
        e,nn=fe.wer(ref,h.lower()); res[k][0]+=e; res[k][1]+=nn; hyps[k].append(h)
    if n%20==0: print(n, round(time.time()-t0,1), flush=True)
for k in 'ABCD': print(k, res[k][0],'/',res[k][1], '=', round(100*res[k][0]/res[k][1],2),'%')
json.dump({'sel':[r['k'] for r in sel],'hyps':hyps,'res':res},open(D+'adv/ab.json','w'))
print('example ref :', sel[0]['ref'].lower())
for k in 'ABCD': print('  ',k, hyps[k][0])

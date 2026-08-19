# NOT RUNNABLE AS CHECKED IN. This script belongs to the Gate 1 host evaluation
# and reads a helper module and a data directory from the scratch tree it was
# written in, which no longer exists. It is kept as the PROVENANCE of the numbers
# in eval/results/ and eval/GATE1.md -- so the published figures have visible
# code behind them -- not as a tool to run. Retargeting the two paths at the top
# is Gate 0/1 work; nothing in QUICKSTART.md depends on it.
import sys, json, numpy as np, soundfile as sf, onnxruntime as ort, librosa, scipy.signal as ss
sys.path.insert(0,'/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/adv')
import fe
D='/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/'
recs=json.load(open(D+'adv/recs.json'))
pool=[r for r in recs if r['d']<=3.5]
rng=np.random.default_rng(0)
sel=[pool[i] for i in rng.permutation(len(pool))[:100]]
so=ort.SessionOptions(); so.intra_op_num_threads=4
s=ort.InferenceSession(D+'citrinet/model.onnx',so,providers=['CPUExecutionProvider'])
def infer(f):
    o=s.run(None,{'audio_signal':f[None].astype(np.float32),'length':np.array([f.shape[1]],dtype=np.int64)})[0]
    return fe.greedy(o[0])
NW=399*160+1; OFF=4800
SR,NFFT,WIN,HOP,NM=16000,512,400,160,80
def melspec(x, preemph=0.97, win='sym', htk=False, mnorm='slaney', power=2.0,
            logbase='ln', guard=2.0**-24, center=True, fmin=0, fmax=8000, ftype='log'):
    if preemph: x=np.concatenate([x[:1], x[1:]-preemph*x[:-1]])
    w = ss.get_window('hann',WIN,fftbins=(win=='per'))
    S=librosa.stft(x,n_fft=NFFT,hop_length=HOP,win_length=WIN,window=w,center=center,pad_mode='constant')
    P=(np.abs(S)**power).astype(np.float32)
    fb=librosa.filters.mel(sr=SR,n_fft=NFFT,n_mels=NM,fmin=fmin,fmax=fmax,norm=mnorm,htk=htk).astype(np.float32)
    M=fb@P
    if ftype=='log': out=np.log(M+guard)
    elif ftype=='log10': out=np.log10(M+guard)
    elif ftype=='db': out=10*np.log10(np.maximum(M,1e-10))
    return out
def place(x):
    b=np.zeros(NW,dtype=np.float32); L=min(len(x),NW-OFF); b[OFF:OFF+L]=x[:L]; return b
CONDS={
 'ref (nemo spec)':      dict(),
 'no preemphasis':       dict(preemph=0.0),
 'periodic hann':        dict(win='per'),
 'HTK mel scale':        dict(htk=True),
 'mel norm=None':        dict(mnorm=None),
 'magnitude (power=1)':  dict(power=1.0),
 'log10 not ln':         dict(logbase='log10', ftype='log10'),
 'dB floor 1e-10':       dict(ftype='db'),
 'guard 1e-6':           dict(guard=1e-6),
 'guard 1e-2':           dict(guard=1e-2),
 'fmin 20 fmax 7600':    dict(fmin=20,fmax=7600),
 'center=False':         dict(center=False),
 'no per-feat norm(gl)': None,   # global mean/var instead of per-bin
}
acc={k:[0,0] for k in CONDS}
for n,r in enumerate(sel):
    w,_=sf.read(r['f']); w=w.astype(np.float32); ref=r['ref'].lower(); b=place(w)
    for k,kw in CONDS.items():
        if kw is None:
            m=melspec(b); f=(m-m.mean())/(m.std(ddof=1)+1e-5)
        else:
            m=melspec(b,**kw); f=fe.norm_pf(m)
        h=infer(f.astype(np.float32)); e,nn=fe.wer(ref,h.lower()); acc[k][0]+=e; acc[k][1]+=nn
    if n%25==0: print(' ',n,flush=True)
for k in CONDS: print(f'{k:22s} WER {100*acc[k][0]/acc[k][1]:7.2f}%  ({acc[k][0]}/{acc[k][1]})')

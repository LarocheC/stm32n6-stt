import sys, json, time, numpy as np, soundfile as sf, onnxruntime as ort, scipy.signal as ss
sys.path.insert(0,'/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/adv')
import fe
D='/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/'
recs=json.load(open(D+'adv/recs.json'))
short=[r for r in recs if r['d']<=3.5]
rng=np.random.default_rng(0)
sel=[short[i] for i in rng.permutation(len(short))[:120]]
so=ort.SessionOptions(); so.intra_op_num_threads=4
s=ort.InferenceSession(D+'citrinet/model.onnx',so,providers=['CPUExecutionProvider'])
def infer(f):
    o=s.run(None,{'audio_signal':f[None].astype(np.float32),'length':np.array([f.shape[1]],dtype=np.int64)})[0]
    return fe.greedy(o[0])
NW=399*160+1; OFF=4800
def synth_rir(rt60, drr_db, sr=16000):
    n=int(sr*min(rt60*1.5,1.0)); t=np.arange(n)/sr
    tail=rng.normal(0,1,n)*np.exp(-3*np.log(10)*t/rt60); tail[0]=0
    tail/= np.sqrt((tail**2).sum())
    a=10**(-drr_db/20.0)
    h=np.zeros(n); h[0]=1.0; h+= a*tail
    return h.astype(np.float32)
real=sf.read('/home/claroche/azuredevops/multichannel_denoising/ir_19307.wav')[0][:,0].astype(np.float32)
real/=np.abs(real).max()
conds={'clean':None,'realRIR_T30_0.1':real,
       'rt60_0.3_drr+10':synth_rir(0.3,10),'rt60_0.4_drr+5':synth_rir(0.4,5),
       'rt60_0.6_drr+0':synth_rir(0.6,0),'rt60_0.6_drr-3':synth_rir(0.6,-3)}
# also a mic-ish response: 100 Hz HP (MEMS) + slight 5 kHz+ rolloff
bhp,ahp=ss.butter(2,100/8000,'high')
EXTRA=['mic_hp100','clip_3pct','gain_-24dB','gain_-40dB_i16','gain_-54dB_i16','gain_-54dB_f32','gain_+clip']
acc={k:[0,0] for k in list(conds)+EXTRA}
ex={}
def place(x):
    b=np.zeros(NW,dtype=np.float32); L=min(len(x),NW-OFF); b[OFF:OFF+L]=x[:L]; return b
t0=time.time()
for n,r in enumerate(sel):
    w,_=sf.read(r['f']); w=w.astype(np.float32); ref=r['ref'].lower()
    for k,h in conds.items():
        x = w if h is None else ss.fftconvolve(w,h)[:len(w)]
        hyp=infer(fe.norm_pf(fe.nemo_mel(place(x))))
        e,nn=fe.wer(ref,hyp.lower()); acc[k][0]+=e; acc[k][1]+=nn
        if n==0: ex[k]=hyp
    for k in EXTRA:
        if k=='mic_hp100': x=ss.lfilter(bhp,ahp,w).astype(np.float32)
        elif k=='clip_3pct':
            th=np.quantile(np.abs(w),0.97); x=np.clip(w,-th,th)
        elif k=='gain_-24dB': x=w*10**(-24/20.)
        elif k=='gain_-40dB_i16': x=(np.round(w*10**(-40/20.)*32768)/32768).astype(np.float32)
        elif k=='gain_-54dB_i16': x=(np.round(w*10**(-54/20.)*32768)/32768).astype(np.float32)
        elif k=='gain_-54dB_f32': x=(w*10**(-54/20.)).astype(np.float32)
        else: x=np.clip(w*8.0,-1,1)
        hyp=infer(fe.norm_pf(fe.nemo_mel(place(x))))
        e,nn=fe.wer(ref,hyp.lower()); acc[k][0]+=e; acc[k][1]+=nn
        if n==0: ex[k]=hyp
    if n%20==0: print(n, round(time.time()-t0,1), flush=True)
for k in acc: print(f'{k:20s} WER {100*acc[k][0]/acc[k][1]:6.2f}%  ex: {ex[k][:80]}')
json.dump({k:[int(v[0]),int(v[1])] for k,v in acc.items()},open(D+'adv/room.json','w'))

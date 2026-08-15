import sys, json, numpy as np, soundfile as sf, onnxruntime as ort
sys.path.insert(0,'/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/adv')
import fe
D='/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/'
recs=json.load(open(D+'adv/recs.json'))
pool=[r for r in recs if r['d']<=3.5]
rng=np.random.default_rng(0); sel=[pool[i] for i in rng.permutation(len(pool))[:100]]
so=ort.SessionOptions(); so.intra_op_num_threads=4
s=ort.InferenceSession(D+'citrinet/model.onnx',so,providers=['CPUExecutionProvider'])
def infer(f):
    o=s.run(None,{'audio_signal':f[None].astype(np.float32),'length':np.array([f.shape[1]],dtype=np.int64)})[0]
    return fe.greedy(o[0])
NW=399*160+1; OFF=4800
def place(x):
    b=np.zeros(NW,dtype=np.float32); L=min(len(x),NW-OFF); b[OFF:OFF+L]=x[:L]; return b
G=2.0**-24
w0,_=sf.read(sel[0]['f']); w0=w0.astype(np.float32)
print('LibriSpeech RMS of sample:', 20*np.log10(np.sqrt((w0**2).mean())), 'dBFS')
for g in [0,-30,-40,-48,-54,-60]:
    x=place(w0*10**(g/20.)); 
    Mp = fe.nemo_mel(x)      # log(M+G)
    M = np.exp(Mp)-G
    frac = (M < G).mean()
    print(f'gain {g:4d} dB : fraction of mel bins with power < log-guard(2^-24) = {100*frac:5.1f}%')
conds=['g0','g-54_raw','g-54_rmsnorm','g-54_peaknorm','g-54_int16_rmsnorm','g-60_rmsnorm']
acc={c:[0,0] for c in conds}
for r in sel:
    w,_=sf.read(r['f']); w=w.astype(np.float32); ref=r['ref'].lower()
    for c in conds:
        if c=='g0': x=w
        else:
            g=-54 if 'g-54' in c else -60
            x=w*10**(g/20.)
            if 'int16' in c: x=(np.round(x*32768)/32768).astype(np.float32)
            if 'rmsnorm' in c:
                x=x*(10**(-25/20.)/ (np.sqrt((x**2).mean())+1e-20))
            elif 'peaknorm' in c:
                x=x*(0.9/(np.abs(x).max()+1e-20))
        h=infer(fe.norm_pf(fe.nemo_mel(place(x.astype(np.float32)))))
        e,nn=fe.wer(ref,h.lower()); acc[c][0]+=e; acc[c][1]+=nn
for c in conds: print(f'{c:22s} WER {100*acc[c][0]/acc[c][1]:6.2f}%  ({acc[c][0]}/{acc[c][1]})')

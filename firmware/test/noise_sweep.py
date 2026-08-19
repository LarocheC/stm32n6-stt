"""Does lowering the capture level help when the capture has a NOISE FLOOR?

level_sweep.py scaled clean canned audio and found level neutral over a 26 dB
plateau.  It could not see the effect the board just showed, because canned audio
has no noise floor: LibriSpeech's inter-word passages are exact digital zeros,
so there is nothing for the 2^-24 log guard to gate.  A live capture's floor is
room noise, and the MDF gain moves speech AND that floor together against an
ABSOLUTE threshold.  So: add a floor, then sweep the level the same way.
"""
import sys, os, struct, json, numpy as np, librosa
sys.path.insert(0,'firmware/test')
import score_wav as SW
from score_corpus import levenshtein
REPO=SW.REPO; fe=SW._load_fe()
SCALE=0.120522417128086; MODEL='artifacts/onnx/q800_relu4d_all.onnx'
vocab=[l.rsplit(" ",1)[0] for l in open(os.path.join(REPO,"tokenizer","vocab.txt"),encoding="utf-8")]
raw=open('artifacts/corpus/wav_blob.bin','rb').read()
n_utt,n_samp=struct.unpack_from('<II',raw,4)
U={u['index']:u for u in json.load(open('artifacts/corpus/wav_ref.json'))['utterances']}

def pink(n, rng):
    w=rng.standard_normal(n)
    W=np.fft.rfft(w); f=np.arange(len(W)); f[0]=1
    return np.fft.irfft(W/np.sqrt(f), n)

def feats(x_counts):
    x=(np.clip(np.round(x_counts),-32768,32767)).astype(np.float32)/np.float32(32768.0)
    y=np.concatenate([x[:1],x[1:]-np.float32(0.97)*x[:-1]])
    S=librosa.stft(y,n_fft=512,hop_length=160,win_length=400,window=fe._w,
                   center=True,pad_mode="constant")
    P=(np.abs(S)**2.0).astype(np.float32)
    E=(fe._fb@P)[:,:SW.T]
    below=int((E<SW.LOG_GUARD).sum()); zero=int((E==0).sum())
    mel=fe.norm_pf(np.log(E+SW.LOG_GUARD))
    return np.clip(np.round(mel/SCALE),-128,127).astype(np.int8), below, zero

IDX=[0,1,4,8,9,13]
LEVELS=[-3.8,-10.0,-15.0,-20.0,-25.0,-30.0,-35.0]
SNRS=[("clean",None),("SNR 30 dB",30.0),("SNR 20 dB",20.0)]
rng=np.random.default_rng(7)
acc={}
for i in IDX:
    x=np.frombuffer(raw,dtype=np.int16,count=n_samp,offset=0x40+i*n_samp*2).astype(np.float64)
    ref=U[i]['ref'].lower().split()
    act=x[np.abs(x)>0]
    srms=float(np.sqrt((act**2).mean()))          # speech RMS over non-silent samples
    nz=pink(len(x),rng); nz/=np.sqrt((nz**2).mean())
    for name,snr in SNRS:
        y = x if snr is None else x + nz*(srms/(10**(snr/20.0)))
        pk=float(np.abs(y).max())
        for L in LEVELS:
            q,b,z=feats(y*((10**(L/20.0))*32768.0/pk))
            ids=SW._ort_ids(q,SCALE,MODEL)
            hyp=SW.ids_to_text(SW.ctc_greedy(np.asarray(ids),SW.BLANK),vocab).split()
            S_,I_,D_=levenshtein(ref,hyp)
            a=acc.setdefault((name,L),[0,0,[],[]])
            a[0]+=S_+I_+D_; a[1]+=len(ref)
            a[2].append(100*b/64000); a[3].append(100*SW.guard_fraction(b,z))
    print(f"  u{i} done",flush=True)
print()
hdr=f"{'peak dBFS':>10}"+"".join(f"{n:>22}" for n,_ in SNRS)
print(hdr); print(f"{'':>10}"+"".join(f"{'WER':>9}{'raw g':>7}{'excl':>6}" for _ in SNRS))
for L in LEVELS:
    row=f"{L:>10.1f}"
    for name,_ in SNRS:
        a=acc[(name,L)]
        row+=f"{100*a[0]/a[1]:>8.2f}%{np.median(a[2]):>6.0f}%{np.median(a[3]):>6.0f}%"
    print(row)

import sys, os, struct, json, numpy as np, librosa
sys.path.insert(0,'firmware/test')
import score_wav as SW
from score_corpus import levenshtein
REPO=SW.REPO
fe=SW._load_fe()
SCALE=0.120522417128086
MODEL='artifacts/onnx/q800_relu4d_all.onnx'
vocab=[l.rsplit(" ",1)[0] for l in open(os.path.join(REPO,"tokenizer","vocab.txt"),encoding="utf-8")]
raw=open('artifacts/corpus/wav_blob.bin','rb').read()
n_utt,n_samp=struct.unpack_from('<II',raw,4)
side=json.load(open('artifacts/corpus/wav_ref.json'))
U={u['index']:u for u in side['utterances']}

def feats(xf_i16_float, truncate):
    """xf is a float array already in int16 counts."""
    if truncate:
        x=(np.clip(np.round(xf_i16_float),-32768,32767)).astype(np.float32)/np.float32(32768.0)
    else:
        x=(xf_i16_float/32768.0).astype(np.float32)
    y=np.concatenate([x[:1],x[1:]-np.float32(0.97)*x[:-1]])
    S=librosa.stft(y,n_fft=512,hop_length=160,win_length=400,window=fe._w,
                   center=True,pad_mode="constant")
    P=(np.abs(S)**2.0).astype(np.float32)
    E=(fe._fb@P)[:,:SW.T]
    below=int((E<SW.LOG_GUARD).sum()); zero=int((E==0).sum())
    mel=fe.norm_pf(np.log(E+SW.LOG_GUARD))
    return np.clip(np.round(mel/SCALE),-128,127).astype(np.int8), below, zero

IDX=[0,1,4,8,9,11,13,15]
LEVELS=[-3.8,-7.6,-12.0,-19.0,-23.0,-30.0,-40.0,-54.0]
acc={}
for i in IDX:
    x=np.frombuffer(raw,dtype=np.int16,count=n_samp,offset=0x40+i*n_samp*2).astype(np.float64)
    pk=float(np.abs(x).max()); ref=U[i]['ref'].lower().split()
    for L in LEVELS:
        xf=x*((10**(L/20.0))*32768.0/pk)
        for mode in ("int16","float"):
            q,b,z=feats(xf, mode=="int16")
            ids=SW._ort_ids(q,SCALE,MODEL)
            hyp=SW.ids_to_text(SW.ctc_greedy(np.asarray(ids),SW.BLANK),vocab).split()
            S_,I_,D_=levenshtein(ref,hyp)
            k=(L,mode); a=acc.setdefault(k,[0,0,[],[]])
            a[0]+=S_+I_+D_; a[1]+=len(ref); a[2].append(100*b/64000)
            a[3].append(100*SW.guard_fraction(b,z))
    print(f"  u{i} done", flush=True)
print()
print(f"{'peak dBFS':>10}{'int16 WER':>11}{'float WER':>11}{'trunc cost':>12}"
      f"{'raw guard':>11}{'zero-excl':>11}")
for L in LEVELS:
    a=acc[(L,'int16')]; b=acc[(L,'float')]
    wa,wb=100*a[0]/a[1],100*b[0]/b[1]
    print(f"{L:>10.1f}{wa:>10.2f}%{wb:>10.2f}%{wa-wb:>+11.2f} "
          f"{np.median(a[2]):>10.1f}%{np.median(a[3]):>10.1f}%")

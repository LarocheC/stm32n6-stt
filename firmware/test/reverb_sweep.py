"""What is the 26-point gap between canned waveforms (4.3 %) and the board's own
microphone (30.3 %) made of?

Eliminated by measurement already: capture level (flat over 26 dB,
level_sweep.py), int16 quantisation (+-1.4 points, no trend), additive noise at
the SNR the board actually measures (20-26 dB -> 6.5-9 %, noise_sweep.py), and
the front end (6 int8 values of 960,000, FRONTEND.md section 11).

Untested: the acoustic channel. LibriSpeech is close-mic'd; the board is a MEMS
microphone across a room. This convolves the canned utterances with a crude
synthetic room impulse response -- exponentially decaying white noise, the
standard cheap model -- and adds the measured noise floor on top.
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
SR=16000

def rir(rt60, rng, direct=1.0):
    """Exponentially decaying noise. rt60 = time to -60 dB."""
    if rt60 <= 0: return np.array([1.0])
    n=int(rt60*SR*1.2)
    h=rng.standard_normal(n)*np.exp(-6.907*np.arange(n)/(rt60*SR))
    h[0]=direct*abs(h[0]) + direct          # keep a direct path
    return h/np.sqrt((h**2).sum())

def pink(n,rng):
    w=rng.standard_normal(n); W=np.fft.rfft(w); f=np.arange(len(W)); f[0]=1
    return np.fft.irfft(W/np.sqrt(f),n)

def feats(c):
    x=(np.clip(np.round(c),-32768,32767)).astype(np.float32)/np.float32(32768.0)
    y=np.concatenate([x[:1],x[1:]-np.float32(0.97)*x[:-1]])
    S=librosa.stft(y,n_fft=512,hop_length=160,win_length=400,window=fe._w,
                   center=True,pad_mode="constant")
    E=(fe._fb@(np.abs(S)**2.0).astype(np.float32))[:,:SW.T]
    return np.clip(np.round(fe.norm_pf(np.log(E+SW.LOG_GUARD))/SCALE),-128,127).astype(np.int8)

IDX=[0,1,4,8,9,13]
RT60=[0.0,0.15,0.30,0.50,0.80]
SNR=22.0                    # the board's own measured range
rng=np.random.default_rng(11)
print(f"{'RT60 s':>8}{'WER':>9}   (peak -15 dBFS, pink floor at {SNR:.0f} dB SNR)")
for rt in RT60:
    e=w=0
    for i in IDX:
        x=np.frombuffer(raw,dtype=np.int16,count=n_samp,offset=0x40+i*n_samp*2).astype(np.float64)
        ref=U[i]['ref'].lower().split()
        y=np.convolve(x,rir(rt,rng))[:len(x)] if rt>0 else x.copy()
        act=y[np.abs(y)>1e-9]; srms=float(np.sqrt((act**2).mean()))
        nz=pink(len(y),rng); nz/=np.sqrt((nz**2).mean())
        y=y+nz*(srms/(10**(SNR/20.0)))
        y*=(10**(-15/20.0))*32768.0/np.abs(y).max()
        ids=SW._ort_ids(feats(y),SCALE,MODEL)
        hyp=SW.ids_to_text(SW.ctc_greedy(np.asarray(ids),SW.BLANK),vocab).split()
        S_,I_,D_=levenshtein(ref,hyp); e+=S_+I_+D_; w+=len(ref)
    print(f"{rt:>8.2f}{100*e/w:>8.2f}%")

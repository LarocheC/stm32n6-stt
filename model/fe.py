import os
import numpy as np, librosa, scipy.signal
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR=16000; N_FFT=512; WIN=400; HOP=160; N_MELS=80
PREEMPH=0.97; LOG_GUARD=2.0**-24; STD_EPS=1e-5
_w = scipy.signal.get_window("hann", WIN, fftbins=False)
_fb = librosa.filters.mel(sr=SR,n_fft=N_FFT,n_mels=N_MELS,fmin=0,fmax=SR/2,norm="slaney").astype(np.float32)
def nemo_mel(x):
    x = np.concatenate([x[:1], x[1:]-PREEMPH*x[:-1]])
    S = librosa.stft(x,n_fft=N_FFT,hop_length=HOP,win_length=WIN,window=_w,center=True,pad_mode="constant")
    P = (np.abs(S)**2.0).astype(np.float32)
    return np.log(_fb@P + LOG_GUARD)
def norm_pf(m, seq_len=None):
    if seq_len is None: seq_len = m.shape[1]
    v = m[:,:seq_len]
    mu = v.mean(1,keepdims=True); sd = v.std(1,ddof=1,keepdims=True)+STD_EPS
    out = (m-mu)/sd; out[:,seq_len:]=0.0
    return out
VOCAB=[l.rsplit(" ",1)[0] for l in open(os.path.join(_REPO,"tokenizer","vocab.txt"),encoding="utf-8").read().split("\n") if l.strip()!=""]
def greedy(logits_TxV, blank=1024):
    out=[]; prev=-1
    for i in logits_TxV.argmax(-1):
        if i!=prev and i!=blank: out.append(VOCAB[i])
        prev=i
    return "".join(out).replace("▁"," ").strip()
def wer(ref, hyp):
    r=ref.split(); h=hyp.split()
    d=np.zeros((len(r)+1,len(h)+1),dtype=np.int32)
    d[:,0]=np.arange(len(r)+1); d[0,:]=np.arange(len(h)+1)
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(r[i-1]!=h[j-1]))
    return d[len(r),len(h)], len(r)

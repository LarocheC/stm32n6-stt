# NOT RUNNABLE AS CHECKED IN. This script belongs to the Gate 1 host evaluation
# and reads a helper module and a data directory from the scratch tree it was
# written in, which no longer exists. It is kept as the PROVENANCE of the numbers
# in eval/results/ and eval/GATE1.md -- so the published figures have visible
# code behind them -- not as a tool to run. Retargeting the two paths at the top
# is Gate 0/1 work; nothing in QUICKSTART.md depends on it.
import sys, json, time, numpy as np, soundfile as sf, onnxruntime as ort
sys.path.insert(0,'/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/adv')
import fe
D='/tmp/claude-1000/-home-claroche-stm32n6-tts/f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/'
recs=json.load(open(D+'adv/recs.json'))
rng=np.random.default_rng(1)
so=ort.SessionOptions(); so.intra_op_num_threads=4
s=ort.InferenceSession(D+'citrinet/model.onnx',so,providers=['CPUExecutionProvider'])
def infer(f):
    o=s.run(None,{'audio_signal':f[None].astype(np.float32),'length':np.array([f.shape[1]],dtype=np.int64)})[0]
    return fe.greedy(o[0])
NW=399*160+1; OFF=4800
def place(x,off=OFF):
    b=np.zeros(NW,dtype=np.float32); L=min(len(x),NW-off); b[off:off+L]=x[:L]; return b

print("=== PART 1: occupancy / room-tone floor (short utts <=2.0s in a 4 s window) ===",flush=True)
sh=[r for r in recs if r['d']<=2.0]
sel=[sh[i] for i in rng.permutation(len(sh))[:100]]
print('n short<=2s',len(sel), 'median dur', np.median([r['d'] for r in sel]))
conds=['native_exact','win_zeropad','win_tone-60','win_tone-50','win_tone-40','win_tone-30']
acc={c:[0,0] for c in conds}
for n,r in enumerate(sel):
    w,_=sf.read(r['f']); w=w.astype(np.float32); ref=r['ref'].lower()
    for c in conds:
        if c=='native_exact': sig=None; f=fe.norm_pf(fe.nemo_mel(w))
        else:
            b=place(w)
            if c!='win_zeropad':
                db=float(c.split('tone')[1]); b=b+rng.normal(0,10**(db/20.),NW).astype(np.float32)
            f=fe.norm_pf(fe.nemo_mel(b))
        h=infer(f); e,nn=fe.wer(ref,h.lower()); acc[c][0]+=e; acc[c][1]+=nn
    if n%25==0: print(' ',n,flush=True)
for c in conds: print(f'{c:16s} WER {100*acc[c][0]/acc[c][1]:6.2f}%  ({acc[c][0]}/{acc[c][1]})')

print("\n=== PART 2: 4 s hard cap on natural-length utterances ===",flush=True)
lg=[r for r in recs if 4.5<=r['d']<=12.0]
sel2=[lg[i] for i in rng.permutation(len(lg))[:100]]
tot_e=tot_n=0; frag_words=0; ref_words=0
ex=[]
for n,r in enumerate(sel2):
    w,_=sf.read(r['f']); w=w.astype(np.float32); ref=r['ref'].lower()
    b=place(w,0)                       # first 4 s only, no lead-in
    h=infer(fe.norm_pf(fe.nemo_mel(b)))
    e,nn=fe.wer(ref,h.lower()); tot_e+=e; tot_n+=nn
    frag_words+=len(h.split()); ref_words+=nn
    if n<3: ex.append((round(r['d'],1),ref,h))
    if n%25==0: print(' ',n,flush=True)
print(f'utterances 4.5-12 s, first 4 s captured: WER vs FULL reference = {100*tot_e/tot_n:.1f}%  ({tot_e}/{tot_n})')
print(f'words returned / words spoken = {frag_words}/{ref_words} = {frag_words/ref_words:.2f}')
for d,ref,h in ex: print(f'  [{d}s] REF: {ref}\n        HYP: {h}')

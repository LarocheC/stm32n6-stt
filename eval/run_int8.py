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
pool=[r for r in recs if r['d']<=3.5]
rng=np.random.default_rng(0)
sel=[pool[i] for i in rng.permutation(len(pool))[:120]]
rng7=np.random.default_rng(7)
cal=set(pool[i]['k'] for i in rng7.permutation(len(pool))[300:364])
ov=[r['k'] for r in sel if r['k'] in cal]
print('calib/eval overlap:', len(ov))
sel=[r for r in sel if r['k'] not in cal]
print('eval n', len(sel))
so=ort.SessionOptions(); so.intra_op_num_threads=4
f32=ort.InferenceSession(D+'citrinet/clean_400.onnx',so,providers=['CPUExecutionProvider'])
i8 =ort.InferenceSession(D+'adv/q400_real.onnx',so,providers=['CPUExecutionProvider'])
dyn=ort.InferenceSession(D+'citrinet/model.onnx',so,providers=['CPUExecutionProvider'])
NW=399*160+1; OFF=4800
def place(x):
    b=np.zeros(NW,dtype=np.float32); L=min(len(x),NW-OFF); b[OFF:OFF+L]=x[:L]; return b
longs=[r for r in recs if r['d']>10]
bab=np.zeros(16000*40,dtype=np.float32)
for i in rng.permutation(len(longs))[:6]:
    w,_=sf.read(longs[i]['f']); w=w.astype(np.float32)
    bab+=np.tile(w,int(np.ceil(len(bab)/len(w))))[:len(bab)]
bab/=np.sqrt((bab**2).mean())
def addnoise(buf,snr):
    act=buf[OFF:]; ps=(act[act!=0]**2).mean()
    st=rng.integers(0,len(bab)-NW); n=bab[st:st+NW]
    return buf+n*np.sqrt(ps/10**(snr/10.))
acc={k:[0,0] for k in ['f32_clean','int8_clean','f32_bab15','int8_bab15']}
agree=[0,0]
t0=time.time()
for n,r in enumerate(sel):
    w,_=sf.read(r['f']); w=w.astype(np.float32); ref=r['ref'].lower()
    for tag,sig in [('clean',place(w)),('bab15',addnoise(place(w),15))]:
        f=fe.norm_pf(fe.nemo_mel(sig))[None].astype(np.float32)
        oa=f32.run(None,{'audio_signal':f})[0][0]
        ob=i8.run(None,{'audio_signal':f})[0][0]
        agree[0]+=int((oa.argmax(-1)==ob.argmax(-1)).sum()); agree[1]+=oa.shape[0]
        for pfx,o in [('f32_',oa),('int8_',ob)]:
            h=fe.greedy(o); e,nn=fe.wer(ref,h.lower()); acc[pfx+tag][0]+=e; acc[pfx+tag][1]+=nn
    if n%20==0: print(n, round(time.time()-t0,1), flush=True)
for k in acc: print(f'{k:12s} WER {100*acc[k][0]/acc[k][1]:6.2f}%  ({acc[k][0]}/{acc[k][1]})')
print('frame argmax agreement f32 vs int8:', agree[0]/agree[1])
json.dump({k:[int(v[0]),int(v[1])] for k,v in acc.items()},open(D+'adv/int8.json','w'))

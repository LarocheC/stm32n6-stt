#!/usr/bin/env python3
"""Is one convolution's emitted hardware configuration an outlier?

atonn turns the ONNX quantisation scales into concrete ConvAcc shift/round
fields and an ArithAcc requantisation shift.  Those emitted fields, not the
JSON scales, are what the silicon sees.  This compares one named node against
every structurally comparable sibling in the same generated network.c and
flags each field as UNIQUE / RARE / modal.

  python3 conv_outlier.py network.c Conv2D_853
"""
import re, sys
from collections import Counter, defaultdict

def parse(src, typename):
    pat = re.compile(r'static const %s (\w+)_init(\d+) = \{(.*?)\n  \};' % typename, re.S)
    out = []
    for m in pat.finditer(src):
        f = {'_node': m.group(1), '_epoch': int(m.group(2))}
        for km in re.finditer(r'\.(\w+)\s*=\s*([^,\n]+),', m.group(3)):
            v = km.group(2).strip()
            try: f[km.group(1)] = int(v)
            except ValueError: f[km.group(1)] = v
        out.append(f)
    return out

def report(target, others, label):
    print(f"\n=== {label}: {target['_node']} (epoch {target['_epoch']}) vs {len(others)} siblings ===")
    for k in [k for k in target if not k.startswith('_')]:
        cnt = Counter(str(o.get(k, '<absent>')) for o in others)
        tv = str(target[k]); n = cnt.get(tv, 0)
        if len(tv) > 40: tv = tv[:37] + '...'
        flag = "  <<< UNIQUE" if n == 0 else (f"  <<< RARE({n})" if n <= 3 else "")
        top = sorted(cnt.items(), key=lambda x: -x[1])[:4]
        top = [(a[:20], b) for a, b in top]
        print(f"  {k:18s} = {tv:42s} others={top}{flag}")

def main():
    src = open(sys.argv[1]).read(); name = sys.argv[2]
    convs = parse(src, 'LL_Convacc_InitTypeDef')
    lead = [c for c in convs if not re.search(r'_ca_pipe_\d+$', c['_node'])]
    t = [c for c in lead if c['_node'] == name]
    if t:
        t = t[0]
        peers = [c for c in lead
                 if c is not t
                 and c.get('kernelWidth') == t.get('kernelWidth')
                 and c.get('kernelHeight') == t.get('kernelHeight')]
        report(t, peers, f"ConvAcc, same kernel {t.get('kernelHeight')}x{t.get('kernelWidth')}")
    else:
        print(f"no ConvAcc init named {name}")
    ar = parse(src, 'LL_Arithacc_InitTypeDef')
    offb = [a for a in ar if '_off_bias_' in a['_node']]
    ta = [a for a in offb if a['_node'].startswith(name + '_off_bias')]
    if ta:
        report(ta[0], [a for a in offb if a is not ta[0]],
               "ArithAcc bias-add / requantisation")

if __name__ == '__main__':
    main()

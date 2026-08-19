#!/usr/bin/env python3
"""Dump the generated epoch-block table out of an atonn-generated network.c.

The table `ll_atonn_rt_epoch_block_array[]` is authoritative: with
-DLL_ATON_EB_DBG_INFO (which Projects/GS/Makefile:199 defines) every entry
carries epoch_num, wait_mask, in_streng_mask, out_streng_mask and the
compiler's own per-epoch cycle estimates.  Struct: ll_aton_NN_interface.h:108-128.

  python3 epoch_table.py network.c                # summary + histograms
  python3 epoch_table.py network.c 340 360        # per-epoch detail for a range

Streaming engines: ATON_STRENG_NUM = 10 on STM32N6 (ATON.h:20221); engines 0-4
sit on memory port 0 and 5-9 on port 1 (Utilities/configs/stm32n6.mdesc).
"""
import re, sys
from collections import Counter

def num(x):
    x = x.strip()
    return int(x, 0) if x.lower().startswith('0x') else int(x)

def load(path):
    src = open(path).read()
    i = src.index("ll_atonn_rt_epoch_block_array[] = {")
    tbl = src[i:src.index("\n  };", i)]
    recs = []
    for item in re.findall(r'\{(.*?)\n *\},', tbl, re.S):
        f = dict(re.findall(r'\.(\w+)\s*=\s*([^,\n]+),', item))
        if 'epoch_num' not in f:      # LL_ATON_EB_DBG_INFO not compiled in
            continue
        recs.append(dict(ep=num(f['epoch_num']),
                         wait=num(f['wait_mask']),
                         inm=num(f['in_streng_mask']),
                         outm=num(f['out_streng_mask']),
                         npu=num(f.get('estimated_npu_cycles', '0')),
                         tot=num(f.get('estimated_tot_cycles', '0')),
                         flags=f.get('flags', '').strip()))
    # node names per epoch, from the Start_EpochBlock_<n> function bodies
    nodes = {}
    starts = [(m.start(), int(m.group(1))) for m in
              re.finditer(r'^static void LL_ATON_Start_EpochBlock_(\d+)\(', src, re.M)]
    for k, (pos, n) in enumerate(starts):
        stop = starts[k + 1][0] if k + 1 < len(starts) else len(src)
        nodes[n] = sorted(set(re.findall(r'/\* kind=(\w+) node=([\w.]+) \*/', src[pos:stop])))
    return recs, nodes

def main():
    recs, nodes = load(sys.argv[1])
    pc = lambda m: bin(m).count('1')
    print(f"epochs: {len(recs)}")
    print("input  streaming engines per epoch:", sorted(Counter(pc(r['inm']) for r in recs).items()))
    print("output streaming engines per epoch:", sorted(Counter(pc(r['outm']) for r in recs).items()))
    bad = [r['ep'] for r in recs if r['wait'] != r['outm']]
    print("epochs where wait_mask != out_streng_mask:", bad[:20], f"({len(bad)} total)")
    if len(sys.argv) > 3:
        lo, hi = int(sys.argv[2]), int(sys.argv[3])
        print()
        for r in recs:
            if lo <= r['ep'] <= hi:
                ns = ';'.join(f"{k}:{v}" for k, v in nodes.get(r['ep'], []))
                print(f"ep{r['ep']:4d} in=0x{r['inm']:03x}({pc(r['inm'])}) "
                      f"out=0x{r['outm']:03x}({pc(r['outm'])}) "
                      f"npu={r['npu']:>9} tot={r['tot']:>9}  {ns}")

if __name__ == '__main__':
    main()

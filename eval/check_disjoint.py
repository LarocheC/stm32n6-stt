"""Gate 1 Part A: is the int8 calibration set disjoint from every evaluation set?

model/quant_real.py's own comment deferred this check ("check overlap later").
This script performs it, by utterance key, for all three quantised graphs.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sets import load_recs, CAL_SETS, EVAL_SETS

recs = load_recs()
out = {"cal_sets": {}, "eval_sets": {}, "overlap": {}, "total_overlap": 0}

cals = {}
for name, f in CAL_SETS.items():
    s = f(recs)
    cals[name] = {r["k"] for r in s}
    out["cal_sets"][name] = {"n": len(s), "n_unique": len(cals[name]),
                             "keys": sorted(cals[name])}
    print(f"{name:10s} n={len(s):4d} unique={len(cals[name]):4d}")

print()
for ename, ef in EVAL_SETS.items():
    es = ef(recs)
    ek = {r["k"] for r in es}
    out["eval_sets"][ename] = {"n": len(es), "n_unique": len(ek)}
    for cname, ck in cals.items():
        ov = sorted(ek & ck)
        out["overlap"][f"{cname} x {ename}"] = {"n": len(ov), "keys": ov}
        out["total_overlap"] += len(ov)
        flag = "  <-- OVERLAP" if ov else ""
        print(f"{cname:10s} x {ename:18s} eval_n={len(es):4d} overlap={len(ov):3d}{flag}")
        if ov:
            print("     ", ov)

print()
print("TOTAL OVERLAP ACROSS ALL PAIRS:", out["total_overlap"])
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "disjoint.json")
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)

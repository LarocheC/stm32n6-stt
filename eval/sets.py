"""Canonical reconstruction of every calibration and evaluation utterance set
used in this repo, so that disjointness can be checked mechanically (Gate 1A).

Each set is rebuilt from `eval/results/recs.json` with the *exact* filter, seed
and slice the original script used. `recs.json` order is load-bearing: the RNG
permutes indices into the filtered list, so the record order in that file is
part of the definition of every set below.
"""
import json, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECS = os.path.join(REPO, "eval", "results", "recs.json")


def load_recs():
    return json.load(open(RECS))


def _perm(n, seed):
    return np.random.default_rng(seed).permutation(n)


# ---------------------------------------------------------------- calibration
def cal_400(recs):
    """model/quant_real.py -> artifacts/onnx/q400_real.onnx"""
    pool = [r for r in recs if r["d"] <= 3.5]
    return [pool[i] for i in _perm(len(pool), 7)[300:364]]


def cal_800(recs):
    """model/q800.py -> artifacts/onnx/q800_real.onnx  (the SHIPPED 8 s model)"""
    pool = [r for r in recs if 4.0 <= r["d"] <= 7.5]
    return [pool[i] for i in _perm(len(pool), 7)[:48]]


def cal_1200(recs):
    """model/q1200.py -> artifacts/onnx/q1200_real.onnx"""
    pool = [r for r in recs if 5.0 <= r["d"] <= 11.5]
    return [pool[i] for i in _perm(len(pool), 7)[:48]]


CAL_SETS = {"cal_400": cal_400, "cal_800": cal_800, "cal_1200": cal_1200}


# ----------------------------------------------------------------- evaluation
def eval_int8(recs):        # eval/run_int8.py
    pool = [r for r in recs if r["d"] <= 3.5]
    return [pool[i] for i in _perm(len(pool), 0)[:120]]


def eval_8s(recs):          # eval/run_8s.py  (unfiltered, all durations)
    return [recs[i] for i in _perm(len(recs), 1)[:150]]


def eval_fe(recs):          # eval/run_fe.py and eval/run_gain.py
    pool = [r for r in recs if r["d"] <= 3.5]
    return [pool[i] for i in _perm(len(pool), 0)[:100]]


def eval_snr(recs):         # eval/run_snr.py (and the deleted run_room.py)
    pool = [r for r in recs if r["d"] <= 3.5]
    return [pool[i] for i in _perm(len(pool), 0)[:120]]


def eval_ab(recs):          # eval/run_ab.py
    pool = [r for r in recs if r["d"] <= 3.8]
    return [pool[i] for i in _perm(len(pool), 0)[:120]]


def eval_occ1(recs):        # eval/run_occ.py part 1
    pool = [r for r in recs if r["d"] <= 2.0]
    return [pool[i] for i in _perm(len(pool), 1)[:100]]


def eval_pad(recs):         # eval/run_pad.py
    pool = [r for r in recs if r["d"] <= 2.0]
    return [pool[i] for i in _perm(len(pool), 1)[:38]]


def eval_occ12(recs):       # eval/run_occ12.py
    pool = [r for r in recs if r["d"] <= 3.0]
    return [pool[i] for i in _perm(len(pool), 3)[:30]]


EVAL_SETS = {
    "run_int8": eval_int8, "run_8s": eval_8s, "run_fe/run_gain": eval_fe,
    "run_snr/run_room": eval_snr, "run_ab": eval_ab, "run_occ.p1": eval_occ1,
    "run_pad": eval_pad, "run_occ12": eval_occ12,
}


def cal_keys_all(recs):
    """Union of every calibration key across all three quantised graphs."""
    k = set()
    for f in CAL_SETS.values():
        k |= {r["k"] for r in f(recs)}
    return k

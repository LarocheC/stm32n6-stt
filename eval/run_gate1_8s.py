"""Gate 1 Part B — fp32 vs int8 WER at the SHIPPED 8-second window (T=800).

  fp32 : artifacts/onnx/clean_800.onnx
  int8 : artifacts/onnx/q800_real.onnx   (QDQ, real-speech calibration)

Evaluation set is drawn from LibriSpeech dev-clean with every calibration
utterance of all three quantised graphs removed (see eval/sets.py and
eval/check_disjoint.py).

Two WERs are reported, because they answer different questions:

  (a) utterance-fits-window : only utterances whose audio fits entirely inside
      the 8 s buffer after the 0.3 s lead-in. No truncation. This is the
      model's real recognition accuracy, and the number the int8 gate is
      judged on.
  (b) full-reference        : every utterance, scored against the complete
      spoken reference regardless of length, so truncation counts as
      deletions. This is what a user experiences, and is the same kind of
      number as the 4s/6s/8s/12s table in docs/FEASIBILITY.md §2(a).

Text normalisation for WER: uppercase; delete every character that is not
A-Z, apostrophe or space; collapse runs of whitespace; split on whitespace.
LibriSpeech references are already uppercase A-Z + apostrophe and the model
vocabulary is lowercase a-z + apostrophe, so this is uppercasing plus a
no-op guard on this corpus (the guard exists to catch the <unk>/<blk>
pieces, which would otherwise survive as literal angle-bracket tokens).

Usage:  python eval/run_gate1_8s.py [N]     (default N=600)
"""
import json, os, re, sys, time
import numpy as np, soundfile as sf, onnxruntime as ort

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "model"))
sys.path.insert(0, os.path.join(REPO, "eval"))
import fe                                    # NeMo-exact frontend + greedy CTC + WER
from sets import load_recs, cal_keys_all, CAL_SETS

SR = 16000
T = 800
NW = (T - 1) * 160 + 1          # 127841 samples of waveform -> exactly 800 mel frames
OFF = 4800                      # 0.3 s lead-in, same placement quantisation used
FITS = NW - OFF                 # 123041 samples = 7.690 s of speech fits untruncated
SEED = 20260815                 # fresh seed: this draw is not any prior eval set
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600

_KEEP = re.compile(r"[^A-Z' ]+")


def norm(s):
    return _KEEP.sub(" ", s.upper()).split()


def wer_counts(ref, hyp):
    """Levenshtein word distance -> (errors, ref_words). Same DP as fe.wer,
    but on the normalised token lists, and returning S/D/I as well."""
    r, h = norm(ref), norm(hyp)
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return int(d[n, m]), n, m


def place(w):
    b = np.zeros(NW, dtype=np.float32)
    L = min(len(w), FITS)
    b[OFF:OFF + L] = w[:L]
    return b


def main():
    recs = load_recs()
    calk = cal_keys_all(recs)
    held = [r for r in recs if r["k"] not in calk]
    rng = np.random.default_rng(SEED)
    sel = [held[i] for i in rng.permutation(len(held))[:N]]

    print(f"corpus {len(recs)}  calibration keys excluded {len(calk)}  "
          f"held-out pool {len(held)}  drawn {len(sel)} (seed {SEED})")
    durs = np.array([r["d"] for r in sel])
    print(f"durations: median {np.median(durs):.2f}s  mean {durs.mean():.2f}s  "
          f"max {durs.max():.2f}s  fits-window {int((durs <= FITS / SR).sum())}")
    assert not (set(r["k"] for r in sel) & calk), "calibration leak in eval set"

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.log_severity_level = 3
    f32 = ort.InferenceSession(os.path.join(REPO, "artifacts/onnx/clean_800.onnx"),
                               so, providers=["CPUExecutionProvider"])
    i8 = ort.InferenceSession(os.path.join(REPO, "artifacts/onnx/q800_real.onnx"),
                              so, providers=["CPUExecutionProvider"])

    acc = {k: [0, 0, 0] for k in ("f32_fit", "i8_fit", "f32_full", "i8_full")}
    agree_hit = agree_tot = 0
    per = []
    t0 = time.time()
    for n, r in enumerate(sel):
        w, sr = sf.read(r["f"])
        assert sr == SR
        w = w.astype(np.float32)
        fits = len(w) <= FITS
        feat = fe.norm_pf(fe.nemo_mel(place(w)))[:, :T][None].astype(np.float32)
        oa = f32.run(None, {"audio_signal": feat})[0][0]
        ob = i8.run(None, {"audio_signal": feat})[0][0]
        aa, ab = oa.argmax(-1), ob.argmax(-1)
        agree_hit += int((aa == ab).sum())
        agree_tot += aa.shape[0]
        row = {"k": r["k"], "d": r["d"], "fits": bool(fits)}
        for tag, o in (("f32", oa), ("i8", ob)):
            hyp = fe.greedy(o)
            e, nref, nhyp = wer_counts(r["ref"], hyp)
            row[tag] = [e, nref, nhyp]
            acc[tag + "_full"][0] += e
            acc[tag + "_full"][1] += nref
            acc[tag + "_full"][2] += nhyp
            if fits:
                acc[tag + "_fit"][0] += e
                acc[tag + "_fit"][1] += nref
                acc[tag + "_fit"][2] += nhyp
        per.append(row)
        if n % 50 == 0:
            print(f"  {n:4d}/{len(sel)}  {time.time()-t0:6.1f}s", flush=True)

    def pct(k):
        return 100.0 * acc[k][0] / acc[k][1]

    nfit = sum(1 for p in per if p["fits"])
    print()
    print(f"(a) utterance-fits-window   n={nfit}  ref words {acc['f32_fit'][1]}")
    print(f"    fp32 WER {pct('f32_fit'):6.2f}%  ({acc['f32_fit'][0]}/{acc['f32_fit'][1]})")
    print(f"    int8 WER {pct('i8_fit'):6.2f}%  ({acc['i8_fit'][0]}/{acc['i8_fit'][1]})")
    print(f"    delta    {pct('i8_fit') - pct('f32_fit'):+6.2f} points")
    print()
    print(f"(b) full-reference          n={len(per)}  ref words {acc['f32_full'][1]}")
    print(f"    fp32 WER {pct('f32_full'):6.2f}%  ({acc['f32_full'][0]}/{acc['f32_full'][1]})")
    print(f"    int8 WER {pct('i8_full'):6.2f}%  ({acc['i8_full'][0]}/{acc['i8_full'][1]})")
    print(f"    delta    {pct('i8_full') - pct('f32_full'):+6.2f} points")
    print(f"    words returned/spoken  fp32 {acc['f32_full'][2]/acc['f32_full'][1]:.3f}"
          f"   int8 {acc['i8_full'][2]/acc['i8_full'][1]:.3f}")
    print()
    print(f"frame argmax agreement fp32 vs int8: {agree_hit/agree_tot:.4f} "
          f"({agree_hit}/{agree_tot})")

    out = {
        "window_s": 8, "T": T, "window_samples": NW, "lead_in_samples": OFF,
        "fits_threshold_samples": FITS, "fits_threshold_s": FITS / SR,
        "fp32_model": "artifacts/onnx/clean_800.onnx",
        "int8_model": "artifacts/onnx/q800_real.onnx",
        "corpus": "LibriSpeech dev-clean (corpus/LibriSpeech/dev-clean)",
        "seed": SEED, "n_drawn": len(sel), "n_fits_window": nfit,
        "calibration_keys_excluded": sorted(calk),
        "n_calibration_keys_excluded": len(calk),
        "calibration_sets": {k: sorted(r["k"] for r in f(recs))
                             for k, f in CAL_SETS.items()},
        "normalisation": "uppercase; drop chars not in [A-Z' ]; collapse whitespace",
        "counts": {k: {"errors": v[0], "ref_words": v[1], "hyp_words": v[2]}
                   for k, v in acc.items()},
        "wer_pct": {k: pct(k) for k in acc},
        "delta_points": {"fits_window": pct("i8_fit") - pct("f32_fit"),
                         "full_reference": pct("i8_full") - pct("f32_full")},
        "frame_argmax_agreement": agree_hit / agree_tot,
        "frame_argmax_hits": agree_hit, "frame_argmax_total": agree_tot,
        "eval_keys": [p["k"] for p in per],
        "per_utterance": per,
    }
    dst = os.path.join(REPO, "eval", "results", "gate1_8s.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()

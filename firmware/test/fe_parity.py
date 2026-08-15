#!/usr/bin/env python3
"""Host parity harness for firmware/src/citrinet_fe.c.

model/fe.py is the oracle.  For each test case this script

  1. builds an 8 s, 128,000-sample 16-bit WAV from a LibriSpeech dev-clean flac
     (4,800 samples of lead-in silence, as eval/ and gen_canned_features.py do),
  2. runs a natively compiled test_fe_host binary on it,
  3. reads back the *exact int16 buffer the C code was fed* and pushes it through
     model/fe.py, so the two sides cannot disagree about the input,
  4. compares the log-mel plane, the normalised plane and — the number that
     actually matters — the 64,000 int8 values.

Usage (normally via run_fe_parity.sh, which compiles the binaries first):

    python firmware/test/fe_parity.py --bin BUILD/fe_cmsis [--bin BUILD/fe_portable ...]
                                      [--n 5] [--json OUT.json]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "model"))
import fe as oracle  # noqa: E402  model/fe.py

SCALE = 0.120522417128086          # compile/reports/g800_st_autosched/io_contract.h
T, NMEL = 800, 80
NSAMPLES = T * 160                 # 128,000
LEAD_IN = 4800
GUARD = 2.0 ** -24


# --------------------------------------------------------------------- fixtures
def pick_utterances(n):
    """Deterministic pick of n dev-clean flacs long enough to fill the window."""
    root = os.path.join(REPO, "corpus", "LibriSpeech", "dev-clean")
    files = sorted(glob.glob(os.path.join(root, "*", "*", "*.flac")))
    if not files:
        raise SystemExit("no flacs under %s" % root)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(files))
    out = []
    for i in order:
        f = files[i]
        info = sf.info(f)
        if info.samplerate != 16000:
            continue
        if info.frames < 3 * 16000:          # want real speech across the window
            continue
        out.append(f)
        if len(out) == n:
            break
    return out


def make_case_wav(path, flac, atten_db=0.0):
    """Write the 128,000-sample deployment window as 16-bit PCM."""
    wav, sr = sf.read(flac, dtype="float32")
    assert sr == 16000
    if atten_db:
        wav = wav * (10.0 ** (atten_db / 20.0))
    buf = np.zeros(NSAMPLES, dtype=np.float32)
    k = min(len(wav), NSAMPLES - LEAD_IN)
    buf[LEAD_IN:LEAD_IN + k] = wav[:k]
    sf.write(path, buf, 16000, subtype="PCM_16")


# ----------------------------------------------------------------------- oracle
def oracle_planes(pcm_i16):
    """model/fe.py on exactly the int16 the C side used."""
    x = pcm_i16.astype(np.float32) / 32768.0
    logmel = oracle.nemo_mel(x)[:, :T].astype(np.float32)      # (80, 800)
    norm = oracle.norm_pf(logmel).astype(np.float32)
    q = np.clip(np.round(norm / SCALE), -128, 127).astype(np.int8)
    return logmel, norm, q


def mel_energy(pcm_i16):
    """The 80x800 mel energies BEFORE the logarithm, from model/fe.py's own
    window and filterbank.  This is the quantity ST's library and NeMo treat
    differently."""
    import librosa
    x = pcm_i16.astype(np.float32) / 32768.0
    xp = np.concatenate([x[:1], x[1:] - oracle.PREEMPH * x[:-1]])
    S = librosa.stft(xp, n_fft=oracle.N_FFT, hop_length=oracle.HOP,
                     win_length=oracle.WIN, window=oracle._w,
                     center=True, pad_mode="constant")
    P = (np.abs(S) ** 2.0).astype(np.float32)
    return (oracle._fb @ P)[:, :T]


def oracle_guard_count(pcm_i16):
    """How many of the 64,000 mel energies were below 2^-24 before the log."""
    return int((mel_energy(pcm_i16) < GUARD).sum())


FLT_MIN = np.float32(1.1754943508222875e-38)
F16_MIN_SUBNORMAL = 5.960464477539063e-08


def st_library_ablation(pcm_i16):
    """What ST's LogMelSpectrogramColumn_q15_Q8 would have produced instead.

    ST clamps (feature_extraction.c:293-298, 315-317)      if (x <= 0) x = FLT_MIN;
    NeMo adds (model/fe.py:12)                             log(x + 2**-24)

    Same everything else — same window, same filterbank, same normalisation —
    so this isolates the one line."""
    M = mel_energy(pcm_i16)

    def quant(logmel):
        n = oracle.norm_pf(logmel.astype(np.float32)).astype(np.float32)
        return np.clip(np.round(n / SCALE), -128, 127).astype(np.int8)

    nemo = np.log(M + np.float32(GUARD))
    st_f32 = np.log(np.where(M <= 0, FLT_MIN, M))

    q_nemo, q_st = quant(nemo), quant(st_f32)
    d = np.abs(q_nemo.astype(np.int32) - q_st.astype(np.int32))
    return dict(
        int8_diff=int((d != 0).sum()),
        int8_diff_frac=float((d != 0).mean()),
        int8_max_abs=int(d.max()),
        logmel_max_abs=float(np.abs(nemo - st_f32).max()),
        nemo_floor=float(nemo.min()), st_floor=float(st_f32.min()),
        below_guard=int((M < GUARD).sum()),
        exact_zero=int((M == 0).sum()),
        # the f16 path assigns FLT_MIN into a float16, which underflows to 0 and
        # makes the next line logf(0) = -inf
        below_f16_subnormal=int((M < F16_MIN_SUBNORMAL).sum()),
    )


# ------------------------------------------------------------------- comparison
def compare(binary, wav, prefix, extra_args=()):
    cmd = [binary, "--wav", wav, "--out", prefix, "--scale", repr(SCALE)] + list(extra_args)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode not in (0,):
        raise SystemExit("%s failed (%d):\n%s" % (binary, p.returncode, p.stderr))
    tel = json.loads(p.stdout.strip().splitlines()[-1])
    tel["report"] = p.stderr.strip()

    pcm = np.fromfile(prefix + ".pcm.s16", dtype=np.int16)
    c_lm = np.fromfile(prefix + ".logmel.f32", dtype=np.float32).reshape(NMEL, T)
    c_nm = np.fromfile(prefix + ".norm.f32", dtype=np.float32).reshape(NMEL, T)
    c_q = np.fromfile(prefix + ".int8", dtype=np.int8).reshape(NMEL, T)
    assert pcm.size == NSAMPLES

    p_lm, p_nm, p_q = oracle_planes(pcm)

    d_lm = np.abs(c_lm.astype(np.float64) - p_lm.astype(np.float64))
    d_nm = np.abs(c_nm.astype(np.float64) - p_nm.astype(np.float64))
    d_q = c_q.astype(np.int32) - p_q.astype(np.int32)
    ad = np.abs(d_q)

    # How close was the oracle to a rounding tie?  A tie is the one place where a
    # 1e-6 float difference is allowed to flip a whole int8 LSB, and it is also
    # where np.round (half-to-even) and C's roundf (half-away-from-zero) part
    # company.  exact_tie counts values sitting on x.5 exactly; near_tie counts
    # values within 1e-4 LSB of one.
    fracs = (p_nm.astype(np.float32) / np.float32(SCALE)).astype(np.float64)
    off = np.abs(np.abs(fracs) - np.floor(np.abs(fracs)) - 0.5)
    near_tie = int((off < 1e-4).sum())
    exact_tie = int((off == 0.0).sum())

    worst = np.unravel_index(np.argmax(ad), ad.shape) if ad.max() else (0, 0)

    res = dict(
        telemetry=tel,
        logmel_max_abs=float(d_lm.max()), logmel_mean_abs=float(d_lm.mean()),
        logmel_rel_max=float((d_lm / (np.abs(p_lm) + 1e-30)).max()),
        norm_max_abs=float(d_nm.max()), norm_mean_abs=float(d_nm.mean()),
        int8_total=int(ad.size),
        int8_diff=int((ad != 0).sum()),
        int8_diff_frac=float((ad != 0).mean()),
        int8_max_abs=int(ad.max()),
        int8_hist={str(k): int((ad == k).sum()) for k in range(0, min(int(ad.max()), 4) + 1)},
        int8_gt1=int((ad > 1).sum()),
        near_tie=near_tie, exact_tie=exact_tie,
        worst_at=[int(worst[0]), int(worst[1])],
        worst_norm_c=float(c_nm[worst]), worst_norm_py=float(p_nm[worst]),
        oracle_guard_below=oracle_guard_count(pcm),
        c_guard_below=int(tel["guard_below"]),
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="append", required=True,
                    help="a compiled test_fe_host; may be repeated")
    ap.add_argument("--n", type=int, default=5, help="utterances per binary")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-onnx", action="store_true",
                    help="skip the ONNX Runtime end-to-end decode")
    ap.add_argument("--quiet-db", type=float, default=-54.0,
                    help="attenuation for the gain-staging case")
    args = ap.parse_args()

    utts = pick_utterances(args.n)
    tmp = tempfile.mkdtemp(prefix="fe_parity_")
    report = {"scale": SCALE, "utterances": [os.path.basename(u) for u in utts],
              "binaries": {}}

    for binary in args.bin:
        name = os.path.basename(binary)
        rows = []
        print("\n=== %s ===" % name)
        for u in utts:
            key = os.path.basename(u).replace(".flac", "")
            wav = os.path.join(tmp, key + ".wav")
            make_case_wav(wav, u)
            r = compare(binary, wav, os.path.join(tmp, name + "_" + key))
            r["utt"] = key
            rows.append(r)
            print("  %-22s logmel max|d| %.3e   int8 differ %5d/%d (%.4f%%)  max|d| %d  "
                  "guard C %d / py %d"
                  % (key, r["logmel_max_abs"], r["int8_diff"], r["int8_total"],
                     100 * r["int8_diff_frac"], r["int8_max_abs"],
                     r["c_guard_below"], r["oracle_guard_below"]))

        agg = dict(
            logmel_max_abs=max(r["logmel_max_abs"] for r in rows),
            norm_max_abs=max(r["norm_max_abs"] for r in rows),
            int8_total=sum(r["int8_total"] for r in rows),
            int8_diff=sum(r["int8_diff"] for r in rows),
            int8_max_abs=max(r["int8_max_abs"] for r in rows),
            int8_gt1=sum(r["int8_gt1"] for r in rows),
            near_tie=sum(r["near_tie"] for r in rows),
            exact_tie=sum(r["exact_tie"] for r in rows),
            int8_hist={str(k): sum(r["int8_hist"].get(str(k), 0) for r in rows) for k in range(5)},
            guard_delta=max(abs(r["c_guard_below"] - r["oracle_guard_below"]) for r in rows),
            scratch_bytes=rows[0]["telemetry"]["scratch_bytes"],
            ctx_bytes=rows[0]["telemetry"]["ctx_bytes"],
            cmsis=rows[0]["telemetry"]["cmsis"],
            scratch_f16=rows[0]["telemetry"]["scratch_f16"],
            var_mode=rows[0]["telemetry"]["var_mode"],
            tab_fnv1a=rows[0]["telemetry"]["tab_fnv1a"],
        )
        agg["int8_diff_frac"] = agg["int8_diff"] / agg["int8_total"]
        print("  ---- %s: %d/%d int8 differ (%.5f%%), max |d| %d, >1 LSB: %d"
              % (name, agg["int8_diff"], agg["int8_total"], 100 * agg["int8_diff_frac"],
                 agg["int8_max_abs"], agg["int8_gt1"]))
        print("       log-mel max |d| %.3e, normalised max |d| %.3e, "
              "rounding ties exact %d / near %d, scratch %d B, ctx %d B"
              % (agg["logmel_max_abs"], agg["norm_max_abs"], agg["exact_tie"],
                 agg["near_tie"], agg["scratch_bytes"], agg["ctx_bytes"]))
        report["binaries"][name] = {"rows": rows, "aggregate": agg}

    # ---- gain-staging / log-guard refusal case, on the first binary only ----
    b0 = args.bin[0]
    key = os.path.basename(utts[0]).replace(".flac", "")
    print("\n=== log-guard refusal, %s at %.0f dB ===" % (key, args.quiet_db))
    guard_rows = []
    for label, atten, gain in (("native", 0.0, None),
                               ("quiet", args.quiet_db, None),
                               ("quiet+peaknorm", args.quiet_db, 0.9)):
        wav = os.path.join(tmp, "%s_%s.wav" % (key, label))
        make_case_wav(wav, utts[0], atten_db=atten)
        extra = ["--gain", str(gain)] if gain else []
        pref = os.path.join(tmp, "guard_" + label)
        r = compare(b0, wav, pref, extra)
        t = r["telemetry"]
        print("  %-16s guard %6.2f%%  usable=%d  rc=%d  peak %.4f  gain %.1f  "
              "logmel [%.2f, %.2f]  int8 differ %d"
              % (label, 100 * t["guard_frac"], t["usable"], t["rc"], t["input_peak"],
                 t["gain_applied"], t["logmel_min"], t["logmel_max"], r["int8_diff"]))
        guard_rows.append(dict(label=label, atten_db=atten, gain=gain, **r))
    report["guard_cases"] = guard_rows

    # ---- end to end: does the C tensor transcribe? ----
    onnx = os.path.join(REPO, "artifacts", "onnx", "q800_real.onnx")
    if os.path.exists(onnx) and not args.no_onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        print("\n=== greedy CTC on the C front end's own int8 tensor (%s) ==="
              % os.path.basename(onnx))
        e2e = []
        for u in utts[:5]:
            key = os.path.basename(u).replace(".flac", "")
            q = np.fromfile(os.path.join(tmp, os.path.basename(args.bin[0]) + "_" + key + ".int8"),
                            dtype=np.int8).reshape(NMEL, T)
            logits = sess.run(None, {iname: (q.astype(np.float32) * SCALE)[None]})[0]
            txt = oracle.greedy(logits[0])
            print("  %-22s %s" % (key, txt))
            e2e.append({"utt": key, "hyp": txt})
        report["end_to_end"] = e2e

    # ---- what ST's clamp would have cost, same audio, same everything else ----
    print("\n=== ST's clamp-to-FLT_MIN vs NeMo's +2^-24, same audio ===")
    ab = []
    for u in utts:
        key = os.path.basename(u).replace(".flac", "")
        pcm = np.fromfile(os.path.join(tmp, os.path.basename(args.bin[0]) + "_" + key + ".pcm.s16"),
                          dtype=np.int16)
        a = st_library_ablation(pcm)
        a["utt"] = key
        ab.append(a)
        print("  %-22s int8 differ %5d/64000 (%5.2f%%) max|d| %3d   floor NeMo %.2f / ST %.2f   "
              "below guard %5d, exact zero %5d, below fp16 subnormal %5d"
              % (key, a["int8_diff"], 100 * a["int8_diff_frac"], a["int8_max_abs"],
                 a["nemo_floor"], a["st_floor"], a["below_guard"], a["exact_zero"],
                 a["below_f16_subnormal"]))
    print("  ---- ST clamp changes %d of %d int8 values (%.2f%%), max |d| %d"
          % (sum(a["int8_diff"] for a in ab), 64000 * len(ab),
             100 * sum(a["int8_diff"] for a in ab) / (64000 * len(ab)),
             max(a["int8_max_abs"] for a in ab)))
    report["st_clamp_ablation"] = ab

    # ---- determinism: same binary, same input, byte-identical output ----
    wav = os.path.join(tmp, os.path.basename(utts[0]).replace(".flac", "") + ".wav")
    for tag in ("rep1", "rep2"):
        subprocess.run([b0, "--wav", wav, "--out", os.path.join(tmp, tag),
                        "--scale", repr(SCALE)], capture_output=True, check=True)
    a = open(os.path.join(tmp, "rep1.int8"), "rb").read()
    b = open(os.path.join(tmp, "rep2.int8"), "rb").read()
    report["deterministic"] = (a == b)
    print("\ndeterminism: two runs of %s produce %s int8 output"
          % (os.path.basename(b0), "identical" if a == b else "DIFFERENT"))

    if args.json:
        json.dump(report, open(args.json, "w"), indent=1)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

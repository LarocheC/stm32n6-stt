#!/usr/bin/env python3
"""Emit artifacts/corpus/corpus_blob.bin + corpus_ref.json — N int8 feature
tensors for the device, and the host ONNX Runtime reference for each.

Gate 4 closed the execution question (448 epochs, 124.0 ms, deterministic) but
left one open: the device disagrees with host onnxruntime on 5 of 100 frames of
the single canned utterance.  Four word errors on 17 words is a Wilson interval
of [9.6 %, 47.3 %] — it contains both "as good as the host" and "much worse".
This script builds the corpus that turns that into a number: device WER versus
host WER over N utterances.

    python firmware/tools/gen_corpus.py [-n N] [--seed S] [--no-anchor]

Run from the repo root with the zoo venv.

BLOB CONTRACT — the firmware is written against this; do not change it.
Flashed to external flash at 0x71000000 (the weight blob ends ~0x70D9C000).

    offset  size      content
    0x00    4         magic, ASCII "STTC"
    0x04    4         uint32 LE  N     (number of utterances)
    0x08    4         uint32 LE  T     (frames per utterance, 800)
    0x0C    4         uint32 LE  NMEL  (mel bins, 80)
    0x10    48        zero padding
    0x40    N*64000   N int8 feature tensors, back to back

Each tensor is 64,000 B, MEL-MAJOR (index = mel*800 + frame), identical in
layout and quantisation to kCannedFeatures in firmware/inc/canned_features.h.
Utterance i starts at 0x40 + i*64000.
"""
import argparse, hashlib, json, os, struct, sys
import numpy as np, soundfile as sf, onnxruntime as ort

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCALE   = 0.120522417128086       # compile/reports/g800_st_autosched/io_contract.h
T, NMEL, NVOCAB, BLANK = 800, 80, 1025, 1024
LEAD_IN = 4800                    # samples of silence before speech, as eval/ does
MAX_DUR = 7.9                     # "fits the 8 s window", per the task brief
MAGIC   = b"STTC"
HDR     = 0x40
TENSOR  = NMEL * T                # 64000 B

# The deployed graph.  q800_fold.onnx and q800_real.onnx give bit-identical
# argmax; this is the one the board is actually running (GATE4.md Round 19).
MODEL   = "artifacts/onnx/q800_relu4d_all.onnx"

# recs.json's 'f' paths are stale — they name the tts repo, which does not exist.
STALE_PREFIX = "/home/claroche/stm32n6-tts/corpus/"
LIVE_PREFIX  = os.path.join(REPO, "corpus") + "/"

# Anchor: the utterance already canned in firmware/inc/canned_features.h.  Pinned
# at index 0 so the device's corpus run can be cross-checked, frame for frame,
# against the 5/100 result in GATE4.md Round 19 before anything else is believed.
ANCHOR = "1272-128104-0000"

DEFAULT_SEED = 20260819


# ------------------------------------------------------------------ front end
def load_fe():
    """model/fe.py is the authoritative spec.  Load it exactly the way
    firmware/tools/gen_canned_features.py does — that script carried a
    replacement for a hardcoded scratchpad vocab path; fe.py now resolves the
    vocab from its own location, so the replacement is a no-op, but it is kept
    so the two loaders cannot drift."""
    path = os.path.join(REPO, "model", "fe.py")
    src = open(path).read()
    src = src.replace(
        "/tmp/claude-1000/-home-claroche-stm32n6-tts/"
        "f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/citrinet/vocab.txt",
        os.path.join(REPO, "tokenizer", "vocab.txt"))
    fe = type(sys)("fe"); fe.__file__ = path
    exec(compile(src, "fe.py", "exec"), fe.__dict__)
    return fe


def features_int8(fe, flac_path):
    """int8 tensor for one utterance, byte-for-byte as gen_canned_features.py
    produces kCannedFeatures (that script, lines 37-48 and 65)."""
    wav, sr = sf.read(flac_path)
    assert sr == 16000, (flac_path, sr)
    wav = wav.astype(np.float32)

    nwin = (T - 1) * 160 + 1
    buf = np.zeros(nwin, dtype=np.float32)
    n = min(len(wav), nwin - LEAD_IN)
    buf[LEAD_IN:LEAD_IN + n] = wav[:n]

    mel = fe.norm_pf(fe.nemo_mel(buf))                     # (80, 800) float32
    assert mel.shape == (NMEL, T), mel.shape
    q = np.clip(np.round(mel / SCALE), -128, 127).astype(np.int8)
    flat = q.ravel()                                       # mel-major
    assert flat.size == TENSOR
    return q, flat, (len(wav) > nwin - LEAD_IN)


# ------------------------------------------------------------------- decoding
def load_vocab():
    p = os.path.join(REPO, "tokenizer", "vocab.txt")
    return [l.rsplit(" ", 1)[0] for l in open(p, encoding="utf-8").read().split("\n") if l.strip()]


def ctc_decode(ids, vocab):
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK:
            out.append(vocab[i])
        prev = i
    return "".join(out).replace("▁", " ").strip()


# ---------------------------------------------------- calibration exclusion
def calibration_keys(recs):
    """Rebuild the calibration sets from eval/sets.py — the canonical
    reconstruction checked by eval/check_disjoint.py (Gate 1 Part A).

    cal_800 is the set that calibrated the SHIPPED 8 s graph q800_real.onnx,
    from which q800_fold / q800_relu4d_all are value-preserving rewrites
    (GATE4.md Round 19: max|diff| = 0 over 3,075,000 elements).  It is excluded
    unconditionally.  The union of all three calibration sets is excluded too:
    cal_400 and cal_1200 did not touch this graph, so this is belt-and-braces,
    and it costs 94 of 1,789 candidates."""
    sys.path.insert(0, os.path.join(REPO, "eval"))
    import sets as S                                       # eval/sets.py
    cal800 = {r["k"] for r in S.cal_800(recs)}
    union = set()
    for name, f in S.CAL_SETS.items():
        union |= {r["k"] for r in f(recs)}
    return cal800, union


# ------------------------------------------------------------------ the build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=64, help="number of utterances")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-anchor", action="store_true",
                    help="do not pin %s at index 0" % ANCHOR)
    ap.add_argument("--outdir", default=os.path.join(REPO, "artifacts", "corpus"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fe = load_fe()
    vocab = load_vocab()

    recs = json.load(open(os.path.join(REPO, "eval", "results", "recs.json")))
    by_key = {r["k"]: r for r in recs}
    cal800, cal_union = calibration_keys(recs)

    # ---- selection ---------------------------------------------------------
    pool_all = [r for r in recs if r["d"] <= MAX_DUR]
    pool = [r for r in pool_all if r["k"] not in cal_union]
    n_excl_800 = sum(1 for r in pool_all if r["k"] in cal800)
    n_excl_union = len(pool_all) - len(pool)

    chosen = []
    if not args.no_anchor:
        assert ANCHOR not in cal_union, "anchor is a calibration utterance"
        chosen.append(by_key[ANCHOR])
        pool = [r for r in pool if r["k"] != ANCHOR]

    need = args.n - len(chosen)
    perm = np.random.default_rng(args.seed).permutation(len(pool))
    chosen += [pool[i] for i in perm[:need]]
    assert len(chosen) == args.n
    assert len({r["k"] for r in chosen}) == args.n, "duplicate key selected"

    # ---- host reference ----------------------------------------------------
    # ORT_ENABLE_ALL is onnxruntime's ACTUAL default -- the level you get when no
    # SessionOptions is passed at all -- and it is what artifacts/corpus/
    # corpus_ref.json's host_ids were produced at.  An earlier revision of this
    # file set ORT_ENABLE_BASIC and labelled it "the default"; it was neither, and
    # a rerun would have reproduced corpus_blob.bin byte for byte (the features do
    # not go through onnxruntime) while disagreeing with its own sidecar's ids on
    # 9 of 16 utterances.  See firmware/tools/gen_wav_corpus.py, which measured
    # the level sweep: ALL 16/16, EXTENDED 16/16, BASIC 7/16, DISABLE_ALL 7/16.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(REPO, MODEL), so,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    tensors, utts, truncated = [], [], []
    for i, r in enumerate(chosen):
        path = r["f"].replace(STALE_PREFIX, LIVE_PREFIX)
        if not os.path.exists(path):
            sys.exit("missing audio: %s (from %s)" % (path, r["f"]))
        q, flat, trunc = features_int8(fe, path)
        if trunc:
            truncated.append(r["k"])
        tensors.append(flat)

        # Feed the DEQUANTISED int8 so host and device see identical values.
        logits = sess.run(None, {iname: (q.astype(np.float32) * SCALE)[None]})[0]
        lg = logits.reshape(-1, NVOCAB)
        ids = lg.argmax(-1).astype(np.int32)
        part = np.partition(lg, -2, axis=-1)
        margin = (part[:, -1] - part[:, -2]).astype(np.float64)
        text = ctc_decode(ids, vocab)

        utts.append({
            "index": i,
            "key": r["k"],
            "duration_s": r["d"],
            "n_words_ref": r["nw"],
            "audio": path,
            "ref": r["ref"],
            "host_ids": [int(v) for v in ids],
            "host_text": text,
            "host_margin": [round(float(v), 6) for v in margin],
            "int8_min": int(q.min()), "int8_max": int(q.max()),
        })
        print("[%3d] %-22s %5.2fs  %s" % (i, r["k"], r["d"], text), flush=True)

    # ---- blob --------------------------------------------------------------
    hdr = MAGIC + struct.pack("<III", args.n, T, NMEL) + b"\0" * 48
    assert len(hdr) == HDR
    blob = bytearray(hdr)
    for flat in tensors:
        blob += flat.tobytes()
    assert len(blob) == HDR + args.n * TENSOR

    blob_path = os.path.join(args.outdir, "corpus_blob.bin")
    open(blob_path, "wb").write(bytes(blob))
    md5 = hashlib.md5(bytes(blob)).hexdigest()

    # ---- self-check: index 0 must equal kCannedFeatures byte for byte ------
    canned_check = check_canned(tensors, chosen)

    side = {
        "generated_by": "firmware/tools/gen_corpus.py",
        "model": MODEL,
        "ort_graph_optimization_level": "ORT_ENABLE_ALL (onnxruntime's actual default; matches corpus_ref.json)",
        "onnxruntime": ort.__version__,
        "scale": SCALE, "T": T, "NMEL": NMEL, "NVOCAB": NVOCAB, "BLANK": BLANK,
        "lead_in_samples": LEAD_IN,
        "layout": "mel-major, index = mel*800 + frame",
        "N": args.n,
        "seed": args.seed,
        "anchor_index0": None if args.no_anchor else ANCHOR,
        "selection": {
            "filter": "d <= %.1f" % MAX_DUR,
            "pool_before_exclusion": len(pool_all),
            "excluded_cal_800": n_excl_800,
            "excluded_cal_union": n_excl_union,
            "pool_after_exclusion": len(pool_all) - n_excl_union,
            "rule": ("index 0 = anchor %s (already canned in "
                     "firmware/inc/canned_features.h); indices 1..N-1 = "
                     "np.random.default_rng(seed).permutation(pool)[:N-1] over the "
                     "calibration-free pool with the anchor removed"
                     % ANCHOR) if not args.no_anchor else
                    "np.random.default_rng(seed).permutation(pool)[:N]",
            "cal_800_keys": sorted(cal800),
            "cal_union_keys": sorted(cal_union),
        },
        "keys": [r["k"] for r in chosen],
        "blob": {
            "path": os.path.relpath(blob_path, REPO),
            "bytes": len(blob),
            "md5": md5,
            "header_bytes": HDR,
            "tensor_bytes": TENSOR,
            "flash_address": "0x71000000",
        },
        "canned_byte_identity_check": canned_check,
        "truncated_utterances": truncated,
        "utterances": utts,
    }
    ref_path = os.path.join(args.outdir, "corpus_ref.json")
    json.dump(side, open(ref_path, "w"), indent=1)

    print()
    print("blob   %s" % blob_path)
    print("bytes  %d  (0x40 header + %d * 64000)" % (len(blob), args.n))
    print("md5    %s" % md5)
    print("side   %s" % ref_path)
    print("canned byte-identity: %s" % canned_check["result"])
    print("excluded %d cal_800 keys (%d for the union of all three cal sets) "
          "from a %d-utterance pool" % (n_excl_800, n_excl_union, len(pool_all)))
    if truncated:
        print("WARNING: %d utterance(s) truncated by the window: %s"
              % (len(truncated), truncated))
    print()
    print("flash it with:")
    print("  STM32_Programmer_CLI -c port=SWD mode=UR --extload <loader> "
          "-w %s 0x71000000" % blob_path)


def check_canned(tensors, chosen):
    """Byte-compare whichever tensor is the anchor against kCannedFeatures in
    firmware/inc/canned_features.h.  Everything downstream depends on the device
    and the host seeing identical features, and this is the one place where a
    known-good byte string exists to check against."""
    hdr = os.path.join(REPO, "firmware", "inc", "canned_features.h")
    idx = next((i for i, r in enumerate(chosen) if r["k"] == ANCHOR), None)
    if idx is None:
        return {"result": "SKIPPED — anchor %s not in selection" % ANCHOR}
    txt = open(hdr).read()
    body = txt.split("kCannedFeatures[64000] = {", 1)[1].split("};", 1)[0]
    ref = np.array([int(v) for v in body.replace("\n", "").split(",") if v.strip()],
                   dtype=np.int8)
    if ref.size != TENSOR:
        return {"result": "FAIL — parsed %d values from the header" % ref.size}
    ours = tensors[idx]
    ndiff = int((ref != ours).sum())
    return {
        "result": "PASS — 64000/64000 bytes identical" if ndiff == 0
                  else "FAIL — %d of 64000 bytes differ" % ndiff,
        "header": os.path.relpath(hdr, REPO),
        "utterance": ANCHOR, "index": idx,
        "bytes_compared": TENSOR, "bytes_differing": ndiff,
        "md5_header": hashlib.md5(ref.tobytes()).hexdigest(),
        "md5_generated": hashlib.md5(ours.tobytes()).hexdigest(),
    }


if __name__ == "__main__":
    main()
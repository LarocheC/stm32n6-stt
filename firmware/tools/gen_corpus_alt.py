#!/usr/bin/env python3
"""FALLBACK corpus-blob builder.  artifacts/corpus/corpus_ref.json + corpus_blob.bin,
produced by the parallel blob task, are AUTHORITATIVE -- they additionally exclude
the int8 calibration utterances and byte-check index 0 against
firmware/inc/canned_features.h.  This file was written by the scorer task before
those artifacts appeared; it was originally at firmware/tools/gen_corpus.py and may
have overwritten the parallel task's generator at that path.  It is kept only so the
blob remains reproducible if that source was lost, and it writes to corpus_alt.* so
it can never be confused with the flashed blob.

Build the multi-utterance corpus blob for the board plus the host sidecar.

Blob layout (the contract the firmware reads, flashed at 0x71000000):

    offset  size      content
    0x00    4         magic, ASCII "STTC"
    0x04    4         uint32 LE  N     (number of utterances)
    0x08    4         uint32 LE  T     (frames per utterance, 800)
    0x0C    4         uint32 LE  NMEL  (mel bins, 80)
    0x10    48        zero padding
    0x40    N*64000   N int8 feature tensors, back to back

Each tensor is MEL-MAJOR (index = mel*800 + frame), quantised with scale
0.120522417128086, round-to-nearest, clipped to [-128,127] -- identical in
layout and quantisation to kCannedFeatures in firmware/inc/canned_features.h.
Preprocessing follows firmware/tools/gen_canned_features.py exactly (4800-sample
lead-in of silence, fe.norm_pf(fe.nemo_mel(...))), so device and host see the
same numbers.

The sidecar is the host's answer key: per utterance the reference transcript,
the 100 host argmax ids, the top-2 id and the top1-top2 logit margin per frame.
firmware/test/score_corpus.py scores a UART capture against it.

    python firmware/tools/gen_corpus.py --n 64 --out artifacts/corpus

Run with the zoo venv: /home/claroche/stm32n6-deployment-zoo/.venv/bin/python
"""
import argparse, hashlib, json, os, random, struct, sys
import numpy as np, soundfile as sf, onnxruntime as ort

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCALE = 0.120522417128086          # compile/reports/g800_st_autosched/io_contract.h
T, NMEL, NVOCAB, BLANK = 800, 80, 1025, 1024
LEAD_IN = 4800                     # samples of silence before speech, as eval/ does
TENSOR_BYTES = NMEL * T            # 64000
HDR_BYTES = 0x40
CANNED_KEY = "1272-128104-0000"    # the utterance Gate 4 has been running; keep it at index 0

# recs.json's 'f' paths name a sibling repo that does not exist here.
STALE_PREFIX, LIVE_PREFIX = "/home/claroche/stm32n6-tts/", REPO + "/"


def load_fe():
    """model/fe.py is the authoritative front end. gen_canned_features.py patches a
    hardcoded scratchpad vocab path into it; that path is gone from the current
    fe.py (it uses _REPO), but the replace is kept so both scripts stay in step."""
    src = open(f"{REPO}/model/fe.py").read()
    fe = type(sys)("fe"); fe.__file__ = f"{REPO}/model/fe.py"
    exec(compile(src, "fe.py", "exec"), fe.__dict__)
    return fe


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def features(fe, wav_path):
    wav, sr = sf.read(wav_path)
    if sr != 16000:
        raise SystemExit(f"{wav_path}: sample rate {sr}, expected 16000")
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(1)
    nwin = (T - 1) * 160 + 1
    buf = np.zeros(nwin, dtype=np.float32)
    n = min(len(wav), nwin - LEAD_IN)
    buf[LEAD_IN:LEAD_IN + n] = wav[:n]
    mel = fe.norm_pf(fe.nemo_mel(buf))
    assert mel.shape == (NMEL, T), mel.shape
    q = np.clip(np.round(mel / SCALE), -128, 127).astype(np.int8)
    truncated = len(wav) > nwin - LEAD_IN
    return q, truncated, len(wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="number of utterances")
    ap.add_argument("--max-dur", type=float, default=7.69,   # (799*160+1-4800)/16000 = 7.69006 s
                    help="skip utterances longer than this (they would be truncated)")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--model", default="artifacts/onnx/q800_relu4d_all.onnx",
                    help="the DEPLOYED graph, relative to repo root")
    ap.add_argument("--out", default="artifacts/corpus")
    args = ap.parse_args()

    outdir = os.path.join(REPO, args.out)
    os.makedirs(outdir, exist_ok=True)
    model = os.path.join(REPO, args.model)

    fe = load_fe()
    vocab = [l.rsplit(" ", 1)[0] for l in
             open(f"{REPO}/tokenizer/vocab.txt", encoding="utf-8").read().split("\n") if l.strip()]
    assert len(vocab) == NVOCAB and vocab[BLANK] == "<blk>", (len(vocab), vocab[BLANK])

    recs = json.load(open(f"{REPO}/eval/results/recs.json"))
    for r in recs:
        r["f"] = r["f"].replace(STALE_PREFIX, LIVE_PREFIX)
    by_key = {r["k"]: r for r in recs}

    elig = sorted((r for r in recs
                   if r["d"] <= args.max_dur and r["nw"] >= args.min_words
                   and os.path.exists(r["f"])), key=lambda r: r["k"])
    print(f"eligible: {len(elig)} of {len(recs)} records "
          f"(d <= {args.max_dur}, nw >= {args.min_words}, file present)")
    if CANNED_KEY not in by_key:
        raise SystemExit("canned utterance missing from recs.json")

    rng = random.Random(args.seed)
    pool = [r for r in elig if r["k"] != CANNED_KEY]
    rng.shuffle(pool)
    chosen = [by_key[CANNED_KEY]] + pool[:args.n - 1]
    if len(chosen) < args.n:
        raise SystemExit(f"only {len(chosen)} eligible utterances, asked for {args.n}")

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    tensors, utts = [], []
    for i, r in enumerate(chosen):
        q, truncated, nsamp = features(fe, r["f"])
        flat = q.ravel()
        assert flat.size == TENSOR_BYTES
        logits = sess.run(None, {inp: (q.astype(np.float32) * SCALE)[None]})[0].reshape(-1, NVOCAB)
        order = np.argsort(-logits, axis=-1)
        ids = order[:, 0].astype(np.int32)
        top2 = order[:, 1].astype(np.int32)
        margin = (logits[np.arange(len(ids)), ids] - logits[np.arange(len(ids)), top2]).astype(np.float64)
        collapsed, prev = [], -1
        for t in ids:
            if t != prev and t != BLANK:
                collapsed.append(int(t))
            prev = int(t)
        text = "".join(vocab[t] for t in collapsed).replace("▁", " ").strip()
        utts.append(dict(i=i, k=r["k"], f=r["f"], d=r["d"], nw=r["nw"], ref=r["ref"],
                         n_samples=int(nsamp), truncated=bool(truncated),
                         sha256=hashlib.sha256(flat.tobytes()).hexdigest(),
                         int8_min=int(flat.min()), int8_max=int(flat.max()),
                         host_ids=[int(x) for x in ids],
                         host_top2=[int(x) for x in top2],
                         host_margin=[round(float(x), 6) for x in margin],
                         host_text=text))
        tensors.append(flat.tobytes())
        if truncated:
            print(f"  [{i:3d}] {r['k']} TRUNCATED ({r['d']:.2f}s)")
        if i % 8 == 0:
            print(f"  [{i:3d}] {r['k']}  {r['d']:.2f}s  {text[:60]}")

    n_out = len(utts[0]["host_ids"])
    assert all(len(u["host_ids"]) == n_out for u in utts)

    blob = bytearray()
    blob += b"STTC" + struct.pack("<III", len(utts), T, NMEL) + b"\x00" * 48
    assert len(blob) == HDR_BYTES
    for t in tensors:
        blob += t
    blob_path = os.path.join(outdir, "corpus_alt.bin")
    open(blob_path, "wb").write(bytes(blob))

    side = dict(magic="STTC-sidecar", version=1,
                repo=REPO, model=os.path.relpath(model, REPO), model_sha256=sha256_file(model),
                vocab_sha256=sha256_file(f"{REPO}/tokenizer/vocab.txt"),
                scale=SCALE, T=T, NMEL=NMEL, NVOCAB=NVOCAB, blank=BLANK, lead_in=LEAD_IN,
                N=len(utts), n_out_frames=n_out, tensor_bytes=TENSOR_BYTES,
                header_bytes=HDR_BYTES, flash_addr="0x71000000",
                seed=args.seed, max_dur=args.max_dur, min_words=args.min_words,
                blob=os.path.relpath(blob_path, REPO),
                blob_sha256=hashlib.sha256(bytes(blob)).hexdigest(),
                blob_bytes=len(blob), utts=utts)
    side_path = os.path.join(outdir, "corpus_alt_sidecar.json")
    json.dump(side, open(side_path, "w"), indent=1)

    print(f"\nwrote {blob_path}  {len(blob)} bytes  sha256 {side['blob_sha256'][:16]}")
    print(f"wrote {side_path}  N={len(utts)}  {n_out} frames/utt  "
          f"{sum(u['nw'] for u in utts)} reference words")
    print(f"flash: 0x71000000 .. 0x{0x71000000 + len(blob):08X}")


if __name__ == "__main__":
    main()

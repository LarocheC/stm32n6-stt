#!/usr/bin/env python3
"""Emit artifacts/corpus/wav_blob.bin + wav_ref.json -- N int16 WAVEFORMS for the
device, so the M55 front end can be brought up on canned audio.

Gate 4 closed the NPU question.  Gate 5 -- firmware/src/citrinet_fe.c, the log-mel
front end -- reproduces model/fe.py exactly on a workstation (0 of 768,000 int8
values differ, firmware/FRONTEND.md section 5) but has never run on the M55, and
its cost there is the largest unmeasured number in the project.  This script
builds the input for that bring-up: waveform -> M55 front end -> NPU -> C CTC
decoder, in one image, with no microphone and no gain stage involved.

    python firmware/tools/gen_wav_corpus.py [-n N]

Run from the repo root with the zoo venv.

WHY THE UTTERANCES ARE NOT RE-SELECTED
--------------------------------------
The keys, and their order, are read verbatim from artifacts/corpus/corpus_ref.json
-- the sidecar of the existing 64-utterance FEATURE corpus, which is already
flashed at 0x71000000.  We take its first N entries.  So utterance i of THIS blob
is the same recording as index i of that one, and the features the device computes
on the M55 can be compared byte for byte against a host feature tensor that is
already on record (artifacts/corpus/corpus_blob.bin, offset 0x40 + i*64000) and
already sitting in the board's flash.  Re-running a selection here would have
broken that correspondence for no gain.  Every downstream exclusion -- the 48
cal_800 calibration keys, the union of all three calibration sets, the d <= 7.9 s
filter -- is therefore inherited from gen_corpus.py rather than re-derived.

AUDIO PREPARATION -- identical to gen_corpus.py:features_int8(), lines 81-89
---------------------------------------------------------------------------
    wav, sr = sf.read(flac)          float64 in [-1,1), sr == 16000
    wav = wav.astype(np.float32)
    nwin = (800-1)*160 + 1 = 127841  the sample count gen_corpus.py fed to fe.py
    n    = min(len(wav), nwin - 4800)  =  min(len(wav), 123041)
    buf[4800 : 4800+n] = wav[:n]     everything else zero

The blob carries 128,000 samples per utterance, not 127,841, because that is what
CITRINET_FE_NSAMPLES is (firmware/inc/citrinet_fe.h:51) and what the device will
hand citrinet_fe_run().  For an utterance that is NOT truncated the extra 159
samples are zero, zero is what librosa's centre padding supplied there on the
host, and the 800 frames are bit-identical.  For a TRUNCATED utterance they are
not, and this is measured, not assumed -- see WINDOW DRIFT below.  The TRUNCATION LIMIT stays 123041, NOT 123200: raising it would feed
frames 798-799 real audio the host never saw, and utterance 11 of 16
(1272-141231-0012) is long enough for that to matter.

int16 conversion is exact, not approximate.  LibriSpeech FLAC is 16-bit, so
soundfile's float64 sample is k/32768 for an integer k, which float32 represents
exactly; k = rint(f*32768) recovers it, and the device's
(float)pcm[i] * (1.0f/32768.0f) (citrinet_fe.c:227) reconstructs the same float32.
The script asserts the round trip rather than trusting it.

BLOB CONTRACT -- the firmware is written against this; do not change it.
Flashed to external flash at 0x72000000.  Weights occupy 0x70400000..~0x70D9C000
and the feature corpus 0x71000000 + 4,096,064 B, so 0x72000000 collides with
neither.

    offset  size        content
    0x00    4           magic, ASCII "STTW"
    0x04    4           uint32 LE  N          (number of utterances)
    0x08    4           uint32 LE  NSAMPLES   (samples per utterance, 128000)
    0x0C    4           uint32 LE  0          (reserved)
    0x10    48          zero padding
    0x40    N*256000    N int16 little-endian PCM waveforms, back to back

Utterance i starts at 0x40 + i*256000.  Per utterance the device prints

    # w <i> fe <cycles> npu <cycles> rc <n> guard <n> hash <8 hex> ids: <100 ints>

and "# wav done" at the end.  'hash' is FNV-1a 32-bit over the 64,000-byte int8
feature tensor the DEVICE computed; wav_ref.json carries the host's hash of the
same tensor, computed the same way, for direct comparison.
"""
import argparse, hashlib, json, os, struct, sys
import numpy as np, soundfile as sf, onnxruntime as ort

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCALE   = 0.120522417128086       # compile/reports/g800_st_autosched/io_contract.h
T, NMEL, NVOCAB, BLANK = 800, 80, 1025, 1024
LEAD_IN   = 4800                  # gen_corpus.py:38
NWIN      = (T - 1) * 160 + 1     # 127841 -- gen_corpus.py:85
NSAMPLES  = T * 160               # 128000 -- CITRINET_FE_NSAMPLES
KEEP_MAX  = NWIN - LEAD_IN        # 123041 -- gen_corpus.py:87
MAGIC     = b"STTW"
HDR       = 0x40
WAV_BYTES = NSAMPLES * 2          # 256000
TENSOR    = NMEL * T              # 64000
LOG_GUARD = 2.0 ** -24            # 5.9604644775390625e-8, CITRINET_FE_LOG_GUARD
GUARD_MAX_FRAC = 0.50             # CITRINET_FE_GUARD_MAX_FRAC, citrinet_fe.h
FLASH_ADDR = "0x72000000"

MODEL = "artifacts/onnx/q800_relu4d_all.onnx"   # the deployed graph, GATE4.md Round 19

FEAT_REF  = os.path.join(REPO, "artifacts", "corpus", "corpus_ref.json")
FEAT_BLOB = os.path.join(REPO, "artifacts", "corpus", "corpus_blob.bin")


# ------------------------------------------------------------------ front end
def load_fe():
    """model/fe.py is the authoritative spec.  Loaded exactly the way
    gen_corpus.py:load_fe() loads it, so the two cannot drift."""
    path = os.path.join(REPO, "model", "fe.py")
    src = open(path).read()
    fe = type(sys)("fe"); fe.__file__ = path
    exec(compile(src, "fe.py", "exec"), fe.__dict__)
    return fe


def prepare_int16(flac_path):
    """The 128,000-sample int16 waveform for one utterance.

    Returns (pcm int16[128000], n_kept, n_source, truncated)."""
    wav, sr = sf.read(flac_path)
    assert sr == 16000, (flac_path, sr)
    wav = wav.astype(np.float32)

    buf = np.zeros(NSAMPLES, dtype=np.float32)
    n = min(len(wav), KEEP_MAX)
    buf[LEAD_IN:LEAD_IN + n] = wav[:n]

    # Exact, not nearly-exact: 16-bit source, so every sample is k/32768.
    k = np.rint(buf.astype(np.float64) * 32768.0)
    assert k.min() >= -32768.0 and k.max() <= 32767.0, (flac_path, k.min(), k.max())
    pcm = k.astype(np.int16)
    back = (pcm.astype(np.float64) / 32768.0).astype(np.float32)
    assert np.array_equal(back, buf), "int16 round trip is lossy for %s" % flac_path
    return pcm, n, len(wav), len(wav) > KEEP_MAX


def features_int8(fe, pcm):
    """int8 tensor the DEVICE should produce from this waveform.

    Dequantise exactly as citrinet_fe.c:227 does, then run the oracle.  801 whole
    frames fit 128,000 samples; the graph is frozen to the first 800 and the
    per-bin statistics are taken over those 800 only -- which is what
    citrinet_fe_run() does (it loops t in 0..799 before citrinet_fe_finish()
    computes any mean).  Truncating AFTER norm_pf would normalise over 801 and is
    the obvious way to get this wrong."""
    x = (pcm.astype(np.float64) / 32768.0).astype(np.float32)
    lm_full = fe.nemo_mel(x)                       # (80, 801) float64
    assert lm_full.shape == (NMEL, T + 1), lm_full.shape
    mel = fe.norm_pf(lm_full[:, :T])
    q = np.clip(np.round(mel / SCALE), -128, 127).astype(np.int8)
    return q, q.ravel()                            # mel-major, index = mel*800 + frame


def features_int8_nwin(fe, pcm):
    """The same tensor computed the way gen_corpus.py computed it: from the
    127,841-sample buffer, which is what corpus_blob.bin holds.  Kept so the two
    references can be compared without re-deriving either."""
    x = (pcm[:NWIN].astype(np.float64) / 32768.0).astype(np.float32)
    lm = fe.nemo_mel(x)
    assert lm.shape == (NMEL, T), lm.shape
    q = np.clip(np.round(fe.norm_pf(lm) / SCALE), -128, 127).astype(np.int8)
    return q.ravel()


def guard_counts(fe, pcm):
    """Host counterpart of citrinet_fe.c:273-274 -- the mel energies, BEFORE the
    log, that fall below the guard, and those that are exactly zero.

    fe.nemo_mel() folds the +2^-24 and the ln in one expression, so the energies
    are recomputed here from the same float32 pieces fe.py uses (_fb float32,
    P float32) to keep the comparison honest.  The device accumulates 500 sparse
    MACs in a fixed order and this uses BLAS, so a bin sitting within an ulp of
    the guard may be counted differently; the zero count is exact, because a mel
    energy is exactly 0.0 only where the whole power column is."""
    import librosa
    x = (pcm.astype(np.float64) / 32768.0).astype(np.float32)
    xp = np.concatenate([x[:1], x[1:] - fe.PREEMPH * x[:-1]])
    S = librosa.stft(xp, n_fft=fe.N_FFT, hop_length=fe.HOP, win_length=fe.WIN,
                     window=fe._w, center=True, pad_mode="constant")
    P = (np.abs(S) ** 2.0).astype(np.float32)[:, :T]
    E = (fe._fb @ P).astype(np.float32)            # (80, 800) mel energies
    below = int((E < np.float32(LOG_GUARD)).sum())
    zero = int((E == np.float32(0.0)).sum())
    total = E.size
    # citrinet_fe_guard_fraction(): exactly-zero bins are excluded from both terms.
    if zero <= below and zero < total:
        frac = (below - zero) / float(total - zero)
    else:
        frac = below / float(total)
    return below, zero, total, frac


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


def fnv1a32(b):
    """FNV-1a 32-bit, the parameters the device uses: offset basis 2166136261,
    prime 16777619, over unsigned bytes."""
    h = 2166136261
    for v in b:
        h = ((h ^ v) * 16777619) & 0xFFFFFFFF
    return h


# ------------------------------------------------------------------ the build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=16,
                    help="number of utterances, taken as the FIRST n of the feature corpus")
    ap.add_argument("--allow-window-drift", action="store_true",
                    help="emit the blob even if a truncated utterance's 128,000-sample "
                         "features differ from corpus_blob.bin (see WINDOW DRIFT)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "artifacts", "corpus"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fe = load_fe()
    vocab = load_vocab()

    feat = json.load(open(FEAT_REF))
    assert feat["T"] == T and feat["NMEL"] == NMEL and feat["scale"] == SCALE
    assert feat["lead_in_samples"] == LEAD_IN
    if args.n > feat["N"]:
        sys.exit("feature corpus has only %d utterances" % feat["N"])
    chosen = feat["utterances"][:args.n]
    feat_bytes = open(FEAT_BLOB, "rb").read()
    assert len(feat_bytes) == 0x40 + feat["N"] * TENSOR, len(feat_bytes)

    # ORT_ENABLE_ALL, and not by accident.  corpus_ref.json records
    # "ORT_ENABLE_ALL (onnxruntime default, no SessionOptions passed)" -- that is
    # what produced the host_ids the device is scored against.  The copy of
    # gen_corpus.py currently on disk sets ORT_ENABLE_BASIC and labels it "the
    # onnxruntime default", which it is not; at BASIC this graph disagrees with the
    # shipping sidecar on 7 of 100 tokens for utterance 0 and 2 of 100 for
    # utterance 1.  Measured, all 16: ALL 16/16, EXTENDED 16/16, BASIC 7/16,
    # DISABLE_ALL 7/16.  Using BASIC here would have invented a reference that
    # disagrees with the one already on the board.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(REPO, MODEL), so,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    waves, utts, drift = [], [], []
    n_identical, ids_match, n_drift = 0, 0, 0
    for i, u in enumerate(chosen):
        path = u["audio"]
        if not os.path.exists(path):
            sys.exit("missing audio: %s" % path)
        pcm, kept, nsrc, trunc = prepare_int16(path)
        waves.append(pcm)

        q, flat = features_int8(fe, pcm)

        # ---- requirement 2: the whole comparison depends on this ------------
        ref = np.frombuffer(feat_bytes, dtype=np.int8,
                            count=TENSOR, offset=0x40 + i * TENSOR)
        ndiff = int((ref != flat).sum())
        # Second opinion: the same audio through gen_corpus.py's own 127,841-sample
        # buffer.  If THIS disagrees, the audio preparation is wrong and nothing
        # below is worth reading.  If only the 128,000 one disagrees, the cause is
        # window length alone.
        nwin_flat = features_int8_nwin(fe, pcm)
        ndiff_nwin = int((ref != nwin_flat).sum())
        if ndiff_nwin != 0:
            sys.exit("STOP: utterance %d %s -- audio preparation does not "
                     "reproduce gen_corpus.py: %d of %d bytes differ even from the "
                     "127,841-sample buffer.  Lead-in, truncation or source record "
                     "is wrong." % (i, u["key"], ndiff_nwin, TENSOR))
        if ndiff == 0:
            n_identical += 1
        else:
            n_drift += 1
            dmask = (ref != flat).reshape(NMEL, T)
            dcols = np.where(dmask.any(0))[0]
            drift.append({
                "index": i, "key": u["key"], "truncated": bool(trunc),
                "int8_bytes_differing": ndiff,
                "max_abs_delta_lsb": int(np.abs(flat.astype(int) - ref.astype(int)).max()),
                "frames_touched": [int(dcols.min()), int(dcols.max())],
                "n_frames_touched": int(len(dcols)),
                "bytes_in_frames_798_799": int(dmask[:, T - 2:].sum()),
                "cause": ("truncated utterance: last kept sample is at index "
                          "%d, so on the device's 128,000-sample buffer "
                          "pre-emphasis continues past it (citrinet_fe.c:226 "
                          "skips only i >= n_samples) where gen_corpus.py's "
                          "127,841-sample buffer ended.  Columns 798-799 move, "
                          "and the per-bin mean/std over all 800 frames carries "
                          "the change into earlier frames."
                          % (LEAD_IN + kept - 1)),
                "authoritative": ("host_feature_fnv1a32 (the 128,000-sample value) "
                                  "-- that is what the device computes"),
            })
            print("DRIFT: utterance %d %s -- %d of %d feature bytes differ from "
                  "%s offset 0x%x (127,841-sample buffer agrees exactly, so this is "
                  "window length, not audio preparation)"
                  % (i, u["key"], ndiff, TENSOR, os.path.relpath(FEAT_BLOB, REPO),
                     0x40 + i * TENSOR), flush=True)

        fhash = fnv1a32(flat.tobytes())
        below, zero, gtotal, gfrac = guard_counts(fe, pcm)

        logits = sess.run(None, {iname: (q.astype(np.float32) * SCALE)[None]})[0]
        ids = logits.reshape(-1, NVOCAB).argmax(-1).astype(np.int32)
        text = ctc_decode(ids, vocab)
        if [int(v) for v in ids] == u["host_ids"]:
            ids_match += 1

        pk = int(np.abs(pcm.astype(np.int32)).max())
        dbfs = 20.0 * np.log10(pk / 32768.0) if pk else float("-inf")

        utts.append({
            "index": i,
            "key": u["key"],
            "duration_s": u["duration_s"],
            "audio": path,
            "ref": u["ref"],
            "n_words_ref": u["n_words_ref"],
            "wav_samples_source": nsrc,
            "wav_samples_kept": kept,
            "truncated": bool(trunc),
            "peak_abs_int16": pk,
            "peak_dbfs": round(float(dbfs), 3),
            "host_feature_fnv1a32": "%08x" % fhash,
            "host_feature_md5": hashlib.md5(flat.tobytes()).hexdigest(),
            "host_ids": [int(v) for v in ids],
            "host_text": text,
            "host_guard_below": below,
            "host_guard_zero": zero,
            "host_guard_total": gtotal,
            "host_guard_fraction": round(float(gfrac), 6),
            "predicted_device_rc": 0 if gfrac <= GUARD_MAX_FRAC else -4,
            "int8_min": int(q.min()), "int8_max": int(q.max()),
            "host_feature_fnv1a32_nwin127841": "%08x" % fnv1a32(nwin_flat.tobytes()),
            "feature_bytes_vs_corpus_blob": ndiff,
            "feature_identical_to_corpus_blob": ndiff == 0,
            "feature_bytes_vs_corpus_blob_nwin127841": ndiff_nwin,
            "ids_match_corpus_ref": [int(v) for v in ids] == u["host_ids"],
        })
        print("[%2d] %-20s %5.2fs peak %7.2f dBFS guard %5d/%5d hash %08x  %s"
              % (i, u["key"], u["duration_s"], dbfs, below, gtotal, fhash, text),
              flush=True)

    if n_identical != args.n and not args.allow_window_drift:
        sys.exit("\nSTOP: only %d of %d utterances reproduce corpus_blob.bin byte "
                 "for byte; %d drifted.  Every drifting utterance reproduced it "
                 "exactly from the 127,841-sample buffer, so the audio preparation "
                 "is right and the cause is window length -- see WINDOW DRIFT in "
                 "this file's docstring.  No blob written.  Re-run with "
                 "--allow-window-drift to emit it with the drift recorded."
                 % (n_identical, args.n, n_drift))

    # ---- blob ---------------------------------------------------------------
    hdr = MAGIC + struct.pack("<III", args.n, NSAMPLES, 0) + b"\0" * 48
    assert len(hdr) == HDR
    blob = bytearray(hdr)
    for pcm in waves:
        b = pcm.astype("<i2").tobytes()
        assert len(b) == WAV_BYTES
        blob += b
    assert len(blob) == HDR + args.n * WAV_BYTES
    blob_path = os.path.join(args.outdir, "wav_blob.bin")
    open(blob_path, "wb").write(bytes(blob))
    md5 = hashlib.md5(bytes(blob)).hexdigest()

    dbfs_vals = [u["peak_dbfs"] for u in utts]
    side = {
        "generated_by": "firmware/tools/gen_wav_corpus.py",
        "purpose": ("Gate 5 bring-up: replay canned waveforms from external flash "
                    "through citrinet_fe.c on the M55, then the NPU, then the C CTC "
                    "decoder.  No microphone, no gain stage."),
        "provenance": {
            "selection": ("first %d utterances of %s, in that file's order -- NOT "
                          "re-selected.  Utterance i here is index i there, so the "
                          "device's features can be compared against "
                          "artifacts/corpus/corpus_blob.bin offset 0x40 + i*64000, "
                          "which is already flashed at 0x71000000."
                          % (args.n, os.path.relpath(FEAT_REF, REPO))),
            "feature_corpus_ref": os.path.relpath(FEAT_REF, REPO),
            "feature_corpus_blob": os.path.relpath(FEAT_BLOB, REPO),
            "feature_corpus_blob_md5": hashlib.md5(feat_bytes).hexdigest(),
            "feature_corpus_N": feat["N"],
            "inherited_seed": feat["seed"],
            "inherited_anchor_index0": feat.get("anchor_index0"),
            "inherited_exclusions": ("48 cal_800 keys and the union of all three "
                                     "calibration sets, applied by gen_corpus.py"),
        },
        "audio_preparation": {
            "source": "16-bit LibriSpeech dev-clean FLAC, native level, no gain applied",
            "lead_in_samples": LEAD_IN,
            "keep_max_samples": KEEP_MAX,
            "window_samples": NSAMPLES,
            "rule": ("buf = zeros(128000); n = min(len(wav), 123041); "
                     "buf[4800:4800+n] = wav[:n].  The 123041 limit and the 4800 "
                     "lead-in are gen_corpus.py:87 and :38 verbatim; the buffer is "
                     "128000 rather than 127841 because that is CITRINET_FE_NSAMPLES, "
                     "and the 159 extra samples are the zeros librosa's centre "
                     "padding supplied anyway."),
            "int16": "k = rint(f*32768), asserted lossless for every sample",
        },
        "model": MODEL,
        "ort_graph_optimization_level": "ORT_ENABLE_ALL (onnxruntime's actual default; matches corpus_ref.json)",
        "onnxruntime": ort.__version__,
        "scale": SCALE, "T": T, "NMEL": NMEL, "NVOCAB": NVOCAB, "BLANK": BLANK,
        "log_guard": LOG_GUARD,
        "N": args.n,
        "keys": [u["key"] for u in utts],
        "hash": {
            "algorithm": "FNV-1a 32-bit",
            "offset_basis": 2166136261, "prime": 16777619,
            "over": "the 64,000-byte int8 feature tensor, mel-major, as unsigned bytes",
            "format": "big-endian hex, lower case, no 0x -- matches the device's print",
        },
        "guard": {
            "definition": ("count of mel energies, before the log, strictly below "
                           "2^-24 over the 80x800 matrix -- citrinet_fe.c:273"),
            "note": ("host uses BLAS for the mel projection, the device 500 sparse "
                     "MACs in a fixed order, so a bin within an ulp of the guard may "
                     "be counted differently; host_guard_zero is exact"),
        },
        "blob": {
            "path": os.path.relpath(blob_path, REPO),
            "bytes": len(blob),
            "md5": md5,
            "magic": "STTW",
            "header_bytes": HDR,
            "waveform_bytes": WAV_BYTES,
            "samples_per_utterance": NSAMPLES,
            "dtype": "int16 little-endian",
            "flash_address": FLASH_ADDR,
            "collision_check": ("weights 0x70400000..~0x70D9C000; feature corpus "
                                "0x71000000 + 4,096,064 B ends 0x713E8040; this blob "
                                "0x72000000 + %d B ends 0x%08X" % (len(blob),
                                0x72000000 + len(blob))),
        },
        "byte_identity_check": {
            "result": ("PASS -- %d/%d utterances reproduce corpus_blob.bin exactly"
                       % (n_identical, args.n)) if n_drift == 0 else
                      ("PARTIAL -- %d/%d reproduce corpus_blob.bin exactly; %d "
                       "drifted on window length alone.  All %d reproduce it "
                       "exactly from gen_corpus.py's own 127,841-sample buffer, "
                       "so the audio preparation is verified for every utterance."
                       % (n_identical, args.n, n_drift, args.n)),
            "utterances_identical": n_identical,
            "utterances_checked": args.n,
            "utterances_drifted": n_drift,
            "audio_preparation_verified": args.n,
            "bytes_per_utterance": TENSOR,
            "compared_against": os.path.relpath(FEAT_BLOB, REPO) + " offset 0x40 + i*64000",
            "window_drift": drift,
            "which_hash_the_device_must_match": (
                "host_feature_fnv1a32 for every utterance -- it is computed over the "
                "same 128,000 samples the device is handed.  For a drifted utterance "
                "that is NOT corpus_blob.bin; host_feature_fnv1a32_nwin127841 is the "
                "corpus_blob.bin value and is recorded only for the audit trail."),
        },
        "ids_cross_check": {
            "result": "%d/%d host id sequences match corpus_ref.json"
                      % (ids_match, args.n),
            "note": ("independent re-run of the deployed graph on the regenerated "
                     "features, at ORT_ENABLE_ALL with the default intra-op thread "
                     "count -- the configuration that produced corpus_ref.json"),
            "sensitivity_measured": {
                "ORT_ENABLE_ALL": "16/16", "ORT_ENABLE_EXTENDED": "16/16",
                "ORT_ENABLE_BASIC": "7/16", "ORT_DISABLE_ALL": "7/16",
                "ORT_ENABLE_ALL, intra_op_num_threads=1": "14/16",
                "comment": ("the host reference is not invariant to onnxruntime's "
                            "optimisation level or thread count; utterance 0 moves "
                            "by 7 of 100 tokens between ALL and BASIC.  Anything "
                            "comparing against corpus_ref.json must use ALL with "
                            "default threads."),
            },
        },
        "predicted_device_rc": {
            "note": ("citrinet_fe_run() returns CITRINET_FE_E_GUARD (-4) when the "
                     "guard fraction exceeds CITRINET_FE_GUARD_MAX_FRAC = 0.50.  "
                     "Predicted from the host guard counts so a -4 on the board is "
                     "recognised as expected rather than investigated as a bug."),
            "expect_rc_0": [u["index"] for u in utts if u["predicted_device_rc"] == 0],
            "expect_rc_minus4": [u["index"] for u in utts if u["predicted_device_rc"] == -4],
            "max_guard_fraction": max(u["host_guard_fraction"] for u in utts),
            "caveat": ("a predicted -4 here is NOT a gain problem.  The threshold was "
                       "derived over utterances that FILL the 8 s window "
                       "(firmware/inc/citrinet_fe.h, max 33.1 %); a short utterance "
                       "leaves most of the window near-silent, and the "
                       "exactly-zero correction in citrinet_fe_guard_fraction() only "
                       "removes bins that are identically 0, not the near-zero ones "
                       "either side of the speech."),
        },
        "peak_dbfs": {"min": min(dbfs_vals), "max": max(dbfs_vals),
                      "median": float(np.median(dbfs_vals))},
        "device_line_format": ("# w <i> fe <cycles> npu <cycles> rc <n> guard <n> "
                               "hash <8 hex> ids: <100 ints>, then '# wav done'"),
        "utterances": utts,
    }
    ref_path = os.path.join(args.outdir, "wav_ref.json")
    json.dump(side, open(ref_path, "w"), indent=1)

    print()
    print("blob   %s" % blob_path)
    print("bytes  %d  (0x40 header + %d * %d)" % (len(blob), args.n, WAV_BYTES))
    print("md5    %s" % md5)
    print("side   %s" % ref_path)
    print("byte-identity vs corpus_blob.bin: %d/%d utterances exact"
          % (n_identical, args.n))
    print("host ids vs corpus_ref.json:      %d/%d match  (ORT_ENABLE_ALL)"
          % (ids_match, args.n))
    if ids_match != args.n:
        print("WARNING: %d id sequence(s) disagree with corpus_ref.json despite "
              "identical features -- check the onnxruntime configuration"
              % (args.n - ids_match))
    print("peak level: %.2f .. %.2f dBFS (median %.2f), LibriSpeech native"
          % (min(dbfs_vals), max(dbfs_vals), float(np.median(dbfs_vals))))
    e4 = [u["index"] for u in utts if u["predicted_device_rc"] == -4]
    print("predicted rc: 0 for %d utterance(s); -4 (E_GUARD) for %s"
          % (args.n - len(e4), e4 if e4 else "none"))
    print()
    print("flash it with:")
    print("  STM32_Programmer_CLI -c port=SWD mode=UR --extload <loader> "
          "-w %s %s" % (blob_path, FLASH_ADDR))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score a board UART capture of the WAVEFORM-REPLAY run: waveform -> M55 front
end -> NPU -> C CTC decoder, all on the device.

    python firmware/test/score_wav.py --build-host-ref [--wav-blob <blob>]   # once
    python firmware/test/score_wav.py --log board/traces/wav_run.log          # then

The first command writes artifacts/corpus/wav_host_ref.json: the host front end
recomputed over the DEVICE's 128,000-sample window, with the guard counters that
have no other source, and -- where that window is not the tensor in
corpus_blob.bin -- its own onnxruntime ids.  It is already built for the 64
sidecar utterances; rerun it with --wav-blob once the waveform blob exists, which
removes every assumption about how the blob was prepared.

Gate 4 fed the NPU features computed on a workstation.  This run feeds it
features the M55 computed itself, from int16 PCM in external flash, and answers
three questions:

  A. did the M55's front end compute the same features as the host?
     -- device FNV-1a over its 64,000-byte int8 tensor vs the same hash over the
        host tensor in artifacts/corpus/corpus_blob.bin.  Bit-exact or not.
  B. what did the front end and the NPU each cost, in cycles?
     -- per-invocation counter reads, reported as measured values.  Never a
        wall-clock total divided by a count.
  C. does waveform -> features -> NPU -> text still produce the right text?
     -- greedy CTC over the device ids, WER against the LibriSpeech reference,
        paired against the host's WER on the same utterances.

INPUT 1 -- the capture.  Any text containing lines of the form

    # w <i> fe <cycles> npu <cycles> rc <n> guard <n> hash <8 hex> ids: <100 ints>

terminated by "# wav done".  ANSI escapes are stripped, terminal hard-wraps are
rejoined and repeated passes are split, exactly as
firmware/test/score_corpus.py:parse_log() does -- that function's approach is
reused here rather than reinvented, and the primitives (Levenshtein, CTC
collapse, Wilson, the paired bootstrap) are imported from it.

INPUT 2 -- artifacts/corpus/corpus_ref.json, the 64-utterance sidecar written by
firmware/tools/gen_corpus.py: host ids, host text, the LibriSpeech reference.

INPUT 3 -- artifacts/corpus/corpus_blob.bin, the host int8 features, md5-checked
against the sidecar, and artifacts/corpus/wav_host_ref.json (above).

Those two are NOT always the same tensor, which was measured, not assumed:
corpus_blob.bin comes from gen_corpus.py's 127,841-sample buffer, which ends on
the last audio sample, so the 0.97 pre-emphasis never emits its one-sample
ring-out; the device's 128,000-sample window has zeros after the audio and
citrinet_fe.c:226-230 does emit it.  61 of the 64 utterances are byte-identical
either way; the 3 the 8 s window truncates differ by 4, 106 and 866 int8 values,
and on one of them the host transcript itself changes ("mar anne" ->
"maryianne").  Scoring those three against corpus_blob.bin would have charged the
device for a word error it did not make.  The device's window is therefore the
primary oracle, corpus_blob.bin is kept as a second candidate, and a device that
matches the second one is reported as a blob-preparation problem rather than a
front-end fault.

Self-test, no board needed:

    python firmware/test/score_wav.py --self-test

synthesises a capture from the sidecar and the feature blob, injects a wrong
hash, a token substitution, an implausible cycle count, a missing utterance and a
truncated line, and checks each is reported exactly as injected.
"""
import argparse, hashlib, importlib.util, json, math, os, re, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ reuse
# score_corpus.py already solves ANSI stripping, terminal wrapping, repeated
# passes, Levenshtein, CTC collapse, Wilson and the paired bootstrap.  Import it
# rather than growing a second copy that can drift.
_spec = importlib.util.spec_from_file_location("score_corpus",
                                               os.path.join(HERE, "score_corpus.py"))
SC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SC)

strip_ansi   = SC.strip_ansi
wilson       = SC.wilson
levenshtein  = SC.levenshtein
ctc_greedy   = SC.ctc_greedy
ids_to_text  = SC.ids_to_text
load_sidecar = SC.load_sidecar
pct          = SC.pct

# ------------------------------------------------------------------ constants
T, NMEL, BLANK = 800, 80, 1024
TENSOR = NMEL * T                       # 64,000 B, mel-major: index = mel*800 + frame
NSAMPLES = 128000                       # the 8.000 s deployment window
LEAD_IN = 4800                          # gen_corpus.py:LEAD_IN
NWIN = (T - 1) * 160 + 1                # 127,841 -- gen_corpus.py's buffer length
LOG_GUARD = np.float32(2.0 ** -24)      # citrinet_fe.h:CITRINET_FE_LOG_GUARD
GUARD_MAX_FRAC = 0.50                   # citrinet_fe.h:CITRINET_FE_GUARD_MAX_FRAC
FE_OK, FE_E_GUARD = 0, -4               # citrinet_fe.h:98-103
WAV_MAGIC = b"STTW"
WAV_HDR = 0x40

# Round 19, board/GATE4.md:1964 -- "# invoke 74421588 cycles = 124.035 ms at
# 600000000 Hz", one canned tensor already in RAM.
ROUND19_CYCLES, ROUND19_MS = 74421588, 124.035
# Round 20, board/GATE4.md:2098 -- median over the 64-utterance corpus run, which
# memcpy'd 64,000 B from external flash immediately before every invoke.
ROUND20_CYCLES, ROUND20_MS = 83997678, 140.0
# Round 18, board/GATE4.md:1771 -- the 1064-epoch --force-all-in-out-to-mem build.
ROUND18_CYCLES = 116393913

DEFAULT_HZ = 600_000_000                # SYSCLK, board/GATE4.md:611

# Plausibility bands.  Outside -> the number is not a measurement of the thing it
# claims to measure and is reported as such.  Stated here so they can be argued
# with rather than buried in a conditional.
#   fe floor  200,000 cy (0.33 ms): the front end alone performs 80*800 = 64,000
#     mel dot products of length 257 (16.4 M MACs) plus 800 512-point real FFTs;
#     0.33 ms at 600 MHz is 200 k cycles for 16.4 M MACs, i.e. 82 MACs/cycle.
#     Below this the counter did not run.
#   fe ceiling 3,000,000,000 cy (5.00 s): more than half the 8 s of audio it is
#     supposed to be keeping up with, so worth refusing to average in silently.
#   npu band [5.0e7, 2.5e8] cy (83.3 ms .. 416.7 ms): the fastest schedule ever
#     measured on this graph is 74.4 M cycles (Round 19) and the slowest is
#     116.4 M (Round 18, 1064 epochs).  A value below 5.0e7 beats the best
#     measured schedule by 33 %; above 2.5e8 is more than twice the worst.
FE_MIN, FE_MAX = 200_000, 3_000_000_000
FE_SOFT_MIN, FE_SOFT_MAX = 4_000_000, 1_200_000_000
NPU_MIN, NPU_MAX = 50_000_000, 250_000_000


# ------------------------------------------------------------------ FNV-1a
def fnv1a(buf):
    """FNV-1a 32-bit, the standard parameters: basis 2166136261, prime 16777619,
    over unsigned bytes.  h ^= b; h *= prime (mod 2^32)."""
    h = 2166136261
    for b in bytes(buf):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def fnv1a_np(buf):
    """Same function, vectorised over the 64,000-byte tensor.  FNV-1a is a strict
    fold so it cannot be parallelised; this only avoids the Python byte loop by
    working in numpy uint32 arithmetic 1 byte at a time.  Checked against
    fnv1a() on every call site in the self-test."""
    a = np.frombuffer(bytes(buf), dtype=np.uint8)
    h = np.uint32(2166136261)
    p = np.uint32(16777619)
    with np.errstate(over="ignore"):
        for b in a:
            h = np.uint32(h ^ np.uint32(b))
            h = np.uint32(h * p)
    return int(h)


def fhex(h):
    """8 hex digits, big-endian, lower case, no 0x -- the contract's format."""
    return "%08x" % (h & 0xFFFFFFFF)


def row_hashes(tensor):
    """Point 5's follow-up instrument, host side.  The tensor is mel-major
    (citrinet_fe.h: out[b*800 + t]), so row b is bytes [b*800, (b+1)*800).
    80 rows of 800 bytes, each hashed with the same FNV-1a as the whole tensor.

    UNTESTED PATH: no firmware prints these yet.  It exists so that if the
    whole-tensor hash disagrees, a rebuild that prints 80 row hashes can be
    diffed against this without any further host work."""
    b = bytes(tensor)
    assert len(b) == TENSOR, len(b)
    return [fhex(fnv1a(b[r * T:(r + 1) * T])) for r in range(NMEL)]


# ------------------------------------------------------------------ log parsing
# Loose head match first, so a line that is recognisably a record but malformed is
# REPORTED as malformed instead of silently ignored.  Field values are pulled out
# individually and validated, which is what lets the self-test's truncated line be
# attributed to the right utterance index.
HEAD_RE = re.compile(r"#\s*w\s+(\d+)\b(.*)$")
IDS_SPLIT_RE = re.compile(r"\bids\s*:\s*(.*)$", re.S)
FIELD_RE = re.compile(r"\b(fe|npu|rc|guard|hash)\s+(\S+)")
DONE_RE = re.compile(r"#\s*wav\s+done")
HZ_RE = re.compile(r"at\s+(\d{6,})\s*Hz")
SYSCLK_RE = re.compile(r"SYSCLK\s+clock\s*:\s*(\d+)\s*MHz")
INT_ONLY_RE = re.compile(r"[\d\s]+")

REQUIRED = ("fe", "npu", "rc", "guard", "hash")


def _parse_fields(head, ids_txt, n_out_frames, lines, li):
    """Returns (record, li, error).  li may advance when a hard-wrapped id list is
    absorbed."""
    kv = dict(FIELD_RE.findall(head))
    missing = [k for k in REQUIRED if k not in kv]
    if missing:
        return None, li, "missing field(s) " + ",".join(missing)
    rec = {}
    for k in ("fe", "npu", "rc", "guard"):
        v = kv[k]
        if not re.fullmatch(r"-?\d+", v):
            return None, li, f"{k}={v!r} is not an integer"
        rec[k] = int(v)
    if not re.fullmatch(r"[0-9a-fA-F]{8}", kv["hash"]):
        return None, li, f"hash={kv['hash']!r} is not 8 hex digits"
    rec["hash"] = kv["hash"].lower()
    for k in ("fe", "npu", "guard"):
        if rec[k] < 0:
            return None, li, f"{k} is negative"
    try:
        ids = [int(x) for x in ids_txt.split()]
    except ValueError:
        return None, li, "unparsable ids"
    # a terminal can hard-wrap a 100-number line; absorb following integer-only
    # lines until the count is satisfied (score_corpus.py:parse_log does this).
    while len(ids) < n_out_frames and li + 1 < len(lines):
        j = li + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines) or not INT_ONLY_RE.fullmatch(lines[j].strip()):
            break
        ids += [int(x) for x in lines[j].split()]
        li = j
    if len(ids) != n_out_frames:
        return None, li, f"{len(ids)} ids, expected {n_out_frames}"
    if min(ids) < 0 or max(ids) > BLANK:
        return None, li, f"id out of range [{min(ids)},{max(ids)}]"
    rec["ids"] = ids
    return rec, li, None


def parse_log(text, n_out_frames):
    """Split a capture into passes.  A new pass starts when an index repeats or
    after a '# wav done' marker.  Returns
    dict(passes=[{i: rec}], bad=[(i, reason, raw)], n_records, hz, notes)."""
    text = strip_ansi(text)
    lines = text.split("\n")
    passes, cur, bad, notes = [], {}, [], []
    hz = None
    n_records = 0
    li = -1
    while li + 1 < len(lines):
        li += 1
        raw = lines[li]
        m_hz = HZ_RE.search(raw)
        if m_hz:
            hz = int(m_hz.group(1))
        m_sys = SYSCLK_RE.search(raw)
        if m_sys and hz is None:
            hz = int(m_sys.group(1)) * 1_000_000
        if DONE_RE.search(raw):
            if cur:
                passes.append(cur)
                cur = {}
            continue
        m = HEAD_RE.search(raw)
        if not m:
            continue
        idx = int(m.group(1))
        rest = m.group(2)
        mi = IDS_SPLIT_RE.search(rest)
        if not mi:
            # the header itself may have been wrapped: join ONE following line,
            # but only if that line does not start a record of its own.
            if li + 1 < len(lines) and not lines[li + 1].lstrip().startswith("#"):
                joined = rest + " " + lines[li + 1]
                mi = IDS_SPLIT_RE.search(joined)
                if mi:
                    rest = joined
                    li += 1
                    notes.append(f"u{idx}: header rejoined across a wrap")
        if not mi:
            bad.append((idx, "no 'ids:' field (truncated line?)", raw.strip()[:100]))
            continue
        head = rest[:mi.start()]
        rec, li, err = _parse_fields(head, mi.group(1), n_out_frames, lines, li)
        if err:
            bad.append((idx, err, raw.strip()[:100]))
            continue
        n_records += 1
        rec["i"] = idx
        if idx in cur:
            passes.append(cur)
            cur = {}
        cur[idx] = rec
    if cur:
        passes.append(cur)
    return dict(passes=passes, bad=bad, n_records=n_records, hz=hz, notes=notes)


def compare_passes(passes):
    """Are the repeats identical?  Cycle counts legitimately vary run to run and
    are compared separately; ids/hash/rc/guard must not vary."""
    notes = []
    if len(passes) < 2:
        return True, notes
    ref = passes[0]
    identical = True
    for p, cur in enumerate(passes[1:], start=1):
        if set(cur) != set(ref):
            identical = False
            notes.append(f"pass {p}: index set differs "
                         f"(missing {sorted(set(ref)-set(cur))[:8]}, "
                         f"extra {sorted(set(cur)-set(ref))[:8]})")
        for i in sorted(set(cur) & set(ref)):
            for f in ("ids", "hash", "rc", "guard"):
                if cur[i][f] != ref[i][f]:
                    identical = False
                    if f == "ids":
                        n = sum(a != b for a, b in zip(ref[i]["ids"], cur[i]["ids"]))
                        notes.append(f"pass {p}: u{i} ids differ from pass 0 in {n} frames")
                    else:
                        notes.append(f"pass {p}: u{i} {f} {ref[i][f]} -> {cur[i][f]}")
    return identical, notes


# ------------------------------------------------------------------ host side
def load_features(path, n_expect=None):
    """The host int8 tensors, straight out of the flashed feature blob.  These are
    byte-for-byte what the device should have computed."""
    raw = open(path, "rb").read()
    if raw[:4] != b"STTC":
        raise SystemExit(f"{path}: magic {raw[:4]!r}, expected b'STTC'")
    n, t, nmel = np.frombuffer(raw, dtype="<u4", count=3, offset=4)
    if (int(t), int(nmel)) != (T, NMEL):
        raise SystemExit(f"{path}: T={t} NMEL={nmel}, expected {T}/{NMEL}")
    body = len(raw) - 0x40
    if body != int(n) * TENSOR:
        raise SystemExit(f"{path}: {body} body bytes for N={n}")
    feats = {i: raw[0x40 + i * TENSOR:0x40 + (i + 1) * TENSOR] for i in range(int(n))}
    return dict(path=path, N=int(n), feats=feats,
                md5=hashlib.md5(raw).hexdigest(), bytes=len(raw))


def read_wav_blob(path):
    """The waveform blob the device replays.  Header per the contract."""
    raw = open(path, "rb").read()
    if raw[:4] != WAV_MAGIC:
        raise SystemExit(f"{path}: magic {raw[:4]!r}, expected {WAV_MAGIC!r}")
    n, ns, resv = np.frombuffer(raw, dtype="<u4", count=3, offset=4)
    if int(ns) != NSAMPLES:
        raise SystemExit(f"{path}: NSAMPLES={ns}, expected {NSAMPLES}")
    body = len(raw) - WAV_HDR
    if body != int(n) * NSAMPLES * 2:
        raise SystemExit(f"{path}: {body} body bytes for N={n}")
    wavs = {i: np.frombuffer(raw, dtype="<i2", count=NSAMPLES,
                             offset=WAV_HDR + i * NSAMPLES * 2)
            for i in range(int(n))}
    return dict(path=path, N=int(n), reserved=int(resv), wavs=wavs,
                md5=hashlib.md5(raw).hexdigest(), bytes=len(raw))


def _load_fe():
    """model/fe.py is the spec AND the oracle; gen_corpus.py:load_fe() loads it
    this way, and the two must not drift."""
    path = os.path.join(REPO, "model", "fe.py")
    src = open(path).read()
    mod = type(sys)("fe")
    mod.__file__ = path
    exec(compile(src, "fe.py", "exec"), mod.__dict__)
    return mod


def host_window(audio_path):
    """The int16 window the waveform blob carries, built the way
    gen_corpus.py:features_int8() builds its float buffer: 4,800 samples of
    lead-in, the audio truncated at NWIN-LEAD_IN, zero to 128,000."""
    import soundfile as sf
    wav, sr = sf.read(audio_path)
    if sr != 16000:
        raise SystemExit(f"{audio_path}: {sr} Hz")
    n = min(len(wav), NWIN - LEAD_IN)
    buf = np.zeros(NSAMPLES, dtype=np.float64)
    buf[LEAD_IN:LEAD_IN + n] = wav[:n]
    q = np.clip(np.round(buf * 32768.0), -32768, 32767).astype(np.int16)
    residual = float(np.abs(buf * 32768.0 - np.round(buf * 32768.0)).max())
    return q, residual, len(wav), n


def host_features_and_guard(fe, pcm_i16, scale):
    """Reproduce citrinet_fe_run() on the host: /32768, pre-emphasis, STFT, mel,
    log guard, per-bin normalisation, quantise.  Also count what the firmware's
    guard counters count -- citrinet_fe.c:273-274, `acc < 2^-24` (strict) and
    `acc == 0.0f` -- so the device's 'guard' field has something to be checked
    against."""
    import librosa
    x = pcm_i16.astype(np.float32) * np.float32(1.0 / 32768.0)
    xx = np.concatenate([x[:1], x[1:] - np.float32(0.97) * x[:-1]])
    S = librosa.stft(xx, n_fft=512, hop_length=160, win_length=400,
                     window=fe._w, center=True, pad_mode="constant")
    P = (np.abs(S) ** 2.0).astype(np.float32)
    E = (fe._fb @ P)[:, :T]
    below = int((E < LOG_GUARD).sum())
    zero = int((E == 0).sum())
    mel = fe.norm_pf(np.log(E + LOG_GUARD))
    q = np.clip(np.round(mel / scale), -128, 127).astype(np.int8)
    return q, below, zero


def guard_fraction(below, zero, total=TENSOR):
    """citrinet_fe.c:389-410 exactly, including the exclusion of the exactly-zero
    bins that the zero-filled tail contributes."""
    b, t = below, total
    if zero <= below and zero < total:
        b -= zero
        t -= zero
    return (b / t) if t else 0.0


_ORT = {}


def _ort_ids(q_int8, scale, model_rel):
    """Host argmax ids for a tensor the feature blob does not contain.  Same
    session settings as gen_corpus.py and gen_wav_corpus.py: ORT_ENABLE_ALL --
    onnxruntime's actual default -- CPU, dequantised int8 fed in, so host and
    device see the same values.  The level matters: gen_wav_corpus.py measured
    ALL 16/16 and EXTENDED 16/16 against corpus_ref.json, but BASIC only 7/16."""
    import onnxruntime as ort
    if "sess" not in _ORT:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ORT["sess"] = ort.InferenceSession(os.path.join(REPO, model_rel), so,
                                            providers=["CPUExecutionProvider"])
        _ORT["in"] = _ORT["sess"].get_inputs()[0].name
    lg = _ORT["sess"].run(None, {_ORT["in"]: (q_int8.astype(np.float32) * scale)[None]})[0]
    lg = lg.reshape(-1, lg.shape[-1])
    return [int(v) for v in lg.argmax(-1)]


def _host_entry(side, feats, i, key, pcm, variants_pcm, scale, want_ids):
    """One host-reference record: the tensor the device should compute from `pcm`,
    its guard counters, and -- when it is NOT the tensor in corpus_blob.bin -- its
    own host ids, because the sidecar's ids belong to the blob's tensor."""
    fe = _load_fe()
    q, below, zero = host_features_and_guard(fe, pcm, scale)
    flat = q.ravel().tobytes()
    ndiff = int((np.frombuffer(flat, dtype=np.int8) !=
                 np.frombuffer(feats["feats"][i], dtype=np.int8)).sum()) \
        if i in feats["feats"] else None
    ent = dict(key=key, guard_below=below, guard_zero=zero,
               guard_frac=guard_fraction(below, zero),
               rc_expect=FE_OK if guard_fraction(below, zero) <= GUARD_MAX_FRAC
               else FE_E_GUARD,
               fnv=fhex(fnv1a(flat)), bytes_vs_feature_blob=ndiff, variants={})
    for vname, vpcm in (variants_pcm or {}).items():
        vq, vb, vz = host_features_and_guard(fe, vpcm, scale)
        ent["variants"][vname] = dict(fnv=fhex(fnv1a(vq.ravel().tobytes())),
                                      guard_below=vb, guard_zero=vz)
    if want_ids and ndiff:
        try:
            ent["host_ids"] = _ort_ids(q, scale, side["raw"]["model"])
            ent["host_ids_note"] = ("recomputed with onnxruntime: this tensor is not "
                                    "the one in corpus_blob.bin, so the sidecar's ids "
                                    "do not describe it")
        except Exception as e:                                    # noqa: BLE001
            ent["host_ids_error"] = f"{type(e).__name__}: {e}"
    return ent


def build_host_ref(side, feats, indices, verbose=True, want_ids=True):
    """Recompute, from the audio, the tensor and the guard counters the DEVICE
    should produce -- i.e. over the 128,000-sample window the waveform blob
    carries, not over gen_corpus.py's 127,841-sample buffer.

    Those two windows are not always the same tensor.  gen_corpus.py's buffer ends
    at the last audio sample, so the 0.97 pre-emphasis never emits its one-sample
    ring-out; the device's buffer has 159 zeros after it, so
    citrinet_fe_build_frame() (citrinet_fe.c:226-230, `i < n_samples`) does emit it
    at index 127,841, inside frame 799.  Per-bin normalisation then spreads that
    one frame over the whole tensor.  Measured on this corpus: identical for 61 of
    64 utterances, 4 / 106 / 866 int8 values different for the three the 8 s window
    truncates.  Both hashes are recorded so a device that matches either can be
    told apart from one that matches neither.

    The second variant, `fill_128000`, is the other defensible reading of "same
    truncation": keep min(len, 123,200) samples so the 128,000-sample window is
    full.  It only differs for utterances long enough to reach the end."""
    scale = side["raw"]["scale"]
    raw_u = {u["index"]: u for u in side["raw"]["utterances"]}
    out = {}
    for i in indices:
        u = raw_u[i]
        pcm, resid, n_wav, n_kept = host_window(u["audio"])
        variants = {}
        if n_wav > NWIN - LEAD_IN:
            import soundfile as sf
            wav, _ = sf.read(u["audio"])
            n2 = min(len(wav), NSAMPLES - LEAD_IN)
            b2 = np.zeros(NSAMPLES, dtype=np.float64)
            b2[LEAD_IN:LEAD_IN + n2] = wav[:n2]
            variants["fill_128000"] = np.clip(np.round(b2 * 32768.0),
                                              -32768, 32767).astype(np.int16)
        ent = _host_entry(side, feats, i, u["key"], pcm, variants, scale, want_ids)
        ent.update(int16_round_residual=resid, wav_samples=int(n_wav),
                   wav_samples_kept=int(n_kept), source="audio, 128000-sample window")
        out[i] = ent
        if verbose:
            print(f"  [{i:3d}] {u['key']:<22} guard {ent['guard_below']:>6} "
                  f"(zero {ent['guard_zero']:>6}, frac {100*ent['guard_frac']:5.2f} %)  "
                  f"fnv {ent['fnv']}  vs corpus_blob.bin: "
                  f"{'identical' if ent['bytes_vs_feature_blob'] == 0 else str(ent['bytes_vs_feature_blob']) + ' int8 differ'}"
                  + ("  [host ids recomputed]" if "host_ids" in ent else ""), flush=True)
    return out


# ------------------------------------------------------------------ cost
def cycles_to_ms(c, hz):
    return 1000.0 * c / hz


def cost_stats(vals, hz):
    v = np.asarray(sorted(vals), dtype=np.int64)
    return dict(n=len(v), min=int(v.min()), median=float(np.median(v)), max=int(v.max()),
                min_ms=cycles_to_ms(int(v.min()), hz),
                median_ms=cycles_to_ms(float(np.median(v)), hz),
                max_ms=cycles_to_ms(int(v.max()), hz),
                spread_pct=100.0 * (int(v.max()) - int(v.min())) / max(float(np.median(v)), 1.0))


def cycle_flags(recs):
    """Every cycle count that cannot be a measurement of what it claims to
    measure, with the band it violated."""
    out = []
    for i in sorted(recs):
        r = recs[i]
        if not (FE_MIN <= r["fe"] <= FE_MAX):
            out.append((i, "fe", r["fe"],
                        f"outside the plausible band [{FE_MIN}, {FE_MAX}] cycles"))
        elif not (FE_SOFT_MIN <= r["fe"] <= FE_SOFT_MAX):
            out.append((i, "fe", r["fe"], "inside the plausible band but outside the "
                        f"expected [{FE_SOFT_MIN}, {FE_SOFT_MAX}] -- advisory"))
        if not (NPU_MIN <= r["npu"] <= NPU_MAX):
            out.append((i, "npu", r["npu"],
                        f"outside the plausible band [{NPU_MIN}, {NPU_MAX}] cycles "
                        f"(Round 19 measured {ROUND19_CYCLES}, Round 18 {ROUND18_CYCLES})"))
    return out


# ------------------------------------------------------------------ scoring
def score(side, feats, recs, hostref, B=10000, seed=1, hz=DEFAULT_HZ):
    rng = np.random.default_rng(seed)
    blank = side["blank"]
    vocab = [l.rsplit(" ", 1)[0] for l in
             open(os.path.join(REPO, "tokenizer", "vocab.txt"), encoding="utf-8")
             .read().split("\n") if l.strip()]
    utts = {u["i"]: u for u in side["utts"]}
    idxs = sorted(i for i in recs if i in utts)
    unknown = sorted(set(recs) - set(utts))

    blob_hash = {i: fhex(fnv1a(feats["feats"][i])) for i in idxs if i in feats["feats"]}

    def candidates(i):
        """Every host tensor this device tensor could legitimately be, primary
        first.  Matching a non-primary candidate is a different diagnosis from
        matching none: it says the arithmetic is right and the input window is
        not."""
        out = []
        g = hostref.get(i)
        if g and g.get("fnv"):
            out.append((g.get("source", "recomputed host window"), g["fnv"]))
        if i in blob_hash:
            out.append(("corpus_blob.bin (gen_corpus.py's 127,841-sample buffer)",
                        blob_hash[i]))
        for vn, v in ((g or {}).get("variants") or {}).items():
            out.append((f"variant {vn}", v["fnv"]))
        return out

    rows = []
    for i in idxs:
        r = recs[i]
        u = utts[i]
        g = hostref.get(i)
        cands = candidates(i)
        primary = cands[0] if cands else (None, None)
        matched = next((n for n, v in cands if v == r["hash"]), None)
        host_ids = (g.get("host_ids") if g and g.get("host_ids") else u["host_ids"])
        ids_src = ("recomputed for the device's window (onnxruntime)"
                   if g and g.get("host_ids") else "sidecar")
        h = np.asarray(host_ids)
        d = np.asarray(r["ids"])
        hc, dc = ctc_greedy(h, blank), ctc_greedy(d, blank)
        htext, dtext = ids_to_text(hc, vocab), ids_to_text(dc, vocab)
        ref = u["ref"].lower().split()
        hS, hI, hD = levenshtein(ref, htext.split())
        dS, dI, dD = levenshtein(ref, dtext.split())
        xS, xI, xD = levenshtein(htext.split(), dtext.split())
        hh = primary[1]
        rows.append(dict(
            i=i, k=u["k"], nw=len(ref), ref=u["ref"], truncated=bool(u.get("truncated")),
            host_text=htext, dev_text=dtext, sidecar_text=u["host_text"],
            n_frames=len(h), n_diff=int((h != d).sum()),
            host_S=hS, host_I=hI, host_D=hD, host_err=hS + hI + hD,
            dev_S=dS, dev_I=dI, dev_D=dD, dev_err=dS + dI + dD,
            xS=xS, xI=xI, xD=xD, x_err=xS + xI + xD,
            n_host_words=len(htext.split()),
            same_ids=bool(hc == dc), same_text=bool(htext == dtext),
            dev_hash=r["hash"], host_hash=hh, host_hash_source=primary[0],
            blob_hash=blob_hash.get(i), matched=matched,
            n_oracles_differing=(0 if not cands else
                                 len({v for _, v in cands}) - 1),
            host_ids_source=ids_src,
            bytes_vs_blob=(g.get("bytes_vs_feature_blob") if g else None),
            hash_ok=(None if hh is None else r["hash"] == hh),
            fe=r["fe"], npu=r["npu"], rc=r["rc"], guard=r["guard"],
            host_guard=(g["guard_below"] if g else None),
            host_guard_zero=(g["guard_zero"] if g else None),
            host_guard_frac=(g["guard_frac"] if g else None),
            rc_expect=(g["rc_expect"] if g else None),
            host_fnv_recomputed=(g["fnv"] if g else None),
        ))

    words = [r["nw"] for r in rows]
    hw = [max(r["n_host_words"], 1) for r in rows]
    he = [r["host_err"] for r in rows]
    de = [r["dev_err"] for r in rows]
    xe = [r["x_err"] for r in rows]
    diff = [a - b for a, b in zip(de, he)]

    def agg(errs, wds, S, I, D):
        tw, te = sum(wds), sum(errs)
        lo, hi, over = SC.wilson_ratio(te, tw)
        blo, bhi = SC.boot_wer(errs, wds, B, rng)
        return dict(S=sum(S), I=sum(I), D=sum(D), err=te, words=tw,
                    wer=te / tw if tw else float("nan"),
                    wilson=(lo, hi), over=over, boot=(blo, bhi))

    obs, blo, bhi, bp, _ = SC.paired_bootstrap(de, he, words, B, rng)
    perm_obs, perm_p = SC.signflip_test(diff, words, B, rng)
    sn, sk, sp = SC.sign_test(diff)

    nfr = sum(r["n_frames"] for r in rows)
    ndf = sum(r["n_diff"] for r in rows)
    res = dict(
        rows=rows, unknown=unknown, n_utt=len(rows),
        n_frames=nfr, n_diff=ndf,
        frame_rate=ndf / nfr if nfr else float("nan"),
        frame_wilson=wilson(ndf, nfr),
        host=agg(he, words, [r["host_S"] for r in rows], [r["host_I"] for r in rows],
                 [r["host_D"] for r in rows]),
        dev=agg(de, words, [r["dev_S"] for r in rows], [r["dev_I"] for r in rows],
                [r["dev_D"] for r in rows]),
        xhost=agg(xe, hw, [r["xS"] for r in rows], [r["xI"] for r in rows],
                  [r["xD"] for r in rows]),
        paired=dict(obs=obs, lo=blo, hi=bhi, p=bp, B=B,
                    mean_diff=float(np.mean(diff)) if diff else float("nan"),
                    worse=sum(1 for x in diff if x > 0),
                    better=sum(1 for x in diff if x < 0),
                    tied=sum(1 for x in diff if x == 0),
                    perm_p=perm_p, sign_n=sn, sign_k=sk, sign_p=sp, diff=diff),
        hash_match=[r["i"] for r in rows if r["hash_ok"] is True],
        hash_mismatch=[r["i"] for r in rows if r["hash_ok"] is False],
        hash_unknown=[r["i"] for r in rows if r["hash_ok"] is None],
        cycles=dict(fe=cost_stats([r["fe"] for r in rows], hz) if rows else None,
                    npu=cost_stats([r["npu"] for r in rows], hz) if rows else None),
        cycle_flags=cycle_flags({r["i"]: r for r in rows}),
        hz=hz,
    )
    if rows:
        ratios = [r["fe"] / r["npu"] for r in rows if r["npu"] > 0]
        res["ratio"] = dict(median=float(np.median(ratios)), min=float(min(ratios)),
                            max=float(max(ratios)),
                            of_medians=res["cycles"]["fe"]["median"] /
                            max(res["cycles"]["npu"]["median"], 1.0))
        tot = [r["fe"] + r["npu"] for r in rows]
        res["total"] = cost_stats(tot, hz)
    return res


# ------------------------------------------------------------------ the report
def report(res, side, feats, meta, out=sys.stdout):
    P = lambda *a: print(*a, file=out)
    rows = res["rows"]
    hz = res["hz"]
    P("=" * 78)
    P("WAVEFORM REPLAY -- M55 front end + NPU + C CTC, %d utterances" % res["n_utt"])
    P("=" * 78)
    for k, v in meta.items():
        P(f"  {k:<24} {v}")
    P(f"  {'sidecar':<24} {side['path']}  (N={side['N']}, {side['n_out_frames']} frames)")
    P(f"  {'host features':<24} {feats['path']}  md5 {feats['md5']}")
    if res["unknown"]:
        P(f"  !! indices in the capture that are not in the sidecar: {res['unknown']}")

    # ---- 1
    P("\n" + "-" * 78)
    P("1. FEATURE PARITY -- did the M55 compute the host's features?")
    P("-" * 78)
    P("   device FNV-1a over its own 64,000-byte int8 tensor, against the same hash")
    P("   over the host tensor in the feature blob.  Bit-exact or not; there is no")
    P("   'close' in a hash.")
    nm, nx = len(res["hash_match"]), len(res["hash_mismatch"])
    srcs = {}
    for r in rows:
        srcs[r["host_hash_source"]] = srcs.get(r["host_hash_source"], 0) + 1
    P("\n  host oracle: " + ", ".join(f"{v} x {k}" for k, v in srcs.items()))
    div = [r for r in rows if r["bytes_vs_blob"]]
    if div:
        P(f"\n  !! On {len(div)} utterance(s) the two host oracles are not the same")
        P("     tensor, so which one the device is scored against matters:")
        for r in div:
            P(f"       u{r['i']:<3} {r['k']:<22} {r['bytes_vs_blob']:>5} of 64,000 int8 "
              f"differ between the 128,000-sample window and corpus_blob.bin")
        P("     Cause, measured: gen_corpus.py builds a 127,841-sample buffer that ends")
        P("     on the last audio sample, so the 0.97 pre-emphasis never emits its")
        P("     one-sample ring-out.  The device's buffer is 128,000 samples with zeros")
        P("     after the audio, and citrinet_fe.c:226-230 emits x[i]-0.97*x[i-1] for")
        P("     every i < n_samples -- including the first zero.  That single sample")
        P("     lands in frame 799 and per-bin normalisation spreads it over the whole")
        P("     tensor.  Zeroing that one sample restores byte-identity exactly.")
        P("     It only bites utterances the 8 s window truncates; here the sidecar")
        P(f"     lists {len(side['raw'].get('truncated_utterances', []))} such utterances.")
        P("     The DEVICE's window is the primary oracle above, because the device is")
        P("     what this run measures.  Their host ids were recomputed to match ("
          + ", ".join(f"u{r['i']}: {r['host_ids_source']}" for r in div[:4]) + ").")
    P(f"\n  hashes identical    {nm} / {res['n_utt']}")
    P(f"  hashes differing    {nx} / {res['n_utt']}")
    if res["hash_unknown"]:
        P(f"  no host tensor for  {res['hash_unknown']}")
    if nx == 0 and nm == res["n_utt"] and nm:
        P("\n  VERDICT: bit-exact.  Every one of the %d device tensors is byte-identical" % nm)
        P("  to the tensor the workstation computes -- 64,000 int8 values each,")
        P("  {:,} values total.  This is the strongest available outcome: the M55".format(nm * TENSOR))
        P("  front end is not close to firmware/test/fe_parity.py's oracle, it is the")
        P("  same function of the same input, now measured on the real part.")
    else:
        P("\n  VERDICT: NOT bit-exact.  See section 5 for how to localise it.")
        for r in rows:
            if r["hash_ok"] is False:
                P(f"    u{r['i']:<3} {r['k']:<22} device {r['dev_hash']}  "
                  f"host {r['host_hash']}  ({r['host_hash_source']})")
                if r["matched"]:
                    P(f"         but it MATCHES another legitimate host tensor: "
                      f"{r['matched']}.")
                    P("         That is not an arithmetic fault.  The front end computed a")
                    P("         correct tensor for a different input window, so the thing to")
                    P("         fix is how the waveform blob was prepared, not citrinet_fe.c.")
                else:
                    P("         and matches none of the "
                      f"{r['n_oracles_differing']+1} distinct host tensor(s) considered: "
                      "this is a genuine front-end disagreement.")

    # ---- 2
    P("\n" + "-" * 78)
    P("2. COST -- front end and NPU, in cycles")
    P("-" * 78)
    P(f"  conversion: ms = cycles / {hz} * 1000.  {meta.get('clock source', '')}")
    P(f"  Cross-check on a published measurement: board/GATE4.md:1964 prints")
    P(f"  '{ROUND19_CYCLES} cycles = {ROUND19_MS} ms at 600000000 Hz'; the same")
    P(f"  arithmetic here gives {cycles_to_ms(ROUND19_CYCLES, 600_000_000):.3f} ms.")
    P("  Every number below is a per-invocation counter read reported directly.")
    P("  None of them is a wall-clock total divided by a count.")
    fe_s, np_s = res["cycles"]["fe"], res["cycles"]["npu"]
    P(f"\n  {'':<10}{'n':>4}{'min cy':>13}{'median cy':>13}{'max cy':>13}"
      f"{'min ms':>10}{'med ms':>10}{'max ms':>10}")
    for name, s in (("front end", fe_s), ("NPU", np_s), ("fe+NPU", res.get("total"))):
        if not s:
            continue
        P(f"  {name:<10}{s['n']:>4}{s['min']:>13}{s['median']:>13.0f}{s['max']:>13}"
          f"{s['min_ms']:>10.2f}{s['median_ms']:>10.2f}{s['max_ms']:>10.2f}")
    rt = res["ratio"]
    P(f"\n  front end / NPU cycles: median of the per-utterance ratio {rt['median']:.4f} "
      f"(min {rt['min']:.4f}, max {rt['max']:.4f})")
    P(f"  ratio of the medians {rt['of_medians']:.4f} -- the front end costs "
      f"{rt['of_medians']:.2f}x the NPU")
    P(f"  real-time factor: the window is 8.000 s of audio; fe+NPU median "
      f"{res['total']['median_ms']:.1f} ms = "
      f"{res['total']['median_ms']/8000.0:.4f} x real time")
    P(f"  per-utterance spread: fe {fe_s['spread_pct']:.3f} %, "
      f"NPU {np_s['spread_pct']:.3f} % of the median")

    d19 = 100.0 * (np_s["median"] - ROUND19_CYCLES) / ROUND19_CYCLES
    d20 = 100.0 * (np_s["median"] - ROUND20_CYCLES) / ROUND20_CYCLES
    P(f"\n  NPU median {np_s['median']:.0f} cy = {np_s['median_ms']:.1f} ms")
    P(f"    vs Round 19  {ROUND19_CYCLES} cy = {ROUND19_MS} ms  ({d19:+.1f} %)"
      "   canned tensor already in RAM")
    P(f"    vs Round 20  {ROUND20_CYCLES} cy = {ROUND20_MS} ms  ({d20:+.1f} %)"
      "   64,000 B memcpy'd from flash before each invoke")
    which = "Round 19" if abs(d19) < abs(d20) else "Round 20"
    P(f"  This run resembles {which}.")
    if which == "Round 19":
        P("  That is what the Round 20 hypothesis predicts (board/GATE4.md:2098): there")
        P("  the CPU read 64,000 B through xSPI2 immediately before every invoke, and")
        P("  the proposed mechanism was that this evicted weights the NPU would have")
        P("  found cached.  Here the features are produced by the M55 into RAM and no")
        P("  flash read precedes the invoke -- except the waveform read, which happens")
        P("  before ~%.0f ms of front-end work, far enough ahead to be irrelevant."
          % fe_s["median_ms"])
        P("  Consistent with the hypothesis; not a test of it.  A test would toggle the")
        P("  flash read with everything else fixed.")
    else:
        P("  That is NOT what the Round 20 hypothesis predicts (board/GATE4.md:2098).")
        P("  Nothing memcpy's features from flash before the invoke here, so if the cost")
        P("  is still ~%.0f ms the cache-eviction story is at best incomplete: the front"
          % np_s["median_ms"])
        P("  end's own traffic over %.0f ms and 256 kB of scratch is the other candidate"
          % fe_s["median_ms"])
        P("  for evicting the same weights.  Untested either way.")
    if res["cycle_flags"]:
        P(f"\n  !! {len(res['cycle_flags'])} cycle count(s) refused as measurements:")
        for i, f, v, why in res["cycle_flags"]:
            P(f"     u{i:<3} {f:<4} {v:>13}  {why}")
        P("     Bands are declared at score_wav.py:FE_MIN..NPU_MAX with their reasoning.")
    else:
        P("\n  every cycle count falls inside its plausibility band "
          f"(fe [{FE_MIN}, {FE_MAX}], npu [{NPU_MIN}, {NPU_MAX}])")

    # ---- 3
    P("\n" + "-" * 78)
    P("3. END TO END -- waveform -> features -> NPU -> text")
    P("-" * 78)
    P(f"  per-frame argmax agreement with the host: {res['n_diff']} of {res['n_frames']} "
      f"frames differ ({pct(res['frame_rate'])})")
    P(f"  WER is corpus-level: total errors / total reference words.")
    P(f"\n  {'comparison':<26}{'S':>5}{'I':>5}{'D':>5}{'err':>6}{'words':>7}   WER"
      f"       Wilson 95%           bootstrap 95%")
    for name, key in (("host vs reference", "host"), ("DEVICE vs reference", "dev"),
                      ("device vs host", "xhost")):
        a = res[key]
        P(f"  {name:<26}{a['S']:>5}{a['I']:>5}{a['D']:>5}{a['err']:>6}{a['words']:>7}   "
          f"{pct(a['wer']):<10}[{pct(a['wilson'][0])}, {pct(a['wilson'][1])}]  "
          f"[{pct(a['boot'][0])}, {pct(a['boot'][1])}]")
        if a["over"]:
            P("      (errors exceed reference words; Wilson clamped -- trust the bootstrap)")
    p = res["paired"]
    P(f"\n  paired per-utterance difference (device errors - host errors):")
    P(f"    mean {p['mean_diff']:+.4f} errors/utterance; device worse on {p['worse']}, "
      f"better on {p['better']}, tied on {p['tied']}")
    P(f"    corpus WER difference {100*p['obs']:+.3f} points, bootstrap 95% "
      f"[{100*p['lo']:+.3f}, {100*p['hi']:+.3f}] (B={p['B']}, unit = utterance), "
      f"p = {p['p']:.4f}")
    P(f"    sign-flip permutation p = {p['perm_p']:.4f}; exact sign test on the "
      f"{p['sign_n']} non-tied utterances ({p['sign_k']} worse) p = {p['sign_p']:.4f}")
    P(f"\n  {'i':>3} {'key':<20}{'nw':>4}{'dfr':>5}{'hErr':>5}{'dErr':>5}{'dif':>5}"
      f"  hash  same_text")
    for r in rows:
        P(f"  {r['i']:>3} {r['k']:<20}{r['nw']:>4}{r['n_diff']:>5}{r['host_err']:>5}"
          f"{r['dev_err']:>5}{r['dev_err']-r['host_err']:>+5}"
          f"  {'ok  ' if r['hash_ok'] else ('DIFF' if r['hash_ok'] is False else '?   ')}"
          f"  {'yes' if r['same_text'] else 'NO'}")
    P("\n  reference / host / device, every utterance:")
    for r in rows:
        P(f"    u{r['i']} {r['k']}{'  [TRUNCATED BY THE 8 s WINDOW]' if r['truncated'] else ''}")
        P(f"      ref : {r['ref'].lower()}")
        P(f"      host: {r['host_text']}")
        P(f"      dev : {r['dev_text']}" +
          ("" if r["same_text"] else "        <-- differs from host"))

    # ---- 4
    P("\n" + "-" * 78)
    P("4. GUARD TELEMETRY -- are the two front ends seeing the same levels?")
    P("-" * 78)
    P("  'guard' is citrinet_fe_t.guard_below (citrinet_fe.c:273): mel energies")
    P("  strictly below 2^-24, out of 80*800 = 64,000 per utterance.  The host count")
    P("  is the same predicate on the host's mel energies.  firmware/FRONTEND.md §5")
    P("  reports these matching exactly in all 60 host-parity runs, on counts from")
    P("  3,098 to 37,721, so exact agreement is the expectation, not a hope.")
    have = [r for r in rows if r["host_guard"] is not None]
    pred = [r for r in rows if r["rc_expect"] == FE_E_GUARD]
    if pred:
        P(f"\n  PREDICTED BEFORE THE RUN: {len(pred)} of {len(rows)} utterance(s) exceed")
        P(f"  CITRINET_FE_GUARD_MAX_FRAC (0.50) on the host, so citrinet_fe_run() returns")
        P("  CITRINET_FE_E_GUARD (-4) for them and firmware that refuses to invoke on")
        P("  rc != 0 prints no ids at all:")
        for r in pred:
            P(f"    u{r['i']:<3} {r['k']:<22} non-silent guard fraction "
              f"{100*r['host_guard_frac']:.2f} %  (device rc {r['rc']})")
        P("  This is the corpus, not the front end: the utterance is mostly silence")
        P("  inside the 8 s window, so few bins carry signal and the ones that do sit")
        P("  low.  An absent utterance here is not a hang.")
    if not have:
        P("\n  NO HOST GUARD REFERENCE.  Rebuild it with --build-host-ref (needs the")
        P("  audio and the zoo venv); only the device column is available:")
        gs = sorted(r["guard"] for r in rows)
        P(f"    device guard: min {gs[0]}  median {int(np.median(gs))}  max {gs[-1]}")
    else:
        ex = sum(1 for r in have if r["guard"] == r["host_guard"])
        dif = [abs(r["guard"] - r["host_guard"]) for r in have]
        P(f"\n  exact agreement on {ex} / {len(have)} utterances; "
          f"max |device - host| = {max(dif)}")
        P(f"  {'i':>3} {'key':<20}{'device':>9}{'host':>9}{'diff':>7}{'zero':>8}"
          f"{'frac':>8}{'rc':>5}{'rc exp':>7}")
        for r in have:
            P(f"  {r['i']:>3} {r['k']:<20}{r['guard']:>9}{r['host_guard']:>9}"
              f"{r['guard']-r['host_guard']:>+7}{r['host_guard_zero']:>8}"
              f"{100*r['host_guard_frac']:>7.2f}%{r['rc']:>5}{r['rc_expect']:>7}")
        bad_rc = [r for r in have if r["rc"] != r["rc_expect"]]
        if bad_rc:
            P(f"  !! rc disagrees with the host's prediction on {len(bad_rc)}: "
              f"{[r['i'] for r in bad_rc]}")
            P("     rc is CITRINET_FE_E_GUARD (-4) exactly when the non-silent guard")
            P("     fraction exceeds %.2f (citrinet_fe.c:389-415)." % GUARD_MAX_FRAC)
        else:
            P(f"  rc matches the host's prediction on all {len(have)} "
              f"(CITRINET_FE_E_GUARD would be -4 above a {GUARD_MAX_FRAC:.2f} non-silent "
              "fraction)")
        if max(dif) == 0:
            P("  The two front ends see identical levels, bin for bin.")
        elif max(dif) <= 8:
            P("  Small disagreements (<= 8 bins) are bins sitting exactly on 2^-24 where")
            P("  a different float32 summation order in the mel matmul tips the")
            P("  comparison.  They are not evidence of a level difference.")
        else:
            P("  A disagreement this large means the two front ends are NOT seeing the")
            P("  same levels -- the first thing to check before anything downstream.")
    rcs = {}
    for r in rows:
        rcs[r["rc"]] = rcs.get(r["rc"], 0) + 1
    P(f"\n  return codes: " + ", ".join(
        f"{k} ({'OK' if k == 0 else 'E_GUARD' if k == -4 else 'unexpected'}): {v}"
        for k, v in sorted(rcs.items())))

    # ---- 5
    P("\n" + "-" * 78)
    P("5. IF A HASH DISAGREES -- how to localise it")
    P("-" * 78)
    if not res["hash_mismatch"]:
        P("  Not needed: every hash matched.  The instrument below exists anyway and")
        P("  is UNTESTED until a real mismatch exercises it.")
    else:
        P(f"  Needed: {len(res['hash_mismatch'])} utterance(s) disagree -- "
          f"{res['hash_mismatch']}.")
    P("")
    P("  A whole-tensor hash says 'different' and nothing else.  The next run should")
    P("  print, for each failing utterance, ONE hash per mel row:")
    P("")
    P('      for (int b = 0; b < 80; b++) {')
    P('          uint32_t h = 2166136261u;')
    P('          for (int t = 0; t < 800; t++) { h ^= (uint8_t)out[b*800 + t];')
    P('                                          h *= 16777619u; }')
    P('          my_printf("# row %d %d %08x\\n", i, b, (unsigned)h);')
    P('      }')
    P("")
    P("  80 rows x 800 bytes, mel-major, the same FNV-1a.  The host side of that diff")
    P("  is implemented here and needs no further host work: --row-hashes PATH writes")
    P("  {index: {key, rows:[80 hex]}} for every utterance in the capture.")
    P("  Reading the result: rows are mel bins.  A handful of adjacent failing rows")
    P("  points at the mel filterbank or its table; all 80 failing points at the FFT,")
    P("  the pre-emphasis or the input samples; rows failing only where the filter")
    P("  weights are smallest points at the log guard.  Per-bin normalisation mixes")
    P("  the whole row, so a single bad frame still shows as one whole bad row --")
    P("  which is why the next step after that is a frame-range hash, not a finer one.")
    P("")
    P("  UNTESTED PATH: no firmware prints row hashes today, and nothing has ever")
    P("  been diffed against the host side of this.  It is written now so that a")
    P("  failing run costs one rebuild instead of a rebuild plus a host script.")

    # ---- 6
    P("\n" + "-" * 78)
    P("6. VERDICT")
    P("-" * 78)
    sup, uns = [], []
    if nx == 0 and nm:
        sup.append(f"the M55 front end reproduces the host's int8 features BIT-EXACTLY on "
                   f"{nm}/{res['n_utt']} utterances ({nm*TENSOR} int8 values)")
    else:
        uns.append(f"feature parity FAILED on {nx} utterance(s): {res['hash_mismatch']}")
    sup.append(f"front end median {fe_s['median_ms']:.1f} ms, NPU median "
               f"{np_s['median_ms']:.1f} ms, both per-invocation counter reads at "
               f"{hz/1e6:.0f} MHz")
    sup.append(f"the front end costs {rt['of_medians']:.2f}x the NPU; the pipeline is "
               f"{res['total']['median_ms']/8000.0:.3f}x real time on an 8 s window")
    sup.append(f"device text equals host text on {sum(1 for r in rows if r['same_text'])} "
               f"of {len(rows)} utterances; device WER {pct(res['dev']['wer'])} vs host "
               f"{pct(res['host']['wer'])}, paired difference {100*p['obs']:+.3f} points "
               f"(p = {p['p']:.4f})")
    if have:
        sup.append(f"guard counters agree exactly on {ex}/{len(have)} utterances "
                   f"(max |diff| {max(dif)}): the two front ends see the same levels")
    if any(r["bytes_vs_blob"] for r in rows):
        uns.append("on the utterances the 8 s window truncates, the host tensor scored "
                   "against is NOT corpus_blob.bin -- it is a recomputation over the "
                   "device's 128,000-sample window, which differs by one sample of "
                   "pre-emphasis ring-out; their host ids were recomputed to match")
    uns.append("nothing here involves the microphone or the gain stage: the waveforms are "
               "canned, at the level gen_corpus.py's source records carry")
    uns.append("the front-end cost is for THIS build (CMSIS arm_rfft_fast_f32, float32 "
               "scratch, 256,000 B); firmware/FRONTEND.md §5 measures fp16 scratch as "
               "0.52 % of int8 values differing, so halving the buffer is not free")
    uns.append(f"{res['n_utt']} utterances of clean read speech <= 7.9 s; the paired CI is "
               f"{100*(p['hi']-p['lo']):.2f} points wide, so a true difference smaller "
               f"than that is invisible here")
    if not res["cycle_flags"]:
        uns.append("the cycle counts are self-consistent and inside their bands, which is "
                   "not the same as the counter being correctly configured; only the "
                   "cross-check against Round 19's published 124.0 ms constrains that")
    P("\n  SUPPORTED by these numbers:")
    for s in sup:
        P(f"    - {s}")
    P("\n  NOT supported by these numbers:")
    for s in uns:
        P(f"    - {s}")
    P("=" * 78)


# ------------------------------------------------------------------ helpers
def missing_indices(expect, recs):
    """Indices the run should have printed and did not."""
    return sorted(set(expect) - set(recs))


def build_host_ref_from_wavs(side, feats, wavs, verbose=True, want_ids=True):
    """Same, but the input samples are the bytes actually in the flashed waveform
    blob.  This is the stronger oracle when the blob is available: it removes every
    assumption about how the blob was prepared, including the one the variants
    above exist to hedge."""
    scale = side["raw"]["scale"]
    raw_u = {u["index"]: u for u in side["raw"]["utterances"]}
    out = {}
    for i in sorted(wavs):
        ent = _host_entry(side, feats, i, raw_u[i]["key"] if i in raw_u else "?",
                          np.asarray(wavs[i], dtype=np.int16), None, scale, want_ids)
        ent.update(int16_round_residual=0.0, source="wav blob")
        out[i] = ent
        if verbose:
            print(f"  [{i:3d}] guard {ent['guard_below']:>6} zero {ent['guard_zero']:>6}  "
                  f"fnv {ent['fnv']}  vs corpus_blob.bin: "
                  f"{'identical' if ent['bytes_vs_feature_blob'] == 0 else str(ent['bytes_vs_feature_blob']) + ' int8 differ'}",
                  flush=True)
    return out


def resolve_clock(pr, args):
    """The ms conversion must not rest on an assumption.  Order: the capture
    itself, then an explicit --hz, then board/GATE4.md."""
    if pr and pr.get("hz"):
        return pr["hz"], f"read from the capture ({pr['hz']} Hz)"
    if args and getattr(args, "hz", None):
        return args.hz, f"given on the command line (--hz {args.hz})"
    g = os.path.join(REPO, "board", "GATE4.md")
    hz, src = None, None
    if os.path.exists(g):
        txt = open(g, encoding="utf-8", errors="replace").read()
        m = SYSCLK_RE.search(txt)
        m2 = re.search(r"#\s*invoke\s+(\d+)\s+cycles\s*=\s*([\d.]+)\s*ms\s+at\s+(\d+)\s*Hz",
                       txt)
        if m:
            hz = int(m.group(1)) * 1_000_000
            src = f"board/GATE4.md 'SYSCLK clock : {m.group(1)} MHz'"
        if m2:
            hz2 = int(m2.group(3))
            implied = 1000.0 * int(m2.group(1)) / hz2
            ok = abs(implied - float(m2.group(2))) < 0.005
            src = (f"board/GATE4.md 'SYSCLK clock : {m.group(1)} MHz' and its printed "
                   f"'{m2.group(1)} cycles = {m2.group(2)} ms at {hz2} Hz' "
                   f"({'consistent' if ok else 'INCONSISTENT'}: {implied:.3f} ms implied)")
            if hz is None or hz == hz2:
                hz = hz2
            else:
                src += f"  !! disagrees with SYSCLK {hz} Hz"
    if hz is None:
        hz, src = DEFAULT_HZ, "fallback default (nothing found to verify against)"
    return hz, src


# ------------------------------------------------------------------ self-test
def _fmt_wav_log(recs, ansi=False, wrap=0, done=True, banner=False, omit=(),
                 truncate=None, truncate_ids=None):
    """Synthesise a capture in the contract's format."""
    out = []
    if banner:
        out.append("SYSCLK clock : 600 MHz    HCLK clock : 400 MHz")
    for i in sorted(recs):
        if i in omit:
            continue
        r = recs[i]
        head = (f"# w {i} fe {r['fe']} npu {r['npu']} rc {r['rc']} guard {r['guard']} "
                f"hash {r['hash']}")
        if truncate is not None and i == truncate:
            out.append(head + " guard")          # cut mid-record: no ids: at all
            continue
        ids = [str(int(x)) for x in r["ids"]]
        if truncate_ids is not None and i == truncate_ids:
            out.append(head + " ids: " + " ".join(ids[:40]))
            continue
        if wrap:
            out.append(head + " ids: " + " ".join(ids[:wrap]))
            k = wrap
            while k < len(ids):
                out.append(" ".join(ids[k:k + wrap]))
                k += wrap
        elif ansi:
            out.append("\x1b[0m" + head.replace("# w", "\x1b[32m# w") + " ids:\x1b[0m" +
                       "".join("\x1b[0m %s\x1b[0m" % v for v in ids))
        else:
            out.append(head + " ids: " + " ".join(ids))
    if done:
        out.append("# wav done")
    return "\r\n".join(out) + "\r\n"


def self_test(sidecar, features_path, B=2000, subset=(0, 1, 2, 23, 30)):
    fails = []
    _c = lambda name, got, want: SC._check(name, got, want, fails)
    print("=" * 78)
    print("SELF-TEST -- score_wav.py is shown wrong answers it already knows")
    print("=" * 78)

    # ---------------- A. primitives
    print("\n  A. primitives")
    _c("FNV-1a('') == 0x811c9dc5", fhex(fnv1a(b"")), "811c9dc5")
    _c("FNV-1a('a') == 0xe40c292c", fhex(fnv1a(b"a")), "e40c292c")
    _c("FNV-1a('foobar') == 0xbf9cf968", fhex(fnv1a(b"foobar")), "bf9cf968")
    _c("hex format keeps leading zeros", fhex(0x00abcdef), "00abcdef")
    _c("hex format is lower case", fhex(0xDEADBEEF), "deadbeef")
    _c("levenshtein 1 sub", levenshtein("a b c".split(), "a x c".split()), (1, 0, 0))
    _c("levenshtein 1 ins", levenshtein("a b c".split(), "a b x c".split()), (0, 1, 0))
    _c("levenshtein 1 del", levenshtein("a b c".split(), "a c".split()), (0, 0, 1))
    _c("ctc collapse+blank", ctc_greedy([1024, 5, 5, 1024, 5, 7, 7, 1024], 1024), [5, 5, 7])
    _c("guard_fraction excludes the exactly-zero pad (30000 below, 20000 zero)",
       round(guard_fraction(30000, 20000), 6), round(10000 / 44000, 6))
    _c("guard_fraction with zero == total does not divide by zero",
       guard_fraction(64000, 64000), 1.0)
    _c("guard_fraction of a clean utterance", guard_fraction(0, 0), 0.0)
    ms = cycles_to_ms(ROUND19_CYCLES, 600_000_000)
    _c("74421588 cy at 600 MHz == board/GATE4.md's 124.035 ms (within 5 us)",
       abs(ms - ROUND19_MS) < 0.005, True)
    hz, src = resolve_clock(None, None)
    print(f"      clock provenance: {src}")
    _c("clock recovered without assuming it", hz, 600_000_000)

    # ---------------- B. the host oracle
    print("\n  B. host oracle")
    side = load_sidecar(sidecar)
    feats = load_features(features_path)
    want_md5 = side["digest"][1]
    _c("feature blob md5 == the sidecar's", feats["md5"], want_md5)
    _c("feature blob N == sidecar N", feats["N"], side["N"])
    _c("tensor size", len(feats["feats"][0]), TENSOR)
    _c("vectorised FNV-1a == the byte-loop FNV-1a on a 64,000-byte tensor",
       fhex(fnv1a_np(feats["feats"][0])), fhex(fnv1a(feats["feats"][0])))
    rh = row_hashes(feats["feats"][0])
    _c("80 row hashes", len(rh), NMEL)
    t = bytearray(feats["feats"][0])
    t[17 * T + 400] = (t[17 * T + 400] + 1) & 0xFF
    rh2 = row_hashes(bytes(t))
    _c("perturbing one byte changes exactly one row hash",
       [r for r in range(NMEL) if rh[r] != rh2[r]], [17])
    _c("and changes the whole-tensor hash",
       fhex(fnv1a(bytes(t))) != fhex(fnv1a(feats["feats"][0])), True)

    hostref = {}
    try:
        print(f"     recomputing the host front end from audio for {list(subset)} "
              f"(int16 window, the same bytes the blob carries):")
        hostref = build_host_ref(side, feats, list(subset))
    except Exception as e:                                    # noqa: BLE001
        print(f"     SKIPPED host recomputation: {type(e).__name__}: {e}")
    if hostref:
        trunc_keys = set(side["raw"].get("truncated_utterances", []))
        tr = [i for i in hostref if hostref[i]["key"] in trunc_keys]
        nt = [i for i in hostref if hostref[i]["key"] not in trunc_keys]
        _c("recomputed int8 == corpus_blob.bin byte for byte on every utterance the "
           "8 s window does NOT truncate",
           sorted({hostref[i]["bytes_vs_feature_blob"] for i in nt}), [0])
        _c("float->int16 rounding residual is exactly 0 (soundfile's float IS n/32768)",
           max(v["int16_round_residual"] for v in hostref.values()), 0.0)
        _c("recomputed FNV-1a == the feature blob's FNV-1a on those",
           [i for i in nt if hostref[i]["fnv"] != fhex(fnv1a(feats["feats"][i]))], [])
        if tr:
            _c("a truncated utterance's two host windows are NOT the same tensor "
               "(the pre-emphasis ring-out at sample 127,841)",
               [i for i in tr if hostref[i]["bytes_vs_feature_blob"] == 0], [])
            _c("and its host ids were recomputed for the device's window",
               [i for i in tr if "host_ids" not in hostref[i]], [])
            _c("the recomputed ids are a full frame vector",
               sorted({len(hostref[i]["host_ids"]) for i in tr}),
               [side["n_out_frames"]])
            _c("the fill_128000 variant is offered for them too",
               [i for i in tr if "fill_128000" not in hostref[i]["variants"]], [])
            print("      truncated members of the subset: " +
                  ", ".join("u%d %s (%d int8 differ from corpus_blob.bin)"
                            % (i, hostref[i]["key"], hostref[i]["bytes_vs_feature_blob"])
                            for i in sorted(tr)))
        print("      guard counts: " +
              ", ".join(f"u{i} {hostref[i]['guard_below']}" for i in sorted(hostref)))

    # ---------------- build a clean synthetic capture
    utts = {u["i"]: u for u in side["utts"]}
    clean = {}
    for i in sorted(utts):
        if i not in feats["feats"]:
            continue
        clean[i] = dict(i=i,
                        ids=list(hostref[i]["host_ids"]) if i in hostref
                        and hostref[i].get("host_ids") else list(utts[i]["host_ids"]),
                        hash=(hostref[i]["fnv"] if i in hostref
                              else fhex(fnv1a(feats["feats"][i]))),
                        fe=31_400_000 + 137 * i, npu=74_400_000 + 211 * i, rc=0,
                        guard=(hostref[i]["guard_below"] if i in hostref
                               else 20_000 + 3 * i))
    N = len(clean)

    # ---------------- C. parsing
    print("\n  C. parsing")
    pr = parse_log(_fmt_wav_log(clean, banner=True), side["n_out_frames"])
    _c("passes found", len(pr["passes"]), 1)
    _c("records parsed", len(pr["passes"][0]) if pr["passes"] else 0, N)
    _c("no malformed lines", len(pr["bad"]), 0)
    _c("clock read out of the capture banner", pr["hz"], 600_000_000)
    _c("fields recovered exactly",
       {k: pr["passes"][0][5][k] for k in ("fe", "npu", "rc", "guard", "hash")},
       {k: clean[5][k] for k in ("fe", "npu", "rc", "guard", "hash")})
    _c("ids recovered exactly", pr["passes"][0][5]["ids"], clean[5]["ids"])
    pa = parse_log(_fmt_wav_log(clean, ansi=True), side["n_out_frames"])
    _c("ESC[0m between every number survives",
       (len(pa["passes"]), len(pa["passes"][0]) if pa["passes"] else 0), (1, N))
    _c("ANSI-stripped ids identical", pa["passes"][0][7]["ids"] if pa["passes"] else None,
       clean[7]["ids"])
    pw = parse_log(_fmt_wav_log(clean, wrap=37), side["n_out_frames"])
    _c("terminal hard-wrap rejoined",
       (len(pw["passes"]), len(pw["passes"][0]) if pw["passes"] else 0), (1, N))
    _c("wrapped ids identical", pw["passes"][0][3]["ids"] if pw["passes"] else None,
       clean[3]["ids"])
    p3 = parse_log(_fmt_wav_log(clean) * 3, side["n_out_frames"])
    ident, notes = compare_passes(p3["passes"])
    _c("3 repeats split and found identical", (len(p3["passes"]), ident), (3, True))
    var = {i: dict(v) for i, v in clean.items()}
    var[4] = dict(var[4]); var[4]["hash"] = "deadbeef"
    ident2, notes2 = compare_passes(parse_log(_fmt_wav_log(clean) + _fmt_wav_log(var),
                                              side["n_out_frames"])["passes"])
    _c("a repeat whose hash changed is flagged", ident2, False)
    print(f"      note: {notes2[0] if notes2 else '(none)'}")
    cyc_only = {i: dict(v, fe=v["fe"] + 900, npu=v["npu"] - 700) for i, v in clean.items()}
    identc, _ = compare_passes(parse_log(_fmt_wav_log(clean) + _fmt_wav_log(cyc_only),
                                         side["n_out_frames"])["passes"])
    _c("cycle jitter alone does NOT count as a differing repeat", identc, True)
    _c("a record with 7 hex digits of hash is refused",
       len(parse_log("# w 1 fe 1 npu 2 rc 0 guard 3 hash abcdef1 ids: " +
                     " ".join(["0"] * 100) + "\n", 100)["bad"]), 1)
    _c("a record with an out-of-range id is refused",
       len(parse_log("# w 1 fe 1 npu 2 rc 0 guard 3 hash abcdef12 ids: " +
                     " ".join(["9999"] * 100) + "\n", 100)["bad"]), 1)
    _c("a line with no id list at all is refused",
       len(parse_log("# w 1 fe 1 npu 2 rc 0 guard 3 hash abcdef12\n", 100)["bad"]), 1)
    _c("junk with no records gives no passes",
       len(parse_log("hello\n# wav done\n", 100)["passes"]), 0)

    # ---------------- D. injected faults
    print("\n  D. injected faults")
    HASH_BAD, SUB, CYC_NPU, CYC_FE, MISS, TRUNC, TRUNC_IDS, GUARD_BAD, RC_BAD = \
        3, 7, 11, 12, 20, 25, 26, 30, 1
    WINDOW_BAD = next((i for i in hostref
                       if hostref[i]["key"] in
                       set(side["raw"].get("truncated_utterances", []))), None)
    dev = {i: dict(v, ids=list(v["ids"])) for i, v in clean.items()}
    inj = {}

    # (1) one wrong hash -- the front end computed different features
    good = dev[HASH_BAD]["hash"]
    dev[HASH_BAD]["hash"] = fhex((int(good, 16) ^ 0x00000001))
    inj["wrong_hash_utterance"] = HASH_BAD
    inj["wrong_hash_value"] = dev[HASH_BAD]["hash"]

    # (2) one token substitution -- same features, the NPU emitted a different id
    ids = dev[SUB]["ids"]
    f = next(t for t in range(1, len(ids) - 1)
             if ids[t] != BLANK and ids[t - 1] != (ids[t] + 1) % BLANK
             and ids[t + 1] != (ids[t] + 1) % BLANK)
    inj["sub_utterance"], inj["sub_frame"] = SUB, f
    inj["sub_from"], inj["sub_to"] = ids[f], (ids[f] + 1) % BLANK
    ids[f] = inj["sub_to"]

    # (3) two impossible cycle counts
    dev[CYC_NPU]["npu"] = 12345
    dev[CYC_FE]["fe"] = 0
    inj["implausible"] = [(CYC_NPU, "npu", 12345), (CYC_FE, "fe", 0)]

    # (4) a guard count that disagrees with the host
    if GUARD_BAD in hostref:
        dev[GUARD_BAD]["guard"] = hostref[GUARD_BAD]["guard_below"] + 5000
        inj["guard_offset"] = (GUARD_BAD, 5000)

    # (5) a return code the guard fraction does not justify
    if RC_BAD in hostref:
        dev[RC_BAD]["rc"] = FE_E_GUARD
        inj["rc_bad"] = (RC_BAD, FE_E_GUARD, hostref[RC_BAD]["rc_expect"])

    # (6) a device that computed the OTHER legitimate window: right arithmetic,
    #     wrong input.  It must be diagnosed as that, not as a front-end fault.
    if WINDOW_BAD is not None:
        dev[WINDOW_BAD]["hash"] = fhex(fnv1a(feats["feats"][WINDOW_BAD]))
        inj["wrong_window_utterance"] = WINDOW_BAD

    print(f"     injected: {json.dumps(inj)}")
    print(f"     plus: utterance {MISS} omitted, utterance {TRUNC} truncated mid-record, "
          f"utterance {TRUNC_IDS} truncated mid-id-list")

    log = _fmt_wav_log(dev, banner=True, omit=(MISS,), truncate=TRUNC,
                       truncate_ids=TRUNC_IDS)
    pr = parse_log(log, side["n_out_frames"])
    recs = pr["passes"][0] if pr["passes"] else {}

    # (6) missing + (7) truncated
    _c("every record that did not reach the scored pass is reported missing",
       missing_indices(clean, recs), sorted([MISS, TRUNC, TRUNC_IDS]))
    _c("and only the omitted one is missing WITHOUT a malformed-line explanation",
       sorted(set(missing_indices(clean, recs)) - {b[0] for b in pr["bad"]}), [MISS])
    _c("truncated records are reported malformed, not silently dropped",
       sorted(b[0] for b in pr["bad"]), [TRUNC, TRUNC_IDS])
    byidx = {b[0]: b[1] for b in pr["bad"]}
    _c("mid-record truncation names the reason", "no 'ids:' field" in byidx[TRUNC], True)
    _c("short id list names the count", byidx[TRUNC_IDS], "40 ids, expected 100")
    _c("neither truncated record leaks into the scored pass",
       [i for i in (TRUNC, TRUNC_IDS) if i in recs], [])
    _c("every other record survived", len(recs), N - 3)

    res = score(side, feats, recs, hostref, B=B, seed=1, hz=600_000_000)
    by = {r["i"]: r for r in res["rows"]}

    # (1)
    want_bad = sorted([HASH_BAD] + ([WINDOW_BAD] if WINDOW_BAD is not None else []))
    _c("exactly the injected utterances fail feature parity",
       res["hash_mismatch"], want_bad)
    _c("every other hash matches", len(res["hash_match"]), len(recs) - len(want_bad))
    if WINDOW_BAD is not None:
        _c("the wrong-window utterance is diagnosed as matching another legitimate "
           "host tensor, not as a front-end fault",
           by[WINDOW_BAD]["matched"].startswith("corpus_blob.bin"), True)
        _c("the genuinely wrong hash matches nothing", by[HASH_BAD]["matched"], None)
    _c("the reported device hash is the injected one",
       by[HASH_BAD]["dev_hash"], inj["wrong_hash_value"])
    _c("the reported host hash is the feature blob's", by[HASH_BAD]["host_hash"], good)

    # (2)
    _c("exactly one utterance has differing frames",
       [r["i"] for r in res["rows"] if r["n_diff"]], [SUB])
    _c("and exactly one frame in it", by[SUB]["n_diff"], 1)
    _c("its transcript no longer equals the host's (same_text)",
       by[SUB]["same_text"], False)
    _c("device-vs-host edit distance on it is >= 1", by[SUB]["x_err"] >= 1, True)
    _c("the substituted utterance's FEATURES still match "
       "(the two failure axes are separated)", by[SUB]["hash_ok"], True)
    _c("total frame disagreements == 1", res["n_diff"], 1)

    # (3)
    _c("exactly the injected cycle counts are refused",
       sorted((i, f) for i, f, v, w in res["cycle_flags"]),
       sorted([(CYC_NPU, "npu"), (CYC_FE, "fe")]))
    _c("the refused values are reported verbatim",
       sorted((i, v) for i, f, v, w in res["cycle_flags"]),
       sorted([(CYC_NPU, 12345), (CYC_FE, 0)]))

    # (4) (5)
    if GUARD_BAD in hostref:
        _c("the guard disagreement is reported with its exact size",
           by[GUARD_BAD]["guard"] - by[GUARD_BAD]["host_guard"], 5000)
        _c("no other utterance with a host reference disagrees on guard",
           [r["i"] for r in res["rows"] if r["host_guard"] is not None
            and r["guard"] != r["host_guard"]], [GUARD_BAD])
    if RC_BAD in hostref:
        _c("the unjustified return code is caught",
           (by[RC_BAD]["rc"], by[RC_BAD]["rc_expect"]), (FE_E_GUARD, FE_OK))
    # positive control for the rc prediction: it must also say -4 when the level
    # really does justify it, or it is just a constant.
    hi = [i for i in hostref if hostref[i]["guard_frac"] > GUARD_MAX_FRAC]
    if hi:
        _c("rc prediction is not stuck on OK: it says E_GUARD for the utterances "
           "whose measured non-silent guard fraction exceeds 0.50",
           sorted({hostref[i]["rc_expect"] for i in hi}), [FE_E_GUARD])
        print("      NEGATIVE RESULT, and it is about the run, not the scorer: "
              + ", ".join("u%d %s %.2f %%" % (i, hostref[i]["key"],
                                              100 * hostref[i]["guard_frac"]) for i in hi)
              + " exceed CITRINET_FE_GUARD_MAX_FRAC (0.50), so citrinet_fe_run() will")
        print("      return CITRINET_FE_E_GUARD (-4) on them and a firmware that refuses "
              "to invoke on rc != 0 will print no ids for those utterances.")

    # WER bookkeeping
    tot_err = sum(r["dev_err"] for r in res["rows"])
    tot_w = sum(r["nw"] for r in res["rows"])
    _c("corpus WER == total errors / total reference words",
       round(res["dev"]["wer"], 12), round(tot_err / tot_w, 12))
    _c("device errors == host errors + the damage from the one substitution",
       res["dev"]["err"] - res["host"]["err"],
       by[SUB]["dev_err"] - by[SUB]["host_err"])
    _c("paired worse+better+tied == utterances scored",
       sum(res["paired"][k] for k in ("worse", "better", "tied")), res["n_utt"])

    # ---------------- E. control
    print("\n  E. control -- a capture with nothing wrong in it")
    pc = parse_log(_fmt_wav_log(clean, banner=True), side["n_out_frames"])
    r0 = score(side, feats, pc["passes"][0], hostref, B=B, seed=1, hz=600_000_000)
    _c("no malformed lines", len(pc["bad"]), 0)
    _c("nothing missing", missing_indices(clean, pc["passes"][0]), [])
    _c("every hash matches", r0["hash_mismatch"], [])
    _c("no frame disagreements", r0["n_diff"], 0)
    _c("no cycle count refused", r0["cycle_flags"], [])
    _c("device WER == host WER", r0["dev"]["wer"] == r0["host"]["wer"], True)
    _c("paired difference == 0", r0["paired"]["obs"], 0.0)
    _c("device-vs-host WER == 0", r0["xhost"]["err"], 0)
    if hostref:
        _c("guard agrees everywhere a host reference exists",
           [r["i"] for r in r0["rows"] if r["host_guard"] is not None
            and r["guard"] != r["host_guard"]], [])
    fe_ms = r0["cycles"]["fe"]["median_ms"]
    npu_ms = r0["cycles"]["npu"]["median_ms"]
    print(f"      synthetic cost readout: fe {fe_ms:.2f} ms, NPU {npu_ms:.2f} ms, "
          f"ratio {r0['ratio']['of_medians']:.3f}")
    _c("the synthetic NPU median is called Round-19-like",
       abs(r0["cycles"]["npu"]["median"] - ROUND19_CYCLES) <
       abs(r0["cycles"]["npu"]["median"] - ROUND20_CYCLES), True)

    # ---------------- F. the host side is the host side
    print("\n  F. the host reference reproduces the published host numbers")
    same_src = [r for r in r0["rows"] if r["host_ids_source"] == "sidecar"]
    _c("host text decoded here == host text in the sidecar, wherever the sidecar's "
       "ids are the ones that describe the device's window",
       [r["i"] for r in same_src if r["host_text"] != r["sidecar_text"]], [])
    resc = [r for r in r0["rows"] if r["host_ids_source"] != "sidecar"]
    _c("and the utterances whose ids were recomputed are exactly the ones whose "
       "window differs from corpus_blob.bin",
       sorted(r["i"] for r in resc),
       sorted(i for i in hostref if hostref[i]["bytes_vs_feature_blob"]))
    for r in resc:
        same = r["host_text"] == r["sidecar_text"]
        print(f"      u{r['i']} {r['k']}: recomputed host text "
              f"{'is identical to' if same else 'DIFFERS from'} the sidecar's")
        if not same:
            print(f"        sidecar   : {r['sidecar_text']}")
            print(f"        recomputed: {r['host_text']}")
    # Golden: the published host WER for this corpus.  README.md and board/GATE4.md
    # Round 20 both quote 5.92 % for the host over these 64 utterances.  Recompute
    # it from the sidecar's own ids, through this file's decode and Levenshtein.
    vocab = [l.rsplit(" ", 1)[0] for l in
             open(os.path.join(REPO, "tokenizer", "vocab.txt"), encoding="utf-8")
             .read().split("\n") if l.strip()]
    terr = tw = 0
    for u in side["utts"]:
        txt = ids_to_text(ctc_greedy(u["host_ids"], side["blank"]), vocab)
        S, I, D = levenshtein(u["ref"].lower().split(), txt.split())
        terr += S + I + D
        tw += len(u["ref"].lower().split())
    _c("host corpus WER from the sidecar's ids == the 5.92 % published in "
       "README.md / board/GATE4.md Round 20", round(100 * terr / tw, 2), 5.92)
    _c("host WER scored here == the same number, decoded through the capture path",
       round(100 * r0["host"]["wer"], 2), 5.92)
    print(f"      {terr} errors over {tw} reference words")

    print("\n" + "=" * 78)
    if fails:
        print(f"SELF-TEST FAILED: {len(fails)} checks -- {fails}")
    else:
        print("SELF-TEST PASSED: every injected fault was reported exactly as injected.")
    print("=" * 78)
    return 1 if fails else 0


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help="captured UART text from the waveform-replay run")
    ap.add_argument("--sidecar", default="artifacts/corpus/corpus_ref.json")
    ap.add_argument("--features", help="host int8 feature blob "
                                       "(default: the sidecar's blob path)")
    ap.add_argument("--wav-blob", help="the flashed STTW waveform blob; header-checked, "
                                       "and used as the host oracle by --build-host-ref")
    ap.add_argument("--host-ref", default="artifacts/corpus/wav_host_ref.json",
                    help="cache of host guard counts + hashes (built by --build-host-ref)")
    ap.add_argument("--build-host-ref", action="store_true",
                    help="recompute the host front end (needs librosa/soundfile) and "
                         "write --host-ref; also byte-compares against the feature blob")
    ap.add_argument("--indices", help="restrict --build-host-ref, e.g. 0-7,30")
    ap.add_argument("--expect-n", type=int,
                    help="how many utterances the run should have printed "
                         "(default: the wav blob's N, else the sidecar's N)")
    ap.add_argument("--hz", type=int, help="CPU clock, when the capture does not say")
    ap.add_argument("--pass", dest="which", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", help="also dump the numbers here")
    ap.add_argument("--row-hashes", help="write the per-mel-row host hashes here "
                                         "(section 5's follow-up instrument)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    side_path = a.sidecar if os.path.isabs(a.sidecar) else os.path.join(REPO, a.sidecar)
    side = load_sidecar(side_path)
    fpath = a.features or os.path.join(REPO, side["blob_path"])
    if not os.path.isabs(fpath):
        fpath = os.path.join(REPO, fpath)

    if a.self_test:
        sys.exit(self_test(side_path, fpath, B=min(a.bootstrap, 2000)))

    feats = load_features(fpath)
    alg, want = side["digest"]
    blob_ok = (hashlib.new(alg, open(fpath, "rb").read()).hexdigest() == want)

    wav = read_wav_blob(a.wav_blob) if a.wav_blob else None

    if a.build_host_ref:
        if a.indices:
            idx = []
            for part in a.indices.split(","):
                if "-" in part:
                    lo, hi = part.split("-")
                    idx += list(range(int(lo), int(hi) + 1))
                else:
                    idx.append(int(part))
        else:
            idx = sorted(feats["feats"])
        print(f"building the host reference for {len(idx)} utterances "
              f"({'from the wav blob' if wav else 'from the audio files'}):")
        hr = (build_host_ref_from_wavs(side, feats, {i: wav["wavs"][i] for i in idx})
              if wav else build_host_ref(side, feats, idx))
        path = a.host_ref if os.path.isabs(a.host_ref) else os.path.join(REPO, a.host_ref)
        json.dump({"generated_by": "firmware/test/score_wav.py --build-host-ref",
                   "sidecar": side_path, "features": fpath,
                   "wav_blob": (wav["path"] if wav else None),
                   "utterances": {str(k): v for k, v in hr.items()}},
                  open(path, "w"), indent=1)
        print(f"wrote {path}")
        nb = [i for i in hr if hr[i]["bytes_vs_feature_blob"]]
        print("byte-identity against corpus_blob.bin: "
              + ("all identical" if not nb else f"{len(nb)} utterances differ: {nb[:8]}"))
        if wav:
            print(f"index mapping: {len(hr)-len(nb)} of {len(hr)} waveforms reproduce the "
                  f"feature-blob tensor AT THE SAME INDEX, so the waveform blob is in the "
                  f"sidecar's order" if len(hr) - len(nb) > len(hr) // 2 else
                  f"!! index mapping: only {len(hr)-len(nb)} of {len(hr)} waveforms match "
                  f"the feature blob at the same index -- the blob may not be in the "
                  f"sidecar's order, and every index-keyed comparison below is suspect")
        if not a.log:
            return

    if not a.log:
        ap.error("--log is required (or --self-test, or --build-host-ref)")

    hostref = {}
    hpath = a.host_ref if os.path.isabs(a.host_ref) else os.path.join(REPO, a.host_ref)
    if os.path.exists(hpath):
        hostref = {int(k): v for k, v in
                   json.load(open(hpath))["utterances"].items()}

    raw = open(a.log, "rb").read().decode("utf-8", "replace")
    pr = parse_log(raw, side["n_out_frames"])
    if not pr["passes"]:
        raise SystemExit(f"{a.log}: no '# w <i> ... ids: ...' records parsed "
                         f"({len(pr['bad'])} malformed)")
    if a.which >= len(pr["passes"]):
        raise SystemExit(f"--pass {a.which} but only {len(pr['passes'])} passes")
    ident, notes = compare_passes(pr["passes"])
    hz, clk_src = resolve_clock(pr, a)

    expect_n = a.expect_n if a.expect_n else (wav["N"] if wav else side["N"])
    recs = pr["passes"][a.which]
    missing = missing_indices(range(expect_n), recs)

    meta = {"log": a.log,
            "records parsed": f"{pr['n_records']} over {len(pr['passes'])} pass(es), "
                              + ("identical" if ident else "NOT IDENTICAL"),
            "scoring pass": a.which,
            "clock source": clk_src,
            "feature blob md5": ("MATCHES the sidecar" if blob_ok
                                 else "!! DOES NOT MATCH THE SIDECAR !!")}
    if wav:
        meta["waveform blob"] = (f"{wav['path']}  N={wav['N']} x {NSAMPLES} samples, "
                                 f"{wav['bytes']} B, md5 {wav['md5']}")
    if missing:
        meta["!! MISSING from capture"] = str(missing)
    if pr["bad"]:
        meta["!! malformed records"] = f"{len(pr['bad'])} -- {pr['bad'][:3]}"
    for j, n in enumerate(notes):
        meta[f"  repeat note {j}"] = n
    for j, n in enumerate(pr["notes"][:5]):
        meta[f"  parse note {j}"] = n
    if len(pr["passes"]) > 1:
        for fld in ("fe", "npu"):
            vs = [p[i][fld] for p in pr["passes"] for i in p]
            meta[f"{fld} cycles, all passes"] = (
                f"n={len(vs)} min={min(vs)} median={int(np.median(vs))} max={max(vs)} "
                f"(per-invocation counter reads)")

    res = score(side, feats, recs, hostref, B=a.bootstrap, seed=a.seed, hz=hz)
    report(res, side, feats, meta)

    if a.row_hashes or res["hash_mismatch"]:
        path = a.row_hashes or (a.log + ".host_row_hashes.json")
        json.dump({"generated_by": "firmware/test/score_wav.py",
                   "note": "FNV-1a per mel row, 80 rows x 800 bytes, mel-major. "
                           "UNTESTED against firmware until a build prints row hashes.",
                   "features": fpath,
                   "utterances": {str(i): {"key": {u["i"]: u["k"] for u in side["utts"]}
                                           .get(i, "?"),
                                           "rows": row_hashes(feats["feats"][i])}
                                  for i in sorted(recs) if i in feats["feats"]}},
                  open(path, "w"), indent=1)
        print(f"\nwrote per-mel-row host hashes to {path}")
    if a.json:
        dump = {k: v for k, v in res.items()}
        json.dump(dump, open(a.json, "w"), indent=1, default=float)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()

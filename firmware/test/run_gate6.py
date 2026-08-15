#!/usr/bin/env python3
"""Gate 6 host oracle test: the C greedy CTC decoder vs model/fe.py:greedy().

Builds firmware/src/citrinet_ctc.c natively (the same .c the MCU compiles),
feeds it the int8 logit tensors that ONNX Runtime produces from
artifacts/onnx/q800_real.onnx on real LibriSpeech dev-clean utterances, and
requires the C text to equal the Python text *character for character*.

Where the int8 logits come from
------------------------------
q800_real.onnx is a QDQ graph whose final pair is

    Transpose -> QuantizeLinear -> DequantizeLinear -> /Transpose_output_0

so the tensor the NPU would emit is literally the QuantizeLinear output.  This
script adds that intermediate to the graph outputs and reads it directly, so
the bytes handed to the C decoder are the graph's own int8, not a re-quantised
approximation.  As a cross-check it also verifies

    round(dequantised / 0.265415638685226)  ==  int8 output

element for element, which is the identity that makes "argmax on int8" and
"argmax on float" the same operation.

Also runs a randomised stress pass over synthetic logit matrices, which is
where tie-breaking and repeat-collapse edge cases actually live: real logits
almost never tie, so 24 utterances alone would not exercise them.

Usage:  python firmware/test/run_gate6.py [N_UTTERANCES]      (default 24)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from onnx import TensorProto, helper

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "model"))
sys.path.insert(0, os.path.join(REPO, "eval"))
import fe                                          # noqa: E402
from sets import cal_keys_all, load_recs          # noqa: E402

SCRATCH = os.environ.get(
    "GATE6_SCRATCH",
    "/tmp/claude-1000/-home-claroche-stm32n6-tts/"
    "f2087db5-f2ba-4413-ba1b-2f3dbcb6780e/scratchpad/gate6",
)
MODEL = os.path.join(REPO, "artifacts", "onnx", "q800_real.onnx")
QNODE = "/Transpose_output_0_QuantizeLinear_Output"
OUT_SCALE = 0.265415638685226          # per-tensor, offset 0 (io_contract.h)

SR, T, FRAMES, CLASSES = 16000, 800, 100, 1025
NW = (T - 1) * 160 + 1                 # 127,841 samples -> exactly 800 mel frames
OFF = 4800                             # 0.3 s lead-in, as eval/run_gate1_8s.py
FITS = NW - OFF
SEED = 20260816                        # fresh draw; not any prior eval set

CFLAGS = ["-std=c99", "-O2", "-Wall", "-Wextra", "-Wpedantic",
          "-Wconversion", "-Wshadow", "-Werror"]


def log(*a):
    print(*a, flush=True)


def build_c():
    os.makedirs(SCRATCH, exist_ok=True)
    exe = os.path.join(SCRATCH, "ctc_host_test")
    cmd = (["gcc"] + CFLAGS + ["-I", os.path.join(REPO, "firmware", "inc"),
           os.path.join(REPO, "firmware", "src", "citrinet_ctc.c"),
           os.path.join(REPO, "firmware", "test", "ctc_host_test.c"),
           "-o", exe])
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    r = subprocess.run([exe, "selftest"], capture_output=True, text=True)
    log(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        raise SystemExit("C selftest failed")
    return exe


def session():
    """q800_real.onnx with the pre-Dequantize int8 tensor exposed as output 1."""
    patched = os.path.join(SCRATCH, "q800_real.i8out.onnx")
    m = onnx.load(MODEL)
    names = {o.name for o in m.graph.output}
    if QNODE not in names:
        m.graph.output.append(
            helper.make_tensor_value_info(QNODE, TensorProto.INT8,
                                          [1, FRAMES, CLASSES]))
    onnx.save(m, patched)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.log_severity_level = 3
    return ort.InferenceSession(patched, so, providers=["CPUExecutionProvider"])


def place(w):
    b = np.zeros(NW, dtype=np.float32)
    L = min(len(w), FITS)
    b[OFF:OFF + L] = w[:L]
    return b


def run_c(exe, jobs, tag):
    """jobs: list of (key, int8 [100,1025] array) -> {key: (status, text, ids)}"""
    man = os.path.join(SCRATCH, f"manifest_{tag}.tsv")
    idf = os.path.join(SCRATCH, f"ids_{tag}.tsv")
    with open(man, "w") as f:
        for key, arr in jobs:
            assert arr.dtype == np.int8 and arr.shape == (FRAMES, CLASSES)
            p = os.path.join(SCRATCH, f"{tag}_{key}.i8")
            arr.tofile(p)
            f.write(f"{key}\t{p}\n")
    r = subprocess.run([exe, "decode", man, idf], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"C decoder failed:\n{r.stderr}")
    out = {}
    for line in r.stdout.split("\n"):
        if not line:
            continue
        key, st, ln, text = line.split("\t", 3)
        assert len(text) == int(ln), (key, len(text), ln)
        out[key] = [int(st), text]
    for line in open(idf):
        if not line.strip():
            continue
        key, ids = line.rstrip("\n").split("\t", 1)
        out[key].append(np.array([int(v) for v in ids.split(" ")], dtype=np.int64))
    return {k: tuple(v) for k, v in out.items()}


def main():
    n_utt = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    os.makedirs(SCRATCH, exist_ok=True)
    exe = build_c()

    # ---------------------------------------------------------- corpus pass
    recs = load_recs()
    calk = cal_keys_all(recs)
    held = [r for r in recs if r["k"] not in calk]
    rng = np.random.default_rng(SEED)
    sel = [held[i] for i in rng.permutation(len(held))[:n_utt]]
    assert not (set(r["k"] for r in sel) & calk), "calibration leak"
    log(f"corpus {len(recs)}  held-out {len(held)}  drawn {len(sel)} (seed {SEED})")

    sess = session()
    jobs, py = [], {}
    roundtrip_ok = True
    t0 = time.time()
    for i, r in enumerate(sel):
        w, sr = sf.read(r["f"])
        assert sr == SR
        feat = fe.norm_pf(fe.nemo_mel(place(w.astype(np.float32))))[:, :T]
        f32, q = sess.run(None, {"audio_signal": feat[None].astype(np.float32)})
        f32, q = f32[0], q[0].astype(np.int8)
        rt = np.clip(np.round(f32.astype(np.float64) / OUT_SCALE), -128, 127)
        if not (rt.astype(np.int8) == q).all():
            roundtrip_ok = False
        # oracle: model/fe.py greedy, on the very same int8 the C sees
        py[r["k"]] = {
            "text_i8": fe.greedy(q.astype(np.int32)),
            "text_f32": fe.greedy(f32),
            "ids": q.astype(np.int32).argmax(-1),
        }
        jobs.append((r["k"], np.ascontiguousarray(q)))
        if i % 8 == 0:
            log(f"  {i:3d}/{len(sel)}  {time.time() - t0:5.1f}s")
    log(f"  int8 == round(dequantised/scale) on every element: {roundtrip_ok}")

    c = run_c(exe, jobs, "utt")

    per, n_text_bad, n_ids_bad, n_f32_bad = [], 0, 0, 0
    for key, arr in jobs:
        st, ctext, cids = c[key]
        o = py[key]
        text_ok = (ctext == o["text_i8"])
        ids_ok = bool((cids == o["ids"]).all())
        f32_ok = (o["text_f32"] == o["text_i8"])
        n_text_bad += (not text_ok)
        n_ids_bad += (not ids_ok)
        n_f32_bad += (not f32_ok)
        per.append({"k": key, "status": st, "chars": len(ctext),
                    "text_match": text_ok, "ids_match": ids_ok,
                    "f32_vs_int8_text_match": f32_ok,
                    "c_text": ctext, "py_text": o["text_i8"]})
        if not text_ok:
            log(f"  DISAGREE {key}\n    C  : {ctext!r}\n    PY : {o['text_i8']!r}")
        if not ids_ok:
            bad = np.nonzero(cids != o["ids"])[0]
            log(f"  IDS DISAGREE {key} at frames {bad.tolist()}: "
                f"C {cids[bad].tolist()} vs PY {o['ids'][bad].tolist()}")
        if st != 0:
            log(f"  NON-ZERO STATUS {key}: {st}")

    log("")
    log("--- corpus utterances ---")
    log(f"  utterances            : {len(per)}")
    log(f"  text disagreements    : {n_text_bad}")
    log(f"  argmax-id disagreements: {n_ids_bad}")
    log(f"  frames compared       : {len(per) * FRAMES}")
    log(f"  chars compared        : {sum(p['chars'] for p in per)}")
    log(f"  fp32-vs-int8 greedy text differences (informational): {n_f32_bad}")
    for p in per[:3]:
        log(f"    {p['k']}: {p['c_text']!r}")

    # ------------------------------------------------------- stress pass
    # Small alphabets and small logit ranges force ties, repeat runs and
    # blank-separated repeats, which real logits essentially never produce.
    srng = np.random.default_rng(4242)
    cases, meta = [], []
    n_case = 0
    for lo, hi, span, blank_bias in [
        (-128, 128, CLASSES, 0),     # full range, full alphabet
        (-2, 3, CLASSES, 0),         # heavy ties across the whole vocabulary
        (-1, 2, 6, 0),               # tiny alphabet -> repeats and ties
        (-1, 2, 6, 1),               # same, blank slightly favoured
        (0, 1, CLASSES, 0),          # every logit identical: argmax must be 0
        (-3, 4, 40, 3),              # blank-dominated, sparse emissions
    ]:
        for _ in range(80):
            a = np.zeros((FRAMES, CLASSES), dtype=np.int32) - 128
            a[:, :span] = srng.integers(lo, hi, size=(FRAMES, span))
            a[:, CLASSES - 1] = srng.integers(lo, hi + blank_bias, size=FRAMES)
            a = np.clip(a, -128, 127).astype(np.int8)
            key = f"s{n_case:04d}"
            cases.append((key, a))
            meta.append(key)
            n_case += 1
    cs = run_c(exe, cases, "stress")
    s_text_bad = s_ids_bad = 0
    n_tie = n_rep = n_blank = 0
    for key, a in cases:
        st, ctext, cids = cs[key]
        w = a.astype(np.int32)
        ref = fe.greedy(w)
        rid = w.argmax(-1)
        # how much of the hard part is actually being exercised
        n_tie += int(((w == w.max(-1, keepdims=True)).sum(-1) > 1).sum())
        n_rep += int((rid[1:] == rid[:-1]).sum())
        n_blank += int((rid == CLASSES - 1).sum())
        if ctext != ref:
            s_text_bad += 1
            log(f"  STRESS DISAGREE {key}\n    C : {ctext!r}\n    PY: {ref!r}")
        if not (cids == rid).all():
            s_ids_bad += 1
            log(f"  STRESS IDS DISAGREE {key}")
    log("")
    log("--- randomised stress ---")
    log(f"  synthetic logit matrices : {len(cases)}")
    log(f"  frames with a tied argmax: {n_tie} / {len(cases) * FRAMES}")
    log(f"  collapsed repeat frames  : {n_rep}")
    log(f"  blank frames             : {n_blank}")
    log(f"  text disagreements       : {s_text_bad}")
    log(f"  argmax-id disagreements  : {s_ids_bad}")

    ok = (n_text_bad == 0 and n_ids_bad == 0
          and s_text_bad == 0 and s_ids_bad == 0)
    log("")
    log("GATE 6 DECODER: " + ("PASS" if ok else "FAIL"))

    res = {
        "model": os.path.relpath(MODEL, REPO),
        "int8_source": f"graph output {QNODE} (pre-DequantizeLinear)",
        "out_scale": OUT_SCALE,
        "int8_equals_round_dequant_over_scale": roundtrip_ok,
        "oracle": "model/fe.py:greedy() + tokenizer/vocab.txt",
        "decoder": "firmware/src/citrinet_ctc.c (built with " + " ".join(CFLAGS) + ")",
        "corpus": "LibriSpeech dev-clean, calibration keys excluded",
        "seed": SEED, "n_utterances": len(per),
        "frames_compared": len(per) * FRAMES,
        "chars_compared": sum(p["chars"] for p in per),
        "text_disagreements": n_text_bad,
        "ids_disagreements": n_ids_bad,
        "fp32_vs_int8_text_differences": n_f32_bad,
        "stress_cases": len(cases),
        "stress_frames": len(cases) * FRAMES,
        "stress_tied_argmax_frames": n_tie,
        "stress_collapsed_repeat_frames": n_rep,
        "stress_blank_frames": n_blank,
        "stress_text_disagreements": s_text_bad,
        "stress_ids_disagreements": s_ids_bad,
        "pass": ok,
        "per_utterance": per,
    }
    dst = os.path.join(REPO, "firmware", "test", "results", "gate6_ctc.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(res, open(dst, "w"), indent=1)
    log("wrote " + dst)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

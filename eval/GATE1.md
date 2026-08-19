# Gate 1 — WER at the shipped 8-second window, fp32 vs int8

**Verdict: PASS.** int8 costs **+0.50 points** of WER against fp32 at the 8 s
window on 373 held-out utterances that fit the window. The pass band is ~1.0
point; the fail threshold is 2.0.

**Calibration/evaluation disjointness: the deferred check is now done.** The
comment in `model/quant_real.py` claiming disjointness was **wrong** — the 4 s
calibration set overlaps six evaluation sets. But **no published int8 number is
contaminated**, for two independent reasons given in §1. No re-quantisation was
needed.

Everything below is host-side: ONNX Runtime 1.25.1, LibriSpeech dev-clean,
`model/fe.py`. **Nothing here touched the board.**

---

## 1. Part A — is the calibration set disjoint from the evaluation sets?

Reproduced in `eval/sets.py`, checked by `eval/check_disjoint.py`, raw output in
`eval/results/disjoint.json`.

```
source /home/claroche/stm32n6-deployment-zoo/.venv/bin/activate
python eval/check_disjoint.py
```

Three calibration sets exist, one per quantised graph, and they are **not** the
same selection rule:

| set | script | product | filter | seed | slice | n |
|---|---|---|---|---:|---|---:|
| `cal_400` | `model/quant_real.py` | `q400_real.onnx` | `d <= 3.5` (591) | 7 | `[300:364]` | 64 |
| **`cal_800`** | **`model/q800.py`** | **`q800_real.onnx`** | **`4.0 <= d <= 7.5` (952)** | **7** | **`[:48]`** | **48** |
| `cal_1200` | `model/q1200.py` | `q1200_real.onnx` | `5.0 <= d <= 11.5` | 7 | `[:48]` | 48 |

Note this first: the gate brief, `model/README.md` and `docs/FEASIBILITY.md` all
point at `quant_real.py`'s selection, but **that script built the 4 s model.**
The shipped 8 s graph was calibrated by `model/q800.py` with a different filter
and a different slice. Both were checked.

### Overlap by utterance key, every calibration × evaluation pair

| eval set | script | n | ∩ cal_400 | ∩ cal_800 | ∩ cal_1200 |
|---|---|---:|---:|---:|---:|
| `run_int8` | `eval/run_int8.py` | 120 | **11** | 0 | 0 |
| `run_8s` | `eval/run_8s.py` | 150 | **6** | 0 | **2** |
| `run_fe` / `run_gain` | `eval/run_fe.py`, `run_gain.py` | 100 | **7** | 0 | 0 |
| `run_snr` / `run_room` | `eval/run_snr.py`; `run_room.py` since deleted, see `eval/README.md` | 120 | **11** | 0 | 0 |
| `run_ab` | `eval/run_ab.py` | 120 | **13** | 0 | 0 |
| `run_occ` part 1 | `eval/run_occ.py` | 38 | **7** | 0 | 0 |
| `run_pad` | `eval/run_pad.py` | 38 | **7** | 0 | 0 |
| `run_occ12` | `eval/run_occ12.py` | 30 | **3** | 0 | 0 |

**The disjointness claim in `quant_real.py` is false.** "perm seed differs" was
not a valid argument: `run_int8.py` draws from the *same* `d <= 3.5` pool with
seed 0, and 11 of its 120 draws land in `cal_400`. Different seeds over the same
591-element pool give overlapping samples at these sizes — the birthday problem,
not a guarantee.

### Why nothing published is contaminated anyway

**(i) `run_int8.py` already removed them at runtime.** Lines 9–13 rebuild the
`cal_400` key set with seed 7 and filter the evaluation selection against it. It
printed `calib/eval overlap: 11` and scored 109 utterances. Confirmed exactly:
the 109 held-out utterances contain **804 reference words**, matching the 804 in
`eval/results/int8.json` to the word. So the published 4 s figures — fp32
5.60 %, int8 6.09 % — were computed on genuinely held-out audio. The defence was
in the evaluation script, not in the calibration script's comment.

**(ii) `cal_800` — the calibration set of the shipped model — overlaps nothing.**
Zero intersection with all eight evaluation sets. The duration filter
(`4.0–7.5 s`) does most of the work: every other eval set except `run_8s` is
drawn from `d <= 3.8`, a disjoint duration band.

The remaining overlaps (`cal_400` and `cal_1200` against `run_8s`, `run_fe`,
`run_gain`, `run_snr`, `run_room`, `run_ab`, `run_occ`, `run_pad`, `run_occ12`)
are **inert**: every one of those scripts evaluates the fp32 `model.onnx`.
Calibration data cannot influence an fp32 graph.

**Conclusion: no re-quantisation. Part B proceeds against the existing
`q800_real.onnx`.** The Part B evaluation set below nonetheless excludes the
union of all three calibration sets (158 unique keys), so it is clean by
construction rather than by argument.

---

## 2. Part B — the measurement

```
python eval/run_gate1_8s.py 600
```

Raw counts and per-utterance records: `eval/results/gate1_8s.json`.

### Method

- **Window.** T = 800, i.e. 127,841 waveform samples → exactly 800 mel frames →
  100 output frames. Speech is placed at a **4,800-sample (0.3 s) lead-in**,
  identical to the placement `model/q800.py` used to calibrate, so 123,041
  samples (**7.690 s**) of speech fit untruncated.
- **Features.** `model/fe.py` — the NeMo-exact frontend, unchanged. Per-feature
  normalisation over the full 800 frames.
- **Decode.** `fe.greedy` — argmax, collapse repeats, drop blank 1024,
  concatenate, `▁` → space.
- **Models.** fp32 `artifacts/onnx/clean_800.onnx`; int8
  `artifacts/onnx/q800_real.onnx`. Both fed the **identical** fp32 feature
  tensor, so this isolates the quantisation and excludes any frontend effect.
- **Evaluation set.** 600 utterances drawn with seed 20260815 from the 2,545
  dev-clean utterances remaining after removing all 158 calibration keys.
  Median 6.32 s, mean 7.28 s, max 32.15 s. 373 of the 600 fit the window.
- **Text normalisation.** Uppercase; delete every character not in `[A-Z' ]`;
  collapse whitespace; split on whitespace. On this corpus that is uppercasing
  plus a no-op guard — dev-clean references are already `A-Z`, apostrophe and
  space, and the vocabulary is `a-z`, apostrophe and `▁`. The guard exists to
  strip the `<unk>`/`<blk>` pieces if greedy ever emits one. WER is Levenshtein
  word distance ÷ reference words, summed over the corpus (not averaged
  per-utterance).

### (a) utterance-fits-window WER — the model's real recognition accuracy

Only the 373 utterances that fit in 8 s. No truncation. **This is what the int8
gate is judged on.**

| | WER | errors / ref words |
|---|---:|---|
| fp32 `clean_800.onnx` | **4.91 %** | 227 / 4,622 |
| int8 `q800_real.onnx` | **5.41 %** | 250 / 4,622 |
| **int8 cost** | **+0.50 points** | +23 errors |

Paired bootstrap over utterances (10,000 resamples, seed 0): **95 % CI
[+0.07, +0.94] points**, P(cost > 1.0 point) = **0.013**. int8 is worse on 37
utterances, better on 20, identical on 316. The penalty is real but small and
sits inside the pass band with room to spare.

### (b) full-reference WER — what a user experiences

All 600 utterances, scored against the complete spoken reference regardless of
length, so truncation past 7.690 s counts as deletions. Same kind of number as
the 4s/6s/8s/12s table in `docs/FEASIBILITY.md` §2(a).

| | WER | errors / ref words | words returned / spoken |
|---|---:|---|---:|
| fp32 `clean_800.onnx` | **24.27 %** | 2,948 / 12,148 | 0.797 |
| int8 `q800_real.onnx` | **24.65 %** | 2,994 / 12,148 | 0.798 |
| **int8 cost** | **+0.38 points** | +46 errors | — |

Paired bootstrap 95 % CI [+0.16, +0.61]. int8 barely moves this number, which is
the point: **at 8 s the dominant error term is truncation, not quantisation.**
19.4 points of the 24.27 % — four fifths of it — is words the window never
heard.

### (c) frame-level argmax agreement

**0.9716** (58,297 / 60,000 frames; 600 utterances × 100 output frames), fp32 vs
int8 on identical inputs. Comparable to the 0.9649 measured at 4 s.

---

## 3. Two things worth knowing before quoting these numbers

**The 0.3 s lead-in costs 1.4 points of full-reference WER.** Same 600
utterances, fp32, lead-in removed:

| lead-in | speech that fits | full-reference WER | coverage | fits-window WER |
|---|---:|---:|---:|---:|
| 4,800 samples (0.3 s) — as calibrated | 7.690 s | 24.27 % | 0.797 | 4.91 % (n=373) |
| 0 | 7.990 s | **22.85 %** | 0.812 | 4.84 % (n=391) |

It is pure truncation: 0.3 s of a fixed 8 s buffer spent on silence is 0.3 s of
speech lost, and it moves the fits-window WER not at all. If the firmware's
push-to-talk capture does not need a lead-in, **it is 1.4 points of free
accuracy.** Worth deciding deliberately at Gate 5 rather than inheriting from
the calibration script.

**The harness reproduces the published 8 s figure exactly.** Re-running
`run_8s.py`'s selection (seed 1, n = 150, no lead-in) through this harness after
retargeting the corpus path gives **19.98 % WER, coverage 0.838** against the
20.0 % / 0.84 in `docs/FEASIBILITY.md` §2(a). So the 24.27 % above is not a
regression — it is a **different, larger, longer draw**: median utterance 6.32 s
here versus 5.55 s there, so more of it is truncated. The 600-utterance number
is the better estimate; the 150-utterance one is not wrong, just noisier and
luckier.

---

## 4. What was changed in the repo

- `eval/results/recs.json` — all 2,703 `f` paths retargeted from the dead
  `/tmp/claude-1000/.../scratchpad/adv/LibriSpeech/...` to
  `corpus/LibriSpeech/dev-clean/...`. All 2,703 files verified present.
- `model/fe.py` — vocabulary path was hardcoded into the scratchpad; now
  resolves `tokenizer/vocab.txt` relative to the repository root.
- `eval/sets.py` (new) — every calibration and evaluation set reconstructed from
  its original filter/seed/slice, in one place, so disjointness is checkable
  mechanically rather than by reading eight scripts.
- `eval/check_disjoint.py` (new) — Part A.
- `eval/run_gate1_8s.py` (new) — Part B.
- `eval/results/disjoint.json`, `eval/results/gate1_8s.json` (new) — raw counts.

`recs.json` record **order** is load-bearing: every set is an RNG permutation of
indices into a filtered slice of that list. Do not sort or regenerate it, or
every set above changes and the disjointness result stops being reproducible.

The other eight `eval/run_*.py` scripts still carry scratchpad paths. They were
not touched, because rerunning them was not in this gate's scope and editing
them without rerunning would destroy the correspondence between each script and
the `results/` file it produced. `eval/sets.py` documents their selections.

---

## 5. Gate verdict

**PASS.** int8 costs +0.50 points at the 8 s window (95 % CI [+0.07, +0.94]),
against a ~1.0-point pass band and a 2.0-point stop. Frame argmax agreement
0.9716. The calibration set of the shipped model is disjoint from every
evaluation set in the repository. No re-quantisation is required and
`artifacts/onnx/q800_real.onnx` is cleared for firmware.

The caveat that outlives this gate is unchanged and is **risk 2 in
`docs/FEASIBILITY.md`**: every number here is **ONNX Runtime QDQ** int8. ST's
Neural-ART int8 is a different implementation, and the zoo has already recorded
it diverging at cosine 0.996 with a systematic bias on another model. This gate
bounds the cost of quantisation *as a concept* at 0.5 points. It does not bound
ST's. Gate 4 is still the first thing that can contradict it.

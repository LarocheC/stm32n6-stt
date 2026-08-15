# Evaluation harness

Product-quality gates: does this work for a person? (Toolchain gates — will an
arbitrary graph run on this part — live in `stm32n6-deployment-zoo`.)

All results in `results/` are **host** measurements: LibriSpeech dev-clean,
ONNX Runtime, degradations applied in simulation. Nothing here touched the board.
Scripts carry hardcoded scratchpad paths and need retargeting before rerunning.

| script | question | result |
|---|---|---|
| `run_8s.py` | WER at a given window against the full spoken reference | window table, `docs/FEASIBILITY.md` §2(a) |
| `run_int8.py` | what does int8 quantisation cost | `results/int8.json` |
| `run_occ.py` / `run_occ12.py` | does padding a short utterance to a long window hurt | `results/occ.log`, `occ12.log` |
| `run_pad.py` | padding placement sensitivity | `results/pad.log` |
| `run_snr.py` | white / pink / babble noise at 30–5 dB | `results/snr.json` |
| `run_room.py` | reverberation, mic HP, clipping, gain | `results/room.json` |
| `run_gain.py` | input level vs the log-zero guard | `results/gain.log` |
| `run_fe.py` | frontend ablation — which spec choices matter | `results/fe.log` |
| `run_ab.py` | per-utterance A/B records | `results/ab.json` |

## Reading the JSON

Values are `[word_errors, total_words]` pairs, so WER is the quotient. For
example `results/int8.json`:

```json
{"f32_clean": [45, 804], "int8_clean": [49, 804],
 "f32_bab15": [77, 804], "int8_bab15": [89, 804]}
```

→ fp32 5.60 %, int8 6.09 % clean; 9.58 % → 11.07 % under 15 dB babble. int8
costs about half a point clean and about 1.5 points under babble.

## The two findings that changed the plan

**Window length dominates everything else.** 4 s → 8 s moves WER against the
full reference from 47.7 % to 20.0 %, because at 4 s the model only ever hears
56 % of the words. Nothing else in the system is worth a quarter as much.

**Input level is a silent killer.** `results/gain.log` — at −54 dBFS, where
ordinary desk speech lands on this microphone, 97.9 % of mel bins fall below the
log guard and WER goes 5.83 % → 35.28 % with no error reported anywhere. The fix
(peak-normalise to 0.9 before the STFT) is free, but only if applied before
int16 truncation: after it, the same fix recovers to 10.45 % rather than 5.83 %.

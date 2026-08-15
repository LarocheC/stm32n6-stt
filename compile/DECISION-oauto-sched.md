# Decision: ship ST's option string **plus** `--Oauto-sched`

Gate 2 established that ST's audio application compiles without `--Oauto-sched`,
and that this costs 300 KB of activations against the zoo's profile. The open
question was whether to match ST exactly or add the flag back.

**Adopted: ST's mpool and ST's option string, with `--Oauto-sched` appended.**

## The compile that settles it

```
stedgeai generate -m artifacts/onnx/q800_real.onnx --target stm32n6 \
  --st-neural-art default@<ST options + --Oauto-sched, compile/st_audio.mpool>
```

| | ST as shipped | **ST + `--Oauto-sched`** | zoo profile (wrong flash base) |
|---|---:|---:|---:|
| epochs | 618 | **628** | 628 |
| pure SW epochs | 0 | **0** | 0 |
| hybrid epochs | 0 | **0** | 0 |
| cpuRAM2 | 500 KB (48.83 %) | **200 KB (19.53 %)** | 200 KB |
| npuRAM6 | 425 KB (94.87 %) | **425 KB (94.87 %)** | 425 KB |
| total activations | 925 KB (62.8 %) | **625 KB (42.5 %)** | 625 KB |
| hyperRAM | 0 B | **0 B** | 0 B |
| octoFlash base | `0x70180000` ✓ | **`0x70180000` ✓** | `0x71000000` ✗ |
| scheduler estimate | 91.89 ms | **91.2 ms** | 91.2 ms |

Evidence: `compile/reports/g800_st_autosched/summary.txt`. The weight blob is
10,200,817 B, byte-identical in size to every previous 8 s compile.

## Why

It is not a trade. `--Oauto-sched` is **smaller and faster at once** — 300 KB
less activation and about 0.7 ms fewer scheduled cycles — with 0 SW and 0 hybrid
epochs preserved and the correct flash base retained. There is no axis on which
ST's shipped configuration wins.

Secondary: the 12-second window stays on-chip with the flag and spills 150 KB to
PSRAM without it, so this keeps a longer window available as a future option
rather than closing it.

## Correcting a wrong reason

An earlier version of this argument claimed the flag relieves pressure on
npuRAM6, which sits at 94.87 % with roughly 23 KB spare. **It does not.**
npuRAM6 holds 425 KB under both option sets; the entire saving is in cpuRAM2.
The tight bank stays tight either way, and any future graph change still has to
fit 448 KB there.

Since cpuRAM2 is claimed by nothing but the model — the application links into
AXISRAM1 (`STM32N657XX_LRUN.ld:47`) and the mel buffers live there too — the
freed 300 KB has no immediate consumer. The practical value is headroom for
graph changes and the 12 s option, not relief of a current constraint.

## The risk taken, and where it is caught

ST presumably validated their shipped configuration and we are diverging from
it. The flag is documented ST Edge AI and the deployment zoo has compiled seven
board-measured models with it, so this is not exotic — but "ST did not ship it"
is a real signal and it is being overridden on evidence rather than dismissed.

The failure mode would be a schedule that reorders epochs incorrectly, which
shows up as wrong output rather than as a compile error. **Gate 4 catches it
directly**: it compares on-device argmax token IDs against host ONNX Runtime on
the same canned input. If Gate 4 diverges, drop `--Oauto-sched` and re-run before
looking anywhere else.

**Regression constant: 628 epochs / 0 SW / 0 hybrid / 625 KB / `0x70180000`.**
Any toolchain bump must reproduce it.

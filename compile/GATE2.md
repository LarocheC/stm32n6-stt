# Gate 2 — recompile against the firmware's memory map

**Verdict: PASS.** `artifacts/onnx/q800_real.onnx` compiled against ST's own audio-application
memory pool with ST's own option string yields **0 software epochs, 0 hybrid epochs,
947,200 B of activations entirely in cpuRAM2 + npuRAM6, zero bytes in hyperRAM**, and a
weights base of **0x70180000** — the address `sign-and-flash-model.sh` writes the blob to.

The deployment contract (input/output shapes, formats, per-tensor scales) is **byte-identical**
to the screening compile, so nothing downstream of the model changes.

Evidence: `compile/reports/g800_st/`. Tool: ST Edge AI Core v4.0.1-20581 (compiler 1.1.3-275).

> **The verdict stands; the geometry moved at Gate 4.** The graph that ships is
> `q800_relu4d_all.onnx` (`q800_real.onnx` with both NPU workarounds applied), it
> compiles to **448 epochs / 0 SW / 0 hybrid / 300 kB cpuRAM2 + 425 kB npuRAM6**,
> and its weight blob is based at **`0x70400000`** — `compile/stt_audio.mpool`,
> which moves octoFlash up so the application has 3 MB rather than the 512 kB ST's
> own mpool leaves. Gate 2's pass criterion is unchanged and is still the check
> that catches a base mismatch; `compile/gen_model.sh` reads the base out of the
> mpool instead of hardcoding it, and `compile/score_build.py` runs the whole
> check set. See `board/GATE4.md` Round 19 and `compile/DECISION-oauto-sched.md`.

---

## 1. ST's actual invocation

Found at `vendor/STM32N6-GettingStarted-Audio/Projects/X-CUBE-AI/models/`:

`generate-n6-model.sh`
```bash
$generateCmd generate -m $1 --target stm32n6 --st-neural-art default@user_neural_art.json
...
cp ./st_ai_output/network_atonbuf.xSPI2.raw network_data.bin
arm-none-eabi-objcopy -I binary network_data.bin --change-addresses 0x70180000 -O ihex network_data.hex
```

`user_neural_art.json` — **ST's exact option string**
```json
{ "Profiles": { "default": {
      "memory_pool": "./stm32n6.mpool",
      "options": "--Ocache-opt -O3 --Os --native-float --Omax-ca-pipe 4 --cache-maintenance --csv-file network --all-buffers-info"
} } }
```

`sign-and-flash-model.sh` — confirms the same offset on the wire:
```bash
$prog -c port=swd mode=HOTPLUG ap=1 --extload $el -w $weight 0x70180000   # weight="network_data.bin"
```

**Confirmed from ST's files, not assumed:** the option string contains neither
`--enable-virtual-mem-pools` nor `--Oauto-sched`. Independently reproduced in the compiler's
own `st_ai_ws/neural_art__network/atonn_options.ini` from this run, which lists exactly
`cache-maintenance, native-float, optimization=3, Os, Omax-ca-pipe=4, Ocache-opt, csv-file,
all-buffers-info` and nothing else.

### Option-set diff: ST vs `compile/audio_profile.json` (the zoo profile used for screening)

| option | ST audio app | zoo `audio` profile |
|---|:--:|:--:|
| `-O3` / `--optimization 3` | yes | yes |
| `--Os` | yes | yes |
| `--Ocache-opt` | yes | yes |
| `--cache-maintenance` | yes | yes |
| `--native-float` | yes | yes |
| `--Omax-ca-pipe 4` | yes | yes |
| `--all-buffers-info` | yes | yes |
| `--csv-file` | `network` | `mem_traffic.csv` |
| **`--Oauto-sched`** | **no** | **yes** |
| **`--enable-virtual-mem-pools`** | **no** | **yes** |
| `--mapping-recap` (+ file) | no | yes |
| `--sched-stats sched_stats.json` | no | yes |
| `--num-ops-info` | no | yes |

ST's set is a strict subset. Because `--sched-stats` is absent, **no `sched_stats.json` is
produced**; the cycle figures in `compile/reports/g800_st/cycles.json` come instead from the
summary rows of ST's own `--csv-file network` output, and its `_source` field records that.

## 2. mpool geometry, side by side

`vendor/…/Projects/X-CUBE-AI/models/stm32n6.mpool` (md5 `3962913c702e781e24ba6bbe431ee10c`,
copied verbatim to `compile/st_audio.mpool`) versus `compile/audio_strict.mpool`:

| pool (fname) | ST audio app — base | ST — size | audio_strict — base | audio_strict — size |
|---|---|---:|---|---:|
| flexMEM (AXIFLEXMEM) | 0x34000000 | 0 KB | 0x34000000 | 0 KB |
| cpuRAM1 (AXISRAM1) | 0x34000000 | 0 KB | **0x34064000** | 0 KB |
| cpuRAM2 (AXISRAM2) | 0x34100000 | 1024 KB | 0x34100000 | 1024 KB |
| npuRAM3 (AXISRAM3) | **absent** | — | 0x34200000 | 0 KB |
| npuRAM4 (AXISRAM4) | **absent** | — | 0x34270000 | 0 KB |
| npuRAM5 (AXISRAM5) | **absent** | — | 0x342e0000 | 0 KB |
| npuRAM6 (AXISRAM6) | 0x34350000 | 448 KB | 0x34350000 | 448 KB |
| **hyperRAM (xSPI1)** | 0x90000000 | **16 MB** | 0x90000000 | **0 MB** |
| **octoFlash (xSPI2)** | **0x70180000** | **63 MB** | **0x71000000** | **112 MB** |

`params.max_onchip_sram_size = 1024 KB` and the cacheinfo block (512 lines × 64 B, 8-way,
`bypass_enable: 1`) are identical in both.

Two differences matter. The **octoFlash base** is the one the gate existed to catch: at
0x71000000 the generated weight blob would have been addressed 14.5 MB above where
`sign-and-flash-model.sh` actually writes it. The **hyperRAM 16 MB** is the one that could have
silently undone the on-chip result — with PSRAM available the allocator is permitted to spill
activations there. It did not, at 8 s. It **does** at 12 s (§5).

The npuRAM3/4/5 rows being absent rather than present-at-zero makes no difference: they carry
zero bytes either way. Whether `Int_Mem_Config()` could enable them is still unread and
still irrelevant at 8 s.

## 3. Result — the gate

Command:
```
stedgeai generate -m artifacts/onnx/q800_real.onnx --target stm32n6 \
  --st-neural-art default@<profile with ST's option string + compile/st_audio.mpool>
```

```
	cpuRAM2    [0x34100000 - 0x34200000]:    500.000 kB /      1.000 MB  ( 48.83 % used)  activations: 500.000 kB
	npuRAM6    [0x34350000 - 0x343C0000]:    425.000 kB /    448.000 kB  ( 94.87 % used)  activations: 425.000 kB
	octoFlash  [0x70180000 - 0x74080000]:      9.728 MB /     63.000 MB  ( 15.44 % used)  weights:     9.728 MB
	hyperRAM   [0x90000000 - 0x91000000]:          0  B /     16.000 MB  (  0.00 % used)  activations:       0  B
Total number of epochs                               618
>> pure software (SW) epochs                           0
>> hybrid epochs (using both software and hardware)    0
```

| PASS criterion | required | measured | |
|---|---|---|:--:|
| pure software epochs | 0 | **0** | pass |
| hybrid epochs | 0 | **0** | pass |
| activations | ≤ 1,507,328 B | **947,200 B** (925.000 kB, 62.8 % of pool) | pass |
| activation placement | cpuRAM2/npuRAM6 only, 0 B hyperRAM | **500 kB cpuRAM2 + 425 kB npuRAM6, 0 B hyperRAM** | pass |
| weights/flash base | 0x70180000 | **0x70180000** (range 0x70180000–0x70B3A700) | pass |

Three independent confirmations that hyperRAM is untouched:
1. report line — `hyperRAM … 0 B / 16.000 MB ( 0.00 % used)`;
2. `st_ai_ws/neural_art__network/atonbuf.xSPI1.raw` is **0 bytes** (only `atonbuf.xSPI2.raw`,
   10,200,817 B, carries content);
3. every `hyperRAM (r/w/cycles)` column of `network.csv` sums to **0.0** across all 621 rows.

**IO contract unchanged.** `compile/reports/g800_st/io_contract.h` is identical to
`compile/reports/g800_real/io_contract.h` over every `STAI_NETWORK_IN*`/`OUT*` define —
S8 `{80,800,1}` 64,000 B scale 0.120522417128086 offset 0 CHANNEL_FIRST in; S8 `{100,1025,1}`
102,500 B scale 0.265415638685226 offset 0 out; both `SCALE_OFFSET_NUM (1)`, i.e. per-tensor.

## 4. What ST's option set costs (isolated, not inferred)

Screening produced 625 kB and 628 epochs; ST's options give 925 kB and 618. The gate only
required a flag diff on failure, but the +307,200 B is real, so it was isolated by three
further compiles on ST's mpool:

| ST opts plus… | epochs | SW | cpuRAM2 | npuRAM6 | total act | hyperRAM | cycles |
|---|---:|---:|---:|---:|---:|---:|---:|
| *(nothing — ST as shipped)* | 618 | 0 | 500.000 kB | 425.000 kB | **925.000 kB** | 0 B | 91,893,224 |
| `--enable-virtual-mem-pools` | 618 | 0 | 500.000 kB | 425.000 kB | 925.000 kB | 0 B | 91,893,224 |
| `--Oauto-sched` | 628 | 0 | 200.000 kB | 425.000 kB | **625.000 kB** | 0 B | 91,212,624 |
| both | 628 | 0 | 200.000 kB | 425.000 kB | 625.000 kB | 0 B | 91,212,624 |

- **`--Oauto-sched` accounts for the entire delta**: −300 kB of cpuRAM2, +10 epochs,
  −680,600 cycles (−0.68 ms). Adding it to ST's profile reproduces the screening numbers
  *exactly* — 625.000 kB / 628 epochs / 91,212,624 cycles.
- **`--enable-virtual-mem-pools` is a no-op** for this graph on this mpool: byte- and
  cycle-identical with and without. It was not the thing to worry about.
- Therefore the mpool geometry change (flash base/size, hyperRAM 16 MB, absent npuRAM3/4/5)
  costs **nothing** in activations, epochs or cycles at 8 s. It only moves the flash base.

Latency estimate under ST's shipped configuration: **91,893,224 cycles → 91.89 ms @ 1 GHz**
(ideal 38,926,712, efficiency 42.4 %), versus 91.21 ms for the screening figure. The
~100–250 ms honest band in `docs/FEASIBILITY.md` is unchanged; this is still a scheduler
estimate, not silicon.

## 5. Finding: the 12 s graph *does* spill to PSRAM under ST's option set

Out of the gate's scope but discovered while checking placement, and it is the exact failure
mode the gate was written to catch. `q1200_real.onnx`, ST's mpool, ST's options:

```
	cpuRAM2    [0x34100000 - 0x34200000]:    900.000 kB /   1.000 MB  ( 87.89 % used)
	npuRAM6    [0x34350000 - 0x343C0000]:    412.500 kB /   448.000 kB ( 92.08 % used)
	hyperRAM   [0x90000000 - 0x91000000]:    150.000 kB /   16.000 MB (  0.92 % used)   <-- SPILL
	                                          range 0x90000000-0x90025800
Total number of epochs 618   >> pure SW 0   >> hybrid 0
```

150 KB of activations land in PSRAM, and **the epoch table still reads 0 SW / 0 hybrid** — a
clean-looking compile that has quietly moved a working buffer onto a 2-byte-wide, high-latency
bus. Measured traffic: 153,600 B read + 153,600 B written, 1,152,000 modelled cycles.
Adding `--Oauto-sched` fixes it: 600.000 kB cpuRAM2 + 417.188 kB npuRAM6, **0 B hyperRAM**,
628 epochs, 0 SW.

Note the scheduler's own totals do *not* flag the spill as a cost — the spilled build reports
123,775,024 cycles against 124,080,680 for the on-chip build. Treat the cycle estimate as
blind to PSRAM pressure and gate on the placement line instead.

Evidence: `compile/reports/g1200_st/summary.txt`.

## 6. Recommendation for firmware integration

Ship ST's `user_neural_art.json` **with `--Oauto-sched` added**:

```
--Ocache-opt -O3 --Os --native-float --Omax-ca-pipe 4 --cache-maintenance \
--csv-file network --all-buffers-info --Oauto-sched
```

It costs nothing, saves 300 KB of cpuRAM2 and 0.68 ms at 8 s, and is the difference between
on-chip and PSRAM-spilled at 12 s. Leave `--enable-virtual-mem-pools` off — it does nothing here.

Either way, make the placement line a build-time regression check, not just the epoch table:
**assert `hyperRAM … 0 B` and `octoFlash [0x70180000`**. Both are single greps over
`network_generate_report.txt`.

## 7. Files

| path | what |
|---|---|
| `compile/st_audio.mpool` | verbatim copy of ST's `stm32n6.mpool` (md5 `3962913c702e781e24ba6bbe431ee10c`) |
| `compile/reports/g800_st/summary.txt` | compilation details + memory + epoch section of `network_generate_report.txt` |
| `compile/reports/g800_st/io_contract.h` | INPUTS/OUTPUTS section of the generated `stai_network.h` |
| `compile/reports/g800_st/cycles.json` | cycle/traffic totals, derived from `network.csv` (no `sched_stats.json` under ST's options) |
| `compile/reports/g800_st/mem_traffic_tail.csv` | header + summary rows of ST's `--csv-file network` output |
| `compile/reports/g800_st/user_neural_art.used.json` | the profile actually passed to the compiler |
| `compile/reports/g1200_st/summary.txt` | the 12 s PSRAM-spill evidence of §5 |

Nothing under `vendor/` was modified. The board was not touched.

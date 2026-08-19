# Blocker 2 — minimal reproducer

**The STM32N6570-DK's Neural-ART NPU hangs forever on any epoch in which an
activation accelerator's output port drives a convolutional accelerator's data
input port through the stream switch.**

```
ATONN_DSTPORT(STRSWITCH, 0, CONVACC, u, 0)  <-  ATONN_SRCPORT(STRSWITCH, 0, ACTIV, v, 0)
```

`LL_ATON_RT_RunEpochBlock()` returns `LL_ATON_RT_WFE` forever at that epoch. No
fault, no interrupt, no illegal access, no timeout — the epoch's `POST_END`
callback never fires and `AiDPUProcess()` never returns.

This document is the smallest self-contained case, its matched control, and the
experiment that sharpens the report. It is written to be handed to ST.
Provenance, discovery and the board runs are [`GATE4.md`](GATE4.md) rounds 15-18c.

---

## 1. Environment

| | |
|---|---|
| part | STM32N6570-DK, `DevID:0x0486 (STM32N6) RevID:0x0000` |
| clocks | SYSCLK 600 MHz, HCLK 400 MHz, NPU 800 MHz, NIC 800 MHz |
| tools | **ST Edge AI Core v4.0.1-20581 7ed50de05**, atonn **compiler 1.1.3-275** |
| runtime | `network rt lib v1.1.3-71120109`, `ll_aton` 1.1.3-275 |
| target | `--target stm32n6`, mdesc `stm32n6.mdesc`, cdesc `cortex-m55.cdesc` (stock) |
| boot | boot from external flash (both DK switches left), power cycle |

Memory pool: [`compile/stt_audio.mpool`](../compile/stt_audio.mpool) — ST's
`stm32n6.mpool` with the `xSPI2` (octoFlash) pool moved from `0x70180000` to
`0x70400000` / 60 MB so the signed application fits below the weights. Every
other pool is ST's: `cpuRAM2 0x34100000` / 1024 kB, `npuRAM6 0x34350000` /
448 kB, `hyperRAM 0x90000000` / 16 MB.

Options are ST's own audio application string, verbatim from
`Projects/X-CUBE-AI/models/user_neural_art.json`, held in
[`compile/st_stock.json`](../compile/st_stock.json):

```
--Ocache-opt -O3 --Os --native-float --Omax-ca-pipe 4 --cache-maintenance --csv-file network --all-buffers-info
```

---

## 2. The reproducer graph

**`artifacts/onnx/repro2/repro_actv_convacc.onnx` — 9 nodes, 9 initializers,
2,453 bytes.** Opset 17, IR 8, int8 QDQ, float32 in and out.

```
x  [1,256,200] f32
 |
 +-- QuantizeLinear   (scale 0.01582539, zp 0)   -> xq  int8
 +-- DequantizeLinear                            -> xd
 +-- Relu                                        -> r
 +-- QuantizeLinear   (scale 0.01582539, zp 0)   -> rq  int8
 +-- DequantizeLinear                            -> rd
 |                       w  [256,1,7] int8  --DequantizeLinear(0.00220463, 0)--> wd
 +-- Conv(rd, wd)    group=256, kernel_shape=[7], pads=[3,3], strides=[1], dilations=[1]
 |                                               -> y
 +-- QuantizeLinear   (scale 0.00724166, zp 0)   -> yq int8
 +-- DequantizeLinear                            -> out [1,256,200] f32
```

There is no bias, matching the original. 358,400 MACs. 1,792 B of weights.

### Where it comes from

It is the failing site of the deployed model, cut out and shrunk. The site is
`/encoder/encoder.13/mconv.5/conv/Conv` of
[`artifacts/onnx/q800_fold.onnx`](../artifacts/onnx/q800_fold.onnx) — `Conv2D_859`
in the compiled graph, the convolution the board stalls on — together with the
`Relu` that feeds it, `/encoder/encoder.13/fc.1/Relu`. It is one of **36
identical sites**: `/encoder/encoder.{13..21}/mconv.{5,10,15,20}/conv/Conv`,
every one `group=256, kernel_shape=[7]`.

[`model/repro_blocker2.py`](../model/repro_blocker2.py) `extract` regenerates the
whole chain from the full model:

| stage | file | nodes | bytes | epochs / SW / hybrid | `ACTIV->CONVACC` | max\|diff\| vs the full graph |
|---|---|---:|---:|---:|---:|---:|
| 0 | `s0_extract.onnx` — `onnx.utils.extract_model` between `/encoder/encoder.13/mconv.1/conv/Conv_output_0` and `…/mconv.5/conv/Conv_output_0_DequantizeLinear_Output` | 9 | 7,358 | 3 / 0 / 0 | **1** | **0** |
| 1 | `s1_shortnames.onnx` — tensor and node names shortened, nothing else | 9 | 3,751 | 3 / 0 / 0 | **1** | **0** |
| 2 | `repro_actv_convacc.onnx` — the 256 per-channel weight scales collapsed to one | 9 | 2,453 | 3 / 0 / 0 | **1** | 0.00724167 |

max\|diff\| is against the full model's own intermediate tensors: the 800-frame
graph is run under onnxruntime with `ORT_DISABLE_ALL`, its tensor at the cut
point is fed to the reproducer, and the reproducer's output is compared with the
full graph's at the same place. Stages 0 and 1 are **bit-exact over 3 random
inputs** (`max|diff| = 0.0`, reference `|max|` 0.29-0.34). Stage 2's
0.00724167 is exactly one output quantisation step — the price of one scale
instead of 256, taken because it halves the file. `s1_shortnames.onnx` is the
bit-exact article and compiles to the same epoch structure.

The compiled epoch structure is identical at every stage: 3 epochs, 0 software,
0 hybrid, one `ACTIV -> CONVACC` epoch with a fan-out of three.

---

## 3. The exact compile command

```bash
cat > neural_art.json <<'JSON'
{ "Globals": {}, "Profiles": { "default": {
    "memory_pool": "/home/claroche/stm32n6-stt/compile/stt_audio.mpool",
    "options": "--Ocache-opt -O3 --Os --native-float --Omax-ca-pipe 4 --cache-maintenance --csv-file network --all-buffers-info"
} } }
JSON

/home/claroche/stedgeai/install/4.0/Utilities/linux/stedgeai generate \
    -m artifacts/onnx/repro2/repro_actv_convacc.onnx \
    --target stm32n6 \
    --st-neural-art default@$PWD/neural_art.json
```

In this repository, equivalently:

```bash
compile/gen_model.sh artifacts/onnx/repro2/repro_actv_convacc.onnx \
    repro2_repro_actv_convacc --profile-file compile/st_stock.json --no-hex
compile/score_build.py artifacts/compile/repro2_repro_actv_convacc
```

Result — `artifacts/compile/repro2_repro_actv_convacc/st_ai_output/network_generate_report.txt`:

```
Total number of epochs                               3
>> pure software (SW) epochs                         0
>> hybrid epochs (using both software and hardware)  0
>> pure hardware (HW or EC) epochs                   3

npuRAM6   [0x34350000 - 0x343C0000]:  350.000 kB / 448.000 kB (78.12 %)  activations
octoFlash [0x70400000 - 0x74000000]:    1.752 kB /  60.000 MB           weights
cpuRAM2, hyperRAM: 0 B
```

Everything on-chip. No PSRAM. No software epochs.

---

## 4. The observed epoch configuration

All citations are into
`artifacts/compile/repro2_repro_actv_convacc/st_ai_output/network.c`.

`epoch_num 1` (`network.c:148`, table entry `network.c:1128`,
`in_streng_mask = 0x00000309`, `out_streng_mask = 0x000000c4`,
`estimated_tot_cycles = 51200`) programs one activation accelerator and three
convolutional accelerators:

```
network.c:154    /* kind=Relu node=Relu_4 */              -> ACTIV_ACC_V2 0   (network.c:153)
network.c:180    /* kind=Conv node=Conv2D_6_subm_2 */     -> CONV_ACC_V2  0
network.c:228    /* kind=Conv node=Conv2D_6_subm_1 */     -> CONV_ACC_V2  1
network.c:276    /* kind=Conv node=Conv2D_6_subm_0 */     -> CONV_ACC_V2  2
```

The compiler splits the `k=7` depthwise convolution into three submasks by
kernel width — `kernelWidth` 3, 2, 2 with `batchDepth = 1` — and drives all
three straight out of the activation accelerator:

```c
/* network.c:562 */ { LL_Switch_Init_Dest() = ATONN_DSTPORT(STRSWITCH, 0, CONVACC, 0, 0),
                      LL_Switch_Init_Source(0) = ATONN_SRCPORT(STRSWITCH, 0, ACTIV, 0, 0), … },
                    /* Conv2D_6_subm_2 IN: in unit=CONV_ACC_V2 0 in port=0 out unit=ACTIV_ACC_V2 0 out port=0 */
/* network.c:565 */ { … CONVACC, 1, 0 … ACTIV, 0, 0 … },   /* Conv2D_6_subm_1 IN */
/* network.c:568 */ { … CONVACC, 2, 0 … ACTIV, 0, 0 … },   /* Conv2D_6_subm_0 IN */
```

(`network.c:604/607/610` repeat them in the re-init block; port 1 of each
convolutional accelerator is the kernel stream and comes from a streaming
engine as usual, `network.c:563/566/569`.)

This is byte-for-byte the configuration that hangs in the deployed model.
`artifacts/compile/g800_noauto/st_ai_output/network.c`, `epoch_num 353`
(`network.c:84583`, table entry `network.c:172393`):

```
network.c:84589   /* kind=Relu node=Relu_857 */                  ACTIV_ACC_V2 1
network.c:84997   CONVACC 0 port 0 <- ACTIV 1 port 0    /* Conv2D_859_subm_2 IN */
network.c:85000   CONVACC 2 port 0 <- ACTIV 1 port 0    /* Conv2D_859_subm_1 IN */
network.c:85003   CONVACC 1 port 0 <- ACTIV 1 port 0    /* Conv2D_859_subm_0 IN */
```

---

## 5. The control that completes

**`artifacts/onnx/repro2/control_streng_convacc.onnx` — 6 nodes, 2,322 bytes.**
The reproducer with the `Relu` and its requantisation pair deleted, nothing else
changed. Same convolution, same shapes, same three-way submask split, same three
convolutional accelerators — the only difference is that the broadcast source is
a streaming engine instead of an activation accelerator.

`artifacts/compile/repro2_control_streng_convacc/st_ai_output/network.c`,
`epoch_num 2` (`network.c:150`, table entry `network.c:1100`,
`in_streng_mask = 0x00000382`, `out_streng_mask = 0x00000058`):

```
network.c:156   /* kind=Conv node=Conv2D_5_subm_2 */    CONV_ACC_V2 0
network.c:204   /* kind=Conv node=Conv2D_5_subm_1 */    CONV_ACC_V2 1
network.c:252   /* kind=Conv node=Conv2D_5_subm_0 */    CONV_ACC_V2 2

network.c:537   CONVACC 0 port 0 <- STRENG 7 port 0     /* Conv2D_5_subm_2 IN */
network.c:540   CONVACC 1 port 0 <- STRENG 7 port 0     /* Conv2D_5_subm_1 IN */
network.c:543   CONVACC 2 port 0 <- STRENG 7 port 0     /* Conv2D_5_subm_0 IN */
```

3 epochs, 0 SW, 0 hybrid, 350 kB npuRAM6 — the same build shape as the
reproducer, and **zero `ACTIV -> CONVACC` links**.

That configuration is proven on silicon. In the deployed model it is
`epoch_num 348` of `artifacts/compile/g800_noauto/st_ai_output/network.c`
(`network.c:82718`, table entry `network.c:172323`) — `Conv2D_850`, also
`group=256 k=[7]`, also split three ways:

```
network.c:83170   CONVACC 0 port 0 <- STRENG 0 port 0   /* Conv2D_850_subm_2 IN */
network.c:83173   CONVACC 3 port 0 <- STRENG 0 port 0   /* Conv2D_850_subm_1 IN */
network.c:83176   CONVACC 1 port 0 <- STRENG 0 port 0   /* Conv2D_850_subm_0 IN */
```

and the board executes it in **178,933 cycles**.

---

## 6. The board symptom

From [`board/traces/round18_noauto_discriminator.log`](traces/round18_noauto_discriminator.log),
`artifacts/compile/g800_noauto` booted from flash. The trace format is
`<NNNfFFFiIIIoOOO>DDDDDDDD`: a **0-based software execution counter** (add 2 for
atonn's `epoch_num`), flags, input and output streaming-engine masks in decimal,
and the epoch's PMU cycle count. A line that opens and never closes is the stall.

```
line 74:  <346f019i681o326>00178933      epoch_num 348   STRENG -> 3x CONVACC   COMPLETES
line 78:  <350f019i016o128>00048024
line 79:  <351f019i834o049               epoch_num 353   ACTIV  -> 3x CONVACC   NEVER CLOSES
```

`i681 = 0x2a9` and `o326 = 0x146` are `epoch_num 348`'s masks; `i834 = 0x342`
and `o049 = 0x031` are `epoch_num 353`'s. Both were predicted from the epoch
block table before the run and both landed exactly where predicted, which is
what pins the numbering.

Counted over the whole 618-epoch graph: **36 epochs carry an `ACTIV -> CONVACC`
link and the board has never completed one; 582 do not and the board completes
them.** The fan-out histogram of those 36 is `{3: 36}` — every one drives
exactly three convolutional accelerators, which is why §7 exists.

Removing every such link makes the whole model run.
[`board/traces/round18_forcemem_pass.log`](traces/round18_forcemem_pass.log):
compiled with `--force-all-in-out-to-mem`, 1064 epochs, 0 SW, 0 hybrid, 0
`ACTIV -> CONVACC`, `AiDPUProcess()` returns in 116,393,913 cycles = 193.989 ms,
repeatable to within 1,957 cycles across runs.

---

## 7. The fan-out experiment

The deployed graph cannot answer whether the defect is *any* `ACTIV -> CONVACC`
link or specifically *an `ACTIV` source driving three destinations*, because all
36 of its instances have a fan-out of exactly three. These graphs separate them.

`fanout1..4.onnx` are one `Relu` feeding **N independent depthwise convolutions**
(`C = 256`, `L = 32`, `K = 3`, `group=256`, so each convolution occupies exactly
one convolutional accelerator and is not split). Everything else is identical
across the four. Each compiles to a **single epoch**, 0 SW, 0 hybrid.

| graph | bytes | nodes | epochs | `ACTIV -> CONVACC` fan-out | in / out mask | est. cycles | npuRAM6 | citation |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `fanout1.onnx` | 1,398 | 9 | 1 | **1** | `0x018` / `0x040` | 8,192 | 16 kB | `repro2_fanout1/…/network.c:385` |
| `fanout2.onnx` | 2,477 | 13 | 1 | **2** | `0x118` / `0x0c0` | 9,216 | 24 kB | `…fanout2/…/network.c:547, 551` |
| `fanout3.onnx` | 3,542 | 17 | 1 | **3** | `0x119` / `0x0c4` | 13,824 | 32 kB | `…fanout3/…/network.c:709, 713, 717` |
| `fanout4.onnx` | 4,607 | 21 | 1 | **4** | `0x11b` / `0x2c4` | 18,432 | 40 kB | `…fanout4/…/network.c:871, 875, 879, 883` |

**The finding, from the compiler:** the `ACTIV -> CONVACC` link is emitted at
fan-out 1, 2, 3 and 4 alike. It is not a property of three-way fan-out, and it
is not a property of the submask split — `fanout1..4` contain no submasks at
all, just N separate `Conv` nodes reading one `Relu`. So a fan-out of exactly
three, which is all the deployed model ever produces, is **incidental**. The
board can now decide between the two remaining readings in four flashes:

| outcome | reading |
|---|---|
| `fanout1` hangs | **any** `ACTIV -> CONVACC` link is broken. The workaround must break all 36 chains, as `--force-all-in-out-to-mem` does |
| `fanout1` and `fanout2` complete, `fanout3` hangs | the defect needs three or more destinations; a rewrite that keeps fan-out ≤ 2 would suffice |
| all four complete | the link alone is not sufficient and something in the deployed model's epochs is co-varying; the primary reproducer of §2 is then the case to test |

`repro_actv_convacc.onnx` (§2, fan-out 3 via a genuine three-way submask split)
and `control_streng_convacc.onnx` (§5, no link) are the matched pair that should
be flashed first: they are the two configurations already decided on silicon at
`epoch_num 353` and `348`, so they validate the reproducer itself before the
ladder is trusted.

### How the fan-out was varied, and what does not vary it

The submask count of a single depthwise convolution is set by the temporal
kernel width and **cannot be made 2**. Sweeping `K` at `C = 256`, `L = 200`
(`artifacts/onnx/repro2/ksweep/`, builds `artifacts/compile/repro2_k*`):

| `K` | conv accelerators | submask `kernelWidth`s | epochs / SW / hybrid | `ACTIV -> CONVACC` fan-out |
|---:|---:|---|---:|---:|
| 1 | 1 | — | 7 / 2 / 1 | none (falls back to software) |
| 2-6 | 1 | one unit, `batchDepth = 4` | 4 / 0 / 0 | **none** — the `Relu` gets its own epoch and writes to memory |
| 7 | 3 | 3, 2, 2 | 3 / 0 / 0 | **3** |
| 8 | 3 | 3, 3, 2 | 3 / 0 / 0 | **3** |
| 9 | 3 | 3, 3, 3 | 3 / 0 / 0 | **3** |
| 10-12 | 4 | ≤3 each | 3 / 0 / 0 | **4** |
| 13-15 | 5 | ≤3 each | 5 / 0 / 0 | **none** — back through memory |

A split unit takes `kernelWidth ≤ 3`; an unsplit one takes up to 6 with
`batchDepth = 4`. Two submasks are therefore unreachable: anything needing ≤ 6
taps fits in one unit, and 7 taps already needs three. Hence the N-consumer
construction above.

Shape sweep (`artifacts/onnx/repro2/shapes/`, builds `artifacts/compile/repro2_sh_*`),
all `group=C` depthwise, ST's stock options:

| C | L | K | consumers | epochs / SW / hybrid | `ACTIV -> CONVACC` |
|---:|---:|---:|---:|---:|---|
| 8 | 200 | 7 | 1 | 3 / 0 / 0 | fan-out 3 |
| 64 | 200 | 7 | 1 | 3 / 0 / 0 | fan-out 3 |
| 512 | 200 | 7 | 1 | 3 / 0 / 0 | fan-out 3 |
| 256 | 64 | 7 | 1 | 3 / 0 / 0 | fan-out 3 |
| 256 | 400 | 7 | 1 | 3 / 0 / 0 | fan-out 3 |
| 1 | 32 | 7 | 1 | 2 / 0 / 0 | fan-out 3 |
| 256 | 32 | 3 | 1 | 1 / 0 / 0 | fan-out 1 |
| 256 | 200 | 3 | 1 | 4 / 0 / 0 | **none** |
| 256 | 200 | 3 | 2 | 4 / 0 / 0 | **none** |

Channel count and time length do not affect *whether* the link is emitted at
`K = 7`; they affect only whether the `Relu` and an *unsplit* convolution are
packed into one epoch, which is what the `L = 32` in the fan-out ladder buys.

---

## 8. Compiler options

The same sweep that was run on the deployed model, run on
`repro_actv_convacc.onnx` (builds `artifacts/compile/repro2_opt_*`). It
reproduces the full-model result exactly: **one option removes the link, and it
is documented "only for debugging".**

| options added to ST's stock string | epochs | SW | hybrid | `ACTIV -> CONVACC` |
|---|---:|---:|---:|---:|
| none (stock) | 3 | 0 | 0 | **1** |
| `--Oauto-sched` | 3 | 0 | 0 | **1** |
| `--Omax-ca-pipe 1` (instead of 4) | 3 | 0 | 0 | **1** |
| `--Oconv-split-cw` | 3 | 0 | 0 | **1** |
| `--Oconv-split-kw` | 3 | 0 | 0 | **1** |
| `--Oconv-split-stripe` | 3 | 0 | 0 | **1** |
| `--Ono-clone-dma` | 3 | 0 | 0 | **1** |
| **`--force-all-in-out-to-mem`** | 5 | 0 | 0 | **0** |

On the deployed 800-frame model the same option costs 618 -> 1064 epochs and
cpuRAM2 500 -> 800 kB, and is the only reason Gate 4 runs at all
([`GATE4.md`](GATE4.md) Round 18c). `--Omax-ca-pipe 2` fails to compile on the
full model. `--onnx-omit-opt split_large_mask_conv` removes every three-submask
epoch on the full model but drops the depthwise convolutions to software (755
epochs, 92 SW, 46 hybrid) and is not a deployment.

---

## 9. Shrinks that were tried and did not work

Reported so they are not retried. All three are `s1_shortnames.onnx` with one
further simplification, regenerable with
`model/repro_blocker2.py negatives`, built as `artifacts/compile/repro2_neg_*`.
Each one makes the ST preprocessor stop recognising the QDQ pattern; the graph
falls back to float on the M55 and the link disappears with it — so the failure
is not reproduced, not fixed.

| shrink | file | epochs | SW | hybrid | `ACTIV -> CONVACC` |
|---|---|---:|---:|---:|---:|
| int8 graph **input** (drop the leading `QuantizeLinear`) | `neg_int8_input.onnx` | 6 | **3** | 0 | 0 |
| int8 graph **output** (drop the trailing `DequantizeLinear`) | `neg_int8_output.onnx` | 6 | **3** | 0 | 0 |
| no requantisation between `Relu` and `Conv` | `neg_no_relu_qdq.onnx` | 6 | **3** | 0 | 0 |

The float32-in / float32-out QDQ envelope is load-bearing. **Nine nodes is the
floor** for this graph.

Bytes can still come down, at a cost. `min_actv_convacc.onnx` is `C = 1,
L = 32, K = 7`: **631 bytes**, 9 nodes, 2 epochs, 224 B of activations, and it
still emits `ACTIV -> CONVACC` with a fan-out of three
(`repro2_min_actv_convacc/…/network.c:562, 565, 568`). It is *not* recommended
as the board case: it also emits an `ARITH.1 <- ARITH` stream-switch link that
no epoch the board has ever executed uses, so a hang would have two candidate
causes instead of one. `repro_actv_convacc.onnx` and the `fanout*` ladder emit
no unproven link other than `CONVACC.0 <- ACTIV` itself.

---

## 10. File index

Regenerate everything: `model/repro_blocker2.py extract`, then
`model/repro_blocker2.py build --C … --L … --K … --consumers … -o …`,
then `compile/gen_model.sh <onnx> <tag> --profile-file compile/st_stock.json --no-hex`,
then `compile/score_build.py artifacts/compile/<tag>`.

| file | bytes | what |
|---|---:|---|
| `artifacts/onnx/repro2/repro_actv_convacc.onnx` | 2,453 | **the reproducer** — `ACTIV -> 3x CONVACC` |
| `artifacts/onnx/repro2/control_streng_convacc.onnx` | 2,322 | **the control** — same convolution, `STRENG -> 3x CONVACC`, completes |
| `artifacts/onnx/repro2/fanout1.onnx` | 1,398 | fan-out 1 |
| `artifacts/onnx/repro2/fanout2.onnx` | 2,477 | fan-out 2 |
| `artifacts/onnx/repro2/fanout3.onnx` | 3,542 | fan-out 3, no submasks |
| `artifacts/onnx/repro2/fanout4.onnx` | 4,607 | fan-out 4 |
| `artifacts/onnx/repro2/min_actv_convacc.onnx` | 631 | smallest that still emits the link; see the caveat in §9 |
| `artifacts/onnx/repro2/s0_extract.onnx` | 7,358 | the raw cut, bit-exact with the full model |
| `artifacts/onnx/repro2/s1_shortnames.onnx` | 3,751 | renamed, still bit-exact |
| `artifacts/onnx/repro2/neg_*.onnx` | — | the three refuted shrinks (§9) |
| `artifacts/onnx/repro2/ksweep/k{1..15}.onnx` | — | the kernel-width sweep (§7) |
| `artifacts/onnx/repro2/shapes/*.onnx` | — | the shape sweep (§7) |
| `model/repro_blocker2.py` | — | generates all of the above |
| `compile/st_stock.json` | — | ST's stock option string as a profile |
| `compile/gen_model.sh` | — | compile driver, keeps the whole workspace per tag |
| `compile/score_build.py` | — | counts epochs, SW, hybrid, `ACTIV -> CONVACC` and unproven links |
| `artifacts/compile/repro2_*/` | — | the preserved compiler workspaces for every build cited here |

---

## 11. Summary for ST

1. `stedgeai generate` 4.0.1 emits, for a `Relu` feeding a depthwise `Conv`,
   an epoch that routes the activation accelerator's output port directly into
   the convolutional accelerator's data input port through the stream switch.
2. On STM32N6570-DK silicon that epoch never completes. There is no fault, no
   interrupt and no timeout; `LL_ATON_RT_RunEpochBlock()` waits forever.
3. The identical epoch with a streaming engine as the source instead of the
   activation accelerator — same convolution, same three destination
   accelerators, same broadcast — completes in 178,933 cycles.
4. `--force-all-in-out-to-mem` avoids it and is the only option that does, at
   +72 % epochs and +60 % cpuRAM2 on a real model.
5. The two questions we cannot answer without ST: **is the stream-switch route
   from `ACTIV_ACC_V2` to `CONV_ACC_V2` port 0 supported on this part at all,
   and if it is, is the compiler missing a synchronisation or a descriptor
   field when it emits it?**

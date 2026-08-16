# Gate 4 — Citrinet on silicon, canned features

**Status: NOT PASSING under gdb. The NPU executes but is pathologically slow.**

> Superseded reading: this document first said the invoke "hangs". It does not.
> The runtime is progressing through epoch blocks the whole time — see
> "The NPU is running" below. The symptom is speed, not a stall.

Goal: feed one host-computed int8 `[80,800]` tensor straight to the NPU and compare
the device's per-frame argmax against host ONNX Runtime, with the microphone, the
MDF decimator, the log-mel front end and the gain stage all removed from the
comparison.

## What works

Under ST's documented dev-mode gdb workflow the application reaches, in order:

```
UART_Config  →  gate4_canned  →  AiDPULoadModel  →  AiDPUProcess
```

and prints:

```
# gate4: canned features, mic bypassed
# in=0x34350000 out=0x34350000 scale=8.297212 off=0
```

Three things that were open are now settled:

- **The relaxed `AiDPUCheckModel()` works.** The int8 rank-3 output is accepted and
  `AiDPULoadModel()` returns `DPU_OK`, where the stock check hung the part.
- **The input scale is right.** `8.297212 = 1 / 0.120522417128086`, matching the
  compiled contract, read at runtime rather than hardcoded.
- **Input and output aliasing at `0x34350000` is expected, not a bug.** The input is
  dead by the time the CTC head writes, so the allocator legitimately reuses the
  buffer. Reading the output after the run is correct.

## What does not

`AiDPUProcess()` — which is a thin wrapper over `stai_network_run(..., STAI_MODE_SYNC)`
— **never returns**. A breakpoint on the statement immediately after it
(`audio_bm.c:190`, the invoke-cycle print) was not reached in **400 seconds**, and
neither `HardFault_Handler` nor `Error_Handler` was hit. The part is not faulting;
it is waiting.

For scale: the scheduler estimate for this graph is 91.2 ms.

## Hypothesis tested and REFUTED: the weights are reachable

The first suspicion was that the NPU could not read the 9,728 KB weight blob at
`0x70400000`, on the theory that external flash is memory-mapped by the FSBL in
boot-from-flash mode and there is no FSBL under gdb.

**Wrong.** Halted inside `gate4_canned()` — after `Ext_Mem_Config()` has run —
the target reads external flash correctly at every probe:

| address | read | expected |
|---|---|---|
| `0x70000000` (FSBL) | `53 54 4d 32` | `STM2` magic |
| `0x70100000` (app) | `53 54 4d 32` | `STM2` magic |
| `0x70400000` (weights) | `00 00 00 00` | matches `network_data.bin` |
| `0x70401000` | `02 03 eb 01` | matches |
| `0x70500000` | `ed 02 ee f9` | matches |
| `0x70900000` | `0b 3c fa 0b` | matches |

External memory is mapped, and the weight blob is flashed correctly and in full.
That whole class of explanation is eliminated, and cheaply — without a single
boot-switch flip.

## Remaining candidates

The strongest is the **runtime's completion signal**. The build defines
`LL_ATON_RT_MODE=LL_ATON_RT_ASYNC`, so `stai_network_run(..., STAI_MODE_SYNC)`
plausibly waits on an NPU interrupt to know the epoch chain finished. Note that
`Projects/GS/Src/stm32n6xx_it.c:381` has **`NPU0_IRQHandler` commented out**,
while NPU1/2/3 and `CACHEAXI_IRQHandler` are present. That is ST's own shipped
arrangement and works for their model, so it is suggestive rather than damning —
but a run that neither faults nor progresses for 400 s is exactly what a missed
completion interrupt looks like.

Worth distinguishing, in order:

1. Read the ATON status registers while halted — does the NPU report *busy*,
   *idle*, or *never started*? That separates "waiting on a signal that never
   comes" from "the chain never launched".
2. Break inside `stai_network_run` and walk down to where it blocks.
3. Compare against ST's own AED model in the same harness: if their model also
   hangs under gdb but runs from flash, the difference is the boot path, not our
   graph. This is the single most informative experiment and needs no flips.

## What this does not cast doubt on

The graph itself compiles to 628 epochs with **0 software and 0 hybrid epochs**
against ST's own memory geometry, verified three times. Gate 1's accuracy result
(int8 costs +0.50 WER points) is a host measurement and is unaffected. What is
unproven is only that this part can *execute* the graph — which is precisely what
Gate 4 exists to establish, and precisely why it is sequenced before any of our
front-end or decoder code exists to be blamed.


---

# Round 2 — two hypotheses refuted, and the invoke is not stuck

## Refuted: epoch count

The suggestion was that the part stalls on graphs with many epochs (ours has 628;
ST's AED model has 44 and runs). Tested directly by truncating `q800_real.onnx`
at `/encoder/encoder.2/mconv.5/conv/...` and compiling the prefix with **identical**
options, mpool, and runtime:

| | full | truncated |
|---|---:|---:|
| ONNX nodes | 1922 | 138 |
| epochs | 628 | **45** |
| SW / hybrid epochs | 0 / 0 | 0 / 0 |
| weights in octoFlash | 9,728 KB | **540 KB** |
| input | 64,000 B | 64,000 B |

Same operator mix, including the squeeze-excite blocks. **The 45-epoch graph
behaves exactly like the 628-epoch one.** Epoch count is not the variable, and
neither is weight volume at an 18× difference.

## Refuted: the async completion interrupt

The build defines `LL_ATON_RT_MODE=LL_ATON_RT_ASYNC`, so a synchronous run
plausibly waited on an NPU interrupt that never fired. Rebuilt with
`LL_ATON_RT_MODE=LL_ATON_RT_POLLING` (`ll_aton_config.h:86`), which removes
interrupts from the path entirely. **Identical behaviour.**

## The NPU is running

With breakpoints inside the generated epoch code, the runtime is demonstrably
progressing:

```
LL_ATON_EC_Inference_Init_network      network.c:142
LL_ATON_End_EpochBlock_2               network.c:264
LL_ATON_End_EpochBlock_10              network.c:2477
  ← __LL_ATON_RT_ExecEndEpochBlock   ll_aton_runtime.c:242
  ← LL_ATON_RT_RunEpochBlock         ll_aton_runtime.c:665
  ← __ll_aton_stai_run_synchonously  ll_aton_stai_internal.c:371
```

Epochs start, execute and complete. Nothing is deadlocked. But **45 epochs do not
finish in ten minutes**, against a scheduler estimate of tens of milliseconds for
the whole 628-epoch graph. That is four to five orders of magnitude, which is a
throughput problem, not a logic one.

## Leading hypothesis: external-flash bandwidth without the FSBL

The NPU streams weights from octoFlash at `0x70400000`. In **boot-from-flash**
mode the FSBL configures xSPI2 for fast octal DTR before handing over. Under
**gdb in development mode there is no FSBL** — only the application's own
`Ext_Mem_Config()`, written for a stock app whose weights ST flashes as part of
the same image.

Reads *work* in this mode — that was verified byte-for-byte at six addresses —
but working and fast are different claims, and only the first was tested. If the
interface falls back to single-line SPI at a low clock, every weight fetch costs
orders of magnitude more, epochs still complete one by one, and the run takes
minutes instead of milliseconds. That matches every observation, including why
weight volume did not change the outcome: 540 KB at a crippled rate is still
far beyond any patience.

**This predicts Gate 4 passes when booted from flash**, which is how the demo
actually runs. The gdb harness would then be unusable for NPU timing — useful for
control flow, useless for throughput — which is worth knowing on its own.

## Next

1. Restore the full Citrinet model (the truncated graph is currently installed),
   flash app and weights, boot from flash, and read the UART. One switch flip.
2. If it passes there, record that dev-mode gdb cannot be used for inference
   timing and move the Gate 4 measurement to flash boot permanently.
3. If it also stalls from flash, the hypothesis is wrong and the next suspect is
   the xSPI configuration itself rather than who performed it.

---

# Round 3 — RAM pressure refuted; the two boot paths disagree

Two more changes, both sound in themselves, neither of which produced output:

- **The inference now loops** every 3 s with a run counter. This was forced by a
  measured fact, not a guess: the catcher logged the USB port lost at 98.5 s and
  reopened at **106.3 s**, a 7.8 s re-enumeration gap under usbipd. A one-shot
  print at power-on is unobservable on this bench no matter how the host is
  arranged, so every earlier "0 bytes from flash" reading was uninterpretable.
- **RAM went 99.04 % → 73.46 %** by dropping `AudioBM_proc_t` from the
  `GATE4_CANNED` path — 269 KB of capture, preprocessing and playback state this
  path never touches. The suspicion was that at 99 % of a 1023 KB region there
  was no headroom for the FSBL's own copy and stack, where Gate 3's *working*
  flash boot sat at 57.62 %.

**Result: still not one byte from flash boot**, with the loop running and 26 % of
RAM free. RAM pressure is refuted as the explanation.

## The state worth recording

The same image behaves differently under the two boot paths:

| | under gdb (dev mode) | booted from flash |
|---|---|---|
| reaches `UART_Config` | yes | **no output at all** |
| prints its header | yes | — |
| loads the model | yes, `DPU_OK` | — |
| executes epoch blocks | yes, verified by backtrace | — |
| completes an inference | no — minutes, not ms | — |

Under gdb it runs and is pathologically slow. From flash it is silent from the
first instruction that would print. Those are two different failures, and the
second is not explained by anything tested so far.

## Recommended next step, and it is not another power cycle

Every hypothesis so far has been tested by changing *our* build and asking for a
boot-switch cycle. That loop has now cost far more than it has returned. The
experiment that actually discriminates is a **known-good control on our own
runtime**:

Regenerate ST's AED model with ST Edge AI Core 4.0.1 (the version our middleware
now requires) and run *their* unmodified application. That separates three things
a single test:

- if ST's app runs from flash on our runtime → our graph or our app changes are at
  fault, and the difference is bisectable on the host;
- if ST's app is also silent from flash → the middleware upgrade or the relocated
  flash layout broke the boot path, independent of our model;
- if ST's app runs *fast* under gdb → the "no FSBL, slow xSPI" theory for the
  gdb slowness is wrong too.

This is host work plus one flip, and it is the first test in this gate whose every
outcome is informative.

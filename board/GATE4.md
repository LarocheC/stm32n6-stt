# Gate 4 — Citrinet on silicon, canned features

**Status: NOT PASSING. The model loads; the NPU invoke does not return.**

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

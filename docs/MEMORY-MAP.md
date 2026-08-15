# The 1.47 MB audio pool is a policy choice, not a hardware ceiling

`docs/FEASIBILITY.md` §5 left open "whether the audio app's `Int_Mem_Config()`
enables npuRAM3/4/5." It does. All of them.

## What the app actually powers on

`vendor/STM32N6-GettingStarted-Audio/Projects/GS/Src/audio_bm.c:741-760`, in the
default (non-`APP_LP`) build:

```c
RCC->MEMENR |= RCC_MEMENR_AXISRAM3EN | RCC_MEMENR_AXISRAM4EN
             | RCC_MEMENR_AXISRAM5EN | RCC_MEMENR_AXISRAM6EN;
RCC->MEMENR |= RCC_MEMENR_CACHEAXIRAMEN;
/* then HAL_RAMCFG_EnableAXISRAM on SRAM2, SRAM3, SRAM4, SRAM5, SRAM6 */
/* then __HAL_RCC_AXISRAM{2,3,4,5,6}_MEM_CLK_ENABLE() */
```

Only the low-power variant (`#ifdef APP_LP`) narrows this to AXISRAM6 alone.

## What is actually claimed by anything

| bank | address range | size | claimed by |
|---|---|---:|---|
| AXISRAM1 | `0x34000400`–`0x34100000` | 1023 K | **the application** — the sole `RAM` region in `STM32N657XX_LRUN.ld:47` |
| AXISRAM2 (`cpuRAM2`) | `0x34100000`–`0x34200000` | 1024 K | the model, via ST's mpool |
| AXISRAM3 (`npuRAM3`) | `0x34200000`–`0x34270000` | 448 K | **nothing** |
| AXISRAM4 (`npuRAM4`) | `0x34270000`–`0x342E0000` | 448 K | **nothing** |
| AXISRAM5 (`npuRAM5`) | `0x342E0000`–`0x34350000` | 448 K | **nothing** |
| AXISRAM6 (`npuRAM6`) | `0x34350000`–`0x343C0000` | 448 K | the model, via ST's mpool |

The application's linker script declares exactly one region and it fits AXISRAM1
exactly (`0x34000400 + 1023K = 0x34100000`). It never touches AXISRAM2–6.

ST's `Projects/X-CUBE-AI/models/stm32n6.mpool` declares `cpuRAM1` (AXISRAM1) at
size 0 — correctly reserving it for the app — and then declares only `cpuRAM2`
and `npuRAM6` to the compiler. **npuRAM3/4/5 are powered, clocked, contiguous,
and offered to nobody: 1,376,256 B sitting idle.**

## What this changes

The binding constraint quoted throughout this project — 1,507,328 B — is ST's
conservative default allocation, not a limit. Declaring npuRAM3/4/5 in the mpool
raises the on-chip pool to the full **2,883,584 B** (`0x34100000`–`0x343C0000`,
2816 K contiguous), with no conflict against the application's linker script.

Consequences, in order of interest:

- **The 12-second window stops being a memory question.** At 1,017 KB of
  activations it currently occupies 69.1 % of the narrow pool; against the full
  pool it is 35.3 %.
- **8 s stops being tight in npuRAM6.** Today it fills that bank to 94.9 %,
  which leaves the allocator almost no room to manoeuvre. Widening the pool lets
  the compiler rebalance, and may change the schedule for the better.
- **It weakens the case for ever spilling to PSRAM**, which was risk 3's
  neighbouring concern.

## Before relying on this

Two things are unverified:

1. **Whether the compiler produces a *better* schedule with the wider pool, or
   merely a legal one.** More banks is not automatically faster — allocation
   across banks affects NPU port contention. Recompile and compare cycles
   rather than assuming.
2. **Whether anything else in the final application wants AXISRAM3/4/5.** The
   LCD framebuffer is destined for PSRAM (768,000 B) and the mel feature buffer
   is 256,000 B, so nothing currently planned needs them — but the audio DMA
   ring buffers and any future additions should be checked against the linker
   script before the banks are handed to the compiler.

This is upside, not a dependency: **8 s already fits the narrow pool at 42.5 %.**
Nothing in the current plan needs the wider one.

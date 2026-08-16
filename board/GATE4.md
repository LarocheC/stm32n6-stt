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

---

# Round 4 — the AED control PASSES, and that narrows it sharply

ST's `yamnet_1024_64x96_tl_qdq_int8.onnx` regenerated with **ST Edge AI Core
4.0.1** (31 epochs, 3,282,785 B of weights — the same size as their shipped blob),
installed into their **unmodified** application, ST's original mpool with weights
back at `0x70180000`, ST's option string, ST's heap restored. Links at
**57.62 % RAM — identical to the Gate 3 build that booted successfully.**

Booted from flash, it runs:

```
| 110     |  2.07%|  0.88|  1.19|  0.00|
```

## What this exonerates

| | verdict |
|---|---|
| the board and the boot chain | **fine** |
| the ll_aton 262 → 275 middleware upgrade | **fine** |
| our vendor patches (`ai_dpu.c`, `cpu_stats.c`, Makefile, linker script) | **fine** |
| our build / sign / flash procedure | **fine** |

Every global change this project made to ST's package is cleared. The fault is
specific to the Citrinet configuration.

## The suspect that now stands out

| | AED control (works) | Citrinet (silent from flash) |
|---|---:|---:|
| signed app size | **243,296 B** | **714,560 B** |
| app slot ST designed for | 512 KB | 512 KB |
| weights base | `0x70180000` | `0x70400000` |
| epochs | 31 | 628 |
| RAM | 57.62 % | 73.46 % |

**Our application is 714 KB against a 512 KB slot.** Moving the weight blob to
`0x70400000` made room in the *address map*, but nothing has ever verified that
the **FSBL will load an SSBL larger than 512 KB**. It reads the image length from
the signed header, but a ceiling — or a layout assumption baked into `ai_fsbl.hex`
— produces precisely the observed signature: correct under gdb, which loads
sections directly and never involves the FSBL, and silent from flash at any epoch
count, any weight volume, and any RAM occupancy.

This also retro-explains round 2: the 45-epoch truncated graph was only ever run
*under gdb*, never from flash, so it never tested this.

## Next test, and it is cheap

Build the **truncated** Citrinet (45 epochs) with `GATE4_CANNED`. Its `network.c`
is a fraction of the full one and the image linked at 47.72 % RAM, so the signed
app should land **under 512 KB**. Flash it with weights at `0x70400000` and boot.

- **prints** → app size is the blocker, and the fix is a bigger app slot (move the
  weights further out and confirm the FSBL follows the header) or a smaller image.
- **silent** → app size is not it either, and the remaining difference is the
  Citrinet graph itself or the relocated weight base.

Either way it is one build and one boot cycle, and it discriminates.

---

# Round 5 — isolated: the Citrinet graph, and it dies before `main`'s first print

Four more single-variable tests, all from flash boot:

| build | app | weights | result |
|---|---:|---|---|
| ST AED model, ST's unmodified app | 243 KB | `0x70180000` | **runs** |
| Citrinet full, our harness | 714 KB | `0x70400000` | silent |
| Citrinet truncated (45 ep), our harness | 178 KB | `0x70400000` | silent |
| Citrinet truncated (45 ep), our harness | 178 KB | **`0x70180000`** | silent |
| **ST AED model, our harness** | **186 KB** | `0x70180000` | **runs** |

```
# ---- run 7 ----
# fed 6144 B input
# invoke returned
# output too small for CTC argmax (control run)
```

## What is now eliminated

App size (714 KB → 178 KB, and AED runs at a *larger* 186 KB), the relocated weight
base, epoch count, RAM occupancy, the middleware upgrade, every vendor patch, the
`GATE4_CANNED` harness itself, and the board. The harness runs ST's graph
perfectly, including a completed NPU inference.

**The variable is the Citrinet network, and nothing else.**

## The important detail

The Citrinet builds emit **not one byte** — not even
`# gate4: canned features, mic bypassed`, which is the first statement of
`gate4_canned()` and runs long before `AiDPULoadModel()`. `UART_Config()` is
`audio_bm.c:122`, earlier still.

So the Citrinet image fails **before `UART_Config`** — in early init, in startup,
or in the FSBL's load of the image. That is not an NPU fault and not a runtime
fault. It is the image failing to come up at all, and it happens with a 178 KB
Citrinet image while a 186 KB AED image on the same addresses boots fine.

Under gdb the same image *does* boot and *does* execute epochs, because gdb loads
sections directly and never involves the FSBL. That is the whole difference.

## Where to look next (no more bisection by power cycle)

The remaining suspects are all static and inspectable on the host:

1. **Section layout.** Diff the two link maps. Citrinet's `network.c` contributes
   ~521 KB of `.rodata` even truncated; check for a section landing somewhere the
   FSBL's own copy or stack occupies, or a `.data` region the startup code copies
   over itself.
2. **The signed image vs what the FSBL expects.** Compare the two headers field by
   field, especially the length, and confirm the FSBL honours it rather than a
   fixed size.
3. **Startup-time initialisers.** Whether `network.c` introduces anything running
   before `main` — constructors, or a `.data` block large enough that the copy
   loop overruns.

All three are host work on artefacts already on disk, and none needs the board.

---

# Round 6 — the boot chain is exonerated by reading its code, and all three suspects are dead

No board time was spent on this round. Everything below comes from artefacts already
on disk: the two ELFs, the two signed images, and a disassembly of `FSBL/ai_fsbl.hex`.

## Suspect 2 — "the signed image vs what the FSBL expects" — REFUTED

`ai_fsbl.hex` disassembles cleanly. Its loader (`0x34180824`) and its handover
(`0x3418099c`) are short and unambiguous:

```c
/* load */
dst = 0x34000000;                       /* NB: image base, header included   */
src = flash_base + 0x100000;            /* = 0x70100000, the app slot        */
n   = *(uint32_t *)(src + 0x6C) + 0x400;/* header length field + header size  */
for (i = 0; i < n; i++)                 /* byte-by-byte, from memory-mapped   */
    ((uint8_t *)dst)[i] = ((uint8_t *)src)[i];

/* handover */
__disable_irq();
SCB->VTOR = 0x34000400;                 /* payload base = image base + 0x400  */
entry     = *(uint32_t *)0x34000404;    /* the APP'S VECTOR TABLE, word 1     */
__set_MSPLIM(0);                        /* cleared, so no stack-limit trap    */
__set_MSP(*(uint32_t *)0x34000400);     /* MSP = the app's _estack            */
__set_PRIMASK(primask);
blx entry;
```

Three things follow, and they close the question:

- **There is no size cap, no checksum check, and no validation of any kind.** The
  FSBL copies whatever length the header claims and jumps. It cannot tell our two
  images apart.
- **The entry point comes from the app's vector table at `0x34000404`, not from the
  header field at `0x70`.** That also finally explains Gate 3's `-align` bug
  properly: without `-align` the payload did not begin at file offset `0x400`, so
  the copy put the vector table somewhere other than `0x34000400` and the FSBL
  jumped through garbage. The header entry field was a *symptom* we could check,
  not the mechanism.
- **`MSPLIM` is explicitly cleared before `MSP` is set**, so the "FSBL leaves a
  stack limit armed and the app faults on its first push" theory is dead too.

Both images were measured against this, and both are correct:

| | AED (runs) | Citrinet truncated (silent) |
|---|---:|---:|
| header version (`0x68`) | `0x00020300` (v2.3.0) | `0x00020300` |
| header length (`0x6C`) | `0x0002D5C0` | `0x0002B660` |
| `.bin` on disk | 185,344 B | 177,312 B |
| length − payload | 448 | 448 |
| header entry (`0x70`) | `0x3400F59D` | `0x3400ECA9` |
| `Reset_Handler` \| 1 | `0x3400F59D` ✓ | `0x3400ECA9` ✓ |

## Suspect 1 — "section layout" — REFUTED

The two ELFs are structurally interchangeable. Nothing lands anywhere unusual, and
the earlier note that `network.c` contributes ~521 KB of `.rodata` even truncated
was simply wrong — it is 102 KB, against the AED build's 104 KB.

| section | AED (runs) | Citrinet truncated (silent) |
|---|---|---|
| `.isr_vector` | `0x34000400` +0x34C | `0x34000400` +0x34C |
| `.text` | `0x34000750` +0x12D90 | `0x34000750` +0x112A4 |
| `.rodata` | `0x340134E0` +0x196E8 | `0x340119F8` +0x18EDC |
| `.data` | `0x3402CBD8` +0xBEC | `0x3402A8E4` +0xF9C |
| `.bss` | `0x3402D800` +0xC18 | `0x3402B8A0` +0x5AC |
| RAM used | 274,456 B (26.20 %) | 264,784 B (25.28 %) |

Both fit inside `0x34000400`–`0x34041000`, a quarter of the region, with the stack
top at `0x34100000` untouched.

## Suspect 3 — "startup-time initialisers" — REFUTED

`.init_array` holds exactly one entry in **both** builds (`frame_dummy`); there are
no constructors from `network.c`. The `.data` copy is RAM-to-RAM in this LRUN
layout (`_sidata == _sdata`), so the copy loop cannot overrun anything.

## And the last model-dependent input before `UART_Config` — also REFUTED

Walking `init_bm()` (`audio_bm.c:104`), exactly one thing that runs before
`UART_Config()` takes an input that differs between the two builds:
`MPU_Config()` builds its non-cacheable MPU region from `__snoncacheable` /
`__enoncacheable`, and that section is **4 bytes in the AED build and 0 bytes in
the Citrinet build**. A degenerate region is legal Armv8-M, and
`MPU_ConfigRegion()` (`stm32n6xx_hal_cortex.c:718`) asserts on the region number,
the permissions and the attribute index — never on base versus limit. It does not
fire.

Everything else on that path — `HAL_Init`, `SystemClock_Config_Full`,
`fuse_vddio`, `Int_Mem_Config`, `Ext_Mem_Config`, `NPU_Config`, `IAC_Config`,
the cache enables — is model-independent code, and `NPU_Config()`
(`misc_toolbox.c:229`) was read line by line to confirm it never touches the graph.

## Where that leaves it

The boot chain is no longer a suspect; it is *understood*. The Citrinet image is
loaded to the right address, at the right length, with `VTOR`, `MSPLIM` and `MSP`
set correctly, and `Reset_Handler` is entered. From there to the first UART byte
every instruction executed is byte-identical to the AED build that works.

That is a contradiction, and it means one of the two premises is false. The premise
that has never actually been verified is the *other* one:

> **that the bytes in flash are the bytes we built.**

For every Citrinet attempt the check performed was that `0x70100000` begins
`53 54 4D 32`. That confirms a header is present. It does not confirm the image is
intact — and the full Citrinet build (714,560 B) provably overflowed the 512 KB app
slot at least once, so this flash has held a too-large image at that address.

## Next test — a readback, not a hypothesis

Flash a Citrinet build, then **read the app region back off the board and
byte-compare it against the signed binary**:

```bash
STM32_Programmer_CLI -c port=SWD mode=UR --extload <...stldr> \
                     -r 0x70100000 <size> readback.bin
cmp readback.bin BuildGCC/BM/GS_Audio_N6_sign.bin
```

This needs the development switch position and no boot cycle. It is a verification
of the one link in the chain that has only ever been spot-checked, and either
outcome is informative: a mismatch explains everything at once, and a match
eliminates the last mechanical explanation and forces the search into the app's own
startup with a debugger on the flash-booted part.

---

# Round 7 — ST's boot documentation, and what it does and does not settle

Source: [`Doc/Boot-Overview.md`](https://github.com/STMicroelectronics/STM32N6-GettingStarted-Audio/blob/main/Doc/Boot-Overview.md)
and `_htmresc/FSBL.png`, plus two files from the same repo that the doc points at.

## It confirms the Round 6 disassembly exactly

ST's own FSBL linker script,
`Drivers/CMSIS/Device/ST/STM32N6xx/Source/Templates/gcc/linker/STM32N657XX_AXISRAM2_fsbl.ld`:

```
ROM (xrw) : ORIGIN = 0x34180400, LENGTH = 255K
RAM (xrw) : ORIGIN = 0x341C0000, LENGTH = 256K
_estack   = ORIGIN(RAM) + LENGTH(RAM)   /* 0x34200000 */
_Min_Stack_Size = 0x800                  /* so _sstack = 0x341FF800 */
```

Every number matches what was read out of the binary in Round 6: image base
`0x34180400`, stack top `0x34200000`, `MSPLIM` `0x341FF800`. **The FSBL owns
`0x34180000` – `0x34200000`: the top 512 KB of AXISRAM2.**

And the boot diagram states the copy explicitly: *"1.5MB of User App binary
(including sign header) is copied from external FLASH to internal SRAM"*, with the
User App region drawn from `0x34000000` up to `0x34180000` — i.e. the app may grow
to 1.5 MB, and the ceiling is where the FSBL starts.

## It also confirms the cpuRAM2 defect, and shows it is ST's, not ours

`Doc/Boot-Overview.md` says:

> 1MB of SRAM1 is reserved for the User App … and **1MB of SRAM2 is reserved for
> the network activations** (see `stm32n6.mpool`).

`Projects/X-CUBE-AI/models/stm32n6.mpool`, shipped, unmodified, ours byte-identical:

```json
{ "fname": "AXISRAM2", "name": "cpuRAM2",
  "offset": { "value": "0x34100000" }, "size": { "value": "1024", "magnitude": "KBYTES" } }
```

**1024 KB starting at `0x34100000` runs to `0x34200000` — straight through the
FSBL.** The documentation and the mpool agree with each other and both are wrong
by 512 KB. It never fires for ST because none of their models allocate in cpuRAM2
at all (the AED model: `cpuRAM2: 0 B, 0.00 % used`). The usable part is
`0x34100000` – `0x34180000`, **512 KB**.

Our 8 s Citrinet at 625 KB of activations is the first thing in this project large
enough to cross that line.

## What it does not explain

The **truncated** graph allocates 200 KB, `0x34100000` – `0x34132000`, nowhere near
the FSBL — and it is silent too. So this is a real constraint on the full model and
a real bug to report upstream, but it is not the Gate 4 fault.

## Two things eliminated on the way

- **The FSBL is not out of date.** `FSBL/ai_fsbl.hex` here is md5
  `652478c952cb1a7a95e7b6c80f862433`, byte-identical to upstream `main`, which is
  v1.4.0 / November 2025 — the newest there is.
- **`BOOT_GetApplicationSize()` is a known-fragile spot but is behaving.** The
  v1.4.0 release note records that ST had to re-append this function by hand
  "*to return correct size for a boot header v2.3*". Round 6 disassembled what
  they wrote — `*(uint32_t *)(hdr + 0x6C) + 0x400` — and it over-copies by exactly
  448 bytes for every image, ours and ST's alike, because the signing tool writes
  `payload + 448` into the length field. Those 448 bytes land in `.bss` and are
  zeroed moments later by `Reset_Handler`. Systematic, and harmless in both builds.

## So the next step is to look, not to deduce

Six rounds of elimination have exhausted what can be inferred. Everything from the
FSBL's jump to the first UART byte is now byte-identical between a build that
works and a build that does not, which means the remaining information is only
obtainable from the part itself, in the state that fails.

**Attach a debugger to the flash-booted board and read the program counter.**
`mode=UR` is the connection that works here (`GATE3.md`), and it resets into the
same flash boot. Halt, then read `PC`, `LR`, `xPSR` and the fault registers
`CFSR` / `HFSR` / `MMFAR` / `BFAR`. That distinguishes, in one shot and with no
further guessing, between:

- `HardFault_Handler` / `Error_Handler` / an `assert_failed()` loop — the address
  says which, and `MMFAR`/`BFAR` say what it touched;
- spinning somewhere in `init_bm()` before `UART_Config`;
- running normally, in which case the fault is the UART or the bench, not the app.

`board/flash_and_verify.sh` covers the other unverified link in the same session:
it reads the app region back off the flash and byte-compares it against the signed
binary, which has never been done for a Citrinet build.

---

# Round 8 — the boot theory was wrong. It boots fine and hangs in the NPU invoke.

**Rounds 3-7 were chasing an artefact.** The Citrinet image does not die before
`UART_Config`. It boots completely, initialises every subsystem, loads the model,
feeds the input, enters `stai_network_run()` — and never comes out.

## What made six rounds read "silent"

The board prints about five lines and then hangs, all within the first two seconds
of boot. A power cycle detaches the USB device and usbipd takes ~8 s to re-attach.
**Every one of those lines was emitted into a window in which nothing was
listening**, and "0 bytes" was then read as "the board is dead".

ST's AED model looked different only because its inference *returns*, so its loop
prints forever and is eventually observable no matter when you start listening.
That single asymmetry — not the graph, not the memory map, not the FSBL —
produced the entire "Citrinet dies before `main`" conclusion.

The fix was to stop reasoning from absence and make the output continuous: hold
each progress marker for ~2 s so the boot becomes a stream, and the last character
received names the last step that completed.

## What the board actually does

```
Ux20 1x20 2x20 3x20 4x20 Rx20 5x20 6x20 7x20 8x20 9x20
System configuration (Bare Metal)
 SYSCLK clock : 600 MHz    HCLK clock : 400 MHz    CACHE conf. : $I/$D=(True,True)
 NPU clock    : 800 MHz    NIC clock  : 800 MHz
Dx20 Ex20
# gate4: canned features, mic bypassed
Lx20
 tools version   : v4.0.1        network rt lib : v1.1.3-71120109
AI_DPU: Activation are already allocated
Mx20
# ---- run 0 ----
# in=0x34350000 out=0x34350000 scale=8.297212 off=0
# fed 64000 B of 64000 B input
# invoking...
<x20          <-- enters AiDPUProcess
              <-- and never returns.  No '>'.
```

`U`…`9` are `UART_Config`, `MPU_Config`, `Int_Mem_Config`, `Ext_Mem_Config`,
`NPU_Config`, `RISAF_Config`, `IAC_Config`, the two cache enables, the sleep
clocks and the BSP init. `D` is `displaySystemSetting()` returning, which also
proves the printf path. `M` is `AiDPULoadModel()` returning `DPU_OK`.

Everything works. The input scale is right (`8.297212 = 1/0.120522`), all 64,000 B
are fed, and the invoke is entered. **The fault is inside the NPU run, exactly
where Round 2 put it under gdb, and nowhere near the boot.**

## Refuted this round, on the board, with a validated instrument

| hypothesis | test | result |
|---|---|---|
| flash contents corrupt | read back FSBL (62,752 B), app (178,336 B) and weights and byte-compare | **all identical** |
| the app dies before `UART_Config` | raw `USART1->TDR` beacon, no printf, no buffering | **refuted — reaches `UART_Config` and far beyond** |
| RISAF firewall blocks the NPU from AXISRAM2 | called `RISAF_Config()`, which ST define and never call (`--gc-sections` drops it entirely) | **no change — still hangs at `<`** |
| an unexpected interrupt is trapped | replaced all **195** `while (1) {}` handlers in `stm32n6xx_it.c` with a reporter that streams `IPSR` forever | **no trap fires — not NPU1/2/3, not IAC, not any fault** |

That last one is worth stating plainly: almost every handler in ST's
`stm32n6xx_it.c` is a deliberate `while (1) {};` trap, so *any* unexpected
interrupt would hang the CPU inside an ISR with no output — a perfect match for
the symptom. It is not what is happening. Nothing fires.

Note also that `NPU0_IRQHandler` being commented out is **correct, not a bug**:
the ll_aton runtime supplies its own. NPU1/2/3 are the "should never happen"
traps, and they stay quiet.

## Where it is

`AiDPUProcess()` → `stai_network_run(..., STAI_MODE_SYNC)` →
`__ll_aton_stai_run_synchonously()` (`ll_aton_stai_internal.c:359`):

```c
do {
  ll_aton_rt_ret = LL_ATON_RT_RunEpochBlock(nn_instance);
  if (ll_aton_rt_ret == LL_ATON_RT_WFE) LL_ATON_OSAL_WFE();
} while (ll_aton_rt_ret != LL_ATON_RT_DONE);
```

No interrupt is trapped and no fault is raised, so the M55 is going round this
loop while an epoch it started never reports completion. That is the same
condition Round 2 saw under gdb, where debug events kept nudging it forward and
made it look merely "pathologically slow" rather than stopped.

## Next: which epoch

The runtime exposes exactly the hook needed —
`LL_ATON_RT_SetNetworkCallback(nn_instance, cb)` with
`LL_ATON_RT_Callbacktype_PRE_START` / `POST_END` per epoch block
(`ll_aton_rt_user_api.h:150`), and this build already defines `LL_ATON_EB_DBG_INFO`,
so each `LL_ATON_RT_EpochBlockItem_t` carries `epoch_num`, the streaming-engine
masks and the estimated cycle counts.

Emitting one raw character per epoch start and end turns the hang into a count:
whichever epoch starts and never ends is the one to look at, and its
`in_streng_mask` / `out_streng_mask` say which NPU streaming engines it was
waiting on. The instance pointer is reachable from the stai handle —
`_stai_aton_context` has `network_instance` as a member and `stai_network *` is
that context.

That is one build and one boot cycle, and it converts "the graph hangs" into
"epoch N, on streaming engine M" — which is both actionable here and reportable
upstream.

---

# Round 9 — the hang is epoch 32 of 45, and it is waiting on a streaming engine

Per-epoch tracing now works on the flash-booted board. Epochs **1 to 31 start and
complete normally. Epoch 32 starts and never ends.**

```
<029f019i064o008>
<030f019i064o008>
<031f019i512o004>      <- ends
<032f019i144o001       <- starts, and that is the last byte the board ever sends
```

Format is `<`epoch `f`flags `i`in_streng_mask `o`out_streng_mask, decimal, with `>`
on completion.

## Epoch 32

| | |
|---|---|
| flags | `19` = `0x13` = `epoch_start \| epoch_end \| pure_hw` |
| input streaming engines | `144` = `0x90` — engines 4 and 7 |
| output streaming engine | `1` = `0x01` — engine 0 |
| operator | `Conv2D_70` |
| reads | `Reshape_69_out_0_inserted_out434`, npuRAM6 +204,800, 204,800 B, live 31..32 |
| writes | `Conv2D_70_off_bias_out_82`, npuRAM6 +0, 102,400 B, `[1,256,400,1]` int8, live 32..33 |

The epoch's completion condition is that its output streaming engines signal. Engine
0 never does, so `LL_ATON_RT_RunEpochBlock()` keeps returning `LL_ATON_RT_WFE` and
the sync loop never leaves.

## What this is not

- **Not an allocation overrun.** Highest buffer end in npuRAM6 is 422,400 B of a
  458,752 B pool; in cpuRAM2, 204,800 B of 1,048,576 B. Both fit.
- **Not AXISRAM2.** Only two buffers ever live in cpuRAM2 —
  `Conv2D_30_off_bias_out_28` (epochs 11..12) and `Reshape_33_out_0` (epochs
  12..38) — and epoch 32 reads and writes npuRAM6 only. Epochs 12 through 31 span
  the same cpuRAM2 residency and complete fine.
- **Not streaming engine 0 being unusable.** Epochs 7 and 8 both drive
  `out_streng = 0x01` and complete.
- **Not a trapped interrupt or a fault** (Round 8), **not the firewall** (Round 8),
  **not the flash contents** (Round 8), **not the boot chain** (Rounds 6-7).

## A middleware defect found on the way

The first attempt at this traced nothing, and the reason is a real bug in the
shipped middleware rather than in the instrumentation.
`__ll_aton_stai_run()` (`ll_aton_stai_internal.c:417-425`) rewrites the epoch
callback on **every** run:

```c
if (nn_context->callback != NULL)
  LL_ATON_RT_SetNetworkCallback(inst, _stai_aton_internal_epoch_block_callback);
else
  LL_ATON_RT_SetNetworkCallback(inst, NULL);   /* <- silently discards the user's */
```

and `_stai_aton_context.callback` **is never assigned anywhere in the middleware**.
There is no public per-network setter — `stai_runtime_set_callback()` feeds a
different, runtime-wide hook. So `LL_ATON_RT_SetNetworkCallback()`, which
`ll_aton_rt_user_api.h` documents as the way to trace epochs, cannot work through
the stai layer at all: whatever you register is set to `NULL` before the first
epoch runs.

Working around it means writing `_stai_aton_context.callback` directly, which is
what this build does. Worth reporting upstream alongside the cpuRAM2/FSBL overlap
from Round 7.

## Where to take it

The question is now narrow and mechanical: why does output streaming engine 0 not
raise completion for `Conv2D_70`, when the same engine completes for epochs 7 and 8
and the same operator class completes 31 times before it. Options, cheapest first:

1. **Read the ATON streaming-engine and CA registers while hung.** The beacon can
   dump `ATON_STRENG*_CTRL/STATUS` and the convolutional accelerator status from
   inside the WFE loop — no debugger needed, and it says whether the engine is
   busy, idle, or errored.
2. **Recompile without `--Oauto-sched`.** ST's plain option set schedules
   differently; if the hang moves or disappears, it is a scheduling artefact
   rather than a property of the operator.
3. **Bisect the graph around node 70.** The truncation point is already a tool we
   have: cutting just before `Conv2D_70` should run clean, and just after should
   hang, which isolates the operator exactly.

---

# Round 10 — it is the stride-2 convolution, and it follows the operator across schedules

## The controlled comparison

Recompiling with `--Omax-ca-pipe 1` produces a **different schedule** — 44 epochs
instead of 45, still 0 software epochs, same memory, same weights size — and moves
`Conv2D_70` from epoch 32 to epoch 31. The hang moves with it:

| | `--Omax-ca-pipe 4` (45 epochs) | `--Omax-ca-pipe 1` (44 epochs) |
|---|---|---|
| `Conv2D_70` scheduled at | epoch 32 | epoch 31 |
| **hangs at** | **epoch 32** | **epoch 31** |
| in / out streaming engines | `0x90` / `0x01` | `0x84` / `0x10` |
| epochs completed before it | 31 | 30 |

Two schedules, two different epoch numbers, two different streaming-engine
assignments, one operator. **The fault is a property of `Conv2D_70`, not of the
schedule, the epoch index, or any particular streaming engine.**

## What is special about Conv2D_70

Nothing, except its stride. The graph contains a run of structurally identical
depthwise convolutions, all with inflated weights of shape `[256,4,3,1]`, 3,072 B:

| conv | epoch | in | out | |
|---|---:|---:|---:|---|
| `Conv2D_34` | 12 | 204,800 | 204,800 | stride 1 — completes |
| `Conv2D_43` | 17 | 204,800 | 204,800 | stride 1 — completes |
| `Conv2D_52` | 22 | 204,800 | 204,800 | stride 1 — completes |
| `Conv2D_61` | 27 | 204,800 | 204,800 | stride 1 — completes |
| **`Conv2D_70`** | **32** | **204,800** | **102,400** | **stride 2 — HANGS** |
| `Conv2D_98` | 40 | 102,400 | 102,400 | (never reached) |

`Conv2D_70` is the **first convolution in the graph that halves the time axis**,
`[1,256,800,1]` → `[1,256,400,1]`, 1,280,000 MACs, weights from octoFlash +524,544.
Four instances of the same operator with the same weight geometry complete
immediately before it. The only difference is the stride.

## Compiler options do not avoid it

All four option sets produce the same 45-epoch, 0-software-epoch graph and all
contain the stride-2 epoch:

| variant | epochs | SW | cpuRAM2 | flash |
|---|---:|---:|---:|---:|
| ST's options + `--Oauto-sched` (shipping choice) | 45 | 0 | 200 kB | 540 kB |
| without `--Oauto-sched` | 45 | 0 | 500 kB | 540 kB |
| `--Omax-ca-pipe 1` | 44 | 0 | 200 kB | 540 kB |
| without `--Ocache-opt` | 45 | 0 | 200 kB | 540 kB |

> **Correction to `compile/DECISION-oauto-sched.md`.** An earlier run of this sweep
> reported that dropping `--Oauto-sched` collapsed the graph to 84 epochs with 57
> software epochs. That was wrong: the already-preprocessed `*_OE_3_3_1.onnx` had
> been fed back into `stedgeai generate`, which preprocessed it a second time. A
> control run against the known-good configuration caught it. `--Oauto-sched` is
> still the right choice, but only because it halves cpuRAM2 occupancy (200 kB vs
> 500 kB) — not because it avoids software epochs.

`--d-dead` (the compiler's own "epoch deadlock analysis and removal" debug level)
reports nothing for this graph at level 9.

## The reproducer

Everything needed to report this upstream is on disk and small:

- `scratchpad/trunc.onnx` (622,844 B) — a 138-node prefix of
  `stt_en_citrinet_256_gamma_0_25`, int8, cut so the failure is reached in 45 epochs
- ST Edge AI Core 4.0.1, `stm32n6.mpool` unmodified, options as shipped
- STM32N6570-DK, boot from flash, ll_aton `v1.1.3-71120109`
- symptom: `LL_ATON_RT_RunEpochBlock()` returns `LL_ATON_RT_WFE` forever at the
  epoch containing the first stride-2 depthwise convolution; no fault, no
  interrupt, no illegal access

## Workaround to try next

A stride-2 convolution is exactly a stride-1 convolution followed by taking every
other column, provided the padding is unchanged — so the graph can be rewritten to
use only the operator form that is known to work here:

```
Conv(stride=2)  ->  Conv(stride=1) + Slice(starts=0, steps=2, axis=time)
```

It costs 2x the MACs of that one layer (1.28M -> 2.56M, against a graph total in
the hundreds of millions) and is numerically exact in int8 as long as the
requantisation parameters are carried over unchanged. Citrinet-256 downsamples 8x
in total, so the full model has a small number of these to rewrite, not dozens.

---

# Round 11 — PASS. The stride-2 depthwise convolution was the only blocker.

```
<050f019i008o256>
<051f019i513o256>
<052f019i064o016>Ox20# invoke returned
# invoke 1153851768 cycles
# ---- run 7 ----
```

**350 epoch events, 350 completed. `AiDPUProcess()` returns. The loop is on run 7.**
The NPU executes the whole graph, repeatedly, booted from flash.

## What it took

Two changes, and only two:

1. **Rewrite the stride-2 depthwise convolution.** In `trunc.onnx`,
   `/encoder/encoder.1/mconv.20/conv/Conv` (`group=256`, `kernel=[3]`,
   `strides=[2]`) becomes `strides=[1]` followed by
   `Slice(starts=0, ends=INT64_MAX, axes=[2], steps=[2])`. With `pads=[1,1]`,
   `k=3`, `L=800` the stride-2 output index *i* is exactly the stride-1 output
   index *2i* for *i* in 0..399, so the rewrite is a selection, not an
   approximation. Verified against the original with onnxruntime on three random
   inputs: **max |diff| = 0, exactly**.

   Note the *other* stride-2 convolution in the graph — `/encoder/encoder.1/res.0.0/conv/Conv`,
   `group=1`, `kernel=[1]` — was left alone and always worked. So the defect is
   specific to **stride-2 depthwise**, not to stride 2.

2. **`-DUSE_EXT_SRAM`.** The rewrite roughly doubles the activation footprint (the
   stride-1 convolution produces an 800-wide tensor where the stride-2 one produced
   400), and the compiler spilled 800 kB to `hyperRAM` at `0x90000000`.
   `Ext_Mem_Config()` gates `BSP_XSPI_RAM_Init()` behind `USE_EXT_SRAM`, which the
   audio application does not define, so that address was never mapped — every
   write to it raised an imprecise bus error that escalated to HardFault
   (`CFSR = 0x00000400`, `BFSR.IMPRECISERR`, `HFSR.FORCED`). Defining it
   initialises the board's APS256 PSRAM and memory-maps the region.

## Sequence of failures, for the record

| build | result |
|---|---|
| original graph | NPU stalls forever at the stride-2 depthwise conv |
| `--Omax-ca-pipe 1` (different schedule) | stalls at the same conv, different epoch index and engines |
| stride-2 rewritten, PSRAM **not** initialised | clears the conv, then HardFault on the unmapped spill |
| stride-2 rewritten, PSRAM initialised | **runs to completion, repeatedly** |

## What this is not yet

`# invoke 1153851768 cycles` is not a usable latency figure — `HAS_DWT_CTRL` is 0
on this part and ST use the PMU instead, so the DWT read in `gate4_canned()` is
meaningless. But the configuration is slow by construction and must not ship as
is:

- **800 kB of activations are in PSRAM**, off-chip over xSPI1.
- **The `Slice` costs 4 software epochs** on the M55, where the graph previously
  had zero.

Both are consequences of expressing the decimation as a separate node. The next
step is to fold it into the following pointwise convolution instead — make the
depthwise `stride=1` and give the **pointwise** `1x1` convolution `stride=2`. That
is the same selection (everything between them is elementwise, so decimation
commutes), it removes the `Slice` node entirely, and it uses the operator form
already proven to work here: `res.0.0` is a `group=1, kernel=1, stride=2`
convolution that executes correctly in this very graph.

Gate 4's question — *can this part execute this graph* — is answered: **yes.**

---

# Round 12 — the shippable fix: fold the decimation into the pointwise convolution

The `Slice` workaround of Round 11 works but costs 4 software epochs and 800 kB of
PSRAM. Neither is necessary. In the Citrinet block the depthwise convolution is
followed by a pointwise one with only quantise/dequantise in between:

```
Conv  group=256 k=3 stride=2      <- hangs the NPU
QuantizeLinear / DequantizeLinear <- elementwise, commutes with decimation
Conv  group=1   k=1 stride=1
```

Moving the stride one operator downstream gives the identical selection:

```
Conv  group=256 k=3 stride=1
QuantizeLinear / DequantizeLinear
Conv  group=1   k=1 stride=2      <- the form res.0.0 already proves works here
```

`max |diff| = 0` against the original over four random inputs, and it adds no node.

## It is free

| | original (hangs) | Round 11 `Slice` | **fold** |
|---|---:|---:|---:|
| epochs | 45 | 50 | **45** |
| software epochs | 0 | 4 | **0** |
| cpuRAM2 | 200 kB | 900 kB | **200 kB** |
| hyperRAM / PSRAM | 0 | 800 kB | **0** |
| octoFlash weights | 540.157 kB | 548.704 kB | **540.126 kB** |

The resource profile is that of the graph that hung. The only difference is which
operator performs the decimation.

## On the board

```
<046f019i002o064>Ox20# invoke returned
# ---- run 7 ----
```

**315 epoch events, 315 completed, 7 inferences returned, no fault**, booted from
flash with PSRAM disabled and `BSP_XSPI_RAM_Init` not even linked.

## Still outstanding

`# invoke 1021881927 cycles` remains meaningless: `ai_device_adaptor.h` sets
`HAS_DWT_CTRL 0` for this part and maps `port_dwt_get_cycles()` to
`ARM_PMU_Get_CCNTR()`, but `gate4_canned()` reads `DWT->CYCCNT` directly. The
harness must use the PMU before any latency number is quoted.

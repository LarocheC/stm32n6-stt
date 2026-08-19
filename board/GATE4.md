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

---

# Round 13 — the full model executes correctly, and the open problem is throughput

## Correctness: done

The stride-2 fold applies to the full 800-frame graph. `q800_real_OE_3_3_1.onnx`
contains **three** stride-2 depthwise convolutions — `Conv2D_70` (k=3),
`Conv2D_478` (k=3), `Conv2D_954` (k=7) — each followed by
`QuantizeLinear -> DequantizeLinear -> Conv(group=1, k=1, stride=1)`. Folding all
three is **bit-exact**: 0 of 615,000 output elements differ over three random
inputs on the `[1,100,1025]` logits.

Compiled through `atonn` with the deployment invocation:

| | original full | **folded full** |
|---|---:|---:|
| epoch blocks | 628 | 629 |
| software epochs | 0 | **0** |
| cpuRAM2 | 625 kB (overlaps the FSBL) | **200 kB** (clear of it) |
| npuRAM6 | — | 425 kB |
| PSRAM | — | **none** |
| weights (octoFlash) | 9,728 kB | 9,960 kB |

The fold *reduced* cpuRAM2 to 200 kB, so the Round 7 FSBL overlap no longer
applies. On the board the graph runs with **no hang and no fault** — 352 epochs
completed and still going when the trace window closed at epoch 354 of 628.

**Gate 4's question, "can this part execute this graph", is answered: yes.**

## Throughput: roughly 2000x off, and not yet explained

The compiler's own estimate for this graph, from `c_info.json`:

```
compute_cycles = 40,317,554        ->  50.4 ms at 800 MHz
max_cycles     = 91,770,192        -> 114.7 ms
memory         = 200,731,988 reads / 45,607,528 writes
                 60,045,789 cycles ->  75.1 ms
```

Measured: **one inference does not complete in 200 s**, in either wait mode. That
is about 0.35 s per epoch, or ~280 million NPU cycles for work estimated in
microseconds. The truncated 45-epoch graph shows the same per-epoch figure, so it
scales with epoch count, not with graph size.

### Eliminated

| candidate | test | verdict |
|---|---|---|
| async completion event never fires | rebuilt with `LL_ATON_RT_MODE=LL_ATON_RT_POLLING`, which removes interrupts and `__WFE()` from the path entirely | **no change** — confirms Round 2 |
| NPU cache disabled | `app_config.h:76` defines `USE_NPU_CACHE 1` and `misc_toolbox.c:22` includes it; preprocessing the translation unit yields `npu_cache_enable()` and no `npu_cache_disable()` | **enabled** |
| software fallback | 0 software epochs, 0 hybrid | **not it** |
| the epoch trace itself | ~20 characters per epoch at 14400 baud with `TC` waits, about 14 ms/epoch; disabling it changed nothing measurable | **not the cause** |

### Not yet tested, in order of promise

1. **xSPI2 prefetch is deliberately disabled.** `Ext_Mem_Config()` ends with
   `MODIFY_REG(XSPI2->CR, XSPI_CR_NOPREF, HAL_XSPI_AUTOMATIC_PREFETCH_DISABLE)`,
   commented "Hotfix for xspi: no prefetch". Every weight fetch from external
   flash then pays full command latency with no lookahead. Removing that line is a
   one-line test.
   Counter-evidence to weigh: ST's AED model streams 106 kB of weights per epoch
   and completes quickly, where this graph streams only 16 kB per epoch. So the
   cost may be per-*transaction* rather than per-byte, which is exactly what
   losing prefetch would produce on a graph with 20x more epochs.
2. **Per-epoch cache maintenance.** `--cache-maintenance` is on. If the runtime
   cleans or invalidates D-cache over each epoch's buffers, the cost is per-epoch
   and this graph has 628 of them against the AED model's 31.
3. **Measure it properly rather than infer.** The PMU counter is now wired up
   (`port_dwt_get_cycles()`); running a graph small enough to finish and printing
   per-epoch cycles from the epoch callback would separate "each epoch is slow" from
   "some epochs are very slow".

## Repository state

`LL_ATON_RT_MODE` is restored to `LL_ATON_RT_ASYNC` in the Makefile. The epoch
trace is now opt-in behind `GATE4_EPOCH_TRACE` — it costs ~14 ms per epoch and
must be off for any measurement. `board/flash_and_verify.sh` takes the weights base
as its second argument and sizes the app slot from it, so the 722 kB Citrinet image
passes the overflow check against a 3 MB slot instead of failing against ST's
512 kB one.

---

# Round 14 — the NPU is fast. There is a second blocker, around epoch 353.

## Correction to Round 13

Round 13 said throughput was "~2000x off, and not yet explained". **That was wrong,
and the error was mine.** It divided wall-clock by epoch count. The wall clock was
dominated by this build's own instrumentation: `BEACON(c)` holds each marker for
20 x `HAL_Delay(100)` = 2 s, and `init_bm()` emits about sixteen of them, so
**roughly 32 seconds elapse before the invoke is even entered.**

Measuring the epochs directly with the PMU says the opposite:

| epoch | measured cycles | compiler estimate | ratio |
|---:|---:|---:|---:|
| 0-6 | 21,891 - 263,218 | 49,152 - 256,000 | **1x** |
| **7** | **5,077,147** | 204,800 | **25x** |
| 8-23 | 157,003 - 670,867 | 204,800 - 819,200 | **1x** |

and the cumulative total across the graph:

```
epoch     0    32    64    96   128   160   192   224   256   288   320
cum ms    0    20    35    46    57    68    79    86    92    98   106
```

**106 ms of NPU time by epoch 320**, growing 6-11 ms per 32 epochs — extrapolating
to roughly 200 ms for all 628, against the compiler's 114.7 ms estimate from
`c_info.json`. The NPU is performing as designed. Epoch 7 at 25x is the only
outlier and costs 8.5 ms.

So the eliminations in Round 13 (async wait mode, NPU cache, software fallback,
xSPI2 prefetch) were all answering a question that was not being asked. The
prefetch test in particular was run against a symptom that did not exist.

## What is actually wrong

The stream ends at `E320t000106` and then goes silent for the remaining ~200 s of
the capture. The earlier full-trace run reached epoch 354. **The full graph stalls
at approximately epoch 353**, in the same manner as the original stride-2 stall:
an epoch starts and never ends, with no fault and no trapped interrupt.

The 45-epoch truncated graph could never have shown this — it stops at epoch 45.
Folding the three stride-2 depthwise convolutions removed the *first* blocker; this
is a second, further in.

Epochs 350-361 are a repeating block:

```
ep 350..377  Reshape_849_out_0            [1,256,200]      51,200 B
ep 351..352  Conv2D_850_off_bias_out_970  [1,256,200,1]    51,200 B
ep 352..353  Conv2D_853_off_bias_out_976  [1,256,200,1]    51,200 B
ep 353..354  Reshape_856_out_0            [1,256,200]      51,200 B
ep 354..356  ____4812 / ____4808 / ____4811  three temporaries, 102,400 B each
ep 356..357  Conv2D_859_off_bias_out_982  [1,256,200,1]    51,200 B
```

The identical block at epochs 344-347 completes, so the shape is not by itself the
problem.

## Next step, and it is the same method that worked before

Round 9 localised the first blocker by tracing every epoch. Do the same in a
window: print per-epoch identity and cycles only for epochs >= 340, so the UART
cost does not distort the measurement and the trace does not stop before reaching
the interesting region. Then read the stalling epoch's operator, flags and
streaming-engine masks out of `c_info.json` exactly as in Rounds 9-10, and check
whether it is another operator form the accelerator mishandles.

Two things worth carrying into that:

- **`BEACON()` must be shortened or disabled** for any timed run. Two seconds per
  marker is right for catching a boot that dies early and completely wrong for
  anything else.
- **Epoch 7 at 25x** the estimate is unexplained and worth a look once the stall is
  resolved.

---

# Round 15 — epoch 352 hangs, and something turns pathological around epoch 321

Windowed tracing (silent below epoch 335, every epoch above it) on the folded full
model, booted from flash:

```
<351f019i008o032>07550957      ends,  12.58 ms
<352f019i393o098               starts, and that is the last byte the board sends
```

## Two distinct findings

### 1. Epoch 352 never ends

`flags = 0x13` (`epoch_start | epoch_end | pure_hw`), input streaming engines
`0x189` (0,3,7,8), output engines `0x062` (1,5,6). It produces
`Conv2D_853_off_bias_out_976`, `[1,256,200,1]`, 51,200 B in npuRAM6.

`Conv2D_853` is **the most ordinary operator in the graph**:

```
op=Conv  group=1  kernel_shape=[1,1]  strides=[1,1]  pads=[0,0,0,0]  dilations=[1,1]
```

A plain pointwise convolution, hundreds of which complete earlier. Unlike the first
blocker (`Conv2D_70`, stride-2 depthwise) there is nothing unusual about the
operator form, so the stride-2 explanation does not transfer.

### 2. Every epoch from ~321 onward costs a uniform ~12.6 ms

| region | per-epoch cost | vs compiler estimate |
|---|---:|---|
| epochs 0-320 | ~0.33 ms (106 ms cumulative) | ~1x |
| epochs 335-351 | **7,524,635 - 7,676,945 cycles**, i.e. 12.54 - 12.79 ms | ~40x |

The spread across those seventeen epochs is **±1 %**, over different operators,
different tensor sizes and different streaming-engine assignments. A constant that
stable is not computation — it is a fixed wait, most likely a timeout being hit and
retried once per epoch. At epoch 352 the same condition apparently becomes
terminal.

## Ruled out for this region

| candidate | evidence |
|---|---|
| cpuRAM2 / AXISRAM2 access | cpuRAM2 is only live at epochs 11-38, 348-349 and 377-380. Epoch 352 does not touch it, and epochs 11-38 ran at full speed |
| number of streaming engines | epoch 346 waits on three output engines (`0x0c4`) and completes in 7,676,945 cycles — indistinguishable from the single-engine epochs. Popcount 1/2/3 means: 7,663,590 / 7,662,956 / 7,676,945 |
| the operator form | `Conv2D_853` is `group=1, k=[1,1], stride=[1,1]` |
| the trace overhead | the printing is ~14 ms per traced epoch but epochs 0-320 are untraced and fast, and the cumulative counter excludes print time |

## Where to look next

1. **Find where the step change begins.** Re-run with the window opening at epoch
   ~280 instead of 335. Cumulative time is 106 ms at epoch 320 and the penalty is
   in force by 335, so the transition is inside a 15-epoch span. Whatever changes
   there — a pool crossing, a weight-streaming boundary, a different accelerator
   configuration — is the thing to look at.
2. **Identify the 12.6 ms constant.** 7.55M cycles at 600 MHz CPU / 9.4 ms at
   800 MHz NPU. If a runtime or ATON timeout has that period, that names the
   mechanism directly.
3. **Epoch 7 at 25x** (5,077,147 cycles against a 204,800 estimate) is a smaller
   instance of the same kind of anomaly in the fast region and may share a cause.

## Where Gate 4 stands

- The graph **executes**: 0 software epochs, everything on-chip, no PSRAM, weights
  at `0x70400000`, and epochs 0-320 run at the compiler's estimated speed.
- The stride-2 depthwise fold is **correct and free** (Round 12) and applies to all
  three occurrences in the full model, bit-exact.
- Two problems remain, and they may be the same problem: a uniform ~40x per-epoch
  penalty from around epoch 321, and a hard stall at epoch 352.

---

# Round 16 — correction: the "40x penalty" was a bug in my own instrument

**Round 15's second finding is withdrawn.** There is no per-epoch penalty. The
callback set the timestamp and *then* printed:

```c
g4_ep_t0 = port_dwt_get_cycles();
if (...) { beacon('<'); g4_num(...); ... }   /* 18 chars INSIDE the window */
```

Eighteen characters at 14400 baud, with a `USART_ISR_TC` wait on each, is
18 x 694 us = **12.5 ms** — which is precisely the "uniform 12.6 ms, +/-1 % across
seventeen epochs" that Round 15 reported as a timeout. The stability that made it
look like a fixed hardware wait was the stability of a fixed-length UART write.

The first measurement (epochs 0-23) printed only in `POST_END`, after `dt` was
computed, which is why those came out clean at ~0.33 ms and appeared to contradict
the later ones. That contradiction was the clue and it was not followed up.

## Re-measured, with the print moved ahead of the timestamp

Epochs 280-351, traced individually:

```
mean = 128,819 cycles = 0.215 ms      min = 21,695      max = 1,274,497
cumulative: epoch 64 = 35 ms, 128 = 57 ms, 192 = 79 ms, 256 = 92 ms
```

Consistent with epochs 0-320 and with the compiler's 114.7 ms estimate for the
whole graph. Three epochs — **289, 317 and 345, spaced exactly 28 apart** — cost
~1,274,400 cycles each; that is the heaviest operator of each repeating block, not
an anomaly.

**The NPU runs this graph at the speed it was designed to.**

## So there is exactly one problem left

Epoch **352** starts and never ends. It is `Conv2D_853`, and the block it sits in
is entirely regular:

```
Reshape_849 -> Conv2D_850 (g=256, k=[7,1], s=[1,1])  -> Q/DQ
            -> Conv2D_853 (g=1,   k=[1,1], s=[1,1])  -> Q/DQ
            -> Reshape_856 -> Relu_857 -> Reshape_858
            -> Conv2D_859 (g=256, k=[7,1]) -> Q/DQ -> Conv2D_862 (g=1, k=[1,1]) -> ...
```

`Conv2D_859`/`Conv2D_862` repeat the same pattern, and the equivalent blocks at
epochs ~334-343 and earlier all complete. The operator form is not the
discriminator this time, unlike `Conv2D_70`.

What is left that distinguishes epoch 352 is its **resource assignment** —
`in_streng 0x189` (engines 0,3,7,8), `out_streng 0x062` (engines 1,5,6). That is a
property of the schedule, not of the graph, and it is directly testable: recompile
with a different scheduler configuration and see whether the stall moves, changes
operator, or disappears. Round 10 used exactly this test to prove the *opposite* for
`Conv2D_70` — there, the hang followed the operator across two schedules.

## Method note

Three separate conclusions in this gate have now been overturned by controls or by
re-reading the instrument: "the image dies before `UART_Config`" (Round 8), "the
NPU is 2000x too slow" (Round 14), and "there is a uniform per-epoch penalty"
(here). Each time the instrument, not the target, was what changed. Two rules that
would have caught all three:

- **Never report a timing figure derived from wall-clock divided by a count.**
  Measure the thing itself.
- **When two measurements of the same quantity disagree, that is the finding.**
  Epochs 0-23 at 0.33 ms and epochs 335-351 at 12.6 ms could not both be right.

---

# Round 17 — the second blocker follows the operator too: `Conv2D_853`

Recompiled the folded full model with `--Omax-ca-pipe 1`, which yields a genuinely
different schedule — **617 epoch blocks instead of 629**, still 0 software epochs,
and it moves `Conv2D_853` from epoch 352 to epoch 349. The stall moved with it:

| | 629-epoch schedule | 617-epoch schedule |
|---|---|---|
| `Conv2D_853` scheduled at | epoch 352 | epoch 349 |
| **stalls at** | **352** | **349** |
| in / out streaming engines | `0x189` / `0x062` | `0x0ca` / `0x031` |
| epochs completed before it | 351 | 348 |

Same operator, two schedules, two epoch indices, two different engine assignments.
**The fault follows `Conv2D_853`** — the same conclusion, by the same test, as
Round 10 reached for `Conv2D_70`.

Timing in this run is again healthy: cumulative 116 ms by epoch 256, individual
epochs 48,096 to 260,574 cycles.

## What it is not

| candidate | evidence |
|---|---|
| operator form | `group=1, k=[1,1], stride=[1,1]`. It is **one of 126 `[256,256,1,1]` pointwise convolutions in this graph; the other 125 execute** |
| schedule position / epoch index | moved 352 -> 349, stall moved with it |
| streaming-engine assignment | `out` changed `0x062` -> `0x031`, stall unaffected |
| memory pool | cpuRAM2 is live only at epochs 11-38, 348-349, 377-380; npuRAM6 otherwise |
| weight or bias magnitude | `\|w\|max` 1.052 (rank 111 of 126), `\|bias\|max` 0.3496 (rank 61 of 126) — thoroughly ordinary. Median `\|w\|max` across the 126 is 2.325, range 0.553 to 22.21 |
| the block it sits in | `Reshape -> dw(k=7) -> Q/DQ -> pw(1x1) -> Q/DQ -> Reshape -> Relu -> ...`, and `Conv2D_859`/`Conv2D_862` repeat it identically and complete |

## Next, and it is the obvious remaining variable

The float weights are ordinary, but the **int8 quantisation parameters have not been
checked**. `q800_real_OE_3_3_1_Q.json` holds the per-tensor and per-channel scales
and zero points that atonn turns into the accelerator's requantisation multiplier
and shift. A multiplier or shift at the edge of the representable range is exactly
the kind of thing that would follow one specific convolution across schedules while
125 structurally identical ones are fine.

Concretely: pull the input, weight and output scales for `Conv2D_853` and for the
125 siblings, compute `scale_in * scale_w / scale_out` for each, and see whether
`Conv2D_853` is an outlier. If it is, the workaround is to nudge that layer's
quantisation — or to force this one node to software, since the graph currently has
zero software epochs and could absorb one.

## Gate 4 scoreboard

| | status |
|---|---|
| the part executes the graph | **yes** — 0 software epochs, all on-chip, no PSRAM |
| speed | **as designed** — ~0.2 ms/epoch, cumulative 116 ms by epoch 256 against a 114.7 ms whole-graph estimate |
| blocker 1: `Conv2D_70`, stride-2 depthwise | **fixed** — fold the stride into the following pointwise conv, bit-exact, free (Round 12) |
| blocker 2: `Conv2D_853`, plain 1x1 | **open** — localised and schedule-invariant, cause not yet found |

---

# Round 18 — the stalling operator was misidentified. It is `Conv2D_859`, a depthwise convolution.

Rounds 15 to 17 named `Conv2D_853` — an ordinary `group=1, k=[1,1]` pointwise
convolution — as the second blocker, and then spent three rounds failing to find
anything unusual about it. Nothing was unusual about it. **It is not the operator
that stalls.**

## The instrument, again

`g4_epoch_cb()` prints `g4_ep_n` (`audio_bm.c:203`), a **software counter
initialised to zero and incremented once per `POST_END`**. It is not
`eb->epoch_num`. atonn's `epoch_num` starts at **2** — the first two entries of
the graph are the parameter and input pseudo-epochs — so

```
trace epoch N  ==  ll_atonn_rt_epoch_block_array[N]  ==  epoch_num N+2
```

Every epoch number in rounds 9 through 17 is therefore **two too low**.

The masks printed alongside it were always correct, and they are what settles it.
`g4_num()` prints **decimal**, so the trace line `<352f019i393o098` carries
`in_streng_mask = 393 = 0x189` and `out_streng_mask = 98 = 0x062` — exactly the
values Round 15 quoted. Looking those masks up in the epoch-block table of the
build that produced them:

| build | trace printed | array index | real `epoch_num` | masks in the table |
|---|---:|---:|---:|---|
| folded, `--Omax-ca-pipe 4` | 352 | 352 | **354** | `in 0x189 / out 0x062` — matches Round 15 exactly |
| folded, `--Omax-ca-pipe 1` (Round 17) | 349 | 349 | **351** | `in 0x0ca / out 0x031` — matches Round 17 exactly |

In each build exactly one entry carries those masks, and in each it is the entry
at the printed index. The identification is not an inference from one artifact;
it is the same result twice, from two separately compiled schedules.

## What actually stalls

Both entries hold the same thing:

```
epoch_num 354 (ca-pipe 4)  /  epoch_num 351 (ca-pipe 1)
    Relu_857               ACTIV_ACC_V2 1
    Reshape_858
    Conv2D_859_subm_0      CONV_ACC_V2 2
    Conv2D_859_subm_1      CONV_ACC_V2 0
    Conv2D_859_subm_2      CONV_ACC_V2 1
```

`Conv2D_859` is `group=256, kernel_shape=[7,1], strides=[1,1], pads=[3,0,3,0]` —
**a depthwise convolution**, `/encoder/encoder.13/mconv.5/conv/Conv`. It is split
into three submasks driven on three convolutional accelerators inside a single
epoch.

**Round 17's reasoning was sound and its conclusion holds — it just named the
wrong node.** The fault does follow the operator across schedules: `Conv2D_859`
carries it from `epoch_num 354` to `epoch_num 351` when the schedule changes,
with different engine assignments both times. Everything Round 17 ruled out
about `Conv2D_853` is moot, because `Conv2D_853` executes correctly. It sits at
`epoch_num 352`, two epochs before the stall, and the board gets past it.

Note also that **both blockers are depthwise convolutions.** Blocker 1 was
`Conv2D_70`, `group=256, k=[3,1]`, stride 2. Blocker 2 is `group=256, k=[7,1]`.
Round 16's "the operator form is not the discriminator this time" was an artifact
of the misidentification.

## The discriminator, and it has no counterexample

Counting epochs that pack three or more `Conv..._subm_N` pieces into one epoch
block, over all 628 epochs of the folded graph:

| | |
|---|---:|
| epochs with **≥3** conv submasks | 37 |
| of those, **below** `epoch_num 354` | **0** |
| of those that also carry a `Relu` in the same epoch | 36 |
| of those, below 354 | **0** |

`epoch_num 354` is the **first epoch in the graph that ever programs three
convolutional accelerators from one convolution**, and the board has never
executed one. Every epoch it does execute successfully uses at most two.

The control is inside the same Citrinet block. `Conv2D_850` is *also* a `k=7`
`group=256` depthwise convolution, on the same tensor shape, six epochs earlier —
and the compiler spreads its three submasks across **two** epochs:

```
epoch_num 348   Conv2D_850_subm_1  CONV_ACC_V2 0     Conv2D_850_subm_2  CONV_ACC_V2 3
epoch_num 349   Conv2D_850_subm_0  CONV_ACC_V2 0     (with Conv2D_846 + its ca_pipe)
epoch_num 354   Conv2D_859_subm_0/1/2  all three, one epoch   <- stalls
```

Same operator form, same shapes, same block. The one that is split across two
epochs completes; the one packed into a single epoch does not.

**The confound, stated plainly.** All 37 three-submask epochs lie at or above
354, so "first occurrence" and "deepest part of the graph" coincide, and this
round cannot separate them from the artifacts alone. What kills or confirms it is
a build in which `Conv2D_859` is not split three ways — see below.

## Refuted this round

| hypothesis | verdict | what killed it |
|---|---|---|
| `Conv2D_853`'s requantisation multiplier is an outlier (Round 17's proposal) | **refuted** | its multiplier spans log2 −9.19 to −7.62; the 73 siblings proven to execute span **−32.30 to +0.04**. Inside the range at both ends. Rank 59 of 73 by minimum. Moot in any case — `Conv2D_853` is not the blocker |
| the stalling epoch needs 4 input streaming engines | **refuted twice** | the premise came from the mis-looked-up epoch. The real masks are 3-in/1-out (ca-pipe 4) and 3-in/2-out (ca-pipe 1) — not invariant. And 39 epochs below the stall use ≥4 input engines and complete |
| `ca_pipe` splitting across two conv accelerators | **refuted** | 175 epochs split, 75 of them below the stall. The exact unit pair `(CONV_ACC_V2 0, CONV_ACC_V2 3)` occurs in 70 epochs |
| MACC magnitude | **refuted** | `epoch_num 354` is 358,400 MACs; `epoch_num 28` does 52,531,200 and completes |
| live activation bytes / npuRAM6 pressure | **refuted** | 153,600 B in 3 non-overlapping aligned buffers, npuRAM6 at 33.5 %, rank 418 of 631 |
| a unique `(in,out)` engine-mask pair | **refuted** | 46 % of all epochs have a mask pair that occurs exactly once. Unique is the norm |
| descriptor fields (`batch_depth`, `kfilt_*`, `simd`, `vshift`, `frame_tot_cnt`) | **refuted** | every value `Conv2D_853` carries is shared with 31–73 executed siblings; `simd`, `vshift` and `kfilt_last` are uniform across all 126 |

## A separate finding: the squeeze-excitation epochs are 26× to 103× over estimate

Round 14 flagged epoch 7 as a 25× outlier and left it unexplained. Round 16
found epochs 289, 317 and 345 at ~1,274,400 cycles and attributed them to "the
heaviest operator of each repeating block, not an anomaly". **That attribution is
wrong.** Those epochs are among the *lightest* the compiler estimates:

```
44 epochs carry estimated_tot_cycles = 49,152 / estimated_npu_cycles = 2,731
they occur in PAIRS spaced 27-28 apart:  7,8  36,37  64,65 ... 289,290  317,318  345,346
each pair is the squeeze-excitation block:
    Conv -> Add(bias) -> Reshape -> Relu -> Reshape
    Conv -> Add(bias) -> Reshape -> Sigmoid
```

against a whole-graph maximum estimate of 3,960,600. So epochs 7, 289, 317 and
345 are not heavy operators; they are trivial ones that overrun their estimate by
26× (and epoch 7 by 103×). The tensors are tiny — the SE branch reduces to 256
values — so the cost is almost certainly fixed per-epoch overhead rather than
computation. 44 epochs is enough to matter for throughput and it is worth
settling once Gate 4 closes.

(These indices are the compiler's `epoch_num`; Round 16's measured 289/317/345
were trace numbers, hence `epoch_num` 291/319/347. The 28-epoch spacing and the
pairing are what identify the family, and both match.)

## Reproducibility, restored

The scratchpad holding rounds 11-17's artifacts was wiped. Everything needed to
continue has been regenerated and is now checked in rather than living in `/tmp`:

| | |
|---|---|
| `model/fold_stride2.py` | the blocker-1 fold, discovering its sites rather than hardcoding them. **0 of 3,075,000 output elements differ over 30 random inputs**, max\|diff\| = 0; identical decoded transcript on real speech |
| `compile/gen_model.sh` | the compile driver, preserving the **whole** workspace per tag — `network_c_info.json`, `*_OE_3_3_1_Q.json`, `network.csv`, the weight blob — which is exactly what was lost |
| `board/BUILD.md` | build, sign (`-align` is mandatory), flash, and read, with the trace format decoded |
| `firmware/vendor-mods/gate4.patch` | a real patch. The file it replaces contained no diff markers, stopped mid-function, and was rejected by both `patch(1)` and `git apply` — there had been no automated way to apply this step |
| `artifacts/compile/g800_fold/` | the folded compile, reproducing Round 13: **628 epochs, 0 SW, 0 hybrid**, cpuRAM2 200 kB, npuRAM6 425 kB, no PSRAM, weights 9.726 MB at `0x70400000` |
| `artifacts/compile/round17_capipe1_rescued/` | Round 17's `--Omax-ca-pipe 1` build, rescued from the vendor tree before step 6 of `apply_vendor_mods.sh` would have overwritten it. 616 blocks; it is what confirms the mask lookup above |

`board/flash_and_verify.sh` and `board/probe_flashboot.sh` pointed at
`/home/claroche/stm32n6-**tts**/`, which does not exist; they now derive the repo
root from their own location. `compile/audio_profile.json` pointed its
`memory_pool` at a wiped scratchpad.

## Two candidate discriminators, and neither can be separated from the artifacts

An adversarial re-check of the streaming-engine analysis produced a second
property with no counterexample, and it is not the same set:

| property | epochs in the graph | occurring **below** the stall |
|---|---:|---:|
| ≥3 `Conv..._subm_N` pieces in one epoch block | 37 | **0** |
| ≥3 concurrent output streams into npuRAM6 | 45 | **0** |

The first set is a strict subset of the second, and `epoch_num 354` is the first
member of both. The eight epochs in the second set but not the first all lie
above 354. **Nothing in the compiled artifacts can separate these two, because
the graph never exercises either property before the point where it stops.**

The coarser version of the second property — three output streams to *anywhere* —
is refuted: `epoch_num 348` drives three (`out = 0x0c4`) and completes. That is
the completion Round 15 recorded. Two of its three go to npuRAM6 and one to
cpuRAM2; the stalling epoch's three all go to npuRAM6.

## The experiment that separates them, and it is one flash

Compiling the same folded graph with **ST's stock option string** — that is,
dropping `--Oauto-sched`, which is `compile/GATE2.md`'s own shipping
configuration and still gives **618 epochs, 0 SW, 0 hybrid** — rearranges the
packing:

```
epoch_num 348   Conv2D_850_subm_0/1/2  + Identity     in 0x2a9 / out 0x146
epoch_num 353   Conv2D_859_subm_0/1/2  + Relu_857     in 0x342 / out 0x031
```

`Conv2D_850` is the depthwise convolution that **completes** under
`--Oauto-sched`, where its three submasks are spread across two epochs. Here they
are packed into one, and it now precedes `Conv2D_859`.

| the board stalls at trace | printed masks | conclusion |
|---:|---|---|
| **346** | `i681 o326` | the three-submask packing is causal. `Conv2D_859` was never special, and `Conv2D_850` stalls the moment it is packed the same way |
| **351** | `i834 o049` | it is `Conv2D_859` specifically; the packing is incidental |
| neither, runs to completion | — | the property belongs to the `--Oauto-sched` schedule itself |

The two outcomes print different masks as well as different numbers, so the
answer does not depend on trusting the epoch counter that caused this round's
correction. Note cpuRAM2 rises to 500 kB without `--Oauto-sched` (Gate 2 measured
exactly this), which is still clear of the FSBL.

## The escape hatch, measured rather than assumed

There is **no compiler option that pins a named node to software**. The full
option set was enumerated from the `atonn` binary, including hidden options via
the undocumented `--emit-options-md` / `--help-full`; `--node-processor-confs`
is keyed by operator *kind*, not node name, and is a capability table rather than
a placement directive.

A single node *can* be forced to software by graph surgery, and it was measured
on a real subgraph before being applied: replace the weight `DequantizeLinear`
with a float32 initializer and the compiler emits `DequantizeLinear → Conv(float)
→ QuantizeLinear` as three pure-SW epochs, leaving the int8 tensors either side
untouched. `-DLL_ATON_SW_FALLBACK` is already in the build
(`Projects/GS/Makefile:198`).

For `Conv2D_859` this is cheap: the epoch is **358,400 MACs**, not the 13.1 M of
the pointwise convolution that was wrongly blamed. `artifacts/onnx/q800_fold.onnx`
with that rewrite applied verifies at **max|diff| = 0 over 820,000 output
elements**.

Two things that do *not* work, recorded so they are not retried:

- `--onnx-omit-opt split_large_mask_conv` does remove every three-submask epoch,
  but the depthwise convolutions then fall back wholesale: **755 epochs, 92 SW,
  46 hybrid**. Not a deployment.
- the `dilations=[2]` trick — a numerical no-op that makes the compiler emit
  `SpaceToDepth → Conv → DepthToSpace` — applies only to `k=1` convolutions.
  `Conv2D_859` is `k=7`, so it does not transfer.

## Round 18 board runbook

Three images are staged under `artifacts/images/`, each with its matching weight
blob. Nothing needs building; flash, power-cycle, read.

| image | build defines | what it answers |
|---|---|---|
| `baseline_fold/` | `-DGATE4_CANNED` | control. Does the regenerated folded graph still stall where Round 17 left it? |
| `noauto_discriminator/` | `+ -DGATE4_BEACON -DGATE4_EPOCH_TRACE -DGATE4_EPOCH_FROM=330` | **the decisive one.** Stall at trace 346 (`i681 o326`) ⇒ the three-submask packing is causal; at trace 351 (`i834 o049`) ⇒ `Conv2D_859` specifically |
| `float859_escape/` | `-DGATE4_CANNED` | the escape hatch. `Conv2D_859` in float on the M55, 4 SW epochs, 358,400 MACs. Should print `# GATE4 PASS` |

Run `noauto_discriminator` first — it is the only one that produces new
information regardless of outcome.

```bash
export PATH=/home/claroche/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin:$PATH
CLI=STM32_Programmer_CLI
EL=/home/claroche/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr
IMG=artifacts/images/noauto_discriminator

# switches BOTH RIGHT (development), then:
$CLI -c port=SWD mode=UR --extload $EL -w $IMG/GS_Audio_N6_sign.bin 0x70100000
$CLI -c port=SWD mode=UR --extload $EL -w $IMG/network_data.bin     0x70400000

# switches BOTH LEFT (boot from flash), POWER CYCLE -- not the reset button
python board/read_uart.py 120
```

Reading the trace: `<NNNfFFFiIIIoOOO>DDDDDDDD` where `NNN` is the **0-based
execution counter** (add 2 for atonn's `epoch_num`), `FFF`/`III`/`OOO` are flags
and the input/output streaming-engine masks **in decimal**, and `DDDDDDDD` is the
epoch's cycle count from the PMU. A line that opens and never closes is the
stall. Note `g4_num(..., 3)` prints only three digits, so a mask ≥ 1000 wraps —
`i681` and `i834` are both safe, but do not trust a printed mask without checking
it against the epoch-block table.

`board/flash_and_verify.sh` reads from the vendored build tree rather than
`artifacts/images/`, so it verifies whatever was built last; use it when you have
just rebuilt, and the two `-w` commands above when flashing a staged image.
Weights go at **`0x70400000`**, which is also the app-slot end to pass as the
script's second argument.

---

# Round 18b — the board answers: it is the activation unit, not the submask split

The discriminating build ran. `artifacts/images/noauto_discriminator/`, ST's
stock option string, 618 epochs, 0 SW, 0 hybrid, booted from flash. Full trace:
[`board/traces/round18_noauto_discriminator.log`](traces/round18_noauto_discriminator.log).

```
<345f019i600o002>01274665
<346f019i681o326>00178933      <- Conv2D_850_subm_0/1/2   COMPLETES
<347f019i460o048>00249361
<348f019i129o256>00081560
<349f019i228o018>00177744
<350f019i016o128>00048024
<351f019i834o049               <- Conv2D_859_subm_0/1/2   OPENS AND NEVER CLOSES
```

Both predicted masks landed exactly where Round 18 said they would — `i681 o326`
at trace 346 and `i834 o049` at trace 351 — which independently confirms the
epoch-numbering correction, since those masks were derived from the epoch-block
table and not from the board.

**The three-submask packing is refuted.** `Conv2D_850` is a `k=7`, `group=256`
depthwise convolution split into three submasks across three convolutional
accelerators in a single epoch, and it **completes in 178,933 cycles**. That was
this round's leading hypothesis and the board killed it.

## What is left, with a control on both sides

The two epochs differ in exactly one thing:

| `epoch_num` | trace | conv submasks | activation unit | result |
|---:|---:|---:|---|---|
| 348 | 346 | 3 | none (`Identity`) | **completes**, 178,933 cycles |
| 353 | 351 | 3 | **`Relu_857`**, `ACTIV_ACC_V2` | **stalls forever** |

Counting over the whole graph: **36 epochs program three or more convolutional
accelerators *and* an activation accelerator, and not one of them lies below the
stall.** The only three-submask epoch without an activation unit is 348 — and it
runs. So the shape of the defect is

> three convolutional accelerators driven from one depthwise convolution in an
> epoch that also programs an activation accelerator

with one positive case and one negative control, both measured on silicon.

## The affected set is perfectly regular

The 36 are exactly `/encoder/encoder.{13..21}/mconv.{5,10,15,20}/conv/Conv` —
nine encoder blocks by four depthwise convolutions each, all `group=256`. They
are **not** all `k=[7,1]`: 20 are (`encoder.13..17`) and 16 are `k=[9,1]`
(`encoder.18..21`). What they share is a temporal kernel long enough to split
into three submasks, with a `Relu` chained into it. Blocks 0-12 are unaffected because their temporal kernels are shorter
and split into fewer than three submasks; `mconv.0` of each block escapes because
its epoch carries no activation, which is precisely the `epoch_num 348` control.

**This kills the escape hatch as built.** `artifacts/images/float859_escape/`
forces only `Conv2D_859` to software, so it would clear `epoch_num 353` and stall
at 358, the next `mconv.10` site. A per-node software fallback would need 36
sites and 144 software epochs; at 358,400 MACs each that is 12.9 MMAC of float
convolution on the M55 plus dequantise/requantise of 51,200 elements per site,
which is not obviously affordable against a ~115 ms budget and has not been
measured.

## Where this leaves blocker 2

Both blockers are now depthwise convolutions with a *specific emitted hardware
configuration*, not with the operator's arithmetic:

| | blocker 1 | blocker 2 |
|---|---|---|
| operator | `Conv2D_70`, `group=256 k=[3,1]` | `Conv2D_859` and 35 siblings, `group=256 k=[7,1]` |
| trigger | stride 2 | 3 conv accelerators + an activation accelerator in one epoch |
| sites | 3 | **36** |
| fix | move the stride to the pointwise partner — bit-exact, free | **open** |

The next thing to try is whatever prevents the activation from being scheduled
into the same epoch as a three-way-split depthwise convolution, because
`epoch_num 348` proves that configuration executes. That is a scheduling
property, so an option sweep is the cheap first move; a graph rewrite that splits
the depthwise convolution across channels is the fallback.

## The mechanism: an activation accelerator feeding convolutional accelerators

Reading the stream-switch configuration of the two epochs the board decided
between, in the build it actually ran (`g800_noauto`):

```
epoch_num 353  -- STALLS
  ACTIV   1 port 0  <- STRENG 8 port 0     Relu_857 reads memory
  CONVACC 0 port 0  <- ACTIV  1 port 0     Conv2D_859_subm_2 data in
  CONVACC 2 port 0  <- ACTIV  1 port 0     Conv2D_859_subm_1 data in
  CONVACC 1 port 0  <- ACTIV  1 port 0     Conv2D_859_subm_0 data in

epoch_num 348  -- COMPLETES
  CONVACC 0/1/3 port 0 <- STRENG 0 port 0  Conv2D_850_subm_0/1/2 data in
```

Both epochs broadcast one stream-switch source port to **three** convolutional
accelerator input ports. The only difference is what the source is: a streaming
engine in the epoch that runs, an **activation accelerator** in the epoch that
hangs.

| property | epochs | completed below the stall |
|---|---:|---:|
| max source fan-out ≥ 3 (any source) | 54 | **6** — incl. `epoch_num 348`, the control |
| max source fan-out ≥ 4 (any source) | 9 | **5** |
| **any `ACTIV → CONVACC` link** | **36** | **0** |

So three-way fan-out through the stream switch is fine; it is fine at four-way.
What this part does not do is route an activation accelerator's output straight
into a convolutional accelerator.

**The obvious refinement cannot be measured on this graph.** All 36 `ACTIV →
CONVACC` epochs have a fan-out of exactly three — the histogram is
`{0: 582, 3: 36}` — so nothing distinguishes "an `ACTIV → CONVACC` link is broken"
from "an `ACTIV` source driving three destinations is broken". Answering that needs
a graph that exercises a fan-out of one or two, which this one never does.

Note this also explains the shape of the affected set without appealing to kernel
size. `mconv.0` of each block escapes not because it is different arithmetic but
because the block's `Relu` is not adjacent to it in the schedule; the four sites
per block that do stall are exactly those where a `Relu` immediately precedes a
depthwise convolution and the compiler chains them through the switch rather than
through a buffer.

---

# Round 18c — the full model executes. `--force-all-in-out-to-mem` clears blocker 2.

```
# ---- run 1 ----
# in=0x34350000 out=0x343820d0 scale=8.297212 off=0
# fed 64000 B of 64000 B input
# invoking...
# invoke returned
# invoke 116393913 cycles = 193.989 ms at 600000000 Hz
# mismatches 5 / 100
```

**`AiDPUProcess()` returns.** All 1064 epochs of the full 800-frame Citrinet
encoder execute on the NPU, booted from flash, repeatably — run 2 came back at
116,391,956 cycles, 1,957 cycles from run 1. Trace:
[`board/traces/round18_forcemem_pass.log`](traces/round18_forcemem_pass.log).

This is the fix predicted by the mechanism, and it works by intervention rather
than by argument: removing every `ACTIV → CONVACC` stream-switch link removes the
stall.

## The option sweep, and only one option does it

| option | epochs | SW | hybrid | `ACTIV→CONVACC` epochs |
|---|---:|---:|---:|---:|
| baseline (ST stock) | 618 | 0 | 0 | 36 |
| `--Oconv-split-cw` | 640 | 0 | 0 | 36 |
| `--Oconv-split-kw` | 794 | 3 | 1 | 36 |
| `--Oconv-split-stripe` | 719 | 0 | 4 | 36 |
| `--Ono-clone-dma` | 629 | 0 | 0 | 36 |
| `--Omax-ca-pipe 2` | compile failed | | | |
| **`--force-all-in-out-to-mem`** | **1064** | **0** | **0** | **0** |

It is documented "only for debugging" and it is a blunt instrument — it forces
every unit to read and write through memory instead of chaining through the
stream switch. The cost is smaller than that description suggests:

| | baseline | force-to-mem |
|---|---:|---:|
| epochs | 618 | 1064 |
| SW / hybrid | 0 / 0 | **0 / 0** |
| cpuRAM2 | 500 kB | 800 kB (78.1 %) |
| npuRAM6 | 425 kB | 425 kB (94.87 %) |
| PSRAM | none | **none** |
| compiler estimate | 92,450,794 cyc | 119,283,496 cyc |
| **measured** | — | **116,393,913 cyc = 194.0 ms** |

Measured is **2.4 % under** the compiler's estimate for this build, which is the
first end-to-end confirmation that the scheduler model predicts this graph. For an
8 s window, 194 ms is ~41x real time.

## Correctness: 5 frames of 100, and the reference is not unique

`# mismatches 5 / 100` fails the gate as written. The differences:

| frame | host | device | host top1/top2 margin | margin rank of 100 | device's rank in host logits |
|---:|---:|---:|---:|---:|---:|
| 13 | 58 | 552 | 1.327 | **3** | 6 |
| 48 | 53 | 29 | 1.327 | **4** | **1** (runner-up) |
| 55 | 38 | 1024 | 8.759 | 57 | 6 |
| 56 | 62 | 38 | 9.024 | 59 | 39 |
| 57 | 1024 | 62 | 3.450 | 12 | **1** (runner-up) |

Frames 55-57 are not five independent errors, they are **one blank-placement
shift**: the device emits blank at 55 and then reproduces the host's 55 and 56 at
56 and 57. CTC greedy decoding collapses blanks and repeats, so it **vanishes from
the transcript**. What survives is two substitutions, at the 3rd and 4th tightest
margins in the whole utterance:

```
reference  mister quilter is the apostle of the middle classes and we are glad ...
host int8  mister crter   is the apostle of the middle classes and we are glad ...
device     mister quirter is the apostle of the middle classes and were    glad ...
```

One in each direction — the device is closer to the reference on "quilter"
(`qui` vs `▁c`) and further on "we are" (`re` vs `▁are`).

**And the host reference is not a fixed point.** `kExpectedTokens` was generated
with onnxruntime's default graph optimisations enabled. Running the *same graph*
on the *same input* with `ORT_DISABLE_ALL` instead:

| comparison | frames differing of 100 |
|---|---:|
| ORT `ENABLE_ALL` vs ORT `DISABLE_ALL` | **7** |
| **device vs ORT `ENABLE_ALL`** | **5** |

**The device agrees with the host reference more closely than the host reference
agrees with itself.** Exact per-frame argmax equality is therefore the wrong
criterion: it asks silicon to reproduce one arbitrary onnxruntime configuration
more faithfully than onnxruntime reproduces it under a flag change. (The same run
re-confirms the fold: `q800_fold.onnx` and `q800_real.onnx` give identical argmax
at both optimisation levels.)

## Where Gate 4 stands

| | status |
|---|---|
| the part executes the graph | **YES** — 1064 epochs, 0 SW, 0 hybrid, all on-chip, no PSRAM, `AiDPUProcess()` returns, repeatable |
| latency | **194.0 ms** measured, 2.4 % under the compiler's estimate |
| blocker 1 — stride-2 depthwise | **fixed**, bit-exact and free (Round 12) |
| blocker 2 — `ACTIV → CONVACC` chaining | **worked around** by `--force-all-in-out-to-mem`; root cause is a silicon or compiler defect and should go upstream |
| per-frame argmax equality with host ORT | **5 / 100**, against 7 / 100 for ORT against itself |

The remaining work is not correctness, it is cost. `--force-all-in-out-to-mem`
is global; the graph only needs the 36 `Relu → depthwise` chains broken. A
targeted fix should recover most of the gap between 618 and 1064 epochs.

---

# Round 19 — the root cause, and a graph rewrite that beats the workaround

## Why the compiler chains the Relu forward

atonn canonicalises every convolution to 2-D. The source graph is 1-D, so atonn
wraps each convolution in `Reshape` 3D↔4D pairs. At the 36 defective sites the
`Relu` is a **3-D** operator sandwiched between two **4-D** convolutions:

```
Conv2D_853(4D) -> Q/DQ -> Reshape_856(4D->3D) -> Relu_857(3D) -> Reshape_858(3D->4D) -> Conv2D_859(4D)
```

`Reshape_856` becomes an epoch of its own — a real `STRENG -> STRENG` copy — which
separates the `Relu` from its **producer**. Having nothing to chain backwards to,
the compiler chains it **forwards into its consumer**: `ACTIV 1 -> CONVACC 0/1/2
port 0`, the stall (`g800_noauto/network.c:84583`).

Where no `Reshape` separates a convolution from its `Relu`, the compiler chains the
other way — `CONVACC -> ARITH(bias) -> ACTIV -> STRENG` — which lands the `Relu`'s
result **in memory**. Twenty-six such epochs execute below the stall in the build
the board ran. That is the whole difference.

## The fix: keep the Relu on the 4-D tensor

```
DQ -> Reshape([0,0,-1,1]) -> Relu -> Q -> DQ -> Reshape([0,0,-1]) -> Conv
```

atonn's own `fuse_consecutive_reshapes` / `eliminate_nop_reshape` cancel its
inserted pair against ours, the `Relu` ends up adjacent to its producing
convolution, and the depthwise convolution reads from a streaming engine — exactly
the `epoch_num 348` configuration the board completed in 178,933 cycles. `Reshape`
does not change values, so the rewrite is exact by construction.

`model/break_relu_chain.py`, discovering its sites rather than hardcoding them.

## It beats `--force-all-in-out-to-mem` on every axis

| | baseline (stalls) | `--force-all-in-out-to-mem` (flashed, works) | **rewrite, 84 sites** |
|---|---:|---:|---:|
| epochs | 618 | 1064 | **448** |
| SW / hybrid | 0 / 0 | 0 / 0 | **0 / 0** |
| `ACTIV→CONVACC` epochs | 36 | 0 | **0** |
| cpuRAM2 | 500 kB | 800 kB | **300 kB** |
| npuRAM6 | 425 kB | 425 kB | 425 kB |
| PSRAM | none | none | **none** |
| estimated cycles | 92,450,794 | 119,283,496 | **76,000,592** |
| latency | — | **194.0 ms measured** | ~127 ms predicted |

−58 % epochs, −36 % estimated cycles, −62 % cpuRAM2 against the workaround — and
it beats the *stalling* baseline too, because the rewrite deletes 132 of the 169
pure copy epochs (15,760,000 → 3,568,000 cycles of `Reshape`/`Identity`).

Verified `max|diff| = 0` over 3,075,000 output elements, 30 random inputs. The
check is sensitive: flipping one LSB of one int8 weight makes it report
`max|diff| = 3.45` with 83 % of elements differing. Total MACs are unchanged
across every build (2,235,241,315), so the epoch reduction is not work going
missing. The weight blob is byte-identical (`md5 c81d84a7a9cbff549bfa9fa4df8923ff`)
— **only the application image needs reflashing.**

## The fan-out question, answered as a side effect

Round 18b could not separate "any `ACTIV → CONVACC` link is broken" from "an
`ACTIV` source driving *three* destinations is broken", because all 36 sites had
fan-out exactly 3. The rewrite settles it from the other direction: ACTIV-source
fan-out goes `{1:117, 3:36}` → `{1:153}`. Every one of the 36 three-way links
becomes a one-way link **to a streaming engine**, and none to a CONVACC. Purpose-built
reproducers with fan-out 1, 2 and 4 are in `artifacts/onnx/repro2/`.

## Negative results, so they are not retried

| tried | outcome |
|---|---|
| `--Oconv-split-cw` / `-kw` / `-stripe`, `--Ono-clone-dma` | all leave 36 |
| `--Oconv-split-stripe-full` | compile aborts, `E103` stripe validation |
| `--d-onnx-temps` on the 36 Relu edges — the only *targeted* materialisation lever in the option set | silently ignored (epoch table byte-identical to baseline), then crashes the wrapper |
| `--Oauto` | 192 combinations, ~4.4 h, and contains no chaining lever — every axis in it was measured to leave all 36 |
| rewriting `/encoder/encoder.22/mconv.0`, whose `Relu` is produced by the residual `Add` rather than a `Conv` | **creates** a defect the baseline did not have. The script now skips any site whose `Relu` producer is not a `Conv` |
| `--force-all-in-out-to-mem` without `--Oauto-sched` | 1052 epochs but *more* cycles than with it |

## A minimal reproducer for ST

`board/REPRO-blocker2.md`, with `artifacts/onnx/repro2/`: a **9-node, 2,453-byte**
ONNX that compiles to three epochs and reproduces `CONVACC 0/1/2 port 0 <- ACTIV 0
port 0`, beside a 6-node control that produces `CONVACC 0/1/2 port 0 <- STRENG 7
port 0` and differs in nothing else. That is the bug report.

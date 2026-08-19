# Gate 5 — what audio input this board actually has

Gate 5 needs speech in front of `firmware/src/citrinet_fe.c`. This document
establishes, from the board support package and the vendored application, what
the STM32N6570-DK can physically deliver, and recommends the cheapest way to get
Gate 5 moving.

Every line number below is against the tree as it sits on disk, i.e. the
vendored ST package with `firmware/apply_vendor_mods.sh` and
`firmware/vendor-mods/gate4.patch` applied. Where that shifts a file relative to
the pristine clone it is called out.

`Patch/...audio.c` throughout means
`vendor/.../Projects/Common/Patch/stm32n6570_discovery_audio.c`, which is the BSP
audio driver the build actually compiles (`Projects/GS/Makefile:19`); the copy
under `Drivers/BSP/STM32N6570-DK/` is not in `C_SOURCES`. The `bm` config also
links the clock patch `Patch/stm32n6570_discovery_audio.patch.hsi_600_400.c`
(`Projects/GS/bm.mk:2`), which overrides the weak `MX_MDF1_ClockConfig()` and
`MX_SAI1_ClockConfig()`.

---

## 1. Is there a built-in microphone?

**Yes. One. It is not an IMP34DT05 — it is an MP23DB01HP.**

```
/* MP23DB01HPTR digital microphone */
#define AUDIO_IN_DEVICE_DIGITAL_MIC      0x10U
```

`Drivers/BSP/STM32N6570-DK/stm32n6570_discovery_audio.h:194-195`. That is the
only microphone part number any *board support* file names, in either vendored
package, and the second package carries the identical line:
`vendor/STM32N6-GettingStarted-ObjectDetection/STM32Cube_FW_N6/Drivers/BSP/STM32N6570-DK/stm32n6570_discovery_audio.h:194`.

**Where "IMP34DT05" came from, and why it is wrong.** `firmware/WORKLIST.md:129`
cites `Projects/Dpu/ai_model_config.h:38`:

```c
#define CTRL_X_CUBE_AI_SENSOR_TYPE               COM_TYPE_MIC
#define CTRL_X_CUBE_AI_SENSOR_NAME               "imp34dt05"
```

That is a **string in a model-configuration header, not a board fact**, and it
is dead: `grep -rn CTRL_X_CUBE_AI_SENSOR_NAME` over the whole package finds the
definition, one `#ifndef` fallback, and **no use**. The fallback
(`Projects/Dpu/dpu_config.h:77-78`) is `"ism330dhcx"` — a 6-axis IMU. The field
is model-zoo provenance metadata that travels with the AED model, and it names
whatever board that model was recorded on. The BSP is what talks to the
hardware.

Five documents in this repository repeat the wrong part: `README.md:190`,
`firmware/FRONTEND.md:245`, `docs/FEASIBILITY.md:129`, `docs/GATES-5-6.md:232`,
`firmware/WORKLIST.md:129`. See §7.

**How it is wired.** PDM into MDF1, not I²S, not through a codec:

| | | source |
|---|---|---|
| bitstream in | **PE8**, `GPIO_AF4_MDF1`, `MDF1_DATIN0` | `..._audio.h:263-266` |
| clock out | **PE2**, `GPIO_AF4_MDF1`, `MDF1_CCK0` | `..._audio.h:259-262` |
| filter | `MDF1_Filter0` | `Patch/stm32n6570_discovery_audio.c:2004` |
| DMA | `GPDMA1_Channel0`, `GPDMA1_REQUEST_MDF1_FLT0` | `..._audio.h:271-275` |
| BSP instance | **1** (instance 0 is the SAI path) | `..._audio.h:182-186` |

**One microphone, mono, and the BSP will not pretend otherwise.** There is a
single `DATIN0` pin, a single filter, and `FilterBistream = MDF_BITSTREAM0_FALLING`
(`Patch/...audio.c:3067`) — one edge of one bitstream.
`BSP_AUDIO_IN_SetChannelsNbr()` returns `BSP_ERROR_FEATURE_NOT_SUPPORTED` for
anything but 1 (`Patch/...audio.c:2694-2697`), commented "only mono channel is
supported" (`:2705`). ST's own board description agrees: "One MEMS digital
microphone".

The MP23DB01HP datasheet — **not in the vendored tree**, so this is the one
number here that is not primary — gives sensitivity −24 dBFS typ (−25 to −23) at
94 dB SPL, SNR 64 dB(A), AOP 135 dB SPL. `docs/FEASIBILITY.md:129` assumed
−26 dBFS at 94 dB SPL. The 2 dB difference does not change any conclusion in
§2(d) of that document; the part number does need fixing.

---

## 2. Every input the BSP supports

`AUDIO_IN_INSTANCES_NBR == 2` (`..._audio.h:186`), and the comment above it says
what they are: instance 0 is the SAI path, instance 1 is the MDF path.
`BSP_AUDIO_IN_SetDevice()` is a no-op — "Nothing to do because there is only one
device for each instance" (`Patch/...audio.c:2443`) — so *instance* is the only
real selector.

| instance | device | peripheral | what it needs that the DK does not already have | selected by ST's app? |
|---|---|---|---|---|
| **1** | `AUDIO_IN_DEVICE_DIGITAL_MIC` — on-board MP23DB01HP | MDF1 Filter0, PE8/PE2 | **nothing** | **yes** |
| 0 | `AUDIO_IN_DEVICE_ANALOG_MIC` — "Analog microphone input from 3.5 audio jack connector" (`..._audio.h:191`) | SAI1_Block_B, SD on PE3 (`..._audio.h:239-245`), through the codec over I²C | a headset with an electret mic on the jack | no |

The codec is chosen by board revision, not at runtime:
`STM32N6570_DK_REV` defaults to `STM32N6570_DK_C01`
(`Projects/GS/Inc/stm32n6570_discovery_conf.h:39-41`) and revisions ≥ B01 select
**WM8904** (`:71-76`, I²C address `0x34`, `..._audio.h:140-144`). CS42L51 is the
A01 path only. Both drivers ship (`Projects/GS/Makefile:52-61`).

For the analog path the BSP hard-codes `codec_init.InputDevice = WM8904_IN_MIC1`
(`Patch/...audio.c:1705`, `:1973`, and the ternary at `:443-446`). `WM8904_IN_MIC1`
routes **IN1L** with `MICBIAS_ENA = 1`, `LIN_VOL = +14 dB` and
`ADC_VOL = +17 dB` (`Drivers/BSP/Components/wm8904/wm8904.c:146-147, 288-297`) —
that is a bias-fed electret headset-microphone front end, not a line input.

`WM8904_IN_LINE2` exists in the codec driver (`wm8904.h:115-116`,
`wm8904.c:268-286`, routing IN2L/IN2R at −1.5 dB with no mic bias) but **no BSP
path ever selects it**, and nothing in the vendored tree says IN2L/IN2R are
routed to a connector on the MB1939. Using it means editing the BSP *and*
establishing from the board schematic that the pins go somewhere.

**There is no USB audio path and no way to build one from this tree.** The
package ships `stm32n6xx_hal_pcd.c` and `stm32n6xx_ll_usb.c` in the HAL, and
nothing else: no USB device library, no CDC or UAC class, no descriptors, no
USB source in `Projects/GS/Makefile`. `find . -iname '*usb*'` outside the HAL
returns one header, `Drivers/BSP/Components/Common/usbtypecswitch.h`.

**The ST-LINK VCP is `USART1` on PE5/PE6** (`stm32n6570_discovery.h:237-254`,
configured at `Projects/Common/misc_toolbox.c:181-227`), 8N1, FIFO disabled
(`:226`), `UART_MODE_TX_RX` (`:215`). No code anywhere reads it —
`grep -rn 'HAL_UART_Receive' Projects/` returns nothing; `_read()` is the weak
newlib stub (`Projects/GS/Src/syscalls.c:67-74`). And the baud rate is **14400**
(`Projects/GS/Inc/app_config.h:63`), with a measured note in that same file that
921600 breaks `UART_Config` at this clock.

---

## 3. The on-board path in detail

### Rate, depth, decimation

| | value | source |
|---|---|---|
| sample rate | `AUDIO_FREQUENCY_16K` = 16000 | `stm32n6570_discovery_conf.h:55` |
| resolution requested | `AUDIO_RESOLUTION_16B` | `audio_bm.c:1211` |
| channels | 1 (enforced) | `audio_bm.c:1212`, `Patch/...audio.c:2694` |
| CIC | `MDF_ONE_FILTER_SINC4` | `Patch/...audio.c:172-180` |
| CIC decimation | **32** at 16 kHz | `Patch/...audio.c:152-160` |
| reshape filter | on, ratio **4** | `Patch/...audio.c:3081-3082` |
| high-pass | on, `MDF_HPF_CUTOFF_0_000625FPCM` (≈10 Hz at 16 kHz) | `Patch/...audio.c:3084-3085` |
| integrator | off | `Patch/...audio.c:3087` |
| offset | 0 | `Patch/...audio.c:3079` |
| acquisition | `MDF_MODE_ASYNC_CONT` | `Patch/...audio.c:3089` |
| serial interface | `MDF_SITF_NORMAL_SPI_MODE`, clock `MDF_SITF_CCK0_SOURCE` | `Patch/...audio.c:3064-3065` |
| MDF gain | **2** at 16 kHz | `Patch/...audio.c:162-170` |
| proc clock divider | 2; output clock divider 12 | `Patch/...audio.c:182-200` |

Total decimation from bitstream to PCM is **32 × 4 = 128**, so the PDM clock on
PE2 is 128 · Fs. *Derived, not measured:* the 16 kHz branch of
`MX_MDF1_ClockConfig` (`Patch/stm32n6570_discovery_audio.patch.hsi_600_400.c:63-88`)
builds PLL4 from HSI as 64/8 · 172 = 1376 MHz, ÷7 ÷4 = 49.142857 MHz, IC8 divider 1;
÷2 (proc) ÷12 (output) = 2.047619 MHz, and 2.047619 MHz / 128 = **15,997.0 Hz**,
0.019 % below 16 kHz. That assumes the output-clock divider divides the
processing clock, which the HAL header does not state. Nothing in Citrinet cares
about 0.019 %; it is recorded so nobody re-derives it later.

### Data type and scaling — where the `/256` really is, and what it does

The DMA is peripheral-to-memory, **WORD to WORD**, circular linked list
(`Patch/...audio.c:3766-3782, 3812-3816`), into

```c
static int32_t Audio_DigMicRecBuff[DEFAULT_AUDIO_IN_BUFFER_SIZE] __NON_CACHEABLE;   /* :271 */
```

32-byte aligned, `DEFAULT_AUDIO_IN_BUFFER_SIZE == CAPTURE_BUFFER_SIZE == 320`
samples = 20 ms (`stm32n6570_discovery_conf.h:56, 79`). The source is
`&MDF1_Filter0->DFLTDR` with `MsbOnly = DISABLE`
(`Patch/...audio.c:2400-2402`, `Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_mdf.c:1569-1570`),
so each buffer entry is the **whole 32-bit register**.

That matters, because the register is not right-aligned:

```
#define MDF_DFLTDR_DR_Pos                   (8U)
#define MDF_DFLTDR_DR_Msk                   (0xFFFFFFUL << MDF_DFLTDR_DR_Pos)       /* 0xFFFFFF00 */
```

`Drivers/CMSIS/Device/ST/STM32N6xx/Include/stm32n657xx.h:25307-25308`. The 24-bit
sample sits in bits [31:8].

The conversion to int16 is:

```c
tmp = Audio_DigMicRecBuff[index] / 256;         /* :3234 and :3259 */
tmp = SaturaLH(tmp, -32768, 32767);
Audio_In_Ctx[1].pBuff[2U * index]        = (uint8_t) tmp;
Audio_In_Ctx[1].pBuff[(2U * index) + 1U] = (uint8_t) ((uint32_t) tmp >> 8);
```

**Two corrections to `WORKLIST §5.8` here.**

*First, the line numbers.* `WORKLIST` cites `Patch/...audio.c:3172,3197`. Those
lines are inside `#if (USE_HAL_MDF_REGISTER_CALLBACKS == 1)` (`:3156`), and
`Projects/GS/Inc/stm32n6xx_hal_conf.h:191` sets that to **0**. The compiled
copies are `HAL_MDF_AcqCpltCallback` and `HAL_MDF_AcqHalfCpltCallback` in the
`#else` branch (`:3218`), and the live `/256` is at **`:3234` and `:3259`**.
Editing 3172/3197 would change nothing and the build would still work, which is
the worst kind of wrong.

*Second, and more important, what `/256` does.* `WORKLIST §5.8` says it
"discards 8 bits" of the sample and that "at −54 dBFS the signal occupies the
bottom ~9 of those 24 bits, so this shift is where the resolution dies." Given
`MDF_DFLTDR_DR_Pos = 8`, that is backwards. Dividing the register word by 256 is
an arithmetic shift that **extracts** the 24-bit sample; the eight bits thrown
away are `DFLTDR[7:0]`, which are not sample data. What actually loses
information is the `SaturaLH(-32768, 32767)` on the next line, and it loses the
**top** 8 bits by clipping, not the bottom.

The consequence is the opposite of what `WORKLIST` assumes: the int16 stream is
the MDF output at a fixed **+48 dB** relative to a straight 24→16-bit alignment.
A signal at −54 dBFS of MDF full scale arrives as ±16,747 counts — about 14 bits
of the int16 range, not 9. What this staging costs is **headroom**: anything
above −48 dBFS of MDF full scale clips hard at the saturate.

Both statements — `WORKLIST`'s and this one — are still *reasoning about
registers*, not measurements. Nobody has captured a sample from this microphone
on this board. The correct next move is not to change the shift but to print
`Audio_DigMicRecBuff[]` peaks for a known acoustic level. **Widening `/256` to
`/16` as `WORKLIST §5.8` proposes would remove 24 dB of headroom, not add 24 dB
of resolution, and should not be done before that measurement.**

### Where the gain is

Two knobs, both upstream of the int16 truncation, and one downstream safety net.

1. **`MDF_GAIN`**, currently **2** at 16 kHz (`Patch/...audio.c:165`), reaching
   `Audio_MdfFilterConfig.Gain` at `:3080`. The HAL field is
   `int32_t Gain`, "in step of around 3 dB (from −48 dB to 72 dB)", range −16..+24
   (`Drivers/STM32N6xx_HAL_Driver/Inc/stm32n6xx_hal_mdf.h:275-277`).
   `HAL_MDF_SetGain()` works **only** while `State == HAL_MDF_STATE_ACQUISITION`
   (`Src/stm32n6xx_hal_mdf.c:1800-1811`), so a slow AGC across utterances is free,
   and a pre-roll gain change is not.
2. **The `/256` + saturate** at `:3234`/`:3259`, discussed above. Treat as a
   headroom setting, not a gain.
3. **`citrinet_fe_peak_normalize()`** (`firmware/inc/citrinet_fe.h:146`), which is
   *after* the truncation and which `eval/results/gain.log` prices at 10.45 % WER
   rather than 5.83 %. It is a safety net, as `FRONTEND.md §5` already says.

---

## 4. What the application does today, and what Gate 5 can reuse

### The stock chain

```
MDF1 Filter0 --DMA(circular, 2x160 words)--> Audio_DigMicRecBuff[320] int32   Patch:271
  HAL_MDF_Acq{Half,}CpltCallback  /256 + saturate -> int16                    Patch:3224,3249
  BSP_AUDIO_IN_{Half,}TransferComplete_CallBack(1)                            audio_bm.c:982,995
  AudioCapture_half_buf_cb -> AudioCapture_ring_buff_feed(160 samples)        audio_bm.c:970-974
exec_bm: poll ring_buff.availableSamples >= AUDIO_ACQ_LEN                     audio_bm.c:587
  audio_process: memcpy overlap, ring_buff_consume, PreProc_DPU,              audio_bm.c:645-712
                 AiDPUProcess, PostProc_DPU, AudioPlayBack
```

`Record_Init()` is at `audio_bm.c:1200-1247` (pristine `:771-818`) and selects
`AUDIO_IN_DEVICE_DIGITAL_MIC` at `:1209`. Note `:1227` calls
`BSP_AUDIO_IN_SetDevice(1, AUDIO_IN_DEVICE_ANALOG_MIC)` — that is a self-test of
the API, not a device change; `SetDevice` ignores its argument
(`Patch/...audio.c:2430, 2443`) and `:1239` immediately asserts the device is
still `DIGITAL_MIC`.

**Reusable as is:** `Record_Init()`, `MDF_MspInit()`, the clock config, the DMA
node, the two acquisition callbacks, and the `AudioCapture_ring_buff` FIFO.
That is the entire acquisition front half, and none of it is model-specific.

**Must be replaced:** everything from `audio_process()` inward. `PreProc_DPU` is
float16 (`app_config.h:51`, `preproc_dpu.h:39-46`) and cannot express NeMo's log
guard or per-feature normalisation — `FRONTEND.md §2` measured that at 96.01 % of
the feature matrix moved. `citrinet_fe.c` replaces it outright.

**One BSP constraint to design around:** `BSP_AUDIO_IN_Record()` rejects
instance 1 requests where `NbrOfBytes / 2 > DEFAULT_AUDIO_IN_BUFFER_SIZE`
(`Patch/...audio.c:2172-2175`), i.e. the MDF half-buffer is capped at 320
samples until `DEFAULT_AUDIO_IN_BUFFER_SIZE` is raised
(`stm32n6570_discovery_conf.h:79`). 800 frames at hop 160 is 128,000 samples =
400 callback pairs. The streaming decomposition
`citrinet_fe_reset()`/`_column()`/`_finish()` exists for exactly this
(`FRONTEND.md §3`).

### The memory arithmetic of `WORKLIST §5.4` — verified, and worse than stated

`WORKLIST §5.4` says the stock buffer geometry at `COL = 800` needs 1,026,240 B
against a 1,047,552 B region and "the malloc fails at runtime rather than at
link time". The region and the ring buffer are right. The rest is not.

Measured, not estimated: `sizeof()` compiled for the real target with the real
flags, changing only `CTRL_X_CUBE_AI_SPECTROGRAM_{NMEL,COL}` in a scratch copy
of `Projects/Dpu/`, `arm-none-eabi-gcc 14.3.1 -mcpu=cortex-m55 -mfpu=fpv5-d16`:

| | stock (`COL` 96, `NMEL` 64) | Gate 5 (`COL` 800, `NMEL` 80) |
|---|---:|---:|
| `PATCH_LENGTH` | 15,600 | 128,240 |
| `sizeof(AudioProcCtx_f16_t)` | 102,752 | **826,464** |
| of which `pCplxSpectrum` | 98,688 | **822,400** |
| `sizeof(AudioBM_proc_t)` | **268,048** | **2,166,032** |
| `sizeof(AudioBM_acq_t)` | 672 | 672 |
| ring buffer `malloc` | 62,720 | 513,280 |

Every row but the last is a `sizeof` read out of the object file. The ring
buffer is a runtime `malloc(nbSamples * 2 * nbFrames)` with
`nbSamples = ((PATCH_LENGTH / 320) + 1) * 320` and `nbFrames = 2`
(`audio_bm.c:721-725`, `AudioCapture_ring_buff.c:81-90`), so that row is
arithmetic.

The stock column reproduces the 268,048 B that
`Projects/GS/Inc/audio_bm.h:51-56` already records, which is the check that the
harness is measuring the right thing.

`WORKLIST §5.4`'s table counts `proc_buff`, `audio_out` and the ring buffer and
stops. It omits `AudioProcCtx_f16_t.pCplxSpectrum[(NFFT/2 + 1) * 2 * COL]`
(`Projects/Dpu/audio_proc.h:64`), which is 822,400 B at `COL = 800` — **and
`AudioBM_proc_t` holds two of them**, because `AudioPostProcCtx_t` is `#define`d
to the same type (`Projects/Dpu/postproc_dpu.h:37`, `audio_bm.h:61-62`).

So the real figure is `2,166,032 + 513,280 = 2,679,312 B` against
`RAM (xrw) : ORIGIN = 0x34000400, LENGTH = 1023K` = 1,047,552 B
(`Projects/GS/STM32CubeIDE/STM32N657XX_LRUN.ld:47`). **`AudioBM_proc_t` alone
overruns the region by 2.07×**, so the failure is at *link* time, before
`AudioCapture_ring_buff_alloc()` is ever reached. `WORKLIST §5.4`'s conclusion
holds and is understated; its stated mechanism is wrong.

None of this is on the critical path for the option recommended below, because
that option instantiates none of these structures — the same trick
`GATE4_CANNED` already plays at `audio_bm.h:51-56`.

### RAM actually available today

`arm-none-eabi-size` on the `GATE4_CORPUS` build currently in
`vendor/.../Projects/GS/BuildGCC/BM/GS_Audio_N6.elf` (identified by its
`# gate4 corpus @ %p ...` string):

```
   text     data      bss      dec
 522156    38944    88292   649392
```

`bss` includes `._user_heap_stack` = 86,020 B (heap `0x10000`, stack `0x5000`,
`STM32N657XX_LRUN.ld:41-42`). 649,392 / 1,047,552 = **61.99 %**, leaving
**398,160 B** free. `citrinet_fe.c`'s float32 pass-1 scratch is 256,000 B and
`citrinet_fe_t` is about 5.9 KB (`FRONTEND.md §5`, `citrinet_fe.h:106-130`), so
the front end fits with ~136 KB to spare — but **not** with a second 256,000 B
copy of the waveform in RAM. See §6.

---

## 5. The options

Effort is one engineer, and assumes the board is available.

| # | option | hardware needed | code that exists | code to write | what could go wrong | effort |
|---|---|---|---|---|---|---|
| **A** | on-board MP23DB01HP via MDF1 | **none** | the whole acquisition chain (§4) | replace `audio_process()`; restructure buffers (§4); solve gain staging | gain staging is unsolved and unmeasured, and the `/256` reasoning in `WORKLIST §5.8` is wrong (§3). Silent failure mode: a clean-looking run at 35 % WER | **2–3 d** |
| B | headset electret mic on the 3.5 mm jack, WM8904 IN1L, SAI1_B, BSP instance 0 | a CTIA headset with a mic | `BSP_AUDIO_IN_Init/Record(0,…)`, WM8904 driver | `Record_Init()` for instance 0; SAI DMA callbacks; same downstream work as A | +14 dB analog and +17 dB digital are baked into the codec init (`wm8904.c:290-297`); level is as unmeasured as A's, and now depends on which headset. Adds a variable rather than removing one | 1.5–2 d |
| C | line-in through WM8904 IN2L/IN2R (`WM8904_IN_LINE2`) | a source, a cable, probably an attenuator, **and a schematic check** | codec driver code path only (`wm8904.c:268-286`) | a new BSP device; everything in B | **no BSP path selects LINE2 and nothing in the tree says IN2L/IN2R reach a connector.** Could be unbuildable on this board | 2 d + unknown |
| **D** | **canned waveform in external flash, replayed through the real front end** | **none** | `gate4_corpus()` proves memory-mapped replay end to end (`firmware/vendor-mods/gate4.patch`); `firmware/tools/gen_corpus.py` proves the blob contract; `citrinet_fe_run()` is bit-exact on the host | a `gen_wave_corpus.py`; a `gate5_wave()` in `audio_bm.c`; DWT brackets | measuring XSPI latency instead of front-end cost (§6) | **0.5–1 d** |
| E | USB audio class | a second USB cable | **nothing** — HAL `pcd`/`ll_usb` only, no device library, no class, no descriptors | a USB device stack | this is a project, not a task | 1–2 w |
| F | waveform over the ST-LINK VCP | none | `UART_Config()` (TX/RX enabled, `misc_toolbox.c:215`) | a receive path — none exists (§2) | 14400 baud 8N1 = 1,440 B/s; one 128,000-sample utterance is 256,000 B = **178 s**, and `app_config.h:63` records that 921600 breaks `UART_Config` at this clock | 1 d, then unusable |

---

## 6. Recommendation — option D, then A

**Do D first. It is half a day and it unblocks the thing Gate 5 is actually
blocked on.**

Flash a small corpus of 16-bit PCM utterances to external flash next to the
feature corpus, and run `citrinet_fe_run()` on the M55 over them.

Concretely, and reusing what Gate 4 already proved:

- The blob mechanism is done. `gate4_corpus()` reads a header and N tensors
  straight out of memory-mapped XSPI at `0x71000000` and needs no RAM copy
  (`firmware/vendor-mods/gate4.patch`, the `GATE4_CORPUS` block). The flash is
  128 MB (`Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.h:52`); weights sit at
  `0x70400000` (`board/BUILD.md:315-317`) and the 64-utterance feature corpus
  occupies `0x71000000` + 4,096,064 B. A waveform blob at `0x72000000` collides
  with nothing.
- The size is trivial: 128,000 samples × 2 B = 256,000 B per utterance, so 16
  utterances is 4.1 MB.
- The entry point exists: `citrinet_fe_run(fe, pcm, n, scale, zp, out)` writes
  straight into the NPU's `p_stai_inputs[0]`, and the corpus loop then reuses
  `AiDPUProcess()` and the argmax printing verbatim.
- Bracket it with `port_dwt_reset()` / `port_dwt_get_cycles()` exactly as
  `gate4_corpus()` already brackets the inference.

**What D de-risks.** The M55 cost of the front end — the single largest
unmeasured number in the project, and the reason `WORKLIST §5` calls 5.5, 5.6 and
5.10 RISKY. It also proves, on silicon: that `arm_rfft_fast_f32` is the backend
actually linked (`citrinet_fe_backend()`), that the 3,840 B of tables land in
`.rodata` as measured on the host, that the guard telemetry and the
`CITRINET_FE_E_GUARD` refusal behave, that cache maintenance between the front
end and the NPU is right, and that waveform → features → NPU → CTC → text runs
end to end in one image. It closes the loop `FRONTEND.md §7` leaves open:
"proven on a workstation, not on silicon".

**What D does not de-risk. Say this out loud.** It touches neither the
microphone, nor MDF1, nor `MDF_GAIN`, nor the `/256`, nor the ring buffer, nor
real-time streaming, nor the fact that nobody has yet seen a single sample come
off this board's microphone. It converts a canned-*feature* test into a
canned-*waveform* test. That is exactly one link further down the chain, and the
gain-staging landmine of `docs/FEASIBILITY.md §2(d)` is still entirely ahead.

**Then do A.** The on-board microphone is the right answer for the product: it
needs no extra hardware, ST's application already selects it, and the acquisition
half of the chain is written and shipping. B and C add a cable, a level unknown
and — for C — a schematic question, in exchange for nothing D does not give
sooner. E and F are not viable.

### The single most important thing that could go wrong with D

**The measurement measures the flash, not the front end.**

`citrinet_fe_run()` walks the waveform once for the peak (`citrinet_fe.c:372-376`)
and then 800 more times in overlapping 400-sample windows through
`citrinet_fe_build_frame()`, which loads both `pcm[i]` and `pcm[i-1]` per position
(`citrinet_fe.c:224-231`) — up to 640,000 `int16` loads, 1.28 MB of traffic over a
256 KB buffer. Pointed straight at `0x72000000`, every one of those is a
memory-mapped XSPI read, and `Ext_Mem_Config()` **disables XSPI2 automatic
prefetch** by default
("Hotfix for xspi: no prefetch", `firmware/vendor-mods/gate4.patch` at the
`Ext_Mem_Config` hunk; `GATE4_XSPI_PREFETCH` flips it). Gate 4 Round 13 already
saw per-transaction cost dominate on this bus. The number that comes back would
be a plausible-looking millisecond figure that is mostly flash latency, and it
would be believed, because there is nothing to compare it against.

The obvious fix does not fit: 649,392 B (current build) + 261,900 B (f32 scratch
and context) + 256,000 B (a RAM copy of the waveform) = 1,167,292 B against
1,047,552 B — **over by 119,740 B**. Dropping to the fp16 scratch gets it to
1,039,292 B, which fits by 8,260 B, but costs the 3,982-of-768,000 int8
divergence `FRONTEND.md §5` measured and puts the image back at 99.2 % of a
region that Gate 4 Round 18 spent a day suspecting.

So stage the waveform in **blocks**: a 2 s / 64,000 B RAM window refilled four
times, driving `citrinet_fe_build_frame()` + `citrinet_fe_column()` — the
streaming decomposition that already exists for `WORKLIST §5.7`. Then report
three numbers, not one: cycles with the input in RAM, cycles with the input in
flash, and the delta. If they differ, the RAM number is the front end's cost and
the delta is the bus's.

---

## 7. Corrections this document makes

Recorded here rather than by editing the originals, per house rule.

1. **The DK's microphone is an MP23DB01HP, not an IMP34DT05.**
   `Drivers/BSP/STM32N6570-DK/stm32n6570_discovery_audio.h:194` is the proof.
   Affects `README.md:190`, `firmware/FRONTEND.md:245`,
   `docs/FEASIBILITY.md:129`, `docs/GATES-5-6.md:232`, `firmware/WORKLIST.md:129`.
   `firmware/WORKLIST.md:129` additionally cites `ai_model_config.h:38` as the
   authority; that symbol is unused model metadata whose default value is an IMU.
2. **The sensitivity in `docs/FEASIBILITY.md:129` is for the wrong part.**
   MP23DB01HP is −24 dBFS typ at 94 dB SPL, not −26. The −54 dBFS working
   assumption and every WER number derived from it are unaffected.
3. **The live `/256` is at `Patch/stm32n6570_discovery_audio.c:3234` and `:3259`,
   not `:3172,3197`.** `USE_HAL_MDF_REGISTER_CALLBACKS` is 0
   (`Projects/GS/Inc/stm32n6xx_hal_conf.h:191`), so the lines `WORKLIST §5.8` and
   `docs/GATES-5-6.md:225-226` name are compiled out.
4. **`/256` extracts the sample, it does not discard sample bits.**
   `MDF_DFLTDR_DR_Pos = 8` (`stm32n657xx.h:25307`). The information loss is in the
   following `SaturaLH`, and it costs headroom above −48 dBFS, not resolution
   below it. `WORKLIST §5.8`'s proposal to widen the shift to `/16` would remove
   24 dB of headroom. Do not do it before the level is measured on silicon.
5. **`WORKLIST §5.4`'s subtotal is 1,026,240 B; the measured figure is
   2,679,312 B**, because the table omits two `pCplxSpectrum[(NFFT/2+1)*2*COL]`
   arrays (`Projects/Dpu/audio_proc.h:64`, one in the pre- and one in the
   post-processing context via `postproc_dpu.h:37`) at 822,400 B each. The
   failure is at link time, not in `malloc`. The conclusion is unchanged.

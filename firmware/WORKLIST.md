# Firmware work list — gates 3 to 7

File-level plan for putting Citrinet-256 on the STM32N6570-DK, derived by reading the
two cloned ST working trees rather than GitHub. Every path below is real and was opened.

- Base app: `vendor/STM32N6-GettingStarted-Audio/` (v2.2.0, 12-Jan-2025)
- LCD donor: `vendor/STM32N6-GettingStarted-ObjectDetection/`
- Context and gate definitions: `README.md`, `docs/FEASIBILITY.md` — not restated here.

**No board access in this pass.** Nothing below has been run on silicon.
Effort is developer-time for one person who has read this document.

Legend: **RISKY** = can fail in a way that invalidates a design assumption ·
*mechanical* = tedious but the outcome is known.

---

## 0. What changed since `docs/FEASIBILITY.md` was written

Read this first; three established numbers moved.

| | FEASIBILITY says | Actually |
|---|---|---|
| Gate 2 | "recompile against ST's mpool — 30 min, not done" | **Done.** `compile/reports/g800_st/` |
| 8 s epochs / SW | 628 / 0 | **618 / 0** under ST's option string |
| 8 s activations | 625 KB (42.5 % of pool) | **925 KB (62.8 %)** — 500 KB cpuRAM2 + 425 KB npuRAM6 |
| 8 s scheduler latency | 91.2 ms | **91.893 ms** @ 1 GHz |
| "Whether the audio app enables npuRAM3/4/5" — open | unknown | **It does**, `audio_bm.c:741-759` |
| "~40× sparse-filterbank win is available" | available | **already implemented** by ST (`MelFilterbank()` is a run-length dot product) |
| LCD line width "47 characters, already derived by ST" | 47 | 47 is `N_PRINTABLE_CHARS`, sized for **Font24** (17 px); the OD app renders **Font20** (14 px) → **57** fit |

Gate 2 evidence (`compile/reports/g800_st/summary.txt` lines 40-45,
`compile/reports/g800_st/cycles.json`):

```
Total number of epochs                               618
>> pure software (SW) epochs                           0
>> hybrid epochs                                       0
octoFlash  [0x70180000 - 0x74080000]:  9.728 MB   used 0x70180000-0x70B3A700
cpuRAM2    500.000 kB / 1.000 MB      npuRAM6  425.000 kB / 448.000 kB
```

The flash base reads `0x70180000` — Gate 2's pass criterion. `compile/reports/g800_st/user_neural_art.used.json`
records ST's exact option string was used (`--Ocache-opt -O3 --Os --native-float --Omax-ca-pipe 4
--cache-maintenance --csv-file network --all-buffers-info`, no `--enable-virtual-mem-pools`,
no `--Oauto-sched`). **Gate 2 is closed.** Full write-up and option ablation in
`compile/GATE2.md`; pool arithmetic in `docs/MEMORY-MAP.md`.

`Int_Mem_Config()` (`Projects/GS/Src/audio_bm.c:741-759`, non-`APP_LP` branch) sets
`RCC_MEMENR_AXISRAM3EN|4EN|5EN|6EN` and calls `HAL_RAMCFG_EnableAXISRAM` on SRAM2..SRAM6.
So AXISRAM3/4/5 (448 KB each at 0x34200000 / 0x34270000 / 0x342E0000, by arithmetic from
npuRAM6 @ 0x34350000) are clocked and out of shutdown at runtime, but `stm32n6.mpool`
does not declare them as pools, so the compiler cannot place activations there. Adding
three pool entries would widen 1,507,328 B → ~2,884,608 B. **Not needed at 8 s. Do not do
this speculatively** — it invalidates the 618/0 evidence.

---

## Map of the audio application

Answers "what is the structure" once, so the gates below can just name files.

```
vendor/STM32N6-GettingStarted-Audio/
  Projects/GS/                          the application
    Makefile                            all four configs; C_SOURCES lists every .c
    bm.mk  bm_lp.mk  freertos.mk  freertos_lp.mk
    STM32CubeIDE/STM32N657XX_LRUN.ld    linker: one region, RAM @0x34000400 len 1023K
    Src/main.c                          init_bm(); exec_bm()      (39 lines)
    Src/audio_bm.c                      THE bare-metal app: init, main loop, callbacks
    Src/AudioCapture_ring_buff.c        malloc'd ring FIFO, IRQ-fed / loop-drained
    Src/test.c stm32n6xx_it.c syscalls.c sysmem.c
    Inc/app_config.h                    LOG_LEVEL, PREPROC_FLOAT_16, USE_UART_BAUDRATE 14400
    Inc/stm32n6570_discovery_conf.h     AUDIO_FREQUENCY_16K, CAPTURE_BUFFER_SIZE=320
    Inc/stm32n6xx_hal_conf.h            HAL module enables
    freertos/                           the RTOS variant (audio_acq_task.c, audio_proc_task.c)
  Projects/Dpu/                         the "data processing units"
    ai_model_config.h                   per-model constants; .aed / .se variants alongside
    dpu_config.h                        defaults + CTRL_* enum
    preproc_dpu.c/.h                    log-mel or STFT front end + f16/f32 selector macros
    ai_dpu.c/.h                         STAI invoke wrapper
    postproc_dpu.c/.h                   iSTFT (speech-enhancement only; compiles to nothing
                                        when CTRL_X_CUBE_AI_POSTPROC == CTRL_AI_BYPASS)
    user_mel_tables.c/.h                generated LUTs; .aed / .se variants alongside
    audio_proc.h                        AudioProcCtx_f16_t — all scratch buffers live here
  Projects/Common/
    misc_toolbox.c/.h                   NPU_Config(), RISAF_Config(), UART_Config(),
                                        fuse_vddio(), system_init_post()
    Patch/stm32n6570_discovery_audio.c  *** the BSP audio driver actually compiled ***
    Patch/stm32n6570_discovery_audio.patch.hsi_600_400.c    clock config for bm/freertos
    Patch/stm32n6570_discovery_audio.patch.msi_4_4.c        clock config for the _lp configs
    cpu_stats.c (DWT timing), logging.c, lc_print.c, system_clock_config.c, pm_dvfs.c
  Projects/X-CUBE-AI/models/            the model drop point (see Gate 4)
  Middlewares/ST/STM32_AI_AudioPreprocessing_Library/       the log-mel library (see Gate 5)
  Middlewares/ST/AI/                    stai + ll_aton runtime + NetworkRuntime1200_CM55_GCC.a
  Drivers/BSP/STM32N6570-DK/            discovery.c/.h, _audio, _bus, _xspi.  NO _lcd.
  FSBL/ai_fsbl.hex                      first-stage boot loader, 62,752 B @ 0x70000000
```

There is **one** board — MB1939 STM32N6570-DK. `Binary/` has only `STM32N6570-DK/`.
The four "configs" are software variants, not boards:

| config | make target | build dir | clock patch | defines |
|---|---|---|---|---|
| bare metal | `make bm` | `BuildGCC/BM` | `...patch.hsi_600_400.c` | `LL_ATON_OSAL_BARE_METAL`, `APP_BARE_METAL` |
| bare metal low power | `make bm_lp` | `BuildGCC/BM_LP` | `...patch.msi_4_4.c` | + `APP_LP`, `APP_DVFS` |
| FreeRTOS | `make freertos` | `BuildGCC/FREERTOS` | `...patch.hsi_600_400.c` | `LL_ATON_OSAL_FREERTOS` |
| FreeRTOS low power | `make freertos_lp` | `BuildGCC/FREERTOS_LP` | `...patch.msi_4_4.c` | + `APP_LP` |

**Build command** (from `Projects/GS/`, `arm-none-eabi-gcc` on PATH):

```bash
make bm -j8            # -> BuildGCC/BM/GS_Audio_N6.{elf,bin}
```

`make all` builds all four. `make flash_bm` signs and flashes to 0x70100000 (board only).

**Take `bm`.** Rationale: `CPU_STATS` (DWT per-stage timing, `audio_bm.c:444-457`) is only
compiled `#ifdef APP_BARE_METAL`, and the utterance-based design has no concurrency to
schedule. `_lp` variants also swap in the MSI 4/4 clock patch and gate NPU SRAM on and off
around each inference — both irrelevant and both extra failure surface.

Local toolchain is `arm-none-eabi-gcc 10.3.1`; ST's own log banner says the reference
binaries were built with GCC 13.3.1. Not necessarily a problem, but if anything smells,
that is the first difference to eliminate.

### The capture path

- **Microphone**: on-board MEMS PDM mic **IMP34DT05** (`ai_model_config.h:38`), read through
  **MDF1** (`AUDIO_IN_DEVICE_DIGITAL_MIC`, BSP instance **1**). `AUDIO_IN_Init` is at
  `audio_bm.c:771-818` (`Record_Init`).
- **Rate**: `AUDIO_FREQUENCY_16K` (`stm32n6570_discovery_conf.h:55`), 16-bit, 1 channel.
  MDF decimation 32, SINC4, reshape ÷4, HPF on at 0.000625·Fpcm (≈10 Hz), `Offset = 0`
  (`Patch/stm32n6570_discovery_audio.c:3053-3094`).
- **Buffers**:
  - MDF DMA lands **32-bit** samples in `Audio_DigMicRecBuff[DEFAULT_AUDIO_IN_BUFFER_SIZE]`,
    `__NON_CACHEABLE`, 32-byte aligned (`Patch/...audio.c:271`). `DEFAULT_AUDIO_IN_BUFFER_SIZE
    == CAPTURE_BUFFER_SIZE == 320` samples (20 ms).
  - `MDF_AcqHalfCpltCallback` / `MDF_AcqCpltCallback` (`Patch/...audio.c:3187-3205` and
    `:3160-3180`) convert to int16 **and this is the truncation point**:

    ```c
    tmp = Audio_DigMicRecBuff[index] / 256;     /* :3172 and :3197 */
    tmp = SaturaLH(tmp, -32768, 32767);
    ```

  - They then call `BSP_AUDIO_IN_{Half,}TransferComplete_CallBack(1)` →
    `AudioCapture_half_buf_cb()` (`audio_bm.c:541-545`) → `AudioCapture_ring_buff_feed()`.
  - `exec_bm()` polls `ring_buff.availableSamples >= AUDIO_ACQ_LEN` and calls
    `audio_process()` (`audio_bm.c:159-165`).
- **Gain knob**: `MDF_GAIN(AUDIO_FREQUENCY_16K)` = **2** (`Patch/...audio.c:164`). The HAL
  field is `int32_t Gain`, range **−16..+24 in ~3 dB steps (−48 dB..+72 dB)**
  (`Drivers/STM32N6xx_HAL_Driver/Inc/stm32n6xx_hal_mdf.h:275-277`), and
  `HAL_MDF_SetGain()` (`stm32n6xx_hal_mdf.c:1800`) changes it **during acquisition**.
  This is the lever `docs/FEASIBILITY.md` §2(d) says must exist. It does, in two places
  (MDF gain register, and the `/256` shift), both **before** int16 truncation.

### The STAI invoke wrapper and cache maintenance

- `Projects/Dpu/ai_dpu.c`
  - `AiDPULoadModel()` :117 — `stai_runtime_init()`, `stai_network_init()`,
    `stai_network_get_info()`, `AiDPUCheckModel()`, `stai_network_get_inputs/outputs()`,
    reads `info.inputs[0].scale.data[0]` → `input_Q_inv_scale` and
    `zeropoint.data[0]` → `input_Q_offset`.
  - `AiDPUProcess()` :220 — `port_dwt_reset()`, `stai_network_run(p_network, STAI_MODE_SYNC)`,
    `time_stats_store(TIME_STAT_AI_PROC, ...)`.
  - `AiDPUCheckModel()` :45 — **see Gate 4, it rejects our model.**
- Network context: `STAI_NETWORK_CONTEXT_DECLARE(network, STAI_NETWORK_CONTEXT_SIZE)` at
  `ai_dpu.c:36`.
- **Cache maintenance is in the pre-processor, not the AI wrapper**:
  `Projects/Dpu/preproc_dpu.c:144` —
  `mcu_cache_clean_invalidate_range((uint32_t)p_spectro, (uint32_t)(p_spectro + NMEL*COL))`,
  called after the feature tensor is written and before `AiDPUProcess()`.
  API in `Middlewares/ST/AI/Npu/Devices/STM32N6xx/mcu_cache.h`
  (`mcu_cache_clean_range`, `_invalidate_range`, `_clean_invalidate_range`).
  There is **no** post-inference invalidate of the output buffer anywhere in the app,
  because the stock AED output is float32 read straight from a coherent buffer.
  **Gate 4 must add one** (see below).

---

## Gate 3 — build and flash the unmodified ST audio app

Goal: prove the toolchain, the boot chain and the board, with ST's own model in the loop.

| # | Item | Files | Effort | Marker |
|---|---|---|---|---|
| 3.1 | Read OTP fuse state **before** anything else | `STM32_Programmer_CLI` + `OTP_FUSES_STM32N6xx.stldr` | 0.5 h | **RISKY — irreversible** |
| 3.2 | Decide on `fuse_vddio()` | `Projects/GS/Src/audio_bm.c:106-109` | 0.5 h | **RISKY** |
| 3.3 | `make bm -j8` | `Projects/GS/` | 0.5 h | mechanical |
| 3.4 | Sign + flash FSBL, app, AED weights | see below | 1 h | mechanical |
| 3.5 | Confirm the AED JSON banner on UART @ 14400 8N1 | — | 0.5 h | mechanical |

**3.2 is the one to read twice.** `init_bm()` calls `fuse_vddio()` unconditionally:

```c
/* Force fusing of the OTP when using a Nucleo/DK board only */
#if (defined(USE_STM32N6xx_NUCLEO) || defined(USE_STM32N6570_DK))
  fuse_vddio();
#endif
```

`USE_STM32N6570_DK` is not in the Makefile's `C_DEFS`, so on the Makefile path this compiles
out — but the CubeIDE `.cproject` may define it. `Projects/GS/STM32CubeIDE/.cproject` must be
grepped before any flash, and `fuse_vddio()` (`Projects/Common/misc_toolbox.c:326`, under
`#ifdef HAL_BSEC_MODULE_ENABLED`) read line by line. The package README states plainly:
*"when executing the project on the board, these two OTP fuses are set if not already"* and
*"when OTP fuses are set, they can not be reset."* VDDIO2_HSLV/VDDIO3_HSLV must be **1** for
the octoFlash to run at speed, so they almost certainly need blowing eventually — but blow
them deliberately, having read the state first, not as a side effect of `make`.

**3.4 flash layout**, verified by parsing the shipped Intel-hex records:

| region | address | size | source |
|---|---:|---:|---|
| FSBL | `0x70000000` | 62,752 B | `FSBL/ai_fsbl.hex` |
| app (SSBL) | `0x70100000` | ~244,544 B stock | `BuildGCC/BM/GS_Audio_N6_sign.bin` |
| weights | `0x70180000` | 3,282,785 B (AED) / **9,728 KB (ours)** | `network_data.bin` |

```bash
STM32_SigningTool_CLI -s -bin BuildGCC/BM/GS_Audio_N6.bin -nk -t ssbl -hv 2.3 \
                      -o BuildGCC/BM/GS_Audio_N6_sign.bin
export DKEL="$(dirname $(which STM32_Programmer_CLI))/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr"
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el $DKEL -hardRst -w FSBL/ai_fsbl.hex
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el $DKEL -hardRst \
                     -w BuildGCC/BM/GS_Audio_N6_sign.bin 0x70100000
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el $DKEL -hardRst \
                     -w Projects/X-CUBE-AI/models/aed_weights.hex
```

Signing is always `-nk -t ssbl -hv 2.3` — unsigned, SSBL type, header v2.3. No keys.
BOOT1 right → flash, BOOT1 left → run, power-cycle between.

> **Hard constraint, permanent.** The app slot is `0x70100000..0x70180000` = **512 KiB**.
> The signed binary must stay under it or it overwrites the weight blob and the network
> reads garbage while every log looks clean. Stock is 238.8 KiB, so there is ~273 KiB of
> headroom for everything gates 4-7 add (vocab 8 KB, fonts 16 KB, LCD stack ~40 KB of
> code). Add `arm-none-eabi-size` against 0x80000 to the build script and fail loudly.

*Stop if* the stock app does not print. Debug with `Binary/STM32N6570-DK/STM32N6_GettingStarted_Audio_aed_bm.hex`
via `Binary/flash-bin.sh aed bm`, which needs no toolchain at all.

---

## Gate 4 — swap in Citrinet, feed a canned feature vector, ignore the mic

Goal: the first point where silicon can contradict desk research.

### 4.0 The model drop-in procedure, as the scripts actually implement it

`Projects/X-CUBE-AI/models/` is the drop point. Four scripts, all thin:

**`generate-n6-model.sh $1`** — runs the compiler and copies eight artefacts up one level:

```bash
stedgeai generate -m $1 --target stm32n6 --st-neural-art default@user_neural_art.json
cp ./st_ai_output/{network.c,network.h,stai_network.c,stai_network.h} .
cp ./st_ai_output/{network_c_info.json,network_generate_report.txt} .
cp ./st_ai_output/network_atonbuf.xSPI2.raw network_data.bin
arm-none-eabi-objcopy -I binary network_data.bin --change-addresses 0x70180000 -O ihex network_data.hex
```

So: **`network.c` + `stai_network.c` are compiled into the app** (they are in the Makefile
at lines 45-47), and **`network_atonbuf.xSPI2.raw` is renamed to `network_data.bin` and
objcopy'd to a hex based at 0x70180000**. That `--change-addresses 0x70180000` is the only
thing tying the blob to the mpool's `octoFlash` offset — which is why Gate 2's flash-base
check mattered. `stmaic_STM32N6570-DK.conf` also lists `network_ecblobs.h` as a template
output; no such file exists in the package and the g800_st compile reports *"0 epoch
controller blobs"*, so it is not produced and not needed.

**`generate-n6-model-headers.sh {aed|se}`** — `python GenHeader/GenHeaders.py --config-name
user_config_{aed,se}.yaml`, then `cp -r ./GenHeaderOutput/C_Header/*.* ../../Dpu`.
Emits `ai_model_config.h`, `user_mel_tables.c`, `user_mel_tables.h`.
Needs `hydra-core omegaconf numpy munch seaborn librosa` (`GenHeader/requirements.txt`).

**`build-firmware.sh` / `sign-and-flash-model.sh`** — Windows CubeIDE headless build and
`STM32_Programmer_CLI.exe`; both have literal `<PathtoCube IDE>` placeholders and neither
runs as shipped. `sign-and-flash-model.sh` is worth reading anyway because it is the
authoritative statement of the three writes:

```bash
$sign -s -bin $bin -nk -t ssbl -hv 2.3 -o aed_bm.bin
$prog ... -w ../../../FSBL/ai_fsbl.hex          # FSBL,   0x70000000 (from the hex)
$prog ... -w aed_bm.bin        0x70100000       # SSBL/app
$prog ... -w network_data.bin  0x70180000       # weights
```

Note the mismatch: this script writes `network_data.bin` while
`stmaic_STM32N6570-DK.conf` writes `network_atonbuf.xSPI2.bin`. Same bytes, two names for
the same file; use `network_data.bin`, which is what `generate-n6-model.sh` produces.

**Do not run `deploy-model.sh`.** It chains all four, and the last one flashes.

| # | Item | Files | Effort | Marker |
|---|---|---|---|---|
| 4.1 | Write `firmware/scripts/gen_model.sh`: Linux port of `generate-n6-model.sh`, hard-wired to `artifacts/onnx/q800_real.onnx` + copies of ST's `stm32n6.mpool` and `user_neural_art.json`. **Read `compile/GATE2.md` §"option ablation" first** — it measures that adding `--Oauto-sched` to ST's string saves 300 KB of cpuRAM2 and 0.68 ms at 8 s (625 kB / 628 epochs / 91,212,624 cycles vs 925 kB / 618 / 91,893,224), and is what keeps 12 s off hyperRAM | new file | 1 h | mechanical |
| 4.2 | Drop `network.c`, `network.h`, `stai_network.c`, `stai_network.h`, `network_data.bin` into the working tree | `firmware/models/` | 0.5 h | mechanical |
| 4.3 | **Relax `AiDPUCheckModel()`** — see below | `Projects/Dpu/ai_dpu.c:99-107` | 0.5 h | mechanical |
| 4.4 | Write `firmware/inc/canned_features.h`: one host-computed int8 `[80,800]` tensor, 64,000 B `const` | new, generated | 1 h | mechanical |
| 4.5 | Bypass mic+preproc: `memcpy` the canned tensor into `p_stai_inputs[0]`, keep the `mcu_cache_clean_invalidate_range()` call | new `audio_bm.c` variant | 1 h | mechanical |
| 4.6 | **Add a post-inference cache invalidate** on the 102,500 B output — the stock app has none | new | 0.5 h | **RISKY** |
| 4.7 | Argmax over `[100,1025]` int8, print 100 token ids over UART | new | 1 h | mechanical |
| 4.8 | Compare against host ORT argmax; DWT-time the invoke | host script + board | 2 h | **RISKY** |

**4.3 is a hard blocker, not a nicety.** `ai_dpu.c:99-107`:

```c
for (int i=0; i< pxInfo->n_outputs ; i++ ) {
  if (STAI_FORMAT_FLOAT32 != pxInfo->outputs[i].format &&
      2 != pxInfo->outputs[i].shape.size) {
    LogError("AI_DPU: Output format not supported\n\r");
    res = DPU_ERROR;
  }
}
```

`stai_shape` is `stai_array_s32` (`Middlewares/ST/AI/Inc/stai.h:300-303, 320`), so
`.size` is the **rank**, and `.data[]` the dims. Our output is `STAI_FORMAT_S8`, rank 3,
`{100, 1025, 1}` (`compile/reports/g800_real/io_contract.h:76-88`). Both conjuncts are
true → `DPU_ERROR` → `AiDPULoadModel()` calls `Error_Handler()` at `ai_dpu.c:148-152`,
which is `__disable_irq(); while(1);`. **The app hangs silently at startup with no output.**
This survives ST's own AED model only because that output is float32.

The input check (`ai_dpu.c:88-98`) is fine — it short-circuits on `STAI_FORMAT_S8`.

Also note `AiDPULoadModel()` indexes `shape.data[AI_DPU_WIDTH=1]` / `[AI_DPU_HEIGHT=2]`
into `in_width`/`in_height`/`out_height`; on our rank-3 tensors that yields
`in_width=800, in_height=1, out_height=1025`, which is meaningless but harmless
(nothing downstream reads them once `printInferenceResults` is replaced).

**4.6.** `preproc_dpu.c:144` cleans the *input* before the run. Nothing invalidates the
*output* after it. The NPU writes 102,500 B via its own path; the M55 D-cache may hold
stale lines. Add, immediately after `stai_network_run()` returns:

```c
mcu_cache_invalidate_range((uint32_t)p_out, (uint32_t)p_out + STAI_NETWORK_OUT_1_SIZE_BYTES);
```

If argmax comes out plausible-but-wrong at Gate 4, this is the first suspect.

**Pass:** on-device token ids match host ORT argmax on the same canned tensor, and wall
time is within ~2× of 91.9 ms. Honest band from `docs/FEASIBILITY.md` §5: **100-250 ms**.

**Deployment contract to hard-check at runtime** (`compile/reports/g800_real/io_contract.h`):

| | format | flags | rank | shape | bytes | scale | offset |
|---|---|---|---|---|---:|---|---:|
| in `Input_0_out_0` | S8 | PREALLOCATED\|CHANNEL_FIRST | 3 | {80, 800, 1} | 64,000 | 0.120522417128086 | 0 |
| out `Transpose_1488_out_0` | S8 | PREALLOCATED\|OVERRIDE\|CHANNEL_FIRST | 3 | {100, 1025, 1} | 102,500 | 0.265415638685226 | 0 |

Both alignments 32. Read the scale from `info.inputs[0].scale.data[0]` at runtime, as
`ai_dpu.c:170` already does; do not hardcode it.

---

## Gate 5 — log-mel front end on the M55

This is the largest and riskiest gate. 2-3 days, and the ordering below is deliberate:
the level self-test goes in with the *first* commit, not at the end.

### 5.0 What ST's library is and what it cannot do

`vendor/STM32N6-GettingStarted-Audio/Middlewares/ST/STM32_AI_AudioPreprocessing_Library/`

```
Inc/ Src/  feature_extraction{,_f16}.c   spectrogram / mel / log-mel / MFCC columns
           mel_filterbank{,_f16}.c       MelFilterbank_Init(), MelFilterbank()
           window{,_f16}.c               Window_Init() — Hann/Hamming/Blackman
           audio_din{,_f16}.c            int16 -> float with padding
           audio_prePost_process{,_f16}.c  batch STFT/iSTFT helpers
           common_tables{,_f16}.c        pre-baked hann/mel LUTs for stock sizes
Examples/  melspectrogram_example.c  mfcc_example.c
Python/    LogMelSpectrogram.py  MFCC.py
```

The entry point the app uses is
`LogMelSpectrogramColumn_q15_Q8(LogMelSpectrogramTypeDef *S, const int16_t *pInSignal,
int8_t *pOutCol, int8_t offset, float32_t inv_scale)` —
`feature_extraction.h:207`, body `feature_extraction.c:250-324`. It does
int16→float (`arm_q15_to_float`, i.e. **÷32768**, so the scale already matches librosa),
centre-pad the 400-sample window inside the 512-point frame, apply the window LUT,
`arm_rfft_fast_f32`, power or magnitude spectrum, sparse mel dot-products, `/= Ref`,
clamp non-positive to `FLT_MIN`, `logf`, then quantise:

```c
pOutCol[i] = (int8_t)__SSAT((int32_t)roundf(p_out[i]*inv_scale + (float)offset), 8);
```

The f16 twin is `LogMelSpectrogramColumn_q15_f16_Q8` (`feature_extraction_f16.c:264`).

**`docs/FEASIBILITY.md` §Gate 5 says "float32 variant, not `_q15_Q8`". That naming is
inverted.** `_q15_Q8` *is* the float32 variant; `_q15_f16_Q8` is the float16 one. The
instruction is right, the name is wrong: take `LogMelSpectrogramColumn_q15_Q8` from
`feature_extraction.c`.

### 5.1 Every place ST's library cannot express what NeMo needs

Reference spec: `model/fe.py` / `model/fe_reference.py`.

| # | NeMo needs | ST gives | Verdict | Fix |
|---|---|---|---|---|
| 1 | `ln(x + 2⁻²⁴)` — **additive** guard | `if (x <= 0) x = FLT_MIN; x = logf(x);` — a **clamp**, and at 1.175e-38 (`feature_extraction.c:293-298, 315-317`) | **CANNOT EXPRESS. Fatal.** | Fork the function; replace the clamp+log with `logf(x + 5.9604645e-8f)` |
| 2 | per-feature (per-mel-bin) mean/std over the window, `ddof=1`, `+1e-5` | **absent — no normalisation of any kind exists in the library** | **CANNOT EXPRESS. Fatal.** | Write it (§5.3) |
| 3 | float32 dynamic range down to ~1e-14 | app is wired to **float16** (`app_config.h:51`, `preproc_dpu.h:39-46`, and `audio_bm.h:28-33` `#error`s if you change it) | **CANNOT EXPRESS in the shipped config. Fatal.** | Switch the whole preproc to f32 (§5.2) |
| 4 | pre-emphasis `x[n] − 0.97·x[n−1]`, `x[0]` kept | absent | not expressible in-library | 4 lines before the STFT — **or skip it**: `eval/results/fe.log` says no-pre-emphasis is 5.29 % vs 5.83 % reference |
| 5 | symmetric Hann, `periodic=False` | `Window_Init()` uses `cos(2πi/len)` = **periodic** (`window.c:78`); the LUT generator uses `librosa.filters.get_window('hann', 400)`, also periodic (`lookup_tables_generator.py:100`) | expressible — it is a LUT | Generate the LUT with `scipy.signal.get_window("hann",400,fftbins=False)`. `fe.log`: **no WER difference** |
| 6 | 80 mels, Slaney scale + Slaney norm, fmin 0, fmax 8000 | `MEL_SLANEY` + `Normalize=1` + `Mel2F=1` all exist (`mel_filterbank.c:175-189`, `preproc_dpu.c:62-64`) | **expressible** | config only |
| 7 | power spectrum (`power=2.0`) | `SPECTRUM_TYPE_POWER` (`feature_extraction.h:76`) | **expressible** | config only |
| 8 | `n_fft 512`, `win 400`, `hop 160`, `sr 16000` | already exactly this in the AED config (`ai_model_config.h:45-47`) | **expressible** | config only |
| 9 | `center=True`, reflect/constant pad of the *signal* by n_fft/2 | app slides the window over the buffer with no signal-edge padding | not expressible | ignore — `fe.log`: `center=False` is **5.43 %**, better than reference |
| 10 | 257 FFT bins | `MelFilterbank_Init()` scans `j < n_fft/2` = 256, dropping the Nyquist bin (`mel_filterbank.c:147`) | latent bug | **harmless here**: measured max stop index for our filterbank is **255**. Also moot — we ship a librosa-generated LUT, not the runtime generator |
| 11 | `log_zero_guard` applied to the mel energy, not the power spectrum | same place in ST | expressible | — |

**Items 1, 2 and 3 are the work.** Items 6-8 are a config file. Items 4, 5, 9 are free.

**On item 3, with numbers.** `p_out[i] = FLT_MIN;` in `feature_extraction_f16.c:310`
assigns 1.175e-38 to a `float16_t`, which **underflows to exactly 0.0**, and the next line
takes `logf(0.0f)` = **−∞**, then `(int16_t)roundf(-INF * inv_scale + offset)` — undefined,
in practice `__SSAT` to −128. That alone disqualifies the f16 path. Worse, float16's
smallest subnormal is 5.96e-8 — numerically identical to NeMo's log guard — so the guard
sits at the very bottom of the type. Measured on `corpus/LibriSpeech/dev-clean`, first
utterance, 8 s window, Slaney 80-mel power spectrum:

| condition | mel energies below fp16 smallest **normal** 6.10e-5 | below fp16 smallest **subnormal** 5.96e-8 (→ hard zero) |
|---|---:|---:|
| native level (peak −8.2 dBFS) | **47.40 %** | 5.54 % |
| scaled to −54 dBFS | 99.77 % | **76.22 %** |

At the level the DK's mic actually delivers, three quarters of the feature matrix is
identically zero before the logarithm is even reached. float32 is not an optimisation here.

### 5.2 LUT generation

`Projects/X-CUBE-AI/models/GenHeader/lookup_tables_generator.py` is the generator;
`gen_h_file.py` writes `ai_model_config.h`; both are driven from a hydra YAML
(`GenHeader/user_config_aed.yaml`). It emits four symbols into
`Projects/Dpu/user_mel_tables.{c,h}`:

```c
extern const PREPROC_FLOAT_T user_win[400];
extern const PREPROC_FLOAT_T user_melFiltersLut[N];
extern const uint32_t user_melFiltersStartIndices[80];
extern const uint32_t user_melFiltersStopIndices[80];
```

(`PREPROC_FLOAT_T` when `serie == STM32N6`, plain `float32_t` otherwise —
`lookup_tables_generator.py:277-304`.) The mel LUT is `librosa.filters.mel(...)` flattened
over its non-zero entries, with per-row start/stop indices; `MelFilterbank()`
(`mel_filterbank.c:213-229`) then walks it as one `arm_dot_prod_f32` per mel bin.

Measured for our parameters (`sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000,
norm="slaney", htk=False`):

- **500 non-zero coefficients out of 20,560 = 2.43 %**; run lengths 2..18; no empty rows,
  so ST's `ValueError` assert (`lookup_tables_generator.py:63-68`) passes.
- Max stop index **255**, so item 10 above never bites.
- **500 MAC/frame** vs 20,560 for a dense 80×257 matmul — **41.1×**. `docs/FEASIBILITY.md`
  §5 lists this as an available optimisation; ST's `MelFilterbank()` already does it.

LUT flash cost at float32: `user_melFiltersLut` 500×4 = 2,000 B, start/stop 2×80×4 = 640 B,
`user_win` 400×4 = 1,600 B → **4,240 B**.

Write `firmware/tools/gen_mel_tables.py` — a 60-line standalone replacement for the hydra
stack (which pulls in hydra-core, omegaconf, munch and seaborn to write two C files) that
emits float32 LUTs with a **symmetric** Hann window. Do not try to drive
`GenHeaders.py`: its YAML has no `norm: slaney` + `htk: False` + `power: 2.0` + 80-mel
combination validated, and it hardcodes `SILENCE_THR 3000.0F`.

### 5.3 Per-feature normalisation — what must be written

Confirmed absent: `grep -r` over the whole
`STM32_AI_AudioPreprocessing_Library` finds no mean, no variance, no per-bin statistic
anywhere. The library's only notion of level is `AudioSpENormalize_t {maxNormValue,
minThreshold}` used by the **STFT/speech-enhancement** path (`preproc_dpu.c:104-107`),
which is a peak-to-INT16_MAX rescale of the *waveform*, not a per-mel-bin z-score.

What NeMo does (`model/fe_reference.py:46-54`): for each of the 80 mel bins independently,
over the T=800 valid frames, `mu = mean`, `sd = std(ddof=1) + 1e-5`, then `(m - mu)/sd`.

This forces a **two-pass structure**, and that is the real architectural consequence:
the int8 quantisation cannot happen inside the per-column call, because the mean and
std are not known until the last column has been computed. `LogMelSpectrogramColumn_q15_Q8`
quantises inline (`feature_extraction.c:320-323`), so it cannot be used as-is.

Implementation, `firmware/src/citrinet_fe.c`:

```
pass 1, per column c in 0..799:
    LogMelColumn_f32(...)  -> float32 logmel[80]        (forked from _q15_Q8, no quantise)
    store into  float16_t  scratch[80][800]             128,000 B
    accumulate  double/float acc_sum[80], acc_sumsq[80]     (80 x 2 x 4 B)
pass 2:
    mu[b]  = acc_sum[b]/T
    var[b] = (acc_sumsq[b] - T*mu[b]*mu[b])/(T-1)       ddof=1
    sd[b]  = sqrtf(var[b]) + 1e-5f
    for c, b:  ai_in[b*800 + c] = SSAT8( roundf(((float)scratch[b][c]-mu[b])/sd[b] * inv_scale) )
    mcu_cache_clean_invalidate_range(ai_in, ai_in+64000)
```

Two notes. **(a)** `float16_t` is safe for `scratch[][]` *after* the logarithm — log-mel
values live in roughly [−32, +5], well inside fp16's range, and fp16's ~3 decimal digits
are far finer than the int8 grid the values land on. Only the pre-log power spectrum
needs float32. That halves the scratch buffer from 256,000 B to 128,000 B, which matters
(§5.4). **(b)** Do the sum/sumsq accumulation in float32 with the running-mean form or in
double; naive `sumsq` over 800 values in [−32,5] is fine in float32, but check it against
`model/fe_reference.py` rather than assuming.

**Do not** reuse `LogMelSpectrogramColumn_q15_os8_batch` (`feature_extraction.c:355`) —
it quantises per column and transposes, i.e. exactly the single-pass shape that
per-feature normalisation forbids.

### 5.4 The memory problem — this gate's real blocker

`preproc_dpu.h:29-33` derives the buffer sizes from the spectrogram geometry:

```c
#define PATCH_OVERLAP    (WINDOW_LENGTH - HOP_LENGTH)   /* 400 - 160 = 240      */
#define PATCH_NO_OVERLAP (COL * HOP_LENGTH)             /* 800 * 160 = 128,000  */
#define PATCH_LENGTH     (PATCH_OVERLAP + PATCH_NO_OVERLAP)   /* 128,240 samples */
```

With `CTRL_X_CUBE_AI_SPECTROGRAM_COL = 800`, the stock structures become:

| allocation | site | bytes |
|---|---|---:|
| `AudioBM_proc_t.proc_buff[128240]` int16 | `audio_bm.h:51` | 256,480 |
| `AudioBM_proc_t.audio_out[128240]` int16 | `audio_bm.h:52` | 256,480 |
| ring buffer `malloc(nbSamples·2·nbFrames)`, `nbSamples = ((128240/320)+1)·320 = 128,320`, `nbFrames = 2` | `audio_bm.c:292-296`, `AudioCapture_ring_buff.c:81-90` | 513,280 |
| **subtotal** | | **1,026,240** |
| linker region `RAM (xrw) : ORIGIN = 0x34000400, LENGTH = 1023K` | `STM32N657XX_LRUN.ld:47` | 1,047,552 |

That leaves **21,312 B** for all code, rodata, bss, the 64 KiB heap and the 20 KiB stack.
It will not link. The heap is `_end`→`_estack − 0x5000` (`sysmem.c` `_sbrk`), so the
`malloc` fails at runtime rather than at link time — `AudioCapture_ring_buff_alloc()`
logs and spins in `while(1)` (`AudioCapture_ring_buff.c:87-89`).

The utterance-based design does not need any of this. Restructure:

| # | Item | Files | Effort | Marker |
|---|---|---|---|---|
| 5.1 | Copy the two BSP patch files and the four Dpu files into `firmware/src/` and repoint the Makefile (vendor/ is read-only) | `firmware/Makefile` | 2 h | mechanical |
| 5.2 | Switch preproc to float32: drop `PREPROC_FLOAT_16`, remap `preproc_dpu.h:39-46` to the f32 symbols, delete the `#error` in `audio_bm.h:28-33`, swap `AudioProcCtx_f16_t` → a float32 context | `app_config.h`, `preproc_dpu.h`, `audio_proc.h`, `audio_bm.h` | 3 h | *mechanical but wide* |
| 5.3 | `firmware/tools/gen_mel_tables.py` → float32 LUTs, symmetric Hann | new | 2 h | mechanical |
| 5.4 | `firmware/inc/ai_model_config.h`: NMEL 80, COL 800, MEL_SLANEY, NORMALIZE 1, FMIN 0, FMAX 8000, SPECTRUM_TYPE_POWER, SILENCE_THR 0 | new | 1 h | mechanical |
| 5.5 | Fork `LogMelSpectrogramColumn_q15_Q8` → `LogMelColumn_f32()` with `logf(x + 2⁻²⁴)` and **no** inline quantise | `firmware/src/citrinet_fe.c` | 3 h | **RISKY** |
| 5.6 | Two-pass per-feature normalisation + quantise (§5.3) | same | 4 h | **RISKY** |
| 5.7 | **Replace the capture path**: drop `proc_buff`/`audio_out`/the ring buffer; run pass 1 incrementally on 160-sample hops straight out of the MDF callback into a 128,000 B fp16 scratch | `firmware/src/audio_capture.c` | 6 h | **RISKY** |
| 5.8 | Gain stage: raise `MDF_GAIN` from 2, and/or `HAL_MDF_SetGain()` AGC, and/or change the `/256` at `Patch/...audio.c:3172,3197` | copied patch file | 3 h | **RISKY** |
| 5.9 | **Guard-occupancy self-test, in from the first commit**: count mel bins where the pre-log energy < 2⁻²⁴, print the fraction each utterance, refuse to invoke above 20 % | `citrinet_fe.c` | 2 h | mechanical |
| 5.10 | Bit-compare against `model/fe_reference.py` on a canned waveform | host + board | 4 h | **RISKY** |

**Budget after restructuring:** fp16 log-mel scratch 128,000 B + 80×2 float accumulators
640 B + a 512-sample float32 FFT scratch ~2 KB + the 320-sample MDF buffers.
Call it **~135 KB** against 1,047,552 B. Comfortable. The int8 feature tensor itself
(64,000 B) is already allocated by the NPU runtime at `p_stai_inputs[0]` and is *not*
in this region.

**On 5.8.** Two independent knobs, both upstream of the int16 truncation that
`docs/FEASIBILITY.md` §2(d) shows only recovers to 10.45 % if you gain up afterwards:

1. `MDF_GAIN(AUDIO_FREQUENCY_16K)` = 2 today (`Patch/...audio.c:164`); range −16..+24 in
   ~3 dB steps. Going from −54 dBFS to ≈ −20 dBFS is +34 dB ≈ **+11 steps → Gain 13**,
   with headroom to 24. `HAL_MDF_SetGain()` works during acquisition, so a slow AGC
   driven by the previous utterance's peak is available for free.
2. The `/256` in the two MDF callbacks. `Audio_DigMicRecBuff` is `int32_t` holding MDF's
   24-bit output; `/256` discards 8 bits. At −54 dBFS the signal occupies the bottom ~9 of
   those 24 bits, so this shift is where the resolution dies. Widening to `/16` buys 24 dB
   *before* `SaturaLH(-32768, 32767)`.

Prefer knob 1 (analogue-domain, no clipping risk from a fixed shift), instrument both.

**Pass:** features match `model/fe_reference.py` to within a few LSB of the int8 grid,
**and** guard occupancy < 20 % on live speech.
**Stop if** occupancy stays high after gain — the gain is in the wrong place in the chain.

**Unverified:** M55 log-mel cost. 800 frames × (512-pt real FFT + 500 MAC + 80 `logf` +
80 quantise), plus a second pass of 64,000 multiply-adds. Nobody on this machine has
measured any on-device audio front end. DWT-time it with `port_dwt_reset()` /
`port_dwt_get_cycles()` (`Projects/Common/cpu_stats.h`), which `preproc_dpu.c:126-128,
153-155` already brackets the call with under `APP_BARE_METAL`.

---

## Gate 6 — greedy CTC + detokeniser

Cheap, self-contained, fully testable on the host. **0.5 d.**

| # | Item | Files | Effort | Marker |
|---|---|---|---|---|
| 6.1 | ~~Tokenizer table generator~~ | `firmware/tools/gen_tokenizer.py` | **done** | mechanical |
| 6.2 | ~~Generated vocabulary header~~ | `firmware/inc/citrinet_vocab.h` | **done** | mechanical |
| 6.3 | ~~`citrinet_ctc_decode()` — argmax, collapse repeats, drop blank, concatenate~~ | `firmware/src/citrinet_ctc.c`, `firmware/inc/citrinet_ctc.h` | **done** | mechanical |
| 6.4 | ~~Host oracle test against `model/fe.py:greedy()` on ≥20 dev-clean utterances~~ | `firmware/test/` | **done** | mechanical |

**Gate 6 is closed.** 100 dev-clean utterances (calibration-disjoint, seed 20260816)
decoded by the C implementation and by `model/fe.py:greedy()` from the same int8
tensor: **0 text disagreements over 9,226 characters, 0 argmax disagreements over
10,000 frames**, plus 480 synthetic logit matrices (39,972 of 48,000 frames carrying
a tied argmax) with 0 disagreements. Evidence: `firmware/test/results/gate6_ctc.json`,
reproduce with `python firmware/test/run_gate6.py 100`.

### 6.1/6.2 are complete and verified

`firmware/tools/gen_tokenizer.py` reads `tokenizer/vocab.txt` (1025 lines, `<piece> <id>`,
ids asserted dense and in file order) and emits `firmware/inc/citrinet_vocab.h`.

**Charset check: 0 violations.** Every one of the 1025 pieces is ASCII apart from U+2581,
which appears in **812** of them. The generator substitutes U+2581 → ASCII `0x20` at
generation time, so `kPieces[]` is pure 7-bit ASCII — which is also all the ST font tables
can render — and detokenisation is plain concatenation with no UTF-8 handling on the MCU.
It also asserts no piece already contains a literal space (none do) and no piece contains NUL.

**Measured flash size:**

```
$ arm-none-eabi-size -A tv_arm.o        # -mcpu=cortex-m55 -mthumb -mfpu=fpv5-d16 -Os
.rodata           8222
```

| symbol | bytes |
|---|---:|
| `kPieces[6170]` — NUL-separated blob | 6,170 |
| `kOffset[1026]` — `uint16_t` | 2,052 |
| **total `.rodata`** | **8,222** |

Longest piece 12 B. Max offset 6,170, comfortably inside `uint16_t`.
API, both `static inline`:

```c
const char *citrinet_piece(uint32_t id);      /* -> NUL-terminated ASCII, &kPieces[kOffset[id]] */
uint32_t    citrinet_piece_len(uint32_t id);  /* kOffset[id+1] - kOffset[id] - 1               */
```

plus `CITRINET_VOCAB_SIZE 1025`, `CITRINET_BLANK_ID 1024`, `CITRINET_UNK_ID 0`,
`CITRINET_PIECES_BYTES 6170`, `CITRINET_MAX_PIECE_LEN 12`.
Tables are file-static; include the header from **exactly one** translation unit.

Verified by compiling the header and dumping all 1025 strings from C, then diffing against
the Python-side `vocab.txt` with U+2581→space: **identical, 1025/1025**.
`strlen(citrinet_piece(i)) == citrinet_piece_len(i)` for all i.
Re-run and re-check at any time with `python3 firmware/tools/gen_tokenizer.py --check`,
which now also diffs the regenerated header against the one on disk, so it fails if
`citrinet_vocab.h` has been hand-edited or has drifted from `tokenizer/vocab.txt`.

**One defect found and fixed at Gate 6.** `kPieces[]` was emitted as adjacent string
literals, which concatenate into a single **6,170-character** literal. C99 5.2.4.1 only
requires an implementation to support 4,095, so `-Wpedantic` rejects the header
(`-Woverlength-strings`) and MISRA C:2012 Rule 1.1 forbids it. The generator now emits
character constants instead. The `.rodata` is **byte-for-byte identical** between the two
forms (`cmp` of `objcopy --only-section=.rodata` output, 8,222 B both ways) — only the
source representation changed, and the whole Gate 6 unit now builds under
`-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`.

### 6.3 The decoder

Both scales are per-tensor, so argmax runs directly on the int8 logits — no dequantisation
(`docs/FEASIBILITY.md` §2(g)). The output is `{100, 1025, 1}` CHANNEL_FIRST with vocabulary
as the fast axis, so frame `t` is `&out[t*1025]` and the inner loop is a contiguous
1025-element `int8_t` max — a natural `vmax` reduction. ~50 lines:

```c
uint32_t prev = 0xFFFFFFFFu;  char *w = text;
for (uint32_t t = 0; t < 100; t++) {
    const int8_t *row = &logits[t * CITRINET_VOCAB_SIZE];
    uint32_t best = 0; int8_t bv = row[0];
    for (uint32_t v = 1; v < CITRINET_VOCAB_SIZE; v++) if (row[v] > bv) { bv = row[v]; best = v; }
    if (best != prev && best != CITRINET_BLANK_ID) {
        uint32_t n = citrinet_piece_len(best);
        if (w - text + n < cap) { memcpy(w, citrinet_piece(best), n); w += n; }
    }
    prev = best;
}
*w = '\0';   /* then strip one leading space */
```

Ties: `>` keeps the lowest index, matching NumPy `argmax`. Worst case output length is
100 frames × 12 B = 1,200 B; a 1,280 B buffer is enough for any input.

**What shipped** (`firmware/inc/citrinet_ctc.h`, `firmware/src/citrinet_ctc.c`):

```c
citrinet_ctc_status_t citrinet_ctc_argmax(const int8_t *logits, uint32_t n_frames,
                                          uint16_t *ids);
citrinet_ctc_status_t citrinet_ctc_ids_to_text(const uint16_t *ids, uint32_t n_frames,
                                               char *text, uint32_t cap, uint32_t *n_written);
citrinet_ctc_status_t citrinet_ctc_decode(const int8_t *logits, uint32_t n_frames,
                                          char *text, uint32_t cap, uint32_t *n_written,
                                          uint16_t *ids /* may be NULL */);
const char *citrinet_ctc_piece(uint32_t id);
```

`citrinet_ctc_argmax()` is split out because **Gate 4 needs exactly that and nothing
else** — it takes no vocabulary and no text buffer. `cap` includes the NUL; overflow
returns `CITRINET_CTC_E_TRUNC` with a valid NUL-terminated prefix rather than
truncating silently. `CITRINET_CTC_TEXT_CAP` is 1,201 B (100 × 12 + 1), asserted at
compile time against `CITRINET_MAX_PIECE_LEN`; `citrinet_ctc_decode()` with
`ids == NULL` uses a 512 B automatic array, so nothing allocates.

Measured cost, `arm-none-eabi-gcc 14.3.1 -mcpu=cortex-m55 -Os -ffunction-sections`:

| symbol | bytes |
|---|---:|
| `citrinet_ctc_argmax` | 68 |
| `citrinet_ctc_ids_to_text` | 364 |
| `citrinet_ctc_decode` | 66 |
| `citrinet_ctc_piece` | 36 + 1 rodata |
| `kPieces` + `kOffset` (§6.2) | 8,222 |
| **total flash for Gate 6** | **8,757** |

`.data` and `.bss` are both **0** and `arm-none-eabi-nm -u` lists **no undefined
symbols**: the translation unit's only include is `<stdint.h>`, so it links against
neither libc nor the HAL.

---

## Gate 7 — LCD graft

Last, deliberately: it cannot fail in a way that invalidates the model. **1-1.5 d.**

`STM32N6-GettingStarted-Audio` is strictly headless — there is no `stm32n6570_discovery_lcd.c`
in its BSP, no `Utilities/` directory at all, and no font tables. Everything comes from the
ObjectDetection package.

### 7.1 Files to copy — exact list

From `vendor/STM32N6-GettingStarted-ObjectDetection/`:

| # | source | destination | bytes |
|---|---|---|---:|
| 1 | `STM32Cube_FW_N6/Drivers/BSP/STM32N6570-DK/stm32n6570_discovery_lcd.c` | `firmware/bsp/` | 45,911 |
| 2 | `STM32Cube_FW_N6/Drivers/BSP/STM32N6570-DK/stm32n6570_discovery_lcd.h` | `firmware/bsp/` | 11,899 |
| 3 | `STM32Cube_FW_N6/Drivers/BSP/Components/rk050hr18/rk050hr18.h` | `firmware/bsp/Components/rk050hr18/` | 2,163 |
| 4 | `STM32Cube_FW_N6/Utilities/lcd/stm32_lcd.c` | `firmware/lcd/` | 35,583 |
| 5 | `STM32Cube_FW_N6/Utilities/lcd/stm32_lcd.h` | `firmware/lcd/` | 6,969 |
| 6 | `STM32Cube_FW_N6/Utilities/Fonts/fonts.h` | `firmware/lcd/Fonts/` | 1,522 |
| 7 | `STM32Cube_FW_N6/Utilities/Fonts/font20.c` (+ others as wanted) | `firmware/lcd/Fonts/` | 46,104 src |
| 8 | `Application/STM32N6570-DK/Src/stm32_lcd_ex.c` | `firmware/src/` | 1,626 |
| 9 | `Application/STM32N6570-DK/Inc/stm32_lcd_ex.h` | `firmware/inc/` | 1,261 |

Already present in the audio package, do **not** copy: `Drivers/BSP/Components/Common/lcd.h`
(the abstract `LCD_Drv_t` that `stm32n6570_discovery_lcd.h:41` includes),
`stm32n6570_discovery.c` (**byte-identical** between the two packages — verified with
`diff`; `stm32n6570_discovery.h` differs only in the four BSP_VERSION macros, 1.1.0 vs 1.3.0),
`stm32n6570_discovery_bus.c/.h`, and the LTDC/DMA2D HAL drivers.

There is **no `rk050hr18.c`** — the panel driver is a header-only descriptor table.
There is **no GT911 component** in either package, so `stm32n6570_discovery_ts.c` cannot
compile (`stm32n6570_discovery_ts.h:30` includes `../Components/gt911/gt911.h`, absent).
Touch remains unavailable, exactly as `docs/FEASIBILITY.md` §2(e) says.

Font table sizes, **measured** (`arm-none-eabi-size`, cortex-m55 -Os):

| font | glyph W×H | `.text` | chars per 800 px line | lines per 480 px |
|---|---|---:|---:|---:|
| Font8 | 5×8 | 760 | 160 | 60 |
| Font12 | 7×12 | 1,140 | 114 | 40 |
| Font16 | 11×16 | 3,040 | 72 | 30 |
| **Font20** | **14×20** | **3,800** | **57** | **24** |
| Font24 | 17×24 | 6,840 | 47 | 20 |
| all five | | **15,580** | | |

`N_PRINTABLE_CHARS` is hardcoded to **47** in `stm32_lcd_ex.c:25` with the comment
*"800px wide screen / 17px wide font"* — that is a **Font24** figure, but the OD app calls
`UTIL_LCD_SetFont(&Font20)` (`main.c:497`). So ST's own 47 is conservative by 10 characters
for the font it actually uses. For a caption, ship **Font20** and raise `N_PRINTABLE_CHARS`
to 57, or ship **Font16** (72×30) and raise it to 72. An 8 s utterance is ~20 words ≈ 120
characters ≈ 3 lines at Font20.

### 7.2 Text drawing

```c
BSP_LCD_Init(0, LCD_ORIENTATION_LANDSCAPE);          /* 800x480, configures LTDC clock  */
BSP_LCD_ConfigLayer(0, LTDC_LAYER_1, &LayerConfig);  /*   from IC16 inside its MSP init */
UTIL_LCD_SetFuncDriver(&LCD_Driver);                 /* stm32n6570_discovery_lcd.c:111  */
UTIL_LCD_SetLayer(LTDC_LAYER_1);
UTIL_LCD_SetFont(&Font20);
UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
UTIL_LCD_Clear(UTIL_LCD_COLOR_BLACK);
UTIL_LCDEx_PrintfAt(0, LINE(2), CENTER_MODE, "%s", transcript);
SCB_CleanDCache_by_Addr(fb, FB_BYTES);
```

`LINE(x)` is `((x) * UTIL_LCD_GetFont()->Height)` (`stm32_lcd.h:107`), i.e. 20 px at Font20.
`Text_AlignModeTypdef` is `{CENTER_MODE, RIGHT_MODE, LEFT_MODE}`. The full drawing API
(`UTIL_LCD_FillRect`, `DrawRect`, `DisplayStringAt`, …) is at `stm32_lcd.h:162-193`.

The OD app uses two layers (camera background RGB565 + ARGB4444 overlay). A captioner needs
**one**: a single `LTDC_LAYER_1` in `LCD_PIXEL_FORMAT_RGB565`, **800 × 480 × 2 = 768,000 B**.
Double-buffering doubles that to 1,536,000 B; a text-only display with a full clear-and-redraw
per utterance does not need it.

### 7.3 Linker script changes

`Projects/GS/STM32CubeIDE/STM32N657XX_LRUN.ld` has **one** region:

```
MEMORY { RAM (xrw) : ORIGIN = 0x34000400, LENGTH = 1023K }
```

768,000 B does not fit alongside the front end. Add the PSRAM region and section, copying
`Application/STM32N6570-DK/STM32CubeIDE/STM32N657xx.ld:16` and `:159-164`:

```ld
MEMORY {
  RAM   (xrw) : ORIGIN = 0x34000400, LENGTH = 1023K
  PSRAM (xrw) : ORIGIN = 0x91000000, LENGTH = 16M     /* XSPI1 / APS256XX, memory-mapped */
}
...
  .psram_section (NOLOAD) : { . = ALIGN(32); *(.psram_bss) . = ALIGN(32); } >PSRAM
```

then `__attribute__((section(".psram_bss"))) static uint8_t lcd_fb[800*480*2];`
(`(NOLOAD)` matters: without it the linker puts 768 KB of zeros in the .bin, which alone
would breach the 512 KiB flash slot).

Three enabling changes on the audio side, all in the audio package's own files:

1. **PSRAM init.** `audio_bm.c:834-853` `Ext_Mem_Config()` already contains the code —
   `BSP_XSPI_RAM_Init(0); BSP_XSPI_RAM_EnableMemoryMappedMode(0);` plus the no-prefetch
   hotfix — but it is behind `#ifdef USE_EXT_SRAM`, and `USE_EXT_SRAM` is defined
   **nowhere** in the package (only that one `#ifdef`). Add `-DUSE_EXT_SRAM` to `C_DEFS`.
   `Drivers/BSP/Components/aps256xx/aps256xx.c` is already in `C_SOURCES` (Makefile:50).
2. **RIF.** `NPU_Config()` (`Projects/Common/misc_toolbox.c:229-257`) configures the RIF
   master/slave attributes for the NPU only. Add LTDC1, LTDC2 and DMA2D, copying
   `Security_Config()` from `Application/STM32N6570-DK/Src/main.c:390-408` — the six
   lines it adds beyond the audio app's version are `RIF_MASTER_INDEX_{DMA2D,LTDC1,LTDC2}`
   and `RIF_RISC_PERIPH_INDEX_{DMA2D,LTDC,LTDCL1,LTDCL2}`.
   Separately, `RISAF_Config()` (`misc_toolbox.c:259-289`) exists and **is never called** —
   `grep -rn RISAF_Config Projects/` finds only the definition and the prototype. It gates
   `RISAF11_S` (OCTOSPI1 @ 0x90000000, i.e. the PSRAM) behind
   `USE_EXTERNAL_MEMORY_DEVICES == 1`, which is likewise never defined. If PSRAM accesses
   fault, this is why; the FSBL may already have set it, so verify before changing.
3. **HAL modules — both LTDC and DMA2D are off.** `stm32n6570_discovery_lcd.c` references
   DMA2D 59 times. In `Projects/GS/Inc/stm32n6xx_hal_conf.h`, *both* the module macros a
   grep appears to find are sitting **inside block comments**:

   ```c
   :47  /* #define HAL_DMA2D_MODULE_ENABLED
   :48     #define HAL_DTS_MODULE_ENABLED
   :49     #define HAL_ETH_MODULE_ENABLED */          <- comment closes here
   ...
   :59  /* #define HAL_I3C_MODULE_ENABLED
   ...
   :64     #define HAL_LTDC_MODULE_ENABLED            <- inside the comment
   :66     #define HAL_MCE_MODULE_ENABLED */          <- comment closes here
   ```

   A plain `grep -n LTDC` reports line 64 as a `#define` and it is not one. Both comment
   runs must be restructured so `HAL_DMA2D_MODULE_ENABLED` and `HAL_LTDC_MODULE_ENABLED`
   are live. Then add three already-present sources to the Makefile:
   `Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_{ltdc,ltdc_ex,dma2d}.c`.
   (The HAL `.c` files self-guard on these macros, which is why
   `stm32n6xx_hal_icache.c` is in `C_SOURCES` today while `HAL_ICACHE_MODULE_ENABLED` is
   commented out — it compiles to an empty object. Adding a source without enabling the
   macro therefore fails at *link* time, not compile time, with undefined `HAL_LTDC_*`.)
4. **Fix `LCD_LAYER_0_ADDRESS`.** `Projects/GS/Inc/stm32n6570_discovery_conf.h:67-68`
   already defines both symbols, so the BSP header compiles unmodified — but the values
   are wrong for this app:

   ```c
   :67  #define LCD_LAYER_0_ADDRESS   0x34000000 /* SRAM1 */
   :68  #define LCD_LAYER_1_ADDRESS   0x340C0000 /* SRAM1 */
   ```

   `BSP_LCD_InitEx()` programs layer 0 at `LCD_LAYER_0_ADDRESS` before the application
   gets a chance to call `BSP_LCD_ConfigLayer()`
   (`stm32n6570_discovery_lcd.c:286-293`, comment: *"This configuration can be override
   by calling BSP_LCD_ConfigLayer() at application level"*). 0x34000000 is 1 KiB below
   the app's own `ORIGIN = 0x34000400`, and 0x340C0000 is squarely inside its `.text`.
   The LTDC would begin DMA-scanning the application's code as pixels the moment
   `BSP_LCD_Init()` returns. Reads only, so nothing corrupts — but it is bus traffic and
   visible garbage for the window between init and `ConfigLayer`. Point both at the PSRAM
   framebuffer (the OD package uses 0x34200000 / 0x32100000 instead).

| # | Item | Effort | Marker |
|---|---|---|---|
| 7.1 | Copy the nine files; add include paths | 1 h | mechanical |
| 7.2 | Makefile: 3 HAL sources + `stm32_lcd.c` + `font20.c` + `stm32_lcd_ex.c`; `-DUSE_EXT_SRAM` | 1 h | mechanical |
| 7.3 | `hal_conf.h`: un-comment **both** `HAL_DMA2D_MODULE_ENABLED` and `HAL_LTDC_MODULE_ENABLED` | 0.5 h | mechanical |
| 7.4 | Linker: PSRAM region + `.psram_section (NOLOAD)` | 1 h | mechanical |
| 7.5 | RIF for LTDC1/LTDC2/DMA2D; decide on `RISAF_Config()` | 2 h | **RISKY** |
| 7.6 | `LCD_init()` + word-wrap at 57 chars; raise `N_PRINTABLE_CHARS` | 3 h | mechanical |
| 7.7 | Push-to-talk UI states (idle / recording / thinking / result) | 3 h | mechanical |

---

## Gate 7b — the button

Not a gate of its own; folded into Gate 5/7. **1 h, mechanical.**

`BUTTON_USER1` **exists** in the audio package's own BSP and is already initialised and
wired to an interrupt by the stock app.

- Enum: `BUTTON_USER1 = B2` — `Drivers/BSP/STM32N6570-DK/stm32n6570_discovery.h:74`
- Pins: `BUTTON_USER1_PIN GPIO_PIN_13`, `BUTTON_USER1_GPIO_PORT GPIOC`,
  `BUTTON_USER1_EXTI_IRQn EXTI13_IRQn`, `BUTTON_USER1_EXTI_LINE EXTI_LINE_13` — `:197-202`
- IRQ priority `BSP_BUTTON_USER1_IT_PRIORITY 15U` — `Projects/GS/Inc/stm32n6570_discovery_conf.h:84`

```c
int32_t  BSP_PB_Init    (Button_TypeDef Button, ButtonMode_TypeDef ButtonMode);   /* :316 */
int32_t  BSP_PB_DeInit  (Button_TypeDef Button);                                  /* :317 */
uint32_t BSP_PB_GetState(Button_TypeDef Button);                                  /* :318 */
void     BSP_PB_Callback(Button_TypeDef Button);                                  /* :319 — weak, override */
void     BSP_PB_IRQHandler(Button_TypeDef Button);                                /* :336 */
```

`ButtonMode_TypeDef` is `{BUTTON_MODE_GPIO = 0, BUTTON_MODE_EXTI = 1}` (`:81-82`).

Already done for you by the stock app:

- `init_bm()` calls `BSP_PB_Init(BUTTON_USER1, BUTTON_MODE_EXTI)` — `audio_bm.c:123`
- `EXTI13_IRQHandler()` calls `BSP_PB_IRQHandler(BUTTON_USER1)` — `Projects/GS/Src/stm32n6xx_it.c:186-189`
- `BSP_PB_Callback()` is overridden at `audio_bm.c:605-611`, currently calling
  `toggle_audio_proc()`. Replace the body.

So push-to-talk is: set a `volatile bool g_ptt` in `BSP_PB_Callback`, and read
`BSP_PB_GetState(BUTTON_USER1)` in the main loop for the hold/release edges.
`BUTTON_TAMP` (`B4`, PE0, `EXTI0`) is likewise already wired if a second button is wanted.

---

## Cross-cutting rules

1. **`vendor/` is read-only.** Everything that must change — `preproc_dpu.c`, `ai_dpu.c`,
   `audio_bm.c`, `audio_bm.h`, `audio_proc.h`, `ai_model_config.h`, `user_mel_tables.c`,
   `Patch/stm32n6570_discovery_audio.c`, `stm32n6xx_hal_conf.h`, the linker script,
   the Makefile — gets copied into `firmware/` and built from there. Write
   `firmware/Makefile` as a fork of `Projects/GS/Makefile` whose `C_SOURCES` point at
   `firmware/src/` first and fall through to `vendor/` for the untouched 95 %.
2. **Pin the toolchain.** ST Edge AI Core **v4.0.1-20581**. All 107 grouped convolutions
   reaching hardware is an empirical property of this build with no vendor commitment
   (`docs/FEASIBILITY.md` risk 4). Make `618 epochs / 0 SW / 0 hybrid` a regression gate on
   any compiler bump: `grep -c "pure software (SW) epochs *0"` the summary.
3. **Never interleave `zoo measure` with a demo flash.** Both write the external flash at
   0x70180000 (`docs/FEASIBILITY.md` risk 5).
4. **Do not touch the board in this pass.** No `STM32_Programmer_CLI`, no OTP, no
   `stedgeai validate --mode target`.

## Effort roll-up

| gate | effort | of which RISKY |
|---|---|---|
| 3 — build & flash stock app | 0.5 d | OTP fuse state |
| 4 — Citrinet + canned features | 1 d | ST int8 ≠ ORT int8; real latency; output cache |
| 5 — log-mel front end | **3 d** | log guard; per-feature norm; f32 migration; capture rewrite; gain staging |
| 6 — CTC + detokeniser | 0.5 d | — (6.1/6.2 already done) |
| 7 — LCD graft | 1.5 d | RIF/PSRAM |
| **total** | **6.5 d** | |

Down from the 8-11 d in `docs/FEASIBILITY.md` because gates 0-2 are closed and Gate 6 is
half-built — but Gate 5 grew, because the float16 finding and the `PATCH_LENGTH` memory
blow-up were not in the original estimate.

# Quickstart — from `git clone` to a talking board

Speech recognition on an STM32N6570-DK: hold the button, speak, and the
transcript appears on the panel. NVIDIA NeMo **Citrinet-256 CTC**, int8, running
its encoder on the Neural-ART NPU and its log-mel front end on the Cortex-M55.

**Read this first: what a clone does *not* contain.** Two directories are
deliberately outside git, and both are needed to flash:

| | why | how to get it |
|---|---|---|
| `vendor/` | ST's application packages — theirs to distribute, ~1 GB | §1, two `git clone`s |
| `artifacts/` | 4.8 GB of compiler workspaces, and a model derived from third-party weights | §2 |

Everything this project *wrote* — the firmware, the graph rewrites, the compile
driver, the host tooling, the evidence — is in the clone.

---

## 0. What you need

**Hardware.** One STM32N6570-DK (MB1939). Nothing else: the microphone, the
800×480 panel and the button are all on the board. No PSRAM is used.

**Tools.**

| tool | needed for | notes |
|---|---|---|
| STM32CubeProgrammer | flashing, signing | free from st.com; supplies `STM32_Programmer_CLI`, `STM32_SigningTool_CLI` and the `MX66UW1G45G_STM32N6570-DK.stldr` external loader |
| `arm-none-eabi-gcc` | building | STM32CubeCLT ships one. Built here with 10.3.1; ST's own binaries used 13.3.1 |
| ST Edge AI Core **4.0.1** | only to **recompile** the network | version matters — see `compile/GATE2.md` |
| Python 3 + numpy, onnx, onnxruntime, librosa, soundfile, pyserial | host tooling and the model pipeline | only for §2b and the scorers |

Tell the scripts where they are — once, in one place:

```bash
cp env.sh env.local.sh    # optional: keep your edits out of git
$EDITOR env.sh            # or just export the variables below
source env.sh             # prints what it found
```

`env.sh` takes `STM32CUBEPROG`, `ARM_BIN`, `STEDGEAI_ROOT` and `STT_PYTHON` from
the environment if they are already set, and otherwise looks in the usual places.
Every script in this repository sources it.

---

## 1. Fetch ST's packages

```bash
mkdir -p vendor && cd vendor
git clone https://github.com/STMicroelectronics/STM32N6-GettingStarted-Audio.git
git clone https://github.com/STMicroelectronics/STM32N6-GettingStarted-ObjectDetection.git
cd ..
```

Developed against **v2.3.0** of both:

| package | commit | needed for |
|---|---|---|
| `STM32N6-GettingStarted-Audio` | `46f1f97` (2026-04-16) | the application, the BSP, the FSBL, the AI runtime |
| `STM32N6-GettingStarted-ObjectDetection` | `7ae96b5` (2026-04-16) | **nothing at build time** — the nine LCD files were copied into `firmware/lcd/` and are in this repo. Clone it only to re-derive them |

Later versions will probably work; if something behaves oddly, check out those
commits before suspecting anything else.

---

## 2. Get the model

The board needs two generated things: `network.c` + `stai_network.c`, compiled
into the application, and a 9.7 MB weight blob flashed at `0x70400000`.

### 2a. If you already have an `artifacts/` tree

Skip to §3. `firmware/apply_vendor_mods.sh` verifies the md5 of both against the
deployed build and refuses to install anything else — see
`artifacts/model_c/MANIFEST.txt` for why that check exists.

### 2b. Building it from the upstream model

```bash
# 1. the upstream export
mkdir -p artifacts/onnx && cd artifacts/onnx
#    OpenVoiceOS/stt_en_citrinet_256_gamma_0_25_onnx  ->  model.onnx
cd ../..

# 2. freeze the shape, clean the graph, quantise  (model/README.md has the detail)
python model/mkstatic.py 800                      # -> static_800.onnx
python model/clean.py    static_800.onnx clean_800.onnx
python model/q800.py                              # -> q800_real.onnx   (needs LibriSpeech dev-clean)

# 3. THE TWO REWRITES THAT MAKE IT RUN ON THE NPU. Not optional.
python model/fold_stride2.py                      # -> q800_fold.onnx
python model/break_relu_chain.py                  # -> q800_relu4d_all.onnx

# 4. compile
compile/gen_model.sh artifacts/onnx/q800_relu4d_all.onnx deploy --install
```

> **Do not skip step 3, and do not compile `q800_real.onnx`.** It compiles
> cleanly, reports 0 software epochs and 0 hybrid epochs, and then **stalls the
> NPU forever** — twice, for two unrelated reasons, neither documented by ST. A
> stride-2 depthwise convolution hangs it, and so does an activation accelerator
> driving a convolution accelerator's data port. Both rewrites are bit-exact
> (`max|diff| = 0` over 3,075,000 outputs) and the fixed graph is also *faster*:
> 448 epochs against 618. `board/GATE4.md` rounds 10–19 is the whole story;
> `board/REPRO-blocker2.md` is a 9-node reproducer.

---

## 3. Patch the vendored tree

```bash
firmware/apply_vendor_mods.sh
```

Six mechanical changes to ST's package — the AI middleware moves to ST Edge AI
4.0.1, `ai_dpu.c` learns to accept int8 outputs, the Makefile gains the
`EXTRA_SOURCES`/`EXTRA_CFLAGS` hooks this project builds through, the heap
shrinks, and the model is installed over the built-in AED one. Then apply the
firmware itself:

```bash
git -C vendor/STM32N6-GettingStarted-Audio apply "$(pwd)/firmware/vendor-mods/gate4.patch"
```

That patch is the application: the microphone capture path, the front-end call,
the CTC decode, the LCD, the push-to-talk loop and the instrumentation. It
reverse-applies cleanly, so `git apply -R` undoes it.

---

## 4. Build

```bash
firmware/build.sh          # demo: microphone + LCD + push-to-talk
```

Other profiles, all useful for bring-up: `wav` (replay a waveform blob through
the real front end), `corpus` (replay host-computed features straight to the
NPU), `canned` (one built-in tensor — the smallest thing that exercises the
runtime). `board/BUILD.md` documents every define.

The image is ~777 kB against a 3 MB slot, and uses 83.7 % of the 1023 kB RAM
region.

> **The build is not bit-reproducible, and that is ST's design, not a fault.**
> Two builds of identical sources differ in ~409 bytes: ST's generated
> `stai_network.c:617` bakes `__DATE__ " " __TIME__` into the network descriptor,
> and `STM32_SigningTool_CLI` writes a nonce into the header. The *size* is stable
> and is a good check; the content is not. `board/flash_and_verify.sh` verifies
> what matters instead — it reads the image back off the flash and compares it to
> the file it just wrote.

---

## 5. Flash

**Both boot switches RIGHT** (development position), then:

```bash
CP=$STM32CUBEPROG/bin
EL=$CP/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr
V=vendor/STM32N6-GettingStarted-Audio

# once per board: the first-stage boot loader
$CP/STM32_Programmer_CLI -c port=SWD mode=UR --extload "$EL" -w $V/FSBL/ai_fsbl.hex

# once per model build: the weights
$CP/STM32_Programmer_CLI -c port=SWD mode=UR --extload "$EL" \
    -w artifacts/model_c/network_atonbuf.xSPI2.raw 0x70400000

# every build: the application, written and then READ BACK and compared
board/flash_and_verify.sh
```

Then **both switches LEFT** and power-cycle.

`mode=UR` (under reset), not `HOTPLUG` — `HOTPLUG` does not work on this part,
and finding that out cost most of Gate 3 (`board/GATE3.md`).

The two replay blobs are optional. If you want them:
`artifacts/corpus/corpus_blob.bin` at `0x71000000` and
`artifacts/corpus/wav_blob.bin` at `0x72000000`. The `demo` profile checks their
magic and skips them cleanly when absent.

---

## 6. Use it

The panel shows `HOLD USER1 TO TALK`. Hold the blue **USER1** button, speak,
release. You get the transcript, a spectrogram of what you just said, and
`fe … ms  npu … ms  … dBFS  guard …%  gain …`.

![What a working board looks like](docs/images/board-transcript.jpg)

This is what a healthy run looks like, and it doubles as a checklist: a
transcript, a spectrogram with visible formant structure and a dark zero-filled
tail, `fe`/`npu` near 133/140 ms, a peak comfortably inside −20…−5 dBFS with no
`CLIP`, and `guard` somewhere in the tens of percent rather than 0 % or 90 %.

The UART is on the ST-LINK VCP at **14400 8N1** — an unusual rate, and it is
ST's, not a typo (`Projects/GS/Inc/app_config.h`). `board/read_uart.py` reads it.

```
fe   log-mel on the M55        ~136 ms
npu  Citrinet-256 encoder      ~140 ms
     per 8 s window, everything on-chip
```

**What to expect from the accuracy.** Citrinet-256 is trained on LibriSpeech —
read audiobook English, close-mic'd, mostly American. Played that kind of audio
through the board's own microphone it scores **3.2 % WER**, against 4.3 % for the
same model fed host-computed features from flash: the microphone path costs
nothing measurable. Free-form live speech from an accented speaker measured
~30 %, and that gap is the model's training distribution, not this port —
`firmware/FRONTEND.md` §§14–15 rules out level, quantisation, SNR and
reverberation by measurement, and §18 shows the fp32 reference model making the
same errors on the same audio.

---

## 7. When it does not work

| symptom | first thing to check |
|---|---|
| nothing on the UART | switches LEFT and power-cycled? 14400 baud, not 115200? |
| board boots, then silence forever | you compiled `q800_real.onnx`. See §2b |
| `failed to erase memory` when flashing | switches are not in the RIGHT position |
| image flashes but does not run | signed without `-align`. `firmware/build.sh` always passes it; `make flash_bm` does not |
| panel dark, UART fine | check the `# gate7: LCD up … pixel clock` line. It should read ~24,571,428 Hz |
| `rc -4` on every utterance | the capture is too quiet — `citrinet_fe_run()` refusing. Watch the level meter |

`board/GATE3.md` and `board/GATE4.md` are the full debugging records, including
the six rounds spent misattributing a stalled NPU to the boot chain.

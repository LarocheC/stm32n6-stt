# Gate 3 — build and flash ST's unmodified audio app

**Status: PASS.** Built from source on this machine, signed correctly, flashed,
booted from external flash, and printing.

```
| 22      |  2.07%|  0.88|  1.20|  0.00|      <- ours, Arm GNU 13.3, -align
| 66      |  2.11%|  0.91|  1.19|  0.00|      <- ST's prebuilt, for comparison
```

Frame counter, CPU load, three per-stage times in ms. The small timing
difference is the compiler; the entry point on the board reads `0x340167bd`
(ours) rather than `0x34016B19` (ST's), so this is unambiguously our binary.

The toolchain is now proven end to end: **source → compile → sign → flash →
boot → run**, with ST's own model in the loop. Gate 4 can proceed.

Goal was to prove the toolchain, the boot chain and the board with ST's own
model in the loop, before any of our code is involved.

| step | result |
|---|---|
| 3.1 read OTP fuse state | **done** — word 124 = `0x00018000`, both bits already set. [`OTP.md`](OTP.md) |
| 3.2 decide on `fuse_vddio()` | **moot on this board** — the fuses it would program are already programmed |
| 3.3 `make bm` | **done** — but not with the toolchain we had pinned, see below |
| 3.4 sign + flash FSBL, app, weights | **done** — all three regions written |
| 3.5 confirm the UART banner | **done** — prints, after the signing fix below |

## The build recipe that works

```bash
export PATH=/home/claroche/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin:$PATH
cd vendor/STM32N6-GettingStarted-Audio/Projects/GS

# 1. build — vanilla Arm GNU needs -fcyclomatic-complexity stripped (see 3.3);
#    ST's own CubeCLT compiler accepts it directly.
make bm -j8 GCC_PATH=/home/claroche/opt/st/stm32cubeclt_1.21.0/GNU-tools-for-STM32/bin

# 2. sign — -align is MANDATORY on CubeProgrammer 2.21+ and ST's Makefile omits it
STM32_SigningTool_CLI -s -bin BuildGCC/BM/GS_Audio_N6.bin -nk -t ssbl -hv 2.3 \
                      -align -o BuildGCC/BM/GS_Audio_N6_sign.bin

# 3. VERIFY before flashing — costs two commands, saves a boot-switch round trip
test "$(xxd -e -g4 -s 0x70 -l 4 BuildGCC/BM/GS_Audio_N6_sign.bin | awk '{print $2}')" \
   = "$(arm-none-eabi-nm BuildGCC/BM/GS_Audio_N6.elf | awk '/ Reset_Handler/{print $1}' \
        | sed 's/^0*//' | awk '{printf "%x\n", strtonum("0x"$1)+1}')" \
  && echo "entry point OK"

# 4. flash (development switch position; mode=UR, HOTPLUG does not work here)
STM32_Programmer_CLI -c port=SWD mode=UR --extload <MX66UW1G45G_STM32N6570-DK.stldr> \
                     -w BuildGCC/BM/GS_Audio_N6_sign.bin 0x70100000

# 5. both switches LEFT, power-cycle, re-attach usbipd, read UART at 14400 8N1
```

## 3.3 — ST's Makefile does not build with vanilla Arm GNU

The zoo pins `arm-gnu-toolchain-13.3.rel1` (`config/toolchain.toml`), which is
the version ST document as validated for Cortex-M55. It **cannot build this
application**:

```
arm-none-eabi-gcc: error: unrecognized command-line option '-fcyclomatic-complexity'
```

`-fcyclomatic-complexity` is not an upstream GCC option. It is an ST addition,
present in the compiler ST ship inside STM32CubeCLT, and their Makefile passes
it unconditionally (`Projects/GS/Makefile`, `C_FLAGS`).

**Build with ST's own compiler instead**, which is already on this machine:

```bash
export PATH=/home/claroche/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin:$PATH
cd vendor/STM32N6-GettingStarted-Audio/Projects/GS
make bm -j8 GCC_PATH=/home/claroche/opt/st/stm32cubeclt_1.21.0/GNU-tools-for-STM32/bin
```

`GNU Tools for STM32 14.3.rel1.20251027` — accepts the flag, builds clean.

This is worth an entry in the zoo's fault atlas: *the validated toolchain
version and the toolchain that can build ST's own examples are not the same
toolchain*, and the failure is a flag error rather than anything that hints at
the real cause.

### Build result

```
Memory region   Used Size    Region Size   %age Used
        RAM:     603552 B        1023 KB      57.62%
   text 215596   data 26644   bss 361280   dec 603520
```

Signed binary **242,848 B**, entry point `0x34003019`.

Against the hard constraint from `firmware/WORKLIST.md` — the app slot
`0x70100000..0x70180000` is 512 KiB and overflowing it silently overwrites the
weight blob — this leaves **281,440 B of headroom** for everything gates 4-7
add. The RAM figure is the one to watch instead: 57.62 % of AXISRAM1 is gone
before our mel front end asks for its ~135 KB.

## 3.4 — what is on the flash now

This **overwrote what `zoo measure` had there**, as authorised.

| region | address | size | source |
|---|---|---:|---|
| FSBL | `0x70000000` | 61.28 KB | `FSBL/ai_fsbl.hex` |
| app (SSBL) | `0x70100000` | 237.16 KB | `GS_Audio_N6_sign.bin` |
| AED weights | `0x70180000` | 3.13 MB | `Projects/X-CUBE-AI/models/aed_weights.hex` |

All three reported `File download complete`. Commands in the repo history; the
external loader is `MX66UW1G45G_STM32N6570-DK.stldr`.

## 3.5 — why nothing prints yet

15 seconds at 14400 8N1 on `/dev/ttyACM0` returned **0 bytes**. The baud is
right (`app_config.h:58`; 14400 is the ceiling under the app's DVFS clock
changes, not a legacy default). The board simply is not running the app.

The DK has two boot switches. Both **right** is development mode — the mode that
makes `mode=HOTPLUG` SWD work, the mode the zoo's `n6_loader.py` drives, and the
mode the board is in now. Both **left** is boot-from-flash. ST's README §"Boot
modes" is explicit that the sequence is: program in the right position, switch
left, then **power-cycle** — a reset over SWD is not enough.

That is a physical action, so Gate 3 stops here.

### To finish it

1. Move **both** boot switches to the **left** position.
2. Power-cycle the board (unplug/replug, not a reset button).
3. ```bash
   source /home/claroche/stm32n6-deployment-zoo/.venv/bin/activate
   python board/read_uart.py
   ```

**Pass:** the AED application prints JSON event records. Play a sound near the
microphone — it classifies events like `crying_baby`, `clock_tick`, `sneezing`.

**Stop if** it does not print: fall back to ST's prebuilt
`Binary/STM32N6570-DK/STM32N6_GettingStarted_Audio_aed_bm.hex` via
`Binary/flash-bin.sh aed bm`, which needs no toolchain at all. If *that* prints
and ours does not, the problem is our build; if neither prints, it is the boot
chain or the switches.

## 3.5 diagnosis — what has been eliminated

After the switches went LEFT and the board was power-cycled, it printed nothing.
A reset-button press produced nothing either, so the app is not merely quiet
between events — it is not reaching `UART_Config()`.

**That is consistent with a hang anywhere in early init.** `UART_Config()` is
`audio_bm.c:122`, *after* `SystemClock_Config_Full` (101), `MPU_Config` (111),
`Ext_Mem_Config` (113), `NPU_Config` (114) and `IAC_Config` (115). Any of those
hanging is byte-identical, from outside, to a dead board — and `Ext_Mem_Config()`
reconfigures the very xSPI interface the app is executing from.

Eliminated, each by measurement rather than reasoning:

| suspect | how it was eliminated |
|---|---|
| wrong boot mode | SWD refuses the target in flash-boot mode and accepts it in dev mode — the switch flip demonstrably took |
| wrong baud | 14400 confirmed in *both* `#ifdef` branches (`app_config.h:57-61`); swept 9600–921600, nothing at any rate |
| wrong flash layout | parsed ST's prebuilt combined hex: `0x70000000` +62,752 B, `0x70100000` +244,544 B, `0x70180000` +3,282,785 B — exactly the three regions written |
| wrong use case | `network.c`, `stai_network.c`, `ai_model_config.h`, `user_mel_tables.c` all md5-match their `.aed` variants, so firmware and weights agree |
| **writes never committed** | **read back from the board: `0x70000000` and `0x70100000` both begin `53544d32` (`STM3`), and the app header byte-matches our signed binary.** The flash is correctly programmed |

What remains is the firmware itself: we built with ST's GCC 14.3 (the only
compiler on this machine that accepts `-fcyclomatic-complexity`), and ST's
prebuilt binary was built with something else.

**Next experiment:** ST's prebuilt `STM32N6_GettingStarted_Audio_aed_bm.hex` is
now flashed over all three regions. If it prints, our build is at fault. If it
stays silent, the fault is in the board or the boot chain, and nothing about our
model is implicated either way.

## RESULT: ST's prebuilt runs. Our build does not.

ST's `STM32N6_GettingStarted_Audio_aed_bm.hex`, flashed over all three regions
and booted from flash, prints continuously:

```
| 66      |  2.11%|  0.91|  1.19|  0.00|
| 67      |  2.11%|  0.91|  1.19|  0.00|
```

A frame counter, CPU load, and three per-stage times in milliseconds. The board,
the boot chain, the external loader, the signing chain and the flash procedure
are **all proven good**. Gate 3 has done its job: the fault is in our build, and
it was found with ST's own model in the loop, before a line of our code existed
to be blamed.

Note also what this says about the memory map — 2.11 % CPU and ~2.1 ms of work
per frame is ST's Yamnet-1024 AED model, not ours. It is a reference point, not
a target.

### The remaining variable is the compiler

| build | compiler | `.bin` size | runs? |
|---|---|---:|---|
| ours | GNU Tools for STM32 **14.3**.rel1 (CubeCLT) | 242,272 B | **no** |
| ours | Arm GNU **13.3**.Rel1, via a shim stripping `-fcyclomatic-complexity` | 243,616 B | untested |
| ST's prebuilt | unknown; ST support STM32CubeIDE v2.10.0 | 244,544 B | **yes** |

ST document STM32CubeIDE v2.10.0 as the supported toolchain (`README.md:88`).
Its bundled "GNU Tools for STM32" is a third lineage — neither the CubeCLT 14.3
we used nor vanilla Arm GNU 13.3. Only one GNU-Tools-for-STM32 is installed on
this machine.

### Next step is a debugger, not another compiler

Swapping toolchains and reflashing costs a boot-switch round trip per attempt and
answers only "did that one work". A debugger answers *where it stops*.

The application links with `STM32N657XX_LRUN.ld` — load-and-run — so it can be
loaded into RAM and started over SWD in development mode, which is how the zoo's
`n6_loader.py` drives this board. That allows iteration with no further switch
flips. `ST-LINK_gdbserver` is at
`/home/claroche/opt/st/stm32cubeclt_1.21.0/STLink-gdb-server/bin/`.

Plan: break on `SystemClock_Config_Full`, `MPU_Config`, `Ext_Mem_Config`,
`NPU_Config`, `IAC_Config` and `UART_Config` in order, and find the first that
does not return. `Ext_Mem_Config()` is the prior favourite, since it reconfigures
the xSPI interface the code is running from.

**That plan does not work, and the reason is worth recording.**
`ST-LINK_gdbserver` cannot halt this part in either boot-switch position:

```
Target not halted after reset. Force halt
Failed to halt target
Error in initializing ST-LINK device.  Reason: Target not halted.
```

Tried with and without `-k` / `--initialize-reset` (the server's equivalent of
`mode=UR`) and with `--halt`. Meanwhile `STM32_Programmer_CLI -c port=SWD
mode=UR` connects to the same board, reads external flash, and — in the
development switch position — erases and writes it. So the probe and the wiring
are fine; it is specifically the debugger's halt-on-reset handshake that fails.

Consequence: **no source-level debugging on this board yet**, and the toolchain
question falls back to one flash-and-boot per candidate compiler.

### Access matrix, measured

| operation | dev position | flash-boot position |
|---|---|---|
| `mode=HOTPLUG` connect | fails | fails |
| `mode=UR` connect | **works** | **works** |
| read external flash | **works** | **works** |
| erase / write external flash | **works** | fails |
| `ST-LINK_gdbserver` halt | fails | fails |

`mode=HOTPLUG` failing in *both* positions is itself notable — it worked before
the board was first booted from flash, so something about that transition
persists.

**The gdbserver flags above were wrong.** ST document the working invocation in
`README.md:178`:

```bash
ST-LINK_gdbserver -p 61234 -l 1 -d -s -cp <cubeprog-bin> -m 1 -g
```

`-m 1` selects **access port 1** — the same `ap=1` ST's flash script uses — and
`-g` **attaches to a running target** rather than trying to halt it. Neither
`--halt` nor `-k` appears. With ST's flags the server gets further but still
fails in the *flash-boot* position (`Target unknown error 32`); the documented
workflow requires the **development** position, which is the one thing not yet
tried.

## The compiler is NOT the cause

The Arm GNU 13.3 build, flashed alongside ST's known-good FSBL and weights and
booted from flash, is **also silent**. Two compilers from two different lineages,
same source, both dead; ST's prebuilt from that same source runs. That
eliminates the toolchain as the differentiator and moves suspicion to the build
path itself.

### The signed headers say the two images are not the same build

Extracting ST's prebuilt app payload from its hex and diffing against ours:

| header offset | ST prebuilt | ours (13.3) | meaning |
|---|---|---|---|
| `0x00` | `53544d32` | `53544d32` | `STM3` magic, same |
| `0x64` | `0166E84F` | `01677F6E` | checksum, expected to differ |
| `0x68` | `00020300` | `00020300` | header version 2.3, same |
| `0x6C` | `0003B900` (244,480) | `0003B7A0` (244,128) | image length |
| **`0x70`** | **`0x34016B19`** | **`0x34002F79`** | **entry point** |

Both link against `RAM : ORIGIN = 0x34000400`, so ST's reset handler sits
`0x16719` (≈91 KB) into the image where ours sits `0x2B79` (≈11 KB) in. Section
ordering under a different optimiser moves this by kilobytes, not by eighty of
them.

**Working hypothesis: ST's prebuilt binaries are built by STM32CubeIDE from
`Projects/GS/STM32CubeIDE/.cproject`, not by the Makefile.** The README presents
both paths as equals (`README.md:160-190`) but only the CubeIDE output is
shipped as a binary, and only the CubeIDE output is known to run. If the
Makefile path is simply not exercised by ST, its breakage would be invisible to
them.

The hypothesis was wrong in its blame but right in its clue. See below.

## ROOT CAUSE: the signing step omits `-align`

### The code was never broken

Under the documented dev-mode gdb workflow the application runs perfectly. Every
init function is reached, in order:

```
MPU_Config  →  Int_Mem_Config  →  Ext_Mem_Config  →  NPU_Config
            →  IAC_Config      →  UART_Config
```

Nothing hangs. `Ext_Mem_Config()`, the prior favourite, returns fine. So the
fault was never in the source, the compiler, or early init — it is in the
**boot image**, which is why it only manifests when the FSBL loads it and never
under a debugger that loads the ELF directly.

### The entry point in the header is wrong

| | value |
|---|---|
| vector table word[0] (initial SP) | `0x34100000` |
| vector table word[1] (`Reset_Handler`) | `0x340167BD` |
| ELF entry point (`readelf -h`) | `0x340167BD` |
| **signed header `+0x70`, what the FSBL jumps to** | **`0x34002F79`** |

`0x34002F79` is in the middle of `.text` (`0x34000750`–`0x3401ACC8`). The FSBL
jumps into the interior of a function and the part dies instantly — before
`UART_Config()`, hence not one byte of output.

### `-align` is the fix, and it was already in the fault atlas

`STM32_SigningTool_CLI --align`: *"Align the payload to the 0x400 offset by
adding padding bytes at the beginning of the payload."* Signing the identical
`.bin` four ways:

| flags | header entry | size |
|---|---|---:|
| ST's Makefile (`-s -bin -nk -t ssbl -hv 2.3`) | `0x34002F79` ✗ | 244,192 |
| **`… -align`** | **`0x340167BD`** ✓ | 244,640 |
| `… -ep 0x340167bd` | `0x340167BD` ✓ | 244,192 |
| `… -align -ep -la` | `0x340167BD` ✓ | 244,640 |

Without the 0x400 padding the payload begins at the wrong offset, and the
header's computed entry point is wrong by exactly that confusion.

**`bm.mk:39` does not pass `-align`.** ST's Makefile predates the requirement.
The zoo's own `config/toolchain.toml` already carried the knowledge —
*"STM32CubeProgrammer 2.21+ (2.21 introduced the mandatory `-align` on
signing)"* — and this machine runs **2.22.0**. The note existed; it simply was
not connected to a symptom that looks like a dead board.

That is the atlas earning its keep twice in one session: this entry, and the
`DEV_CONNECT_ERR` wedge below.

### The corrected signing command

```bash
STM32_SigningTool_CLI -s -bin GS_Audio_N6.bin -nk -t ssbl -hv 2.3 \
                      -align -o GS_Audio_N6_sign.bin
```

Flashed and verified by read-back: the board's header at `+0x70` now reads
`340167bd`.

**Verification gate for every future signing:** the word at `+0x70` of the
signed image must equal `Reset_Handler` from `nm`. Two commands, and it catches
silently-unbootable images before they cost a boot-switch round trip.

## The probe wedges after a gdb session — and detach/attach clears it

Immediately after killing `ST-LINK_gdbserver`, every flash attempt failed with
`ST-LINK error (DEV_CONNECT_ERR)`, three times running.

`zoo/faults/known_issues.toml:2079` describes exactly this — *"ST-LINK/usbip
link wedges after a killed inference, a gdb weight load, or a completed validate
run"* — and its workaround worked verbatim:

```bash
usbipd detach --busid 1-1 && usbipd attach --wsl --busid 1-1
```

The probe answered immediately afterwards. Note the atlas's caveat that a
*physical* replug is required when the wedge follows a `validate` run; after a
gdb session the software cycle is enough.

## Bench procedure: `mode=UR`, not `mode=HOTPLUG`

Once the board has been in boot-from-flash mode, `mode=HOTPLUG` fails with
`Unable to get core ID` / `No STM32 target found` **even after the switches are
returned to development position** — because the boot configuration is latched
at reset, and flipping a switch under power changes nothing until the next one.

`mode=UR` (connect under reset) asserts NRST and catches the part early:

```
STM32_Programmer_CLI -c port=SWD mode=UR --extload <MX66UW1G45G_STM32N6570-DK.stldr> ...
```

This is worth knowing for the zoo too, whose loader path assumes HOTPLUG. The
symptom looks exactly like the catalogue's wedged-probe entry — which calls for
a physical replug — but is a different fault with a software fix, and reaching
for the replug first would have wasted the trip.

## Switching back

`zoo measure` and `n6_loader.py` need development mode. Both switches go back
**right** for that. The two workflows want opposite switch positions, which is a
second reason — beyond the shared `0x70180000` — to keep them strictly
sequential rather than interleaved.

# Gate 3 — build and flash ST's unmodified audio app

**Status: built, signed and flashed. Verification blocked on a physical switch.**

Goal was to prove the toolchain, the boot chain and the board with ST's own
model in the loop, before any of our code is involved.

| step | result |
|---|---|
| 3.1 read OTP fuse state | **done** — word 124 = `0x00018000`, both bits already set. [`OTP.md`](OTP.md) |
| 3.2 decide on `fuse_vddio()` | **moot on this board** — the fuses it would program are already programmed |
| 3.3 `make bm` | **done** — but not with the toolchain we had pinned, see below |
| 3.4 sign + flash FSBL, app, weights | **done** — all three regions written |
| 3.5 confirm the UART banner | **blocked** — needs both boot switches LEFT and a power cycle |

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

### Switching back

`zoo measure` and `n6_loader.py` need development mode. Both switches go back
**right** for that. The two workflows want opposite switch positions, which is a
second reason — beyond the shared `0x70180000` — to keep them strictly
sequential rather than interleaved.

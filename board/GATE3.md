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

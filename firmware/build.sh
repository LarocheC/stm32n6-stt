#!/usr/bin/env bash
# Build and sign the application image.
#
#   firmware/build.sh [profile]
#
# profiles
#   demo     (default)  microphone + LCD + push-to-talk. What the board ships with.
#   wav                 replay the waveform blob at 0x72000000 through the real
#                       front end and the NPU, forever. No microphone, no LCD.
#   corpus              replay the 64-utterance FEATURE blob at 0x71000000
#                       straight to the NPU. This is what Gate 4 scored.
#   canned              one built-in feature tensor to the NPU, repeatedly. The
#                       smallest thing that exercises the whole runtime.
#
# Output: vendor/.../Projects/GS/BuildGCC/BM/GS_Audio_N6_sign.bin
# Then:   board/flash_and_verify.sh
set -euo pipefail

R="$(cd "$(dirname "$0")/.." && pwd)"
STT_QUIET=1 . "$R/env.sh"

PROFILE="${1:-demo}"
GS="$STT_GS"
L="$R/firmware/lcd"

[ -d "$GS" ] || { echo "vendored audio package missing at $GS -- see QUICKSTART.md section 1"; exit 1; }
[ -x "$ARM_BIN/arm-none-eabi-gcc" ] || { echo "arm-none-eabi-gcc not found in ARM_BIN=$ARM_BIN -- see QUICKSTART.md section 1"; exit 1; }

# Sources that live in this repository rather than in the vendored tree. They
# reach the build through the EXTRA_SOURCES hook that apply_vendor_mods.sh adds
# to the Makefile -- a bare -I is not enough, the .c files must be listed.
SRC="$R/firmware/src/citrinet_fe.c $R/firmware/src/citrinet_ctc.c"
INC="-I$R/firmware/inc"
DEF="-DCITRINET_FE_USE_CMSIS=1"

case "$PROFILE" in
  demo)
    DEF="$DEF -DGATE4_CANNED -DGATE5_WAV -DGATE5_MIC -DGATE7_LCD"
    SRC="$SRC $L/BSP/STM32N6570-DK/stm32n6570_discovery_lcd.c \
             $L/Utilities/lcd/stm32_lcd.c \
             ../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_ltdc.c \
             ../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_ltdc_ex.c \
             ../../Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_dma2d.c"
    INC="$INC -I$L/BSP/STM32N6570-DK -I$L/Utilities/lcd -I../../Drivers/BSP/Components/Common"
    ;;
  wav)    DEF="$DEF -DGATE4_CANNED -DGATE5_WAV" ;;
  corpus) DEF="$DEF -DGATE4_CANNED -DGATE4_CORPUS" ;;
  canned) DEF="$DEF -DGATE4_CANNED" ;;
  *) echo "unknown profile '$PROFILE'; try demo, wav, corpus or canned"; exit 1 ;;
esac

echo "profile   $PROFILE"
echo "defines   $DEF"

cd "$GS"
# Never set OPT= on the command line: that discards every "OPT +=" in the
# Makefile, including -flax-vector-conversions, which breaks CMSIS-DSP.
make bm -j"$(nproc)" \
  GCC_PATH="$ARM_BIN" \
  EXTRA_SOURCES="$SRC" \
  EXTRA_CFLAGS="$DEF $INC"

# -align is mandatory. Without it the payload does not land at file offset 0x400
# and the FSBL jumps through garbage -- board/GATE3.md, board/BUILD.md section 4.
[ -x "$STM32_SIGNING_CLI" ] || { echo "STM32_SigningTool_CLI not found -- set STM32CUBEPROG"; exit 1; }
"$STM32_SIGNING_CLI" -s -bin BuildGCC/BM/GS_Audio_N6.bin -nk -t ssbl -hv 2.3 -align \
                     -o BuildGCC/BM/GS_Audio_N6_sign.bin >/dev/null

echo
echo "signed: $GS/BuildGCC/BM/GS_Audio_N6_sign.bin ($(stat -c%s BuildGCC/BM/GS_Audio_N6_sign.bin) B)"
echo "flash with: board/flash_and_verify.sh"

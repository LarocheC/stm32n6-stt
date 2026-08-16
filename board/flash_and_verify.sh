#!/usr/bin/env bash
# Flash the current build and READ IT BACK, byte for byte.
#
# Every Gate 4 attempt so far checked only that 0x70100000 begins "STM2".
# That confirms a header is present; it does not confirm the image is intact.
# This flash has held a 714,560 B app in a 512 KB slot at least once, so the
# assumption that what we wrote is what is there has never actually been tested.
#
# Requires the DEVELOPMENT switch position (both RIGHT) and mode=UR.
# See board/GATE3.md for why HOTPLUG does not work here.
#
# Usage:  board/flash_and_verify.sh [weights.bin weights_addr]
set -euo pipefail

CP=/home/claroche/STMicroelectronics/STM32Cube/STM32CubeProgrammer
CLI=$CP/bin/STM32_Programmer_CLI
EL=$CP/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr
GS=/home/claroche/stm32n6-tts/vendor/STM32N6-GettingStarted-Audio/Projects/GS
APP=$GS/BuildGCC/BM/GS_Audio_N6_sign.bin
OUT=${TMPDIR:-/tmp}/gate4_readback

APP_ADDR=0x70100000
SLOT_END=0x70180000

[ -f "$APP" ] || { echo "no signed app at $APP -- build and sign first"; exit 1; }
mkdir -p "$OUT"

SIZE=$(stat -c%s "$APP")
printf 'app: %s\n     %d bytes (0x%x)\n' "$APP" "$SIZE" "$SIZE"

# The slot is 512 KB. Overflowing it silently overwrites whatever is at
# 0x70180000, which for the AED layout is the weight blob.
LIMIT=$(( SLOT_END - APP_ADDR ))
if [ "$SIZE" -gt "$LIMIT" ]; then
  echo "!! app is $SIZE B against a $LIMIT B slot -- it will run into $(printf 0x%x $SLOT_END)"
  echo "!! relocate the weights before flashing, or shrink the image"
  exit 1
fi
echo "     fits the $LIMIT B slot with $(( LIMIT - SIZE )) B to spare"

# Entry point must equal Reset_Handler|1, or the FSBL jumps through garbage.
# The FSBL actually takes the entry from the app's vector table at 0x34000404,
# but a mismatch here means the payload did not land at file offset 0x400 --
# which is the -align bug from Gate 3 (board/GATE3.md).
NM=/home/claroche/opt/st/stm32cubeclt_1.21.0/GNU-tools-for-STM32/bin/arm-none-eabi-nm
HDR=$(xxd -e -g4 -s 0x70 -l 4 "$APP" | awk '{print $2}')
RST=$($NM "${APP%_sign.bin}.elf" | awk '/ Reset_Handler/{print $1}' | sed 's/^0*//' \
      | awk '{printf "%08x\n", strtonum("0x"$1)+1}')
if [ "$HDR" != "$RST" ]; then
  echo "!! header entry $HDR != Reset_Handler|1 $RST -- re-sign with -align"
  exit 1
fi
echo "     entry point $HDR OK"

echo
echo "== writing app =="
$CLI -c port=SWD mode=UR --extload "$EL" -w "$APP" $APP_ADDR

if [ $# -ge 2 ]; then
  echo
  echo "== writing weights $1 -> $2 =="
  $CLI -c port=SWD mode=UR --extload "$EL" -w "$1" "$2"
fi

echo
echo "== reading the app region back =="
$CLI -c port=SWD mode=UR --extload "$EL" -r $APP_ADDR "$SIZE" "$OUT/app.bin"

echo
if cmp "$OUT/app.bin" "$APP"; then
  echo "VERIFIED: $SIZE bytes on the flash match the signed image byte for byte."
  echo "The image is intact. If it still does not boot, the fault is inside the"
  echo "application's own startup, not the flash or the FSBL."
else
  echo "MISMATCH -- this is the answer. First differing bytes:"
  cmp -l "$OUT/app.bin" "$APP" | head -20
  echo "readback kept at $OUT/app.bin"
fi

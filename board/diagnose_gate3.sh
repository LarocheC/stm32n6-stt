#!/bin/bash
# Gate 3 diagnostic. Run with BOTH boot switches in the RIGHT (development) position.
#
#   1. proves we can talk to the target at all
#   2. reads back the three flashed regions and compares them to what we wrote,
#      because "File download complete" is the programmer's claim, not evidence
#   3. flashes ST's own prebuilt combined hex using ST's exact invocation
#
# Then: both switches LEFT, power cycle, re-attach usbipd, read the UART.
set -u
V=/home/claroche/stm32n6-tts/vendor/STM32N6-GettingStarted-Audio
CP=/home/claroche/STMicroelectronics/STM32Cube/STM32CubeProgrammer
CLI=$CP/bin/STM32_Programmer_CLI
EL=$CP/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr
T=$(mktemp -d)
strip() { sed -e 's/\x1b\[[0-9;]*m//g'; }

echo "=============== 1. can we reach the target? ==============="
timeout 90 $CLI -c port=SWD mode=HOTPLUG ap=1 2>&1 | strip \
  | grep -iE "Device (ID|name)|core|Error|BL Version" | head -6
echo

echo "=============== 2. read back what we flashed ==============="
# ST's own invocation style: ap=1 + --extload, no -hardRst
for spec in "0x70000000:FSBL" "0x70100000:APP" "0x70180000:WEIGHTS"; do
  addr=${spec%%:*}; name=${spec##*:}
  timeout 120 $CLI -c port=SWD mode=HOTPLUG ap=1 --extload $EL \
      -r $addr 0x100 "$T/$name.bin" >/dev/null 2>&1
  if [ -s "$T/$name.bin" ]; then
    printf "%-8s @ %s  first 32 bytes: %s\n" "$name" "$addr" \
      "$(xxd -p -l 32 "$T/$name.bin" 2>/dev/null)"
  else
    printf "%-8s @ %s  READ FAILED\n" "$name" "$addr"
  fi
done
echo
echo "--- what SHOULD be at 0x70100000 (our signed app, first 32 bytes) ---"
xxd -p -l 32 $V/Projects/GS/BuildGCC/BM/GS_Audio_N6_sign.bin 2>/dev/null
echo "  (an STM32 header starts with the ASCII magic 'STM3' = 53544d32)"
echo

echo "=============== 3. flash ST's prebuilt, ST's way ==============="
HEX=$V/Binary/STM32N6570-DK/STM32N6_GettingStarted_Audio_aed_bm.hex
echo "writing $HEX"
timeout 900 $CLI -c port=swd mode=HOTPLUG ap=1 --extload $EL -w $HEX 2>&1 | strip \
  | grep -iE "Size|Address|download|error|complete|elapsed|segment" | head -20
echo
echo "=============================================================="
echo "Now: BOTH switches LEFT, then power-cycle the board."
echo "=============================================================="
rm -rf "$T"

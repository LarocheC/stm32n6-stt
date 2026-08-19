#!/usr/bin/env bash
# Halt a FLASH-BOOTED board and ask it where it is.
#
# Six rounds of Gate 4 eliminated everything that can be eliminated from the
# host: the boot chain is understood (board/GATE4.md rounds 6-7), and from the
# FSBL's jump to the first UART byte every instruction is byte-identical between
# the AED build that runs and the Citrinet build that says nothing. The only
# remaining information is inside the part, in the state that fails.
#
# This connects to a board that has already booted from flash and gone quiet,
# halts it, and reads the program counter plus the fault registers. That
# separates, in one shot:
#
#   PC in HardFault_Handler       -> CFSR/HFSR say why, MMFAR/BFAR say where
#   PC in Error_Handler           -> a HAL call failed; LR says which
#   PC in assert_failed / __BKPT  -> USE_FULL_ASSERT tripped before the UART existed
#   PC inside init_bm             -> spinning in a specific init step
#   PC in the gate4_canned loop   -> it IS running, and the fault is the UART
#
# Requires: BOOT-FROM-FLASH switch position (both LEFT), board already powered
# and given a few seconds to run. mode=UR is the connection that works on this
# part -- HOTPLUG fails in both switch positions (board/GATE3.md).
set -euo pipefail

# Repo root from this script's own location, the same way
# firmware/apply_vendor_mods.sh does it. Do not hardcode it -- the path
# that used to be here named a sibling directory that does not exist
# (this repository is stm32n6-stt).
R="$(cd "$(dirname "$0")/.." && pwd)"

CLT=/home/claroche/opt/st/stm32cubeclt_1.21.0
GDBSRV=$CLT/STLink-gdb-server/bin/ST-LINK_gdbserver
GDB=$CLT/GNU-tools-for-STM32/bin/arm-none-eabi-gdb
CPBIN=/home/claroche/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin
ELF=${1:-$R/vendor/STM32N6-GettingStarted-Audio/Projects/GS/BuildGCC/BM/GS_Audio_N6.elf}
PORT=61300
OUT=${TMPDIR:-/tmp}/gate4_probe
mkdir -p "$OUT"

[ -f "$ELF" ] || { echo "no elf at $ELF"; exit 1; }
echo "elf: $ELF"

pkill -f "ST-LINK_gdbserver.*$PORT" 2>/dev/null || true

# -m 1 -g is ST's documented invocation (vendor README.md:178). Do NOT pass
# --halt or -k: those produce "Target not halted" on this part.
"$GDBSRV" -m 1 -g -p $PORT -cp "$CPBIN" -d --attach > "$OUT/gdbserver.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for _ in $(seq 40); do
  grep -q "Waiting for connection" "$OUT/gdbserver.log" 2>/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "gdbserver died:"; cat "$OUT/gdbserver.log"; exit 1; }
  read -t 0.25 -u 3 _ 3</dev/null || true
done

cat > "$OUT/probe.gdb" <<'GDBEOF'
set pagination off
set confirm off
target extended-remote localhost:61300
interrupt
echo \n===== WHERE =====\n
info registers pc lr sp xpsr msp psp primask control
echo \n===== FRAME =====\n
frame
echo \n===== BACKTRACE =====\n
bt 20
echo \n===== FAULT STATUS =====\n
printf "CFSR  = 0x%08x\n", *(unsigned int*)0xE000ED28
printf "  MMFSR=0x%02x  BFSR=0x%02x  UFSR=0x%04x\n", \
       (*(unsigned int*)0xE000ED28)&0xff, \
       ((*(unsigned int*)0xE000ED28)>>8)&0xff, \
       ((*(unsigned int*)0xE000ED28)>>16)&0xffff
printf "HFSR  = 0x%08x\n", *(unsigned int*)0xE000ED2C
printf "MMFAR = 0x%08x\n", *(unsigned int*)0xE000ED34
printf "BFAR  = 0x%08x\n", *(unsigned int*)0xE000ED38
printf "VTOR  = 0x%08x   (expect 0x34000400)\n", *(unsigned int*)0xE000ED08
printf "SHCSR = 0x%08x\n", *(unsigned int*)0xE000ED24
echo \n===== DID THE FSBL LOAD US? =====\n
echo -- vector table at 0x34000400: word0 should be _estack, word1 Reset_Handler|1\n
x/2xw 0x34000400
echo -- signed header the FSBL copied to 0x34000000\n
x/4xw 0x3400006c
echo \n===== ADVANCE 3 s AND SEE IF IT MOVES =====\n
continue &
shell sleep 3
interrupt
info registers pc
echo \n===== END =====\n
detach
quit
GDBEOF

"$GDB" -q -batch -x "$OUT/probe.gdb" "$ELF" 2>&1 | tee "$OUT/probe.out"
echo
echo "saved: $OUT/probe.out   (gdbserver log: $OUT/gdbserver.log)"

#!/usr/bin/env python3
"""Read the DK's ST-LINK VCP and print what the app says.

The audio application prints at 14400 8N1 -- not a typo, and not a legacy
default. `app_config.h:58` sets it because 14400 is the maximum the UART can
hold across the DVFS clock changes the app makes at runtime.

    python board/read_uart.py            # 15 s at 14400
    python board/read_uart.py 30 921600  # seconds, baud

Prints nothing? The board is almost certainly in development boot mode. Both
boot switches must be LEFT and the board power-cycled for it to run from
external flash. See board/GATE3.md.
"""
import sys, time

try:
    import serial
except ImportError:
    sys.exit("pyserial missing: pip install pyserial  (see QUICKSTART.md)")

PORT = "/dev/ttyACM0"
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
baud = int(sys.argv[2]) if len(sys.argv) > 2 else 14400

try:
    s = serial.Serial(PORT, baud, timeout=1)
except Exception as e:
    sys.exit(f"cannot open {PORT} at {baud}: {e}")

s.reset_input_buffer()
buf, t0, last = b"", time.time(), 0
while time.time() - t0 < secs:
    n = s.in_waiting
    if n:
        buf += s.read(n)
        chunk = buf[last:].decode("utf-8", "replace")
        sys.stdout.write(chunk)
        sys.stdout.flush()
        last = len(buf)
    else:
        time.sleep(0.05)
s.close()

print(f"\n\n--- {len(buf)} bytes in {secs:g}s at {baud} baud ---")
if not buf:
    print("NOTHING RECEIVED. Both boot switches LEFT + power cycle, then retry.")

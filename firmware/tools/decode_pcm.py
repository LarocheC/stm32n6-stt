#!/usr/bin/env python3
"""Decode the '# PCM BEGIN ... # PCM END <fnv1a>' base64 block out of a UART
capture into a 16 kHz mono WAV, verifying the checksum the firmware printed."""
import base64, re, struct, sys, wave

def fnv1a(b):
    h = 2166136261
    for x in b:
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h

def main(log, out):
    raw = open(log, 'rb').read()
    txt = re.sub(rb'\x1b\[[0-9;]*[A-Za-z]', b'', raw).decode('utf-8', 'replace')
    m = re.search(r'#\s*PCM BEGIN utt (\d+) samples (\d+) rate (\d+) fmt s16le enc b64'
                  r'(.*?)#\s*PCM END utt \1 fnv1a ([0-9a-f]{8})', txt, re.S)
    if not m:
        raise SystemExit("no complete PCM block (missing END marker?)")
    utt, nsamp, rate, body, want = m.group(1), int(m.group(2)), int(m.group(3)), \
                                   m.group(4), m.group(5)
    b64 = re.sub(r'[^A-Za-z0-9+/=]', '', body)
    pcm = base64.b64decode(b64)
    got = f"{fnv1a(pcm):08x}"
    print(f"utterance {utt}: {len(pcm)} B decoded, expected {nsamp*2}")
    print(f"fnv1a device {want}  host {got}  -> {'MATCH' if got == want else 'MISMATCH'}")
    if len(pcm) != nsamp * 2:
        raise SystemExit("length mismatch -- the transfer lost bytes")
    if got != want:
        raise SystemExit("checksum mismatch -- the transfer corrupted bytes")
    with wave.open(out, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm)
    s = struct.unpack(f'<{nsamp}h', pcm)
    pk = max(abs(v) for v in s)
    print(f"wrote {out}: {nsamp} samples, {nsamp/rate:.2f} s, peak {pk} "
          f"({20*__import__('math').log10(pk/32768):.1f} dBFS)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

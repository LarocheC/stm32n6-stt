#!/usr/bin/env python3
"""Generate firmware/inc/citrinet_vocab.h from tokenizer/vocab.txt.

The Citrinet-256 CTC head emits 1025 classes: 1024 SentencePiece *unigram*
pieces (ids 0..1023) plus the CTC blank at id 1024.  The only non-ASCII
character anywhere in the vocabulary is U+2581 (LOWER ONE EIGHTH BLOCK, the
SentencePiece word-boundary marker), so no SentencePiece runtime and no UTF-8
handling are needed on the MCU: detokenisation is "concatenate the pieces and
turn U+2581 into a space".

This script does that substitution at generation time, so kPieces[] is pure
7-bit ASCII -- which is also all the ST Utilities/Fonts tables can render.

Layout emitted
--------------
    kPieces[]  : NUL-separated blob, piece i starts at kOffset[i]
    kOffset[]  : uint16_t[1026]; kOffset[1025] == sizeof(kPieces)
                 len(piece i) == kOffset[i+1] - kOffset[i] - 1   (the -1 is the NUL)

Because every piece is NUL-terminated in place, `&kPieces[kOffset[id]]` is a
plain C string and can be handed straight to strcat/UTIL_LCD_DisplayStringAt.

kPieces[] is emitted as a `char` array of character constants rather than as
one long string literal: C99 5.2.4.1 only requires an implementation to
support 4,095 characters in a string literal after concatenation, and the blob
is 6,170.  GCC accepts the literal form but `-Wpedantic` rejects it
(-Woverlength-strings), and MISRA C:2012 Rule 1.1 forbids exceeding a
translation limit.  The array form costs nothing at run time -- identical
.rodata, byte for byte.

Usage
-----
    python3 firmware/tools/gen_tokenizer.py                  # repo-root relative
    python3 firmware/tools/gen_tokenizer.py --check          # verify only, no write
    python3 firmware/tools/gen_tokenizer.py -i V -o H

--check validates tokenizer/vocab.txt, re-derives the header and diffs it
against the file already on disk, so it is a genuine regression check: it
fails if citrinet_vocab.h has been hand-edited or has drifted from the vocab.
"""

from __future__ import annotations

import argparse
import os
import sys

SPACE_MARKER = "▁"
BLANK_ID = 1024
UNK_ID = 0
EXPECTED_LINES = 1025

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_IN = os.path.join(_REPO, "tokenizer", "vocab.txt")
DEFAULT_OUT = os.path.join(_REPO, "firmware", "inc", "citrinet_vocab.h")


def load_vocab(path: str) -> list[str]:
    """Read '<piece> <id>' lines; assert ids are dense 0..N-1 in file order."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    lines = [l for l in raw.split("\n") if l != ""]
    if len(lines) != EXPECTED_LINES:
        raise SystemExit(
            f"{path}: expected {EXPECTED_LINES} non-empty lines, got {len(lines)}"
        )
    pieces: list[str] = []
    for i, line in enumerate(lines):
        if " " not in line:
            raise SystemExit(f"{path}:{i + 1}: no ' <id>' field: {line!r}")
        piece, sid = line.rsplit(" ", 1)
        if int(sid) != i:
            raise SystemExit(f"{path}:{i + 1}: id {sid} != line index {i}")
        if piece == "":
            raise SystemExit(f"{path}:{i + 1}: empty piece")
        pieces.append(piece)
    return pieces


def check_charset(pieces: list[str]) -> int:
    """Every piece must be ASCII apart from U+2581. Returns U+2581 piece count."""
    bad = []
    for i, p in enumerate(pieces):
        for ch in p:
            if ord(ch) > 0x7F and ch != SPACE_MARKER:
                bad.append((i, p, f"U+{ord(ch):04X}"))
    if bad:
        for i, p, cp in bad[:20]:
            print(f"  id {i}: {p!r} contains {cp}", file=sys.stderr)
        raise SystemExit(f"{len(bad)} non-ASCII characters other than U+2581")
    for i, p in enumerate(pieces):
        if " " in p:
            raise SystemExit(f"id {i}: piece {p!r} already contains a literal space")
        if "\0" in p:
            raise SystemExit(f"id {i}: piece {p!r} contains NUL")
    return sum(1 for p in pieces if SPACE_MARKER in p)


def c_char(ch: str) -> str:
    """One byte as a C character constant."""
    if ch == "\0":
        return "0"
    if ch == "\\":
        return "'\\\\'"
    if ch == "'":
        return "'\\''"
    if 0x20 <= ord(ch) <= 0x7E:
        return f"'{ch}'"
    raise SystemExit(f"unencodable char U+{ord(ch):04X}")


def c_comment(s: str) -> str:
    """Piece rendered safely inside a /* */ comment."""
    return s.replace("*/", "*_/")


def build(pieces: list[str]) -> tuple[list[str], list[int], int]:
    """ASCII-ise pieces and compute NUL-separated offsets."""
    ascii_pieces = [p.replace(SPACE_MARKER, " ") for p in pieces]
    offsets = [0]
    cur = 0
    for p in ascii_pieces:
        cur += len(p.encode("ascii")) + 1  # +1 for the NUL
        offsets.append(cur)
    return ascii_pieces, offsets, cur


def emit(pieces: list[str], ascii_pieces: list[str], offsets: list[int],
         blob_bytes: int, src: str) -> str:
    n = len(ascii_pieces)
    max_len = max(len(p) for p in ascii_pieces)
    n_marker = sum(1 for p in pieces if SPACE_MARKER in p)
    off_bytes = len(offsets) * 2
    rel_src = os.path.relpath(src, _REPO)

    L: list[str] = []
    a = L.append
    a("/* citrinet_vocab.h -- GENERATED, DO NOT EDIT.")
    a(" *")
    a(f" * Source : {rel_src} ({n} lines, '<piece> <id>' per line)")
    a(" * Tool   : firmware/tools/gen_tokenizer.py")
    a(" *")
    a(" * Vocabulary of nvidia/stt_en_citrinet_256_gamma_0_25 (SentencePiece unigram,")
    a(f" * 1024 pieces) plus the CTC blank at id {BLANK_ID}.")
    a(" *")
    a(f" * The SentencePiece word-boundary marker U+2581 (present in {n_marker} of {n}")
    a(" * pieces) has already been replaced by ASCII 0x20 here, so kPieces[] is pure")
    a(" * 7-bit ASCII and detokenisation is plain concatenation -- no UTF-8 decoding")
    a(" * and no SentencePiece runtime on the MCU.")
    a(" *")
    a(" * Flash cost:")
    a(f" *   kPieces  {blob_bytes:>6} B  (NUL-separated blob)")
    a(f" *   kOffset  {off_bytes:>6} B  (uint16_t x {len(offsets)})")
    a(f" *   total    {blob_bytes + off_bytes:>6} B  .rodata")
    a(" */")
    a("")
    a("#ifndef CITRINET_VOCAB_H")
    a("#define CITRINET_VOCAB_H")
    a("")
    a("#include <stdint.h>")
    a("")
    a(f"#define CITRINET_VOCAB_SIZE     {n}u   /* == stai output shape.data[1] */")
    a(f"#define CITRINET_BLANK_ID       {BLANK_ID}u")
    a(f"#define CITRINET_UNK_ID         {UNK_ID}u")
    a(f"#define CITRINET_PIECES_BYTES   {blob_bytes}u")
    a(f"#define CITRINET_MAX_PIECE_LEN  {max_len}u   /* excluding the NUL */")
    a("")
    a("/* Include this header from exactly ONE translation unit (the CTC decoder);")
    a("   the tables are file-static so a second includer would duplicate 8 KB. */")
    a("")
    a("/* NUL-separated pieces.  Emitted as character constants, not as a string")
    a("   literal: the blob exceeds C99's 4,095-character translation limit for a")
    a("   string literal after concatenation (5.2.4.1). */")
    a("static const char kPieces[CITRINET_PIECES_BYTES] = {")

    # Four pieces per line, each byte an explicit character constant and an
    # explicit 0 terminator, with the source pieces echoed in a comment.
    per_line = 4
    for start in range(0, n, per_line):
        chunk = ascii_pieces[start:start + per_line]
        body = " ".join("".join(c_char(ch) + "," for ch in p) + "0," for p in chunk)
        label = "|".join(c_comment(p) for p in chunk)
        a(f"  {body}  /* {start}..{start + len(chunk) - 1}  {label} */")
    a("};")
    a("")
    a(f"static const uint16_t kOffset[CITRINET_VOCAB_SIZE + 1] = {{")
    per_line = 12
    for start in range(0, len(offsets), per_line):
        row = ", ".join(f"{v:5d}" for v in offsets[start:start + per_line])
        a(f"  {row},")
    a("};")
    a("")
    a("/* Piece id -> NUL-terminated ASCII string. Valid for id < CITRINET_VOCAB_SIZE. */")
    a("static inline const char *citrinet_piece(uint32_t id)")
    a("{")
    a("  return &kPieces[kOffset[id]];")
    a("}")
    a("")
    a("/* Length in bytes, excluding the NUL. */")
    a("static inline uint32_t citrinet_piece_len(uint32_t id)")
    a("{")
    a("  return (uint32_t)(kOffset[id + 1u] - kOffset[id]) - 1u;")
    a("}")
    a("")
    a("#endif /* CITRINET_VOCAB_H */")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=DEFAULT_IN)
    ap.add_argument("-o", "--output", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="validate and print stats, write nothing")
    args = ap.parse_args()

    pieces = load_vocab(args.input)
    n_marker = check_charset(pieces)
    ascii_pieces, offsets, blob_bytes = build(pieces)

    assert offsets[-1] == blob_bytes
    assert len(offsets) == len(pieces) + 1
    if blob_bytes > 0xFFFF:
        raise SystemExit(f"blob is {blob_bytes} B; kOffset must widen to uint32_t")
    # round-trip: rebuild each piece from the blob the way C will
    blob = "".join(p + "\0" for p in ascii_pieces)
    for i, p in enumerate(ascii_pieces):
        got = blob[offsets[i]:offsets[i + 1] - 1]
        assert got == p, (i, got, p)
        assert blob[offsets[i + 1] - 1] == "\0", i

    off_bytes = len(offsets) * 2
    print(f"pieces                 : {len(pieces)}")
    print(f"contain U+2581         : {n_marker}")
    print(f"non-ASCII violations   : 0")
    print(f"longest piece          : {max(len(p) for p in ascii_pieces)} B")
    print(f"kPieces  (NUL-sep blob): {blob_bytes} B")
    print(f"kOffset  (uint16 x{len(offsets)}): {off_bytes} B")
    print(f"total .rodata          : {blob_bytes + off_bytes} B")

    text = emit(pieces, ascii_pieces, offsets, blob_bytes, args.input)

    if args.check:
        if not os.path.exists(args.output):
            print(f"CHECK FAILED: {args.output} does not exist", file=sys.stderr)
            return 1
        with open(args.output, encoding="ascii", newline="") as f:
            have = f.read()
        if have != text:
            print(f"CHECK FAILED: {args.output} differs from the regenerated "
                  f"header ({len(have)} B on disk vs {len(text)} B derived)",
                  file=sys.stderr)
            return 1
        print(f"check OK: {args.output} matches the vocabulary byte for byte")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    print(f"wrote {args.output} ({len(text)} B of C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

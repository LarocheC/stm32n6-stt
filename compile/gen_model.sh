#!/usr/bin/env bash
# Regenerate the Neural-ART deployment build, and KEEP THE WHOLE WORKSPACE.
#
# Rounds 11-17 of Gate 4 were run out of a /tmp scratchpad that has since been
# wiped. What went with it was not the eight files ST's own script copies --
# those are regenerable in one command -- but the compiler's intermediate
# workspace: `network_c_info.json` (the epoch/operator/streaming-engine table
# that localised both blockers, board/GATE4.md rounds 9-10 and 15-17),
# `<stem>_OE_3_3_1.onnx` (the preprocessed graph the `Conv2D_NNN` names refer
# to, board/GATE4.md:999), `<stem>_OE_3_3_1_Q.json` (the quantisation dump
# Round 17 named as the next thing to read, board/GATE4.md:1337),
# `network.csv` (the per-epoch memory traffic that proved hyperRAM untouched,
# compile/GATE2.md section 3, confirmation 3) and `network_atonbuf.xSPI2.raw`
# (the weight blob).
#
# So this script does not compile into a temporary directory and copy a
# shortlist out. It compiles *into* artifacts/compile/<tag>/ and leaves
# st_ai_output/ and st_ai_ws/ where they fell. That is the whole reason it
# exists.
#
#   compile/gen_model.sh <model.onnx> <tag> [options]
#
#   compile/gen_model.sh artifacts/onnx/q800_fold.onnx deploy
#   compile/gen_model.sh artifacts/onnx/q800_fold.onnx capipe1 --ca-pipe 1
#   compile/gen_model.sh artifacts/onnx/q800_real.onnx  st-stock --mpool st
#
# The graph the board needs is the FOLDED one -- q800_real.onnx still contains
# the three stride-2 depthwise convolutions that stall the NPU (board/GATE4.md
# Round 12). model/fold_stride2.py produces artifacts/onnx/q800_fold.onnx from
# it. This script does not care which it is handed; it only refuses an input
# that stedgeai has already preprocessed.
#
# Nothing here touches the board and nothing here writes into vendor/.
# Installing the result into the ST application is firmware/apply_vendor_mods.sh
# step 6, which reads artifacts/model_c/ -- pass --install to refresh it.
#
# Exit status: 0 all deployment checks pass; 1 bad usage or the compile failed;
# 2 the compile succeeded but a deployment check failed. The workspace is
# preserved in every case, including 2 -- a failing experiment is evidence.

set -euo pipefail

# ---------------------------------------------------------------- toolchain
# ST Edge AI Core 4.0.1 EXPLICITLY. 3.0 is also installed under the same
# prefix and picking it up would silently change the compiler (1.1.3-275 ->
# whatever 3.0 ships) and the ll_aton the firmware links against, which
# network.c enforces with a hard #error -- see firmware/apply_vendor_mods.sh
# step 1.
STEDGEAI_ROOT=/home/claroche/stedgeai/install/4.0
STEDGEAI="$STEDGEAI_ROOT/Utilities/linux/stedgeai"
STEDGEAI_EXPECT="v4.0.1-20581"
ARM_BIN=/home/claroche/opt/st/stm32cubeclt_1.21.0/GNU-tools-for-STM32/bin
OBJCOPY="$ARM_BIN/arm-none-eabi-objcopy"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------- defaults
#
# MPOOL. compile/st_audio.mpool is ST's stm32n6.mpool verbatim (md5
# 3962913c702e781e24ba6bbe431ee10c, compile/GATE2.md section 7) and puts
# octoFlash at 0x70180000, which leaves the application the 512 kB between
# 0x70100000 and the weight blob. The Citrinet signed image is 727,168 B and
# does not fit that: board/flash_and_verify.sh refuses it against a 512 kB slot
# and, before that check existed, it silently overwrote the weights.
# compile/stt_audio.mpool is the same file with the octoFlash pool moved to
# 0x70400000 / 60 MB, which gives the application 3 MB. Every board run from
# Round 13 on used that layout (board/GATE4.md:1225, "weights at 0x70400000"),
# so it is the default here.
MPOOL="$REPO/compile/stt_audio.mpool"

# OPTIONS. ST's audio application ships this exact string in
# Projects/X-CUBE-AI/models/user_neural_art.json:
#
#   --Ocache-opt -O3 --Os --native-float --Omax-ca-pipe 4 --cache-maintenance \
#   --csv-file network --all-buffers-info
#
# compile/DECISION-oauto-sched.md adopts it with --Oauto-sched appended: on the
# 8 s graph that is 200 kB of cpuRAM2 instead of 500 kB, 628 epochs instead of
# 618, 91.2 ms instead of 91.9 ms, 0 SW and 0 hybrid either way -- smaller and
# faster at once, with no axis on which ST's shipped set wins. Regression
# constant recorded there: 628 epochs / 0 SW / 0 hybrid / 625 kB.
#
# --Omax-ca-pipe is spliced in from $CA_PIPE rather than written literally,
# because Round 17 (board/GATE4.md:1303) changed exactly that one value to get
# a different schedule (629 epoch blocks -> 617) and prove the second blocker
# follows the operator rather than the schedule. Appending a second
# --Omax-ca-pipe would leave which one wins to atonn; substituting leaves
# nothing to guess. At CA_PIPE=4 the string below is byte-for-byte ST's plus
# --Oauto-sched, which is what compile/reports/g800_stt_0x70400000/summary.txt
# records as actually passed to the compiler.
CA_PIPE=4
ATONN_EXTRA=""

PROFILE_NAME=default          # ST's own profile name in user_neural_art.json
PROFILE_FILE=""               # set by --profile-file to use a checked-in profile verbatim
FORCE=0
INSTALL=0
ALLOW_OE=0
MAKE_HEX=1

usage() {
  sed -n '4,39p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

Options
  --mpool PATH|stt|st|strict   memory pool. stt = compile/stt_audio.mpool
                               (octoFlash 0x70400000, 3 MB app slot, DEFAULT);
                               st = compile/st_audio.mpool (ST stock,
                               0x70180000, 512 kB app slot);
                               strict = compile/audio_strict.mpool (screening
                               geometry, octoFlash 0x71000000, no hyperRAM)
  --ca-pipe N                  value for --Omax-ca-pipe (default 4; Round 17
                               used 1 to force a different schedule)
  --atonn "OPTS"               extra atonn options, appended verbatim
  --profile-file FILE          use a checked-in profile JSON as-is instead of
                               synthesising one; refuses --mpool/--ca-pipe/--atonn
  --profile NAME               profile name inside that JSON (default: default)
  --out DIR                    tag directory (default artifacts/compile/<tag>)
  --install                    also refresh artifacts/model_c/, which
                               firmware/apply_vendor_mods.sh step 6 reads
  --no-hex                     skip network_data.hex (~29 MB of ihex)
  --allow-oe                   permit an already-preprocessed *_OE_*.onnx input
  --force                      overwrite an existing tag directory
USAGE
}

die() { printf 'gen_model: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- arguments
[ $# -ge 1 ] || { usage; exit 1; }
case "$1" in -h|--help) usage; exit 0 ;; esac
[ $# -ge 2 ] || { usage; exit 1; }

MODEL_IN="$1"; TAG="$2"; shift 2
TAGDIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --mpool)
      case "${2:-}" in
        stt)    MPOOL="$REPO/compile/stt_audio.mpool" ;;
        st)     MPOOL="$REPO/compile/st_audio.mpool" ;;
        strict) MPOOL="$REPO/compile/audio_strict.mpool" ;;
        "")     die "--mpool needs a value" ;;
        *)      MPOOL="$2" ;;
      esac; shift 2 ;;
    --ca-pipe)      CA_PIPE="${2:?--ca-pipe needs a value}"; shift 2 ;;
    --atonn)        ATONN_EXTRA="${2:?--atonn needs a value}"; shift 2 ;;
    --profile-file) PROFILE_FILE="${2:?--profile-file needs a value}"; shift 2 ;;
    --profile)      PROFILE_NAME="${2:?--profile needs a value}"; shift 2 ;;
    --out)          TAGDIR="${2:?--out needs a value}"; shift 2 ;;
    --install)      INSTALL=1; shift ;;
    --no-hex)       MAKE_HEX=0; shift ;;
    --allow-oe)     ALLOW_OE=1; shift ;;
    --force)        FORCE=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *)              die "unknown option '$1' (try --help)" ;;
  esac
done

case "$TAG" in ""|*/*) die "tag must be a single path component, got '$TAG'" ;; esac
[ -n "$TAGDIR" ] || TAGDIR="$REPO/artifacts/compile/$TAG"

# ---------------------------------------------------------------- preflight
[ -x "$STEDGEAI" ] || die "no ST Edge AI Core at $STEDGEAI"
VER="$("$STEDGEAI" --version 2>&1 | head -1)"
case "$VER" in
  *"$STEDGEAI_EXPECT"*) : ;;
  *) die "expected ST Edge AI Core $STEDGEAI_EXPECT, got: $VER" ;;
esac
[ -f "$MODEL_IN" ] || die "no such model: $MODEL_IN"
MODEL="$(cd "$(dirname "$MODEL_IN")" && pwd)/$(basename "$MODEL_IN")"
[ -f "$MPOOL" ]  || die "no such mpool: $MPOOL"
if [ "$MAKE_HEX" = 1 ]; then
  [ -x "$OBJCOPY" ] || die "no arm-none-eabi-objcopy at $OBJCOPY (or pass --no-hex)"
fi

# Feeding an already-preprocessed graph back to `stedgeai generate` makes it
# preprocess a second time. That produced a bogus 84-epoch / 57-SW-epoch result
# once already and was only caught by a control run -- board/GATE4.md:832.
case "$(basename "$MODEL")" in
  *_OE_*.onnx)
    [ "$ALLOW_OE" = 1 ] || die "$(basename "$MODEL") looks already preprocessed by stedgeai.
  Feeding a *_OE_*.onnx back into 'generate' preprocesses it twice and gives
  wrong epoch counts (board/GATE4.md:832). Pass --allow-oe if this is deliberate." ;;
esac

if [ -n "$PROFILE_FILE" ]; then
  [ -f "$PROFILE_FILE" ] || die "no such profile file: $PROFILE_FILE"
  PROFILE_FILE="$(cd "$(dirname "$PROFILE_FILE")" && pwd)/$(basename "$PROFILE_FILE")"
  [ "$CA_PIPE" = 4 ] && [ -z "$ATONN_EXTRA" ] \
    || die "--profile-file uses the file's option string verbatim; drop --ca-pipe/--atonn"
fi

if [ -e "$TAGDIR" ] && [ -n "$(ls -A "$TAGDIR" 2>/dev/null)" ]; then
  [ "$FORCE" = 1 ] || die "tag '$TAG' already exists at $TAGDIR -- pass --force to replace it"
  rm -rf "$TAGDIR"
fi
mkdir -p "$TAGDIR"

# --------------------------------------------------- mpool -> weights base
# ST's generate-n6-model.sh hardcodes `--change-addresses 0x70180000`, which is
# correct only for its own mpool. The hex base has to be whatever the mpool
# says octoFlash (xSPI2) is, or the blob is addressed somewhere the flash
# writes do not land -- the exact failure compile/GATE2.md section 2 was
# written to catch. Read it out of the mpool instead of restating it.
read -r WEIGHTS_BASE FLASH_SIZE <<EOF
$(python3 - "$MPOOL" <<'PY'
import json, sys
pools = json.load(open(sys.argv[1]))["memory"]["mempools"]
for p in pools:
    if p["fname"] == "xSPI2":
        print(p["offset"]["value"], p["size"]["value"] + " " + p["size"]["magnitude"])
        break
else:
    sys.exit("no xSPI2 (octoFlash) pool in the mpool")
PY
)
EOF
[ -n "$WEIGHTS_BASE" ] || die "could not read the xSPI2 base out of $MPOOL"

# ---------------------------------------------------------------- profile
# The tag directory gets its own copy of the mpool and its own profile JSON,
# both with absolute paths, so it is self-describing after the fact and does
# not depend on any cwd. Absolute memory_pool paths are supported: ST's own
# resources use them and the deployment zoo relies on it.
cp "$MPOOL" "$TAGDIR/$(basename "$MPOOL")"
if [ -n "$PROFILE_FILE" ]; then
  cp "$PROFILE_FILE" "$TAGDIR/neural_art.json"
  OPTIONS="(from $PROFILE_FILE)"
else
  OPTIONS="--Ocache-opt -O3 --Os --native-float --Omax-ca-pipe $CA_PIPE --cache-maintenance --csv-file network --all-buffers-info --Oauto-sched"
  [ -z "$ATONN_EXTRA" ] || OPTIONS="$OPTIONS $ATONN_EXTRA"
  python3 - "$TAGDIR/neural_art.json" "$PROFILE_NAME" \
           "$TAGDIR/$(basename "$MPOOL")" "$OPTIONS" <<'PY'
import json, sys
out, name, mpool, options = sys.argv[1:5]
json.dump({"Globals": {}, "Profiles": {name: {"memory_pool": mpool,
                                              "options": options}}},
          open(out, "w"), indent=4)
open(out, "a").write("\n")
PY
fi

# ---------------------------------------------------------------- compile
# ST's line is
#   $generateCmd generate -m $1 --target stm32n6 --st-neural-art default@user_neural_art.json
# and it relies on the defaults for everything else: --name defaults to
# "network" (`stedgeai generate -h`), --output to ./st_ai_output and
# --workspace to ./st_ai_ws, both relative to the cwd. Running from inside the
# tag directory therefore lands the entire workspace in the tag directory --
# including anything atonn drops relative to its own cwd, such as the
# --csv-file output -- with no copy step that can be forgotten.
CMD=( "$STEDGEAI" generate -m "$MODEL" --target stm32n6
      --st-neural-art "$PROFILE_NAME@$TAGDIR/neural_art.json" )

{ printf '%q ' "${CMD[@]}"; printf '\n# cwd: %s\n' "$TAGDIR"; } > "$TAGDIR/cmd.txt"

echo "model    $MODEL"
echo "tag      $TAG  ->  $TAGDIR"
echo "mpool    $MPOOL"
echo "octoFlash base $WEIGHTS_BASE  size $FLASH_SIZE"
echo "options  $OPTIONS"
echo "tool     $VER"
echo
cat "$TAGDIR/cmd.txt"
echo

cd "$TAGDIR"
set +e
"${CMD[@]}" 2>&1 | tee "$TAGDIR/gen.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" = 0 ] || die "stedgeai exited $RC -- log and partial workspace kept at $TAGDIR"

OUT="$TAGDIR/st_ai_output"
WS="$TAGDIR/st_ai_ws"
REPORT="$OUT/network_generate_report.txt"
[ -f "$REPORT" ] || die "no $REPORT -- the compile did not produce a report"

# ------------------------------------------------- the ST drop-point files
# firmware/WORKLIST.md section 4.0: generate-n6-model.sh copies six files up one
# level, renames network_atonbuf.xSPI2.raw to network_data.bin, and objcopies
# that to an ihex based at the octoFlash offset. Eight artefacts. They are
# assembled in drop/ rather than at the top of the tag so that the preserved
# workspace stays exactly as the compiler left it.
DROP="$TAGDIR/drop"
mkdir -p "$DROP"
for f in network.c network.h stai_network.c stai_network.h \
         network_c_info.json network_generate_report.txt; do
  [ -f "$OUT/$f" ] || die "expected $OUT/$f"
  cp "$OUT/$f" "$DROP/$f"
done
[ -f "$OUT/network_atonbuf.xSPI2.raw" ] || die "expected $OUT/network_atonbuf.xSPI2.raw"
cp "$OUT/network_atonbuf.xSPI2.raw" "$DROP/network_data.bin"
if [ "$MAKE_HEX" = 1 ]; then
  "$OBJCOPY" -I binary "$DROP/network_data.bin" \
             --change-addresses "$WEIGHTS_BASE" -O ihex "$DROP/network_data.hex"
fi

if [ "$INSTALL" = 1 ]; then
  MC="$REPO/artifacts/model_c"
  mkdir -p "$MC"
  for f in network.c network.h stai_network.c stai_network.h \
           network_atonbuf.xSPI2.raw; do
    cp "$OUT/$f" "$MC/$f"
  done
  echo "installed into $MC (firmware/apply_vendor_mods.sh step 6 reads it)"
fi

# ---------------------------------------------------------------- the gate
echo
echo "================ the numbers that decide the gate ================"
sed -n '/^Memory usage information/,/^====/p' "$REPORT" | grep -E '\[0x[0-9a-fA-F]+ - 0x' || true
echo "---"
grep -E '^Total number of epochs|^>> pure software|^>> hybrid epochs|^>> pure hardware' "$REPORT" || true
echo "---"
sed -n '/^Used memory ranges/,/^====/p' "$REPORT" | grep -E '\[0x' || true

# Deployment checks. compile/GATE2.md section 6 asks for exactly these to be a
# build-time regression check rather than eyeballed, because section 5 found a
# 12 s compile that reads 0 SW / 0 hybrid while quietly holding 150 kB of
# activations on a 2-byte-wide PSRAM bus.
echo
echo "================ deployment checks ================"
FAILED=0
chk() { # chk <label> <ok?> <detail>
  if [ "$2" = 1 ]; then printf '  ok    %s -- %s\n' "$1" "$3"
  else printf '  FAIL  %s -- %s\n' "$1" "$3"; FAILED=$((FAILED+1)); fi
}

SW=$(grep -E '^>> pure software' "$REPORT" | grep -oE '[0-9]+$' | head -1)
HY=$(grep -E '^>> hybrid epochs' "$REPORT" | grep -oE '[0-9]+$' | head -1)
chk "0 software epochs" "$([ "${SW:-x}" = 0 ] && echo 1 || echo 0)" "pure SW epochs = ${SW:-unread}"
chk "0 hybrid epochs"   "$([ "${HY:-x}" = 0 ] && echo 1 || echo 0)" "hybrid epochs = ${HY:-unread}"

HYPER=$(sed -n '/^Memory usage information/,/^====/p' "$REPORT" | grep -E '^\s+hyperRAM' || true)
if [ -n "$HYPER" ]; then
  case "$HYPER" in
    *": "*"0  B /"*) chk "0 B in hyperRAM" 1 "activations stay on-chip" ;;
    *)               chk "0 B in hyperRAM" 0 "$(echo "$HYPER" | sed 's/^[[:space:]]*//')" ;;
  esac
else
  chk "0 B in hyperRAM" 1 "no hyperRAM pool in this mpool"
fi

# Independent of the report line: a spill would put bytes in the xSPI1 blob.
X1="$OUT/network_atonbuf.xSPI1.raw"
if [ -f "$X1" ]; then
  S1=$(stat -c%s "$X1")
  chk "xSPI1 weight/activation blob empty" "$([ "$S1" = 0 ] && echo 1 || echo 0)" "$X1 is $S1 B"
fi

FBASE=$(sed -n '/^Memory usage information/,/^====/p' "$REPORT" \
        | grep -E '^\s+octoFlash' | grep -oE '0x[0-9a-fA-F]+' | head -1)
chk "octoFlash base matches the mpool" \
    "$([ "$(printf '%d' "${FBASE:-0}")" = "$(printf '%d' "$WEIGHTS_BASE")" ] && echo 1 || echo 0)" \
    "report ${FBASE:-unread}, mpool $WEIGHTS_BASE, hex based at $WEIGHTS_BASE"

# ------------------------------------------------------------- what is kept
echo
echo "================ preserved at $TAGDIR ================"
show() { [ -e "$1" ] && printf '  %10s  %s\n' "$(stat -c%s "$1")" "${1#$TAGDIR/}" || true; }
show "$TAGDIR/cmd.txt"
show "$TAGDIR/neural_art.json"
show "$TAGDIR/$(basename "$MPOOL")"
show "$TAGDIR/gen.log"
show "$OUT/network_generate_report.txt"
show "$OUT/network_c_info.json"
for f in "$OUT"/*_OE_*.onnx "$OUT"/*_OE_*_Q.json; do show "$f"; done
show "$OUT/network_atonbuf.xSPI2.raw"
show "$WS/neural_art__network/network.csv"
show "$WS/neural_art__network/c_info.json"
show "$WS/neural_art__network/atonn_options.ini"
echo "  ...plus the complete st_ai_output/ ($(find "$OUT" -type f | wc -l) files) and"
echo "     st_ai_ws/ ($(find "$WS" -type f | wc -l) files), left as the compiler wrote them."
echo "  drop/ holds the eight files generate-n6-model.sh copies to the ST app."

if [ "$FAILED" != 0 ]; then
  echo
  echo "$FAILED deployment check(s) failed. The workspace above is kept regardless."
  exit 2
fi

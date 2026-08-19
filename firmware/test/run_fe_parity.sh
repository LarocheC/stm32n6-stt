#!/usr/bin/env bash
# Compile firmware/src/citrinet_fe.c natively, four ways, and check every one of
# them against model/fe.py.  No board, no ARM toolchain.
#
#   bash firmware/test/run_fe_parity.sh [N_UTTERANCES]
#
# Needs the zoo venv for numpy/librosa/scipy/soundfile:
#   source ~/.venvs/stm32n6-stt/bin/activate     (see QUICKSTART.md)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CMSIS="$REPO/vendor/STM32N6-GettingStarted-Audio/Drivers/CMSIS"
BUILD="${BUILD:-$REPO/artifacts/fe_host}"
N="${1:-5}"

mkdir -p "$BUILD" "$REPO/firmware/test/results"
CC="${CC:-gcc}"
BASE="-O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -I$REPO/firmware/inc"
SRC="$REPO/firmware/src/citrinet_fe.c $REPO/firmware/test/test_fe_host.c"

# CMSIS-DSP builds for the host unchanged: __GNUC_PYTHON__ is the escape hatch
# arm_math_types.h already provides for non-Cortex builds (arm_math_types.h:75).
# This is the same scalar C that runs on the M55 when Helium is off.
CMSIS_SRC="
  $CMSIS/DSP/Source/TransformFunctions/arm_rfft_fast_f32.c
  $CMSIS/DSP/Source/TransformFunctions/arm_rfft_fast_init_f32.c
  $CMSIS/DSP/Source/TransformFunctions/arm_cfft_init_f32.c
  $CMSIS/DSP/Source/TransformFunctions/arm_cfft_f32.c
  $CMSIS/DSP/Source/TransformFunctions/arm_cfft_radix8_f32.c
  $CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.c
  $CMSIS/DSP/Source/CommonTables/arm_common_tables.c
  $CMSIS/DSP/Source/CommonTables/arm_const_structs.c"
CMSIS_FLAGS="-D__GNUC_PYTHON__ -DCITRINET_FE_USE_CMSIS=1 -I$CMSIS/DSP/Include -I$CMSIS/DSP/PrivateInclude"

echo "[build] portable FFT, float32 scratch"
$CC $BASE -DCITRINET_FE_USE_CMSIS=0 $SRC -lm -o "$BUILD/fe_portable"

echo "[build] CMSIS arm_rfft_fast_f32, float32 scratch"
# shellcheck disable=SC2086
$CC $BASE $CMSIS_FLAGS $SRC $CMSIS_SRC -lm -o "$BUILD/fe_cmsis"

echo "[build] CMSIS, fp16 scratch (128,000 B instead of 256,000 B)"
# shellcheck disable=SC2086
$CC $BASE $CMSIS_FLAGS -DCITRINET_FE_SCRATCH_F16=1 $SRC $CMSIS_SRC -lm -o "$BUILD/fe_cmsis_f16"

echo "[build] CMSIS, naive sum/sumsq variance, double accumulators"
# shellcheck disable=SC2086
$CC $BASE $CMSIS_FLAGS -DCITRINET_FE_VAR_MODE=1 $SRC $CMSIS_SRC -lm -o "$BUILD/fe_var_sumsq_f64"

echo "[build] CMSIS, naive sum/sumsq variance, float32 accumulators"
# shellcheck disable=SC2086
$CC $BASE $CMSIS_FLAGS -DCITRINET_FE_VAR_MODE=2 $SRC $CMSIS_SRC -lm -o "$BUILD/fe_var_sumsq_f32"

echo
python "$REPO/firmware/tools/gen_mel_tables.py" --check

python "$REPO/firmware/test/fe_parity.py" \
    --bin "$BUILD/fe_cmsis" \
    --bin "$BUILD/fe_portable" \
    --bin "$BUILD/fe_cmsis_f16" \
    --bin "$BUILD/fe_var_sumsq_f64" \
    --bin "$BUILD/fe_var_sumsq_f32" \
    --n "$N" --json "$REPO/firmware/test/results/fe_parity.json"

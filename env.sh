#!/usr/bin/env bash
# Where the external tools live.  Sourced by every script in this repository.
#
# Nothing here is guessed at run time by the scripts themselves: they source this
# file, and this file takes each location from the environment if it is already
# set, otherwise tries a short list of usual places, otherwise leaves the variable
# empty so the caller can fail with a message that names what is missing.
#
#   source env.sh                 # defaults, or whatever is already exported
#   STEDGEAI_ROOT=/opt/stedgeai/4.0 source env.sh
#
# Override by exporting before sourcing, or by editing the defaults below.

# --- helper: first existing directory from a list -----------------------------
_stt_first_dir() { for d in "$@"; do [ -d "$d" ] && { printf '%s' "$d"; return 0; }; done; return 1; }

# --- repository root ----------------------------------------------------------
STT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export STT_ROOT

# --- STM32CubeProgrammer ------------------------------------------------------
# Supplies STM32_Programmer_CLI, STM32_SigningTool_CLI and the MX66UW1G45G
# external loader.  Required to flash.  Free from st.com.
: "${STM32CUBEPROG:=$(_stt_first_dir \
      "$HOME/STMicroelectronics/STM32Cube/STM32CubeProgrammer" \
      "/opt/stm32cubeprog" \
      "/usr/local/STMicroelectronics/STM32Cube/STM32CubeProgrammer")}"
export STM32CUBEPROG
export STM32_PROGRAMMER_CLI="${STM32CUBEPROG:+$STM32CUBEPROG/bin/STM32_Programmer_CLI}"
export STM32_SIGNING_CLI="${STM32CUBEPROG:+$STM32CUBEPROG/bin/STM32_SigningTool_CLI}"
export STM32_EXTLOADER="${STM32CUBEPROG:+$STM32CUBEPROG/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr}"

# --- Arm bare-metal toolchain -------------------------------------------------
# Anything providing arm-none-eabi-gcc will do; STM32CubeCLT ships one.
# board/GATE3.md notes the local build used 10.3.1 while ST's own binaries were
# built with 13.3.1 -- if anything smells, that is the first difference to try.
: "${ARM_BIN:=$(_stt_first_dir \
      "$HOME/opt/st/stm32cubeclt_1.21.0/GNU-tools-for-STM32/bin" \
      "/opt/st/stm32cubeclt/GNU-tools-for-STM32/bin" \
      "/usr/bin")}"
export ARM_BIN
export GCC_PATH="$ARM_BIN"          # the vendored Makefile's own variable name

# --- ST Edge AI Core ----------------------------------------------------------
# Only needed to RECOMPILE the network (compile/gen_model.sh).  Flashing a
# prebuilt artifacts/ tree does not need it.  Version matters: this project is
# pinned to 4.0.1-20581, see compile/GATE2.md.
: "${STEDGEAI_ROOT:=$(_stt_first_dir \
      "$HOME/stedgeai/install/4.0" \
      "/opt/stedgeai/4.0" \
      "$HOME/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.0.0/Utilities/linux")}"
export STEDGEAI_ROOT
export STEDGEAI="${STEDGEAI_ROOT:+$STEDGEAI_ROOT/Utilities/linux/stedgeai}"

# --- Python -------------------------------------------------------------------
# The host tooling needs numpy, onnx, onnxruntime, librosa, soundfile, pyserial.
# Point STT_PYTHON at an interpreter that has them; see QUICKSTART.md.
: "${STT_PYTHON:=$(command -v python3 || true)}"
export STT_PYTHON

# --- vendored ST application packages ----------------------------------------
# Not in git (they are ST's, and large).  QUICKSTART.md §1 says where to get them.
export STT_AUDIO_PKG="$STT_ROOT/vendor/STM32N6-GettingStarted-Audio"
export STT_OD_PKG="$STT_ROOT/vendor/STM32N6-GettingStarted-ObjectDetection"
export STT_GS="$STT_AUDIO_PKG/Projects/GS"

# --- one-line report, unless STT_QUIET is set --------------------------------
if [ -z "${STT_QUIET:-}" ]; then
  _stt_show() { printf '  %-18s %s\n' "$1" "${2:-<not found>}"; }
  echo "stm32n6-stt tool locations (override by exporting before sourcing env.sh):"
  _stt_show STM32CUBEPROG "$STM32CUBEPROG"
  _stt_show ARM_BIN       "$ARM_BIN"
  _stt_show STEDGEAI_ROOT "$STEDGEAI_ROOT"
  _stt_show STT_PYTHON    "$STT_PYTHON"
  _stt_show "audio package" "$([ -d "$STT_AUDIO_PKG" ] && echo "$STT_AUDIO_PKG" || echo '<missing -- QUICKSTART.md section 1>')"
fi

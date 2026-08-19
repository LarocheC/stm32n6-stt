# Gate 6 host tests for the CTC decoder.
#
#   make selftest    build natively, run the argument/capacity contract tests
#   make oracle      the full ONNX-Runtime-vs-fe.py comparison (needs the venv)
#   make arm         cross-compile the decoder and report its flash cost
#
# The decoder itself is target code; this Makefile only builds it for the host
# so it can be diffed against the Python oracle.

REPO    := $(abspath $(CURDIR)/../..)
INC     := $(REPO)/firmware/inc
SRC     := $(REPO)/firmware/src/citrinet_ctc.c
BUILD   ?= $(CURDIR)/build

CFLAGS  := -std=c99 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -I$(INC)

ARMCC   ?= arm-none-eabi-gcc
ARMSIZE ?= arm-none-eabi-size
ARMFLAGS := -mcpu=cortex-m55 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -Os -std=c99 \
            -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror \
            -ffunction-sections -fdata-sections -I$(INC)

# Any interpreter with numpy/librosa/soundfile/onnxruntime. See QUICKSTART.md.
VENV    ?= $(HOME)/.venvs/stm32n6-stt

.PHONY: all selftest oracle arm vocab clean

all: selftest

$(BUILD)/ctc_host_test: $(SRC) ctc_host_test.c $(INC)/citrinet_ctc.h $(INC)/citrinet_vocab.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SRC) ctc_host_test.c -o $@

selftest: $(BUILD)/ctc_host_test
	$<  selftest

# Regenerate the vocabulary header and confirm it still matches tokenizer/vocab.txt.
vocab:
	$(VENV)/bin/python $(REPO)/firmware/tools/gen_tokenizer.py --check

oracle: vocab
	$(VENV)/bin/python $(REPO)/firmware/test/run_gate6.py $(N)

arm:
	@mkdir -p $(BUILD)
	$(ARMCC) -c $(ARMFLAGS) $(SRC) -o $(BUILD)/citrinet_ctc.arm.o
	$(ARMSIZE) -A $(BUILD)/citrinet_ctc.arm.o

clean:
	rm -rf $(BUILD)

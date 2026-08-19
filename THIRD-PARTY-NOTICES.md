# Third-party notices

The Apache-2.0 `LICENSE` covers what this project wrote. It does **not** cover
the components below, which keep their own terms.

## Redistributed in this repository

| path | origin | copyright | licence |
|---|---|---|---|
| `firmware/lcd/BSP/STM32N6570-DK/stm32n6570_discovery_lcd.{c,h}` | STM32N6-GettingStarted-ObjectDetection, *STM32N6570-DK BSP Drivers* | STMicroelectronics | **BSD-3-Clause** |
| `firmware/lcd/BSP/Components/rk050hr18/rk050hr18.h` | same package, *BSP Components* | STMicroelectronics | **BSD-3-Clause** |
| `firmware/lcd/Utilities/lcd/stm32_lcd.{c,h}` | same package, *lcd* | STMicroelectronics | **BSD-3-Clause** |
| `firmware/lcd/Utilities/Fonts/font{8,12,16,20,24}.c`, `fonts.h` | same package, *Fonts* | STMicroelectronics | **BSD-3-Clause** |
| `tokenizer/tokenizer.model`, `tokenizer.vocab`, `vocab.txt` | the upstream Citrinet-256 export | NVIDIA / OpenVoiceOS | see *Model* below |

Licences are as listed in `LICENSE.md` at the root of the ObjectDetection
package, checked per component rather than assumed. The nine LCD files sit in
`firmware/lcd/` in that package's own directory layout, unmodified.

> **One file was removed rather than redistributed.** `stm32_lcd_ex.{c,h}` came
> from that package's `Application/` directory, for which its `LICENSE.md` has no
> row — so its terms are unstated. Nothing in this project called it (drawing goes
> through `UTIL_LCD_DisplayStringAt` and formatting through `snprintf`), so it was
> deleted instead of shipped under an unclear licence.

## Not in this repository, fetched by the user

| component | licence | notes |
|---|---|---|
| `STM32N6-GettingStarted-Audio`, `-ObjectDetection` | mixed: BSD-3-Clause, Apache-2.0, MIT and **SLA0044** | QUICKSTART §1 clones them. `Projects/`, `Binary/` and the AI runtime are SLA0044 — read ST's `LICENSE.md` before redistributing anything from them |
| STM32CubeProgrammer, STM32CubeCLT, ST Edge AI Core | ST's own terms | tools, not linked into the image |
| LibriSpeech `dev-clean` | CC BY 4.0 | calibration and evaluation only |

**`firmware/vendor-mods/gate4.patch` is a patch against ST's SLA0044 sources.**
The patch — the lines this project wrote — is Apache-2.0. The files it applies to
are ST's, and you must obtain them from ST yourself.

## Model

The network derives from **Citrinet-256** (`stt_en_citrinet_256_gamma_0_25`),
trained by NVIDIA and released through NeMo, via the ONNX export
`OpenVoiceOS/stt_en_citrinet_256_gamma_0_25_onnx`. The weights are NVIDIA's under
their terms; `docs/upstream_model_card.md` records the provenance.

`artifacts/` — including the quantised graphs and the compiled `network.c` and
weight blob — is **not** in this repository. Those are derived both from NVIDIA's
weights and from ST Edge AI Core's output, so redistributing them is a decision
for whoever publishes a fork, not something this repository makes on their behalf.
QUICKSTART §2b regenerates them from the upstream model.

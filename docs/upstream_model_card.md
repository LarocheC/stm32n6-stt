---
language:
  - en
license: cc-by-4.0
library_name: onnx-asr
pipeline_tag: automatic-speech-recognition
tags:
  - automatic-speech-recognition
  - speech-to-text
  - ctc
  - citrinet
  - nemo
  - onnx
  - openvoiceos
base_model: nvidia/stt_en_citrinet_256_gamma_0_25
---

# stt_en_citrinet_256_gamma_0_25_onnx

English speech-to-text model. ONNX export of
[stt_en_citrinet_256_gamma_0_25](https://catalog.ngc.nvidia.com/orgs/nvidia/models/stt_en_citrinet_256_gamma_0_25) — an NVIDIA NeMo **Citrinet** CTC model — for
[onnx-asr](https://github.com/istupakov/onnx-asr). Runs offline with ONNX
Runtime; PyTorch and NeMo are not required.

Part of the [OpenVoiceOS STT/ASR ONNX collection](https://huggingface.co/collections/OpenVoiceOS/stt-asr-onnx-699321e8732462509c642fbe).

## Files

| File | Purpose |
|---|---|
| `model.onnx` | Encoder + CTC head, fp32 |
| `vocab.txt` | Token vocabulary (`<token> <id>` per line, `▁` = space, `<blk>` = CTC blank) |
| `config.json` | onnx-asr metadata: `model_type: nemo-conformer-ctc`, `features_size: 80`, `subsampling_factor: 8` |

There is no int8 variant: these architectures are convolution-dominated, and
dynamic quantization produces `ConvInteger` nodes that ONNX Runtime cannot
execute on CPU. int8 requires static QDQ quantization with calibration data.

## Usage

With [onnx-asr](https://github.com/istupakov/onnx-asr) (`pip install onnx-asr[cpu,hub]`):

```python
import onnx_asr

model = onnx_asr.load_model("OpenVoiceOS/stt_en_citrinet_256_gamma_0_25_onnx")
print(model.recognize("speech.wav"))  # 16 kHz PCM wav
```

With [OpenVoiceOS](https://github.com/OpenVoiceOS), through
[ovos-stt-plugin-onnx-asr](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr)
(`mycroft.conf`):

```json
{
  "stt": {
    "module": "ovos-stt-plugin-onnx-asr",
    "ovos-stt-plugin-onnx-asr": {
      "model": "OpenVoiceOS/stt_en_citrinet_256_gamma_0_25_onnx"
    }
  }
}
```

## Export and verification

Exported from the original checkpoint with NeMo's `model.export()`
(see the [conversion guide](https://github.com/istupakov/onnx-asr/blob/main/docs/conversion.md)).
The `subsampling_factor` was measured empirically on the exported graph, and
the export was verified differentially: the ONNX model and the original NeMo
checkpoint produce identical transcriptions on a reference clip.

## Accuracy, training data and limitations

See the [source model card](https://catalog.ngc.nvidia.com/orgs/nvidia/models/stt_en_citrinet_256_gamma_0_25) for benchmark results, training
corpora and known limitations. This repo changes the runtime, not the weights.

## Related projects

- [onnx-asr](https://github.com/istupakov/onnx-asr) — ASR inference with ONNX Runtime
- [ovos-stt-plugin-onnx-asr](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr) — OpenVoiceOS STT plugin
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — framework the source model was trained with

# `config/policy.toml` — three corrections, all with provenance

Every threshold in `policy.toml` is meant to be traceable to a number someone
chose on purpose. Two of its budget values carried a comment rather than a
provenance, and one `[quantize]` block documents a behaviour the code does not
implement. Nothing here changes a verdict for a model already in the zoo; two
of the three change what a reader can check.

---

## 1. `onchip_bytes = 2883576` is right, and the comment should say why

```toml
# Usable on-chip pool for the SCREENING harness mpool: cpuRAM2 (1024 KB) plus
# npuRAM3-6 (4 x 448 KB). The compiler reports the virtual pool at 2,883,576 B.
onchip_bytes = 2883576
```

The arithmetic gives 1,048,576 + 4 x 458,752 = **2,883,584**. The comment is
correct that the compiler says 2,883,576, and the eight missing bytes are not a
typo — but the compiler says **both** numbers, in two files from the same run,
about the same pool:

```
$ grep -o 'name=cpuRAM2_npuRAM3_npuRAM4_npuRAM5_npuRAM6 .*size=[0-9]*' \
    runs/inference-engine-mnist-12/mnist-12/compile/onchip/network.c
name=cpuRAM2_npuRAM3_npuRAM4_npuRAM5_npuRAM6 offset=0x34100000  absolute_mode size=2883576 vpool …
```

```
$ python -c "import json;d=json.load(open('.../ws/neural_art__network/c_info.json'));
             print([p for p in d['memory_pools'] if p['id']==10][0])"
{'name': 'cpuRAM2_npuRAM3_npuRAM4_npuRAM5_npuRAM6', 'id': 10, 'alignment': 8,
 'address': '873463808', 'offset_start': 0, 'size_bytes': 2883584,
 'used_size_bytes': 1972225, …}
```

Where the eight bytes go is visible in the same `network.c` pool table: the
individual banks read `npuRAM3/4/5 = 458752` and `cpuRAM2 = 1048576`, but
`npuRAM6 = 458744` — eight short, and npuRAM6 is the last bank of the virtual
concatenation. It is a per-allocatable-pool reservation, matching the
`alignment: 8` the same record carries. Confirmed identical in six independent
compiles (mnist-12, yunet at two resolutions, pphumanseg, handpose,
whisper-tiny encoder).

It is not confined to the virtual pool. On ST's audio-application mpool, which
declares no virtual pool at all, **every** non-zero pool loses its eight bytes
(`artifacts/model_c/network.c`, the Citrinet compile):

```
name=cpuRAM2   … size=1048568      (1024 KB - 8)
name=npuRAM6   … size=458744       (448 KB  - 8)
name=hyperRAM  … size=16777208     (16 MB   - 8)
name=octoFlash … size=62914552     (60 MB   - 8)
```

**Recommended change — comment only, the value stands.** 2,883,576 is the
allocator's own figure and the conservative one of the two; keep it.

```toml
# Usable on-chip pool for the SCREENING harness mpool: cpuRAM2 (1024 KB) plus
# npuRAM3-6 (4 x 448 KB) = 2,883,584 B of declared bank. The compiler reserves
# 8 B per allocatable pool (`alignment: 8`), so `network.c`'s pool table reports
# the merged virtual pool as 2,883,576 and that is the number to budget against.
# Beware: `network_c_info.json` — which is what zoo/st/cinfo.py parses — reports
# `size_bytes: 2883584` for the SAME pool, so a check that compares a parsed
# pool size against this constant is off by 8 and should compare fractions or
# allow the slack.
onchip_bytes = 2883576
```

## 2. `audio_app_onchip_bytes = 1507328` — provenance, now first-hand

```toml
audio_app_onchip_bytes = 1507328    # 1472 KB: AXISRAM2 + AXISRAM6 only
```

Verified against ST's own file rather than inferred. `STM32N6-GettingStarted-Audio`,
`Projects/X-CUBE-AI/models/stm32n6.mpool` (md5 `3962913c702e781e24ba6bbe431ee10c`,
copied verbatim to `compile/st_audio.mpool`), declares exactly six pools, of
which two internal ones are non-zero:

| pool (fname) | base | size |
|---|---|---:|
| flexMEM (AXIFLEXMEM) | 0x34000000 | 0 KB |
| cpuRAM1 (AXISRAM1) | 0x34000000 | 0 KB — reserved for the application |
| **cpuRAM2 (AXISRAM2)** | **0x34100000** | **1024 KB** |
| **npuRAM6 (AXISRAM6)** | **0x34350000** | **448 KB** |
| hyperRAM (xSPI1) | 0x90000000 | 16 MB |
| octoFlash (xSPI2) | 0x70180000 | 63 MB |

1,048,576 + 458,752 = **1,507,328**. The comment is exactly right; it now has a
file behind it. `npuRAM3/4/5` are absent from ST's mpool rather than declared
at zero — no difference to the compiler, they carry zero bytes either way.

Two things worth adding to the comment, both of which cost this project time:

- **The same mpool offers 16 MB of hyperRAM.** `audio_app_onchip_bytes` is
  therefore not a ceiling the compiler enforces; it is a target the compiler
  will silently exceed by spilling. Citrinet at T=1200 does exactly that under
  ST's option set — 150 KB of activations into PSRAM with the epoch table still
  reading 0 SW / 0 hybrid. See the new
  `psram-spill-with-zero-sw-epochs` fault entry.
- **The 8-byte reservation applies here too**, so the allocator's usable total
  under this mpool is 1,507,312.

## 3. The npuRAM3/4/5 finding: 1,376,256 B powered, clocked, and offered to nobody

`docs/FEASIBILITY.md` §5 in this project listed "whether the audio app enables
npuRAM3/4/5" as unknown. It does — all of them — and the consequence is that
`audio_app_onchip_bytes` is a **policy choice of ST's, not a hardware limit**.

`Projects/GS/Src/audio_bm.c`, `Int_Mem_Config()` at line 794, in the default
(non-`APP_LP`) branch:

```c
RCC->MEMENR |= RCC_MEMENR_AXISRAM3EN | RCC_MEMENR_AXISRAM4EN
             | RCC_MEMENR_AXISRAM5EN | RCC_MEMENR_AXISRAM6EN;   /* :818 */
RCC->MEMENR |= RCC_MEMENR_CACHEAXIRAMEN;
/* then HAL_RAMCFG_EnableAXISRAM on SRAM2, SRAM3, SRAM4, SRAM5, SRAM6 */
/* then __HAL_RCC_AXISRAM{2,3,4,5,6}_MEM_CLK_ENABLE() */
```

Only the low-power variant narrows this to AXISRAM6 alone. And the application
claims none of them: `Projects/GS/STM32CubeIDE/STM32N657XX_LRUN.ld:49` declares
exactly one region, `RAM (xrw) : ORIGIN = 0x34000400, LENGTH = 1023K`, which
fits AXISRAM1 exactly (0x34000400 + 1023K = 0x34100000) and never touches
AXISRAM2-6.

| bank | range | size | claimed by |
|---|---|---:|---|
| AXISRAM1 | 0x34000400–0x34100000 | 1023 K | the application (its sole linker region) |
| AXISRAM2 (cpuRAM2) | 0x34100000–0x34200000 | 1024 K | the model, via ST's mpool |
| AXISRAM3 (npuRAM3) | 0x34200000–0x34270000 | 448 K | **nothing** |
| AXISRAM4 (npuRAM4) | 0x34270000–0x342E0000 | 448 K | **nothing** |
| AXISRAM5 (npuRAM5) | 0x342E0000–0x34350000 | 448 K | **nothing** |
| AXISRAM6 (npuRAM6) | 0x34350000–0x343C0000 | 448 K | the model, via ST's mpool |

Declaring the three idle banks raises the audio application's pool from
1,507,328 to the full **2,883,584 B** (0x34100000–0x343C0000, 2816 K
contiguous) — the same geometry as the screening harness mpool, which is a
useful consistency check on both numbers.

**Recommended change: comment, not value.** Keep 1,507,328 as the default,
because it is what ST ships and what a promoted model actually inherits, and
add the headroom as a note:

```toml
# The demo-tier applications reserve memory for themselves, so a model that
# screens fine can still be too big to promote. Checked at promotion time.
# 1472 KB = cpuRAM2 (1024 KB) + npuRAM6 (448 KB), the only two non-zero internal
# pools in ST's Projects/X-CUBE-AI/models/stm32n6.mpool. This is ST's
# conservative default, NOT a hardware ceiling: audio_bm.c's Int_Mem_Config()
# powers and clocks AXISRAM3/4/5 too and nothing claims them, so declaring them
# in the mpool widens this to 2,883,584 B. Two things to check before relying on
# that: whether the compiler produces a BETTER schedule with the wider pool or
# merely a legal one (more banks is not automatically faster — allocation across
# banks affects NPU port contention), and whether anything else in the final
# application wants those banks.
audio_app_onchip_bytes = 1507328
```

For scale: at 8 s Citrinet uses 42.5 % of the narrow pool, so nothing in this
project needs the wider one. It is upside, not a dependency — and it is the
difference between the 12 s window being a memory question (69.1 % of the
narrow pool) and not being one (35.3 % of the wide one).

## 4. `[quantize]` documents symmetric activations; the code hardcodes asymmetric

Not a value correction — a divergence between the policy file and the code it
is supposed to govern, found while trying to express this model's quantisation
as a recipe.

`policy.toml` `[quantize]` says, "Confirmed against ST's own quantization.html,
not folklore", and lists format/activation type/weight type/per-channel/
calibrate method. It has **no key for symmetry**, and `zoo/quant/qdq.py:386`
hardcodes it:

```python
extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
```

`grep -n "policy" zoo/quant/qdq.py` returns nothing, so no policy value reaches
that call at all.

For Citrinet the shipped artifact was quantised with
`ActivationSymmetric: True`, and that is not cosmetic. Verified on
`artifacts/onnx/q800_real.onnx`: **all 1,419 Q/DQ zero-point tensors are
exactly 0**, the input QuantizeLinear reads scale 0.12052242 / zero-point 0, and
the output DequantizeLinear reads 0.26541564 / 0 — which is what the deployment
contract records as `offset 0` at both ends. The firmware depends on it in one
specific place: the M55 front end quantises features as
`q = clip(round(x / 0.120522417128086), -128, 127)`, with **no offset term**.
Re-quantising this graph through the zoo as it stands would give the input
tensor a non-zero zero-point, and that front end would then be silently biased
by a constant on every feature. (Greedy CTC survives either way — argmax is
invariant under a positive-scale affine map with a per-tensor offset — so the
failure would show up as accuracy loss, not as a crash.)

**Recommended change: make it a policy key and pass it through.**

```toml
# ORT's default is asymmetric activations. Symmetric activations force every
# zero-point to 0, which is what lets a deployment contract read "offset 0" and
# lets a hand-written front end quantise with q = round(x/scale) and no offset
# term. Models that depend on that should be able to say so.
activation_symmetric = false
weight_symmetric = true
```

and in `qdq.py`:

```python
extra_options={
    "ActivationSymmetric": bool(cfg.get("activation_symmetric", False)),
    "WeightSymmetric": bool(cfg.get("weight_symmetric", True)),
},
```

with a per-recipe override, since it is a property of the model rather than of
the bench. Until then, `models/audio/citrinet-256-gamma025.toml` cannot
reproduce the artifact it describes, and that limitation is recorded in
`zoo-contrib/README.md`.

---

### Sources

| claim | file |
|---|---|
| `size=2883576` vpool line, `size_bytes: 2883584` | `~/stm32n6-deployment-zoo/runs/inference-engine-mnist-12/mnist-12/compile/onchip/{network.c,ws/neural_art__network/c_info.json}` |
| 8 B per pool on a non-virtual mpool | `~/stm32n6-tts/artifacts/model_c/network.c` |
| ST's audio mpool geometry | `~/stm32n6-tts/compile/st_audio.mpool` (md5 `3962913c702e781e24ba6bbe431ee10c`), `~/stm32n6-tts/compile/GATE2.md` §2 |
| npuRAM3/4/5 powered and unclaimed | `vendor/STM32N6-GettingStarted-Audio/Projects/GS/Src/audio_bm.c:794-830`, `Projects/GS/STM32CubeIDE/STM32N657XX_LRUN.ld:49`, `~/stm32n6-tts/docs/MEMORY-MAP.md` |
| all zero-points 0 in the shipped int8 graph | `~/stm32n6-tts/artifacts/onnx/q800_real.onnx`, `~/stm32n6-tts/compile/reports/g800_real/io_contract.h` |
| hardcoded `ActivationSymmetric: False` | `~/stm32n6-deployment-zoo/zoo/quant/qdq.py:386` |

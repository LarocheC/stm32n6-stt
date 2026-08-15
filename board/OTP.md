# OTP fuse state — read before Gate 3, and it clears Gate 3

**Read 2026-08-15, read-only, board unchanged.**

```
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG \
  -el <CubeProgrammer>/bin/ExternalLoader/OTP_FUSES_STM32N6xx.stldr -otp displ
```

Board: STM32N6570-DK, ST-LINK SN `003D00443234510E37333934`, FW V3J17M11,
device 0x486 Rev B, 3.29 V. Global state: **Secured**, hardware key not set.
368 OTP words readable — word 124 is well within range.

## The word that matters

```
    124     |     0x00018000  |  0x00000000
```

`0x00018000` = bit 16 | bit 15. **Both are already programmed:**

| bit | fuse | purpose | state |
|---:|---|---|---|
| 16 | `HSLV_VDDIO2` | PSRAM / XSPIM1 high-speed low-voltage | **set** |
| 15 | `HSLV_VDDIO3` | external flash / XSPIM2 high-speed low-voltage | **set** |

## Why this matters

`Projects/GS/Src/audio_bm.c:108` calls `fuse_vddio()` unconditionally on this
board. The guard is
`#if (defined(USE_STM32N6xx_NUCLEO) || defined(USE_STM32N6570_DK))`, and
`Drivers/BSP/STM32N6570-DK/stm32n6570_discovery.h:59-61` **self-defines**
`USE_STM32N6570_DK` — it is not a Makefile option you can decline. The outer
`#ifdef HAL_BSEC_MODULE_ENABLED` is satisfied at `stm32n6xx_hal_conf.h:38`.

`fuse_hardware_conf()` (`Projects/Common/misc_toolbox.c:69-107`) reads OTP word
124 and, **if the bit is clear, programs it permanently** with no prompt:

```c
if ((data & fuse_mask) != fuse_mask) {
  data |= fuse_mask;
  HAL_BSEC_OTP_Program(&sBsecHandler, fuse_id, data, HAL_BSEC_NORMAL_PROG);
  /* ... verify, or spin forever in while(1){} */
}
```

So "just flash ST's unmodified app first" — the conventional safe warm-up, and
what Gate 3 asks for — is *not* inert on a fresh board. It burns two fuses.

**On this board it is inert**, because both bits are already set: the read
branch matches and the program branch never runs. Gate 3 carries no
irreversible action here.

This is consistent with the board's history. The deployment zoo has measured
seven models with weights streaming from xSPI2 external flash, which does not
work without `HSLV_VDDIO3`. The fuses were blown long before this project.

## What this does *not* clear

- These fuses are permanent and correct for a DK. Nothing here should be read
  as encouragement to run this firmware on a **custom board** whose IO rails are
  not 1.8 V — blowing HSLV on a 3.3 V rail is a hardware fault, not a setting.
- Word 124's own status is `0x00000000` — not permanently write-locked — so
  other bits in it remain programmable by anything that asks.
- The separate flash-overwrite hazard is unaffected: the demo's weights at
  `0x70180000` and the app slot at `0x70100000` collide with what `zoo measure`
  writes. See `docs/GATES-1-2.md`.

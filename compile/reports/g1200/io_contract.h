#endif

/********************************** INPUTS ***********************************/
#define STAI_NETWORK_IN_NUM (1)
#define STAI_NETWORK_IN_ALIGNMENTS \
  { \
    32 \
  }
#define STAI_NETWORK_IN_NAMES \
  { \
    "Input_0_out_0" \
  }
#define STAI_NETWORK_IN_FORMATS \
  { \
    STAI_FORMAT_S8 \
  }
#define STAI_NETWORK_IN_SIZES \
  { \
    96000 \
  }
#define STAI_NETWORK_IN_SIZES_BYTES \
  { \
    96000 \
  }

#define STAI_NETWORK_IN_1_ALIGNMENT (32)
#define STAI_NETWORK_IN_1_NAME "Input_0_out_0"
#define STAI_NETWORK_IN_1_FLAGS (STAI_FLAG_PREALLOCATED|STAI_FLAG_CHANNEL_FIRST)
#define STAI_NETWORK_IN_1_FORMAT (STAI_FORMAT_S8)
#define STAI_NETWORK_IN_1_SIZE (96000)
#define STAI_NETWORK_IN_1_SIZE_BYTES (96000)
#define STAI_NETWORK_IN_1_CHANNEL (1)
#define STAI_NETWORK_IN_1_HEIGHT (80)
#define STAI_NETWORK_IN_1_WIDTH (1200)
#define STAI_NETWORK_IN_1_BATCH (1)
#define STAI_NETWORK_IN_1_RANK (3)
#define STAI_NETWORK_IN_1_SHAPE \
  { \
    80, 1200, 1 \
  }
#define STAI_NETWORK_IN_1_SCALE_OFFSET_NUM (1)
#define STAI_NETWORK_IN_1_SCALES \
  { \
    0.143160685896873 \
  }
#define STAI_NETWORK_IN_1_OFFSETS \
  { \
    0 \
  }

/********************************** OUTPUTS **********************************/
#define STAI_NETWORK_OUT_NUM (1)
#define STAI_NETWORK_OUT_ALIGNMENTS \
  { \
    32 \
  }
#define STAI_NETWORK_OUT_NAMES \
  { \
    "Transpose_1488_out_0" \
  }
#define STAI_NETWORK_OUT_FORMATS \
  { \
    STAI_FORMAT_S8 \
  }
#define STAI_NETWORK_OUT_SIZES \
  { \
    153750 \
  }
#define STAI_NETWORK_OUT_SIZES_BYTES \
  { \
    153750 \
  }

#define STAI_NETWORK_OUT_1_ALIGNMENT (32)
#define STAI_NETWORK_OUT_1_NAME "Transpose_1488_out_0"
#define STAI_NETWORK_OUT_1_FLAGS (STAI_FLAG_PREALLOCATED|STAI_FLAG_OVERRIDE|STAI_FLAG_CHANNEL_FIRST)
#define STAI_NETWORK_OUT_1_FORMAT (STAI_FORMAT_S8)
#define STAI_NETWORK_OUT_1_SIZE (153750)
#define STAI_NETWORK_OUT_1_SIZE_BYTES (153750)
#define STAI_NETWORK_OUT_1_CHANNEL (1)
#define STAI_NETWORK_OUT_1_HEIGHT (150)
#define STAI_NETWORK_OUT_1_WIDTH (1025)
#define STAI_NETWORK_OUT_1_BATCH (1)
#define STAI_NETWORK_OUT_1_RANK (3)
#define STAI_NETWORK_OUT_1_SHAPE \
  { \
    150, 1025, 1 \
  }
#define STAI_NETWORK_OUT_1_SCALE_OFFSET_NUM (1)
#define STAI_NETWORK_OUT_1_SCALES \
  { \
    0.279830664396286 \
  }
#define STAI_NETWORK_OUT_1_OFFSETS \
  { \
    0 \
  }

/********************************** WEIGHTS **********************************/
#if LL_ATON_DBG_BUFFER_INFO_EXCLUDED == 0
#define STAI_NETWORK_WEIGHTS_NUM (1006)

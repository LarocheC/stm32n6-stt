/* citrinet_fe.h — NeMo-exact log-mel front end for stt_en_citrinet_256_gamma_0_25
 * on the STM32N6 Cortex-M55.  Self-contained: no dependency on ST's
 * STM32_AI_AudioPreprocessing_Library, which cannot express this transform
 * (see firmware/FRONTEND.md §2).
 *
 * The spec, and the host-side test oracle, is model/fe.py:
 *
 *   sample rate   16000       n_fft 512    win_length 400   hop 160   n_mels 80
 *   window        symmetric Hann, scipy.signal.get_window("hann",400,fftbins=False)
 *   pre-emphasis  y[0]=x[0], y[n]=x[n]-0.97*x[n-1]
 *   framing       centre (librosa center=True, pad_mode="constant" -> zeros)
 *   spectrum      power, |S|^2
 *   mel           librosa slaney-normalised, fmin 0, fmax 8000, htk=False
 *   log           ln(mel + 2^-24)                   <-- the one value that matters
 *   normalise     per mel bin over time: (x - mean_t) / (std_t(ddof=1) + 1e-5)
 *   quantise      q = clip(rint(x / scale) + zp, -128, 127)
 *
 * Two passes are unavoidable: the per-bin mean and standard deviation are not
 * known until the last column exists, so quantisation cannot happen inside the
 * per-column call the way ST's LogMelSpectrogramColumn_q15_Q8 does it.
 *
 * Memory: the caller owns the pass-1 scratch (CITRINET_FE_SCRATCH_BYTES, 256,000 B
 * of float32 by default, 128,000 B if built -DCITRINET_FE_SCRATCH_F16=1) and the
 * 64,000 B int8 output, which on the target is the NPU's own p_stai_inputs[0] and
 * therefore not in the 1023 KB region at all.  The context struct itself is ~6 KB.
 *
 * NOT reentrant: one FFT instance is file-static in citrinet_fe.c.
 */
#ifndef CITRINET_FE_H
#define CITRINET_FE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ geometry */
#define CITRINET_FE_SR          16000
#define CITRINET_FE_NFFT        512
#define CITRINET_FE_WIN         400
#define CITRINET_FE_HOP         160
#define CITRINET_FE_NMEL        80
#define CITRINET_FE_T           800                      /* frames the graph is frozen to */
#define CITRINET_FE_NBINS       (CITRINET_FE_NFFT / 2 + 1)             /* 257 */
#define CITRINET_FE_WOFF        ((CITRINET_FE_NFFT - CITRINET_FE_WIN) / 2)  /* 56 */

/* 8.000 s at 16 kHz.  With centre framing that is 801 whole frames; the graph
 * takes the first 800, exactly as model/fe_reference.py:features_for_fixed_window. */
#define CITRINET_FE_NSAMPLES    (CITRINET_FE_T * CITRINET_FE_HOP)      /* 128000 */
#define CITRINET_FE_OUT_BYTES   (CITRINET_FE_NMEL * CITRINET_FE_T)     /* 64000  */

/* ------------------------------------------------------------------ constants */
/* 2^-24, written so no compiler can round it: NeMo's log_zero_guard_value.
 * It is ADDED, never clamped.  ST's library clamps at FLT_MIN (1.18e-38) and so
 * floors log-mel near -87 where NeMo floors it at ln(2^-24) = -16.6355324. */
#define CITRINET_FE_LOG_GUARD   5.9604644775390625e-8f
#define CITRINET_FE_LOG_FLOOR   (-16.635532333438686f)  /* ln(2^-24) */
#define CITRINET_FE_STD_EPS     1e-5f
#define CITRINET_FE_PREEMPH     0.97f
#define CITRINET_FE_I16_SCALE   (1.0f / 32768.0f)

/* Refuse to invoke the model when more than this fraction of the *spoken* mel
 * energies landed below the log guard.  eval/results/gain.log: at -54 dBFS,
 * 97.9 % of bins are below the guard and WER goes 5.83 % -> 35.28 % while every
 * log still reports a clean NPU run.
 *
 * The threshold is 0.50, derived from a measured distribution rather than
 * guessed.  Over 80 dev-clean utterances that fill the 8 s window at correct
 * gain: median 4.5 %, p90 14.7 %, p95 22.4 %, **max 33.1 %**.  The failure mode
 * this exists to catch sits at 91-97 %.  An earlier value of 0.20 was inside the
 * speech distribution and refused 5 of those 80 utterances even though they
 * transcribed essentially perfectly -- a threshold that rejects good audio is
 * worse than none, because it trains you to ignore it.
 *
 * Note this is evaluated over non-silent bins only; see citrinet_fe_guard_fraction(). */
#define CITRINET_FE_GUARD_MAX_FRAC  0.50f

/* --------------------------------------------------------------- scratch type */
/* The pass-1 log-mel matrix.  float32 by default.  Values live in roughly
 * [-16.7, +6], so IEEE binary16 also holds them with ~3 decimal digits, which is
 * far finer than the int8 grid they land on -- see firmware/FRONTEND.md §5 for the
 * measured cost of halving this buffer. */
#ifndef CITRINET_FE_SCRATCH_F16
#define CITRINET_FE_SCRATCH_F16 0
#endif

#if CITRINET_FE_SCRATCH_F16
typedef uint16_t citrinet_fe_scratch_t;   /* raw IEEE binary16 bit pattern */
#else
typedef float citrinet_fe_scratch_t;
#endif

#define CITRINET_FE_SCRATCH_ELEMS  (CITRINET_FE_NMEL * CITRINET_FE_T)
#define CITRINET_FE_SCRATCH_BYTES  (CITRINET_FE_SCRATCH_ELEMS * sizeof(citrinet_fe_scratch_t))

/* ----------------------------------------------------------------- return codes */
#define CITRINET_FE_OK          0
#define CITRINET_FE_E_SCRATCH  (-1)   /* scratch missing or too small           */
#define CITRINET_FE_E_ARG      (-2)   /* bad pointer / scale / frame index      */
#define CITRINET_FE_E_STATE    (-3)   /* finish() before all 800 columns exist  */
#define CITRINET_FE_E_GUARD    (-4)   /* features produced, but level is unusable */

/* ---------------------------------------------------------------------- context */
typedef struct {
    citrinet_fe_scratch_t *logmel;   /* [NMEL][T], bin-major: logmel[b*T + t]  */
    uint32_t  scratch_elems;
    uint32_t  columns;               /* columns written since reset            */

    /* log-floor telemetry — filled during pass 1, every run, unconditionally */
    uint32_t  guard_below;           /* mel energies strictly < 2^-24          */
    uint32_t  guard_zero;            /* mel energies that came out exactly 0   */
    uint32_t  guard_total;           /* = NMEL * columns                       */

    /* level telemetry */
    float     input_peak;            /* max |sample| / 32768 seen by run()     */
    float     gain_applied;          /* what peak_normalize() last returned    */
    float     logmel_min;
    float     logmel_max;

    /* pass-2 statistics, valid after finish() */
    float     mu[CITRINET_FE_NMEL];
    float     sd[CITRINET_FE_NMEL];  /* std(ddof=1) + 1e-5                     */
    uint32_t  clipped;               /* int8 values that hit the -128/+127 rail */

    /* work buffers, kept out of the stack: 2*512 + 257 floats = 5,124 B */
    float     work[2 * CITRINET_FE_NFFT];
    float     power[CITRINET_FE_NBINS];
} citrinet_fe_t;

/* ------------------------------------------------------------------------- API */

/* Bind a caller-owned scratch buffer.  scratch_bytes must be >=
 * CITRINET_FE_SCRATCH_BYTES.  Also initialises the FFT.  Idempotent. */
int citrinet_fe_init(citrinet_fe_t *fe, void *scratch, size_t scratch_bytes);

/* Gain hook.  Peak-normalises pcm in place so that max|x| == target_peak (in
 * 0..1 full scale), never amplifying by more than max_gain.  Returns the gain
 * actually applied, or 0.0f for an all-zero buffer (nothing written).
 *
 * eval/results/gain.log: peak-normalising to 0.9 takes -54 dBFS input from
 * 35.28 % WER back to 5.83 %.  Doing it here, i.e. AFTER the MDF's int16
 * truncation, only recovers to 10.45 % -- so this is a safety net, not the fix.
 * The fix is MDF_GAIN / the /256 shift in the acquisition callbacks. */
float citrinet_fe_peak_normalize(int16_t *pcm, uint32_t n, float target_peak, float max_gain);

/* One-shot: int16 PCM -> 64,000 int8 features, CHANNEL_FIRST {80,800,1}
 * (mel-major, so out[b*800 + t]).  n_samples may be less than
 * CITRINET_FE_NSAMPLES; the tail is treated as the zeros centre-padding already
 * implies.  quant_scale is the model's input scale, read at runtime from
 * stai_network_get_info()->inputs[0].scale.data[0] (0.120522417128086 today).
 *
 * Returns CITRINET_FE_OK, or CITRINET_FE_E_GUARD when the features were computed
 * but more than CITRINET_FE_GUARD_MAX_FRAC of the mel energies were below the log
 * guard.  A caller that only tests `!= CITRINET_FE_OK` therefore refuses to
 * invoke the model on a mis-gained capture by default. */
int citrinet_fe_run(citrinet_fe_t *fe, const int16_t *pcm, uint32_t n_samples,
                    float quant_scale, int32_t quant_zero_point, int8_t *out);

/* Streaming form of the same thing, for the capture-driven path of WORKLIST §5.7.
 * reset() -> column() x800 -> finish().  `seg` holds CITRINET_FE_NFFT
 * PRE-EMPHASISED float samples covering signal indices
 * [t*HOP - NFFT/2, t*HOP - NFFT/2 + NFFT), zeros outside the signal; use
 * citrinet_fe_build_frame() to produce one from an int16 buffer. */
void citrinet_fe_reset(citrinet_fe_t *fe);
int  citrinet_fe_column(citrinet_fe_t *fe, const float *seg, uint32_t t);
int  citrinet_fe_finish(citrinet_fe_t *fe, float quant_scale, int32_t quant_zero_point,
                        int8_t *out);

/* Fill seg[CITRINET_FE_NFFT] for frame t from an int16 buffer, applying the
 * /32768 conversion, the 0.97 pre-emphasis and the centre zero-padding. */
void citrinet_fe_build_frame(const int16_t *pcm, uint32_t n_samples, uint32_t t, float *seg);

/* -------------------------------------------------------------------- telemetry */
float citrinet_fe_guard_fraction(const citrinet_fe_t *fe);

/* 1 when the capture is usable, 0 when the log guard swallowed too much of it.
 * Call this before stai_network_run().  This is the check docs/FEASIBILITY.md
 * §2(d) and eval/results/gain.log say the product lives or dies on. */
int citrinet_fe_features_usable(const citrinet_fe_t *fe);

/* Read back a pass-1 log-mel value (before per-bin normalisation). */
float citrinet_fe_logmel(const citrinet_fe_t *fe, uint32_t bin, uint32_t t);

/* One-line human-readable level report into buf; returns buf. */
const char *citrinet_fe_report(const citrinet_fe_t *fe, char *buf, size_t n);

/* Build introspection, so a test or a boot banner can state what it is running.
 * fnv1a is CITRINET_FE_TAB_FNV1A from the generated table header; backend is 1
 * when the FFT is CMSIS-DSP's arm_rfft_fast_f32 and 0 for the portable one. */
uint32_t citrinet_fe_tables_fnv1a(void);
int      citrinet_fe_backend(void);

#ifdef __cplusplus
}
#endif
#endif /* CITRINET_FE_H */

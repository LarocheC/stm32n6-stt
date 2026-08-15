/* citrinet_fe.c — NeMo-exact log-mel front end.  See citrinet_fe.h for the spec
 * and firmware/FRONTEND.md for why ST's STM32_AI_AudioPreprocessing_Library is
 * not used and what the host parity measurement says.
 *
 * Build knobs (all optional):
 *   CITRINET_FE_USE_CMSIS=1   use arm_rfft_fast_f32.  Default: on for ARM
 *                             targets, off elsewhere so the host parity harness
 *                             compiles this exact source with a portable FFT.
 *   CITRINET_FE_SCRATCH_F16=1 hold the pass-1 log-mel matrix as IEEE binary16,
 *                             halving it from 256,000 B to 128,000 B.
 *   CITRINET_FE_NAIVE_VAR=1   compute the ddof=1 variance from running
 *                             sum/sum-of-squares instead of the two-pass form.
 *                             Kept so the harness can measure the difference
 *                             WORKLIST §5.3(b) asks about; do not ship it.
 */
#include "citrinet_fe.h"
#include "citrinet_fe_tables.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

/* ---------------------------------------------------------------- FFT backend */
/* How the ddof=1 variance is computed.  0 is what ships.
 *   0  two-pass (mean, then sum of squared deviations), double accumulators
 *   1  running sum / sum-of-squares, double accumulators
 *   2  running sum / sum-of-squares, float32 accumulators   <- WORKLIST §5.3(b)
 * All three are measured against model/fe.py in firmware/FRONTEND.md §6. */
#ifndef CITRINET_FE_VAR_MODE
#define CITRINET_FE_VAR_MODE 0
#endif

#if !defined(CITRINET_FE_USE_CMSIS)
#  if defined(__ARM_ARCH) || defined(ARM_MATH_CM55) || defined(ARM_MATH_CM4)
#    define CITRINET_FE_USE_CMSIS 1
#  else
#    define CITRINET_FE_USE_CMSIS 0
#  endif
#endif

#if CITRINET_FE_USE_CMSIS
#include "arm_math.h"
static arm_rfft_fast_instance_f32 s_rfft;
static int s_rfft_ready = 0;
#else
/* Portable radix-2 DIT FFT, N = 512.  Exists so this file can be compiled and
 * proven natively (firmware/test/test_fe_host.c); the target uses CMSIS. */
#define FE_N CITRINET_FE_NFFT
static float s_tw_re[FE_N / 2];
static float s_tw_im[FE_N / 2];
static int   s_tw_ready = 0;

static void fe_fft_tables(void)
{
    if (s_tw_ready) return;
    for (int i = 0; i < FE_N / 2; i++) {
        double a = -2.0 * 3.14159265358979323846 * (double)i / (double)FE_N;
        s_tw_re[i] = (float)cos(a);
        s_tw_im[i] = (float)sin(a);
    }
    s_tw_ready = 1;
}

static void fe_fft512(float *re, float *im)
{
    /* bit reversal */
    for (int i = 1, j = 0; i < FE_N; i++) {
        int bit = FE_N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= FE_N; len <<= 1) {
        int half = len >> 1;
        int step = FE_N / len;
        for (int i = 0; i < FE_N; i += len) {
            for (int k = 0; k < half; k++) {
                float wr = s_tw_re[k * step], wi = s_tw_im[k * step];
                float xr = re[i + k + half], xi = im[i + k + half];
                float vr = xr * wr - xi * wi;
                float vi = xr * wi + xi * wr;
                float ur = re[i + k], ui = im[i + k];
                re[i + k]        = ur + vr;  im[i + k]        = ui + vi;
                re[i + k + half] = ur - vr;  im[i + k + half] = ui - vi;
            }
        }
    }
}
#endif /* CITRINET_FE_USE_CMSIS */

/* -------------------------------------------------------------- fp16 scratch */
#if CITRINET_FE_SCRATCH_F16
/* Software IEEE binary16 conversion, round-to-nearest-even.  Software rather
 * than __fp16 so the host harness measures exactly what the M55 would store. */
static uint16_t fe_f32_to_f16(float f)
{
    union { float f; uint32_t u; } v; v.f = f;
    uint32_t u = v.u;
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t mant = u & 0x007FFFFFu;
    int32_t  exp  = (int32_t)((u >> 23) & 0xFFu);

    if (exp == 255) {   /* Inf / NaN */
        return (uint16_t)(sign | 0x7C00u | (mant ? (0x0200u | (mant >> 13)) : 0u));
    }
    int32_t e = exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t res   = mant >> shift;
        uint32_t rem   = mant & ((1u << shift) - 1u);
        uint32_t half  = 1u << (shift - 1);
        if (rem > half || (rem == half && (res & 1u))) res++;
        return (uint16_t)(sign | res);
    }
    uint32_t res = ((uint32_t)e << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (res & 1u))) res++;
    return (uint16_t)(sign | res);
}

static float fe_f16_to_f32(uint16_t h)
{
    union { float f; uint32_t u; } v;
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(h & 0x03FFu);
    if (exp == 0) {
        if (mant == 0) { v.u = sign; return v.f; }
        int e = -1;
        do { mant <<= 1; e++; } while (!(mant & 0x400u));
        mant &= 0x3FFu;
        v.u = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        return v.f;
    }
    if (exp == 31) { v.u = sign | 0x7F800000u | (mant << 13); return v.f; }
    v.u = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    return v.f;
}
#define FE_STORE(x)  fe_f32_to_f16(x)
#define FE_LOAD(x)   fe_f16_to_f32(x)
#else
#define FE_STORE(x)  (x)
#define FE_LOAD(x)   (x)
#endif

/* -------------------------------------------------------------------- public */
int citrinet_fe_init(citrinet_fe_t *fe, void *scratch, size_t scratch_bytes)
{
    if (fe == NULL) return CITRINET_FE_E_ARG;
    if (scratch == NULL || scratch_bytes < CITRINET_FE_SCRATCH_BYTES)
        return CITRINET_FE_E_SCRATCH;

    memset(fe, 0, sizeof(*fe));
    fe->logmel = (citrinet_fe_scratch_t *)scratch;
    fe->scratch_elems = CITRINET_FE_SCRATCH_ELEMS;

#if CITRINET_FE_USE_CMSIS
    if (!s_rfft_ready) {
        if (arm_rfft_fast_init_f32(&s_rfft, CITRINET_FE_NFFT) != ARM_MATH_SUCCESS)
            return CITRINET_FE_E_ARG;
        s_rfft_ready = 1;
    }
#else
    fe_fft_tables();
#endif
    citrinet_fe_reset(fe);
    return CITRINET_FE_OK;
}

void citrinet_fe_reset(citrinet_fe_t *fe)
{
    if (fe == NULL) return;
    fe->columns     = 0;
    fe->guard_below = 0;
    fe->guard_zero  = 0;
    fe->guard_total = 0;
    fe->clipped     = 0;
    fe->logmel_min  =  1e30f;
    fe->logmel_max  = -1e30f;
}

float citrinet_fe_peak_normalize(int16_t *pcm, uint32_t n, float target_peak, float max_gain)
{
    if (pcm == NULL || n == 0u) return 0.0f;

    int32_t pk = 0;
    for (uint32_t i = 0; i < n; i++) {
        int32_t a = (int32_t)pcm[i];
        if (a < 0) a = -a;
        if (a > pk) pk = a;
    }
    if (pk == 0) return 0.0f;                       /* digital silence: leave alone */

    float g = target_peak * 32768.0f / (float)pk;
    if (max_gain > 0.0f && g > max_gain) g = max_gain;
    if (g == 1.0f) return 1.0f;

    for (uint32_t i = 0; i < n; i++) {
        float s = rintf((float)pcm[i] * g);
        if (s >  32767.0f) s =  32767.0f;
        if (s < -32768.0f) s = -32768.0f;
        pcm[i] = (int16_t)s;
    }
    return g;
}

void citrinet_fe_build_frame(const int16_t *pcm, uint32_t n_samples, uint32_t t, float *seg)
{
    if (seg == NULL) return;
    memset(seg, 0, sizeof(float) * CITRINET_FE_NFFT);
    if (pcm == NULL) return;

    /* librosa center=True: frame t reads padded[t*HOP .. t*HOP+NFFT), where the
     * signal starts at padded[NFFT/2].  pad_mode="constant" -> zeros. */
    long base = (long)t * (long)CITRINET_FE_HOP - (long)(CITRINET_FE_NFFT / 2);

    /* Only [WOFF, WOFF+WIN) survives the window, so only that span is built. */
    for (int k = CITRINET_FE_WOFF; k < CITRINET_FE_WOFF + CITRINET_FE_WIN; k++) {
        long i = base + (long)k;
        if (i < 0 || i >= (long)n_samples) continue;
        float xi = (float)pcm[i] * CITRINET_FE_I16_SCALE;
        seg[k] = (i == 0)
               ? xi
               : xi - CITRINET_FE_PREEMPH * ((float)pcm[i - 1] * CITRINET_FE_I16_SCALE);
    }
}

int citrinet_fe_column(citrinet_fe_t *fe, const float *seg, uint32_t t)
{
    if (fe == NULL || fe->logmel == NULL || seg == NULL) return CITRINET_FE_E_ARG;
    if (t != fe->columns || t >= (uint32_t)CITRINET_FE_T) return CITRINET_FE_E_ARG;

    float *fr = fe->work;                       /* [0 .. NFFT)  windowed frame  */
    memset(fr, 0, sizeof(float) * CITRINET_FE_NFFT);
    for (int k = 0; k < CITRINET_FE_WIN; k++)
        fr[CITRINET_FE_WOFF + k] = seg[CITRINET_FE_WOFF + k] * kCitrinetHann[k];

    /* ---- power spectrum, 257 bins ---- */
#if CITRINET_FE_USE_CMSIS
    float *sp = fe->work + CITRINET_FE_NFFT;    /* CMSIS packing, destroys fr    */
    arm_rfft_fast_f32(&s_rfft, fr, sp, 0);
    fe->power[0] = sp[0] * sp[0];                                   /* DC        */
    fe->power[CITRINET_FE_NFFT / 2] = sp[1] * sp[1];                /* Nyquist   */
    for (int k = 1; k < CITRINET_FE_NFFT / 2; k++)
        fe->power[k] = sp[2 * k] * sp[2 * k] + sp[2 * k + 1] * sp[2 * k + 1];
#else
    float *im = fe->work + CITRINET_FE_NFFT;
    memset(im, 0, sizeof(float) * CITRINET_FE_NFFT);
    fe_fft512(fr, im);
    for (int k = 0; k <= CITRINET_FE_NFFT / 2; k++)
        fe->power[k] = fr[k] * fr[k] + im[k] * im[k];
#endif

    /* ---- sparse Slaney mel, then ln(x + 2^-24) ---- */
    uint32_t off = 0;
    for (int m = 0; m < CITRINET_FE_NMEL; m++) {
        uint32_t s = kCitrinetMelStart[m];
        uint32_t L = kCitrinetMelLen[m];
        const float *w = &kCitrinetMelW[off];
        const float *p = &fe->power[s];
        float acc = 0.0f;
        for (uint32_t k = 0; k < L; k++) acc += w[k] * p[k];
        off += L;

        /* Log-floor telemetry.  Not optional, not conditional: this is the
         * number that separates "quiet room" from "the gain stage is wrong". */
        if (acc < CITRINET_FE_LOG_GUARD) fe->guard_below++;
        if (acc == 0.0f)                 fe->guard_zero++;

        float lv = logf(acc + CITRINET_FE_LOG_GUARD);
        if (lv < fe->logmel_min) fe->logmel_min = lv;
        if (lv > fe->logmel_max) fe->logmel_max = lv;

        fe->logmel[(uint32_t)m * (uint32_t)CITRINET_FE_T + t] = FE_STORE(lv);
    }
    fe->guard_total += CITRINET_FE_NMEL;
    fe->columns = t + 1u;
    return CITRINET_FE_OK;
}

int citrinet_fe_finish(citrinet_fe_t *fe, float quant_scale, int32_t quant_zero_point,
                       int8_t *out)
{
    if (fe == NULL || fe->logmel == NULL || out == NULL) return CITRINET_FE_E_ARG;
    if (!(quant_scale > 0.0f)) return CITRINET_FE_E_ARG;
    if (fe->columns != (uint32_t)CITRINET_FE_T) return CITRINET_FE_E_STATE;

    const uint32_t T = (uint32_t)CITRINET_FE_T;
    fe->clipped = 0;

    for (uint32_t m = 0; m < (uint32_t)CITRINET_FE_NMEL; m++) {
        const citrinet_fe_scratch_t *row = &fe->logmel[m * T];

#if CITRINET_FE_VAR_MODE == 2
        /* Running sum / sum-of-squares in float32 — the form WORKLIST §5.3(b)
         * says to check rather than assume.  ss reaches ~2e5 while T*mu*mu
         * reaches nearly the same value, so the subtraction cancels most of a
         * 24-bit mantissa away. */
        float s = 0.0f, ss = 0.0f;
        for (uint32_t t = 0; t < T; t++) {
            float v = (float)FE_LOAD(row[t]);
            s += v; ss += v * v;
        }
        float muf  = s / (float)T;
        float varf = (ss - (float)T * muf * muf) / (float)(T - 1u);
        if (varf < 0.0f) varf = 0.0f;
        double mud = (double)muf, vard = (double)varf;
#elif CITRINET_FE_VAR_MODE == 1
        /* Running sum / sum-of-squares, double accumulators.  Streamable, so it
         * is what the capture-driven path of WORKLIST §5.7 would want. */
        double s = 0.0, ss = 0.0;
        for (uint32_t t = 0; t < T; t++) {
            double v = (double)FE_LOAD(row[t]);
            s += v; ss += v * v;
        }
        double mud  = s / (double)T;
        double vard = (ss - (double)T * mud * mud) / (double)(T - 1u);
        if (vard < 0.0) vard = 0.0;
#else
        /* Two-pass, which is what NumPy's std() does and what NeMo therefore
         * gets.  No cancellation; costs one extra sweep of a 3,200 B row that
         * is already in L1. */
        double s = 0.0;
        for (uint32_t t = 0; t < T; t++) s += (double)FE_LOAD(row[t]);
        double mud = s / (double)T;
        double ss = 0.0;
        for (uint32_t t = 0; t < T; t++) {
            double d = (double)FE_LOAD(row[t]) - mud;
            ss += d * d;
        }
        double vard = ss / (double)(T - 1u);     /* ddof = 1: NeMo divides by N-1 */
#endif
        /* Round the statistics to float32 before use: normalize_batch() works in
         * the tensor's dtype, so mu and sd are float32 in the oracle too. */
        float mu = (float)mud;
        float sd = (float)sqrt(vard) + CITRINET_FE_STD_EPS;
        fe->mu[m] = mu;
        fe->sd[m] = sd;

        int8_t *o = &out[m * T];
        for (uint32_t t = 0; t < T; t++) {
            float z = (FE_LOAD(row[t]) - mu) / sd;      /* per-feature z-score   */
            float q = z / quant_scale;                  /* model input scale     */
            /* rintf, not roundf: NumPy's np.round is round-half-to-even, and so
             * is rintf under the default rounding mode.  roundf rounds half away
             * from zero and disagrees with the oracle on exact .5 boundaries. */
            float r = rintf(q) + (float)quant_zero_point;
            if (r >  127.0f) { r =  127.0f; fe->clipped++; }
            else if (r < -128.0f) { r = -128.0f; fe->clipped++; }
            o[t] = (int8_t)r;
        }
    }

    return citrinet_fe_features_usable(fe) ? CITRINET_FE_OK : CITRINET_FE_E_GUARD;
}

int citrinet_fe_run(citrinet_fe_t *fe, const int16_t *pcm, uint32_t n_samples,
                    float quant_scale, int32_t quant_zero_point, int8_t *out)
{
    if (fe == NULL || pcm == NULL || out == NULL) return CITRINET_FE_E_ARG;
    if (fe->logmel == NULL) return CITRINET_FE_E_SCRATCH;

    citrinet_fe_reset(fe);

    int32_t pk = 0;
    for (uint32_t i = 0; i < n_samples; i++) {
        int32_t a = (int32_t)pcm[i];
        if (a < 0) a = -a;
        if (a > pk) pk = a;
    }
    fe->input_peak = (float)pk * CITRINET_FE_I16_SCALE;

    float seg[CITRINET_FE_NFFT];                 /* 2,048 B of stack            */
    for (uint32_t t = 0; t < (uint32_t)CITRINET_FE_T; t++) {
        citrinet_fe_build_frame(pcm, n_samples, t, seg);
        int rc = citrinet_fe_column(fe, seg, t);
        if (rc != CITRINET_FE_OK) return rc;
    }
    return citrinet_fe_finish(fe, quant_scale, quant_zero_point, out);
}

/* ------------------------------------------------------------------ telemetry */
float citrinet_fe_guard_fraction(const citrinet_fe_t *fe)
{
    if (fe == NULL || fe->guard_total == 0u) return 0.0f;

    /* Exclude bins whose energy is *exactly* zero.  citrinet_fe_run() zero-fills
     * the tail of a capture shorter than the 8 s window, and every mel energy over
     * that pad is identically 0 -- below the guard by construction, but carrying no
     * information about gain staging.  Counting it made the statistic a function of
     * utterance length: 30 of 60 arbitrary dev-clean utterances were refused, and a
     * 2.77 s utterance reported 77.4 % occupancy of which 65.1 points were pure pad.
     *
     * The quantity we actually want is "of the bins that carry signal, how many
     * collapsed onto the log floor". */
    uint32_t below = fe->guard_below;
    uint32_t total = fe->guard_total;
    if (fe->guard_zero <= below && fe->guard_zero < total) {
        below -= fe->guard_zero;
        total -= fe->guard_zero;
    }
    if (total == 0u) return 0.0f;
    return (float)below / (float)total;
}

int citrinet_fe_features_usable(const citrinet_fe_t *fe)
{
    return citrinet_fe_guard_fraction(fe) <= CITRINET_FE_GUARD_MAX_FRAC ? 1 : 0;
}

float citrinet_fe_logmel(const citrinet_fe_t *fe, uint32_t bin, uint32_t t)
{
    if (fe == NULL || fe->logmel == NULL ||
        bin >= (uint32_t)CITRINET_FE_NMEL || t >= (uint32_t)CITRINET_FE_T) return 0.0f;
    return FE_LOAD(fe->logmel[bin * (uint32_t)CITRINET_FE_T + t]);
}

uint32_t citrinet_fe_tables_fnv1a(void) { return (uint32_t)CITRINET_FE_TAB_FNV1A; }

int citrinet_fe_backend(void) { return CITRINET_FE_USE_CMSIS ? 1 : 0; }

const char *citrinet_fe_report(const citrinet_fe_t *fe, char *buf, size_t n)
{
    if (buf == NULL || n == 0u) return buf;
    if (fe == NULL) { buf[0] = '\0'; return buf; }

    /* Integer-only formatting on purpose: newlib-nano's printf drops %f unless
     * the link line carries -u _printf_float, and this string must never be the
     * reason a level warning fails to appear. */
    unsigned guard_pm = fe->guard_total
        ? (unsigned)(((uint64_t)fe->guard_below * 1000u) / fe->guard_total) : 0u;
    unsigned zero_pm = fe->guard_total
        ? (unsigned)(((uint64_t)fe->guard_zero * 1000u) / fe->guard_total) : 0u;
    int peak_millifs = (int)(fe->input_peak * 1000.0f + 0.5f);
    int lmin = (int)floorf(fe->logmel_min);
    int lmax = (int)ceilf(fe->logmel_max);

    snprintf(buf, n,
             "FE: peak %d/1000 FS, log-guard %u.%u%%%s, zeros %u.%u%%, "
             "logmel [%d,%d], clipped %u/%d",
             peak_millifs,
             guard_pm / 10u, guard_pm % 10u,
             citrinet_fe_features_usable(fe) ? "" : "  *** TOO QUIET, REFUSING ***",
             zero_pm / 10u, zero_pm % 10u,
             lmin, lmax,
             (unsigned)fe->clipped, CITRINET_FE_OUT_BYTES);
    return buf;
}

/* test_fe_host.c — run firmware/src/citrinet_fe.c natively against a WAV.
 *
 * Compiles the SAME source the M55 will run.  The only substitution is the FFT
 * backend, and the harness builds it both ways: with the vendored CMSIS-DSP
 * arm_rfft_fast_f32 (-D__GNUC_PYTHON__, the scalar C path the M55 also takes
 * when Helium is off) and with citrinet_fe.c's portable fallback.
 *
 *   test_fe_host --wav IN.wav --out PREFIX [--gain P] [--maxgain G] [--scale S]
 *
 * Writes
 *   PREFIX.pcm.s16    the exact int16 buffer the front end was fed (so the
 *                     Python oracle sees byte-identical input, gain included)
 *   PREFIX.logmel.f32 80*800 float32, pass 1, BEFORE per-feature normalisation
 *   PREFIX.norm.f32   80*800 float32, AFTER normalisation, before quantisation
 *   PREFIX.int8       64,000 int8, the tensor that goes to p_stai_inputs[0]
 * and one JSON object of telemetry on stdout.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "citrinet_fe.h"

#define DEFAULT_SCALE 0.120522417128086f

static int read_wav_s16(const char *path, int16_t **out, uint32_t *n_out, uint32_t *sr_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }

    char riff[4], wave[4];
    uint32_t riff_size;
    if (fread(riff, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0) goto bad;
    if (fread(&riff_size, 4, 1, f) != 1) goto bad;
    if (fread(wave, 1, 4, f) != 4 || memcmp(wave, "WAVE", 4) != 0) goto bad;

    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t sr = 0;
    int have_fmt = 0;

    for (;;) {
        char id[4];
        uint32_t sz;
        if (fread(id, 1, 4, f) != 4) break;
        if (fread(&sz, 4, 1, f) != 1) break;

        if (memcmp(id, "fmt ", 4) == 0) {
            uint8_t buf[40];
            uint32_t take = sz < sizeof(buf) ? sz : (uint32_t)sizeof(buf);
            if (fread(buf, 1, take, f) != take) goto bad;
            memcpy(&fmt, buf + 0, 2);
            memcpy(&ch,  buf + 2, 2);
            memcpy(&sr,  buf + 4, 4);
            memcpy(&bits, buf + 14, 2);
            have_fmt = 1;
            if (sz > take) fseek(f, (long)(sz - take), SEEK_CUR);
        } else if (memcmp(id, "data", 4) == 0) {
            if (!have_fmt) goto bad;
            if (fmt != 1 || ch != 1 || bits != 16) {
                fprintf(stderr, "need mono 16-bit PCM, got fmt=%u ch=%u bits=%u\n",
                        fmt, ch, bits);
                fclose(f); return -1;
            }
            uint32_t n = sz / 2u;
            int16_t *d = (int16_t *)malloc(n ? n * 2u : 2u);
            if (!d) goto bad;
            if (fread(d, 2, n, f) != n) { free(d); goto bad; }
            fclose(f);
            *out = d; *n_out = n; *sr_out = sr;
            return 0;
        } else {
            fseek(f, (long)(sz + (sz & 1u)), SEEK_CUR);
        }
        if (sz & 1u) fseek(f, 1, SEEK_CUR);
    }
bad:
    fprintf(stderr, "malformed wav: %s\n", path);
    fclose(f);
    return -1;
}

static int dump(const char *prefix, const char *suffix, const void *p, size_t n)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s%s", prefix, suffix);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return -1; }
    size_t w = fwrite(p, 1, n, f);
    fclose(f);
    return w == n ? 0 : -1;
}

int main(int argc, char **argv)
{
    const char *wav = NULL, *prefix = NULL;
    float gain_target = 0.0f, max_gain = 0.0f, scale = DEFAULT_SCALE;
    int32_t zp = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--wav")     && i + 1 < argc) wav = argv[++i];
        else if (!strcmp(argv[i], "--out")     && i + 1 < argc) prefix = argv[++i];
        else if (!strcmp(argv[i], "--gain")    && i + 1 < argc) gain_target = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--maxgain") && i + 1 < argc) max_gain = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--scale")   && i + 1 < argc) scale = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--zp")      && i + 1 < argc) zp = (int32_t)strtol(argv[++i], NULL, 10);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!wav || !prefix) {
        fprintf(stderr, "usage: %s --wav IN.wav --out PREFIX [--gain P] [--maxgain G] [--scale S]\n",
                argv[0]);
        return 2;
    }

    int16_t *src = NULL; uint32_t nsrc = 0, sr = 0;
    if (read_wav_s16(wav, &src, &nsrc, &sr) != 0) return 1;
    if (sr != CITRINET_FE_SR) {
        fprintf(stderr, "sample rate %u, need %d\n", sr, CITRINET_FE_SR);
        return 1;
    }

    /* the deployment window: exactly CITRINET_FE_NSAMPLES samples, zero padded */
    static int16_t pcm[CITRINET_FE_NSAMPLES];
    memset(pcm, 0, sizeof(pcm));
    uint32_t n = nsrc < (uint32_t)CITRINET_FE_NSAMPLES ? nsrc : (uint32_t)CITRINET_FE_NSAMPLES;
    memcpy(pcm, src, (size_t)n * 2u);
    free(src);

    float g_applied = 1.0f;
    if (gain_target > 0.0f)
        g_applied = citrinet_fe_peak_normalize(pcm, CITRINET_FE_NSAMPLES, gain_target, max_gain);

    void *scratch = malloc(CITRINET_FE_SCRATCH_BYTES);
    static int8_t out[CITRINET_FE_OUT_BYTES];
    citrinet_fe_t fe;
    if (citrinet_fe_init(&fe, scratch, CITRINET_FE_SCRATCH_BYTES) != CITRINET_FE_OK) {
        fprintf(stderr, "citrinet_fe_init failed\n"); return 1;
    }
    fe.gain_applied = g_applied;

    int rc = citrinet_fe_run(&fe, pcm, CITRINET_FE_NSAMPLES, scale, zp, out);

    /* re-derive the two float planes for the parity comparison */
    static float lm[CITRINET_FE_NMEL * CITRINET_FE_T];
    static float nm[CITRINET_FE_NMEL * CITRINET_FE_T];
    for (uint32_t m = 0; m < (uint32_t)CITRINET_FE_NMEL; m++) {
        for (uint32_t t = 0; t < (uint32_t)CITRINET_FE_T; t++) {
            float v = citrinet_fe_logmel(&fe, m, t);
            lm[m * CITRINET_FE_T + t] = v;
            nm[m * CITRINET_FE_T + t] = (v - fe.mu[m]) / fe.sd[m];
        }
    }

    if (dump(prefix, ".pcm.s16", pcm, sizeof(pcm))) return 1;
    if (dump(prefix, ".logmel.f32", lm, sizeof(lm))) return 1;
    if (dump(prefix, ".norm.f32", nm, sizeof(nm))) return 1;
    if (dump(prefix, ".int8", out, sizeof(out))) return 1;

    char msg[256];
    citrinet_fe_report(&fe, msg, sizeof(msg));
    fprintf(stderr, "%s\n", msg);

    printf("{\"rc\": %d, \"usable\": %d, \"guard_below\": %u, \"guard_zero\": %u, "
           "\"guard_total\": %u, \"guard_frac\": %.9g, \"input_peak\": %.9g, "
           "\"gain_applied\": %.9g, \"logmel_min\": %.9g, \"logmel_max\": %.9g, "
           "\"clipped\": %u, \"scale\": %.17g, \"cmsis\": %d, \"scratch_f16\": %d, "
           "\"var_mode\": %d, \"scratch_bytes\": %zu, \"ctx_bytes\": %zu, "
           "\"tab_fnv1a\": \"0x%08x\"}\n",
           rc, citrinet_fe_features_usable(&fe),
           (unsigned)fe.guard_below, (unsigned)fe.guard_zero, (unsigned)fe.guard_total,
           (double)citrinet_fe_guard_fraction(&fe), (double)fe.input_peak,
           (double)g_applied, (double)fe.logmel_min, (double)fe.logmel_max,
           (unsigned)fe.clipped, (double)scale,
           citrinet_fe_backend(),
           (int)CITRINET_FE_SCRATCH_F16,
#ifdef CITRINET_FE_VAR_MODE
           (int)CITRINET_FE_VAR_MODE,
#else
           0,
#endif
           (size_t)CITRINET_FE_SCRATCH_BYTES, sizeof(citrinet_fe_t),
           (unsigned)citrinet_fe_tables_fnv1a());

    free(scratch);
    return 0;
}

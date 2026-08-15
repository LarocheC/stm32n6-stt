/* ctc_host_test.c -- host driver for firmware/src/citrinet_ctc.c.
 *
 * Compiles the FIRMWARE decoder natively (same .c file the MCU builds) and
 * drives it from files, so the C implementation can be diffed against the
 * Python oracle in model/fe.py.  Nothing in here is compiled for the target;
 * the decoder itself has no printf and no stdio dependency.
 *
 *   ctc_host_test decode <manifest> <ids_out>
 *       manifest : "<key>\t<path-to-int8-logits.bin>" per line.  Each blob is
 *                  100*1025 = 102,500 raw int8, frame-major.
 *       stdout   : "<key>\t<status>\t<len>\t<text>" per line.
 *       ids_out  : "<key>\t<id> <id> ..." per line, 100 argmax ids.
 *
 *   ctc_host_test dumpvocab
 *       stdout   : "<id>\t<len>\t<piece>" for all 1025 ids.
 *
 *   ctc_host_test selftest
 *       exercises the argument, capacity and truncation contracts.
 *
 * Text is emitted verbatim: the vocabulary alphabet is [a-z'<>] plus the
 * space that U+2581 became, so it can contain neither a tab nor a newline
 * and TSV needs no escaping (asserted below).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "citrinet_ctc.h"

#define FRAMES   ((uint32_t)CITRINET_CTC_FRAMES)
#define CLASSES  ((uint32_t)CITRINET_CTC_CLASSES)
#define BLOB     (FRAMES * CLASSES)

static int8_t   g_logits[BLOB];
static uint16_t g_ids[FRAMES];
static char     g_text[CITRINET_CTC_TEXT_CAP];

static int read_blob(const char *path, int8_t *dst, size_t n)
{
    FILE  *f = fopen(path, "rb");
    size_t got;
    long   extra;

    if (f == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return -1;
    }
    got = fread(dst, 1u, n, f);
    extra = fgetc(f) == EOF ? 0L : 1L;
    fclose(f);
    if (got != n || extra != 0L) {
        fprintf(stderr, "%s: expected exactly %zu bytes, got %zu%s\n",
                path, n, got, extra ? "+" : "");
        return -1;
    }
    return 0;
}

static int cmd_decode(const char *manifest, const char *ids_path)
{
    FILE *mf = fopen(manifest, "r");
    FILE *idf;
    char  line[4096];
    int   rc = 0;

    if (mf == NULL) {
        fprintf(stderr, "cannot open %s\n", manifest);
        return 1;
    }
    idf = fopen(ids_path, "w");
    if (idf == NULL) {
        fprintf(stderr, "cannot write %s\n", ids_path);
        fclose(mf);
        return 1;
    }

    while (fgets(line, (int)sizeof(line), mf) != NULL) {
        char    *tab;
        char    *key;
        char    *path;
        uint32_t len = 0u;
        citrinet_ctc_status_t st;
        uint32_t t;
        size_t   i;

        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') {
            continue;
        }
        tab = strchr(line, '\t');
        if (tab == NULL) {
            fprintf(stderr, "bad manifest line: %s\n", line);
            rc = 1;
            break;
        }
        *tab = '\0';
        key  = line;
        path = tab + 1;

        if (read_blob(path, g_logits, (size_t)BLOB) != 0) {
            rc = 1;
            break;
        }

        st = citrinet_ctc_decode(g_logits, FRAMES, g_text,
                                 (uint32_t)sizeof(g_text), &len, g_ids);

        for (i = 0u; i < (size_t)len; ++i) {
            if (g_text[i] == '\t' || g_text[i] == '\n' || g_text[i] == '\r') {
                fprintf(stderr, "%s: decoded text contains a TSV-hostile byte\n", key);
                rc = 1;
            }
        }
        if (strlen(g_text) != (size_t)len) {
            fprintf(stderr, "%s: strlen != reported length\n", key);
            rc = 1;
        }
        if (rc != 0) {
            break;
        }

        printf("%s\t%d\t%u\t%s\n", key, (int)st, (unsigned)len, g_text);

        fprintf(idf, "%s\t", key);
        for (t = 0u; t < FRAMES; ++t) {
            fprintf(idf, "%u%s", (unsigned)g_ids[t], (t + 1u == FRAMES) ? "" : " ");
        }
        fputc('\n', idf);
    }

    fclose(idf);
    fclose(mf);
    return rc;
}

static int cmd_dumpvocab(void)
{
    uint32_t id;
    for (id = 0u; id < CLASSES; ++id) {
        const char *p = citrinet_ctc_piece(id);
        printf("%u\t%zu\t%s\n", (unsigned)id, strlen(p), p);
    }
    return 0;
}

/* ------------------------------------------------------------- selftest -- */

static int g_fail = 0;

static void check(int cond, const char *what)
{
    if (!cond) {
        fprintf(stderr, "SELFTEST FAIL: %s\n", what);
        g_fail = 1;
    }
}

static int cmd_selftest(void)
{
    static int8_t   lg[BLOB];
    uint16_t        ids[FRAMES];
    char            buf[CITRINET_CTC_TEXT_CAP];
    char            tiny[4];
    uint32_t        len = 123u;
    citrinet_ctc_status_t st;
    uint32_t        t;

    /* argument contract */
    check(citrinet_ctc_argmax(NULL, FRAMES, ids) == CITRINET_CTC_E_ARG, "argmax NULL logits");
    check(citrinet_ctc_argmax(lg, FRAMES, NULL) == CITRINET_CTC_E_ARG, "argmax NULL ids");
    check(citrinet_ctc_argmax(lg, 0u, ids) == CITRINET_CTC_E_ARG, "argmax n_frames 0");
    check(citrinet_ctc_argmax(lg, CITRINET_CTC_MAX_FRAMES + 1u, ids) == CITRINET_CTC_E_ARG,
          "argmax n_frames overlarge");
    check(citrinet_ctc_decode(lg, FRAMES, buf, 0u, &len, ids) == CITRINET_CTC_E_ARG,
          "decode cap 0");

    /* all-zero logits -> every frame argmaxes to class 0 (<unk>), ties low */
    memset(lg, 0, sizeof(lg));
    st = citrinet_ctc_decode(lg, FRAMES, buf, (uint32_t)sizeof(buf), &len, ids);
    check(st == CITRINET_CTC_OK, "all-zero decode ok");
    for (t = 0u; t < FRAMES; ++t) {
        check(ids[t] == 0u, "all-zero argmax is class 0");
    }
    check(strcmp(buf, "<unk>") == 0, "all-zero collapses to one <unk>");
    check(len == 5u, "all-zero length");

    /* all-blank -> empty string, still NUL-terminated */
    memset(lg, 0, sizeof(lg));
    for (t = 0u; t < FRAMES; ++t) {
        lg[t * CLASSES + CITRINET_CTC_BLANK] = 1;
    }
    st = citrinet_ctc_decode(lg, FRAMES, buf, (uint32_t)sizeof(buf), &len, ids);
    check(st == CITRINET_CTC_OK && len == 0u && buf[0] == '\0', "all-blank -> empty");

    /* leading/trailing marker pieces get trimmed; interior spaces survive.
     * id 2 is "▁the" -> " the".  Frames: [2, blank, 2] -> " the the" -> "the the" */
    memset(lg, 0, sizeof(lg));
    for (t = 0u; t < FRAMES; ++t) {
        lg[t * CLASSES + CITRINET_CTC_BLANK] = 1;
    }
    lg[0u * CLASSES + 2u] = 2;
    lg[2u * CLASSES + 2u] = 2;
    st = citrinet_ctc_decode(lg, FRAMES, buf, (uint32_t)sizeof(buf), &len, ids);
    check(st == CITRINET_CTC_OK, "trim case ok");
    check(strcmp(buf, "the the") == 0, "leading space trimmed, interior kept");
    check(len == 7u, "trim case length");

    /* capacity: the same decode into a 4-byte buffer must truncate cleanly */
    len = 999u;
    st  = citrinet_ctc_decode(lg, FRAMES, tiny, (uint32_t)sizeof(tiny), &len, ids);
    check(st == CITRINET_CTC_E_TRUNC, "tiny buffer reports truncation");
    check(strlen(tiny) == (size_t)len, "tiny buffer NUL-terminated at reported len");
    check(len <= 3u, "tiny buffer stayed within capacity");
    /* " the the" fills 3 bytes as " th", which trims to "th" -- a prefix of
     * the untruncated result.  Truncation is applied before the trim, so the
     * guarantee is "a valid prefix", not "the first cap-1 bytes". */
    check(strncmp(tiny, "the the", strlen(tiny)) == 0, "tiny buffer holds a valid prefix");

    /* out-of-range id is rejected by the detokeniser */
    for (t = 0u; t < FRAMES; ++t) {
        ids[t] = (uint16_t)CITRINET_CTC_BLANK;
    }
    ids[3] = (uint16_t)CLASSES;
    check(citrinet_ctc_ids_to_text(ids, FRAMES, buf, (uint32_t)sizeof(buf), &len)
          == CITRINET_CTC_E_ARG, "out-of-range id rejected");

    /* argmax tie-breaking: two classes share the max, lowest id must win */
    memset(lg, 0, sizeof(lg));
    lg[0u * CLASSES + 700u] = 5;
    lg[0u * CLASSES + 42u]  = 5;
    check(citrinet_ctc_argmax(lg, 1u, ids) == CITRINET_CTC_OK, "tie argmax ok");
    check(ids[0] == 42u, "tie resolves to lowest class id");

    /* decode with ids == NULL must still work (internal stack array) */
    st = citrinet_ctc_decode(lg, 1u, buf, (uint32_t)sizeof(buf), &len, NULL);
    check(st == CITRINET_CTC_OK, "decode with NULL ids");

    /* vocabulary sanity straight out of the generated header */
    check(strcmp(citrinet_ctc_piece(0u), "<unk>") == 0, "piece 0");
    check(strcmp(citrinet_ctc_piece(1024u), "<blk>") == 0, "piece 1024 (blank)");
    check(citrinet_ctc_piece(1025u)[0] == '\0', "piece out of range -> \"\"");

    if (g_fail == 0) {
        printf("selftest: all checks passed\n");
    }
    return g_fail;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "decode") == 0 && argc == 4) {
        return cmd_decode(argv[2], argv[3]);
    }
    if (argc == 2 && strcmp(argv[1], "dumpvocab") == 0) {
        return cmd_dumpvocab();
    }
    if (argc == 2 && strcmp(argv[1], "selftest") == 0) {
        return cmd_selftest();
    }
    fprintf(stderr,
            "usage: %s decode <manifest> <ids_out>\n"
            "       %s dumpvocab\n"
            "       %s selftest\n",
            argv[0], argv[0], argv[0]);
    return 2;
}

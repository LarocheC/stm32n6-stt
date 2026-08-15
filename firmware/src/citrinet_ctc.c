/* citrinet_ctc.c -- greedy CTC decode + SentencePiece detokenisation.
 *
 * Contract, rationale and buffer semantics: see firmware/inc/citrinet_ctc.h.
 *
 * Oracle: model/fe.py:greedy()
 *
 *     prev = -1
 *     for i in logits.argmax(-1):
 *         if i != prev and i != blank: out.append(VOCAB[i])
 *         prev = i
 *     return "".join(out).replace("▁", " ").strip()
 *
 * Note that `prev` is updated on EVERY frame including blanks, so the token
 * run "a a <blk> a" decodes to "aa", not "a".  The loop below does the same.
 *
 * The U+2581 -> ' ' substitution is already applied at table-generation time
 * by firmware/tools/gen_tokenizer.py, so kPieces[] is pure 7-bit ASCII (which
 * is also all the ST font tables can render).  append_piece() nevertheless
 * translates the UTF-8 byte sequence E2 96 81 as it copies, so the decoder is
 * still correct if the header is ever regenerated with the marker left in.
 * On the shipped header that branch is dead: no piece contains a 0xE2 byte.
 *
 * MISRA-ish: no dynamic allocation, no printf/errno, no non-const file-scope
 * data, every loop bounded by a compile-time constant or a validated
 * argument, single point of return per function.
 */

#include "citrinet_ctc.h"

/* The generated vocabulary tables are file-static; this is their ONLY
 * translation unit.  Do not include citrinet_vocab.h anywhere else. */
#include "citrinet_vocab.h"

/* The two headers describe the same network; prove it at compile time. */
#if (CITRINET_VOCAB_SIZE != CITRINET_CTC_CLASSES)
#error "citrinet_vocab.h and citrinet_ctc.h disagree on the class count"
#endif
#if (CITRINET_BLANK_ID != CITRINET_CTC_BLANK)
#error "citrinet_vocab.h and citrinet_ctc.h disagree on the blank id"
#endif
#if ((CITRINET_CTC_FRAMES * CITRINET_MAX_PIECE_LEN) + 1u) != CITRINET_CTC_TEXT_CAP
#error "CITRINET_CTC_TEXT_CAP is no longer the worst-case text length"
#endif

/* SentencePiece word-boundary marker U+2581, UTF-8 encoded. */
#define SP_MARK_B0 0xE2u
#define SP_MARK_B1 0x96u
#define SP_MARK_B2 0x81u

/* The only whitespace the decoder can emit.  Verified over tokenizer/vocab.txt:
 * the piece alphabet is exactly [a-z'<>] plus U+2581, so trimming ' ' is
 * bit-identical to Python's str.strip() on this vocabulary. */
#define SP_SPACE   ' '

/* ---------------------------------------------------------------- argmax -- */

citrinet_ctc_status_t citrinet_ctc_argmax(const int8_t *logits,
                                          uint32_t      n_frames,
                                          uint16_t     *ids)
{
    citrinet_ctc_status_t st = CITRINET_CTC_OK;

    if ((logits == 0) || (ids == 0) ||
        (n_frames == 0u) || (n_frames > CITRINET_CTC_MAX_FRAMES)) {
        st = CITRINET_CTC_E_ARG;
    } else {
        uint32_t t;
        for (t = 0u; t < n_frames; ++t) {
            const int8_t *row  = &logits[(uint32_t)t * CITRINET_CTC_CLASSES];
            int32_t       best = (int32_t)row[0];
            uint32_t      arg  = 0u;
            uint32_t      v;

            /* Strict '>' keeps the lowest index on ties, matching
             * numpy.argmax and therefore model/fe.py:greedy(). */
            for (v = 1u; v < CITRINET_CTC_CLASSES; ++v) {
                const int32_t x = (int32_t)row[v];
                if (x > best) {
                    best = x;
                    arg  = v;
                }
            }
            ids[t] = (uint16_t)arg;
        }
    }
    return st;
}

/* ---------------------------------------------------- piece concatenation -- */

/* Copy one vocabulary piece into text[], translating the UTF-8 encoding of
 * U+2581 to a single space.  Writes nothing past cap-1 (the NUL is added by
 * the caller).  Returns 0 on overflow, 1 otherwise; *w is advanced by the
 * number of bytes actually written. */
static uint32_t append_piece(char *text, uint32_t cap, uint32_t *w, uint32_t id)
{
    const char *p    = citrinet_piece(id);
    const uint32_t n = citrinet_piece_len(id);
    uint32_t       i = 0u;
    uint32_t       ok = 1u;

    while ((i < n) && (i < CITRINET_MAX_PIECE_LEN) && (ok != 0u)) {
        char     c;
        uint32_t adv;

        if (((uint8_t)p[i] == SP_MARK_B0) && ((i + 2u) < n) &&
            ((uint8_t)p[i + 1u] == SP_MARK_B1) &&
            ((uint8_t)p[i + 2u] == SP_MARK_B2)) {
            c   = SP_SPACE;
            adv = 3u;
        } else {
            c   = p[i];
            adv = 1u;
        }

        if ((*w + 1u) < cap) {          /* +1 reserves the terminating NUL */
            text[*w] = c;
            *w      += 1u;
            i       += adv;
        } else {
            ok = 0u;
        }
    }
    return ok;
}

/* ------------------------------------------------------------ detokenise -- */

citrinet_ctc_status_t citrinet_ctc_ids_to_text(const uint16_t *ids,
                                               uint32_t        n_frames,
                                               char           *text,
                                               uint32_t        cap,
                                               uint32_t       *n_written)
{
    citrinet_ctc_status_t st = CITRINET_CTC_OK;

    if ((ids == 0) || (text == 0) || (cap == 0u) ||
        (n_frames == 0u) || (n_frames > CITRINET_CTC_MAX_FRAMES)) {
        st = CITRINET_CTC_E_ARG;
    } else {
        uint32_t w    = 0u;
        uint32_t prev = CITRINET_CTC_NO_ID;
        uint32_t t;
        uint32_t lo;
        uint32_t hi;

        for (t = 0u; (t < n_frames) && (st == CITRINET_CTC_OK); ++t) {
            const uint32_t id = (uint32_t)ids[t];

            if (id >= CITRINET_VOCAB_SIZE) {
                st = CITRINET_CTC_E_ARG;
            } else {
                if ((id != prev) && (id != CITRINET_CTC_BLANK)) {
                    if (append_piece(text, cap, &w, id) == 0u) {
                        st = CITRINET_CTC_E_TRUNC;
                    }
                }
                prev = id;
            }
        }

        if (st != CITRINET_CTC_E_ARG) {
            /* Python's .strip(): drop leading and trailing spaces in place. */
            lo = 0u;
            hi = w;
            while ((lo < hi) && (text[lo] == SP_SPACE)) {
                ++lo;
            }
            while ((hi > lo) && (text[hi - 1u] == SP_SPACE)) {
                --hi;
            }
            if (lo > 0u) {
                uint32_t k;
                for (k = 0u; k < (hi - lo); ++k) {
                    text[k] = text[lo + k];
                }
            }
            w       = hi - lo;
            text[w] = '\0';
            if (n_written != 0) {
                *n_written = w;
            }
        } else {
            text[0] = '\0';
            if (n_written != 0) {
                *n_written = 0u;
            }
        }
    }
    return st;
}

/* ---------------------------------------------------------------- decode -- */

citrinet_ctc_status_t citrinet_ctc_decode(const int8_t *logits,
                                          uint32_t      n_frames,
                                          char         *text,
                                          uint32_t      cap,
                                          uint32_t     *n_written,
                                          uint16_t     *ids)
{
    uint16_t              local[CITRINET_CTC_MAX_FRAMES];
    uint16_t             *dst = (ids != 0) ? ids : local;
    citrinet_ctc_status_t st  = citrinet_ctc_argmax(logits, n_frames, dst);

    if (st == CITRINET_CTC_OK) {
        st = citrinet_ctc_ids_to_text(dst, n_frames, text, cap, n_written);
    } else {
        if ((text != 0) && (cap != 0u)) {
            text[0] = '\0';
        }
        if (n_written != 0) {
            *n_written = 0u;
        }
    }
    return st;
}

/* ----------------------------------------------------------------- piece -- */

const char *citrinet_ctc_piece(uint32_t id)
{
    return (id < CITRINET_VOCAB_SIZE) ? citrinet_piece(id) : "";
}

/* citrinet_ctc.h -- greedy CTC decode + SentencePiece detokenisation.
 *
 * Decodes the Citrinet-256 CTC head's int8 logit tensor into ASCII text.
 *
 * Why this operates on int8 and never dequantises
 * -----------------------------------------------
 * The compiled network's output is STAI_FORMAT_S8, shape {100, 1025, 1},
 * CHANNEL_FIRST, with a *per-tensor* scale (0.265415638685226) and offset 0
 * (compile/reports/g800_real/io_contract.h).  A single positive affine map
 * applied to every element is order-preserving, so
 *
 *     argmax_v (s * (q[v] - z))  ==  argmax_v q[v]      for s > 0
 *
 * i.e. argmax over the raw int8 is *exactly* argmax over the dequantised
 * float.  Dequantisation would cost 102,500 multiplies and change nothing.
 * (This would NOT hold for a per-channel/per-axis output scale.)
 *
 * Memory model
 * ------------
 * No allocation of any kind.  Every output goes to a caller-supplied buffer
 * with an explicit capacity.  No printf, no errno, no globals other than the
 * const vocabulary tables pulled in by citrinet_vocab.h.  All loops are
 * bounded by compile-time constants or by validated arguments.
 *
 * Include citrinet_vocab.h from NOWHERE else: its tables are file-static and
 * citrinet_ctc.c is their single translation unit.
 */

#ifndef CITRINET_CTC_H
#define CITRINET_CTC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Geometry of the shipped 8 s network: {100 frames, 1025 classes, 1}. */
#define CITRINET_CTC_FRAMES      100u
#define CITRINET_CTC_CLASSES     1025u
#define CITRINET_CTC_BLANK       1024u

/* Upper bound accepted for n_frames, so every loop is provably bounded even
 * if a caller passes a corrupt length.  150 frames == the 12 s window. */
#define CITRINET_CTC_MAX_FRAMES  256u

/* Worst-case text: every frame emits the longest piece (12 B), nothing
 * collapses.  100 * 12 + 1 for the NUL.  A buffer of this size can never
 * truncate at CITRINET_CTC_FRAMES frames. */
#define CITRINET_CTC_TEXT_CAP    1201u

/* Sentinel stored in the ids array for "no valid frame"; never a real id. */
#define CITRINET_CTC_NO_ID       0xFFFFu

typedef enum {
    CITRINET_CTC_OK      = 0,   /* complete result written                  */
    CITRINET_CTC_E_ARG   = 1,   /* NULL pointer, zero/oversized n_frames,
                                   zero capacity, or an out-of-range id     */
    CITRINET_CTC_E_TRUNC = 2    /* text did not fit; buffer holds a valid,
                                   NUL-terminated prefix                    */
} citrinet_ctc_status_t;

/* Per-frame argmax over the int8 logits.  This is what Gate 4 compares
 * against host ONNX Runtime; it needs no vocabulary and no text buffer.
 *
 *   logits    [n_frames * 1025] int8, frame-major (frame t at &logits[t*1025])
 *   n_frames  1 .. CITRINET_CTC_MAX_FRAMES
 *   ids       [n_frames] out, one class id per frame
 *
 * Ties resolve to the LOWEST class id, matching numpy.argmax.
 */
citrinet_ctc_status_t citrinet_ctc_argmax(const int8_t *logits,
                                          uint32_t      n_frames,
                                          uint16_t     *ids);

/* Collapse repeats, drop blanks, concatenate pieces, U+2581 -> ' ', trim.
 *
 *   ids        [n_frames] as produced by citrinet_ctc_argmax()
 *   text       out, always NUL-terminated when cap >= 1
 *   cap        sizeof(text), INCLUDING room for the NUL; must be >= 1
 *   n_written  optional, may be NULL; strlen(text) on return
 *
 * Returns CITRINET_CTC_E_TRUNC (not E_ARG) if the text did not fit; the
 * buffer then holds a valid NUL-terminated prefix and *n_written its length.
 */
citrinet_ctc_status_t citrinet_ctc_ids_to_text(const uint16_t *ids,
                                               uint32_t        n_frames,
                                               char           *text,
                                               uint32_t        cap,
                                               uint32_t       *n_written);

/* argmax + detokenise in one call.
 *
 *   ids  optional, may be NULL; if non-NULL, receives the per-frame argmax
 *        (same buffer contract as citrinet_ctc_argmax).
 *
 * When ids == NULL an internal CITRINET_CTC_MAX_FRAMES-entry automatic array
 * is used (512 B of stack), so the call still allocates nothing.
 */
citrinet_ctc_status_t citrinet_ctc_decode(const int8_t *logits,
                                          uint32_t      n_frames,
                                          char         *text,
                                          uint32_t      cap,
                                          uint32_t     *n_written,
                                          uint16_t     *ids);

/* Piece id -> NUL-terminated ASCII string, for callers that want to print the
 * token stream.  Returns "" for an out-of-range id; the blank id 1024 maps to
 * the literal piece stored for it in the vocabulary, so callers that mean
 * "blank" must test against CITRINET_CTC_BLANK themselves. */
const char *citrinet_ctc_piece(uint32_t id);

#ifdef __cplusplus
}
#endif

#endif /* CITRINET_CTC_H */

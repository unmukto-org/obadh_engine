/*
 * obadh.h — stable C ABI for the Obadh transliteration engine.
 *
 * Built when the crate is compiled with the `cabi` feature (native targets only).
 * Link the resulting cdylib/staticlib and include this header.
 *
 * This header is the authoritative *signature*; the project README is the
 * authoritative *behavior* — the linguistic rules, per-function semantics, and a
 * copy-ready client-side auto-insert policy built on suggest_detailed +
 * word_frequency. See its "C ABI", "Autocorrect", and "Autosuggest" sections:
 *   https://github.com/nsssayom/obadh_engine#c-abi
 *
 * Conventions
 * -----------
 * Sizing (snprintf-style): every function that writes bytes takes an output
 *   pointer and capacity and returns the number of bytes the result needs. It
 *   copies only when the buffer is large enough. Pass a small stack scratch;
 *   reallocate and call again only if the return exceeds the capacity.
 *
 * String lists (count + length-prefixed records, no delimiter): a list is packed
 *   into one buffer as little-endian [uint32 count] followed by `count` records
 *   of [uint32 byte_len][utf8 bytes]. No in-band separator — a candidate may
 *   contain any bytes, and an empty string is faithful.
 *
 * Handles: opaque pointers from *_open / *_new, released by the matching *_free.
 *   Do not use a handle after freeing it, and do not use one handle from multiple
 *   threads at once (no internal locking; the caller owns synchronization).
 *
 * UTF-8: inputs are (ptr, len) UTF-8 spans. Invalid UTF-8 makes the call a no-op
 *   (returns 0 / false / null). A zero length is the empty string.
 *
 * The ABI version (obadh_abi_version) is independent of the crate's semver:
 * additive symbols do not bump it.
 */
#ifndef OBADH_H
#define OBADH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OBADH_ABI_VERSION 2

/* Opaque handles. */
typedef struct ObadhEngine ObadhEngine;
typedef struct ObadhAutocorrect ObadhAutocorrect;
typedef struct ObadhAutosuggest ObadhAutosuggest;

/* ----------------------------------------------------------------- version */

uint32_t obadh_abi_version(void);
size_t   obadh_engine_version(uint8_t *out, size_t cap);

/* ------------------------------------------------------ deterministic engine */

ObadhEngine *obadh_engine_new(void);
void         obadh_engine_free(ObadhEngine *engine);

size_t obadh_transliterate(const ObadhEngine *engine,
                           const uint8_t *input, size_t input_len,
                           uint8_t *out, size_t cap);
size_t obadh_transliterate_lenient(const ObadhEngine *engine,
                                   const uint8_t *input, size_t input_len,
                                   uint8_t *out, size_t cap);

/* ----------------------------------------------------------------- autocorrect */

/* Open from a bn.fst path and an optional loanword FST path (length 0 to omit).
 * Returns NULL on failure. */
ObadhAutocorrect *obadh_autocorrect_open(const uint8_t *fst_path, size_t fst_path_len,
                                         const uint8_t *loanword_path, size_t loanword_path_len);
void              obadh_autocorrect_free(ObadhAutocorrect *autocorrect);

/* Content fingerprint of the loaded lexicon FST (crate <-> artifact check). */
uint64_t obadh_autocorrect_fingerprint(const ObadhAutocorrect *autocorrect);

/* Lexicon frequency of `word`: the stored count, on the same scale as
 * suggest_detailed's per-candidate `frequency` (one table). 0 if `word` is not an
 * exact entry (or on invalid input), so presence is `> 0`. This is the baseline
 * signal for a client auto-insert gate's frequency-ratio override; it subsumes a
 * plain membership check. No entry is stored with frequency 0. */
uint64_t obadh_autocorrect_word_frequency(const ObadhAutocorrect *autocorrect,
                                          const uint8_t *word, size_t word_len);

/* Ranked corrections for `roman` with full provenance, as a packed record list:
 *   [uint32 count]
 *     per candidate:
 *       [uint32 text_len][text utf8]
 *       [uint8  source]              // FstCandidateSource code, frozen/append-only:
 *                                    //  0 exact  1 edit_distance  2 diacritic_edit
 *                                    //  3 orthographic_vowel_length  4 prefix_completion
 *                                    //  5 stem_suffix_completion  6 skeleton_vowel_drop
 *                                    //  7 consonant_confusion  8 roman_repair_exact
 *                                    //  9 english_loanword_exact  10 english_loanword_fuzzy
 *                                    // 11 emoticon_exact (opt-in conversion only)
 *                                    // treat an UNKNOWN code as not-auto-replaceable.
 *       [uint16 edit_cost]           // Bangla-side edit distance
 *       [uint16 roman_repair_cost]   // 0xFFFF = none (native-side edit)
 *       [uint64 frequency]           // lexicon frequency of the candidate word
 * The field a client builds its own auto-insert gate on. snprintf-style. */
size_t obadh_autocorrect_suggest_detailed(const ObadhAutocorrect *autocorrect,
                                          const uint8_t *roman, size_t roman_len,
                                          size_t limit, uint8_t *out, size_t cap);

/* Active-typing candidate bar for `roman`: the deterministic baseline first,
 * then corrections. Exact emoticons return the intact literal first, then its
 * emoji. The baseline is always present so the user can keep what they typed
 * even when it is not a lexicon word. Packed string list. */
size_t obadh_compose_suggestions(const ObadhAutocorrect *autocorrect,
                                 const uint8_t *roman, size_t roman_len,
                                 size_t limit, uint8_t *out, size_t cap);

/* Alternative spellings for an already-composed Bengali `word` (a re-correction
 * menu for a committed word). Input is Bengali. Packed string list. */
size_t obadh_autocorrect_word_alternatives(const ObadhAutocorrect *autocorrect,
                                           const uint8_t *word, size_t word_len,
                                           size_t limit, uint8_t *out, size_t cap);

/* ----------------------------------------------------------------- autosuggest */

/* Open from an n-gram artifact path. Returns NULL on failure. */
ObadhAutosuggest *obadh_autosuggest_open(const uint8_t *path, size_t path_len);
void              obadh_autosuggest_free(ObadhAutosuggest *autosuggest);

/* Content fingerprint of the loaded n-gram artifact. */
uint64_t obadh_autosuggest_fingerprint(const ObadhAutosuggest *autosuggest);

/* Commit a token, learning it into the personal overlay. Returns 1 if learned,
 * else 0. */
int32_t obadh_autosuggest_commit(ObadhAutosuggest *autosuggest,
                                 const uint8_t *token, size_t token_len);

/* Next-word suggestions for the current session context as a packed string list.
 * Merges the personal overlay's learned words with the model's. */
size_t obadh_autosuggest_suggest(ObadhAutosuggest *autosuggest, size_t limit,
                                 uint8_t *out, size_t cap);

/* Stateless next-word suggestions for an explicit Bengali `context` string.
 * Model-only — does not use or update the session's learned state. */
size_t obadh_autosuggest_suggest_for_context(const ObadhAutosuggest *autosuggest,
                                             const uint8_t *context, size_t context_len,
                                             size_t limit, uint8_t *out, size_t cap);

void obadh_autosuggest_clear_session(ObadhAutosuggest *autosuggest);
void obadh_autosuggest_clear_personal(ObadhAutosuggest *autosuggest);

/* Export/import the personal overlay snapshot. Export is snprintf-style; import
 * returns 1 on success, else 0. */
size_t  obadh_autosuggest_export_personal(const ObadhAutosuggest *autosuggest,
                                          uint8_t *out, size_t cap);
int32_t obadh_autosuggest_import_personal(ObadhAutosuggest *autosuggest,
                                          const uint8_t *input, size_t input_len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OBADH_H */

//! Exact ASCII emoticon alternatives, independent of the spelling lexicon.

use super::{FstCandidate, FstCandidateSource, FstSuggestResult};

mod table;
use table::{EMOTICON_RULES, MAX_EMOTICON_LEN};

/// Look up a complete, case-sensitive ASCII emoticon without allocating.
///
/// No trimming, substring matching, or fuzzy repair is performed. The caller
/// must supply the intact active input before splitting punctuation.
pub fn emoticon_emoji(input: &str) -> Option<&'static str> {
    // Every alias is short ASCII; reject ordinary words and larger input early.
    if input.len() > MAX_EMOTICON_LEN || !input.is_ascii() {
        return None;
    }
    EMOTICON_RULES
        .binary_search_by(|&(alias, _)| alias.cmp(input))
        .ok()
        .map(|index| EMOTICON_RULES[index].1)
}

/// Return an emoji alternative with a literal baseline for recognized input.
///
/// This channel bypasses dictionary ranking. Its score and frequency are zero;
/// neither expresses spelling confidence. Compose callers must put `baseline`
/// first and never silently replace it unless the client opts into conversion.
/// Unrecognized input returns `None`, allowing normal word correction to run.
pub fn emoticon_suggestions(input: &str, limit: usize) -> Option<FstSuggestResult> {
    let emoji = emoticon_emoji(input)?;
    let candidates = if limit == 0 {
        Vec::new()
    } else {
        vec![FstCandidate {
            text: emoji.to_owned(),
            source: FstCandidateSource::EmoticonExact,
            edit_cost: 0,
            frequency: 0,
            score: 0,
            roman_repair: None,
            roman_repair_kind: None,
            roman_repair_cost: None,
        }]
    };
    Some(FstSuggestResult {
        baseline: input.to_owned(),
        exact_frequency: None,
        max_distance: 0,
        max_edit_cost: None,
        candidate_count: 1,
        returned_candidates: candidates.len(),
        truncated: candidates.is_empty(),
        candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_matches_the_pinned_source_and_is_sorted_unique() {
        let groups: serde_json::Value =
            serde_json::from_str(include_str!("../../data/emoticons/emoticons.json")).unwrap();
        let mut count = 0;
        for group in groups.as_array().unwrap() {
            for alias in group["emoticons"].as_array().unwrap() {
                assert_eq!(
                    emoticon_emoji(alias.as_str().unwrap()),
                    group["emoji"].as_str()
                );
                count += 1;
            }
        }
        assert_eq!(count, EMOTICON_RULES.len());
        assert!(EMOTICON_RULES.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }

    #[test]
    fn both_styles_and_literal_baselines_are_preserved() {
        for (input, emoji) in [(":)", "😃"), (":-)", "😃"), (":D", "😄"), (":-D", "😄")] {
            let result = emoticon_suggestions(input, 2).unwrap();
            assert_eq!(result.baseline, input);
            assert_eq!(result.candidates.len(), 1);
            let candidate = &result.candidates[0];
            assert_eq!(candidate.text, emoji);
            assert_eq!(candidate.source, FstCandidateSource::EmoticonExact);
            assert_eq!(candidate.frequency, 0);
            assert!(candidate.roman_repair.is_none());
        }
    }

    #[test]
    fn only_complete_exact_aliases_are_recognized() {
        for input in [
            "",
            ":",
            ":--",
            ":dd",
            "hi:)",
            " :) ",
            "http://a",
            "😃",
            "বাংলা",
            ":):)",
        ] {
            assert!(emoticon_suggestions(input, 2).is_none(), "{input}");
        }
        let result = emoticon_suggestions(":D", 0).unwrap();
        assert!(result.candidates.is_empty());
        assert_eq!(result.candidate_count, 1);
        assert!(result.truncated);
    }
}

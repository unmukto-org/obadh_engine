//! Definitions for Bengali conjuncts
//!
//! This module provides a comprehensive set of Bengali conjunct definitions
//! based on phonetic components. The engine uses this compiled Rust rule table
//! directly; source CSV data is not parsed or shipped on the runtime path.

mod canonical;
mod rules;
mod trie;

use crate::definitions::consonant_value;
use canonical::{
    canonical_conjunct_part, is_special_form_key, is_ya_phola_marker, ya_phola_attaches_to,
};
use rules::CONJUNCT_RULES;
use trie::ConjunctTrie;

use std::collections::BTreeSet;
use std::sync::OnceLock;

/// Structure to store and manage Bengali conjunct definitions.
#[derive(Debug)]
pub struct ConjunctDefinitions {
    /// Trie for allocation-free conjunct lookup from component parts.
    conjunct_trie: ConjunctTrie,
}

impl ConjunctDefinitions {
    /// Create a new instance of conjunct definitions
    pub fn new() -> Self {
        let conjunct_trie = ConjunctTrie::with_capacity(conjunct_trie_node_capacity());

        let mut instance = ConjunctDefinitions { conjunct_trie };

        for rule in CONJUNCT_RULES {
            instance.add_conjunct(rule.key(), rule.value());
        }

        instance.conjunct_trie.sort_edges();

        instance
    }

    /// Add a conjunct mapping
    fn add_conjunct(&mut self, key: &'static str, value: &'static str) {
        self.conjunct_trie.insert(key, value);
    }

    /// Check if a sequence can form a valid conjunct
    pub fn can_form_conjunct(&self, key: &str) -> bool {
        self.create_conjunct(key).is_some()
    }

    /// Create a conjunct from a sequence of consonants
    pub fn create_conjunct(&self, key: &str) -> Option<&'static str> {
        let node = self.conjunct_trie.advance(self.conjunct_trie.root(), key)?;
        self.conjunct_trie.value(node)
    }

    /// Create a conjunct from already-tokenized component parts without
    /// allocating a joined key.
    pub fn create_conjunct_from_parts(&self, parts: &[&str]) -> Option<&'static str> {
        if parts.len() < 2 {
            return None;
        }

        let mut node = self.conjunct_trie.root();
        for part in parts {
            node = self
                .conjunct_trie
                .advance(node, canonical_conjunct_part(part))?;
        }

        self.conjunct_trie.value(node)
    }

    /// Check component parts without allocating a joined key.
    pub fn can_form_conjunct_from_parts(&self, parts: &[&str]) -> bool {
        self.create_conjunct_from_parts(parts).is_some() || self.is_derived_ya_phola(parts)
    }

    /// Check derived conjunct forms that are intentionally rule-generated
    /// instead of listed in the static source-owned conjunct table.
    pub(crate) fn can_form_derived_conjunct_from_parts(&self, parts: &[&str]) -> bool {
        self.is_derived_ya_phola(parts)
    }

    /// Check whether the current parts can still become a derived conjunct if
    /// the tokenizer consumes one more component.
    pub(crate) fn can_match_derived_conjunct_prefix(&self, parts: &[&str]) -> bool {
        if self.is_derived_ya_phola(parts) {
            return true;
        }
        // The parts so far form a renderable base that could still take a
        // ya-phola once the tokenizer consumes one more component.
        self.base_takes_ya_phola(parts)
    }

    /// A productive ya-phola conjunct: a renderable base (a single consonant or an
    /// enumerated conjunct) directly followed by a ya-phola marker. This lets
    /// loanword clusters such as প্ল্য/ব্ল্য/গ্ল্য form even though they are absent
    /// from the native conjunct table, mirroring how the অ্যা vowel already
    /// attaches to any conjunct.
    fn is_derived_ya_phola(&self, parts: &[&str]) -> bool {
        let Some((phola, base)) = parts.split_last() else {
            return false;
        };
        is_ya_phola_marker(phola) && self.base_takes_ya_phola(base)
    }

    /// Whether `base` is a renderable conjunct base whose final consonant accepts
    /// a ya-phola.
    fn base_takes_ya_phola(&self, base: &[&str]) -> bool {
        let Some((last, rest)) = base.split_last() else {
            return false;
        };
        // A ya-phola composes onto any real consonant base, including the tail of a
        // conjunct (প্ল + য = প্ল্য). Standalone র refuses it (রয়া, kept distinct
        // from reph-ya র‌্য), but an r-phola tail inside a multi-consonant conjunct
        // is not a standalone র and does take the ya-phola (ট্র + য = ট্র্য, used by
        // loanwords such as ট্র্যাক).
        let takes = ya_phola_attaches_to(last)
            || (!rest.is_empty() && canonical_conjunct_part(last) == "r");
        takes && self.base_is_renderable(base)
    }

    /// Whether `base` renders on its own: a single consonant, or an enumerated
    /// conjunct. (Derived ya-phola forms are not themselves valid ya-phola bases.)
    fn base_is_renderable(&self, base: &[&str]) -> bool {
        match base {
            [] => false,
            [single] => consonant_value(canonical_conjunct_part(single)).is_some(),
            _ => self.create_conjunct_from_parts(base).is_some(),
        }
    }

    /// Return the root trie cursor for incremental conjunct matching.
    pub(crate) fn conjunct_match_root(&self) -> usize {
        self.conjunct_trie.root()
    }

    /// Advance an incremental conjunct match by one romanized component.
    pub(crate) fn advance_conjunct_match(&self, node: usize, part: &str) -> Option<usize> {
        self.conjunct_trie
            .advance(node, canonical_conjunct_part(part))
    }

    /// Return the conjunct value at an incremental trie cursor, if terminal.
    pub(crate) fn conjunct_match_value(&self, node: usize) -> Option<&'static str> {
        self.conjunct_trie.value(node)
    }

    /// Get romanized consonants for a conjunct
    pub fn get_components(&self, conjunct: &str) -> Option<Vec<String>> {
        for rule in CONJUNCT_RULES {
            if rule.value() == conjunct {
                return Some(self.components_for_key(rule.key()));
            }
        }
        None
    }

    fn components_for_key(&self, key: &str) -> Vec<String> {
        let mut components = Vec::new();
        let mut i = 0;
        while i < key.len() {
            let mut found = false;
            for len in (1..=key.len() - i).rev() {
                let substr = &key[i..i + len];
                if consonant_value(substr).is_some() || is_special_form_key(substr) {
                    components.push(substr.to_string());
                    i += len;
                    found = true;
                    break;
                }
            }
            if !found {
                components.push(key[i..i + 1].to_string());
                i += 1;
            }
        }
        components
    }

    /// Check if a given sequence is a valid conjunct
    pub fn is_valid_conjunct(&self, components: &[String]) -> bool {
        if components.is_empty() {
            return false;
        }

        let mut node = self.conjunct_trie.root();
        for component in components {
            let Some(next_node) = self
                .conjunct_trie
                .advance(node, canonical_conjunct_part(component))
            else {
                return false;
            };
            node = next_node;
        }

        self.conjunct_trie.value(node).is_some()
    }

    /// Get all valid conjuncts
    pub fn get_all_valid_conjuncts(&self) -> BTreeSet<&'static str> {
        CONJUNCT_RULES.iter().map(|rule| rule.key()).collect()
    }

    /// Check if a form is a special form (like reph, ya-phola, ba-phola)
    pub fn is_special_form(&self, form: &str) -> bool {
        is_special_form_key(form)
    }
}

impl Default for ConjunctDefinitions {
    fn default() -> Self {
        Self::new()
    }
}

/// Return a singleton instance of ConjunctDefinitions
pub fn conjuncts() -> &'static ConjunctDefinitions {
    static INSTANCE: OnceLock<ConjunctDefinitions> = OnceLock::new();
    INSTANCE.get_or_init(ConjunctDefinitions::new)
}

fn conjunct_trie_node_capacity() -> usize {
    1 + CONJUNCT_RULES
        .iter()
        .map(|rule| rule.key().len())
        .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conjunct_rule_table_has_unique_keys() {
        let mut keys = BTreeSet::new();

        for rule in CONJUNCT_RULES {
            assert!(!rule.key().is_empty());
            assert!(!rule.value().is_empty());
            assert!(
                keys.insert(rule.key()),
                "duplicate conjunct rule key: {}",
                rule.key()
            );
        }
    }

    #[test]
    fn conjunct_definitions_load_every_static_rule() {
        let definitions = ConjunctDefinitions::new();

        assert_eq!(
            definitions.get_all_valid_conjuncts().len(),
            CONJUNCT_RULES.len()
        );

        for rule in CONJUNCT_RULES {
            assert_eq!(definitions.create_conjunct(rule.key()), Some(rule.value()));
        }
    }

    #[test]
    fn derived_ya_phola_stacks_on_r_phola_conjunct_tail_not_standalone_r() {
        let defs = ConjunctDefinitions::new();

        // An r-phola tail inside a multi-consonant conjunct takes ya-phola
        // (ট্র + য = ট্র্য, ক্র + য = ক্র্য, স্ত্র + য = স্ত্র্য).
        assert!(defs.can_form_derived_conjunct_from_parts(&["T", "r", "y"]));
        assert!(defs.can_form_derived_conjunct_from_parts(&["k", "r", "Y"]));
        assert!(defs.can_form_derived_conjunct_from_parts(&["s", "t", "r", "y"]));
        // A standalone র refuses ya-phola (রয়া, kept distinct from reph-ya র‌্য).
        assert!(!defs.can_form_derived_conjunct_from_parts(&["r", "y"]));
        // The l-phola tail still works; a base ending in a phola marker still refuses.
        assert!(defs.can_form_derived_conjunct_from_parts(&["p", "l", "y"]));
        assert!(!defs.can_form_derived_conjunct_from_parts(&["sh", "w", "y"]));
    }
}

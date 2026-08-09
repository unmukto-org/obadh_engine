//! Input sanitization for the Obadh Engine
//!
//! This module provides functions to validate and sanitize input text
//! before passing it to the transliteration engine.

use std::collections::{BTreeSet, HashSet};

/// Result of sanitization, containing either the sanitized string or an error message
pub type SanitizeResult = Result<String, String>;

/// Sanitizer for input text
#[derive(Default)]
pub struct Sanitizer {
    /// Optional caller-provided additions to the default deterministic input contract.
    extra_allowed_chars: Option<HashSet<char>>,
}

impl Sanitizer {
    /// Create a new sanitizer with the default allowed character set
    pub fn new() -> Self {
        Self::default()
    }

    /// Add additional allowed characters to the sanitizer
    pub fn with_allowed_chars(mut self, chars: &[char]) -> Self {
        let extra_allowed_chars = self
            .extra_allowed_chars
            .get_or_insert_with(|| HashSet::with_capacity(chars.len()));

        for &c in chars {
            extra_allowed_chars.insert(c);
        }
        self
    }

    pub(crate) fn is_allowed_char(&self, c: char) -> bool {
        is_default_allowed_char(c)
            || self
                .extra_allowed_chars
                .as_ref()
                .is_some_and(|chars| chars.contains(&c))
    }

    /// Sanitize the input text, ensuring it contains only allowed characters
    ///
    /// Returns the sanitized string if successful, or an error message if invalid characters are found
    pub fn sanitize(&self, input: &str) -> SanitizeResult {
        let mut invalid_chars = BTreeSet::new();

        // Check for invalid characters
        for c in input.chars() {
            if !self.is_allowed_char(c) {
                invalid_chars.insert(c);
            }
        }

        // If there are invalid characters, return an error
        if !invalid_chars.is_empty() {
            let invalid_list: String = invalid_chars.into_iter().collect();
            return Err(format!("Invalid characters found: {}", invalid_list));
        }

        // Otherwise, return the sanitized string
        Ok(input.to_string())
    }

    /// Remove invalid characters from the input and return the sanitized string
    pub fn clean(&self, input: &str) -> String {
        input.chars().filter(|&c| self.is_allowed_char(c)).collect()
    }

    /// Check if a string contains only valid characters
    pub fn is_valid(&self, input: &str) -> bool {
        input.chars().all(|c| self.is_allowed_char(c))
    }
}

fn is_default_allowed_char(c: char) -> bool {
    c.is_whitespace()
        || c.is_ascii_alphanumeric()
        || matches!(
            c,
            '\u{0980}'
                ..='\u{09FF}' // Bengali block
                | '\u{0964}' // dari (।)
                | '\u{0965}' // double dari (॥)
                | '\u{200C}' // zero-width non-joiner
                | '\u{200D}' // zero-width joiner
                | ','
                | '.'
                | ':'
                | ';'
                | '!'
                | '?'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '"'
                | '\''
                | '`'
                | '-'
                | '_'
                | '+'
                | '='
                | '/'
                | '\\'
                | '|'
                | '@'
                | '#'
                | '$'
                | '%'
                | '^'
                | '&'
                | '*'
                | '<'
                | '>'
        )
}

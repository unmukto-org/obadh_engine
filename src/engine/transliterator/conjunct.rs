use crate::definitions::{consonant_value, diacritics::HASANT};

use super::Transliterator;

impl Transliterator {
    pub(super) fn append_reph_prefix(output: &mut String) {
        output.push('র');
        output.push_str(HASANT);
    }

    fn conjunct_component(&self, part: &str) -> Option<&'static str> {
        match part {
            // Reph-ya loanword form র‍্য. The joiner is ZWJ (U+200D), the
            // Unicode-prescribed encoding for this form (র‍্যাব, র‍্যান্ডম),
            // distinct from reph-ya র্য and the র + য় glide রয়.
            "rZ" => Some("র\u{200D}"),
            "y" | "Y" => Some("য"),
            "w" => Some("ব"),
            _ => consonant_value(part),
        }
    }

    pub(super) fn append_conjunct_parts(&self, output: &mut String, parts: &[&str]) -> bool {
        let checkpoint = output.len();
        if self.append_conjunct_parts_unchecked(output, parts) {
            true
        } else {
            output.truncate(checkpoint);
            false
        }
    }

    fn append_conjunct_parts_unchecked(&self, output: &mut String, parts: &[&str]) -> bool {
        if parts.len() < 2 {
            return false;
        }

        if let Some(mapped) = self.conjuncts.create_conjunct_from_parts(parts) {
            output.push_str(mapped);
            return true;
        }

        if parts.first() == Some(&"rr") {
            Self::append_reph_prefix(output);
            return self.append_conjunct_parts_unchecked(output, &parts[1..]);
        }

        for (index, part) in parts.iter().enumerate() {
            let Some(component) = self.conjunct_component(part) else {
                return false;
            };
            output.push_str(component);
            if index < parts.len() - 1 {
                output.push_str(HASANT);
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_conjunct_parts_rolls_back_invalid_parts() {
        let transliterator = Transliterator::new();
        let mut output = String::from("আগে");

        assert!(!transliterator.append_conjunct_parts(&mut output, &["rr", "not-a-part"]));
        assert_eq!(output, "আগে");
    }
}

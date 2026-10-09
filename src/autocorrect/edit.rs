use super::bangla::{bangla_units, unit_similarity};

/// A weighted edit cost, saturated at `u16::MAX` for larger distances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EditCost(pub u16);

pub(crate) const INSERT_DELETE_COST: u16 = 2;

pub fn weighted_edit_distance(left: &str, right: &str) -> EditCost {
    if left == right {
        return EditCost(0);
    }
    let left_units = bangla_units(left);
    let right_units = bangla_units(right);
    weighted_unit_edit_distance(&left_units, &right_units)
}

pub(crate) fn weighted_unit_edit_distance(left: &[&str], right: &[&str]) -> EditCost {
    if left.is_empty() {
        return EditCost(insertion_deletion_cost(right.len()));
    }
    if right.is_empty() {
        return EditCost(insertion_deletion_cost(left.len()));
    }

    // The costs are symmetric; keep working memory proportional to the shorter
    // input rather than allocating two rows for the longer one.
    let (left, right) = if left.len() < right.len() {
        (right, left)
    } else {
        (left, right)
    };
    let mut previous = (0..=right.len())
        .map(insertion_deletion_cost)
        .collect::<Vec<_>>();
    let mut current = vec![0; right.len() + 1];

    for (left_index, left_unit) in left.iter().enumerate() {
        current[0] = insertion_deletion_cost(left_index + 1);
        for (right_index, right_unit) in right.iter().enumerate() {
            let substitution =
                previous[right_index].saturating_add(unit_similarity(left_unit, right_unit));
            let deletion = previous[right_index + 1].saturating_add(INSERT_DELETE_COST);
            let insertion = current[right_index].saturating_add(INSERT_DELETE_COST);
            current[right_index + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    EditCost(previous[right.len()])
}

pub(crate) fn insertion_deletion_cost(unit_count: usize) -> u16 {
    unit_count
        .saturating_mul(usize::from(INSERT_DELETE_COST))
        .min(usize::from(u16::MAX)) as u16
}

#[cfg(test)]
mod tests {
    use super::{insertion_deletion_cost, weighted_edit_distance};

    #[test]
    fn weighted_edit_distance_is_bangla_unit_aware() {
        assert_eq!(weighted_edit_distance("বিজ্ঞান", "বিজ্ঞান").0, 0);
        assert!(weighted_edit_distance("বিজান", "বিজ্ঞান").0 <= 2);
        assert!(weighted_edit_distance("কিরণ", "করণ").0 < weighted_edit_distance("আম", "বিজ্ঞান").0);
    }

    #[test]
    fn mark_discounts_preserve_consonant_identity() {
        for (left, right, cost) in [
            ("কি", "মি", 3),
            ("কি", "কী", 1),
            ("কি", "ক", 3),
            ("কাঁ", "কা", 1),
            ("কাঁ", "মা", 3),
            ("ক্ষি", "ক্ষী", 1),
            ("ক্ষি", "ক্মী", 2),
            ("ড\u{09BC}ি", "ডি", 3),
        ] {
            assert_eq!(
                weighted_edit_distance(left, right).0,
                cost,
                "{left} → {right}"
            );
            assert_eq!(
                weighted_edit_distance(right, left).0,
                cost,
                "{right} → {left}"
            );
        }
        // Joiners remain part of consonant identity; differing segmentation
        // may add costs, but a shared vowel sign must not make this a cheap edit.
        assert!(weighted_edit_distance("ক্\u{200D}ষি", "ক্ষি").0 > 1);
    }

    #[test]
    fn long_edit_distances_saturate_without_wrapping() {
        for count in [32_767, 32_768, 65_536] {
            let text = "ক".repeat(count);
            let expected = (count * 2).min(usize::from(u16::MAX)) as u16;
            assert_eq!(weighted_edit_distance("", &text).0, expected);
            assert_eq!(weighted_edit_distance(&text, "").0, expected);
            assert_eq!(weighted_edit_distance(&text, &text).0, 0);
        }
        assert_eq!(insertion_deletion_cost(usize::MAX), u16::MAX);
    }

    #[test]
    fn saturation_preserves_representable_costs_in_nonempty_rows() {
        let text = "ক".repeat(32_768);
        assert_eq!(weighted_edit_distance(&text, "ক").0, 65_534);
        assert_eq!(weighted_edit_distance("ক", &text).0, 65_534);
        assert_eq!(weighted_edit_distance(&text, "ম").0, u16::MAX);
        assert_eq!(weighted_edit_distance("ম", &text).0, u16::MAX);
    }
}

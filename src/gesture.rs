//! Platform-neutral, lexicon-constrained CTC search for phonetic word gestures.
//!
//! A touch encoder supplies log emissions for `a`..`z` and blank at each frame.
//! This module does not know about UIKit, Android, a model format, or a keyboard
//! layout. It only searches engine-verified Roman paths and returns Bangla words.
//! Loading is a one-time operation; no corpus parsing occurs on a touch thread.

use std::collections::HashMap;

const MAX_MODEL_BYTES: usize = 8 * 1024 * 1024;
const MAX_ENTRIES: usize = 100_000;
const MAX_FRAMES: usize = 128;
const MAX_BEAM: usize = 512;
const NEG_INFINITY: f32 = f32::NEG_INFINITY;

#[derive(Debug, PartialEq, Eq)]
pub enum GestureModelError {
    TooLarge,
    InvalidUtf8,
    InvalidRow,
    TooManyEntries,
    Empty,
}

#[derive(Default)]
struct Node {
    edges: Vec<(u8, usize)>,
    outputs: Vec<(String, u32)>,
}

/// The Roman search path comes from the deterministic engine's verified
/// inverse vocabulary. One spelling may lead to several Bengali words.
pub struct GestureLexicon {
    nodes: Vec<Node>,
    entries: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GestureCandidate {
    pub spelling: String,
    pub word: String,
    pub score: f32,
}

#[derive(Clone)]
struct Beam {
    path: Vec<u8>,
    node: usize,
    blank: f32,
    nonblank: f32,
}

impl Beam {
    fn total(&self) -> f32 {
        log_add(self.blank, self.nonblank)
    }
}

fn log_add(a: f32, b: f32) -> f32 {
    if a == NEG_INFINITY {
        return b;
    }
    if b == NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + (-(a - b).abs()).exp().ln_1p()
}

impl GestureLexicon {
    /// Read the engine-verified `roman<TAB>bangla<TAB>frequency` artifact.
    /// The caller must verify the artifact hash before loading. Input size,
    /// path length, count, UTF-8, and frequency are bounded even for bad data.
    pub fn from_tsv(data: &[u8]) -> Result<Self, GestureModelError> {
        if data.len() > MAX_MODEL_BYTES {
            return Err(GestureModelError::TooLarge);
        }
        let text = std::str::from_utf8(data).map_err(|_| GestureModelError::InvalidUtf8)?;
        let mut model = Self {
            nodes: vec![Node::default()],
            entries: 0,
        };
        for line in text.lines() {
            let mut parts = line.split('\t');
            let (Some(path), Some(word), Some(frequency), None) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                return Err(GestureModelError::InvalidRow);
            };
            if !(1..=28).contains(&path.len())
                || !path.bytes().all(|b| b.is_ascii_lowercase())
                || word.is_empty()
                || word.len() > 96
                || !word.chars().all(|c| ('\u{0980}'..='\u{09ff}').contains(&c))
            {
                return Err(GestureModelError::InvalidRow);
            }
            let frequency: u32 = frequency
                .parse()
                .map_err(|_| GestureModelError::InvalidRow)?;
            if frequency == 0 {
                return Err(GestureModelError::InvalidRow);
            }
            model.entries += 1;
            if model.entries > MAX_ENTRIES {
                return Err(GestureModelError::TooManyEntries);
            }
            model.insert(path.as_bytes(), word, frequency);
        }
        if model.entries == 0 {
            return Err(GestureModelError::Empty);
        }
        Ok(model)
    }

    pub fn entry_count(&self) -> usize {
        self.entries
    }

    fn insert(&mut self, path: &[u8], word: &str, frequency: u32) {
        let mut node = 0;
        for &key in path {
            let next = self.nodes[node]
                .edges
                .iter()
                .find(|(letter, _)| *letter == key)
                .map(|(_, index)| *index);
            node = match next {
                Some(next) => next,
                None => {
                    let next = self.nodes.len();
                    self.nodes.push(Node::default());
                    self.nodes[node].edges.push((key, next));
                    next
                }
            };
        }
        self.nodes[node].outputs.push((word.to_owned(), frequency));
    }

    /// Decode 27-value log-probability frames: `a`..`z`, then CTC blank.
    /// Returns no candidate for malformed emissions. A caller should never
    /// auto-insert a low-confidence result; calibrated confidence is a separate
    /// product decision based on human traces and language context.
    pub fn decode(
        &self,
        frames: &[[f32; 27]],
        beam_width: usize,
        limit: usize,
    ) -> Vec<GestureCandidate> {
        if frames.is_empty()
            || frames.len() > MAX_FRAMES
            || beam_width == 0
            || beam_width > MAX_BEAM
            || !(1..=8).contains(&limit)
            || frames
                .iter()
                .flatten()
                .any(|score| score.is_nan() || *score == f32::INFINITY)
        {
            return Vec::new();
        }
        let mut beams = vec![Beam {
            path: Vec::new(),
            node: 0,
            blank: 0.0,
            nonblank: NEG_INFINITY,
        }];
        for frame in frames {
            let mut next: HashMap<Vec<u8>, Beam> = HashMap::with_capacity(beams.len() * 12);
            for beam in &beams {
                let unchanged = next.entry(beam.path.clone()).or_insert_with(|| Beam {
                    path: beam.path.clone(),
                    node: beam.node,
                    blank: NEG_INFINITY,
                    nonblank: NEG_INFINITY,
                });
                unchanged.blank = log_add(unchanged.blank, beam.total() + frame[26]);
                if let Some(last) = beam.path.last() {
                    unchanged.nonblank = log_add(
                        unchanged.nonblank,
                        beam.nonblank + frame[(last - b'a') as usize],
                    );
                }
                for &(letter, child) in &self.nodes[beam.node].edges {
                    let source = if beam.path.last() == Some(&letter) {
                        beam.blank
                    } else {
                        beam.total()
                    };
                    if source == NEG_INFINITY {
                        continue;
                    }
                    let mut path = beam.path.clone();
                    path.push(letter);
                    let extended = next.entry(path.clone()).or_insert_with(|| Beam {
                        path,
                        node: child,
                        blank: NEG_INFINITY,
                        nonblank: NEG_INFINITY,
                    });
                    extended.nonblank =
                        log_add(extended.nonblank, source + frame[(letter - b'a') as usize]);
                }
            }
            beams = next.into_values().collect();
            beams.sort_unstable_by(|a, b| {
                b.total()
                    .total_cmp(&a.total())
                    .then_with(|| a.path.cmp(&b.path))
            });
            beams.truncate(beam_width);
        }
        let mut words: HashMap<&str, GestureCandidate> = HashMap::new();
        for beam in &beams {
            let spelling = String::from_utf8(beam.path.clone()).expect("paths are ASCII");
            for (word, frequency) in &self.nodes[beam.node].outputs {
                let score = beam.total() + 0.10 * (*frequency as f32).ln_1p();
                let candidate = GestureCandidate {
                    spelling: spelling.clone(),
                    word: word.clone(),
                    score,
                };
                match words.get(word.as_str()) {
                    Some(previous) if previous.score >= score => {}
                    _ => {
                        words.insert(word, candidate);
                    }
                }
            }
        }
        let mut candidates: Vec<_> = words.into_values().collect();
        candidates.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.word.cmp(&b.word))
        });
        candidates.truncate(limit);
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emissions(path: &str) -> Vec<[f32; 27]> {
        let mut frames = Vec::new();
        for letter in path.bytes() {
            let mut frame = [-12.0; 27];
            frame[(letter - b'a') as usize] = -0.01;
            frames.push(frame);
            let mut blank = [-12.0; 27];
            blank[26] = -0.01;
            frames.push(blank);
        }
        frames
    }

    #[test]
    fn searches_bangla_paths_and_repeated_letters() {
        let model =
            GestureLexicon::from_tsv("ami\tআমি\t900\nall\tসব\t700\namir\tআমির\t80\n".as_bytes())
                .unwrap();
        assert_eq!(model.entry_count(), 3);
        assert_eq!(model.decode(&emissions("ami"), 32, 3)[0].word, "আমি");
        assert_eq!(model.decode(&emissions("all"), 32, 3)[0].word, "সব");
    }

    #[test]
    fn rejects_bad_artifacts_and_scores() {
        assert_eq!(
            GestureLexicon::from_tsv(b"").err(),
            Some(GestureModelError::Empty)
        );
        assert_eq!(
            GestureLexicon::from_tsv(b"ami\tword\t2\n").err(),
            Some(GestureModelError::InvalidRow)
        );
        let model = GestureLexicon::from_tsv("ami\tআমি\t900\n".as_bytes()).unwrap();
        let mut frame = [-1.0; 27];
        frame[0] = f32::NAN;
        assert!(model.decode(&[frame], 32, 3).is_empty());
        assert!(model.decode(&emissions("ami"), 0, 3).is_empty());
    }

    #[test]
    fn verifies_an_explicit_real_gesture_artifact() {
        let Ok(path) = std::env::var("OBADH_GESTURE_TSV") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        let started = std::time::Instant::now();
        let model = GestureLexicon::from_tsv(&data).unwrap();
        let load = started.elapsed();
        assert!(model.entry_count() >= 60_000);
        for (spelling, expected) in [("ami", "আমি"), ("tumi", "তুমি"), ("bhalo", "ভালো")]
        {
            let started = std::time::Instant::now();
            let candidates = model.decode(&emissions(spelling), 100, 3);
            assert_eq!(candidates[0].word, expected, "{spelling}: {candidates:?}");
            eprintln!("gesture {spelling}: {:?}", started.elapsed());
        }
        eprintln!(
            "gesture model: {} entries loaded in {load:?}",
            model.entry_count()
        );
    }
}

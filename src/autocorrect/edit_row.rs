use std::ops::Deref;

// Keyboard-time searches permit at most two edits. Capping cells at limit + 1
// makes u8 sufficient; short rows stay inline to avoid allocating at each FST
// edge. Longer queries retain the same algorithm through the heap fallback.
const INLINE_CELLS: usize = 32;

#[derive(Debug, Clone)]
pub(super) enum BoundedEditRow {
    Inline {
        cells: [u8; INLINE_CELLS],
        len: usize,
    },
    Heap(Vec<u8>),
}

impl BoundedEditRow {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        if capacity <= INLINE_CELLS {
            Self::Inline {
                cells: [0; INLINE_CELLS],
                len: 0,
            }
        } else {
            Self::Heap(Vec::with_capacity(capacity))
        }
    }

    pub(super) fn initial(query_len: usize, ceiling: u8) -> Self {
        let mut row = Self::with_capacity(query_len + 1);
        for index in 0..=query_len {
            row.push(index.min(usize::from(ceiling)) as u8);
        }
        row
    }

    pub(super) fn push(&mut self, value: u8) {
        match self {
            Self::Inline { cells, len } => {
                cells[*len] = value;
                *len += 1;
            }
            Self::Heap(cells) => cells.push(value),
        }
    }
}

impl Deref for BoundedEditRow {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Inline { cells, len } => &cells[..*len],
            Self::Heap(cells) => cells,
        }
    }
}

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackedBinaryRows {
    height: usize,
    words_per_row: usize,
    words: Vec<u64>,
}

impl PackedBinaryRows {
    pub(crate) fn from_sparse(rows: Vec<Vec<usize>>, width: usize) -> PackedBinaryRows {
        let height = rows.len();
        let words_per_row = width.div_ceil(u64::BITS as usize);
        let mut packed = PackedBinaryRows {
            height,
            words_per_row,
            words: vec![0; height * words_per_row],
        };

        for (row, entries) in rows.into_iter().enumerate() {
            for col in entries {
                packed.set(row, col);
            }
        }

        packed
    }

    pub(crate) fn contains(&self, row: usize, col: usize) -> bool {
        let word = self.word_index(row, col);
        (self.words[word] & bit_mask(col)) != 0
    }

    pub(crate) fn swap_rows(&mut self, a: usize, b: usize) {
        assert!(a < self.height);
        assert!(b < self.height);
        if a == b {
            return;
        }

        let a_start = self.row_start(a);
        let b_start = self.row_start(b);
        for offset in 0..self.words_per_row {
            self.words.swap(a_start + offset, b_start + offset);
        }
    }

    pub(crate) fn xor_suffix_if_contains(
        &mut self,
        dest: usize,
        src: usize,
        start_col: usize,
    ) -> bool {
        if !self.contains(dest, start_col) {
            return false;
        }

        let first_word = start_col / u64::BITS as usize;
        let first_mask = u64::MAX << (start_col % u64::BITS as usize);
        let dest_start = self.row_start(dest);
        let src_start = self.row_start(src);

        self.words[dest_start + first_word] ^= self.words[src_start + first_word] & first_mask;
        for offset in (first_word + 1)..self.words_per_row {
            self.words[dest_start + offset] ^= self.words[src_start + offset];
        }
        true
    }

    pub(crate) fn is_zero(&self, row: usize) -> bool {
        assert!(row < self.height);
        let start = self.row_start(row);
        self.words[start..start + self.words_per_row]
            .iter()
            .all(|&word| word == 0)
    }

    fn set(&mut self, row: usize, col: usize) {
        let word = self.word_index(row, col);
        self.words[word] |= bit_mask(col);
    }

    fn row_start(&self, row: usize) -> usize {
        row * self.words_per_row
    }

    fn word_index(&self, row: usize, col: usize) -> usize {
        assert!(row < self.height);
        self.row_start(row) + col / u64::BITS as usize
    }
}

fn bit_mask(col: usize) -> u64 {
    1u64 << (col % u64::BITS as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_suffix_preserves_lower_columns() {
        let rows = vec![vec![1, 3, 70], vec![0, 3, 64, 70]];
        let mut packed = PackedBinaryRows::from_sparse(rows, 96);

        assert!(packed.xor_suffix_if_contains(1, 0, 3));

        assert!(packed.contains(1, 0));
        assert!(!packed.contains(1, 3));
        assert!(packed.contains(1, 64));
        assert!(!packed.contains(1, 70));
    }
}

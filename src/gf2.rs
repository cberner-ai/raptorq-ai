#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackedBinaryRows {
    height: usize,
    width: usize,
    words_per_row: usize,
    words: Vec<u64>,
}

impl PackedBinaryRows {
    pub(crate) fn from_sparse(rows: Vec<Vec<usize>>, width: usize) -> PackedBinaryRows {
        let height = rows.len();
        let words_per_row = width.div_ceil(u64::BITS as usize);
        let mut packed = PackedBinaryRows {
            height,
            width,
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

    pub(crate) fn first_one_at_or_after(&self, row: usize, start_col: usize) -> Option<usize> {
        assert!(row < self.height);
        if start_col >= self.width {
            return None;
        }

        let row_start = self.row_start(row);
        let mut offset = start_col / u64::BITS as usize;
        let bit_offset = start_col % u64::BITS as usize;
        let mut word = self.words[row_start + offset] & (u64::MAX << bit_offset);

        loop {
            if word != 0 {
                let col = offset * u64::BITS as usize + word.trailing_zeros() as usize;
                return (col < self.width).then_some(col);
            }

            offset += 1;
            if offset >= self.words_per_row {
                return None;
            }
            word = self.words[row_start + offset];
        }
    }

    pub(crate) fn visit_ones_at_or_after<F>(&self, row: usize, start_col: usize, mut visit: F)
    where
        F: FnMut(usize),
    {
        assert!(row < self.height);
        if start_col >= self.width {
            return;
        }

        let row_start = self.row_start(row);
        let mut offset = start_col / u64::BITS as usize;
        let bit_offset = start_col % u64::BITS as usize;
        let mut word = self.words[row_start + offset] & (u64::MAX << bit_offset);

        while offset < self.words_per_row {
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let col = offset * u64::BITS as usize + bit;
                if col >= self.width {
                    return;
                }
                visit(col);
                word &= word - 1;
            }

            offset += 1;
            if offset < self.words_per_row {
                word = self.words[row_start + offset];
            }
        }
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

    #[test]
    fn first_one_scans_across_words() {
        let rows = vec![vec![1, 63, 64, 95]];
        let packed = PackedBinaryRows::from_sparse(rows, 96);

        assert_eq!(packed.first_one_at_or_after(0, 0), Some(1));
        assert_eq!(packed.first_one_at_or_after(0, 2), Some(63));
        assert_eq!(packed.first_one_at_or_after(0, 64), Some(64));
        assert_eq!(packed.first_one_at_or_after(0, 65), Some(95));
        assert_eq!(packed.first_one_at_or_after(0, 96), None);
    }

    #[test]
    fn visit_ones_stays_inside_width() {
        let rows = vec![vec![62, 63, 64, 95]];
        let packed = PackedBinaryRows::from_sparse(rows, 96);
        let mut visited = Vec::new();

        packed.visit_ones_at_or_after(0, 63, |col| visited.push(col));

        assert_eq!(visited, vec![63, 64, 95]);
    }
}

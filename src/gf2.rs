#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::matrix::BinaryMatrix;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedBinaryRows {
    height: usize,
    width: usize,
    words_per_row: usize,
    words: Vec<u64>,
}

impl PackedBinaryRows {
    pub(crate) fn new(height: usize, width: usize) -> PackedBinaryRows {
        let words_per_row = width.div_ceil(u64::BITS as usize);
        PackedBinaryRows {
            height,
            width,
            words_per_row,
            words: vec![0; height * words_per_row],
        }
    }

    #[cfg(test)]
    pub(crate) fn from_sparse(rows: Vec<Vec<usize>>, width: usize) -> PackedBinaryRows {
        let height = rows.len();
        let mut packed = PackedBinaryRows::new(height, width);

        for (row, entries) in rows.into_iter().enumerate() {
            for col in entries {
                packed.set(row, col);
            }
        }

        packed
    }

    pub(crate) fn from_matrix<M: BinaryMatrix>(matrix: &M) -> PackedBinaryRows {
        let mut packed = PackedBinaryRows::new(matrix.height(), matrix.width());

        for row in 0..matrix.height() {
            matrix.visit_row_entries(row, |col| packed.set(row, col));
        }

        packed
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn height(&self) -> usize {
        self.height
    }

    pub(crate) fn contains(&self, row: usize, col: usize) -> bool {
        let word = self.word_index(row, col);
        (self.words[word] & bit_mask(col)) != 0
    }

    pub(crate) fn xor_suffix(&mut self, dest: usize, src: usize, start_col: usize) {
        debug_assert!(self.contains(dest, start_col));
        debug_assert_ne!(dest, src);

        let first_word = start_col / u64::BITS as usize;
        let first_mask = u64::MAX << (start_col % u64::BITS as usize);
        let words_per_row = self.words_per_row;
        let dest_start = self.row_start(dest);
        let src_start = self.row_start(src);
        // Elimination buckets only merge distinct rows; raw row pointers avoid per-word bounds
        // checks in this packed GF(2) hot path.
        let dest_ptr = unsafe { self.words.as_mut_ptr().add(dest_start) };
        let src_ptr = unsafe { self.words.as_ptr().add(src_start) };

        unsafe {
            *dest_ptr.add(first_word) ^= *src_ptr.add(first_word) & first_mask;
        }
        let mut offset = first_word + 1;
        while offset + 4 <= words_per_row {
            unsafe {
                *dest_ptr.add(offset) ^= *src_ptr.add(offset);
                *dest_ptr.add(offset + 1) ^= *src_ptr.add(offset + 1);
                *dest_ptr.add(offset + 2) ^= *src_ptr.add(offset + 2);
                *dest_ptr.add(offset + 3) ^= *src_ptr.add(offset + 3);
            }
            offset += 4;
        }
        while offset < words_per_row {
            unsafe {
                *dest_ptr.add(offset) ^= *src_ptr.add(offset);
            }
            offset += 1;
        }
    }

    #[cfg(test)]
    pub(crate) fn xor_suffix_count_ones(
        &mut self,
        dest: usize,
        src: usize,
        start_col: usize,
    ) -> u32 {
        self.xor_suffix_count_ones_and_first_one(dest, src, start_col)
            .0
    }

    pub(crate) fn xor_suffix_count_ones_and_first_one(
        &mut self,
        dest: usize,
        src: usize,
        start_col: usize,
    ) -> (u32, Option<usize>) {
        debug_assert!(self.contains(dest, start_col));
        debug_assert_ne!(dest, src);

        let first_word = start_col / u64::BITS as usize;
        let first_mask = u64::MAX << (start_col % u64::BITS as usize);
        let width = self.width;
        let words_per_row = self.words_per_row;
        let dest_start = self.row_start(dest);
        let src_start = self.row_start(src);
        // Elimination buckets only merge distinct rows; raw row pointers avoid per-word bounds
        // checks in this packed GF(2) hot path.
        let dest_ptr = unsafe { self.words.as_mut_ptr().add(dest_start) };
        let src_ptr = unsafe { self.words.as_ptr().add(src_start) };

        let first_suffix_word = unsafe {
            let first_ptr = dest_ptr.add(first_word);
            *first_ptr ^= *src_ptr.add(first_word) & first_mask;
            *first_ptr & first_mask
        };
        let mut weight = first_suffix_word.count_ones();
        let mut first_one = if first_suffix_word == 0 {
            None
        } else {
            let col = first_word * u64::BITS as usize + first_suffix_word.trailing_zeros() as usize;
            (col < width).then_some(col)
        };

        let mut offset = first_word + 1;
        while first_one.is_none() && offset < words_per_row {
            let word = unsafe {
                let dest_word = dest_ptr.add(offset);
                let word = *dest_word ^ *src_ptr.add(offset);
                *dest_word = word;
                word
            };
            weight += word.count_ones();
            if word != 0 {
                let col = offset * u64::BITS as usize + word.trailing_zeros() as usize;
                if col < width {
                    first_one = Some(col);
                }
            }
            offset += 1;
        }

        while offset + 4 <= words_per_row {
            let word0;
            let word1;
            let word2;
            let word3;
            unsafe {
                word0 = *dest_ptr.add(offset) ^ *src_ptr.add(offset);
                word1 = *dest_ptr.add(offset + 1) ^ *src_ptr.add(offset + 1);
                word2 = *dest_ptr.add(offset + 2) ^ *src_ptr.add(offset + 2);
                word3 = *dest_ptr.add(offset + 3) ^ *src_ptr.add(offset + 3);
                *dest_ptr.add(offset) = word0;
                *dest_ptr.add(offset + 1) = word1;
                *dest_ptr.add(offset + 2) = word2;
                *dest_ptr.add(offset + 3) = word3;
            }
            weight +=
                word0.count_ones() + word1.count_ones() + word2.count_ones() + word3.count_ones();
            offset += 4;
        }
        while offset < words_per_row {
            let word = unsafe {
                let dest_word = dest_ptr.add(offset);
                let word = *dest_word ^ *src_ptr.add(offset);
                *dest_word = word;
                word
            };
            weight += word.count_ones();
            offset += 1;
        }
        (weight, first_one)
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

    pub(crate) fn weight_at_or_after(&self, row: usize, start_col: usize) -> u32 {
        assert!(row < self.height);
        if start_col >= self.width {
            return 0;
        }

        let row_start = self.row_start(row);
        let first_word = start_col / u64::BITS as usize;
        let bit_offset = start_col % u64::BITS as usize;
        let mut weight =
            (self.words[row_start + first_word] & (u64::MAX << bit_offset)).count_ones();
        for offset in (first_word + 1)..self.words_per_row {
            weight += self.words[row_start + offset].count_ones();
        }
        weight
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

    pub(crate) fn set(&mut self, row: usize, col: usize) {
        let word = self.word_index(row, col);
        self.words[word] |= bit_mask(col);
    }

    pub(crate) fn set_entries(&mut self, row: usize, cols: &[usize]) {
        assert!(row < self.height);
        let row_start = self.row_start(row);
        for &col in cols {
            debug_assert!(col < self.width);
            self.words[row_start + col / u64::BITS as usize] |= bit_mask(col);
        }
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

        packed.xor_suffix(1, 0, 3);

        assert!(packed.contains(1, 0));
        assert!(!packed.contains(1, 3));
        assert!(packed.contains(1, 64));
        assert!(!packed.contains(1, 70));
    }

    #[test]
    fn xor_suffix_count_ones_returns_updated_suffix_weight() {
        let rows = vec![vec![1, 3, 70], vec![0, 3, 64, 70, 95]];
        let mut packed = PackedBinaryRows::from_sparse(rows, 96);

        let weight = packed.xor_suffix_count_ones(1, 0, 3);

        assert_eq!(weight, 2);
        assert!(packed.contains(1, 0));
        assert!(!packed.contains(1, 3));
        assert!(packed.contains(1, 64));
        assert!(!packed.contains(1, 70));
        assert!(packed.contains(1, 95));
    }

    #[test]
    fn xor_suffix_count_ones_and_first_one_returns_combined_suffix_state() {
        let rows = vec![vec![1, 3, 70], vec![0, 3, 64, 70, 95]];
        let mut packed = PackedBinaryRows::from_sparse(rows, 96);

        let (weight, first_one) = packed.xor_suffix_count_ones_and_first_one(1, 0, 3);

        assert_eq!(weight, 2);
        assert_eq!(first_one, Some(64));
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

    #[test]
    fn weight_counts_suffix_bits() {
        let rows = vec![vec![1, 63, 64, 95]];
        let packed = PackedBinaryRows::from_sparse(rows, 96);

        assert_eq!(packed.weight_at_or_after(0, 0), 4);
        assert_eq!(packed.weight_at_or_after(0, 2), 3);
        assert_eq!(packed.weight_at_or_after(0, 64), 2);
        assert_eq!(packed.weight_at_or_after(0, 96), 0);
    }

    #[test]
    fn from_matrix_packs_visited_entries() {
        use crate::matrix::{BinaryMatrix, DenseBinaryMatrix};

        let mut matrix = DenseBinaryMatrix::new(2, 96);
        matrix.set(0, 1, true);
        matrix.set(0, 64, true);
        matrix.set(1, 3, true);
        matrix.set(1, 95, true);

        let packed = PackedBinaryRows::from_matrix(&matrix);

        assert_eq!(packed.width(), 96);
        assert_eq!(packed.height(), 2);
        assert!(packed.contains(0, 1));
        assert!(packed.contains(0, 64));
        assert!(packed.contains(1, 3));
        assert!(packed.contains(1, 95));
        assert!(!packed.contains(0, 95));
    }
}

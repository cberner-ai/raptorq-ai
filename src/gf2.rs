#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(all(feature = "std", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi8, _mm256_and_si256, _mm256_extract_epi64, _mm256_loadu_si256,
    _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_srli_epi16, _mm256_storeu_si256, _mm256_xor_si256,
};

use crate::matrix::BinaryMatrix;

const WIDE_BINARY_ROW_POPCOUNT_MIN_WORDS: usize = 2;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedBinaryRows {
    height: usize,
    width: usize,
    words_per_row: usize,
    words: Vec<u64>,
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    use_avx2_popcount: bool,
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    use_popcnt: bool,
}

impl PackedBinaryRows {
    pub(crate) fn new(height: usize, width: usize) -> PackedBinaryRows {
        let words_per_row = width.div_ceil(u64::BITS as usize);
        let use_wide_popcount = words_per_row >= WIDE_BINARY_ROW_POPCOUNT_MIN_WORDS;
        #[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
        let _ = use_wide_popcount;
        PackedBinaryRows {
            height,
            width,
            words_per_row,
            words: vec![0; height * words_per_row],
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            use_avx2_popcount: use_wide_popcount && std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            use_popcnt: use_wide_popcount && std::arch::is_x86_feature_detected!("popcnt"),
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

    pub(crate) fn from_sparse_entries_with_row_weights_and_first_ones(
        width: usize,
        rows: &[Vec<usize>],
    ) -> (PackedBinaryRows, Vec<u32>, Vec<Option<usize>>) {
        let height = rows.len();
        let mut packed = PackedBinaryRows::new(height, width);
        let mut row_weights = vec![0; height];
        let mut first_ones = vec![None; height];

        fill_sparse_entry_metadata_chunk(
            width,
            packed.words_per_row,
            rows,
            packed.words.as_mut_slice(),
            row_weights.as_mut_slice(),
            first_ones.as_mut_slice(),
        );
        (packed, row_weights, first_ones)
    }

    pub(crate) fn from_sparse_entries_with_first_ones(
        width: usize,
        rows: &[Vec<usize>],
    ) -> (PackedBinaryRows, Vec<Option<usize>>) {
        let height = rows.len();
        let mut packed = PackedBinaryRows::new(height, width);
        let mut first_ones = vec![None; height];

        fill_sparse_entry_first_ones(
            width,
            packed.words_per_row,
            rows,
            packed.words.as_mut_slice(),
            first_ones.as_mut_slice(),
        );
        (packed, first_ones)
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
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        if self.use_avx2_popcount {
            unsafe {
                return self.xor_suffix_count_ones_and_first_one_avx2(dest, src, start_col);
            }
        }

        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if self.use_popcnt {
            unsafe {
                return self.xor_suffix_count_ones_and_first_one_popcnt(dest, src, start_col);
            }
        }

        self.xor_suffix_count_ones_and_first_one_fallback(dest, src, start_col)
    }

    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn xor_suffix_count_ones_and_first_one_avx2(
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

        let lookup = unsafe { avx2_nibble_popcount_table() };
        let low_mask = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        while offset + 16 <= words_per_row {
            unsafe {
                weight += xor_store_popcount_4(dest_ptr, src_ptr, offset, lookup, low_mask, zero)
                    + xor_store_popcount_4(dest_ptr, src_ptr, offset + 4, lookup, low_mask, zero)
                    + xor_store_popcount_4(dest_ptr, src_ptr, offset + 8, lookup, low_mask, zero)
                    + xor_store_popcount_4(dest_ptr, src_ptr, offset + 12, lookup, low_mask, zero);
            }
            offset += 16;
        }
        while offset + 4 <= words_per_row {
            unsafe {
                weight += xor_store_popcount_4(dest_ptr, src_ptr, offset, lookup, low_mask, zero);
            }
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

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[target_feature(enable = "popcnt")]
    unsafe fn xor_suffix_count_ones_and_first_one_popcnt(
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

        while offset + 8 <= words_per_row {
            let word0;
            let word1;
            let word2;
            let word3;
            let word4;
            let word5;
            let word6;
            let word7;
            unsafe {
                word0 = *dest_ptr.add(offset) ^ *src_ptr.add(offset);
                word1 = *dest_ptr.add(offset + 1) ^ *src_ptr.add(offset + 1);
                word2 = *dest_ptr.add(offset + 2) ^ *src_ptr.add(offset + 2);
                word3 = *dest_ptr.add(offset + 3) ^ *src_ptr.add(offset + 3);
                word4 = *dest_ptr.add(offset + 4) ^ *src_ptr.add(offset + 4);
                word5 = *dest_ptr.add(offset + 5) ^ *src_ptr.add(offset + 5);
                word6 = *dest_ptr.add(offset + 6) ^ *src_ptr.add(offset + 6);
                word7 = *dest_ptr.add(offset + 7) ^ *src_ptr.add(offset + 7);
                *dest_ptr.add(offset) = word0;
                *dest_ptr.add(offset + 1) = word1;
                *dest_ptr.add(offset + 2) = word2;
                *dest_ptr.add(offset + 3) = word3;
                *dest_ptr.add(offset + 4) = word4;
                *dest_ptr.add(offset + 5) = word5;
                *dest_ptr.add(offset + 6) = word6;
                *dest_ptr.add(offset + 7) = word7;
            }
            weight += word0.count_ones()
                + word1.count_ones()
                + word2.count_ones()
                + word3.count_ones()
                + word4.count_ones()
                + word5.count_ones()
                + word6.count_ones()
                + word7.count_ones();
            offset += 8;
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

    fn xor_suffix_count_ones_and_first_one_fallback(
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

        while offset + 8 <= words_per_row {
            let word0;
            let word1;
            let word2;
            let word3;
            let word4;
            let word5;
            let word6;
            let word7;
            unsafe {
                word0 = *dest_ptr.add(offset) ^ *src_ptr.add(offset);
                word1 = *dest_ptr.add(offset + 1) ^ *src_ptr.add(offset + 1);
                word2 = *dest_ptr.add(offset + 2) ^ *src_ptr.add(offset + 2);
                word3 = *dest_ptr.add(offset + 3) ^ *src_ptr.add(offset + 3);
                word4 = *dest_ptr.add(offset + 4) ^ *src_ptr.add(offset + 4);
                word5 = *dest_ptr.add(offset + 5) ^ *src_ptr.add(offset + 5);
                word6 = *dest_ptr.add(offset + 6) ^ *src_ptr.add(offset + 6);
                word7 = *dest_ptr.add(offset + 7) ^ *src_ptr.add(offset + 7);
                *dest_ptr.add(offset) = word0;
                *dest_ptr.add(offset + 1) = word1;
                *dest_ptr.add(offset + 2) = word2;
                *dest_ptr.add(offset + 3) = word3;
                *dest_ptr.add(offset + 4) = word4;
                *dest_ptr.add(offset + 5) = word5;
                *dest_ptr.add(offset + 6) = word6;
                *dest_ptr.add(offset + 7) = word7;
            }
            weight += word0.count_ones()
                + word1.count_ones()
                + word2.count_ones()
                + word3.count_ones()
                + word4.count_ones()
                + word5.count_ones()
                + word6.count_ones()
                + word7.count_ones();
            offset += 8;
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

    #[cfg(test)]
    pub(crate) fn xor_columns_update_weight_and_first_one(
        &mut self,
        dest: usize,
        cols: &[usize],
        mut weight: u32,
    ) -> (u32, Option<usize>) {
        let Some(&start_col) = cols.first() else {
            return (weight, None);
        };
        debug_assert!(self.contains(dest, start_col));

        for &col in cols {
            debug_assert!(col < self.width);
            let word = self.word_index(dest, col);
            let mask = bit_mask(col);
            if (self.words[word] & mask) == 0 {
                self.words[word] |= mask;
                weight += 1;
            } else {
                self.words[word] &= !mask;
                weight -= 1;
            }
        }

        let first_one = if weight == 0 {
            None
        } else {
            self.first_one_at_or_after(dest, start_col)
        };
        (weight, first_one)
    }

    pub(crate) fn xor_u16_columns_update_weight_and_first_one(
        &mut self,
        dest: usize,
        cols: &[u16],
        mut weight: u32,
    ) -> (u32, Option<usize>) {
        let Some(&start_col) = cols.first() else {
            return (weight, None);
        };
        let start_col = start_col as usize;
        debug_assert!(self.contains(dest, start_col));

        for &col in cols {
            let col = col as usize;
            debug_assert!(col < self.width);
            let word = self.word_index(dest, col);
            let mask = bit_mask(col);
            if (self.words[word] & mask) == 0 {
                self.words[word] |= mask;
                weight += 1;
            } else {
                self.words[word] &= !mask;
                weight -= 1;
            }
        }

        let first_one = if weight == 0 {
            None
        } else {
            self.first_one_at_or_after(dest, start_col)
        };
        (weight, first_one)
    }

    pub(crate) fn xor_u16_columns_apply_known_state(
        &mut self,
        dest: usize,
        cols: &[u16],
        weight: u32,
        first_one: Option<usize>,
    ) -> (u32, Option<usize>) {
        let Some(&start_col) = cols.first() else {
            return (weight, first_one);
        };
        let start_col = start_col as usize;
        debug_assert!(self.contains(dest, start_col));

        for &col in cols {
            let col = col as usize;
            debug_assert!(col < self.width);
            let word = self.word_index(dest, col);
            self.words[word] ^= bit_mask(col);
        }

        debug_assert_eq!(weight, self.weight_at_or_after(dest, start_col));
        debug_assert_eq!(first_one, self.first_one_at_or_after(dest, start_col));
        (weight, first_one)
    }

    pub(crate) fn weight_at_or_after(&self, row: usize, start_col: usize) -> u32 {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if self.use_popcnt {
            unsafe {
                return self.weight_at_or_after_popcnt(row, start_col);
            }
        }

        self.weight_at_or_after_fallback(row, start_col)
    }

    fn weight_at_or_after_fallback(&self, row: usize, start_col: usize) -> u32 {
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

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[target_feature(enable = "popcnt")]
    unsafe fn weight_at_or_after_popcnt(&self, row: usize, start_col: usize) -> u32 {
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
        let row_words = &mut self.words[row_start..row_start + self.words_per_row];
        for &col in cols {
            debug_assert!(col < self.width);
            row_words[col / u64::BITS as usize] |= bit_mask(col);
        }
    }

    pub(crate) fn set_row_from_le_bytes(
        &mut self,
        row: usize,
        bytes: &[u8],
    ) -> (u32, Option<usize>) {
        assert!(row < self.height);
        assert_eq!(bytes.len(), self.width.div_ceil(8));

        let row_start = self.row_start(row);
        let row_words = &mut self.words[row_start..row_start + self.words_per_row];
        let mut weight = 0u32;
        let mut first_one = None;
        let mut chunks = bytes.chunks_exact(8);
        for (word_index, chunk) in chunks.by_ref().enumerate() {
            let word = u64::from_le_bytes(chunk.try_into().expect("chunk length is 8"));
            row_words[word_index] = word;
            weight += word.count_ones();
            if first_one.is_none() && word != 0 {
                first_one = Some(word_index * u64::BITS as usize + word.trailing_zeros() as usize);
            }
        }

        let tail = chunks.remainder();
        if !tail.is_empty() {
            let word_index = bytes.len() / 8;
            let mut word = 0u64;
            for (offset, &byte) in tail.iter().enumerate() {
                word |= (byte as u64) << (offset * 8);
            }
            let valid_bits = self.width - word_index * u64::BITS as usize;
            if valid_bits < u64::BITS as usize {
                word &= u64::MAX >> (u64::BITS as usize - valid_bits);
            }
            row_words[word_index] = word;
            weight += word.count_ones();
            if first_one.is_none() && word != 0 {
                first_one = Some(word_index * u64::BITS as usize + word.trailing_zeros() as usize);
            }
        }

        (weight, first_one)
    }

    pub(crate) fn set_row_from_bit_slice(
        &mut self,
        row: usize,
        bytes: &[u8],
        start_bit: usize,
    ) -> (u32, Option<usize>) {
        assert!(row < self.height);
        assert!(start_bit + self.width <= bytes.len() * u8::BITS as usize);

        let row_start = self.row_start(row);
        let row_words = &mut self.words[row_start..row_start + self.words_per_row];
        let mut weight = 0u32;
        let mut first_one = None;

        for (word_index, slot) in row_words.iter_mut().enumerate() {
            let col = word_index * u64::BITS as usize;
            let valid_bits = (self.width - col).min(u64::BITS as usize);
            let word = read_le_bits(bytes, start_bit + col, valid_bits);
            *slot = word;
            weight += word.count_ones();
            if first_one.is_none() && word != 0 {
                first_one = Some(col + word.trailing_zeros() as usize);
            }
        }

        (weight, first_one)
    }

    fn row_start(&self, row: usize) -> usize {
        row * self.words_per_row
    }

    fn word_index(&self, row: usize, col: usize) -> usize {
        assert!(row < self.height);
        self.row_start(row) + col / u64::BITS as usize
    }
}

#[inline]
fn read_le_bits(bytes: &[u8], start_bit: usize, valid_bits: usize) -> u64 {
    debug_assert!((1..=u64::BITS as usize).contains(&valid_bits));
    debug_assert!(start_bit + valid_bits <= bytes.len() * u8::BITS as usize);

    let byte_start = start_bit / u8::BITS as usize;
    let bit_offset = start_bit % u8::BITS as usize;
    let byte_len = (bit_offset + valid_bits).div_ceil(u8::BITS as usize);
    let byte_window = &bytes[byte_start..byte_start + byte_len];

    let bits = if byte_window.len() >= 8 {
        u64::from_le_bytes(byte_window[..8].try_into().expect("slice length is 8"))
    } else {
        let mut bits = 0u64;
        for (offset, &byte) in byte_window.iter().enumerate() {
            bits |= (byte as u64) << (offset * u8::BITS as usize);
        }
        bits
    };

    let mut shifted = bits >> bit_offset;
    if bit_offset != 0 && byte_window.len() > 8 {
        shifted |= (byte_window[8] as u64) << (u64::BITS as usize - bit_offset);
    }
    shifted & valid_bit_mask(valid_bits)
}

#[inline]
fn valid_bit_mask(valid_bits: usize) -> u64 {
    if valid_bits == u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    }
}

fn fill_sparse_entry_metadata_chunk(
    width: usize,
    words_per_row: usize,
    rows: &[Vec<usize>],
    words: &mut [u64],
    row_weights: &mut [u32],
    first_ones: &mut [Option<usize>],
) {
    debug_assert_eq!(words.len(), rows.len() * words_per_row);
    debug_assert_eq!(row_weights.len(), rows.len());
    debug_assert_eq!(first_ones.len(), rows.len());

    for (row, entries) in rows.iter().enumerate() {
        row_weights[row] = entries.len() as u32;
        let row_words = &mut words[row * words_per_row..(row + 1) * words_per_row];
        first_ones[row] = fill_sparse_entry_row(width, row_words, entries);
    }
}

fn fill_sparse_entry_first_ones(
    width: usize,
    words_per_row: usize,
    rows: &[Vec<usize>],
    words: &mut [u64],
    first_ones: &mut [Option<usize>],
) {
    debug_assert_eq!(words.len(), rows.len() * words_per_row);
    debug_assert_eq!(first_ones.len(), rows.len());

    for (row, entries) in rows.iter().enumerate() {
        let row_words = &mut words[row * words_per_row..(row + 1) * words_per_row];
        first_ones[row] = fill_sparse_entry_row(width, row_words, entries);
    }
}

#[inline]
fn fill_sparse_entry_row(width: usize, row_words: &mut [u64], entries: &[usize]) -> Option<usize> {
    let mut first_one = width;
    for &col in entries {
        debug_assert!(col < width);
        first_one = first_one.min(col);
        row_words[col / u64::BITS as usize] |= bit_mask(col);
    }
    (first_one != width).then_some(first_one)
}

#[inline]
fn bit_mask(col: usize) -> u64 {
    1u64 << (col % u64::BITS as usize)
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn xor_store_popcount_4(
    dest_ptr: *mut u64,
    src_ptr: *const u64,
    offset: usize,
    lookup: __m256i,
    low_mask: __m256i,
    zero: __m256i,
) -> u32 {
    let updated = unsafe {
        let dest = _mm256_loadu_si256(dest_ptr.add(offset).cast::<__m256i>().cast_const());
        let src = _mm256_loadu_si256(src_ptr.add(offset).cast::<__m256i>());
        let updated = _mm256_xor_si256(dest, src);
        _mm256_storeu_si256(dest_ptr.add(offset).cast::<__m256i>(), updated);
        updated
    };
    unsafe { avx2_popcount_256(updated, lookup, low_mask, zero) }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_popcount_256(
    value: __m256i,
    lookup: __m256i,
    low_mask: __m256i,
    zero: __m256i,
) -> u32 {
    let low = _mm256_and_si256(value, low_mask);
    let high = _mm256_and_si256(_mm256_srli_epi16(value, 4), low_mask);
    let counts = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, low),
        _mm256_shuffle_epi8(lookup, high),
    );
    let sums = _mm256_sad_epu8(counts, zero);
    let sum0 = _mm256_extract_epi64(sums, 0) as u64;
    let sum1 = _mm256_extract_epi64(sums, 1) as u64;
    let sum2 = _mm256_extract_epi64(sums, 2) as u64;
    let sum3 = _mm256_extract_epi64(sums, 3) as u64;
    (sum0 + sum1 + sum2 + sum3) as u32
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_nibble_popcount_table() -> __m256i {
    _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    )
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
    fn wide_xor_suffix_count_matches_fallback() {
        let rows = vec![
            vec![5, 64, 511, 4096, 32760],
            vec![1, 5, 64, 128, 4096, 16384, 32760, 32767],
        ];
        let mut expected = PackedBinaryRows::from_sparse(rows.clone(), 32768);
        let mut packed = PackedBinaryRows::from_sparse(rows, 32768);

        let expected_result = expected.xor_suffix_count_ones_and_first_one_fallback(1, 0, 5);
        let actual_result = packed.xor_suffix_count_ones_and_first_one(1, 0, 5);

        assert_eq!(actual_result, expected_result);
        assert_eq!(packed, expected);
    }

    #[test]
    fn popcount_gate_width_matches_fallback() {
        let width = WIDE_BINARY_ROW_POPCOUNT_MIN_WORDS * u64::BITS as usize;
        let mid = (width / 2).max(65);
        let tail = width - 1;
        let rows = vec![
            vec![7, 64, mid, tail - 1],
            vec![3, 7, 64, mid, tail - 1, tail],
        ];
        let mut expected = PackedBinaryRows::from_sparse(rows.clone(), width);
        let mut packed = PackedBinaryRows::from_sparse(rows, width);

        let expected_result = expected.xor_suffix_count_ones_and_first_one_fallback(1, 0, 7);
        let actual_result = packed.xor_suffix_count_ones_and_first_one(1, 0, 7);

        assert_eq!(actual_result, expected_result);
        assert_eq!(packed, expected);
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
    fn column_xor_updates_weight_and_first_one() {
        let rows = vec![vec![0, 3, 65, 70, 95]];
        let mut packed = PackedBinaryRows::from_sparse(rows, 96);
        let cols = [3, 64, 70];

        let (weight, first_one) = packed.xor_columns_update_weight_and_first_one(0, &cols, 4);

        assert_eq!(weight, 3);
        assert_eq!(first_one, Some(64));
        assert!(packed.contains(0, 0));
        assert!(!packed.contains(0, 3));
        assert!(packed.contains(0, 64));
        assert!(packed.contains(0, 65));
        assert!(!packed.contains(0, 70));
        assert!(packed.contains(0, 95));
    }

    #[test]
    fn u16_column_xor_updates_weight_and_first_one() {
        let rows = vec![vec![0, 3, 65, 70, 95]];
        let mut packed = PackedBinaryRows::from_sparse(rows, 96);
        let cols = [3u16, 64, 70];

        let (weight, first_one) = packed.xor_u16_columns_update_weight_and_first_one(0, &cols, 4);

        assert_eq!(weight, 3);
        assert_eq!(first_one, Some(64));
        assert!(packed.contains(0, 0));
        assert!(!packed.contains(0, 3));
        assert!(packed.contains(0, 64));
        assert!(packed.contains(0, 65));
        assert!(!packed.contains(0, 70));
        assert!(packed.contains(0, 95));
    }

    #[test]
    fn u16_column_xor_can_reuse_known_row_state() {
        let rows = vec![vec![3, 8, 13]];
        let mut packed = PackedBinaryRows::from_sparse(rows, 32);
        let cols = [3u16, 5, 13, 21];

        let (weight, first_one) = packed.xor_u16_columns_apply_known_state(0, &cols, 3, Some(5));

        assert_eq!(weight, 3);
        assert_eq!(first_one, Some(5));
        assert!(!packed.contains(0, 3));
        assert!(packed.contains(0, 5));
        assert!(packed.contains(0, 8));
        assert!(!packed.contains(0, 13));
        assert!(packed.contains(0, 21));
    }

    #[cfg(feature = "std")]
    #[test]
    fn large_sparse_entries_pack_with_metadata() {
        const LARGE_ROWS: usize = 16_384;
        const MIDDLE_ROW: usize = 4_097;

        let mut rows = vec![Vec::new(); LARGE_ROWS];
        rows[0] = vec![70, 3];
        rows[MIDDLE_ROW] = vec![128, 1, 256];
        rows[LARGE_ROWS - 1] = vec![200];

        let (packed, row_weights, first_ones) =
            PackedBinaryRows::from_sparse_entries_with_row_weights_and_first_ones(257, &rows);

        assert_eq!(packed.height(), LARGE_ROWS);
        assert_eq!(packed.width(), 257);
        assert_eq!(row_weights[0], 2);
        assert_eq!(first_ones[0], Some(3));
        assert!(packed.contains(0, 3));
        assert!(packed.contains(0, 70));
        assert_eq!(row_weights[MIDDLE_ROW], 3);
        assert_eq!(first_ones[MIDDLE_ROW], Some(1));
        assert!(packed.contains(MIDDLE_ROW, 1));
        assert!(packed.contains(MIDDLE_ROW, 128));
        assert!(packed.contains(MIDDLE_ROW, 256));
        assert_eq!(row_weights[LARGE_ROWS - 1], 1);
        assert_eq!(first_ones[LARGE_ROWS - 1], Some(200));
        assert!(packed.contains(LARGE_ROWS - 1, 200));
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

    #[test]
    fn read_le_bits_matches_reference_across_byte_offsets() {
        let bytes = (0u8..96)
            .map(|byte| byte.wrapping_mul(37).wrapping_add(11))
            .collect::<Vec<_>>();

        for start_bit in 0..256 {
            for valid_bits in 1..=u64::BITS as usize {
                let actual = read_le_bits(&bytes, start_bit, valid_bits);
                let expected = read_le_bits_reference(&bytes, start_bit, valid_bits);
                assert_eq!(
                    actual, expected,
                    "start_bit={start_bit}, valid_bits={valid_bits}"
                );
            }
        }
    }

    fn read_le_bits_reference(bytes: &[u8], start_bit: usize, valid_bits: usize) -> u64 {
        let mut bits = 0u64;
        for bit in 0..valid_bits {
            let source_bit = start_bit + bit;
            let byte = bytes[source_bit / u8::BITS as usize];
            let mask = 1u8 << (source_bit % u8::BITS as usize);
            if byte & mask != 0 {
                bits |= 1u64 << bit;
            }
        }
        bits
    }
}

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

const WIDE_BINARY_ROW_POPCOUNT_MIN_WORDS: usize = 512;
#[cfg(feature = "std")]
const PARALLEL_SPARSE_PACK_MIN_ROWS: usize = 16_384;
#[cfg(feature = "std")]
const PARALLEL_SPARSE_PACK_ROWS_PER_THREAD: usize = 4_096;
#[cfg(feature = "std")]
const PARALLEL_SPARSE_PACK_MAX_THREADS: usize = 4;

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

    pub(crate) fn from_sparse_entries_with_row_weights_and_first_ones(
        width: usize,
        rows: &[Vec<usize>],
    ) -> (PackedBinaryRows, Vec<u32>, Vec<Option<usize>>) {
        let height = rows.len();
        let mut packed = PackedBinaryRows::new(height, width);
        let mut row_weights = vec![0; height];
        let mut first_ones = vec![None; height];

        #[cfg(feature = "std")]
        if height >= PARALLEL_SPARSE_PACK_MIN_ROWS {
            let available_threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);
            let thread_count = available_threads
                .min(PARALLEL_SPARSE_PACK_MAX_THREADS)
                .min(height.div_ceil(PARALLEL_SPARSE_PACK_ROWS_PER_THREAD));
            if thread_count > 1 {
                let rows_per_chunk = height.div_ceil(thread_count);
                std::thread::scope(|scope| {
                    let mut remaining_words = packed.words.as_mut_slice();
                    let mut remaining_weights = row_weights.as_mut_slice();
                    let mut remaining_first_ones = first_ones.as_mut_slice();
                    for row_chunk in rows.chunks(rows_per_chunk) {
                        let chunk_rows = row_chunk.len();
                        let (word_chunk, next_words) =
                            remaining_words.split_at_mut(chunk_rows * packed.words_per_row);
                        remaining_words = next_words;
                        let (weight_chunk, next_weights) =
                            remaining_weights.split_at_mut(chunk_rows);
                        remaining_weights = next_weights;
                        let (first_one_chunk, next_first_ones) =
                            remaining_first_ones.split_at_mut(chunk_rows);
                        remaining_first_ones = next_first_ones;
                        let words_per_row = packed.words_per_row;
                        scope.spawn(move || {
                            fill_sparse_entry_metadata_chunk(
                                width,
                                words_per_row,
                                row_chunk,
                                word_chunk,
                                weight_chunk,
                                first_one_chunk,
                            );
                        });
                    }
                });
                return (packed, row_weights, first_ones);
            }
        }

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
        if self.words_per_row >= WIDE_BINARY_ROW_POPCOUNT_MIN_WORDS {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            if std::arch::is_x86_feature_detected!("avx2") {
                unsafe {
                    return self.xor_suffix_count_ones_and_first_one_avx2(dest, src, start_col);
                }
            }

            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            if std::arch::is_x86_feature_detected!("popcnt") {
                unsafe {
                    return self.xor_suffix_count_ones_and_first_one_popcnt(dest, src, start_col);
                }
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

    pub(crate) fn weight_at_or_after(&self, row: usize, start_col: usize) -> u32 {
        if self.words_per_row >= WIDE_BINARY_ROW_POPCOUNT_MIN_WORDS {
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            if std::arch::is_x86_feature_detected!("popcnt") {
                unsafe {
                    return self.weight_at_or_after_popcnt(row, start_col);
                }
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
        let mut first_one = None;
        for &col in entries {
            debug_assert!(col < width);
            if first_one.is_none_or(|first| col < first) {
                first_one = Some(col);
            }
            row_words[col / u64::BITS as usize] |= bit_mask(col);
        }
        first_ones[row] = first_one;
    }
}

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

    #[cfg(feature = "std")]
    #[test]
    fn large_sparse_entries_pack_with_metadata() {
        let mut rows = vec![Vec::new(); PARALLEL_SPARSE_PACK_MIN_ROWS];
        rows[0] = vec![70, 3];
        rows[PARALLEL_SPARSE_PACK_ROWS_PER_THREAD + 1] = vec![128, 1, 256];
        rows[PARALLEL_SPARSE_PACK_MIN_ROWS - 1] = vec![200];

        let (packed, row_weights, first_ones) =
            PackedBinaryRows::from_sparse_entries_with_row_weights_and_first_ones(257, &rows);

        assert_eq!(packed.height(), PARALLEL_SPARSE_PACK_MIN_ROWS);
        assert_eq!(packed.width(), 257);
        assert_eq!(row_weights[0], 2);
        assert_eq!(first_ones[0], Some(3));
        assert!(packed.contains(0, 3));
        assert!(packed.contains(0, 70));
        assert_eq!(row_weights[PARALLEL_SPARSE_PACK_ROWS_PER_THREAD + 1], 3);
        assert_eq!(
            first_ones[PARALLEL_SPARSE_PACK_ROWS_PER_THREAD + 1],
            Some(1)
        );
        assert!(packed.contains(PARALLEL_SPARSE_PACK_ROWS_PER_THREAD + 1, 1));
        assert!(packed.contains(PARALLEL_SPARSE_PACK_ROWS_PER_THREAD + 1, 128));
        assert!(packed.contains(PARALLEL_SPARSE_PACK_ROWS_PER_THREAD + 1, 256));
        assert_eq!(row_weights[PARALLEL_SPARSE_PACK_MIN_ROWS - 1], 1);
        assert_eq!(first_ones[PARALLEL_SPARSE_PACK_MIN_ROWS - 1], Some(200));
        assert!(packed.contains(PARALLEL_SPARSE_PACK_MIN_ROWS - 1, 200));
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

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::gf2::PackedBinaryRows;
use crate::octet::Octet;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

pub trait BinaryMatrix: Clone {
    fn new(height: usize, width: usize) -> Self;
    fn height(&self) -> usize;
    fn width(&self) -> usize;
    fn systematic_source_block_symbols(&self) -> Option<u32> {
        None
    }
    fn mark_systematic_source_block_symbols(&mut self, _source_block_symbols: u32) {}
    fn contiguous_single_repair_systematic_rows(&self) -> Option<(u32, usize, usize, u32)> {
        None
    }
    fn mark_contiguous_single_repair_systematic_rows(
        &mut self,
        _source_block_symbols: u32,
        _missing_isi: usize,
        _repair_matrix_row: usize,
        _repair_isi: u32,
    ) {
    }
    fn systematic_row_isis(&self) -> Option<&[Option<u32>]> {
        None
    }
    fn mark_encoded_systematic_isis(
        &mut self,
        _row_offset: usize,
        _source_block_symbols: u32,
        _encoded_isis: &[u32],
    ) {
    }
    fn get(&self, row: usize, col: usize) -> Octet;
    fn set(&mut self, row: usize, col: usize, value: bool);
    fn reserve_row_entries(&mut self, _row: usize, _additional: usize) {}
    fn normalize_rows(&mut self) {}
    fn toggle_unique(&mut self, row: usize, col: usize) {
        self.toggle(row, col);
    }
    fn packed_rows(&self) -> PackedBinaryRows {
        PackedBinaryRows::from_matrix(self)
    }

    fn packed_row_prefix(&self, height: usize) -> PackedBinaryRows {
        assert!(height <= self.height());
        let mut rows = PackedBinaryRows::new(height, self.width());
        for row in 0..height {
            self.visit_row_entries(row, |col| rows.set(row, col));
        }
        rows
    }

    fn packed_row_prefix_with_row_weights_and_first_ones(
        &self,
        height: usize,
    ) -> (PackedBinaryRows, Vec<u32>, Vec<Option<usize>>) {
        let rows = self.packed_row_prefix(height);
        let row_weights = (0..rows.height())
            .map(|row| rows.weight_at_or_after(row, 0))
            .collect();
        let first_ones = (0..rows.height())
            .map(|row| rows.first_one_at_or_after(row, 0))
            .collect();
        (rows, row_weights, first_ones)
    }

    fn packed_rows_with_first_ones(&self) -> (PackedBinaryRows, Vec<Option<usize>>) {
        let rows = self.packed_rows();
        let first_ones = (0..rows.height())
            .map(|row| rows.first_one_at_or_after(row, 0))
            .collect();
        (rows, first_ones)
    }
    fn packed_rows_with_row_weights_and_first_ones(
        &self,
    ) -> (PackedBinaryRows, Vec<u32>, Vec<Option<usize>>) {
        let rows = self.packed_rows();
        let row_weights = (0..rows.height())
            .map(|row| rows.weight_at_or_after(row, 0))
            .collect();
        let first_ones = (0..rows.height())
            .map(|row| rows.first_one_at_or_after(row, 0))
            .collect();
        (rows, row_weights, first_ones)
    }
    fn toggle(&mut self, row: usize, col: usize) {
        let next = self.get(row, col) == Octet::zero();
        self.set(row, col, next);
    }

    fn visit_row_entries<F>(&self, row: usize, mut visit: F)
    where
        F: FnMut(usize),
    {
        for col in 0..self.width() {
            if self.get(row, col) != Octet::zero() {
                visit(col);
            }
        }
    }

    fn visit_row_entries_unordered<F>(&self, row: usize, visit: F)
    where
        F: FnMut(usize),
    {
        self.visit_row_entries(row, visit);
    }

    fn row_entries_unordered_slice(&self, _row: usize) -> Option<&[usize]> {
        None
    }

    fn row_entries(&self, row: usize) -> Vec<usize> {
        let mut entries = Vec::new();
        self.visit_row_entries(row, |col| entries.push(col));
        entries
    }

    fn into_row_entries(self) -> Vec<Vec<usize>>
    where
        Self: Sized,
    {
        let mut rows = Vec::with_capacity(self.height());
        for row in 0..self.height() {
            rows.push(self.row_entries(row));
        }
        rows
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct DenseBinaryMatrix {
    height: usize,
    width: usize,
    data: Vec<u8>,
}

impl DenseBinaryMatrix {
    fn byte_index(&self, row: usize, col: usize) -> (usize, u8) {
        assert!(row < self.height);
        assert!(col < self.width);
        let bit = row * self.width + col;
        (bit / 8, 1 << (bit % 8))
    }
}

impl BinaryMatrix for DenseBinaryMatrix {
    fn new(height: usize, width: usize) -> DenseBinaryMatrix {
        DenseBinaryMatrix {
            height,
            width,
            data: vec![0; (height * width).div_ceil(8)],
        }
    }

    fn height(&self) -> usize {
        self.height
    }

    fn width(&self) -> usize {
        self.width
    }

    fn get(&self, row: usize, col: usize) -> Octet {
        let (byte, mask) = self.byte_index(row, col);
        if self.data[byte] & mask == 0 {
            Octet::zero()
        } else {
            Octet::one()
        }
    }

    fn set(&mut self, row: usize, col: usize, value: bool) {
        let (byte, mask) = self.byte_index(row, col);
        if value {
            self.data[byte] |= mask;
        } else {
            self.data[byte] &= !mask;
        }
    }

    fn toggle(&mut self, row: usize, col: usize) {
        let (byte, mask) = self.byte_index(row, col);
        self.data[byte] ^= mask;
    }

    fn packed_rows(&self) -> PackedBinaryRows {
        let mut rows = PackedBinaryRows::new(self.height, self.width);
        if self.width.is_multiple_of(8) {
            let row_byte_width = self.width / 8;
            for row in 0..self.height {
                let row_start = row * row_byte_width;
                rows.set_row_from_le_bytes(row, &self.data[row_start..row_start + row_byte_width]);
            }
            return rows;
        }

        for row in 0..self.height {
            rows.set_row_from_bit_slice(row, &self.data, row * self.width);
        }
        rows
    }

    fn packed_row_prefix(&self, height: usize) -> PackedBinaryRows {
        assert!(height <= self.height);
        let mut rows = PackedBinaryRows::new(height, self.width);
        if self.width.is_multiple_of(8) {
            let row_byte_width = self.width / 8;
            for row in 0..height {
                let row_start = row * row_byte_width;
                rows.set_row_from_le_bytes(row, &self.data[row_start..row_start + row_byte_width]);
            }
            return rows;
        }

        for row in 0..height {
            rows.set_row_from_bit_slice(row, &self.data, row * self.width);
        }
        rows
    }

    fn packed_rows_with_first_ones(&self) -> (PackedBinaryRows, Vec<Option<usize>>) {
        let mut rows = PackedBinaryRows::new(self.height, self.width);
        let mut first_ones = Vec::with_capacity(self.height);
        if self.width.is_multiple_of(8) {
            let row_byte_width = self.width / 8;
            for row in 0..self.height {
                let row_start = row * row_byte_width;
                let (_, first_one) = rows
                    .set_row_from_le_bytes(row, &self.data[row_start..row_start + row_byte_width]);
                first_ones.push(first_one);
            }
            return (rows, first_ones);
        }

        for row in 0..self.height {
            let (_, first_one) = rows.set_row_from_bit_slice(row, &self.data, row * self.width);
            first_ones.push(first_one);
        }
        (rows, first_ones)
    }

    fn visit_row_entries<F>(&self, row: usize, mut visit: F)
    where
        F: FnMut(usize),
    {
        assert!(row < self.height);
        if self.width.is_multiple_of(8) {
            let row_byte_width = self.width / 8;
            let row_start = row * row_byte_width;
            for (byte_offset, &byte) in self.data[row_start..row_start + row_byte_width]
                .iter()
                .enumerate()
            {
                let mut byte = byte;
                while byte != 0 {
                    let set_bit = byte.trailing_zeros() as usize;
                    visit(byte_offset * 8 + set_bit);
                    byte &= byte - 1;
                }
            }
            return;
        }

        let start_bit = row * self.width;
        let end_bit = start_bit + self.width;
        let mut bit = start_bit;

        while bit < end_bit {
            let byte_index = bit / 8;
            let bit_offset = bit % 8;
            let bits_in_byte = (8 - bit_offset).min(end_bit - bit);
            let mask = (((1u16 << bits_in_byte) - 1) as u8) << bit_offset;
            let mut byte = self.data[byte_index] & mask;

            while byte != 0 {
                let set_bit = byte.trailing_zeros() as usize;
                visit(byte_index * 8 + set_bit - start_bit);
                byte &= byte - 1;
            }

            bit += bits_in_byte;
        }
    }

    fn row_entries(&self, row: usize) -> Vec<usize> {
        let mut entries = Vec::new();
        self.visit_row_entries(row, |col| entries.push(col));
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_row_entries_handles_unaligned_rows() {
        let mut matrix = DenseBinaryMatrix::new(3, 5);
        matrix.set(0, 0, true);
        matrix.set(0, 4, true);
        matrix.set(1, 1, true);
        matrix.set(1, 3, true);
        matrix.set(2, 2, true);

        assert_eq!(matrix.row_entries(0), vec![0, 4]);
        assert_eq!(matrix.row_entries(1), vec![1, 3]);
        assert_eq!(matrix.row_entries(2), vec![2]);
    }

    #[test]
    fn dense_byte_aligned_rows_pack_directly() {
        let mut matrix = DenseBinaryMatrix::new(3, 16);
        matrix.set(0, 1, true);
        matrix.set(0, 15, true);
        matrix.set(1, 8, true);

        let (packed, first_ones) = matrix.packed_rows_with_first_ones();

        assert_eq!(first_ones, vec![Some(1), Some(8), None]);
        assert!(packed.contains(0, 1));
        assert!(packed.contains(0, 15));
        assert!(packed.contains(1, 8));
        assert!(!packed.contains(1, 7));

        let prefix = matrix.packed_row_prefix(2);
        assert_eq!(prefix.height(), 2);
        assert!(prefix.contains(0, 15));
        assert!(prefix.contains(1, 8));
    }

    #[test]
    fn dense_unaligned_rows_pack_directly() {
        let mut matrix = DenseBinaryMatrix::new(3, 13);
        matrix.set(0, 0, true);
        matrix.set(0, 12, true);
        matrix.set(1, 1, true);
        matrix.set(1, 11, true);
        matrix.set(2, 6, true);

        let (packed, first_ones) = matrix.packed_rows_with_first_ones();

        assert_eq!(first_ones, vec![Some(0), Some(1), Some(6)]);
        assert!(packed.contains(0, 0));
        assert!(packed.contains(0, 12));
        assert!(packed.contains(1, 1));
        assert!(packed.contains(1, 11));
        assert!(packed.contains(2, 6));
        assert!(!packed.contains(1, 12));

        let prefix = matrix.packed_row_prefix(2);
        assert_eq!(prefix.height(), 2);
        assert!(prefix.contains(0, 12));
        assert!(prefix.contains(1, 11));
    }
}

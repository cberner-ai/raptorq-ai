#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::octet::Octet;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

pub trait BinaryMatrix: Clone {
    fn new(height: usize, width: usize) -> Self;
    fn height(&self) -> usize;
    fn width(&self) -> usize;
    fn get(&self, row: usize, col: usize) -> Octet;
    fn set(&mut self, row: usize, col: usize, value: bool);
    fn reserve_row_entries(&mut self, _row: usize, _additional: usize) {}
    fn normalize_rows(&mut self) {}
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

    fn row_entries(&self, row: usize) -> Vec<usize> {
        let mut entries = Vec::new();
        self.visit_row_entries(row, |col| entries.push(col));
        entries
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

    fn visit_row_entries<F>(&self, row: usize, mut visit: F)
    where
        F: FnMut(usize),
    {
        assert!(row < self.height);
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
}

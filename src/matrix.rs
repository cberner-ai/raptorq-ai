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

    fn row_entries(&self, row: usize) -> Vec<usize> {
        (0..self.width())
            .filter(|&col| self.get(row, col) != Octet::zero())
            .collect()
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

    fn row_entries(&self, row: usize) -> Vec<usize> {
        assert!(row < self.height);
        (0..self.width)
            .filter(|&col| self.get(row, col) != Octet::zero())
            .collect()
    }
}

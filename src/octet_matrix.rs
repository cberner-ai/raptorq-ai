#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::octet::Octet;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct DenseOctetMatrix {
    height: usize,
    width: usize,
    data: Vec<Octet>,
}

impl DenseOctetMatrix {
    pub fn new(height: usize, width: usize) -> DenseOctetMatrix {
        DenseOctetMatrix {
            height,
            width,
            data: vec![Octet::zero(); height * width],
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn get(&self, row: usize, col: usize) -> Octet {
        assert!(row < self.height);
        assert!(col < self.width);
        self.data[row * self.width + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: Octet) {
        assert!(row < self.height);
        assert!(col < self.width);
        self.data[row * self.width + col] = value;
    }
}

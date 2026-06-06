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

    pub(crate) fn row(&self, row: usize) -> &[Octet] {
        assert!(row < self.height);
        let start = row * self.width;
        &self.data[start..start + self.width]
    }

    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [Octet] {
        assert!(row < self.height);
        let start = row * self.width;
        &mut self.data[start..start + self.width]
    }

    pub(crate) fn as_slice(&self) -> &[Octet] {
        &self.data
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_accessors_expose_contiguous_storage() {
        let mut matrix = DenseOctetMatrix::new(2, 3);
        matrix
            .row_mut(1)
            .copy_from_slice(&[Octet::new(4), Octet::new(5), Octet::new(6)]);

        assert_eq!(
            matrix.row(0),
            &[Octet::zero(), Octet::zero(), Octet::zero()]
        );
        assert_eq!(
            matrix.as_slice(),
            &[
                Octet::zero(),
                Octet::zero(),
                Octet::zero(),
                Octet::new(4),
                Octet::new(5),
                Octet::new(6)
            ]
        );
        assert_eq!(matrix.get(1, 1), Octet::new(5));
    }
}

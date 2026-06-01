#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::matrix::BinaryMatrix;
use crate::octet::Octet;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct SparseBinaryMatrix {
    width: usize,
    rows: Vec<Vec<usize>>,
}

impl BinaryMatrix for SparseBinaryMatrix {
    fn new(height: usize, width: usize) -> SparseBinaryMatrix {
        SparseBinaryMatrix {
            width,
            rows: vec![Vec::new(); height],
        }
    }

    fn height(&self) -> usize {
        self.rows.len()
    }

    fn width(&self) -> usize {
        self.width
    }

    fn get(&self, row: usize, col: usize) -> Octet {
        assert!(row < self.rows.len());
        assert!(col < self.width);
        if self.rows[row].binary_search(&col).is_ok() {
            Octet::one()
        } else {
            Octet::zero()
        }
    }

    fn set(&mut self, row: usize, col: usize, value: bool) {
        assert!(row < self.rows.len());
        assert!(col < self.width);
        match self.rows[row].binary_search(&col) {
            Ok(index) if !value => {
                self.rows[row].remove(index);
            }
            Err(index) if value => {
                self.rows[row].insert(index, col);
            }
            _ => {}
        }
    }

    fn toggle(&mut self, row: usize, col: usize) {
        assert!(row < self.rows.len());
        assert!(col < self.width);
        match self.rows[row].binary_search(&col) {
            Ok(index) => {
                self.rows[row].remove(index);
            }
            Err(index) => {
                self.rows[row].insert(index, col);
            }
        }
    }

    fn visit_row_entries<F>(&self, row: usize, mut visit: F)
    where
        F: FnMut(usize),
    {
        assert!(row < self.rows.len());
        for &col in &self.rows[row] {
            visit(col);
        }
    }

    fn row_entries(&self, row: usize) -> Vec<usize> {
        assert!(row < self.rows.len());
        self.rows[row].clone()
    }
}

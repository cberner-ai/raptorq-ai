#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::gf2::PackedBinaryRows;
use crate::matrix::BinaryMatrix;
use crate::octet::Octet;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

const APPEND_BUILD_MIN_WIDTH: usize = 512;
const PARALLEL_PACK_METADATA_MIN_ROWS: usize = 16_384;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct SparseBinaryMatrix {
    width: usize,
    rows: Vec<Vec<usize>>,
    #[cfg_attr(feature = "serde_support", serde(skip, default))]
    systematic_source_block_symbols: Option<u32>,
    #[cfg_attr(feature = "serde_support", serde(skip, default))]
    contiguous_single_repair_systematic_rows: Option<(u32, usize, usize, u32)>,
    #[cfg_attr(feature = "serde_support", serde(skip, default))]
    systematic_row_isis: Option<Vec<Option<u32>>>,
    #[cfg_attr(feature = "serde_support", serde(skip, default))]
    rows_normalized: bool,
}

impl PartialEq for SparseBinaryMatrix {
    fn eq(&self, other: &Self) -> bool {
        if self.width != other.width || self.rows.len() != other.rows.len() {
            return false;
        }

        self.rows
            .iter()
            .zip(other.rows.iter())
            .all(|(left, right)| sorted_entries(left) == sorted_entries(right))
    }
}

impl Eq for SparseBinaryMatrix {}

fn sorted_entries(entries: &[usize]) -> Vec<usize> {
    let mut sorted = entries.to_vec();
    sorted.sort_unstable();
    sorted
}

impl BinaryMatrix for SparseBinaryMatrix {
    fn new(height: usize, width: usize) -> SparseBinaryMatrix {
        SparseBinaryMatrix {
            width,
            rows: vec![Vec::new(); height],
            systematic_source_block_symbols: None,
            contiguous_single_repair_systematic_rows: None,
            systematic_row_isis: None,
            rows_normalized: true,
        }
    }

    fn height(&self) -> usize {
        self.rows.len()
    }

    fn width(&self) -> usize {
        self.width
    }

    fn systematic_source_block_symbols(&self) -> Option<u32> {
        self.systematic_source_block_symbols
    }

    fn mark_systematic_source_block_symbols(&mut self, source_block_symbols: u32) {
        self.systematic_source_block_symbols = Some(source_block_symbols);
    }

    fn contiguous_single_repair_systematic_rows(&self) -> Option<(u32, usize, usize, u32)> {
        self.contiguous_single_repair_systematic_rows
    }

    fn mark_contiguous_single_repair_systematic_rows(
        &mut self,
        source_block_symbols: u32,
        missing_isi: usize,
        repair_matrix_row: usize,
        repair_isi: u32,
    ) {
        self.contiguous_single_repair_systematic_rows = Some((
            source_block_symbols,
            missing_isi,
            repair_matrix_row,
            repair_isi,
        ));
    }

    fn systematic_row_isis(&self) -> Option<&[Option<u32>]> {
        self.systematic_row_isis.as_deref()
    }

    fn mark_encoded_systematic_isis(
        &mut self,
        row_offset: usize,
        source_block_symbols: u32,
        encoded_isis: &[u32],
    ) {
        assert!(row_offset + encoded_isis.len() <= self.rows.len());
        let mut row_isis = vec![None; self.rows.len()];
        for (offset, &isi) in encoded_isis.iter().enumerate() {
            if isi < source_block_symbols {
                row_isis[row_offset + offset] = Some(isi);
            }
        }
        self.systematic_row_isis = Some(row_isis);
    }

    fn get(&self, row: usize, col: usize) -> Octet {
        assert!(row < self.rows.len());
        assert!(col < self.width);
        let contains = if self.rows_normalized {
            self.rows[row].binary_search(&col).is_ok()
        } else {
            self.rows[row].contains(&col)
        };
        if contains {
            Octet::one()
        } else {
            Octet::zero()
        }
    }

    fn set(&mut self, row: usize, col: usize, value: bool) {
        assert!(row < self.rows.len());
        assert!(col < self.width);
        if self.width < APPEND_BUILD_MIN_WIDTH {
            match self.rows[row].binary_search(&col) {
                Ok(index) if !value => {
                    self.rows[row].remove(index);
                }
                Err(index) if value => {
                    self.rows[row].insert(index, col);
                }
                _ => {}
            }
            return;
        }

        let row_entries = &mut self.rows[row];
        let mut changed = false;
        match row_entries.iter().position(|&entry| entry == col) {
            Some(index) if !value => {
                row_entries.swap_remove(index);
                changed = true;
            }
            None if value => {
                row_entries.push(col);
                changed = true;
            }
            _ => {}
        }
        if changed {
            self.rows_normalized = false;
        }
    }

    fn reserve_row_entries(&mut self, row: usize, additional: usize) {
        assert!(row < self.rows.len());
        self.rows[row].reserve(additional);
    }

    fn toggle(&mut self, row: usize, col: usize) {
        assert!(row < self.rows.len());
        assert!(col < self.width);
        if self.width < APPEND_BUILD_MIN_WIDTH {
            match self.rows[row].binary_search(&col) {
                Ok(index) => {
                    self.rows[row].remove(index);
                }
                Err(index) => {
                    self.rows[row].insert(index, col);
                }
            }
            return;
        }

        let row_entries = &mut self.rows[row];
        match row_entries.iter().position(|&entry| entry == col) {
            Some(index) => {
                row_entries.swap_remove(index);
            }
            None => {
                row_entries.push(col);
            }
        }
        self.rows_normalized = false;
    }

    fn normalize_rows(&mut self) {
        if self.rows_normalized {
            return;
        }
        for row in &mut self.rows {
            row.sort_unstable();
        }
        self.rows_normalized = true;
    }

    fn toggle_unique(&mut self, row: usize, col: usize) {
        assert!(row < self.rows.len());
        assert!(col < self.width);
        debug_assert!(!self.rows[row].contains(&col));
        self.rows[row].push(col);
        self.rows_normalized = false;
    }

    fn packed_rows(&self) -> PackedBinaryRows {
        let mut packed = PackedBinaryRows::new(self.rows.len(), self.width);
        for (row, entries) in self.rows.iter().enumerate() {
            packed.set_entries(row, entries);
        }
        packed
    }

    fn packed_row_prefix(&self, height: usize) -> PackedBinaryRows {
        assert!(height <= self.rows.len());
        let mut packed = PackedBinaryRows::new(height, self.width);
        for (row, entries) in self.rows[..height].iter().enumerate() {
            packed.set_entries(row, entries);
        }
        packed
    }

    fn packed_row_prefix_with_row_weights_and_first_ones(
        &self,
        height: usize,
    ) -> (PackedBinaryRows, Vec<u32>, Vec<Option<usize>>) {
        assert!(height <= self.rows.len());
        let mut packed = PackedBinaryRows::new(height, self.width);
        let mut row_weights = Vec::with_capacity(height);
        let mut first_ones = Vec::with_capacity(height);
        for (row, entries) in self.rows[..height].iter().enumerate() {
            row_weights.push(entries.len() as u32);
            let mut first_one = entries.first().copied();
            for &col in entries {
                if !self.rows_normalized && first_one.is_some_and(|first| col < first) {
                    first_one = Some(col);
                }
            }
            packed.set_entries(row, entries);
            first_ones.push(first_one);
        }
        (packed, row_weights, first_ones)
    }

    fn packed_rows_with_first_ones(&self) -> (PackedBinaryRows, Vec<Option<usize>>) {
        let mut packed = PackedBinaryRows::new(self.rows.len(), self.width);
        let mut first_ones = Vec::with_capacity(self.rows.len());
        for (row, entries) in self.rows.iter().enumerate() {
            let mut first_one = entries.first().copied();
            for &col in entries {
                if !self.rows_normalized && first_one.is_some_and(|first| col < first) {
                    first_one = Some(col);
                }
            }
            packed.set_entries(row, entries);
            first_ones.push(first_one);
        }
        (packed, first_ones)
    }

    fn packed_rows_with_row_weights_and_first_ones(
        &self,
    ) -> (PackedBinaryRows, Vec<u32>, Vec<Option<usize>>) {
        if !self.rows_normalized && self.rows.len() >= PARALLEL_PACK_METADATA_MIN_ROWS {
            return PackedBinaryRows::from_sparse_entries_with_row_weights_and_first_ones(
                self.width, &self.rows,
            );
        }

        let mut packed = PackedBinaryRows::new(self.rows.len(), self.width);
        let mut row_weights = Vec::with_capacity(self.rows.len());
        let mut first_ones = Vec::with_capacity(self.rows.len());
        for (row, entries) in self.rows.iter().enumerate() {
            row_weights.push(entries.len() as u32);
            let mut first_one = entries.first().copied();
            for &col in entries {
                if !self.rows_normalized && first_one.is_some_and(|first| col < first) {
                    first_one = Some(col);
                }
            }
            packed.set_entries(row, entries);
            first_ones.push(first_one);
        }
        (packed, row_weights, first_ones)
    }

    fn visit_row_entries<F>(&self, row: usize, mut visit: F)
    where
        F: FnMut(usize),
    {
        assert!(row < self.rows.len());
        if self.rows_normalized {
            for &col in &self.rows[row] {
                visit(col);
            }
        } else {
            let entries = sorted_entries(&self.rows[row]);
            for col in entries {
                visit(col);
            }
        }
    }

    fn row_entries(&self, row: usize) -> Vec<usize> {
        assert!(row < self.rows.len());
        let mut entries = self.rows[row].clone();
        if !self.rows_normalized {
            entries.sort_unstable();
        }
        entries
    }

    fn into_row_entries(mut self) -> Vec<Vec<usize>> {
        if !self.rows_normalized {
            for row in &mut self.rows {
                row.sort_unstable();
            }
        }
        self.rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_rows_are_sorted_when_read_after_unsorted_toggles() {
        let mut matrix = SparseBinaryMatrix::new(1, APPEND_BUILD_MIN_WIDTH);
        matrix.toggle(0, 1);
        matrix.toggle(0, 5);
        matrix.toggle(0, 3);
        matrix.set(0, 7, true);
        matrix.set(0, 1, false);

        assert!(!matrix.rows_normalized);
        assert_ne!(matrix.rows[0], vec![3, 5, 7]);
        assert_eq!(matrix.row_entries(0), vec![3, 5, 7]);

        let mut visited = Vec::new();
        matrix.visit_row_entries(0, |col| visited.push(col));
        assert_eq!(visited, vec![3, 5, 7]);

        matrix.normalize_rows();
        assert!(matrix.rows_normalized);
        assert_eq!(matrix.rows[0], vec![3, 5, 7]);
        assert_eq!(matrix.row_entries(0), vec![3, 5, 7]);
    }

    #[test]
    fn packed_rows_with_metadata_counts_sparse_entries() {
        let mut matrix = SparseBinaryMatrix::new(2, 96);
        matrix.toggle(0, 70);
        matrix.toggle(0, 3);
        matrix.toggle(1, 64);

        let (packed, row_weights, first_ones) =
            matrix.packed_rows_with_row_weights_and_first_ones();

        assert_eq!(row_weights, vec![2, 1]);
        assert_eq!(first_ones, vec![Some(3), Some(64)]);
        assert!(packed.contains(0, 3));
        assert!(packed.contains(0, 70));
        assert!(packed.contains(1, 64));
    }

    #[test]
    fn packed_rows_with_metadata_returns_first_sparse_entries() {
        let mut matrix = SparseBinaryMatrix::new(3, APPEND_BUILD_MIN_WIDTH);
        matrix.toggle(0, 70);
        matrix.toggle(0, 3);
        matrix.toggle(1, 64);

        let (packed, row_weights, first_ones) =
            matrix.packed_rows_with_row_weights_and_first_ones();

        assert_eq!(row_weights, vec![2, 1, 0]);
        assert_eq!(first_ones, vec![Some(3), Some(64), None]);
        assert!(packed.contains(0, 3));
        assert!(packed.contains(0, 70));
        assert!(packed.contains(1, 64));
    }

    #[test]
    fn packed_row_prefix_with_metadata_omits_suffix_rows() {
        let mut matrix = SparseBinaryMatrix::new(3, APPEND_BUILD_MIN_WIDTH);
        matrix.toggle(0, 70);
        matrix.toggle(0, 3);
        matrix.toggle(1, 64);
        matrix.toggle(2, 95);

        let (packed, row_weights, first_ones) =
            matrix.packed_row_prefix_with_row_weights_and_first_ones(2);

        assert_eq!(packed.height(), 2);
        assert_eq!(row_weights, vec![2, 1]);
        assert_eq!(first_ones, vec![Some(3), Some(64)]);
        assert!(packed.contains(0, 3));
        assert!(packed.contains(0, 70));
        assert!(packed.contains(1, 64));
    }
}

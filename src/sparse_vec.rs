#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::octet::Octet;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SparseOctetVec {
    entries: Vec<(usize, Octet)>,
}

impl SparseOctetVec {
    pub(crate) fn new() -> SparseOctetVec {
        SparseOctetVec {
            entries: Vec::new(),
        }
    }

    pub(crate) fn from_binary_entries(entries: impl IntoIterator<Item = usize>) -> SparseOctetVec {
        SparseOctetVec::from_octet_entries(entries.into_iter().map(|col| (col, Octet::one())))
    }

    pub(crate) fn from_octet_entries(
        entries: impl IntoIterator<Item = (usize, Octet)>,
    ) -> SparseOctetVec {
        let mut entries: Vec<_> = entries
            .into_iter()
            .filter(|&(_, value)| !value.is_zero())
            .collect();
        entries.sort_unstable_by_key(|&(col, _)| col);

        let mut compacted: Vec<(usize, Octet)> = Vec::with_capacity(entries.len());
        for (col, value) in entries {
            if let Some((last_col, last_value)) = compacted.last_mut()
                && *last_col == col
            {
                *last_value += value;
                if last_value.is_zero() {
                    compacted.pop();
                }
                continue;
            }
            compacted.push((col, value));
        }

        SparseOctetVec { entries: compacted }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn get(&self, col: usize) -> Octet {
        self.entries
            .binary_search_by_key(&col, |&(entry_col, _)| entry_col)
            .map(|index| self.entries[index].1)
            .unwrap_or_else(|_| Octet::zero())
    }

    pub(crate) fn scale_from(&mut self, start_col: usize, scalar: Octet) {
        let split = self.entries.partition_point(|&(col, _)| col < start_col);
        if scalar.is_zero() {
            self.entries.truncate(split);
            return;
        }
        if scalar == Octet::one() {
            return;
        }

        for (_, value) in self.entries[split..].iter_mut() {
            *value *= scalar;
        }
        self.entries.retain(|&(_, value)| !value.is_zero());
    }

    pub(crate) fn add_scaled_from(
        &mut self,
        src: &SparseOctetVec,
        start_col: usize,
        scalar: Octet,
    ) {
        if scalar.is_zero() {
            return;
        }

        let mut merged = Vec::with_capacity(self.entries.len() + src.entries.len());
        let mut dest_index = 0usize;
        let mut src_index = src.entries.partition_point(|&(col, _)| col < start_col);

        while dest_index < self.entries.len() || src_index < src.entries.len() {
            match (self.entries.get(dest_index), src.entries.get(src_index)) {
                (Some(&(dest_col, dest_value)), Some(&(src_col, src_value))) => {
                    if dest_col < src_col {
                        merged.push((dest_col, dest_value));
                        dest_index += 1;
                    } else if src_col < dest_col {
                        merged.push((src_col, src_value * scalar));
                        src_index += 1;
                    } else {
                        let value = dest_value + src_value * scalar;
                        if !value.is_zero() {
                            merged.push((dest_col, value));
                        }
                        dest_index += 1;
                        src_index += 1;
                    }
                }
                (Some(&(dest_col, dest_value)), None) => {
                    merged.push((dest_col, dest_value));
                    dest_index += 1;
                }
                (None, Some(&(src_col, src_value))) => {
                    merged.push((src_col, src_value * scalar));
                    src_index += 1;
                }
                (None, None) => break,
            }
        }

        self.entries = merged;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_entries_are_sorted_and_xor_duplicates() {
        let row = SparseOctetVec::from_binary_entries([5, 1, 5, 3]);

        assert_eq!(row.get(1), Octet::one());
        assert_eq!(row.get(3), Octet::one());
        assert_eq!(row.get(5), Octet::zero());
        assert_eq!(row.get(4), Octet::zero());
    }

    #[test]
    fn scale_from_keeps_lower_columns_and_drops_zeroed_tail() {
        let mut row = SparseOctetVec::from_octet_entries([
            (0, Octet::new(7)),
            (2, Octet::new(9)),
            (4, Octet::new(11)),
        ]);

        row.scale_from(2, Octet::zero());

        assert_eq!(row.get(0), Octet::new(7));
        assert_eq!(row.get(2), Octet::zero());
        assert_eq!(row.get(4), Octet::zero());
    }

    #[test]
    fn add_scaled_from_merges_in_gf256_from_start_column() {
        let mut dest = SparseOctetVec::from_octet_entries([
            (0, Octet::new(5)),
            (2, Octet::new(7)),
            (5, Octet::new(1)),
        ]);
        let src = SparseOctetVec::from_octet_entries([
            (1, Octet::new(9)),
            (2, Octet::new(7)),
            (5, Octet::new(3)),
        ]);

        dest.add_scaled_from(&src, 2, Octet::one());

        assert_eq!(dest.get(0), Octet::new(5));
        assert_eq!(dest.get(1), Octet::zero());
        assert_eq!(dest.get(2), Octet::zero());
        assert_eq!(dest.get(5), Octet::new(2));
    }
}

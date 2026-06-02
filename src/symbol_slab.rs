#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::symbol::Symbol;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct SymbolSlab {
    bytes: Vec<u8>,
    symbol_size: usize,
}

impl SymbolSlab {
    pub fn with_zeros(symbols: usize, symbol_size: usize) -> SymbolSlab {
        assert_ne!(symbol_size, 0);
        SymbolSlab {
            bytes: vec![0; symbols * symbol_size],
            symbol_size,
        }
    }

    pub fn from_bytes(bytes: Vec<u8>, symbol_size: usize) -> SymbolSlab {
        assert_ne!(symbol_size, 0);
        assert_eq!(bytes.len() % symbol_size, 0);
        SymbolSlab { bytes, symbol_size }
    }

    #[allow(dead_code)]
    pub fn from_symbols(symbols: Vec<Symbol>, symbol_size: usize) -> SymbolSlab {
        let mut bytes = Vec::with_capacity(symbols.len() * symbol_size);
        for symbol in symbols {
            assert_eq!(symbol.len(), symbol_size);
            bytes.extend_from_slice(symbol.as_bytes());
        }
        SymbolSlab { bytes, symbol_size }
    }

    pub fn len(&self) -> usize {
        self.bytes.len() / self.symbol_size
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn symbol_size(&self) -> usize {
        self.symbol_size
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    pub fn get(&self, index: usize) -> &[u8] {
        let start = index * self.symbol_size;
        &self.bytes[start..start + self.symbol_size]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut [u8] {
        let start = index * self.symbol_size;
        &mut self.bytes[start..start + self.symbol_size]
    }

    pub fn get_disjoint_mut(&mut self, first: usize, second: usize) -> (&mut [u8], &mut [u8]) {
        assert_ne!(first, second);
        let symbol_size = self.symbol_size;
        let first_start = first * symbol_size;
        let second_start = second * symbol_size;

        if first_start < second_start {
            let (left, right) = self.bytes.split_at_mut(second_start);
            (
                &mut left[first_start..first_start + symbol_size],
                &mut right[..symbol_size],
            )
        } else {
            let (left, right) = self.bytes.split_at_mut(first_start);
            (
                &mut right[..symbol_size],
                &mut left[second_start..second_start + symbol_size],
            )
        }
    }

    pub fn swap_symbols(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let (a_symbol, b_symbol) = self.get_disjoint_mut(a, b);
        a_symbol.swap_with_slice(b_symbol);
    }

    pub fn copy_block_from(&mut self, start_symbol: usize, src: &[u8]) {
        assert_eq!(src.len() % self.symbol_size, 0);
        let start = start_symbol * self.symbol_size;
        let end = start + src.len();
        self.bytes[start..end].copy_from_slice(src);
    }
}

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(feature = "std")]
use crate::constraint_matrix::generate_constraint_matrix;
use crate::octet::Octet;
use crate::octets::{fused_addassign_mul_scalar, mulassign_scalar};
#[cfg(feature = "std")]
use crate::pi_solver::fused_inverse_mul_symbols_without_ops;
#[cfg(feature = "std")]
use crate::sparse_matrix::SparseBinaryMatrix;
use crate::symbol_slab::SymbolSlab;
#[cfg(feature = "std")]
use crate::systematic_constants::extended_source_block_symbols;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum SymbolOps {
    Swap(usize, usize),
    Scale(usize, Octet),
    FusedAdd {
        dest: usize,
        src: usize,
        scalar: Octet,
    },
    #[cfg(feature = "std")]
    Solve {
        source_block_symbols: u32,
    },
}

pub fn perform_op(op: &SymbolOps, symbols: &mut SymbolSlab) {
    match *op {
        SymbolOps::Swap(a, b) => {
            symbols.swap_symbols(a, b);
        }
        SymbolOps::Scale(row, scalar) => {
            mulassign_scalar(symbols.get_mut(row), &scalar);
        }
        SymbolOps::FusedAdd { dest, src, scalar } => {
            fused_addassign_symbol(symbols, dest, src, scalar);
        }
        #[cfg(feature = "std")]
        SymbolOps::Solve {
            source_block_symbols,
        } => {
            solve_into(symbols, source_block_symbols);
        }
    }
}

fn fused_addassign_symbol(symbols: &mut SymbolSlab, dest: usize, src: usize, scalar: Octet) {
    if dest == src {
        let combined = Octet::one() + scalar;
        mulassign_scalar(symbols.get_mut(dest), &combined);
    } else {
        let (src_symbol, dest_symbol) = symbols.get_disjoint_mut(src, dest);
        fused_addassign_mul_scalar(dest_symbol, src_symbol, &scalar);
    }
}

#[cfg(feature = "std")]
fn solve_into(symbols: &mut SymbolSlab, source_block_symbols: u32) {
    let extended_symbols = extended_source_block_symbols(source_block_symbols);
    let indices: Vec<u32> = (0..extended_symbols).collect();
    let (matrix, hdpc_rows) =
        generate_constraint_matrix::<SparseBinaryMatrix>(source_block_symbols, &indices);
    let decoded = fused_inverse_mul_symbols_without_ops(
        matrix,
        hdpc_rows,
        symbols.clone(),
        source_block_symbols,
    )
    .expect("intermediate-symbol solve failed");
    for row in 0..decoded.len() {
        symbols.get_mut(row).copy_from_slice(decoded.get(row));
    }
}

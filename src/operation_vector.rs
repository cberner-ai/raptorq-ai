use crate::octet::Octet;
use crate::octets::{fused_addassign_mul_scalar, mulassign_scalar};
#[cfg(feature = "std")]
use crate::pi_solver::apply_cached_systematic_plan;
#[cfg(not(feature = "std"))]
use crate::pi_solver::apply_direct_systematic_solve;
use crate::symbol_slab::SymbolSlab;
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
    ApplyCachedSystematicPlan {
        source_block_symbols: u32,
    },
    #[cfg(not(feature = "std"))]
    DirectSystematicSolve {
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
        SymbolOps::ApplyCachedSystematicPlan {
            source_block_symbols,
        } => {
            apply_cached_systematic_plan(source_block_symbols, symbols);
        }
        #[cfg(not(feature = "std"))]
        SymbolOps::DirectSystematicSolve {
            source_block_symbols,
        } => {
            apply_direct_systematic_solve(source_block_symbols, symbols);
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

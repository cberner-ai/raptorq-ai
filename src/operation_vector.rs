use crate::octet::Octet;
use crate::octets::{add_assign, fused_addassign_mul_scalar, mulassign_scalar};
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

fn addassign_symbol(symbols: &mut SymbolSlab, dest: usize, src: usize) {
    if dest == src {
        symbols.get_mut(dest).fill(0);
    } else {
        let (src_symbol, dest_symbol) = symbols.get_disjoint_mut(src, dest);
        add_assign(dest_symbol, src_symbol);
    }
}

fn fused_addassign_symbol(symbols: &mut SymbolSlab, dest: usize, src: usize, scalar: Octet) {
    if scalar == Octet::one() {
        addassign_symbol(symbols, dest, src);
    } else if dest == src {
        let combined = Octet::one() + scalar;
        mulassign_scalar(symbols.get_mut(dest), &combined);
    } else {
        let (src_symbol, dest_symbol) = symbols.get_disjoint_mut(src, dest);
        fused_addassign_mul_scalar(dest_symbol, src_symbol, &scalar);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_add_matches_fused_add_with_one() {
        let mut fused =
            SymbolSlab::from_bytes(vec![3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], 4);
        let mut add = fused.clone();

        perform_op(
            &SymbolOps::FusedAdd {
                dest: 1,
                src: 0,
                scalar: Octet::one(),
            },
            &mut fused,
        );
        let (src_symbol, dest_symbol) = add.get_disjoint_mut(0, 1);
        add_assign(dest_symbol, src_symbol);

        assert_eq!(add, fused);
    }
}

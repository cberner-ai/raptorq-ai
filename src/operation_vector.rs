#[cfg(feature = "std")]
use std::boxed::Box;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

use crate::octet::Octet;
use crate::octets::{
    AddAssignFastPath, FusedAddAssignMulScalarFastPath, add_assign, bytes_are_zero,
    fused_addassign_mul_scalar, mulassign_scalar,
};
#[cfg(feature = "std")]
use crate::pi_solver::apply_cached_systematic_plan;
use crate::pi_solver::apply_direct_systematic_solve;
use crate::symbol_slab::SymbolSlab;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

const REPLAY_BATCH_FAST_PATH_MIN_SYMBOLS: usize = 500;
const REPLAY_BATCH_FAST_PATH_MIN_DESTS_FOR_LARGE_SYMBOLS: usize = 2;
const REPLAY_BATCH_FAST_PATH_LARGE_SYMBOLS: usize = 20_000;
const REPLAY_BATCH_FAST_PATH_MAX_SYMBOLS: usize = 32_768;

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
    FusedAddBatch {
        src: usize,
        dests: Box<[(usize, Octet)]>,
    },
    #[cfg(feature = "std")]
    ApplyCachedSystematicPlan {
        source_block_symbols: u32,
    },
    #[cfg_attr(feature = "serde_support", serde(alias = "ApplyDirectSystematicSolve"))]
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
        SymbolOps::FusedAddBatch { src, ref dests } => {
            fused_addassign_symbol_batch_for_replay(symbols, src, dests);
        }
        #[cfg(feature = "std")]
        SymbolOps::ApplyCachedSystematicPlan {
            source_block_symbols,
        } => {
            apply_cached_systematic_plan(source_block_symbols, symbols);
        }
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

pub(crate) fn fused_addassign_symbol_batch(
    symbols: &mut SymbolSlab,
    src: usize,
    dests: &[(usize, Octet)],
) {
    fused_addassign_symbol_batch_inner::<false>(symbols, src, dests);
}

fn fused_addassign_symbol_batch_for_replay(
    symbols: &mut SymbolSlab,
    src: usize,
    dests: &[(usize, Octet)],
) {
    fused_addassign_symbol_batch_inner::<true>(symbols, src, dests);
}

#[inline]
fn use_replay_batch_fast_paths(symbol_count: usize, dest_count: usize) -> bool {
    (REPLAY_BATCH_FAST_PATH_MIN_SYMBOLS..=REPLAY_BATCH_FAST_PATH_MAX_SYMBOLS)
        .contains(&symbol_count)
        && dest_count > 1
        && (symbol_count < REPLAY_BATCH_FAST_PATH_LARGE_SYMBOLS
            || dest_count >= REPLAY_BATCH_FAST_PATH_MIN_DESTS_FOR_LARGE_SYMBOLS)
}

fn fused_addassign_symbol_batch_inner<const REUSE_FAST_PATHS: bool>(
    symbols: &mut SymbolSlab,
    src: usize,
    dests: &[(usize, Octet)],
) {
    if dests.is_empty() {
        return;
    }

    let symbol_size = symbols.symbol_size();
    let symbol_count = symbols.len();
    let bytes = symbols.as_mut_bytes();
    let src_start = src * symbol_size;
    assert!(src_start + symbol_size <= bytes.len());
    let src_ptr = unsafe { bytes.as_ptr().add(src_start) };
    let src_symbol = unsafe { core::slice::from_raw_parts(src_ptr, symbol_size) };
    if bytes_are_zero(src_symbol) {
        return;
    }
    let bytes_ptr = bytes.as_mut_ptr();

    if REUSE_FAST_PATHS && use_replay_batch_fast_paths(symbol_count, dests.len()) {
        let add_fast_path = AddAssignFastPath::new(symbol_size);
        let fused_fast_path = FusedAddAssignMulScalarFastPath::new(symbol_size);

        for &(dest, scalar) in dests {
            if scalar.is_zero() {
                continue;
            }
            assert_ne!(dest, src);
            let dest_start = dest * symbol_size;
            assert!(dest_start + symbol_size <= bytes.len());
            unsafe {
                let dest_symbol =
                    core::slice::from_raw_parts_mut(bytes_ptr.add(dest_start), symbol_size);
                if scalar == Octet::one() {
                    add_fast_path.apply_same_len(dest_symbol, src_symbol);
                } else {
                    fused_fast_path.apply_nonzero(dest_symbol, src_symbol, &scalar);
                }
            }
        }
        return;
    }

    for &(dest, scalar) in dests {
        if scalar.is_zero() {
            continue;
        }
        assert_ne!(dest, src);
        let dest_start = dest * symbol_size;
        assert!(dest_start + symbol_size <= bytes.len());
        unsafe {
            let dest_symbol =
                core::slice::from_raw_parts_mut(bytes_ptr.add(dest_start), symbol_size);
            if scalar == Octet::one() {
                add_assign(dest_symbol, src_symbol);
            } else {
                fused_addassign_mul_scalar(dest_symbol, src_symbol, &scalar);
            }
        }
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

    #[test]
    fn batch_fused_add_matches_individual_ops() {
        let mut batch = SymbolSlab::from_bytes(
            vec![3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59],
            4,
        );
        let mut individual = batch.clone();
        let dests = vec![(1, Octet::one()), (2, Octet::new(7)), (3, Octet::new(19))];

        perform_op(
            &SymbolOps::FusedAddBatch {
                src: 0,
                dests: dests.clone().into_boxed_slice(),
            },
            &mut batch,
        );
        for (dest, scalar) in dests {
            perform_op(
                &SymbolOps::FusedAdd {
                    dest,
                    src: 0,
                    scalar,
                },
                &mut individual,
            );
        }

        assert_eq!(individual, batch);
    }

    #[test]
    fn zero_source_batch_does_not_change_destinations() {
        let mut symbols =
            SymbolSlab::from_bytes(vec![0, 0, 0, 0, 13, 17, 19, 23, 29, 31, 37, 41], 4);
        let expected = symbols.clone();

        perform_op(
            &SymbolOps::FusedAddBatch {
                src: 0,
                dests: vec![(1, Octet::one()), (2, Octet::new(7))].into_boxed_slice(),
            },
            &mut symbols,
        );

        assert_eq!(expected, symbols);
    }

    #[test]
    fn replay_batch_fast_path_covers_mid_ci_rows() {
        assert!(!use_replay_batch_fast_paths(10, 2));
        assert!(!use_replay_batch_fast_paths(100, 2));
        assert!(!use_replay_batch_fast_paths(250, 2));
        assert!(use_replay_batch_fast_paths(500, 2));
        assert!(use_replay_batch_fast_paths(1_000, 2));
        assert!(use_replay_batch_fast_paths(5_000, 2));
        assert!(use_replay_batch_fast_paths(20_000, 2));
        assert!(!use_replay_batch_fast_paths(50_000, 4));
    }
}

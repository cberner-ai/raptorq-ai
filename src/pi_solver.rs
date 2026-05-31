#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::base::intermediate_tuple;
use crate::constraint_matrix::{enc_indices, generate_hdpc_rows};
use crate::matrix::BinaryMatrix;
use crate::octet::Octet;
use crate::octet_matrix::DenseOctetMatrix;
use crate::octets::{fused_addassign_mul_scalar, mulassign_scalar};
use crate::operation_vector::SymbolOps;
use crate::symbol_slab::SymbolSlab;
use crate::systematic_constants::num_ldpc_symbols;
use crate::systematic_constants::{
    calculate_p1, extended_source_block_symbols, num_intermediate_symbols, num_lt_symbols,
    num_pi_symbols, systematic_index,
};

type CoefficientRow = Vec<(usize, Octet)>;
const MAX_RECORDED_SOLVER_WIDTH: usize = 4096;

pub fn fused_inverse_mul_symbols<M: BinaryMatrix>(
    matrix: M,
    hdpc_rows: DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let is_systematic_plan = is_full_systematic_planning_matrix(&matrix, source_block_symbols);
    if matrix.width() > MAX_RECORDED_SOLVER_WIDTH {
        if is_systematic_plan {
            #[cfg(feature = "std")]
            {
                return (
                    None,
                    Some(vec![SymbolOps::Solve {
                        source_block_symbols,
                    }]),
                );
            }
        }

        return fused_inverse_mul_symbols_impl(
            matrix,
            hdpc_rows,
            symbols,
            source_block_symbols,
            false,
        );
    }
    fused_inverse_mul_symbols_impl(
        matrix,
        hdpc_rows,
        symbols,
        source_block_symbols,
        is_systematic_plan,
    )
}

#[cfg(feature = "std")]
pub(crate) fn fused_inverse_mul_symbols_without_ops<M: BinaryMatrix>(
    matrix: M,
    hdpc_rows: DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
) -> Option<SymbolSlab> {
    fused_inverse_mul_symbols_impl(matrix, hdpc_rows, symbols, source_block_symbols, false).0
}

fn fused_inverse_mul_symbols_impl<M: BinaryMatrix>(
    matrix: M,
    hdpc_rows: DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
    record_ops: bool,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let width = matrix.width();
    let total_rows = matrix.height() + hdpc_rows.height();
    assert_eq!(symbols.len(), total_rows);
    assert_eq!(hdpc_rows.width(), width);
    assert!(matrix.height() >= s);

    let rows = coefficient_rows(&matrix, &hdpc_rows, source_block_symbols);
    solve(rows, width, symbols, record_ops)
}

fn coefficient_rows<M: BinaryMatrix>(
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    source_block_symbols: u32,
) -> Vec<CoefficientRow> {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let total_rows = matrix.height() + hdpc_rows.height();
    let mut rows = vec![Vec::new(); total_rows];
    for row in 0..s {
        rows[row] = copy_binary_row(matrix, row);
    }
    for row in 0..hdpc_rows.height() {
        let dest = s + row;
        rows[dest] = copy_octet_row(hdpc_rows, row);
    }
    for row in s..matrix.height() {
        let dest = row + hdpc_rows.height();
        rows[dest] = copy_binary_row(matrix, row);
    }

    rows
}

pub fn fused_inverse_mul_symbols_no_hdpc<M: BinaryMatrix>(
    matrix: M,
    symbols: SymbolSlab,
    source_block_symbols: u32,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let width = matrix.width();
    assert_eq!(symbols.len(), matrix.height());
    let mut rows = Vec::with_capacity(matrix.height());
    for row in 0..matrix.height() {
        rows.push(copy_binary_row(&matrix, row));
    }

    let (decoded, ops) = solve(rows, width, symbols, false);
    match decoded {
        Some(decoded) => (verify_no_hdpc_solution(decoded, source_block_symbols), ops),
        None => (None, ops),
    }
}

fn verify_no_hdpc_solution(decoded: SymbolSlab, source_block_symbols: u32) -> Option<SymbolSlab> {
    if decoded.len() != num_intermediate_symbols(source_block_symbols) as usize {
        return Some(decoded);
    }

    let hdpc_rows = generate_hdpc_rows(source_block_symbols);
    if hdpc_rows_satisfied(&decoded, &hdpc_rows) {
        Some(decoded)
    } else {
        None
    }
}

fn hdpc_rows_satisfied(decoded: &SymbolSlab, hdpc_rows: &DenseOctetMatrix) -> bool {
    let mut check = vec![0u8; decoded.symbol_size()];
    for row in 0..hdpc_rows.height() {
        check.fill(0);
        for col in 0..hdpc_rows.width() {
            let coefficient = hdpc_rows.get(row, col);
            if !coefficient.is_zero() {
                fused_addassign_mul_scalar(&mut check, decoded.get(col), &coefficient);
            }
        }
        if !symbol_is_zero(&check) {
            return false;
        }
    }
    true
}

fn copy_binary_row<M: BinaryMatrix>(matrix: &M, row: usize) -> CoefficientRow {
    matrix
        .row_entries(row)
        .into_iter()
        .map(|col| (col, Octet::one()))
        .collect()
}

fn copy_octet_row(matrix: &DenseOctetMatrix, row: usize) -> CoefficientRow {
    let mut result = Vec::new();
    for col in 0..matrix.width() {
        let value = matrix.get(row, col);
        if !value.is_zero() {
            result.push((col, value));
        }
    }
    result
}

fn is_full_systematic_planning_matrix<M: BinaryMatrix>(
    matrix: &M,
    source_block_symbols: u32,
) -> bool {
    let k_prime = extended_source_block_symbols(source_block_symbols);
    let s = num_ldpc_symbols(source_block_symbols);
    if matrix.height() != (s + k_prime) as usize {
        return false;
    }

    let lt_symbols = num_lt_symbols(source_block_symbols);
    let pi_symbols = num_pi_symbols(source_block_symbols);
    let sys_index = systematic_index(source_block_symbols);
    let p1 = calculate_p1(source_block_symbols);
    for isi in 0..k_prime {
        let tuple = intermediate_tuple(isi, lt_symbols, sys_index, p1);
        let mut expected = Vec::new();
        enc_indices(tuple, lt_symbols, pi_symbols, p1, |col| {
            expected.push(col);
        });
        expected.sort_unstable();
        if matrix.row_entries((s + isi) as usize) != expected {
            return false;
        }
    }

    true
}

fn solve(
    mut rows: Vec<CoefficientRow>,
    width: usize,
    mut symbols: SymbolSlab,
    record_ops: bool,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    assert!(
        !record_ops || width <= MAX_RECORDED_SOLVER_WIDTH,
        "operation-recording solver supports at most {MAX_RECORDED_SOLVER_WIDTH} columns"
    );

    let height = rows.len();
    let mut ops = if record_ops { Some(Vec::new()) } else { None };
    let mut pivot_row = 0usize;

    for col in 0..width {
        let Some(pivot) =
            (pivot_row..height).find(|&row| !coefficient_at(&rows[row], col).is_zero())
        else {
            return (None, None);
        };

        if pivot != pivot_row {
            rows.swap(pivot, pivot_row);
            swap_symbol_rows(&mut symbols, pivot, pivot_row);
            if let Some(ops) = ops.as_mut() {
                ops.push(SymbolOps::Swap(pivot, pivot_row));
            }
        }

        let pivot_value = coefficient_at(&rows[pivot_row], col);
        if pivot_value != Octet::one() {
            let scalar = pivot_value.inverse();
            scale_matrix_row(&mut rows[pivot_row], col, scalar);
            mulassign_scalar(symbols.get_mut(pivot_row), &scalar);
            if let Some(ops) = ops.as_mut() {
                ops.push(SymbolOps::Scale(pivot_row, scalar));
            }
        }

        let pivot_coefficients = rows[pivot_row].clone();
        let pivot_symbol = symbols.get(pivot_row).to_vec();
        for row in 0..height {
            if row == pivot_row {
                continue;
            }
            let factor = coefficient_at(&rows[row], col);
            if factor.is_zero() {
                continue;
            }
            add_scaled_matrix_row(&mut rows[row], &pivot_coefficients, col, factor);
            fused_addassign_mul_scalar(symbols.get_mut(row), &pivot_symbol, &factor);
            if let Some(ops) = ops.as_mut() {
                ops.push(SymbolOps::FusedAdd {
                    dest: row,
                    src: pivot_row,
                    scalar: factor,
                });
            }
        }

        pivot_row += 1;
    }

    for row in pivot_row..height {
        if !rows[row].is_empty() || !symbol_is_zero(symbols.get(row)) {
            return (None, None);
        }
    }

    let mut decoded = SymbolSlab::with_zeros(width, symbols.symbol_size());
    for row in 0..width {
        decoded.get_mut(row).copy_from_slice(symbols.get(row));
    }

    (Some(decoded), ops)
}

fn symbol_is_zero(symbol: &[u8]) -> bool {
    symbol.iter().all(|&byte| byte == 0)
}

fn coefficient_at(row: &CoefficientRow, col: usize) -> Octet {
    row.binary_search_by_key(&col, |&(entry_col, _)| entry_col)
        .map(|index| row[index].1)
        .unwrap_or_else(|_| Octet::zero())
}

fn scale_matrix_row(row: &mut CoefficientRow, start_col: usize, scalar: Octet) {
    for (col, value) in row.iter_mut() {
        if *col >= start_col {
            *value *= scalar;
        }
    }
}

fn add_scaled_matrix_row(
    dest: &mut CoefficientRow,
    src: &CoefficientRow,
    start_col: usize,
    scalar: Octet,
) {
    let mut merged = Vec::with_capacity(dest.len() + src.len());
    let mut dest_index = 0usize;
    let mut src_index = src.partition_point(|&(col, _)| col < start_col);

    while dest_index < dest.len() || src_index < src.len() {
        match (dest.get(dest_index), src.get(src_index)) {
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

    *dest = merged;
}

fn swap_symbol_rows(symbols: &mut SymbolSlab, a: usize, b: usize) {
    if a == b {
        return;
    }
    let tmp = symbols.get(a).to_vec();
    let b_bytes = symbols.get(b).to_vec();
    symbols.get_mut(a).copy_from_slice(&b_bytes);
    symbols.get_mut(b).copy_from_slice(&tmp);
}

#[cfg(feature = "benchmarking")]
fn coefficient_storage_bytes<M: BinaryMatrix>(matrix: &M, hdpc_rows: &DenseOctetMatrix) -> usize {
    let binary_entries = (0..matrix.height())
        .map(|row| matrix.row_entries(row).len())
        .sum::<usize>();
    binary_entries * core::mem::size_of::<usize>()
        + hdpc_rows.height() * hdpc_rows.width() * core::mem::size_of::<Octet>()
}

#[cfg(feature = "benchmarking")]
pub struct IntermediateSymbolDecoder<M: BinaryMatrix> {
    matrix: M,
    hdpc_rows: DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
    ops: Vec<SymbolOps>,
}

#[cfg(feature = "benchmarking")]
impl<M: BinaryMatrix> IntermediateSymbolDecoder<M> {
    pub fn new(
        matrix: M,
        hdpc_rows: DenseOctetMatrix,
        symbols: SymbolSlab,
        source_block_symbols: u32,
    ) -> IntermediateSymbolDecoder<M> {
        IntermediateSymbolDecoder {
            matrix,
            hdpc_rows,
            symbols,
            source_block_symbols,
            ops: Vec::new(),
        }
    }

    pub fn execute(&mut self) {
        let (_, ops) = fused_inverse_mul_symbols(
            self.matrix.clone(),
            self.hdpc_rows.clone(),
            self.symbols.clone(),
            self.source_block_symbols,
        );
        self.ops = ops.unwrap_or_default();
    }

    pub fn get_non_symbol_bytes(&self) -> usize {
        coefficient_storage_bytes(&self.matrix, &self.hdpc_rows)
    }

    pub fn get_symbol_mul_ops(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, SymbolOps::Scale(..) | SymbolOps::FusedAdd { .. }))
            .count()
    }

    pub fn get_symbol_add_ops(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, SymbolOps::FusedAdd { .. }))
            .count()
    }

    pub fn get_symbol_mul_ops_by_phase(&self) -> [usize; 5] {
        [self.get_symbol_mul_ops(), 0, 0, 0, 0]
    }

    pub fn get_symbol_add_ops_by_phase(&self) -> [usize; 5] {
        [self.get_symbol_add_ops(), 0, 0, 0, 0]
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::constraint_matrix::generate_constraint_matrix;
    use crate::matrix::{BinaryMatrix, DenseBinaryMatrix};
    use crate::sparse_matrix::SparseBinaryMatrix;
    use crate::systematic_constants::num_intermediate_symbols;

    #[test]
    fn large_non_planning_matrix_uses_non_recording_solver_without_ops() {
        let width = MAX_RECORDED_SOLVER_WIDTH + 1;
        let mut matrix = DenseBinaryMatrix::new(width, width);
        for row in 0..width {
            matrix.set(row, row, true);
        }
        let symbols = SymbolSlab::with_zeros(width, 1);
        let hdpc_rows = DenseOctetMatrix::new(0, width);

        let (decoded, ops) = fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, 1);

        assert!(decoded.is_some());
        assert!(ops.is_none());
    }

    #[test]
    fn small_non_planning_matrix_does_not_record_decode_ops() {
        let width = 16;
        let mut matrix = DenseBinaryMatrix::new(width, width);
        for row in 0..width {
            matrix.set(row, row, true);
        }
        let symbols = SymbolSlab::with_zeros(width, 1);
        let hdpc_rows = DenseOctetMatrix::new(0, width);

        let (decoded, ops) = fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, 1);

        assert!(decoded.is_some());
        assert!(ops.is_none());
    }

    #[test]
    fn large_systematic_plan_returns_non_recording_solve_op_without_factorizing() {
        let source_symbols = 5_000;
        let k_prime = extended_source_block_symbols(source_symbols);
        let symbols = SymbolSlab::with_zeros(num_intermediate_symbols(source_symbols) as usize, 1);
        let indices: Vec<u32> = (0..k_prime).collect();
        let (matrix, hdpc_rows) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &indices);

        let (decoded, ops) = fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_symbols);

        assert!(decoded.is_none());
        assert!(matches!(
            ops.as_deref(),
            Some([SymbolOps::Solve {
                source_block_symbols: 5_000
            }])
        ));
    }

    #[test]
    fn large_plan_detection_does_not_match_repair_decode_matrix() {
        let source_symbols = 5_000;
        let k_prime = extended_source_block_symbols(source_symbols);

        let planning_indices: Vec<u32> = (0..k_prime).collect();
        let (planning_matrix, _) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &planning_indices);
        assert!(is_full_systematic_planning_matrix(
            &planning_matrix,
            source_symbols
        ));

        let decode_indices: Vec<u32> = (1..k_prime).chain(core::iter::once(k_prime)).collect();
        let (decode_matrix, _) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &decode_indices);
        assert!(!is_full_systematic_planning_matrix(
            &decode_matrix,
            source_symbols
        ));
    }

    #[test]
    fn overdetermined_consistent_no_hdpc_system_decodes() {
        let mut matrix = DenseBinaryMatrix::new(2, 1);
        matrix.set(0, 0, true);
        matrix.set(1, 0, true);
        let symbols = SymbolSlab::from_bytes(vec![0x5a, 0x5a], 1);

        let (decoded, ops) = fused_inverse_mul_symbols_no_hdpc(matrix, symbols, 1);

        assert_eq!(decoded.unwrap().get(0), &[0x5a]);
        assert!(ops.is_none());
    }

    #[test]
    fn overdetermined_inconsistent_no_hdpc_system_fails() {
        let mut matrix = DenseBinaryMatrix::new(2, 1);
        matrix.set(0, 0, true);
        matrix.set(1, 0, true);
        let symbols = SymbolSlab::from_bytes(vec![0x5a, 0xa5], 1);

        let (decoded, ops) = fused_inverse_mul_symbols_no_hdpc(matrix, symbols, 1);

        assert!(decoded.is_none());
        assert!(ops.is_none());
    }
}

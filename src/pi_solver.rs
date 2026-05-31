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
use crate::sparse_vec::SparseOctetVec;
use crate::symbol_slab::SymbolSlab;
use crate::systematic_constants::num_ldpc_symbols;
use crate::systematic_constants::{
    MAX_SUPPORTED_INTERMEDIATE_SYMBOLS, calculate_p1, extended_source_block_symbols,
    num_intermediate_symbols, num_lt_symbols, num_pi_symbols, systematic_index,
};

type CoefficientRow = SparseOctetVec;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SolveStats {
    symbol_mul_ops_by_phase: [usize; 5],
    symbol_add_ops_by_phase: [usize; 5],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SolverPhase {
    First,
    Fifth,
}

impl SolverPhase {
    fn index(self) -> usize {
        match self {
            SolverPhase::First => 0,
            SolverPhase::Fifth => 4,
        }
    }
}

impl SolveStats {
    fn record_symbol_op(&mut self, phase: SolverPhase, op: &SymbolOps) {
        let phase = phase.index();
        match op {
            SymbolOps::Swap(..) => {}
            SymbolOps::Scale(..) => {
                self.symbol_mul_ops_by_phase[phase] += 1;
            }
            SymbolOps::FusedAdd { .. } => {
                self.symbol_mul_ops_by_phase[phase] += 1;
                self.symbol_add_ops_by_phase[phase] += 1;
            }
        }
    }

    #[cfg(feature = "benchmarking")]
    fn total_symbol_mul_ops(self) -> usize {
        self.symbol_mul_ops_by_phase.iter().sum()
    }

    #[cfg(feature = "benchmarking")]
    fn total_symbol_add_ops(self) -> usize {
        self.symbol_add_ops_by_phase.iter().sum()
    }
}

// Systematic planning matrices feed SourceBlockEncodingPlan, which replays concrete row
// operations. Recording must therefore stay independent of matrix width and feature set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OperationRecording {
    Record,
    Skip,
}

impl OperationRecording {
    fn for_matrix<M: BinaryMatrix>(matrix: &M, source_block_symbols: u32) -> OperationRecording {
        if is_full_systematic_planning_matrix(matrix, source_block_symbols) {
            OperationRecording::Record
        } else {
            OperationRecording::Skip
        }
    }

    fn new_ops(self) -> Option<Vec<SymbolOps>> {
        match self {
            OperationRecording::Record => Some(Vec::new()),
            OperationRecording::Skip => None,
        }
    }
}

pub fn fused_inverse_mul_symbols<M: BinaryMatrix>(
    matrix: M,
    hdpc_rows: DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let recording = OperationRecording::for_matrix(&matrix, source_block_symbols);
    fused_inverse_mul_symbols_impl(matrix, hdpc_rows, symbols, source_block_symbols, recording)
}

#[cfg(feature = "benchmarking")]
fn fused_inverse_mul_symbols_traced<M: BinaryMatrix>(
    matrix: M,
    hdpc_rows: DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>, SolveStats) {
    let recording = OperationRecording::for_matrix(&matrix, source_block_symbols);
    fused_inverse_mul_symbols_impl_traced(
        matrix,
        hdpc_rows,
        symbols,
        source_block_symbols,
        recording,
    )
}

fn fused_inverse_mul_symbols_impl<M: BinaryMatrix>(
    matrix: M,
    hdpc_rows: DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
    recording: OperationRecording,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let (decoded, ops, _) = fused_inverse_mul_symbols_impl_traced(
        matrix,
        hdpc_rows,
        symbols,
        source_block_symbols,
        recording,
    );
    (decoded, ops)
}

fn fused_inverse_mul_symbols_impl_traced<M: BinaryMatrix>(
    matrix: M,
    hdpc_rows: DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
    recording: OperationRecording,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>, SolveStats) {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let width = matrix.width();
    let total_rows = matrix.height() + hdpc_rows.height();
    assert_eq!(symbols.len(), total_rows);
    assert_eq!(hdpc_rows.width(), width);
    assert!(matrix.height() >= s);

    let rows = coefficient_rows(&matrix, &hdpc_rows, source_block_symbols);
    solve_traced(rows, width, symbols, recording)
}

fn coefficient_rows<M: BinaryMatrix>(
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    source_block_symbols: u32,
) -> Vec<CoefficientRow> {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let total_rows = matrix.height() + hdpc_rows.height();
    let mut rows = vec![SparseOctetVec::new(); total_rows];
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

    let (decoded, ops) = solve(rows, width, symbols, OperationRecording::Skip);
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
    SparseOctetVec::from_binary_entries(matrix.row_entries(row))
}

fn copy_octet_row(matrix: &DenseOctetMatrix, row: usize) -> CoefficientRow {
    SparseOctetVec::from_octet_entries((0..matrix.width()).map(|col| (col, matrix.get(row, col))))
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
    rows: Vec<CoefficientRow>,
    width: usize,
    symbols: SymbolSlab,
    recording: OperationRecording,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let (decoded, ops, _) = solve_traced(rows, width, symbols, recording);
    (decoded, ops)
}

fn solve_traced(
    mut rows: Vec<CoefficientRow>,
    width: usize,
    mut symbols: SymbolSlab,
    recording: OperationRecording,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>, SolveStats) {
    assert!(
        width <= MAX_SUPPORTED_INTERMEDIATE_SYMBOLS as usize,
        "generic RaptorQ solver supports at most {MAX_SUPPORTED_INTERMEDIATE_SYMBOLS} intermediate symbols; optimized large-matrix PI solver is not implemented"
    );

    let height = rows.len();
    let mut ops = recording.new_ops();
    let mut stats = SolveStats::default();
    let mut pivot_row = 0usize;

    for col in 0..width {
        let Some(pivot) =
            (pivot_row..height).find(|&row| !coefficient_at(&rows[row], col).is_zero())
        else {
            return (None, None, stats);
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
            stats.record_symbol_op(SolverPhase::First, &SymbolOps::Scale(pivot_row, scalar));
        }

        let pivot_coefficients = rows[pivot_row].clone();
        let pivot_symbol = symbols.get(pivot_row).to_vec();
        for row in (pivot_row + 1)..height {
            let factor = coefficient_at(&rows[row], col);
            if factor.is_zero() {
                continue;
            }
            add_scaled_matrix_row(&mut rows[row], &pivot_coefficients, col, factor);
            fused_addassign_mul_scalar(symbols.get_mut(row), &pivot_symbol, &factor);
            record_fused_add(
                ops.as_mut(),
                &mut stats,
                SolverPhase::First,
                row,
                pivot_row,
                factor,
            );
        }

        pivot_row += 1;
    }

    for row in pivot_row..height {
        if !rows[row].is_empty() || !symbol_is_zero(symbols.get(row)) {
            return (None, None, stats);
        }
    }

    for col in (0..width).rev() {
        let pivot_coefficients = rows[col].clone();
        let pivot_symbol = symbols.get(col).to_vec();
        for row in 0..col {
            let factor = coefficient_at(&rows[row], col);
            if factor.is_zero() {
                continue;
            }
            add_scaled_matrix_row(&mut rows[row], &pivot_coefficients, col, factor);
            fused_addassign_mul_scalar(symbols.get_mut(row), &pivot_symbol, &factor);
            record_fused_add(
                ops.as_mut(),
                &mut stats,
                SolverPhase::Fifth,
                row,
                col,
                factor,
            );
        }
    }

    let mut decoded = SymbolSlab::with_zeros(width, symbols.symbol_size());
    for row in 0..width {
        decoded.get_mut(row).copy_from_slice(symbols.get(row));
    }

    (Some(decoded), ops, stats)
}

fn symbol_is_zero(symbol: &[u8]) -> bool {
    symbol.iter().all(|&byte| byte == 0)
}

fn record_fused_add(
    ops: Option<&mut Vec<SymbolOps>>,
    stats: &mut SolveStats,
    phase: SolverPhase,
    dest: usize,
    src: usize,
    scalar: Octet,
) {
    let op = SymbolOps::FusedAdd { dest, src, scalar };
    stats.record_symbol_op(phase, &op);
    if let Some(ops) = ops {
        ops.push(op);
    }
}

fn coefficient_at(row: &CoefficientRow, col: usize) -> Octet {
    row.get(col)
}

fn scale_matrix_row(row: &mut CoefficientRow, start_col: usize, scalar: Octet) {
    row.scale_from(start_col, scalar);
}

fn add_scaled_matrix_row(
    dest: &mut CoefficientRow,
    src: &CoefficientRow,
    start_col: usize,
    scalar: Octet,
) {
    dest.add_scaled_from(src, start_col, scalar);
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
    stats: SolveStats,
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
            stats: SolveStats::default(),
        }
    }

    pub fn execute(&mut self) {
        let (_, _, stats) = fused_inverse_mul_symbols_traced(
            self.matrix.clone(),
            self.hdpc_rows.clone(),
            self.symbols.clone(),
            self.source_block_symbols,
        );
        self.stats = stats;
    }

    pub fn get_non_symbol_bytes(&self) -> usize {
        coefficient_storage_bytes(&self.matrix, &self.hdpc_rows)
    }

    pub fn get_symbol_mul_ops(&self) -> usize {
        self.stats.total_symbol_mul_ops()
    }

    pub fn get_symbol_add_ops(&self) -> usize {
        self.stats.total_symbol_add_ops()
    }

    pub fn get_symbol_mul_ops_by_phase(&self) -> [usize; 5] {
        self.stats.symbol_mul_ops_by_phase
    }

    pub fn get_symbol_add_ops_by_phase(&self) -> [usize; 5] {
        self.stats.symbol_add_ops_by_phase
    }
}

#[cfg(test)]
mod recording_tests {
    use super::*;

    #[test]
    fn operation_recording_solver_records_for_supported_width() {
        let width = 64;
        let rows: Vec<CoefficientRow> = (0..width)
            .map(|col| SparseOctetVec::from_octet_entries([(col, Octet::one())]))
            .collect();
        let symbols = SymbolSlab::with_zeros(width, 1);

        let (decoded, ops) = solve(rows, width, symbols, OperationRecording::Record);

        assert!(decoded.is_some());
        assert!(ops.is_some());
    }

    #[test]
    fn traced_solver_counts_forward_and_backward_symbol_ops_by_phase() {
        let rows = vec![
            SparseOctetVec::from_binary_entries([0, 1]),
            SparseOctetVec::from_binary_entries([0]),
        ];
        let symbols = SymbolSlab::with_zeros(2, 1);

        let (decoded, ops, stats) = solve_traced(rows, 2, symbols, OperationRecording::Skip);

        assert!(decoded.is_some());
        assert!(ops.is_none());
        assert_eq!(stats.symbol_mul_ops_by_phase, [1, 0, 0, 0, 1]);
        assert_eq!(stats.symbol_add_ops_by_phase, [1, 0, 0, 0, 1]);
    }

    #[test]
    #[should_panic(expected = "generic RaptorQ solver supports at most 1120 intermediate symbols")]
    fn generic_solver_rejects_width_above_supported_limit() {
        let width = MAX_SUPPORTED_INTERMEDIATE_SYMBOLS as usize + 1;
        let rows: Vec<CoefficientRow> = (0..width)
            .map(|col| SparseOctetVec::from_octet_entries([(col, Octet::one())]))
            .collect();
        let symbols = SymbolSlab::with_zeros(width, 1);

        let _ = solve(rows, width, symbols, OperationRecording::Record);
    }

    #[test]
    #[should_panic(expected = "generic RaptorQ solver supports at most 1120 intermediate symbols")]
    fn oversized_source_block_encoding_plan_rejects_at_solver_limit() {
        crate::SourceBlockEncodingPlan::generate(1_051);
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::constraint_matrix::generate_constraint_matrix;
    use crate::matrix::{BinaryMatrix, DenseBinaryMatrix};
    use crate::sparse_matrix::SparseBinaryMatrix;

    #[test]
    fn large_non_planning_matrix_uses_non_recording_solver_without_ops() {
        let width = MAX_SUPPORTED_INTERMEDIATE_SYMBOLS as usize;
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

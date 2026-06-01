#[cfg(feature = "std")]
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "std")]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::base::intermediate_tuple;
use crate::constraint_matrix::generate_constraint_matrix;
use crate::constraint_matrix::{enc_indices, generate_hdpc_rows};
use crate::gf2::PackedBinaryRows;
use crate::matrix::BinaryMatrix;
use crate::octet::Octet;
use crate::octet_matrix::DenseOctetMatrix;
use crate::octets::{add_assign, fused_addassign_mul_scalar, mulassign_scalar};
use crate::operation_vector::SymbolOps;
use crate::sparse_matrix::SparseBinaryMatrix;
use crate::symbol_slab::SymbolSlab;
use crate::systematic_constants::num_ldpc_symbols;
use crate::systematic_constants::{
    MAX_SUPPORTED_INTERMEDIATE_SYMBOLS, calculate_p1, extended_source_block_symbols,
    num_intermediate_symbols, num_lt_symbols, num_pi_symbols, systematic_index,
};

type CoefficientRow = Vec<(usize, Octet)>;
pub(crate) const MAX_INLINE_RECORDED_SOLVER_WIDTH: usize = 4096;
#[cfg(feature = "std")]
const SYSTEMATIC_PLAN_CACHE_CAPACITY: usize = 16;

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
    if recording == OperationRecording::Record && matrix.width() > MAX_INLINE_RECORDED_SOLVER_WIDTH
    {
        #[cfg(feature = "std")]
        {
            let source_block_symbols = extended_source_block_symbols(source_block_symbols);
            cached_systematic_plan(source_block_symbols);
            return (
                None,
                Some(vec![SymbolOps::ApplyCachedSystematicPlan {
                    source_block_symbols,
                }]),
            );
        }

        #[cfg(not(feature = "std"))]
        {
            // no_std cannot keep the global large-plan cache, so replay performs the solve.
            let source_block_symbols = extended_source_block_symbols(source_block_symbols);
            let op = SymbolOps::DirectSystematicSolve {
                source_block_symbols,
            };
            if symbol_is_zero(symbols.as_bytes()) {
                let decoded = SymbolSlab::with_zeros(matrix.width(), symbols.symbol_size());
                return (Some(decoded), Some(vec![op]));
            }

            let (decoded, _) = fused_inverse_mul_symbols_impl(
                matrix,
                hdpc_rows,
                symbols,
                source_block_symbols,
                OperationRecording::Skip,
            );
            return (decoded, Some(vec![op]));
        }
    }

    fused_inverse_mul_symbols_impl(matrix, hdpc_rows, symbols, source_block_symbols, recording)
}

fn fused_inverse_mul_symbols_impl<M: BinaryMatrix>(
    matrix: M,
    hdpc_rows: DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
    recording: OperationRecording,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let width = matrix.width();
    let total_rows = matrix.height() + hdpc_rows.height();
    assert_eq!(symbols.len(), total_rows);
    assert_eq!(hdpc_rows.width(), width);
    assert!(matrix.height() >= s);

    let rows = coefficient_rows(&matrix, &hdpc_rows, source_block_symbols);
    solve(rows, width, symbols, recording)
}

#[cfg(feature = "std")]
struct CachedSystematicPlan {
    // Large systematic plans keep matrix state out of replay without storing huge row-op vectors.
    rows: Vec<CoefficientRow>,
    width: usize,
}

#[cfg(feature = "std")]
#[derive(Default)]
struct SystematicPlanCache {
    plans: HashMap<u32, Arc<CachedSystematicPlan>>,
    insertion_order: VecDeque<u32>,
}

#[cfg(feature = "std")]
type SystematicPlanCacheLock = Mutex<SystematicPlanCache>;

#[cfg(feature = "std")]
fn systematic_plan_cache() -> &'static SystematicPlanCacheLock {
    static CACHE: OnceLock<SystematicPlanCacheLock> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(SystematicPlanCache::default()))
}

#[cfg(feature = "std")]
fn cached_systematic_plan(source_block_symbols: u32) -> Arc<CachedSystematicPlan> {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    {
        let cache = systematic_plan_cache();
        let guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(plan) = guard.plans.get(&source_block_symbols) {
            return Arc::clone(plan);
        }
    }

    let generated = Arc::new(generate_systematic_plan(source_block_symbols));
    let cache = systematic_plan_cache();
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    insert_systematic_plan(&mut guard, source_block_symbols, generated)
}

#[cfg(feature = "std")]
fn insert_systematic_plan(
    cache: &mut SystematicPlanCache,
    source_block_symbols: u32,
    generated: Arc<CachedSystematicPlan>,
) -> Arc<CachedSystematicPlan> {
    if let Some(plan) = cache.plans.get(&source_block_symbols) {
        return Arc::clone(plan);
    }

    if cache.plans.len() >= SYSTEMATIC_PLAN_CACHE_CAPACITY
        && let Some(evicted_source_block_symbols) = cache.insertion_order.pop_front()
    {
        cache.plans.remove(&evicted_source_block_symbols);
    }

    cache.insertion_order.push_back(source_block_symbols);
    cache
        .plans
        .insert(source_block_symbols, Arc::clone(&generated));
    generated
}

#[cfg(feature = "std")]
fn generate_systematic_plan(source_block_symbols: u32) -> CachedSystematicPlan {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    let indices: Vec<u32> = (0..source_block_symbols).collect();
    let (matrix, hdpc_rows) =
        generate_constraint_matrix::<SparseBinaryMatrix>(source_block_symbols, &indices);
    let width = matrix.width();
    let rows = coefficient_rows(&matrix, &hdpc_rows, source_block_symbols);
    CachedSystematicPlan { rows, width }
}

#[cfg(feature = "std")]
pub(crate) fn apply_cached_systematic_plan(source_block_symbols: u32, symbols: &mut SymbolSlab) {
    let plan = cached_systematic_plan(source_block_symbols);
    let (decoded, _) = solve(
        plan.rows.clone(),
        plan.width,
        symbols.clone(),
        OperationRecording::Skip,
    );
    let decoded = decoded.expect("cached systematic solve failed");
    for row in 0..decoded.len() {
        symbols.get_mut(row).copy_from_slice(decoded.get(row));
    }
}

#[cfg(not(feature = "std"))]
pub(crate) fn apply_direct_systematic_solve(source_block_symbols: u32, symbols: &mut SymbolSlab) {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    let indices: Vec<u32> = (0..source_block_symbols).collect();
    let (matrix, hdpc_rows) =
        generate_constraint_matrix::<SparseBinaryMatrix>(source_block_symbols, &indices);
    let (decoded, _) = fused_inverse_mul_symbols_impl(
        matrix,
        hdpc_rows,
        symbols.clone(),
        source_block_symbols,
        OperationRecording::Skip,
    );
    let decoded = decoded.expect("direct systematic solve failed");
    for row in 0..decoded.len() {
        symbols.get_mut(row).copy_from_slice(decoded.get(row));
    }
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
        rows.push(matrix.row_entries(row));
    }

    let (decoded, ops) = solve_binary(rows, width, symbols);
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
    let mut result = Vec::new();
    matrix.visit_row_entries(row, |col| result.push((col, Octet::one())));
    result
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
    recording: OperationRecording,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let height = rows.len();
    let mut ops = recording.new_ops();
    let mut pivot_row = 0usize;
    let mut row_merge_scratch = Vec::new();

    for col in 0..width {
        let Some(pivot) =
            (pivot_row..height).find(|&row| !coefficient_at(&rows[row], col).is_zero())
        else {
            return (None, None);
        };

        if pivot != pivot_row {
            rows.swap(pivot, pivot_row);
            symbols.swap_symbols(pivot, pivot_row);
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

        let (rows_before, pivot_and_after) = rows.split_at_mut(pivot_row);
        let (pivot_coefficients, rows_after) = pivot_and_after
            .split_first_mut()
            .expect("pivot row must exist");

        for (row, row_coefficients) in rows_before.iter_mut().enumerate() {
            let factor = coefficient_at(row_coefficients, col);
            if factor.is_zero() {
                continue;
            }
            add_scaled_matrix_row(
                row_coefficients,
                pivot_coefficients,
                col,
                factor,
                &mut row_merge_scratch,
            );
            let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(pivot_row, row);
            fused_addassign_mul_scalar(dest_symbol, pivot_symbol, &factor);
            if let Some(ops) = ops.as_mut() {
                ops.push(SymbolOps::FusedAdd {
                    dest: row,
                    src: pivot_row,
                    scalar: factor,
                });
            }
        }

        for (offset, row_coefficients) in rows_after.iter_mut().enumerate() {
            let row = pivot_row + 1 + offset;
            let factor = coefficient_at(row_coefficients, col);
            if factor.is_zero() {
                continue;
            }
            add_scaled_matrix_row(
                row_coefficients,
                pivot_coefficients,
                col,
                factor,
                &mut row_merge_scratch,
            );
            let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(pivot_row, row);
            fused_addassign_mul_scalar(dest_symbol, pivot_symbol, &factor);
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

fn solve_binary(
    rows: Vec<Vec<usize>>,
    width: usize,
    mut symbols: SymbolSlab,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    assert!(
        width <= MAX_SUPPORTED_INTERMEDIATE_SYMBOLS as usize,
        "generic RaptorQ solver supports at most {MAX_SUPPORTED_INTERMEDIATE_SYMBOLS} intermediate symbols; optimized large-matrix PI solver is not implemented"
    );

    let height = rows.len();
    let mut rows = PackedBinaryRows::from_sparse(rows, width);
    let mut bucket_heads = vec![None; width];
    let mut next_in_bucket = vec![None; height];
    for row in 0..height {
        if let Some(col) = rows.first_one_at_or_after(row, 0) {
            push_binary_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
        }
    }

    let mut pivot_for_col = vec![None; width];
    let mut is_pivot_row = vec![false; height];

    for col in 0..width {
        let Some(pivot) = pop_binary_row_bucket(&mut bucket_heads, &mut next_in_bucket, col) else {
            return (None, None);
        };
        pivot_for_col[col] = Some(pivot);
        is_pivot_row[pivot] = true;

        while let Some(row) = pop_binary_row_bucket(&mut bucket_heads, &mut next_in_bucket, col) {
            rows.xor_suffix_if_contains(row, pivot, col);
            let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(pivot, row);
            add_assign(dest_symbol, pivot_symbol);

            if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                push_binary_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
            }
        }
    }

    for (row, is_pivot) in is_pivot_row.into_iter().enumerate() {
        if !is_pivot && (!rows.is_zero(row) || !symbol_is_zero(symbols.get(row))) {
            return (None, None);
        }
    }

    let mut decoded = SymbolSlab::with_zeros(width, symbols.symbol_size());
    for col in (0..width).rev() {
        let pivot = pivot_for_col[col].expect("pivot was recorded for every decoded column");
        decoded.get_mut(col).copy_from_slice(symbols.get(pivot));
        rows.visit_ones_at_or_after(pivot, col + 1, |dependent_col| {
            let (dependent_symbol, dest_symbol) = decoded.get_disjoint_mut(dependent_col, col);
            add_assign(dest_symbol, dependent_symbol);
        });
    }

    (Some(decoded), None)
}

fn push_binary_row_bucket(
    bucket_heads: &mut [Option<usize>],
    next_in_bucket: &mut [Option<usize>],
    col: usize,
    row: usize,
) {
    debug_assert!(next_in_bucket[row].is_none());
    next_in_bucket[row] = bucket_heads[col];
    bucket_heads[col] = Some(row);
}

fn pop_binary_row_bucket(
    bucket_heads: &mut [Option<usize>],
    next_in_bucket: &mut [Option<usize>],
    col: usize,
) -> Option<usize> {
    let row = bucket_heads[col]?;
    bucket_heads[col] = next_in_bucket[row];
    next_in_bucket[row] = None;
    Some(row)
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
    scratch: &mut CoefficientRow,
) {
    if scalar == Octet::one() {
        add_unscaled_matrix_row(dest, src, start_col, scratch);
        return;
    }

    let mut dest_index = 0usize;
    let mut src_index = src.partition_point(|&(col, _)| col < start_col);
    scratch.clear();
    scratch.reserve(dest.len() + src.len() - src_index);

    while dest_index < dest.len() || src_index < src.len() {
        match (dest.get(dest_index), src.get(src_index)) {
            (Some(&(dest_col, dest_value)), Some(&(src_col, src_value))) => {
                if dest_col < src_col {
                    scratch.push((dest_col, dest_value));
                    dest_index += 1;
                } else if src_col < dest_col {
                    scratch.push((src_col, src_value * scalar));
                    src_index += 1;
                } else {
                    let value = dest_value + src_value * scalar;
                    if !value.is_zero() {
                        scratch.push((dest_col, value));
                    }
                    dest_index += 1;
                    src_index += 1;
                }
            }
            (Some(&(dest_col, dest_value)), None) => {
                scratch.push((dest_col, dest_value));
                dest_index += 1;
            }
            (None, Some(&(src_col, src_value))) => {
                scratch.push((src_col, src_value * scalar));
                src_index += 1;
            }
            (None, None) => break,
        }
    }

    core::mem::swap(dest, scratch);
}

fn add_unscaled_matrix_row(
    dest: &mut CoefficientRow,
    src: &CoefficientRow,
    start_col: usize,
    scratch: &mut CoefficientRow,
) {
    let mut dest_index = 0usize;
    let mut src_index = src.partition_point(|&(col, _)| col < start_col);
    scratch.clear();
    scratch.reserve(dest.len() + src.len() - src_index);

    while dest_index < dest.len() || src_index < src.len() {
        match (dest.get(dest_index), src.get(src_index)) {
            (Some(&(dest_col, dest_value)), Some(&(src_col, src_value))) => {
                if dest_col < src_col {
                    scratch.push((dest_col, dest_value));
                    dest_index += 1;
                } else if src_col < dest_col {
                    scratch.push((src_col, src_value));
                    src_index += 1;
                } else {
                    let value = dest_value + src_value;
                    if !value.is_zero() {
                        scratch.push((dest_col, value));
                    }
                    dest_index += 1;
                    src_index += 1;
                }
            }
            (Some(&(dest_col, dest_value)), None) => {
                scratch.push((dest_col, dest_value));
                dest_index += 1;
            }
            (None, Some(&(src_col, src_value))) => {
                scratch.push((src_col, src_value));
                src_index += 1;
            }
            (None, None) => break,
        }
    }

    core::mem::swap(dest, scratch);
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

#[cfg(test)]
mod recording_tests {
    use super::*;

    #[test]
    fn operation_recording_solver_records_for_supported_width() {
        let width = 64;
        let rows: Vec<CoefficientRow> = (0..width).map(|col| vec![(col, Octet::one())]).collect();
        let symbols = SymbolSlab::with_zeros(width, 1);

        let (decoded, ops) = solve(rows, width, symbols, OperationRecording::Record);

        assert!(decoded.is_some());
        assert!(ops.is_some());
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
        let width = 1024;
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
    fn large_systematic_plan_uses_cached_systematic_plan() {
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
            Some([SymbolOps::ApplyCachedSystematicPlan {
                source_block_symbols
            }]) if *source_block_symbols == k_prime
        ));
    }

    #[test]
    fn systematic_plan_cache_insert_evicts_old_entries() {
        let mut cache = SystematicPlanCache::default();

        for source_block_symbols in 0..=SYSTEMATIC_PLAN_CACHE_CAPACITY as u32 {
            let plan = std::sync::Arc::new(CachedSystematicPlan {
                rows: Vec::new(),
                width: 0,
            });
            insert_systematic_plan(&mut cache, source_block_symbols, plan);
        }

        assert_eq!(cache.plans.len(), SYSTEMATIC_PLAN_CACHE_CAPACITY);
        assert!(!cache.plans.contains_key(&0));
        assert!(
            cache
                .plans
                .contains_key(&(SYSTEMATIC_PLAN_CACHE_CAPACITY as u32))
        );
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

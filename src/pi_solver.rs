#[cfg(feature = "std")]
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "std")]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::base::intermediate_tuple;
use crate::constraint_matrix::enc_indices;
use crate::constraint_matrix::generate_constraint_matrix;
use crate::constraint_matrix::generate_hdpc_rows;
use crate::gf2::PackedBinaryRows;
use crate::matrix::BinaryMatrix;
use crate::octet::Octet;
use crate::octet_matrix::DenseOctetMatrix;
use crate::octets::{add_assign, fused_addassign_mul_scalar, mulassign_scalar};
use crate::operation_vector::SymbolOps;
#[cfg(feature = "std")]
use crate::operation_vector::fused_addassign_symbol_batch;
use crate::sparse_matrix::SparseBinaryMatrix;
use crate::symbol_slab::SymbolSlab;
use crate::systematic_constants::num_ldpc_symbols;
use crate::systematic_constants::{
    MAX_SUPPORTED_INTERMEDIATE_SYMBOLS, calculate_p1, extended_source_block_symbols,
    num_intermediate_symbols, num_lt_symbols, num_pi_symbols, systematic_index,
};

type CoefficientRow = Vec<(usize, Octet)>;
pub(crate) const MAX_INLINE_RECORDED_SOLVER_WIDTH: usize = 4096;
const LIGHTEST_PIVOT_MIN_WIDTH: usize = 64;
const COEFFICIENT_BUCKET_SOLVER_MIN_WIDTH: usize = 512;
const TRIANGULAR_RECORDING_MIN_WIDTH: usize = MAX_SUPPORTED_INTERMEDIATE_SYMBOLS as usize + 1;
const SQUARE_HYBRID_MAX_WIDTH: usize = 16_384;
const OVERDETERMINED_HYBRID_MAX_WIDTH: usize = 16_384;
#[cfg(not(test))]
const SINGLE_REPAIR_SYSTEMATIC_MIN_WIDTH: usize = 4096;
#[cfg(test)]
const SINGLE_REPAIR_SYSTEMATIC_MIN_WIDTH: usize = 1;
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

pub(crate) fn fused_inverse_mul_symbols<M: BinaryMatrix>(
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

    #[cfg(feature = "std")]
    if recording == OperationRecording::Skip
        && let Some(decoded) =
            try_single_repair_systematic_decode(&matrix, &hdpc_rows, &symbols, source_block_symbols)
    {
        return (Some(decoded), None);
    }

    let square_hybrid_candidate = total_rows == width && width <= SQUARE_HYBRID_MAX_WIDTH;
    let overdetermined_hybrid_candidate =
        total_rows > width && width <= OVERDETERMINED_HYBRID_MAX_WIDTH;
    if recording == OperationRecording::Skip
        && (square_hybrid_candidate || overdetermined_hybrid_candidate)
        && let Some(decoded) =
            try_hybrid_binary_hdpc_solve(&matrix, &hdpc_rows, &symbols, source_block_symbols)
    {
        return (Some(decoded), None);
    }

    let (rows, symbols) = match recording {
        OperationRecording::Record => (
            coefficient_rows(&matrix, &hdpc_rows, source_block_symbols),
            symbols,
        ),
        OperationRecording::Skip if width >= LIGHTEST_PIVOT_MIN_WIDTH => {
            coefficient_rows_with_hdpc_last(&matrix, &hdpc_rows, symbols, source_block_symbols)
        }
        OperationRecording::Skip => (
            coefficient_rows(&matrix, &hdpc_rows, source_block_symbols),
            symbols,
        ),
    };
    solve(rows, width, symbols, recording)
}

#[cfg(feature = "std")]
struct CachedSystematicPlan {
    // Large systematic plans replay only symbol operations; coefficient elimination is cached.
    forward_steps: Vec<CachedSystematicForwardStep>,
    pivot_symbol_cycles: Vec<Box<[usize]>>,
    back_substitution_rows: Vec<Box<[(usize, Octet)]>>,
    width: usize,
}

#[cfg(feature = "std")]
struct CachedSystematicForwardStep {
    pivot: usize,
    scale: Option<Octet>,
    dests: Box<[(usize, Octet)]>,
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
    prepare_cached_systematic_plan(rows, width)
}

#[cfg(feature = "std")]
pub(crate) fn apply_cached_systematic_plan(source_block_symbols: u32, symbols: &mut SymbolSlab) {
    let plan = cached_systematic_plan(source_block_symbols);
    apply_prepared_systematic_plan(&plan, symbols);
}

#[cfg(feature = "std")]
fn prepare_cached_systematic_plan(
    mut rows: Vec<CoefficientRow>,
    width: usize,
) -> CachedSystematicPlan {
    let height = rows.len();
    assert_eq!(height, width);

    let mut row_merge_scratch = Vec::new();
    let mut bucket_heads = vec![None; width];
    let mut next_in_bucket = vec![None; height];
    for (row, coefficients) in rows.iter().enumerate() {
        if let Some(&(col, _)) = coefficients.first() {
            push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
        }
    }

    let mut forward_steps = Vec::with_capacity(width);
    let mut pivot_for_col = vec![usize::MAX; width];
    let mut is_pivot_row = vec![false; height];

    for (col, pivot_slot) in pivot_for_col.iter_mut().enumerate() {
        let (pivot, pivot_value) =
            pop_lightest_coefficient_row_bucket(&rows, &mut bucket_heads, &mut next_in_bucket, col)
                .expect("systematic plan matrix must be full rank");
        *pivot_slot = pivot;
        is_pivot_row[pivot] = true;

        let scale = if pivot_value != Octet::one() {
            let scalar = pivot_value.inverse();
            scale_matrix_row(&mut rows[pivot], col, scalar);
            Some(scalar)
        } else {
            None
        };

        let pivot_coefficients = rows[pivot].clone();
        let mut dests = Vec::new();
        while let Some(row) = pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col) {
            debug_assert_ne!(row, pivot);
            let factor = rows[row][0].1;
            add_scaled_matrix_row(
                &mut rows[row],
                &pivot_coefficients,
                col,
                factor,
                &mut row_merge_scratch,
            );
            dests.push((row, factor));

            if let Some(&(next_col, _)) = rows[row].first() {
                debug_assert!(next_col > col);
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
            }
        }

        forward_steps.push(CachedSystematicForwardStep {
            pivot,
            scale,
            dests: dests.into_boxed_slice(),
        });
    }

    for (row, is_pivot) in is_pivot_row.into_iter().enumerate() {
        assert!(
            is_pivot || rows[row].is_empty(),
            "systematic plan matrix has a non-pivot residual row"
        );
    }

    let mut back_substitution_rows = Vec::with_capacity(width);
    for (col, &pivot) in pivot_for_col.iter().enumerate() {
        let mut dependencies = Vec::new();
        for &(dependent_col, coefficient) in rows[pivot].iter().rev() {
            if dependent_col <= col {
                break;
            }
            dependencies.push((dependent_col, coefficient));
        }
        back_substitution_rows.push(dependencies.into_boxed_slice());
    }

    CachedSystematicPlan {
        forward_steps,
        pivot_symbol_cycles: pivot_symbol_cycles(&pivot_for_col),
        back_substitution_rows,
        width,
    }
}

#[cfg(feature = "std")]
fn apply_prepared_systematic_plan(plan: &CachedSystematicPlan, symbols: &mut SymbolSlab) {
    assert_eq!(symbols.len(), plan.width);

    for step in &plan.forward_steps {
        if let Some(scale) = step.scale {
            mulassign_scalar(symbols.get_mut(step.pivot), &scale);
        }
        fused_addassign_symbol_batch(symbols, step.pivot, &step.dests);
    }

    move_pivot_symbols_to_columns(symbols, &plan.pivot_symbol_cycles);
    for col in (0..plan.width).rev() {
        for &(dependent_col, coefficient) in &plan.back_substitution_rows[col] {
            let (dependent_symbol, dest_symbol) = symbols.get_disjoint_mut(dependent_col, col);
            fused_addassign_mul_scalar(dest_symbol, dependent_symbol, &coefficient);
        }
    }
}

#[cfg(feature = "std")]
fn pivot_symbol_cycles(pivot_for_col: &[usize]) -> Vec<Box<[usize]>> {
    let mut dest_for_pivot = vec![usize::MAX; pivot_for_col.len()];
    for (col, &pivot) in pivot_for_col.iter().enumerate() {
        assert!(pivot < pivot_for_col.len());
        assert_eq!(dest_for_pivot[pivot], usize::MAX);
        dest_for_pivot[pivot] = col;
    }

    let mut cycles = Vec::new();
    let mut visited = vec![false; pivot_for_col.len()];
    for start in 0..dest_for_pivot.len() {
        if visited[start] {
            continue;
        }

        let mut cycle = Vec::new();
        let mut current = start;
        loop {
            visited[current] = true;
            cycle.push(current);
            let next = dest_for_pivot[current];
            assert!(next < dest_for_pivot.len());
            if next == start {
                break;
            }
            assert!(!visited[next]);
            current = next;
        }

        if cycle.len() > 1 {
            cycles.push(cycle.into_boxed_slice());
        }
    }

    cycles
}

#[cfg(feature = "std")]
fn move_pivot_symbols_to_columns(symbols: &mut SymbolSlab, cycles: &[Box<[usize]>]) {
    if cycles.is_empty() {
        return;
    }

    let mut scratch = vec![0u8; symbols.symbol_size()];
    for cycle in cycles {
        scratch.copy_from_slice(symbols.get(cycle[0]));
        for &next in &cycle[1..] {
            scratch.swap_with_slice(symbols.get_mut(next));
        }
        symbols.get_mut(cycle[0]).copy_from_slice(&scratch);
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

fn coefficient_rows_with_hdpc_last<M: BinaryMatrix>(
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
) -> (Vec<CoefficientRow>, SymbolSlab) {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let total_rows = matrix.height() + hdpc_rows.height();
    let mut rows = Vec::with_capacity(total_rows);
    let mut reordered_symbols = SymbolSlab::with_zeros(total_rows, symbols.symbol_size());

    for row in 0..s {
        rows.push(copy_binary_row(matrix, row));
        reordered_symbols
            .get_mut(row)
            .copy_from_slice(symbols.get(row));
    }

    for row in s..matrix.height() {
        rows.push(copy_binary_row(matrix, row));
        let source_row = row + hdpc_rows.height();
        reordered_symbols
            .get_mut(row)
            .copy_from_slice(symbols.get(source_row));
    }

    for row in 0..hdpc_rows.height() {
        rows.push(copy_octet_row(hdpc_rows, row));
        let dest = matrix.height() + row;
        let source_row = s + row;
        reordered_symbols
            .get_mut(dest)
            .copy_from_slice(symbols.get(source_row));
    }

    (rows, reordered_symbols)
}

pub fn fused_inverse_mul_symbols_no_hdpc<M: BinaryMatrix>(
    matrix: M,
    symbols: SymbolSlab,
    source_block_symbols: u32,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    assert_eq!(symbols.len(), matrix.height());
    let rows = matrix.packed_rows();

    let (decoded, ops) = solve_binary(rows, symbols);
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
    hdpc_rows_satisfied(&decoded, &hdpc_rows).then_some(decoded)
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

// Repair systems with at least L rows can often be reduced mostly over GF(2).
// Any remaining free columns form a small GF(256) system for the HDPC rows.
fn try_hybrid_binary_hdpc_solve<M: BinaryMatrix>(
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    symbols: &SymbolSlab,
    source_block_symbols: u32,
) -> Option<SymbolSlab> {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let h = hdpc_rows.height();
    let width = matrix.width();
    let binary_height = matrix.height();
    let symbol_size = symbols.symbol_size();
    let overdetermined = binary_height + h > width;

    let mut binary_symbols = SymbolSlab::with_zeros(binary_height, symbol_size);
    for row in 0..s {
        binary_symbols
            .get_mut(row)
            .copy_from_slice(symbols.get(row));
    }
    for row in s..binary_height {
        binary_symbols
            .get_mut(row)
            .copy_from_slice(symbols.get(row + h));
    }

    let mut rows = matrix.packed_rows();
    let mut bucket_heads = vec![None; width];
    let mut next_in_bucket = vec![None; binary_height];
    for row in 0..binary_height {
        if let Some(col) = rows.first_one_at_or_after(row, 0) {
            push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
        }
    }

    let mut pivot_for_col = vec![None; width];
    let mut is_pivot_row = vec![false; binary_height];
    for col in 0..width {
        let Some(pivot) =
            pop_lightest_binary_row_bucket(&rows, &mut bucket_heads, &mut next_in_bucket, col)
        else {
            continue;
        };
        pivot_for_col[col] = Some(pivot);
        is_pivot_row[pivot] = true;

        while let Some(row) = pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col) {
            rows.xor_suffix(row, pivot, col);
            let (pivot_symbol, dest_symbol) = binary_symbols.get_disjoint_mut(pivot, row);
            add_assign(dest_symbol, pivot_symbol);

            if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
            }
        }
    }

    for (row, is_pivot) in is_pivot_row.into_iter().enumerate() {
        if !is_pivot && (!rows.is_zero(row) || !symbol_is_zero(binary_symbols.get(row))) {
            return None;
        }
    }

    let free_cols = pivot_for_col
        .iter()
        .enumerate()
        .filter_map(|(col, pivot)| pivot.is_none().then_some(col))
        .collect::<Vec<_>>();
    if free_cols.len() > h {
        return None;
    }

    if overdetermined && free_cols.is_empty() {
        let decoded =
            binary_decoded_solution(&rows, &pivot_for_col, &binary_symbols, width, symbol_size);
        return hdpc_rows_satisfied(&decoded, hdpc_rows).then_some(decoded);
    }

    let mut hdpc_symbols = SymbolSlab::with_zeros(h, symbol_size);
    for row in 0..h {
        hdpc_symbols
            .get_mut(row)
            .copy_from_slice(symbols.get(s + row));
    }
    let mut hdpc_coefficients = dense_hdpc_coefficients(hdpc_rows);

    for (col, pivot) in pivot_for_col.iter().copied().enumerate() {
        let Some(pivot) = pivot else {
            continue;
        };

        for row in 0..h {
            let row_start = row * width;
            let factor = hdpc_coefficients[row_start + col];
            if factor.is_zero() {
                continue;
            }
            rows.visit_ones_at_or_after(pivot, col, |entry_col| {
                hdpc_coefficients[row_start + entry_col] += factor;
            });
            fused_addassign_mul_scalar(
                hdpc_symbols.get_mut(row),
                binary_symbols.get(pivot),
                &factor,
            );
        }
    }

    let free_values = solve_hdpc_free_variables_dense(
        hdpc_coefficients,
        hdpc_symbols,
        &free_cols,
        width,
        symbol_size,
    )?;

    let mut decoded = SymbolSlab::with_zeros(width, symbol_size);
    for (free_index, &col) in free_cols.iter().enumerate() {
        decoded
            .get_mut(col)
            .copy_from_slice(free_values.get(free_index));
    }
    for col in (0..width).rev() {
        if let Some(pivot) = pivot_for_col[col] {
            decoded
                .get_mut(col)
                .copy_from_slice(binary_symbols.get(pivot));
            rows.visit_ones_at_or_after(pivot, col + 1, |dependent_col| {
                let (dependent_symbol, dest_symbol) = decoded.get_disjoint_mut(dependent_col, col);
                add_assign(dest_symbol, dependent_symbol);
            });
        }
    }

    Some(decoded)
}

fn binary_decoded_solution(
    rows: &PackedBinaryRows,
    pivot_for_col: &[Option<usize>],
    binary_symbols: &SymbolSlab,
    width: usize,
    symbol_size: usize,
) -> SymbolSlab {
    let mut decoded = SymbolSlab::with_zeros(width, symbol_size);
    for col in (0..width).rev() {
        let pivot = pivot_for_col[col].expect("full-rank binary solve has every pivot");
        decoded
            .get_mut(col)
            .copy_from_slice(binary_symbols.get(pivot));
        rows.visit_ones_at_or_after(pivot, col + 1, |dependent_col| {
            let (dependent_symbol, dest_symbol) = decoded.get_disjoint_mut(dependent_col, col);
            add_assign(dest_symbol, dependent_symbol);
        });
    }
    decoded
}

fn dense_hdpc_coefficients(matrix: &DenseOctetMatrix) -> Vec<Octet> {
    let mut coefficients = vec![Octet::zero(); matrix.height() * matrix.width()];
    for row in 0..matrix.height() {
        let row_start = row * matrix.width();
        for col in 0..matrix.width() {
            coefficients[row_start + col] = matrix.get(row, col);
        }
    }
    coefficients
}

fn solve_hdpc_free_variables_dense(
    hdpc_coefficients: Vec<Octet>,
    hdpc_symbols: SymbolSlab,
    free_cols: &[usize],
    width: usize,
    symbol_size: usize,
) -> Option<SymbolSlab> {
    let h = hdpc_symbols.len();
    assert_eq!(hdpc_coefficients.len(), h * width);

    if free_cols.is_empty() {
        if hdpc_coefficients.iter().any(|value| !value.is_zero()) {
            return None;
        }
        for row in 0..h {
            if !symbol_is_zero(hdpc_symbols.get(row)) {
                return None;
            }
        }
        return Some(SymbolSlab::with_zeros(0, symbol_size));
    }

    let mut free_index_by_col = vec![usize::MAX; width];
    for (index, &col) in free_cols.iter().enumerate() {
        free_index_by_col[col] = index;
    }

    let mut free_rows = Vec::with_capacity(h);
    for row in 0..h {
        let row_start = row * width;
        let mut free_row = Vec::with_capacity(free_cols.len());
        for col in 0..width {
            let value = hdpc_coefficients[row_start + col];
            if value.is_zero() {
                continue;
            }
            let free_index = free_index_by_col[col];
            if free_index == usize::MAX {
                return None;
            }
            free_row.push((free_index, value));
        }
        free_rows.push(free_row);
    }

    solve_without_recording(free_rows, free_cols.len(), hdpc_symbols).0
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

    for isi in 0..k_prime {
        let expected = systematic_constraint_row_entries(source_block_symbols, isi);
        if matrix.row_entries((s + isi) as usize) != expected {
            return false;
        }
    }

    true
}

fn systematic_constraint_row_entries(source_block_symbols: u32, isi: u32) -> Vec<usize> {
    let lt_symbols = num_lt_symbols(source_block_symbols);
    let pi_symbols = num_pi_symbols(source_block_symbols);
    let sys_index = systematic_index(source_block_symbols);
    let p1 = calculate_p1(source_block_symbols);
    let tuple = intermediate_tuple(isi, lt_symbols, sys_index, p1);
    let mut entries = Vec::new();
    enc_indices(tuple, lt_symbols, pi_symbols, p1, |col| {
        entries.push(col);
    });
    entries.sort_unstable();
    entries
}

fn matrix_row_matches_systematic<M: BinaryMatrix>(
    matrix: &M,
    row: usize,
    source_block_symbols: u32,
    isi: usize,
) -> bool {
    matrix.row_entries(row) == systematic_constraint_row_entries(source_block_symbols, isi as u32)
}

#[cfg(feature = "std")]
struct SingleRepairSystematicRows {
    missing_isi: usize,
    repair_matrix_row: usize,
    systematic_rows: Vec<(usize, usize)>,
}

#[cfg(feature = "std")]
fn try_single_repair_systematic_decode<M: BinaryMatrix>(
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    symbols: &SymbolSlab,
    source_block_symbols: u32,
) -> Option<SymbolSlab> {
    let k_prime = extended_source_block_symbols(source_block_symbols) as usize;
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let h = hdpc_rows.height();
    let width = matrix.width();
    if width < SINGLE_REPAIR_SYSTEMATIC_MIN_WIDTH
        || width != s + h + k_prime
        || matrix.height() != s + k_prime
        || symbols.len() != width
    {
        return None;
    }

    let rows = classify_single_repair_systematic_rows(matrix, source_block_symbols, s, k_prime)?;
    if rows.missing_isi >= source_block_symbols as usize {
        return None;
    }

    let symbol_size = symbols.symbol_size();
    let mut systematic_symbols = SymbolSlab::with_zeros(width, symbol_size);
    for (isi, matrix_row) in rows.systematic_rows {
        systematic_symbols
            .get_mut(s + h + isi)
            .copy_from_slice(symbols.get(matrix_row + h));
    }

    let plan = cached_systematic_plan(k_prime as u32);
    apply_prepared_systematic_plan(&plan, &mut systematic_symbols);

    let mut missing_basis = SymbolSlab::with_zeros(width, 1);
    missing_basis.get_mut(s + h + rows.missing_isi)[0] = 1;
    apply_prepared_systematic_plan(&plan, &mut missing_basis);

    let repair_entries = matrix.row_entries(rows.repair_matrix_row);
    if repair_entries.is_empty() {
        return None;
    }

    let mut predicted_repair = vec![0u8; symbol_size];
    let mut repair_coefficient = Octet::zero();
    for col in repair_entries {
        add_assign(&mut predicted_repair, systematic_symbols.get(col));
        repair_coefficient += Octet::new(missing_basis.get(col)[0]);
    }
    if repair_coefficient.is_zero() {
        return None;
    }

    let mut missing_symbol = symbols.get(rows.repair_matrix_row + h).to_vec();
    add_assign(&mut missing_symbol, &predicted_repair);
    mulassign_scalar(&mut missing_symbol, &repair_coefficient.inverse());

    for col in 0..width {
        let coefficient = Octet::new(missing_basis.get(col)[0]);
        if !coefficient.is_zero() {
            fused_addassign_mul_scalar(
                systematic_symbols.get_mut(col),
                &missing_symbol,
                &coefficient,
            );
        }
    }

    Some(systematic_symbols)
}

#[cfg(feature = "std")]
fn classify_single_repair_systematic_rows<M: BinaryMatrix>(
    matrix: &M,
    source_block_symbols: u32,
    s: usize,
    k_prime: usize,
) -> Option<SingleRepairSystematicRows> {
    let mut systematic_rows = Vec::with_capacity(k_prime.saturating_sub(1));
    let mut expected_isi = 0usize;
    let mut missing_isi = None;
    let mut repair_matrix_row = None;

    for offset in 0..k_prime {
        let row = s + offset;
        if expected_isi < k_prime
            && matrix_row_matches_systematic(matrix, row, source_block_symbols, expected_isi)
        {
            systematic_rows.push((expected_isi, row));
            expected_isi += 1;
            continue;
        }

        if missing_isi.is_none()
            && expected_isi + 1 < k_prime
            && matrix_row_matches_systematic(matrix, row, source_block_symbols, expected_isi + 1)
        {
            missing_isi = Some(expected_isi);
            systematic_rows.push((expected_isi + 1, row));
            expected_isi += 2;
            continue;
        }

        if repair_matrix_row.is_some() {
            return None;
        }
        if missing_isi.is_none() {
            missing_isi = Some(expected_isi);
            expected_isi += 1;
        }
        repair_matrix_row = Some(row);
    }

    let missing_isi = missing_isi?;
    let repair_matrix_row = repair_matrix_row?;
    (expected_isi == k_prime && systematic_rows.len() + 1 == k_prime).then_some(
        SingleRepairSystematicRows {
            missing_isi,
            repair_matrix_row,
            systematic_rows,
        },
    )
}

fn solve(
    mut rows: Vec<CoefficientRow>,
    width: usize,
    mut symbols: SymbolSlab,
    recording: OperationRecording,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    if recording == OperationRecording::Skip {
        return solve_without_recording(rows, width, symbols);
    }
    if width >= TRIANGULAR_RECORDING_MIN_WIDTH {
        return solve_recording_triangular(rows, width, symbols);
    }

    let height = rows.len();
    let mut ops = recording.new_ops();
    let symbols_are_zero =
        recording == OperationRecording::Record && symbol_is_zero(symbols.as_bytes());
    let mut pivot_row = 0usize;
    let mut row_merge_scratch = Vec::new();
    let mut fused_add_batch = Vec::new();

    for col in 0..width {
        let Some((pivot, pivot_value)) =
            select_pivot_row(&rows, pivot_row, height, width, col, recording)
        else {
            return (None, None);
        };

        if pivot != pivot_row {
            rows.swap(pivot, pivot_row);
            if !symbols_are_zero {
                symbols.swap_symbols(pivot, pivot_row);
            }
            if let Some(ops) = ops.as_mut() {
                ops.push(SymbolOps::Swap(pivot, pivot_row));
            }
        }

        if pivot_value != Octet::one() {
            let scalar = pivot_value.inverse();
            scale_matrix_row(&mut rows[pivot_row], col, scalar);
            if !symbols_are_zero {
                mulassign_scalar(symbols.get_mut(pivot_row), &scalar);
            }
            if let Some(ops) = ops.as_mut() {
                ops.push(SymbolOps::Scale(pivot_row, scalar));
            }
        }

        let (rows_before, pivot_and_after) = rows.split_at_mut(pivot_row);
        let (pivot_coefficients, rows_after) = pivot_and_after
            .split_first_mut()
            .expect("pivot row must exist");
        fused_add_batch.clear();

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
            if !symbols_are_zero {
                let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(pivot_row, row);
                fused_addassign_mul_scalar(dest_symbol, pivot_symbol, &factor);
            }
            fused_add_batch.push((row, factor));
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
            if !symbols_are_zero {
                let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(pivot_row, row);
                fused_addassign_mul_scalar(dest_symbol, pivot_symbol, &factor);
            }
            fused_add_batch.push((row, factor));
        }
        push_recorded_fused_add_ops(&mut ops, pivot_row, &mut fused_add_batch);

        pivot_row += 1;
    }

    for row in pivot_row..height {
        if !rows[row].is_empty() || (!symbols_are_zero && !symbol_is_zero(symbols.get(row))) {
            return (None, None);
        }
    }

    let mut decoded = SymbolSlab::with_zeros(width, symbols.symbol_size());
    for row in 0..width {
        decoded.get_mut(row).copy_from_slice(symbols.get(row));
    }

    (Some(decoded), ops)
}

fn solve_recording_triangular(
    mut rows: Vec<CoefficientRow>,
    width: usize,
    mut symbols: SymbolSlab,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let height = rows.len();
    let mut ops = OperationRecording::Record.new_ops();
    let symbols_are_zero = symbol_is_zero(symbols.as_bytes());
    let mut row_merge_scratch = Vec::new();
    let mut fused_add_batch = Vec::new();

    for col in 0..width {
        let Some((pivot, pivot_value)) =
            select_pivot_row(&rows, col, height, width, col, OperationRecording::Record)
        else {
            return (None, None);
        };

        if pivot != col {
            rows.swap(pivot, col);
            if !symbols_are_zero {
                symbols.swap_symbols(pivot, col);
            }
            if let Some(ops) = ops.as_mut() {
                ops.push(SymbolOps::Swap(pivot, col));
            }
        }

        if pivot_value != Octet::one() {
            let scalar = pivot_value.inverse();
            scale_matrix_row(&mut rows[col], col, scalar);
            if !symbols_are_zero {
                mulassign_scalar(symbols.get_mut(col), &scalar);
            }
            if let Some(ops) = ops.as_mut() {
                ops.push(SymbolOps::Scale(col, scalar));
            }
        }

        let (pivot_and_before_after, rows_after) = rows.split_at_mut(col + 1);
        let pivot_coefficients = &pivot_and_before_after[col];
        fused_add_batch.clear();

        for (offset, row_coefficients) in rows_after.iter_mut().enumerate() {
            let row = col + 1 + offset;
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
            if !symbols_are_zero {
                let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(col, row);
                fused_addassign_mul_scalar(dest_symbol, pivot_symbol, &factor);
            }
            fused_add_batch.push((row, factor));
        }
        push_recorded_fused_add_ops(&mut ops, col, &mut fused_add_batch);
    }

    let mut back_substitution_batches = vec![Vec::new(); width];
    for dest in 0..width {
        for &(dependent_col, coefficient) in rows[dest].iter().rev() {
            if dependent_col <= dest {
                break;
            }
            back_substitution_batches[dependent_col].push((dest, coefficient));
        }
    }

    for src in (0..width).rev() {
        let batch = &mut back_substitution_batches[src];
        if !symbols_are_zero {
            apply_fused_add_batch_to_symbols(&mut symbols, src, batch);
        }
        push_recorded_fused_add_ops(&mut ops, src, batch);
    }

    let mut decoded = SymbolSlab::with_zeros(width, symbols.symbol_size());
    for row in 0..width {
        decoded.get_mut(row).copy_from_slice(symbols.get(row));
    }

    (Some(decoded), ops)
}

fn apply_fused_add_batch_to_symbols(
    symbols: &mut SymbolSlab,
    src: usize,
    batch: &[(usize, Octet)],
) {
    for &(dest, factor) in batch {
        let (src_symbol, dest_symbol) = symbols.get_disjoint_mut(src, dest);
        fused_addassign_mul_scalar(dest_symbol, src_symbol, &factor);
    }
}

fn push_recorded_fused_add_ops(
    ops: &mut Option<Vec<SymbolOps>>,
    src: usize,
    batch: &mut Vec<(usize, Octet)>,
) {
    let Some(ops) = ops.as_mut() else {
        batch.clear();
        return;
    };
    match batch.len() {
        0 => {}
        1 => {
            let (dest, scalar) = batch[0];
            ops.push(SymbolOps::FusedAdd { dest, src, scalar });
            batch.clear();
        }
        _ => {
            ops.push(SymbolOps::FusedAddBatch {
                src,
                dests: core::mem::take(batch).into_boxed_slice(),
            });
        }
    }
}

fn solve_without_recording(
    rows: Vec<CoefficientRow>,
    width: usize,
    symbols: SymbolSlab,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    if width >= COEFFICIENT_BUCKET_SOLVER_MIN_WIDTH {
        solve_without_recording_bucketed(rows, width, symbols)
    } else {
        solve_without_recording_scan(rows, width, symbols)
    }
}

fn solve_without_recording_bucketed(
    mut rows: Vec<CoefficientRow>,
    width: usize,
    mut symbols: SymbolSlab,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let height = rows.len();
    let mut row_merge_scratch = Vec::new();
    let mut bucket_heads = vec![None; width];
    let mut next_in_bucket = vec![None; height];
    for (row, coefficients) in rows.iter().enumerate() {
        if let Some(&(col, _)) = coefficients.first() {
            push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
        }
    }
    let mut pivot_for_col = vec![None; width];
    let mut is_pivot_row = vec![false; height];

    for col in 0..width {
        let Some((pivot, pivot_value)) =
            pop_lightest_coefficient_row_bucket(&rows, &mut bucket_heads, &mut next_in_bucket, col)
        else {
            return (None, None);
        };
        pivot_for_col[col] = Some(pivot);
        is_pivot_row[pivot] = true;

        if pivot_value != Octet::one() {
            let scalar = pivot_value.inverse();
            scale_matrix_row(&mut rows[pivot], col, scalar);
            mulassign_scalar(symbols.get_mut(pivot), &scalar);
        }

        let pivot_coefficients = rows[pivot].clone();

        while let Some(row) = pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col) {
            debug_assert_ne!(row, pivot);
            debug_assert_eq!(
                rows[row].first().map(|&(entry_col, _)| entry_col),
                Some(col)
            );
            let factor = rows[row][0].1;
            add_scaled_matrix_row(
                &mut rows[row],
                &pivot_coefficients,
                col,
                factor,
                &mut row_merge_scratch,
            );
            let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(pivot, row);
            fused_addassign_mul_scalar(dest_symbol, pivot_symbol, &factor);

            if let Some(&(next_col, _)) = rows[row].first() {
                debug_assert!(next_col > col);
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
            }
        }
    }

    for (row, is_pivot) in is_pivot_row.into_iter().enumerate() {
        if !is_pivot && (!rows[row].is_empty() || !symbol_is_zero(symbols.get(row))) {
            return (None, None);
        }
    }

    let mut decoded = SymbolSlab::with_zeros(width, symbols.symbol_size());
    // Non-recording solves only need the result, so the upper triangle is resolved here
    // instead of clearing earlier rows during elimination.
    for col in (0..width).rev() {
        let pivot = pivot_for_col[col].expect("pivot was recorded for every decoded column");
        decoded.get_mut(col).copy_from_slice(symbols.get(pivot));
        for &(dependent_col, coefficient) in rows[pivot].iter().rev() {
            if dependent_col <= col {
                break;
            }
            let (dependent_symbol, dest_symbol) = decoded.get_disjoint_mut(dependent_col, col);
            fused_addassign_mul_scalar(dest_symbol, dependent_symbol, &coefficient);
        }
    }

    (Some(decoded), None)
}

fn solve_without_recording_scan(
    mut rows: Vec<CoefficientRow>,
    width: usize,
    mut symbols: SymbolSlab,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let height = rows.len();
    let mut row_merge_scratch = Vec::new();

    for col in 0..width {
        let Some((pivot, pivot_value)) =
            select_pivot_row(&rows, col, height, width, col, OperationRecording::Skip)
        else {
            return (None, None);
        };

        if pivot != col {
            rows.swap(pivot, col);
            symbols.swap_symbols(pivot, col);
        }

        if pivot_value != Octet::one() {
            let scalar = pivot_value.inverse();
            scale_matrix_row(&mut rows[col], col, scalar);
            mulassign_scalar(symbols.get_mut(col), &scalar);
        }

        let (pivot_and_before_after, rows_after) = rows.split_at_mut(col + 1);
        let pivot_coefficients = &pivot_and_before_after[col];

        for (offset, row_coefficients) in rows_after.iter_mut().enumerate() {
            let row = col + 1 + offset;
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
            let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(col, row);
            fused_addassign_mul_scalar(dest_symbol, pivot_symbol, &factor);
        }
    }

    for row in width..height {
        if !rows[row].is_empty() || !symbol_is_zero(symbols.get(row)) {
            return (None, None);
        }
    }

    let mut decoded = SymbolSlab::with_zeros(width, symbols.symbol_size());
    for row in (0..width).rev() {
        decoded.get_mut(row).copy_from_slice(symbols.get(row));
        for &(col, coefficient) in rows[row].iter().rev() {
            if col <= row {
                break;
            }
            let (dependent_symbol, dest_symbol) = decoded.get_disjoint_mut(col, row);
            fused_addassign_mul_scalar(dest_symbol, dependent_symbol, &coefficient);
        }
    }

    (Some(decoded), None)
}

fn select_pivot_row(
    rows: &[CoefficientRow],
    start_row: usize,
    height: usize,
    width: usize,
    col: usize,
    recording: OperationRecording,
) -> Option<(usize, Octet)> {
    if recording == OperationRecording::Record || width < LIGHTEST_PIVOT_MIN_WIDTH {
        return (start_row..height).find_map(|row| {
            let value = coefficient_at(&rows[row], col);
            (!value.is_zero()).then_some((row, value))
        });
    }

    let mut best = None;
    let mut best_suffix_len = usize::MAX;
    let mut best_value = Octet::zero();

    for row in start_row..height {
        let Some((value, suffix_len)) = pivot_value_and_suffix_len(&rows[row], col) else {
            continue;
        };

        if suffix_len < best_suffix_len
            || (suffix_len == best_suffix_len
                && value == Octet::one()
                && best_value != Octet::one())
        {
            best = Some(row);
            best_suffix_len = suffix_len;
            best_value = value;
        }
    }

    best.map(|row| (row, best_value))
}

fn pivot_value_and_suffix_len(row: &CoefficientRow, col: usize) -> Option<(Octet, usize)> {
    if let Some(&(entry_col, value)) = row.first() {
        if entry_col == col {
            return Some((value, row.len()));
        }
        if entry_col > col {
            return None;
        }
    }

    let index = row.partition_point(|&(entry_col, _)| entry_col < col);
    match row.get(index) {
        Some(&(entry_col, value)) if entry_col == col => Some((value, row.len() - index)),
        _ => None,
    }
}

fn solve_binary(
    mut rows: PackedBinaryRows,
    mut symbols: SymbolSlab,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let width = rows.width();
    assert!(
        width <= MAX_SUPPORTED_INTERMEDIATE_SYMBOLS as usize,
        "generic RaptorQ solver supports at most {MAX_SUPPORTED_INTERMEDIATE_SYMBOLS} intermediate symbols; optimized large-matrix PI solver is not implemented"
    );

    let height = rows.height();
    assert_eq!(height, symbols.len());
    let mut bucket_heads = vec![None; width];
    let mut next_in_bucket = vec![None; height];
    for row in 0..height {
        if let Some(col) = rows.first_one_at_or_after(row, 0) {
            push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
        }
    }

    let mut pivot_for_col = vec![None; width];
    let mut is_pivot_row = vec![false; height];

    for col in 0..width {
        let Some(pivot) =
            pop_lightest_binary_row_bucket(&rows, &mut bucket_heads, &mut next_in_bucket, col)
        else {
            return (None, None);
        };
        pivot_for_col[col] = Some(pivot);
        is_pivot_row[pivot] = true;

        while let Some(row) = pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col) {
            rows.xor_suffix(row, pivot, col);
            let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(pivot, row);
            add_assign(dest_symbol, pivot_symbol);

            if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
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

fn push_row_bucket(
    bucket_heads: &mut [Option<usize>],
    next_in_bucket: &mut [Option<usize>],
    col: usize,
    row: usize,
) {
    debug_assert!(next_in_bucket[row].is_none());
    next_in_bucket[row] = bucket_heads[col];
    bucket_heads[col] = Some(row);
}

fn pop_row_bucket(
    bucket_heads: &mut [Option<usize>],
    next_in_bucket: &mut [Option<usize>],
    col: usize,
) -> Option<usize> {
    let row = bucket_heads[col]?;
    bucket_heads[col] = next_in_bucket[row];
    next_in_bucket[row] = None;
    Some(row)
}

fn pop_lightest_coefficient_row_bucket(
    rows: &[CoefficientRow],
    bucket_heads: &mut [Option<usize>],
    next_in_bucket: &mut [Option<usize>],
    col: usize,
) -> Option<(usize, Octet)> {
    let head = bucket_heads[col]?;
    debug_assert_eq!(
        rows[head].first().map(|&(entry_col, _)| entry_col),
        Some(col)
    );
    if next_in_bucket[head].is_none() {
        bucket_heads[col] = None;
        return Some((head, rows[head][0].1));
    }

    let mut best = head;
    let mut best_previous = None;
    let mut best_suffix_len = rows[head].len();
    let mut best_value = rows[head][0].1;
    let mut previous = head;
    let mut current = next_in_bucket[head];

    while let Some(row) = current {
        debug_assert_eq!(
            rows[row].first().map(|&(entry_col, _)| entry_col),
            Some(col)
        );
        let value = rows[row][0].1;
        let suffix_len = rows[row].len();
        if suffix_len < best_suffix_len
            || (suffix_len == best_suffix_len
                && value == Octet::one()
                && best_value != Octet::one())
        {
            best = row;
            best_previous = Some(previous);
            best_suffix_len = suffix_len;
            best_value = value;
            if suffix_len == 1 && value == Octet::one() {
                break;
            }
        }
        previous = row;
        current = next_in_bucket[row];
    }

    if let Some(previous) = best_previous {
        next_in_bucket[previous] = next_in_bucket[best];
    } else {
        bucket_heads[col] = next_in_bucket[best];
    }
    next_in_bucket[best] = None;
    Some((best, best_value))
}

fn pop_lightest_binary_row_bucket(
    rows: &PackedBinaryRows,
    bucket_heads: &mut [Option<usize>],
    next_in_bucket: &mut [Option<usize>],
    col: usize,
) -> Option<usize> {
    let head = bucket_heads[col]?;
    if next_in_bucket[head].is_none() {
        bucket_heads[col] = None;
        return Some(head);
    }

    let mut best = head;
    let mut best_previous = None;
    let mut best_weight = rows.weight_at_or_after(head, col);
    let mut previous = head;
    let mut current = next_in_bucket[head];

    while let Some(row) = current {
        let weight = rows.weight_at_or_after(row, col);
        if weight < best_weight {
            best = row;
            best_previous = Some(previous);
            best_weight = weight;
            if weight == 1 {
                break;
            }
        }
        previous = row;
        current = next_in_bucket[row];
    }

    if let Some(previous) = best_previous {
        next_in_bucket[previous] = next_in_bucket[best];
    } else {
        bucket_heads[col] = next_in_bucket[best];
    }
    next_in_bucket[best] = None;
    Some(best)
}

fn symbol_is_zero(symbol: &[u8]) -> bool {
    symbol.iter().all(|&byte| byte == 0)
}

fn coefficient_at(row: &CoefficientRow, col: usize) -> Octet {
    let Some(&(first_col, first_value)) = row.first() else {
        return Octet::zero();
    };
    if first_col == col {
        return first_value;
    }
    if first_col > col || row.last().is_some_and(|&(last_col, _)| last_col < col) {
        return Octet::zero();
    }

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

#[cfg(all(test, feature = "std"))]
fn add_scaled_binary_matrix_row(
    dest: &mut CoefficientRow,
    src_cols: &[usize],
    scalar: Octet,
    scratch: &mut CoefficientRow,
) {
    let mut dest_index = 0usize;
    let mut src_index = 0usize;
    scratch.clear();
    scratch.reserve(dest.len() + src_cols.len());

    while dest_index < dest.len() || src_index < src_cols.len() {
        match (dest.get(dest_index), src_cols.get(src_index)) {
            (Some(&(dest_col, dest_value)), Some(&src_col)) => {
                if dest_col < src_col {
                    scratch.push((dest_col, dest_value));
                    dest_index += 1;
                } else if src_col < dest_col {
                    scratch.push((src_col, scalar));
                    src_index += 1;
                } else {
                    let value = dest_value + scalar;
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
            (None, Some(&src_col)) => {
                scratch.push((src_col, scalar));
                src_index += 1;
            }
            (None, None) => break,
        }
    }

    core::mem::swap(dest, scratch);
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
    let mut src_index = if src.first().is_some_and(|&(col, _)| col >= start_col) {
        0
    } else {
        src.partition_point(|&(col, _)| col < start_col)
    };
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
    let mut src_index = if src.first().is_some_and(|&(col, _)| col >= start_col) {
        0
    } else {
        src.partition_point(|&(col, _)| col < start_col)
    };
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
            .map(|op| match op {
                SymbolOps::Scale(..) | SymbolOps::FusedAdd { .. } => 1,
                SymbolOps::FusedAddBatch { dests, .. } => dests.len(),
                _ => 0,
            })
            .sum()
    }

    pub fn get_symbol_add_ops(&self) -> usize {
        self.ops
            .iter()
            .map(|op| match op {
                SymbolOps::FusedAdd { .. } => 1,
                SymbolOps::FusedAddBatch { dests, .. } => dests.len(),
                _ => 0,
            })
            .sum()
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

    #[test]
    fn triangular_recording_solves_and_replays_wide_system() {
        let width = TRIANGULAR_RECORDING_MIN_WIDTH;
        let mut rows: Vec<CoefficientRow> =
            (0..width).map(|col| vec![(col, Octet::one())]).collect();
        rows[0] = vec![(0, Octet::one()), (1, Octet::new(3)), (7, Octet::new(11))];
        rows[1] = vec![(0, Octet::one()), (1, Octet::new(2)), (2, Octet::new(5))];

        let expected = (0..width)
            .map(|index| Octet::new((index as u8).wrapping_mul(37).wrapping_add(11)))
            .collect::<Vec<_>>();
        let rhs = rows
            .iter()
            .map(|row| {
                row.iter()
                    .fold(Octet::zero(), |acc, &(col, value)| {
                        acc + value * expected[col]
                    })
                    .value()
            })
            .collect::<Vec<_>>();
        let symbols = SymbolSlab::from_bytes(rhs, 1);

        let (decoded, ops) = solve_recording_triangular(rows, width, symbols.clone());

        let expected_bytes = expected
            .iter()
            .map(|value| value.value())
            .collect::<Vec<_>>();
        assert_eq!(decoded.unwrap().as_bytes(), expected_bytes.as_slice());

        let mut replayed = symbols;
        for op in ops.unwrap() {
            crate::operation_vector::perform_op(&op, &mut replayed);
        }
        assert_eq!(replayed.as_bytes(), expected_bytes.as_slice());
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
    fn non_recording_solver_back_substitutes_non_unit_pivots() {
        let x0 = Octet::new(0x11);
        let x1 = Octet::new(0x22);
        let x2 = Octet::new(0x33);
        let rows = vec![
            vec![(0, Octet::new(5)), (1, Octet::new(2)), (2, Octet::new(3))],
            vec![(1, Octet::new(7)), (2, Octet::new(4))],
            vec![(2, Octet::new(9))],
        ];
        let symbols = SymbolSlab::from_bytes(
            vec![
                (Octet::new(5) * x0 + Octet::new(2) * x1 + Octet::new(3) * x2).value(),
                (Octet::new(7) * x1 + Octet::new(4) * x2).value(),
                (Octet::new(9) * x2).value(),
            ],
            1,
        );

        let (decoded, ops) = solve(rows, 3, symbols, OperationRecording::Skip);

        assert_eq!(
            decoded.unwrap(),
            SymbolSlab::from_bytes(vec![x0.value(), x1.value(), x2.value()], 1)
        );
        assert!(ops.is_none());
    }

    #[test]
    fn prepared_systematic_plan_matches_non_recording_solve() {
        let rows = vec![
            vec![(0, Octet::new(2)), (1, Octet::new(5)), (3, Octet::new(7))],
            vec![(0, Octet::new(6)), (1, Octet::new(3)), (2, Octet::new(11))],
            vec![(1, Octet::new(9)), (2, Octet::new(5)), (3, Octet::new(13))],
            vec![(2, Octet::new(10)), (3, Octet::new(17))],
        ];
        let symbols = SymbolSlab::from_bytes(
            vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 7, 11, 19, 31],
            4,
        );

        let (expected, ops) = solve(rows.clone(), 4, symbols.clone(), OperationRecording::Skip);
        let plan = prepare_cached_systematic_plan(rows, 4);
        let mut replayed = symbols;
        apply_prepared_systematic_plan(&plan, &mut replayed);

        assert_eq!(replayed, expected.unwrap());
        assert!(ops.is_none());
    }

    #[test]
    fn prepared_systematic_plan_replays_pivot_permutation_in_place() {
        let rows = vec![
            vec![(0, Octet::one()), (1, Octet::one())],
            vec![(0, Octet::one())],
            vec![(2, Octet::one())],
        ];
        let symbols = SymbolSlab::from_bytes(vec![9, 4, 12], 1);
        let plan = prepare_cached_systematic_plan(rows, 3);
        let mut replayed = symbols;

        apply_prepared_systematic_plan(&plan, &mut replayed);

        assert_eq!(replayed.as_bytes(), &[4, 13, 12]);
    }

    #[test]
    fn scaled_binary_matrix_row_matches_unit_coefficient_merge() {
        let src_cols = vec![0, 2, 5, 8];
        let src_row = src_cols
            .iter()
            .map(|&col| (col, Octet::one()))
            .collect::<Vec<_>>();

        for scalar in [Octet::one(), Octet::new(7)] {
            let dest = vec![
                (0, scalar),
                (1, Octet::new(3)),
                (4, Octet::new(9)),
                (5, Octet::new(2)),
                (9, Octet::new(11)),
            ];
            let mut generic = dest.clone();
            let mut binary = dest;
            let mut generic_scratch = Vec::new();
            let mut binary_scratch = Vec::new();

            add_scaled_matrix_row(&mut generic, &src_row, 0, scalar, &mut generic_scratch);
            add_scaled_binary_matrix_row(&mut binary, &src_cols, scalar, &mut binary_scratch);

            assert_eq!(binary, generic);
        }
    }

    fn bucketed_rebucket_system(
        inconsistent: bool,
    ) -> (Vec<CoefficientRow>, SymbolSlab, [Octet; 3]) {
        let width = COEFFICIENT_BUCKET_SOLVER_MIN_WIDTH;
        let x0 = Octet::new(0x11);
        let x1 = Octet::new(0x22);
        let x2 = Octet::new(0x33);
        let one = Octet::one();

        let mut rows = Vec::with_capacity(width + 1);
        rows.push(vec![(0, one)]);
        rows.push(vec![(0, one), (1, one), (2, one)]);
        rows.push(vec![(1, Octet::new(7))]);
        for col in 3..width {
            rows.push(vec![(col, one)]);
        }
        rows.push(vec![(0, one), (2, one)]);

        let mut redundant_symbol = x0 + x2;
        if inconsistent {
            redundant_symbol += one;
        }
        let mut symbol_bytes = Vec::with_capacity(width + 1);
        symbol_bytes.push(x0.value());
        symbol_bytes.push((x0 + x1 + x2).value());
        symbol_bytes.push((Octet::new(7) * x1).value());
        symbol_bytes.extend(core::iter::repeat_n(0, width - 3));
        symbol_bytes.push(redundant_symbol.value());
        let symbols = SymbolSlab::from_bytes(symbol_bytes, 1);

        (rows, symbols, [x0, x1, x2])
    }

    #[test]
    fn bucketed_non_recording_solver_rebuckets_eliminated_rows() {
        let width = COEFFICIENT_BUCKET_SOLVER_MIN_WIDTH;
        let (rows, symbols, [x0, x1, x2]) = bucketed_rebucket_system(false);

        let (decoded, ops) = solve(rows, width, symbols, OperationRecording::Skip);

        let decoded = decoded.unwrap();
        assert_eq!(decoded.get(0), &[x0.value()]);
        assert_eq!(decoded.get(1), &[x1.value()]);
        assert_eq!(decoded.get(2), &[x2.value()]);
        assert!(ops.is_none());
    }

    #[test]
    fn bucketed_non_recording_solver_rejects_inconsistent_rebucketed_row() {
        let width = COEFFICIENT_BUCKET_SOLVER_MIN_WIDTH;
        let (rows, symbols, _) = bucketed_rebucket_system(true);

        let (decoded, ops) = solve(rows, width, symbols, OperationRecording::Skip);

        assert!(decoded.is_none());
        assert!(ops.is_none());
    }

    #[test]
    fn non_recording_hdpc_last_ordering_preserves_symbol_mapping() {
        let source_block_symbols = 10;
        let width = LIGHTEST_PIVOT_MIN_WIDTH;
        let s = num_ldpc_symbols(source_block_symbols) as usize;
        let mut matrix = DenseBinaryMatrix::new(width - 1, width);
        for row in 0..(width - 1) {
            matrix.set(row, row, true);
        }

        let mut hdpc_rows = DenseOctetMatrix::new(1, width);
        hdpc_rows.set(0, width - 1, Octet::one());

        let mut expected_bytes = Vec::with_capacity(width * 2);
        for symbol in 0..width {
            expected_bytes.push((symbol as u8).wrapping_mul(3).wrapping_add(1));
            expected_bytes.push((symbol as u8) ^ 0xa5);
        }
        let expected = SymbolSlab::from_bytes(expected_bytes, 2);

        let mut symbols = SymbolSlab::with_zeros(width, expected.symbol_size());
        for row in 0..s {
            symbols.get_mut(row).copy_from_slice(expected.get(row));
        }
        symbols.get_mut(s).copy_from_slice(expected.get(width - 1));
        for row in s..(width - 1) {
            symbols
                .get_mut(row + hdpc_rows.height())
                .copy_from_slice(expected.get(row));
        }

        let original_rows = coefficient_rows(&matrix, &hdpc_rows, source_block_symbols);
        let (original, _) = solve(
            original_rows,
            width,
            symbols.clone(),
            OperationRecording::Record,
        );
        let (optimized, ops) =
            fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_block_symbols);

        assert_eq!(optimized, original);
        assert_eq!(optimized.unwrap(), expected);
        assert!(ops.is_none());
    }

    struct OverdeterminedHybridSystem {
        matrix: DenseBinaryMatrix,
        hdpc_rows: DenseOctetMatrix,
        symbols: SymbolSlab,
        corrupt_symbol_row: usize,
        source_block_symbols: u32,
    }

    fn overdetermined_full_rank_hybrid_system() -> OverdeterminedHybridSystem {
        let source_block_symbols = 10;
        let width = LIGHTEST_PIVOT_MIN_WIDTH;
        let h = 1;
        let binary_height = width + 1;
        let symbol_size = 2;
        let s = num_ldpc_symbols(source_block_symbols) as usize;
        let corruptible_repair_row = s;
        let mut matrix = DenseBinaryMatrix::new(binary_height, width);
        for col in 0..width {
            matrix.set(col, col, true);
        }
        matrix.set(width, 1, true);

        let mut hdpc_rows = DenseOctetMatrix::new(h, width);
        hdpc_rows.set(0, corruptible_repair_row, Octet::one());
        let symbols = SymbolSlab::with_zeros(binary_height + h, symbol_size);

        assert!(width <= OVERDETERMINED_HYBRID_MAX_WIDTH);
        assert!(binary_height + h > width);

        OverdeterminedHybridSystem {
            matrix,
            hdpc_rows,
            symbols,
            corrupt_symbol_row: corruptible_repair_row + h,
            source_block_symbols,
        }
    }

    #[test]
    fn overdetermined_full_rank_hybrid_decode_uses_binary_solution() {
        let system = overdetermined_full_rank_hybrid_system();

        let (decoded, ops) = fused_inverse_mul_symbols(
            system.matrix,
            system.hdpc_rows,
            system.symbols,
            system.source_block_symbols,
        );

        assert_eq!(
            decoded.unwrap(),
            SymbolSlab::with_zeros(LIGHTEST_PIVOT_MIN_WIDTH, 2)
        );
        assert!(ops.is_none());
    }

    #[test]
    fn overdetermined_full_rank_hybrid_decode_rejects_hdpc_failure() {
        let mut system = overdetermined_full_rank_hybrid_system();
        system.symbols.get_mut(system.corrupt_symbol_row)[0] = 0x5a;

        let (decoded, ops) = fused_inverse_mul_symbols(
            system.matrix,
            system.hdpc_rows,
            system.symbols,
            system.source_block_symbols,
        );

        assert!(decoded.is_none());
        assert!(ops.is_none());
    }

    // Row iteration is intentionally empty so these tests prove hybrid dispatch.
    #[derive(Clone)]
    struct PackedOnlyMatrix {
        rows: PackedBinaryRows,
    }

    impl PackedOnlyMatrix {
        fn new(rows: PackedBinaryRows) -> PackedOnlyMatrix {
            PackedOnlyMatrix { rows }
        }
    }

    impl BinaryMatrix for PackedOnlyMatrix {
        fn new(_height: usize, _width: usize) -> PackedOnlyMatrix {
            unimplemented!("test fixture builds packed rows directly")
        }

        fn height(&self) -> usize {
            self.rows.height()
        }

        fn width(&self) -> usize {
            self.rows.width()
        }

        fn get(&self, row: usize, col: usize) -> Octet {
            if self.rows.contains(row, col) {
                Octet::one()
            } else {
                Octet::zero()
            }
        }

        fn set(&mut self, _row: usize, _col: usize, _value: bool) {
            unimplemented!("test fixture builds packed rows directly")
        }

        fn packed_rows(&self) -> PackedBinaryRows {
            self.rows.clone()
        }

        fn visit_row_entries<F>(&self, row: usize, _visit: F)
        where
            F: FnMut(usize),
        {
            assert!(row < self.height());
        }
    }

    fn packed_identity_with_duplicate_row(width: usize) -> PackedBinaryRows {
        let mut rows = PackedBinaryRows::new(width + 1, width);
        for col in 0..width {
            rows.set(col, col);
        }
        rows.set(width, 0);
        rows
    }

    fn packed_identity_with_free_last_column(width: usize) -> PackedBinaryRows {
        let mut rows = PackedBinaryRows::new(width - 1, width);
        for col in 0..(width - 1) {
            rows.set(col, col);
        }
        rows
    }

    #[test]
    fn overdetermined_hybrid_decode_handles_width_above_generic_cap() {
        let width = MAX_SUPPORTED_INTERMEDIATE_SYMBOLS as usize + 1;
        let source_block_symbols = 10;
        let symbol_size = 1;
        let matrix = PackedOnlyMatrix::new(packed_identity_with_duplicate_row(width));

        let hdpc_rows = DenseOctetMatrix::new(1, width);
        let symbols = SymbolSlab::with_zeros(matrix.height() + hdpc_rows.height(), symbol_size);

        let (decoded, ops) =
            fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_block_symbols);

        assert_eq!(decoded.unwrap(), SymbolSlab::with_zeros(width, symbol_size));
        assert!(ops.is_none());
    }

    #[test]
    fn square_hybrid_decode_stays_capped_above_square_limit() {
        let width = SQUARE_HYBRID_MAX_WIDTH + 1;
        let source_block_symbols = 10;
        let symbol_size = 1;
        let matrix = PackedOnlyMatrix::new(packed_identity_with_free_last_column(width));
        let mut hdpc_rows = DenseOctetMatrix::new(1, width);
        hdpc_rows.set(0, width - 1, Octet::one());
        let symbols = SymbolSlab::with_zeros(matrix.height() + hdpc_rows.height(), symbol_size);

        let (decoded, ops) =
            fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_block_symbols);

        assert!(decoded.is_none());
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
                forward_steps: Vec::new(),
                pivot_symbol_cycles: Vec::new(),
                back_substitution_rows: Vec::new(),
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
    fn single_repair_systematic_decode_recovers_missing_source() {
        let source_symbols = 10;
        let k_prime = extended_source_block_symbols(source_symbols);
        let width = num_intermediate_symbols(source_symbols) as usize;
        let s = num_ldpc_symbols(source_symbols) as usize;
        let h = width - s - k_prime as usize;
        let symbol_size = 3;
        let missing_isi = 3;
        let repair_isi = k_prime;

        let mut full_d = SymbolSlab::with_zeros(width, symbol_size);
        for isi in 0..source_symbols as usize {
            full_d.get_mut(s + h + isi).copy_from_slice(&[
                isi as u8 + 1,
                (isi as u8).wrapping_mul(7).wrapping_add(3),
                (isi as u8) ^ 0xa5,
            ]);
        }
        let mut expected_intermediate = full_d.clone();
        apply_cached_systematic_plan(source_symbols, &mut expected_intermediate);

        let encoded_isis = (0..k_prime)
            .filter(|&isi| isi != missing_isi)
            .chain(core::iter::once(repair_isi))
            .collect::<Vec<_>>();
        let (matrix, hdpc_rows) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &encoded_isis);
        let h = hdpc_rows.height();
        let mut decode_symbols = SymbolSlab::with_zeros(matrix.height() + h, symbol_size);
        for (offset, &isi) in encoded_isis.iter().enumerate() {
            let matrix_row = s + offset;
            let symbol_row = matrix_row + h;
            if isi < k_prime {
                decode_symbols
                    .get_mut(symbol_row)
                    .copy_from_slice(full_d.get(s + h + isi as usize));
                continue;
            }

            let mut repair = vec![0u8; symbol_size];
            for col in matrix.row_entries(matrix_row) {
                add_assign(&mut repair, expected_intermediate.get(col));
            }
            decode_symbols.get_mut(symbol_row).copy_from_slice(&repair);
        }

        let fast_decoded = try_single_repair_systematic_decode(
            &matrix,
            &hdpc_rows,
            &decode_symbols,
            source_symbols,
        );
        assert_eq!(fast_decoded.as_ref(), Some(&expected_intermediate));

        let (decoded, ops) =
            fused_inverse_mul_symbols(matrix, hdpc_rows, decode_symbols, source_symbols);
        assert_eq!(decoded, Some(expected_intermediate));
        assert!(ops.is_none());
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

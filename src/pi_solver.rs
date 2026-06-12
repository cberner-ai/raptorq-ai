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
use crate::octets::{
    AddAssignFastPath, add_assign, bytes_are_zero, fused_addassign_mul_scalar, mulassign_scalar,
};
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

type CoefficientColumn = u16;
type CoefficientRow = Vec<(CoefficientColumn, Octet)>;
#[cfg(feature = "std")]
type DirectSystematicFreeRows = Vec<Box<[(CoefficientColumn, Octet)]>>;
const NO_BUCKET_ROW: usize = usize::MAX;
#[cfg(feature = "std")]
const NO_COEFFICIENT_COLUMN: CoefficientColumn = u16::MAX;
pub(crate) const MAX_INLINE_RECORDED_SOLVER_WIDTH: usize = 4096;
#[cfg(feature = "std")]
const SYSTEMATIC_PLAN_FORWARD_DESTS_PER_COL_HINT: usize = 96;
const LIGHTEST_PIVOT_MIN_WIDTH: usize = 64;
const COEFFICIENT_BUCKET_SOLVER_MIN_WIDTH: usize = 512;
const SPARSE_SOURCE_MERGE_DEST_FACTOR: usize = 3;
const SPARSE_SOURCE_MERGE_MAX_SOURCE_LEN: usize = 512;
const SPARSE_SOURCE_LINEAR_SCAN_LIMIT: usize = 24;
const SHORT_ROW_LINEAR_SCAN_LIMIT: usize = 8;
const TRIANGULAR_RECORDING_MIN_WIDTH: usize = MAX_SUPPORTED_INTERMEDIATE_SYMBOLS as usize + 1;
const HYBRID_MAX_WIDTH: usize = u16::MAX as usize;
const SQUARE_HYBRID_MAX_WIDTH: usize = HYBRID_MAX_WIDTH;
const OVERDETERMINED_HYBRID_MAX_WIDTH: usize = HYBRID_MAX_WIDTH;
#[cfg(not(test))]
const SINGLE_REPAIR_SYSTEMATIC_MIN_WIDTH: usize = 4096;
#[cfg(test)]
const SINGLE_REPAIR_SYSTEMATIC_MIN_WIDTH: usize = 1;
#[cfg(feature = "std")]
const SYSTEMATIC_PLAN_CACHE_CAPACITY: usize = 16;
#[cfg(feature = "std")]
const DIRECT_SYSTEMATIC_PLAN_CACHE_CAPACITY: usize = 16;
#[cfg(feature = "std")]
const BATCHED_BACK_SUBSTITUTION_MIN_WIDTH: usize = MAX_INLINE_RECORDED_SOLVER_WIDTH + 1;
#[cfg(feature = "std")]
const FLAT_BACK_SUBSTITUTION_MIN_WIDTH: usize = MAX_INLINE_RECORDED_SOLVER_WIDTH + 1;
#[cfg(feature = "std")]
const CLONE_FREE_PLAN_ELIMINATION_MIN_WIDTH: usize = 16_384;
#[cfg(feature = "std")]
const DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH: usize = 16_384;
#[cfg(feature = "std")]
const SHORT_PIVOT_MERGE_MAX_LEN: usize = 64;
#[cfg(feature = "std")]
const REPAIR_SOURCE_COEFFICIENTS_CACHE_CAPACITY: usize = 16;
#[cfg(all(feature = "std", not(test)))]
const IN_PLACE_HYBRID_REPLAY_MIN_WIDTH: usize = 32_768;
#[cfg(all(feature = "std", test))]
const IN_PLACE_HYBRID_REPLAY_MIN_WIDTH: usize = 64;
#[cfg(all(test, feature = "std"))]
const SINGLE_REPAIR_BASIS_CACHE_CAPACITY: usize = 64;

fn coefficient_col(col: usize) -> CoefficientColumn {
    debug_assert!(CoefficientColumn::try_from(col).is_ok());
    col as CoefficientColumn
}

fn coefficient_col_index(col: CoefficientColumn) -> usize {
    col as usize
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

pub(crate) fn fused_inverse_mul_symbols<M: BinaryMatrix>(
    matrix: M,
    hdpc_rows: DenseOctetMatrix,
    symbols: SymbolSlab,
    source_block_symbols: u32,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let recording = OperationRecording::for_matrix(&matrix, source_block_symbols);
    if recording == OperationRecording::Record && matrix.width() > MAX_INLINE_RECORDED_SOLVER_WIDTH
    {
        let source_block_symbols = extended_source_block_symbols(source_block_symbols);
        #[cfg(feature = "std")]
        {
            if matrix.width() < DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH {
                if cached_hybrid_systematic_plan_from_matrix(
                    source_block_symbols,
                    &matrix,
                    &hdpc_rows,
                )
                .is_some()
                {
                    return (
                        None,
                        Some(vec![SymbolOps::DirectSystematicSolve {
                            source_block_symbols,
                        }]),
                    );
                }
                cached_systematic_plan_from_matrix(source_block_symbols, matrix, &hdpc_rows);
                return (
                    None,
                    Some(vec![SymbolOps::ApplyCachedSystematicPlan {
                        source_block_symbols,
                    }]),
                );
            }

            if matrix.width() > SQUARE_HYBRID_MAX_WIDTH {
                cached_systematic_plan_from_matrix(source_block_symbols, matrix, &hdpc_rows);
                return (
                    None,
                    Some(vec![SymbolOps::ApplyCachedSystematicPlan {
                        source_block_symbols,
                    }]),
                );
            }
        }

        // Wide systematic plans are better represented by a semantic operation. Replay can use a
        // prepared binary+HDPC plan instead of caching a large coefficient plan.
        let op = SymbolOps::DirectSystematicSolve {
            source_block_symbols,
        };
        #[cfg(feature = "std")]
        {
            cached_direct_systematic_plan_from_matrix(source_block_symbols, &matrix, &hdpc_rows);
        }
        if symbol_is_zero(symbols.as_bytes()) {
            let decoded = SymbolSlab::with_zeros(matrix.width(), symbols.symbol_size());
            return (Some(decoded), Some(vec![op]));
        }

        #[cfg(feature = "std")]
        {
            let mut decoded = symbols.clone();
            apply_direct_systematic_solve(source_block_symbols, &mut decoded);
            return (Some(decoded), Some(vec![op]));
        }

        #[cfg(not(feature = "std"))]
        {
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
    forward_dests: CachedSystematicSlices,
    pivot_symbol_cycles: Vec<Box<[usize]>>,
    back_substitution: CachedSystematicBackSubstitution,
    width: usize,
}

#[cfg(feature = "std")]
struct CachedHybridSystematicPlan {
    binary_forward_dests: CachedSystematicSlices,
    hdpc_symbol_steps: Vec<HybridHdpcSymbolStep>,
    free_cols: Box<[usize]>,
    free_rows: Vec<CoefficientRow>,
    pivots: Box<[(usize, usize)]>,
    back_substitution: CachedSystematicSlices,
    output_symbol_cycles: Option<Vec<Box<[usize]>>>,
    s: usize,
    h: usize,
    width: usize,
}

#[cfg(feature = "std")]
struct HybridHdpcSymbolStep {
    row: usize,
    pivot: usize,
    factor: Octet,
}

struct CachedSystematicForwardStep {
    pivot: usize,
    scale: Option<Octet>,
}

#[cfg(feature = "std")]
enum CachedSystematicBackSubstitution {
    Rows(Vec<Box<[(usize, Octet)]>>),
    Batches(Vec<Box<[(usize, Octet)]>>),
    FlatBatches(CachedSystematicSlices),
}

#[cfg(feature = "std")]
struct CachedSystematicSlices {
    ranges: Vec<(usize, usize)>,
    entries: Vec<(CoefficientColumn, Octet)>,
    unit_only: Vec<bool>,
}

#[cfg(feature = "std")]
type CachedSystematicSliceParts = (
    Vec<(usize, usize)>,
    Vec<usize>,
    Vec<(CoefficientColumn, Octet)>,
    usize,
);

#[cfg(feature = "std")]
impl CachedSystematicSlices {
    fn slice(&self, index: usize) -> &[(CoefficientColumn, Octet)] {
        let (start, end) = self.ranges[index];
        &self.entries[start..end]
    }

    fn is_unit_only(&self, index: usize) -> bool {
        self.unit_only[index]
    }
}

#[cfg(feature = "std")]
struct DirectSystematicPlan {
    forward_steps: Vec<DirectSystematicForwardStep>,
    forward_dests: DirectSystematicSlices,
    hdpc_update_pivots: Box<[CoefficientColumn]>,
    hdpc_updates: CachedSystematicSlices,
    hdpc_free_rows: DirectSystematicFreeRows,
    free_cols: Box<[CoefficientColumn]>,
    pivot_symbol_moves: Vec<Box<[usize]>>,
    back_substitution: DirectSystematicSlices,
    width: usize,
    s: usize,
    h: usize,
}

#[cfg(feature = "std")]
struct DirectSystematicForwardStep {
    pivot_symbol: CoefficientColumn,
}

#[cfg(feature = "std")]
struct DirectSystematicSlices {
    ranges: Vec<(usize, usize)>,
    entries: Vec<CoefficientColumn>,
}

#[cfg(feature = "std")]
type DirectSystematicSliceParts = (
    Vec<(usize, usize)>,
    Vec<usize>,
    Vec<CoefficientColumn>,
    usize,
);

#[cfg(feature = "std")]
impl DirectSystematicSlices {
    fn slice(&self, index: usize) -> &[CoefficientColumn] {
        let (start, end) = self.ranges[index];
        &self.entries[start..end]
    }
}

#[cfg(feature = "std")]
#[derive(Default)]
struct SystematicPlanCache {
    plans: HashMap<u32, Arc<CachedSystematicPlan>>,
    insertion_order: VecDeque<u32>,
}

#[cfg(feature = "std")]
#[derive(Default)]
struct HybridSystematicPlanCache {
    plans: HashMap<u32, Arc<CachedHybridSystematicPlan>>,
    insertion_order: VecDeque<u32>,
}

#[cfg(feature = "std")]
type SystematicPlanCacheLock = Mutex<SystematicPlanCache>;

#[cfg(feature = "std")]
type HybridSystematicPlanCacheLock = Mutex<HybridSystematicPlanCache>;

#[cfg(feature = "std")]
fn systematic_plan_cache() -> &'static SystematicPlanCacheLock {
    static CACHE: OnceLock<SystematicPlanCacheLock> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(SystematicPlanCache::default()))
}

#[cfg(feature = "std")]
fn hybrid_systematic_plan_cache() -> &'static HybridSystematicPlanCacheLock {
    static CACHE: OnceLock<HybridSystematicPlanCacheLock> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HybridSystematicPlanCache::default()))
}

#[cfg(feature = "std")]
#[derive(Default)]
struct DirectSystematicPlanCache {
    plans: HashMap<u32, Arc<DirectSystematicPlan>>,
    insertion_order: VecDeque<u32>,
}

#[cfg(feature = "std")]
type DirectSystematicPlanCacheLock = Mutex<DirectSystematicPlanCache>;

#[cfg(feature = "std")]
fn direct_systematic_plan_cache() -> &'static DirectSystematicPlanCacheLock {
    static CACHE: OnceLock<DirectSystematicPlanCacheLock> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(DirectSystematicPlanCache::default()))
}

#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct RepairSourceCoefficientsKey {
    source_block_symbols: u32,
    repair_isi: u32,
}

#[cfg(feature = "std")]
struct CachedRepairSourceCoefficients {
    source_coefficients: Box<[u8]>,
    nonzero_sources: Box<[(usize, Octet)]>,
}

#[cfg(feature = "std")]
#[derive(Default)]
struct RepairSourceCoefficientsCache {
    coefficients: HashMap<RepairSourceCoefficientsKey, Arc<CachedRepairSourceCoefficients>>,
    insertion_order: VecDeque<RepairSourceCoefficientsKey>,
}

#[cfg(feature = "std")]
type RepairSourceCoefficientsCacheLock = Mutex<RepairSourceCoefficientsCache>;

#[cfg(feature = "std")]
fn repair_source_coefficients_cache() -> &'static RepairSourceCoefficientsCacheLock {
    static CACHE: OnceLock<RepairSourceCoefficientsCacheLock> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(RepairSourceCoefficientsCache::default()))
}

#[cfg(all(test, feature = "std"))]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SingleRepairBasisKey {
    source_block_symbols: u32,
    missing_isi: usize,
}

#[cfg(all(test, feature = "std"))]
struct CachedSingleRepairBasis {
    coefficients: Vec<u8>,
    nonzero_cols: Box<[usize]>,
}

#[cfg(all(test, feature = "std"))]
#[derive(Default)]
struct SingleRepairBasisCache {
    bases: HashMap<SingleRepairBasisKey, Arc<CachedSingleRepairBasis>>,
    insertion_order: VecDeque<SingleRepairBasisKey>,
}

#[cfg(all(test, feature = "std"))]
type SingleRepairBasisCacheLock = Mutex<SingleRepairBasisCache>;

#[cfg(all(test, feature = "std"))]
fn single_repair_basis_cache() -> &'static SingleRepairBasisCacheLock {
    static CACHE: OnceLock<SingleRepairBasisCacheLock> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(SingleRepairBasisCache::default()))
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
fn cached_systematic_plan_from_matrix<M: BinaryMatrix>(
    source_block_symbols: u32,
    matrix: M,
    hdpc_rows: &DenseOctetMatrix,
) -> Arc<CachedSystematicPlan> {
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

    let width = matrix.width();
    let matrix_height = matrix.height();
    let (rows, binary_rows) = coefficient_rows_from_binary_entries(
        matrix.into_row_entries(),
        matrix_height,
        hdpc_rows,
        source_block_symbols,
    );
    let generated = Arc::new(prepare_cached_systematic_plan_with_binary_rows(
        rows,
        width,
        binary_rows,
    ));
    let cache = systematic_plan_cache();
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    insert_systematic_plan(&mut guard, source_block_symbols, generated)
}

#[cfg(feature = "std")]
fn cached_hybrid_systematic_plan_if_present(
    source_block_symbols: u32,
) -> Option<Arc<CachedHybridSystematicPlan>> {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    let cache = hybrid_systematic_plan_cache();
    let guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.plans.get(&source_block_symbols).map(Arc::clone)
}

#[cfg(feature = "std")]
fn cached_hybrid_systematic_plan_from_matrix<M: BinaryMatrix>(
    source_block_symbols: u32,
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
) -> Option<Arc<CachedHybridSystematicPlan>> {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    {
        let cache = hybrid_systematic_plan_cache();
        let guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(plan) = guard.plans.get(&source_block_symbols) {
            return Some(Arc::clone(plan));
        }
    }

    let generated = Arc::new(prepare_cached_hybrid_systematic_plan(
        source_block_symbols,
        matrix,
        hdpc_rows,
    )?);
    let cache = hybrid_systematic_plan_cache();
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Some(insert_hybrid_systematic_plan(
        &mut guard,
        source_block_symbols,
        generated,
    ))
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
fn insert_hybrid_systematic_plan(
    cache: &mut HybridSystematicPlanCache,
    source_block_symbols: u32,
    generated: Arc<CachedHybridSystematicPlan>,
) -> Arc<CachedHybridSystematicPlan> {
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
fn cached_direct_systematic_plan(source_block_symbols: u32) -> Arc<DirectSystematicPlan> {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    {
        let cache = direct_systematic_plan_cache();
        let guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(plan) = guard.plans.get(&source_block_symbols) {
            return Arc::clone(plan);
        }
    }

    let generated = Arc::new(generate_direct_systematic_plan(source_block_symbols));
    let cache = direct_systematic_plan_cache();
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    insert_direct_systematic_plan(&mut guard, source_block_symbols, generated)
}

#[cfg(feature = "std")]
fn cached_direct_systematic_plan_from_matrix<M: BinaryMatrix>(
    source_block_symbols: u32,
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
) -> Arc<DirectSystematicPlan> {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    {
        let cache = direct_systematic_plan_cache();
        let guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(plan) = guard.plans.get(&source_block_symbols) {
            return Arc::clone(plan);
        }
    }

    let generated = Arc::new(
        prepare_direct_systematic_plan(matrix, hdpc_rows, source_block_symbols)
            .expect("systematic direct plan generation failed"),
    );
    let cache = direct_systematic_plan_cache();
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    insert_direct_systematic_plan(&mut guard, source_block_symbols, generated)
}

#[cfg(feature = "std")]
fn insert_direct_systematic_plan(
    cache: &mut DirectSystematicPlanCache,
    source_block_symbols: u32,
    generated: Arc<DirectSystematicPlan>,
) -> Arc<DirectSystematicPlan> {
    if let Some(plan) = cache.plans.get(&source_block_symbols) {
        return Arc::clone(plan);
    }

    if cache.plans.len() >= DIRECT_SYSTEMATIC_PLAN_CACHE_CAPACITY
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
fn cached_repair_source_coefficients(
    source_block_symbols: u32,
    repair_isi: u32,
    plan: &CachedSystematicPlan,
    s: usize,
    h: usize,
) -> Arc<CachedRepairSourceCoefficients> {
    let key = RepairSourceCoefficientsKey {
        source_block_symbols,
        repair_isi,
    };
    {
        let cache = repair_source_coefficients_cache();
        let guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(coefficients) = guard.coefficients.get(&key) {
            return Arc::clone(coefficients);
        }
    }

    let repair_entries = systematic_constraint_row_entries(source_block_symbols, repair_isi);
    let generated = Arc::new(generate_repair_source_coefficients(
        plan,
        &repair_entries,
        s,
        h,
        source_block_symbols as usize,
    ));
    let cache = repair_source_coefficients_cache();
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    insert_repair_source_coefficients(&mut guard, key, generated)
}

#[cfg(feature = "std")]
fn insert_repair_source_coefficients(
    cache: &mut RepairSourceCoefficientsCache,
    key: RepairSourceCoefficientsKey,
    generated: Arc<CachedRepairSourceCoefficients>,
) -> Arc<CachedRepairSourceCoefficients> {
    if let Some(coefficients) = cache.coefficients.get(&key) {
        return Arc::clone(coefficients);
    }

    if cache.coefficients.len() >= REPAIR_SOURCE_COEFFICIENTS_CACHE_CAPACITY
        && let Some(evicted_key) = cache.insertion_order.pop_front()
    {
        cache.coefficients.remove(&evicted_key);
    }

    cache.insertion_order.push_back(key);
    cache.coefficients.insert(key, Arc::clone(&generated));
    generated
}

#[cfg(all(test, feature = "std"))]
fn cached_single_repair_basis(
    source_block_symbols: u32,
    missing_isi: usize,
) -> Arc<CachedSingleRepairBasis> {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    let key = SingleRepairBasisKey {
        source_block_symbols,
        missing_isi,
    };
    {
        let cache = single_repair_basis_cache();
        let guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(basis) = guard.bases.get(&key) {
            return Arc::clone(basis);
        }
    }

    let generated = Arc::new(generate_single_repair_basis(
        source_block_symbols,
        missing_isi,
    ));
    let cache = single_repair_basis_cache();
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    insert_single_repair_basis(&mut guard, key, generated)
}

#[cfg(all(test, feature = "std"))]
fn insert_single_repair_basis(
    cache: &mut SingleRepairBasisCache,
    key: SingleRepairBasisKey,
    generated: Arc<CachedSingleRepairBasis>,
) -> Arc<CachedSingleRepairBasis> {
    if let Some(basis) = cache.bases.get(&key) {
        return Arc::clone(basis);
    }

    if cache.bases.len() >= SINGLE_REPAIR_BASIS_CACHE_CAPACITY
        && let Some(evicted_key) = cache.insertion_order.pop_front()
    {
        cache.bases.remove(&evicted_key);
    }

    cache.insertion_order.push_back(key);
    cache.bases.insert(key, Arc::clone(&generated));
    generated
}

#[cfg(all(test, feature = "std"))]
fn generate_single_repair_basis(
    source_block_symbols: u32,
    missing_isi: usize,
) -> CachedSingleRepairBasis {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    let width = num_intermediate_symbols(source_block_symbols) as usize;
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let h = crate::systematic_constants::num_hdpc_symbols(source_block_symbols) as usize;
    assert!(missing_isi < source_block_symbols as usize);

    let plan = cached_systematic_plan(source_block_symbols);
    let coefficients =
        apply_prepared_systematic_plan_to_basis_coefficients(&plan, s + h + missing_isi);
    assert_eq!(coefficients.len(), width);

    let nonzero_cols = coefficients
        .iter()
        .enumerate()
        .filter_map(|(col, &coefficient)| (coefficient != 0).then_some(col))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    CachedSingleRepairBasis {
        coefficients,
        nonzero_cols,
    }
}

#[cfg(all(test, feature = "std"))]
fn apply_prepared_systematic_plan_to_basis_coefficients(
    plan: &CachedSystematicPlan,
    source_col: usize,
) -> Vec<u8> {
    assert!(source_col < plan.width);

    let mut coefficients = vec![0u8; plan.width];
    coefficients[source_col] = 1;

    for (step_index, step) in plan.forward_steps.iter().enumerate() {
        if let Some(scale) = step.scale {
            coefficients[step.pivot] = (Octet::new(coefficients[step.pivot]) * scale).value();
        }

        let src_value = coefficients[step.pivot];
        if src_value == 0 {
            continue;
        }
        for &(dest, scalar) in plan.forward_dests.slice(step_index) {
            let dest = coefficient_col_index(dest);
            add_scaled_coefficient(&mut coefficients[dest], src_value, scalar);
        }
    }

    move_pivot_coefficients_to_columns(&mut coefficients, &plan.pivot_symbol_cycles);
    match &plan.back_substitution {
        CachedSystematicBackSubstitution::Rows(rows) => {
            for col in (0..plan.width).rev() {
                for &(dependent_col, coefficient) in rows[col].iter() {
                    let dependent_value = coefficients[dependent_col];
                    add_scaled_coefficient(&mut coefficients[col], dependent_value, coefficient);
                }
            }
        }
        CachedSystematicBackSubstitution::Batches(batches) => {
            for src in (0..plan.width).rev() {
                let src_value = coefficients[src];
                if src_value == 0 {
                    continue;
                }
                for &(dest, scalar) in batches[src].iter() {
                    add_scaled_coefficient(&mut coefficients[dest], src_value, scalar);
                }
            }
        }
        CachedSystematicBackSubstitution::FlatBatches(batches) => {
            for src in (0..plan.width).rev() {
                let src_value = coefficients[src];
                if src_value == 0 {
                    continue;
                }
                for &(dest, scalar) in batches.slice(src) {
                    let dest = coefficient_col_index(dest);
                    add_scaled_coefficient(&mut coefficients[dest], src_value, scalar);
                }
            }
        }
    }

    coefficients
}

#[cfg(feature = "std")]
fn repair_source_coefficients_for_entries(
    plan: &CachedSystematicPlan,
    repair_entries: &[usize],
) -> Vec<u8> {
    let mut coefficients = vec![0u8; plan.width];
    for &col in repair_entries {
        coefficients[col] ^= 1;
    }

    apply_back_substitution_to_final_coefficients(plan, &mut coefficients);
    move_final_coefficients_to_pivot_rows(&mut coefficients, &plan.pivot_symbol_cycles);
    apply_forward_steps_to_final_coefficients(plan, &mut coefficients);
    coefficients
}

#[cfg(feature = "std")]
fn generate_repair_source_coefficients(
    plan: &CachedSystematicPlan,
    repair_entries: &[usize],
    s: usize,
    h: usize,
    source_block_symbols: usize,
) -> CachedRepairSourceCoefficients {
    let coefficients = repair_source_coefficients_for_entries(plan, repair_entries);
    let mut source_coefficients = Vec::with_capacity(source_block_symbols);
    let mut nonzero_sources = Vec::new();
    for isi in 0..source_block_symbols {
        let coefficient = coefficients[s + h + isi];
        source_coefficients.push(coefficient);
        if coefficient != 0 {
            nonzero_sources.push((isi, Octet::new(coefficient)));
        }
    }

    CachedRepairSourceCoefficients {
        source_coefficients: source_coefficients.into_boxed_slice(),
        nonzero_sources: nonzero_sources.into_boxed_slice(),
    }
}

#[cfg(feature = "std")]
fn apply_back_substitution_to_final_coefficients(
    plan: &CachedSystematicPlan,
    coefficients: &mut [u8],
) {
    match &plan.back_substitution {
        CachedSystematicBackSubstitution::Rows(rows) => {
            for col in 0..plan.width {
                let dest_value = coefficients[col];
                if dest_value == 0 {
                    continue;
                }
                for &(dependent_col, coefficient) in rows[col].iter().rev() {
                    add_scaled_coefficient(
                        &mut coefficients[dependent_col],
                        dest_value,
                        coefficient,
                    );
                }
            }
        }
        CachedSystematicBackSubstitution::Batches(batches) => {
            for src in 0..plan.width {
                for &(dest, coefficient) in batches[src].iter().rev() {
                    let dest_value = coefficients[dest];
                    if dest_value != 0 {
                        add_scaled_coefficient(&mut coefficients[src], dest_value, coefficient);
                    }
                }
            }
        }
        CachedSystematicBackSubstitution::FlatBatches(batches) => {
            for src in 0..plan.width {
                for &(dest, coefficient) in batches.slice(src).iter().rev() {
                    let dest = coefficient_col_index(dest);
                    let dest_value = coefficients[dest];
                    if dest_value != 0 {
                        add_scaled_coefficient(&mut coefficients[src], dest_value, coefficient);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "std")]
fn move_final_coefficients_to_pivot_rows(coefficients: &mut [u8], cycles: &[Box<[usize]>]) {
    for cycle in cycles {
        let first = coefficients[cycle[0]];
        for index in 0..cycle.len() - 1 {
            coefficients[cycle[index]] = coefficients[cycle[index + 1]];
        }
        coefficients[*cycle.last().expect("cycle is non-empty")] = first;
    }
}

#[cfg(feature = "std")]
fn apply_forward_steps_to_final_coefficients(plan: &CachedSystematicPlan, coefficients: &mut [u8]) {
    for step_index in (0..plan.forward_steps.len()).rev() {
        let step = &plan.forward_steps[step_index];
        for &(dest, scalar) in plan.forward_dests.slice(step_index).iter().rev() {
            let dest = coefficient_col_index(dest);
            let dest_value = coefficients[dest];
            if dest_value != 0 {
                add_scaled_coefficient(&mut coefficients[step.pivot], dest_value, scalar);
            }
        }
        if let Some(scale) = step.scale {
            let pivot_value = coefficients[step.pivot];
            if pivot_value != 0 {
                coefficients[step.pivot] = (Octet::new(pivot_value) * scale).value();
            }
        }
    }
}

#[cfg(feature = "std")]
fn add_scaled_coefficient(dest: &mut u8, src: u8, scalar: Octet) {
    if src == 0 {
        return;
    }
    if scalar == Octet::one() {
        *dest ^= src;
    } else {
        *dest ^= (Octet::new(src) * scalar).value();
    }
}

#[cfg(all(test, feature = "std"))]
fn move_pivot_coefficients_to_columns(coefficients: &mut [u8], cycles: &[Box<[usize]>]) {
    for cycle in cycles {
        let mut scratch = coefficients[cycle[0]];
        for &next in &cycle[1..] {
            core::mem::swap(&mut scratch, &mut coefficients[next]);
        }
        coefficients[cycle[0]] = scratch;
    }
}

#[cfg(feature = "std")]
fn generate_systematic_plan(source_block_symbols: u32) -> CachedSystematicPlan {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    let indices: Vec<u32> = (0..source_block_symbols).collect();
    let (matrix, hdpc_rows) =
        generate_constraint_matrix::<SparseBinaryMatrix>(source_block_symbols, &indices);
    let width = matrix.width();
    let (rows, binary_rows) =
        coefficient_rows_with_binary_flags(&matrix, &hdpc_rows, source_block_symbols);
    prepare_cached_systematic_plan_with_binary_rows(rows, width, binary_rows)
}

#[cfg(feature = "std")]
fn generate_direct_systematic_plan(source_block_symbols: u32) -> DirectSystematicPlan {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    let indices: Vec<u32> = (0..source_block_symbols).collect();
    let (matrix, hdpc_rows) =
        generate_constraint_matrix::<SparseBinaryMatrix>(source_block_symbols, &indices);
    prepare_direct_systematic_plan(&matrix, &hdpc_rows, source_block_symbols)
        .expect("systematic direct plan generation failed")
}

#[cfg(feature = "std")]
pub(crate) fn apply_cached_systematic_plan(source_block_symbols: u32, symbols: &mut SymbolSlab) {
    let plan = cached_systematic_plan(source_block_symbols);
    apply_prepared_systematic_plan(&plan, symbols);
}

#[cfg(all(test, feature = "std"))]
fn prepare_cached_systematic_plan(rows: Vec<CoefficientRow>, width: usize) -> CachedSystematicPlan {
    let binary_rows = coefficient_row_binary_flags(&rows);
    prepare_cached_systematic_plan_with_binary_rows(rows, width, binary_rows)
}

#[cfg(feature = "std")]
fn prepare_cached_systematic_plan_with_binary_rows(
    mut rows: Vec<CoefficientRow>,
    width: usize,
    mut binary_rows: Vec<u8>,
) -> CachedSystematicPlan {
    let height = rows.len();
    assert_eq!(height, width);
    assert_eq!(binary_rows.len(), height);

    let mut row_merge_scratch = Vec::with_capacity(width / 4);
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut singleton_bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut bucket_counts = vec![0usize; width];
    let mut next_in_bucket = vec![NO_BUCKET_ROW; height];
    for (row, coefficients) in rows.iter().enumerate() {
        if let Some(&(col, _)) = coefficients.first() {
            push_counted_row_bucket(
                &rows,
                &mut bucket_heads,
                &mut singleton_bucket_heads,
                &mut bucket_counts,
                &mut next_in_bucket,
                coefficient_col_index(col),
                row,
            );
        }
    }

    let mut forward_steps = Vec::with_capacity(width);
    let mut forward_dest_ranges = Vec::with_capacity(width);
    let mut forward_dest_unit_only = Vec::with_capacity(width);
    let mut forward_dest_entries =
        Vec::with_capacity(width.saturating_mul(SYSTEMATIC_PLAN_FORWARD_DESTS_PER_COL_HINT));
    let mut pivot_for_col = vec![usize::MAX; width];
    let mut is_pivot_row = vec![false; height];

    for (col, pivot_slot) in pivot_for_col.iter_mut().enumerate() {
        let (pivot, pivot_value) = pop_lightest_counted_coefficient_row_bucket(
            &rows,
            &mut bucket_heads,
            &mut singleton_bucket_heads,
            &mut bucket_counts,
            &mut next_in_bucket,
            col,
        )
        .expect("systematic plan matrix must be full rank");
        *pivot_slot = pivot;
        is_pivot_row[pivot] = true;

        let scale = if pivot_value != Octet::one() {
            let scalar = pivot_value.inverse();
            scale_matrix_row(&mut rows[pivot], col, scalar);
            binary_rows[pivot] = 0;
            Some(scalar)
        } else {
            None
        };
        let pivot_is_binary = binary_rows[pivot] != 0;

        let bucket_len = bucket_counts[col];
        let dest_start = forward_dest_entries.len();
        let mut dest_unit_only = true;
        forward_dest_entries.reserve(bucket_len);
        if rows[pivot].len() == 1 {
            while let Some(row) = pop_counted_row_bucket(
                &mut bucket_heads,
                &mut singleton_bucket_heads,
                &mut bucket_counts,
                &mut next_in_bucket,
                col,
            ) {
                debug_assert_ne!(row, pivot);
                let factor = rows[row][0].1;
                remove_matrix_row_entry_at(&mut rows[row], 0);
                if dest_unit_only && factor != Octet::one() {
                    dest_unit_only = false;
                }
                forward_dest_entries.push((coefficient_col(row), factor));

                if let Some(&(next_col, _)) = rows[row].first() {
                    debug_assert!(coefficient_col_index(next_col) > col);
                    push_counted_row_bucket(
                        &rows,
                        &mut bucket_heads,
                        &mut singleton_bucket_heads,
                        &mut bucket_counts,
                        &mut next_in_bucket,
                        coefficient_col_index(next_col),
                        row,
                    );
                }
            }
        } else if width >= CLONE_FREE_PLAN_ELIMINATION_MIN_WIDTH {
            while let Some(row) = pop_counted_row_bucket(
                &mut bucket_heads,
                &mut singleton_bucket_heads,
                &mut bucket_counts,
                &mut next_in_bucket,
                col,
            ) {
                debug_assert_ne!(row, pivot);
                let factor = rows[row][0].1;
                let (pivot_coefficients, row_coefficients) =
                    disjoint_coefficient_rows_mut(&mut rows, pivot, row);
                let binary_merge =
                    pivot_is_binary && binary_rows[row] != 0 && factor == Octet::one();
                if binary_merge && pivot_coefficients.len() <= SHORT_PIVOT_MERGE_MAX_LEN {
                    add_normalized_binary_short_matrix_row(row_coefficients, pivot_coefficients);
                } else {
                    if !binary_merge {
                        binary_rows[row] = 0;
                    }
                    if pivot_coefficients.len() <= SHORT_PIVOT_MERGE_MAX_LEN {
                        add_scaled_normalized_short_matrix_row(
                            row_coefficients,
                            pivot_coefficients,
                            col,
                            factor,
                        );
                    } else {
                        add_scaled_matrix_row(
                            row_coefficients,
                            pivot_coefficients,
                            col,
                            factor,
                            &mut row_merge_scratch,
                        );
                    }
                }
                if dest_unit_only && factor != Octet::one() {
                    dest_unit_only = false;
                }
                forward_dest_entries.push((coefficient_col(row), factor));

                if let Some(&(next_col, _)) = rows[row].first() {
                    debug_assert!(coefficient_col_index(next_col) > col);
                    push_counted_row_bucket(
                        &rows,
                        &mut bucket_heads,
                        &mut singleton_bucket_heads,
                        &mut bucket_counts,
                        &mut next_in_bucket,
                        coefficient_col_index(next_col),
                        row,
                    );
                }
            }
        } else {
            let pivot_coefficients = rows[pivot].clone();
            while let Some(row) = pop_counted_row_bucket(
                &mut bucket_heads,
                &mut singleton_bucket_heads,
                &mut bucket_counts,
                &mut next_in_bucket,
                col,
            ) {
                debug_assert_ne!(row, pivot);
                let factor = rows[row][0].1;
                add_scaled_matrix_row(
                    &mut rows[row],
                    &pivot_coefficients,
                    col,
                    factor,
                    &mut row_merge_scratch,
                );
                if dest_unit_only && factor != Octet::one() {
                    dest_unit_only = false;
                }
                forward_dest_entries.push((coefficient_col(row), factor));

                if let Some(&(next_col, _)) = rows[row].first() {
                    debug_assert!(coefficient_col_index(next_col) > col);
                    push_counted_row_bucket(
                        &rows,
                        &mut bucket_heads,
                        &mut singleton_bucket_heads,
                        &mut bucket_counts,
                        &mut next_in_bucket,
                        coefficient_col_index(next_col),
                        row,
                    );
                }
            }
        }
        forward_dest_ranges.push((dest_start, forward_dest_entries.len()));
        forward_dest_unit_only.push(dest_unit_only);

        forward_steps.push(CachedSystematicForwardStep { pivot, scale });
    }

    for (row, is_pivot) in is_pivot_row.into_iter().enumerate() {
        assert!(
            is_pivot || rows[row].is_empty(),
            "systematic plan matrix has a non-pivot residual row"
        );
    }
    let back_substitution = if width >= FLAT_BACK_SUBSTITUTION_MIN_WIDTH {
        CachedSystematicBackSubstitution::FlatBatches(prepare_flat_back_substitution_batches(
            &rows,
            &pivot_for_col,
            width,
        ))
    } else if width >= BATCHED_BACK_SUBSTITUTION_MIN_WIDTH {
        CachedSystematicBackSubstitution::Batches(prepare_back_substitution_batches(
            &rows,
            &pivot_for_col,
            width,
        ))
    } else {
        let mut rows_by_dest = Vec::with_capacity(width);
        for (col, &pivot) in pivot_for_col.iter().enumerate() {
            let mut dependencies = Vec::new();
            for &(dependent_col, coefficient) in rows[pivot].iter().rev() {
                if coefficient_col_index(dependent_col) <= col {
                    break;
                }
                dependencies.push((coefficient_col_index(dependent_col), coefficient));
            }
            rows_by_dest.push(dependencies.into_boxed_slice());
        }
        CachedSystematicBackSubstitution::Rows(rows_by_dest)
    };

    CachedSystematicPlan {
        forward_steps,
        forward_dests: CachedSystematicSlices {
            ranges: forward_dest_ranges,
            entries: forward_dest_entries,
            unit_only: forward_dest_unit_only,
        },
        pivot_symbol_cycles: pivot_symbol_cycles(&pivot_for_col),
        back_substitution,
        width,
    }
}

#[cfg(feature = "std")]
fn prepare_direct_systematic_plan<M: BinaryMatrix>(
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    source_block_symbols: u32,
) -> Option<DirectSystematicPlan> {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let h = hdpc_rows.height();
    let width = matrix.width();
    let binary_height = matrix.height();
    assert!(width < NO_COEFFICIENT_COLUMN as usize);
    assert!(binary_height < NO_COEFFICIENT_COLUMN as usize);
    assert_eq!(hdpc_rows.width(), width);
    assert!(binary_height >= s);

    let mut rows = matrix.packed_rows();
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut next_in_bucket = vec![NO_BUCKET_ROW; binary_height];
    for row in 0..binary_height {
        if let Some(col) = rows.first_one_at_or_after(row, 0) {
            push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
        }
    }

    let mut pivot_for_col = vec![NO_COEFFICIENT_COLUMN; width];
    let mut is_pivot_row = vec![false; binary_height];
    let mut forward_steps = Vec::with_capacity(width);
    let mut forward_ranges = Vec::with_capacity(width);
    let mut forward_entries = Vec::new();
    for col in 0..width {
        let Some(pivot) =
            pop_lightest_binary_row_bucket(&rows, &mut bucket_heads, &mut next_in_bucket, col)
        else {
            continue;
        };
        pivot_for_col[col] = coefficient_col(pivot);
        is_pivot_row[pivot] = true;

        let dest_start = forward_entries.len();
        while let Some(row) = pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col) {
            rows.xor_suffix(row, pivot, col);
            forward_entries.push(coefficient_col(direct_binary_symbol_index(row, s, h)));

            if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
            }
        }

        if dest_start != forward_entries.len() {
            forward_steps.push(DirectSystematicForwardStep {
                pivot_symbol: coefficient_col(direct_binary_symbol_index(pivot, s, h)),
            });
            forward_ranges.push((dest_start, forward_entries.len()));
        }
    }

    if is_pivot_row.into_iter().any(|is_pivot| !is_pivot) {
        return None;
    }

    let free_cols = pivot_for_col
        .iter()
        .enumerate()
        .filter_map(|(col, &pivot)| {
            (pivot == NO_COEFFICIENT_COLUMN).then_some(coefficient_col(col))
        })
        .collect::<Vec<_>>();
    if free_cols.len() > h {
        return None;
    }

    let mut hdpc_coefficients = dense_hdpc_coefficients(hdpc_rows);
    let mut hdpc_update_pivots = Vec::new();
    let mut hdpc_update_ranges = Vec::new();
    let mut hdpc_update_entries = Vec::new();
    let mut hdpc_update_unit_only = Vec::new();
    let mut back_substitution_counts = vec![0usize; width];
    let mut back_substitution_entries = Vec::new();
    let mut pivot_entries = Vec::new();
    for (col, &pivot) in pivot_for_col.iter().enumerate() {
        if pivot == NO_COEFFICIENT_COLUMN {
            continue;
        }
        let pivot = coefficient_col_index(pivot);
        pivot_entries.clear();
        rows.visit_ones_at_or_after(pivot, col + 1, |entry_col| {
            pivot_entries.push(entry_col);
        });
        for &dependent_col in &pivot_entries {
            back_substitution_counts[dependent_col] += 1;
            back_substitution_entries.push((coefficient_col(dependent_col), coefficient_col(col)));
        }

        let update_start = hdpc_update_entries.len();
        let mut update_unit_only = true;
        for row in 0..h {
            let row_start = row * width;
            let factor = hdpc_coefficients[row_start + col];
            if factor.is_zero() {
                continue;
            }
            hdpc_coefficients[row_start + col] = Octet::zero();
            for &entry_col in &pivot_entries {
                hdpc_coefficients[row_start + entry_col] += factor;
            }
            if factor != Octet::one() {
                update_unit_only = false;
            }
            hdpc_update_entries.push((coefficient_col(row), factor));
        }
        if update_start != hdpc_update_entries.len() {
            hdpc_update_pivots.push(coefficient_col(direct_binary_symbol_index(pivot, s, h)));
            hdpc_update_ranges.push((update_start, hdpc_update_entries.len()));
            hdpc_update_unit_only.push(update_unit_only);
        }
    }

    let hdpc_updates = CachedSystematicSlices {
        ranges: hdpc_update_ranges,
        entries: hdpc_update_entries,
        unit_only: hdpc_update_unit_only,
    };
    let hdpc_free_rows = direct_hdpc_free_rows(hdpc_coefficients, &free_cols, width)?;
    let back_substitution = prepare_direct_back_substitution_batches(
        back_substitution_counts,
        back_substitution_entries,
    );
    let pivot_symbol_moves = direct_pivot_symbol_moves(&pivot_for_col, s, h, width);

    Some(DirectSystematicPlan {
        forward_steps,
        forward_dests: DirectSystematicSlices {
            ranges: forward_ranges,
            entries: forward_entries,
        },
        hdpc_update_pivots: hdpc_update_pivots.into_boxed_slice(),
        hdpc_updates,
        hdpc_free_rows,
        free_cols: free_cols.into_boxed_slice(),
        pivot_symbol_moves,
        back_substitution,
        width,
        s,
        h,
    })
}

#[cfg(feature = "std")]
fn prepare_back_substitution_batches(
    rows: &[CoefficientRow],
    pivot_for_col: &[usize],
    width: usize,
) -> Vec<Box<[(usize, Octet)]>> {
    let mut counts = vec![0usize; width];
    for (col, &pivot) in pivot_for_col.iter().enumerate() {
        for &(dependent_col, _) in rows[pivot].iter().rev() {
            let dependent_col = coefficient_col_index(dependent_col);
            if dependent_col <= col {
                break;
            }
            counts[dependent_col] += 1;
        }
    }

    let mut batches = counts
        .into_iter()
        .map(Vec::with_capacity)
        .collect::<Vec<_>>();
    for (col, &pivot) in pivot_for_col.iter().enumerate() {
        for &(dependent_col, coefficient) in rows[pivot].iter().rev() {
            let dependent_col = coefficient_col_index(dependent_col);
            if dependent_col <= col {
                break;
            }
            batches[dependent_col].push((col, coefficient));
        }
    }

    batches.into_iter().map(Vec::into_boxed_slice).collect()
}

#[cfg(feature = "std")]
fn prepare_flat_back_substitution_batches(
    rows: &[CoefficientRow],
    pivot_for_col: &[usize],
    width: usize,
) -> CachedSystematicSlices {
    let mut counts = vec![0usize; width];
    for (col, &pivot) in pivot_for_col.iter().enumerate() {
        for &(dependent_col, _) in rows[pivot].iter().rev() {
            let dependent_col = coefficient_col_index(dependent_col);
            if dependent_col <= col {
                break;
            }
            counts[dependent_col] += 1;
        }
    }

    let (ranges, mut offsets, mut entries, entries_len) = slices_from_counts(counts);
    let mut unit_only = vec![true; width];
    for (col, &pivot) in pivot_for_col.iter().enumerate() {
        for &(dependent_col, coefficient) in rows[pivot].iter().rev() {
            let dependent_col = coefficient_col_index(dependent_col);
            if dependent_col <= col {
                break;
            }
            if unit_only[dependent_col] && coefficient != Octet::one() {
                unit_only[dependent_col] = false;
            }
            let offset = offsets[dependent_col];
            debug_assert!(offset < entries_len);
            // The first pass counted this slot, and each dependent column advances
            // monotonically inside its assigned range.
            unsafe {
                entries
                    .as_mut_ptr()
                    .add(offset)
                    .write((coefficient_col(col), coefficient));
            }
            offsets[dependent_col] += 1;
        }
    }

    for (offset, &(_, end)) in offsets.iter().zip(ranges.iter()) {
        debug_assert_eq!(*offset, end);
    }
    // All counted slots were initialized by the second pass above.
    unsafe {
        entries.set_len(entries_len);
    }

    CachedSystematicSlices {
        ranges,
        entries,
        unit_only,
    }
}

#[cfg(feature = "std")]
fn prepare_direct_back_substitution_batches(
    counts: Vec<usize>,
    back_substitution_entries: Vec<(CoefficientColumn, CoefficientColumn)>,
) -> DirectSystematicSlices {
    let (ranges, mut offsets, mut entries, entries_len) = direct_slices_from_counts(counts);
    for (dependent_col, col) in back_substitution_entries {
        let dependent_col = coefficient_col_index(dependent_col);
        let offset = offsets[dependent_col];
        debug_assert!(offset < entries_len);
        unsafe {
            entries.as_mut_ptr().add(offset).write(col);
        }
        offsets[dependent_col] += 1;
    }

    for (offset, &(_, end)) in offsets.iter().zip(ranges.iter()) {
        debug_assert_eq!(*offset, end);
    }
    unsafe {
        entries.set_len(entries_len);
    }

    DirectSystematicSlices { ranges, entries }
}

#[cfg(feature = "std")]
fn direct_hdpc_free_rows(
    hdpc_coefficients: Vec<Octet>,
    free_cols: &[CoefficientColumn],
    width: usize,
) -> Option<DirectSystematicFreeRows> {
    assert_eq!(hdpc_coefficients.len() % width, 0);
    let h = hdpc_coefficients.len() / width;
    let mut free_index_by_col = vec![usize::MAX; width];
    for (index, &col) in free_cols.iter().enumerate() {
        free_index_by_col[coefficient_col_index(col)] = index;
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
            free_row.push((coefficient_col(free_index), value));
        }
        free_rows.push(free_row.into_boxed_slice());
    }
    Some(free_rows)
}

#[cfg(feature = "std")]
fn slices_from_counts(counts: Vec<usize>) -> CachedSystematicSliceParts {
    let mut ranges = Vec::with_capacity(counts.len());
    let mut next_start = 0usize;
    for count in counts {
        let start = next_start;
        next_start += count;
        ranges.push((start, next_start));
    }
    let offsets = ranges.iter().map(|&(start, _)| start).collect::<Vec<_>>();
    let entries = Vec::with_capacity(next_start);
    (ranges, offsets, entries, next_start)
}

#[cfg(feature = "std")]
fn direct_slices_from_counts(counts: Vec<usize>) -> DirectSystematicSliceParts {
    let mut ranges = Vec::with_capacity(counts.len());
    let mut next_start = 0usize;
    for count in counts {
        let start = next_start;
        next_start += count;
        ranges.push((start, next_start));
    }
    let offsets = ranges.iter().map(|&(start, _)| start).collect::<Vec<_>>();
    let entries = Vec::with_capacity(next_start);
    (ranges, offsets, entries, next_start)
}

#[cfg(feature = "std")]
fn apply_prepared_systematic_plan(plan: &CachedSystematicPlan, symbols: &mut SymbolSlab) {
    assert_eq!(symbols.len(), plan.width);

    for (step_index, step) in plan.forward_steps.iter().enumerate() {
        if let Some(scale) = step.scale {
            mulassign_scalar(symbols.get_mut(step.pivot), &scale);
        }
        fused_addassign_cached_symbol_batch(
            symbols,
            step.pivot,
            plan.forward_dests.slice(step_index),
            plan.forward_dests.is_unit_only(step_index),
        );
    }

    move_pivot_symbols_to_columns(symbols, &plan.pivot_symbol_cycles);
    match &plan.back_substitution {
        CachedSystematicBackSubstitution::Rows(rows) => {
            for col in (0..plan.width).rev() {
                for &(dependent_col, coefficient) in &rows[col] {
                    let (dependent_symbol, dest_symbol) =
                        symbols.get_disjoint_mut(dependent_col, col);
                    fused_addassign_mul_scalar(dest_symbol, dependent_symbol, &coefficient);
                }
            }
        }
        CachedSystematicBackSubstitution::Batches(batches) => {
            for src in (0..plan.width).rev() {
                fused_addassign_symbol_batch(symbols, src, &batches[src]);
            }
        }
        CachedSystematicBackSubstitution::FlatBatches(batches) => {
            for src in (0..plan.width).rev() {
                fused_addassign_cached_symbol_batch(
                    symbols,
                    src,
                    batches.slice(src),
                    batches.is_unit_only(src),
                );
            }
        }
    }
}

#[cfg(feature = "std")]
fn apply_prepared_direct_systematic_plan(plan: &DirectSystematicPlan, symbols: &mut SymbolSlab) {
    assert_eq!(symbols.len(), plan.width);

    let symbol_size = symbols.symbol_size();
    let add_assign_path = AddAssignFastPath::new(symbol_size);

    for (step_index, step) in plan.forward_steps.iter().enumerate() {
        addassign_direct_symbol_batch(
            symbols,
            coefficient_col_index(step.pivot_symbol),
            plan.forward_dests.slice(step_index),
            add_assign_path,
        );
    }

    let mut hdpc_symbols = SymbolSlab::with_zeros(plan.h, symbol_size);
    for row in 0..plan.h {
        hdpc_symbols
            .get_mut(row)
            .copy_from_slice(symbols.get(plan.s + row));
    }
    for (update_index, &pivot) in plan.hdpc_update_pivots.iter().enumerate() {
        let pivot_symbol = symbols.get(coefficient_col_index(pivot));
        for &(row, factor) in plan.hdpc_updates.slice(update_index) {
            fused_addassign_mul_scalar(
                hdpc_symbols.get_mut(coefficient_col_index(row)),
                pivot_symbol,
                &factor,
            );
        }
    }

    let free_values = solve_prepared_hdpc_free_variables(
        &plan.hdpc_free_rows,
        hdpc_symbols,
        plan.free_cols.len(),
        symbol_size,
    )
    .expect("prepared direct systematic HDPC solve failed");

    move_direct_pivot_symbols_to_columns(plan, symbols);
    for (free_index, &col) in plan.free_cols.iter().enumerate() {
        symbols
            .get_mut(coefficient_col_index(col))
            .copy_from_slice(free_values.get(free_index));
    }
    for src in (0..plan.width).rev() {
        addassign_direct_symbol_batch(
            symbols,
            src,
            plan.back_substitution.slice(src),
            add_assign_path,
        );
    }
}

#[cfg(feature = "std")]
fn direct_binary_symbol_index(row: usize, s: usize, h: usize) -> usize {
    if row < s { row } else { row + h }
}

#[cfg(feature = "std")]
fn move_direct_pivot_symbols_to_columns(plan: &DirectSystematicPlan, symbols: &mut SymbolSlab) {
    let mut scratch = vec![0u8; symbols.symbol_size()];
    for symbol_move in &plan.pivot_symbol_moves {
        let (&source, rest) = symbol_move
            .split_first()
            .expect("direct symbol moves are non-empty");
        let (&dest, swap_positions) = rest
            .split_last()
            .expect("direct symbol moves include a destination");

        scratch.copy_from_slice(symbols.get(source));
        for &position in swap_positions {
            scratch.swap_with_slice(symbols.get_mut(position));
        }
        symbols.get_mut(dest).copy_from_slice(&scratch);
    }
}

#[cfg(feature = "std")]
fn direct_pivot_symbol_moves(
    pivot_for_col: &[CoefficientColumn],
    s: usize,
    h: usize,
    width: usize,
) -> Vec<Box<[usize]>> {
    let mut dest_for_source = vec![NO_BUCKET_ROW; width];
    for (col, &pivot) in pivot_for_col.iter().enumerate() {
        if pivot == NO_COEFFICIENT_COLUMN {
            continue;
        }
        let source = direct_binary_symbol_index(coefficient_col_index(pivot), s, h);
        debug_assert_eq!(dest_for_source[source], NO_BUCKET_ROW);
        dest_for_source[source] = col;
    }

    let mut visited = vec![false; width];
    let mut moves = Vec::new();
    for source in 0..width {
        if dest_for_source[source] == NO_BUCKET_ROW || visited[source] {
            continue;
        }

        let mut symbol_move = vec![source];
        let mut current = source;
        loop {
            visited[current] = true;
            let dest = dest_for_source[current];
            debug_assert_ne!(dest, NO_BUCKET_ROW);
            symbol_move.push(dest);
            if dest_for_source[dest] != NO_BUCKET_ROW && !visited[dest] {
                current = dest;
            } else {
                break;
            }
        }

        if symbol_move.len() != 2 || symbol_move[0] != symbol_move[1] {
            moves.push(symbol_move.into_boxed_slice());
        }
    }
    moves
}

#[cfg(feature = "std")]
fn solve_prepared_hdpc_free_variables(
    rows: &[Box<[(CoefficientColumn, Octet)]>],
    hdpc_symbols: SymbolSlab,
    free_count: usize,
    symbol_size: usize,
) -> Option<SymbolSlab> {
    if free_count == 0 {
        if rows.iter().any(|row| !row.is_empty()) {
            return None;
        }
        for row in 0..hdpc_symbols.len() {
            if !symbol_is_zero(hdpc_symbols.get(row)) {
                return None;
            }
        }
        return Some(SymbolSlab::with_zeros(0, symbol_size));
    }

    let rows = rows.iter().map(|row| row.to_vec()).collect::<Vec<_>>();
    solve_without_recording(rows, free_count, hdpc_symbols).0
}

#[cfg(feature = "std")]
fn fused_addassign_cached_symbol_batch(
    symbols: &mut SymbolSlab,
    src: usize,
    dests: &[(CoefficientColumn, Octet)],
    unit_only: bool,
) {
    if dests.is_empty() {
        return;
    }

    let symbol_size = symbols.symbol_size();
    let bytes = symbols.as_mut_bytes();
    let src_start = src * symbol_size;
    assert!(src_start + symbol_size <= bytes.len());
    let src_ptr = unsafe { bytes.as_ptr().add(src_start) };
    let src_symbol = unsafe { core::slice::from_raw_parts(src_ptr, symbol_size) };
    if bytes_are_zero(src_symbol) {
        return;
    }
    let bytes_ptr = bytes.as_mut_ptr();

    if unit_only {
        for &(dest, _) in dests {
            let dest = coefficient_col_index(dest);
            assert_ne!(dest, src);
            let dest_start = dest * symbol_size;
            assert!(dest_start + symbol_size <= bytes.len());
            unsafe {
                let dest_symbol =
                    core::slice::from_raw_parts_mut(bytes_ptr.add(dest_start), symbol_size);
                add_assign(dest_symbol, src_symbol);
            }
        }
        return;
    }

    for &(dest, scalar) in dests {
        let dest = coefficient_col_index(dest);
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

#[cfg(feature = "std")]
fn addassign_direct_symbol_batch(
    symbols: &mut SymbolSlab,
    src: usize,
    dests: &[CoefficientColumn],
    add_assign_path: AddAssignFastPath,
) {
    if dests.is_empty() {
        return;
    }

    let symbol_size = symbols.symbol_size();
    let bytes = symbols.as_mut_bytes();
    let src_start = src * symbol_size;
    assert!(src_start + symbol_size <= bytes.len());
    let src_ptr = unsafe { bytes.as_ptr().add(src_start) };
    let src_symbol = unsafe { core::slice::from_raw_parts(src_ptr, symbol_size) };
    let bytes_ptr = bytes.as_mut_ptr();

    for &dest in dests {
        let dest = coefficient_col_index(dest);
        assert_ne!(dest, src);
        let dest_start = dest * symbol_size;
        assert!(dest_start + symbol_size <= bytes.len());
        unsafe {
            let dest_symbol =
                core::slice::from_raw_parts_mut(bytes_ptr.add(dest_start), symbol_size);
            add_assign_path.apply(dest_symbol, src_symbol);
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

    symbol_cycles_from_destinations(&dest_for_pivot)
}

#[cfg(feature = "std")]
fn hybrid_output_symbol_cycles(
    pivots: &[(usize, usize)],
    free_cols: &[usize],
    s: usize,
    h: usize,
    width: usize,
) -> Option<Vec<Box<[usize]>>> {
    if free_cols.len() != h {
        return None;
    }

    let mut dest_for_source = vec![usize::MAX; width];
    for &(col, pivot) in pivots {
        let source = mapped_binary_symbol_row(pivot, s, h);
        if dest_for_source[source] != usize::MAX {
            return None;
        }
        dest_for_source[source] = col;
    }
    for (free_index, &col) in free_cols.iter().enumerate() {
        let source = s + free_index;
        if source >= width || dest_for_source[source] != usize::MAX {
            return None;
        }
        dest_for_source[source] = col;
    }
    dest_for_source
        .iter()
        .all(|&dest| dest != usize::MAX)
        .then(|| symbol_cycles_from_destinations(&dest_for_source))
}

#[cfg(feature = "std")]
fn symbol_cycles_from_destinations(dest_for_source: &[usize]) -> Vec<Box<[usize]>> {
    let mut cycles = Vec::new();
    let mut visited = vec![false; dest_for_source.len()];
    for start in 0..dest_for_source.len() {
        if visited[start] {
            continue;
        }

        let mut cycle = Vec::new();
        let mut current = start;
        loop {
            visited[current] = true;
            cycle.push(current);
            let next = dest_for_source[current];
            assert!(next < dest_for_source.len());
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

pub(crate) fn apply_direct_systematic_solve(source_block_symbols: u32, symbols: &mut SymbolSlab) {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    #[cfg(feature = "std")]
    {
        if let Some(plan) = cached_hybrid_systematic_plan_if_present(source_block_symbols) {
            apply_cached_hybrid_systematic_plan(&plan, symbols);
            return;
        }
        let plan = cached_direct_systematic_plan(source_block_symbols);
        apply_prepared_direct_systematic_plan(&plan, symbols);
        return;
    }

    #[cfg(not(feature = "std"))]
    {
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
}

#[cfg(feature = "std")]
fn apply_cached_hybrid_systematic_plan(
    plan: &CachedHybridSystematicPlan,
    symbols: &mut SymbolSlab,
) {
    if plan.width >= IN_PLACE_HYBRID_REPLAY_MIN_WIDTH {
        apply_cached_hybrid_systematic_plan_in_place(plan, symbols);
    } else {
        apply_cached_hybrid_systematic_plan_with_binary_slab(plan, symbols);
    }
}

#[cfg(feature = "std")]
fn apply_cached_hybrid_systematic_plan_with_binary_slab(
    plan: &CachedHybridSystematicPlan,
    symbols: &mut SymbolSlab,
) {
    assert_eq!(symbols.len(), plan.width);

    let symbol_size = symbols.symbol_size();
    let binary_height = plan.width - plan.h;
    let mut binary_symbols = SymbolSlab::with_zeros(binary_height, symbol_size);
    let symbol_bytes = symbols.as_bytes();
    let low_binary_bytes = plan.s * symbol_size;
    binary_symbols.copy_block_from(0, &symbol_bytes[..low_binary_bytes]);
    let high_binary_start = (plan.s + plan.h) * symbol_size;
    binary_symbols.copy_block_from(plan.s, &symbol_bytes[high_binary_start..]);

    for (step_index, &(_, pivot)) in plan.pivots.iter().enumerate() {
        fused_addassign_cached_symbol_batch(
            &mut binary_symbols,
            pivot,
            plan.binary_forward_dests.slice(step_index),
            true,
        );
    }

    let mut hdpc_symbols = SymbolSlab::with_zeros(plan.h, symbol_size);
    let hdpc_start = plan.s * symbol_size;
    let hdpc_end = hdpc_start + plan.h * symbol_size;
    hdpc_symbols.copy_block_from(0, &symbol_bytes[hdpc_start..hdpc_end]);
    for step in &plan.hdpc_symbol_steps {
        fused_addassign_mul_scalar(
            hdpc_symbols.get_mut(step.row),
            binary_symbols.get(step.pivot),
            &step.factor,
        );
    }

    let free_values = if plan.free_cols.is_empty() {
        assert!(
            (0..plan.h).all(|row| symbol_is_zero(hdpc_symbols.get(row))),
            "cached hybrid systematic solve has inconsistent HDPC rows"
        );
        SymbolSlab::with_zeros(0, symbol_size)
    } else {
        solve_without_recording(plan.free_rows.clone(), plan.free_cols.len(), hdpc_symbols)
            .0
            .expect("cached hybrid systematic free-column solve failed")
    };

    let mut decoded = SymbolSlab::with_zeros(plan.width, symbol_size);
    for (free_index, &col) in plan.free_cols.iter().enumerate() {
        decoded
            .get_mut(col)
            .copy_from_slice(free_values.get(free_index));
    }
    for &(col, pivot) in plan.pivots.iter() {
        decoded
            .get_mut(col)
            .copy_from_slice(binary_symbols.get(pivot));
    }
    for src in (0..plan.width).rev() {
        fused_addassign_cached_symbol_batch(
            &mut decoded,
            src,
            plan.back_substitution.slice(src),
            true,
        );
    }

    symbols.copy_block_from(0, decoded.as_bytes());
}

#[cfg(feature = "std")]
fn apply_cached_hybrid_systematic_plan_in_place(
    plan: &CachedHybridSystematicPlan,
    symbols: &mut SymbolSlab,
) {
    assert_eq!(symbols.len(), plan.width);

    let symbol_size = symbols.symbol_size();
    for (step_index, &(_, pivot)) in plan.pivots.iter().enumerate() {
        addassign_mapped_binary_symbol_batch(
            symbols,
            pivot,
            plan.binary_forward_dests.slice(step_index),
            plan.s,
            plan.h,
        );
    }

    for step in &plan.hdpc_symbol_steps {
        let src = mapped_binary_symbol_row(step.pivot, plan.s, plan.h);
        let dest = plan.s + step.row;
        let (src_symbol, dest_symbol) = symbols.get_disjoint_mut(src, dest);
        fused_addassign_mul_scalar(dest_symbol, src_symbol, &step.factor);
    }

    let free_values = if plan.free_cols.is_empty() {
        assert!(
            (0..plan.h).all(|row| symbol_is_zero(symbols.get(plan.s + row))),
            "cached hybrid systematic solve has inconsistent HDPC rows"
        );
        SymbolSlab::with_zeros(0, symbol_size)
    } else {
        let mut hdpc_symbols = SymbolSlab::with_zeros(plan.h, symbol_size);
        for row in 0..plan.h {
            hdpc_symbols
                .get_mut(row)
                .copy_from_slice(symbols.get(plan.s + row));
        }
        solve_without_recording(plan.free_rows.clone(), plan.free_cols.len(), hdpc_symbols)
            .0
            .expect("cached hybrid systematic free-column solve failed")
    };

    if let Some(output_symbol_cycles) = &plan.output_symbol_cycles {
        for (free_index, _) in plan.free_cols.iter().enumerate() {
            symbols
                .get_mut(plan.s + free_index)
                .copy_from_slice(free_values.get(free_index));
        }
        move_pivot_symbols_to_columns(symbols, output_symbol_cycles);
        for src in (0..plan.width).rev() {
            fused_addassign_cached_symbol_batch(
                symbols,
                src,
                plan.back_substitution.slice(src),
                true,
            );
        }
        return;
    }

    let mut decoded = SymbolSlab::with_zeros(plan.width, symbol_size);
    for (free_index, &col) in plan.free_cols.iter().enumerate() {
        decoded
            .get_mut(col)
            .copy_from_slice(free_values.get(free_index));
    }
    for &(col, pivot) in plan.pivots.iter() {
        let pivot = mapped_binary_symbol_row(pivot, plan.s, plan.h);
        decoded.get_mut(col).copy_from_slice(symbols.get(pivot));
    }
    for src in (0..plan.width).rev() {
        fused_addassign_cached_symbol_batch(
            &mut decoded,
            src,
            plan.back_substitution.slice(src),
            true,
        );
    }

    symbols.copy_block_from(0, decoded.as_bytes());
}

#[cfg(feature = "std")]
fn mapped_binary_symbol_row(row: usize, s: usize, h: usize) -> usize {
    if row < s { row } else { row + h }
}

#[cfg(feature = "std")]
fn addassign_mapped_binary_symbol_batch(
    symbols: &mut SymbolSlab,
    src: usize,
    dests: &[(CoefficientColumn, Octet)],
    s: usize,
    h: usize,
) {
    let src = mapped_binary_symbol_row(src, s, h);
    if dests.is_empty() {
        return;
    }

    let symbol_size = symbols.symbol_size();
    let bytes = symbols.as_mut_bytes();
    let src_start = src * symbol_size;
    assert!(src_start + symbol_size <= bytes.len());
    let src_ptr = unsafe { bytes.as_ptr().add(src_start) };
    let src_symbol = unsafe { core::slice::from_raw_parts(src_ptr, symbol_size) };
    if bytes_are_zero(src_symbol) {
        return;
    }
    let bytes_ptr = bytes.as_mut_ptr();

    for &(dest, _) in dests {
        let dest = mapped_binary_symbol_row(coefficient_col_index(dest), s, h);
        assert_ne!(dest, src);
        let dest_start = dest * symbol_size;
        assert!(dest_start + symbol_size <= bytes.len());
        unsafe {
            let dest_symbol =
                core::slice::from_raw_parts_mut(bytes_ptr.add(dest_start), symbol_size);
            add_assign(dest_symbol, src_symbol);
        }
    }
}

fn coefficient_rows<M: BinaryMatrix>(
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    source_block_symbols: u32,
) -> Vec<CoefficientRow> {
    coefficient_rows_with_binary_flags(matrix, hdpc_rows, source_block_symbols).0
}

fn coefficient_rows_with_binary_flags<M: BinaryMatrix>(
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    source_block_symbols: u32,
) -> (Vec<CoefficientRow>, Vec<u8>) {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let total_rows = matrix.height() + hdpc_rows.height();
    let mut rows = Vec::with_capacity(total_rows);
    let mut binary_rows = Vec::with_capacity(total_rows);
    for row in 0..s {
        rows.push(copy_binary_row(matrix, row));
        binary_rows.push(1);
    }
    for row in 0..hdpc_rows.height() {
        rows.push(copy_octet_row(hdpc_rows, row));
        binary_rows.push(0);
    }
    for row in s..matrix.height() {
        rows.push(copy_binary_row(matrix, row));
        binary_rows.push(1);
    }
    debug_assert_eq!(rows.len(), total_rows);
    debug_assert_eq!(binary_rows.len(), total_rows);

    (rows, binary_rows)
}

fn coefficient_rows_from_binary_entries(
    binary_entries: Vec<Vec<usize>>,
    matrix_height: usize,
    hdpc_rows: &DenseOctetMatrix,
    source_block_symbols: u32,
) -> (Vec<CoefficientRow>, Vec<u8>) {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    assert_eq!(binary_entries.len(), matrix_height);
    assert!(matrix_height >= s);

    let total_rows = matrix_height + hdpc_rows.height();
    let mut rows = Vec::with_capacity(total_rows);
    let mut binary_rows = Vec::with_capacity(total_rows);
    let mut binary_entries = binary_entries.into_iter();
    for _ in 0..s {
        let entries = binary_entries
            .next()
            .expect("LDPC row entries must be present");
        rows.push(coefficient_row_from_binary_entries(entries));
        binary_rows.push(1);
    }
    for row in 0..hdpc_rows.height() {
        rows.push(copy_octet_row(hdpc_rows, row));
        binary_rows.push(0);
    }
    for entries in binary_entries {
        rows.push(coefficient_row_from_binary_entries(entries));
        binary_rows.push(1);
    }
    debug_assert_eq!(rows.len(), total_rows);
    debug_assert_eq!(binary_rows.len(), total_rows);

    (rows, binary_rows)
}

fn coefficient_row_from_binary_entries(entries: Vec<usize>) -> CoefficientRow {
    entries
        .into_iter()
        .map(|col| (coefficient_col(col), Octet::one()))
        .collect()
}

#[cfg(all(test, feature = "std"))]
fn coefficient_row_binary_flags(rows: &[CoefficientRow]) -> Vec<u8> {
    rows.iter()
        .map(|row| u8::from(row.iter().all(|&(_, value)| value == Octet::one())))
        .collect()
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
        for (col, &coefficient) in hdpc_rows.row(row).iter().enumerate() {
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
#[cfg(feature = "std")]
fn prepare_cached_hybrid_systematic_plan<M: BinaryMatrix>(
    source_block_symbols: u32,
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
) -> Option<CachedHybridSystematicPlan> {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let h = hdpc_rows.height();
    let width = matrix.width();
    let binary_height = matrix.height();
    if binary_height + h != width
        || hdpc_rows.width() != width
        || binary_height < s
        || width > SQUARE_HYBRID_MAX_WIDTH
    {
        return None;
    }

    let use_weighted_buckets = width >= IN_PLACE_HYBRID_REPLAY_MIN_WIDTH;
    let (mut rows, mut row_weights) = if use_weighted_buckets {
        matrix.packed_rows_with_row_weights()
    } else {
        (matrix.packed_rows(), Vec::new())
    };
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut next_in_bucket = vec![NO_BUCKET_ROW; binary_height];
    for row in 0..binary_height {
        if let Some(col) = rows.first_one_at_or_after(row, 0) {
            push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
        }
    }

    let mut pivot_for_col = vec![None; width];
    let mut is_pivot_row = vec![false; binary_height];
    let mut pivots = Vec::with_capacity(width);
    let mut binary_forward_ranges = Vec::with_capacity(width);
    let mut binary_forward_entries = Vec::new();
    let mut binary_forward_unit_only = Vec::with_capacity(width);
    for col in 0..width {
        let pivot = if use_weighted_buckets {
            pop_lightest_weighted_binary_row_bucket(
                &row_weights,
                &mut bucket_heads,
                &mut next_in_bucket,
                col,
            )
        } else {
            pop_lightest_binary_row_bucket(&rows, &mut bucket_heads, &mut next_in_bucket, col)
        };
        let Some(pivot) = pivot else {
            continue;
        };
        pivot_for_col[col] = Some(pivot);
        is_pivot_row[pivot] = true;
        pivots.push((col, pivot));

        let dest_start = binary_forward_entries.len();
        while let Some(row) = pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col) {
            if use_weighted_buckets {
                row_weights[row] = rows.xor_suffix_count_ones(row, pivot, col);
            } else {
                rows.xor_suffix(row, pivot, col);
            }
            binary_forward_entries.push((coefficient_col(row), Octet::one()));

            if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
            }
        }
        binary_forward_ranges.push((dest_start, binary_forward_entries.len()));
        binary_forward_unit_only.push(true);
    }

    for (row, is_pivot) in is_pivot_row.into_iter().enumerate() {
        if !is_pivot && !rows.is_zero(row) {
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

    let mut hdpc_coefficients = dense_hdpc_coefficients(hdpc_rows);
    let mut hdpc_symbol_steps = Vec::new();
    for &(col, pivot) in &pivots {
        for row in 0..h {
            let row_start = row * width;
            let factor = hdpc_coefficients[row_start + col];
            if factor.is_zero() {
                continue;
            }
            hdpc_symbol_steps.push(HybridHdpcSymbolStep { row, pivot, factor });
            rows.visit_ones_at_or_after(pivot, col, |entry_col| {
                hdpc_coefficients[row_start + entry_col] += factor;
            });
        }
    }

    let free_rows = hybrid_hdpc_free_rows(&hdpc_coefficients, &free_cols, width)?;
    let back_substitution = prepare_binary_flat_back_substitution_batches(&rows, &pivots, width);
    let output_symbol_cycles = (width >= IN_PLACE_HYBRID_REPLAY_MIN_WIDTH)
        .then(|| hybrid_output_symbol_cycles(&pivots, &free_cols, s, h, width))
        .flatten();

    Some(CachedHybridSystematicPlan {
        binary_forward_dests: CachedSystematicSlices {
            ranges: binary_forward_ranges,
            entries: binary_forward_entries,
            unit_only: binary_forward_unit_only,
        },
        hdpc_symbol_steps,
        free_cols: free_cols.into_boxed_slice(),
        free_rows,
        pivots: pivots.into_boxed_slice(),
        back_substitution,
        output_symbol_cycles,
        s,
        h,
        width,
    })
}

#[cfg(feature = "std")]
fn prepare_binary_flat_back_substitution_batches(
    rows: &PackedBinaryRows,
    pivots: &[(usize, usize)],
    width: usize,
) -> CachedSystematicSlices {
    let mut counts = vec![0usize; width];
    for &(col, pivot) in pivots {
        rows.visit_ones_at_or_after(pivot, col + 1, |dependent_col| {
            counts[dependent_col] += 1;
        });
    }

    let (ranges, mut offsets, mut entries, entries_len) = slices_from_counts(counts);
    let unit_only = vec![true; width];
    for &(col, pivot) in pivots {
        rows.visit_ones_at_or_after(pivot, col + 1, |dependent_col| {
            let offset = offsets[dependent_col];
            debug_assert!(offset < entries_len);
            // The first pass counted this slot, and each dependent column advances
            // monotonically inside its assigned range.
            unsafe {
                entries
                    .as_mut_ptr()
                    .add(offset)
                    .write((coefficient_col(col), Octet::one()));
            }
            offsets[dependent_col] += 1;
        });
    }

    for (offset, &(_, end)) in offsets.iter().zip(ranges.iter()) {
        debug_assert_eq!(*offset, end);
    }
    // All counted slots were initialized by the second pass above.
    unsafe {
        entries.set_len(entries_len);
    }

    CachedSystematicSlices {
        ranges,
        entries,
        unit_only,
    }
}

#[cfg(feature = "std")]
fn hybrid_hdpc_free_rows(
    hdpc_coefficients: &[Octet],
    free_cols: &[usize],
    width: usize,
) -> Option<Vec<CoefficientRow>> {
    let h = hdpc_coefficients.len() / width;
    assert_eq!(hdpc_coefficients.len(), h * width);

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
            free_row.push((coefficient_col(free_index), value));
        }
        free_rows.push(free_row);
    }

    Some(free_rows)
}

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
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut next_in_bucket = vec![NO_BUCKET_ROW; binary_height];
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
    matrix.as_slice().to_vec()
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
            free_row.push((coefficient_col(free_index), value));
        }
        free_rows.push(free_row);
    }

    solve_without_recording(free_rows, free_cols.len(), hdpc_symbols).0
}

fn copy_binary_row<M: BinaryMatrix>(matrix: &M, row: usize) -> CoefficientRow {
    let mut result = Vec::new();
    matrix.visit_row_entries(row, |col| result.push((coefficient_col(col), Octet::one())));
    result
}

fn copy_octet_row(matrix: &DenseOctetMatrix, row: usize) -> CoefficientRow {
    let mut result = Vec::new();
    for (col, &value) in matrix.row(row).iter().enumerate() {
        if !value.is_zero() {
            result.push((coefficient_col(col), value));
        }
    }
    result
}

fn is_full_systematic_planning_matrix<M: BinaryMatrix>(
    matrix: &M,
    source_block_symbols: u32,
) -> bool {
    let k_prime = extended_source_block_symbols(source_block_symbols);
    if matrix.systematic_source_block_symbols() == Some(k_prime) {
        return true;
    }

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
enum SingleRepairSystematicRowLayout {
    Contiguous,
    Explicit(Vec<(usize, usize)>),
}

#[cfg(feature = "std")]
struct SingleRepairSystematicRows {
    missing_isi: usize,
    repair_matrix_row: usize,
    repair_isi: Option<u32>,
    systematic_rows: SingleRepairSystematicRowLayout,
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

    let rows = tagged_single_repair_systematic_rows(matrix, k_prime).or_else(|| {
        classify_single_repair_systematic_rows(matrix, source_block_symbols, s, k_prime)
    })?;
    if rows.missing_isi >= source_block_symbols as usize {
        return None;
    }

    let symbol_size = symbols.symbol_size();
    let plan = cached_systematic_plan(k_prime as u32);
    let repair_coefficients = if let Some(repair_isi) = rows.repair_isi {
        cached_repair_source_coefficients(source_block_symbols, repair_isi, &plan, s, h)
    } else {
        let repair_entries = matrix.row_entries(rows.repair_matrix_row);
        if repair_entries.is_empty() {
            return None;
        }
        Arc::new(generate_repair_source_coefficients(
            &plan,
            &repair_entries,
            s,
            h,
            source_block_symbols as usize,
        ))
    };
    let missing_coefficient = Octet::new(repair_coefficients.source_coefficients[rows.missing_isi]);
    if missing_coefficient.is_zero() {
        return None;
    }

    let mut missing_symbol = symbols.get(rows.repair_matrix_row + h).to_vec();
    match &rows.systematic_rows {
        SingleRepairSystematicRowLayout::Contiguous => {
            for &(isi, coefficient) in repair_coefficients.nonzero_sources.iter() {
                if isi == rows.missing_isi {
                    continue;
                }
                let matrix_row = s + if isi < rows.missing_isi { isi } else { isi - 1 };
                fused_addassign_mul_scalar(
                    &mut missing_symbol,
                    symbols.get(matrix_row + h),
                    &coefficient,
                );
            }
        }
        SingleRepairSystematicRowLayout::Explicit(systematic_rows) => {
            for &(isi, matrix_row) in systematic_rows {
                if isi >= source_block_symbols as usize {
                    continue;
                }
                let coefficient = Octet::new(repair_coefficients.source_coefficients[isi]);
                if coefficient.is_zero() {
                    continue;
                }
                fused_addassign_mul_scalar(
                    &mut missing_symbol,
                    symbols.get(matrix_row + h),
                    &coefficient,
                );
            }
        }
    }

    mulassign_scalar(&mut missing_symbol, &missing_coefficient.inverse());

    let missing_entries =
        systematic_constraint_row_entries(source_block_symbols, rows.missing_isi as u32);
    let &missing_entry = missing_entries.first()?;
    let mut decoded = SymbolSlab::with_zeros(width, symbol_size);
    // SourceBlockDecoder copies received source rows directly, so the one-erasure
    // fast path only has to provide the entry that rebuilds the missing source.
    decoded
        .get_mut(missing_entry)
        .copy_from_slice(&missing_symbol);

    Some(decoded)
}

#[cfg(feature = "std")]
fn tagged_single_repair_systematic_rows<M: BinaryMatrix>(
    matrix: &M,
    k_prime: usize,
) -> Option<SingleRepairSystematicRows> {
    let (tagged_k_prime, missing_isi, repair_matrix_row, repair_isi) =
        matrix.contiguous_single_repair_systematic_rows()?;
    if tagged_k_prime != k_prime as u32
        || missing_isi >= k_prime
        || repair_matrix_row >= matrix.height()
    {
        return None;
    }

    Some(SingleRepairSystematicRows {
        missing_isi,
        repair_matrix_row,
        repair_isi: Some(repair_isi),
        systematic_rows: SingleRepairSystematicRowLayout::Contiguous,
    })
}

#[cfg(feature = "std")]
fn classify_single_repair_systematic_rows<M: BinaryMatrix>(
    matrix: &M,
    source_block_symbols: u32,
    s: usize,
    k_prime: usize,
) -> Option<SingleRepairSystematicRows> {
    if let Some(rows) = classify_single_repair_systematic_rows_from_metadata(matrix, s, k_prime) {
        return Some(rows);
    }

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
    let source_rows_contiguous =
        repair_matrix_row == s + k_prime - 1 && systematic_rows.len() + 1 == k_prime;
    (expected_isi == k_prime && systematic_rows.len() + 1 == k_prime).then_some(
        SingleRepairSystematicRows {
            missing_isi,
            repair_matrix_row,
            repair_isi: None,
            systematic_rows: if source_rows_contiguous {
                SingleRepairSystematicRowLayout::Contiguous
            } else {
                SingleRepairSystematicRowLayout::Explicit(systematic_rows)
            },
        },
    )
}

#[cfg(feature = "std")]
fn classify_single_repair_systematic_rows_from_metadata<M: BinaryMatrix>(
    matrix: &M,
    s: usize,
    k_prime: usize,
) -> Option<SingleRepairSystematicRows> {
    let row_isis = matrix.systematic_row_isis()?;
    if row_isis.len() != matrix.height() || matrix.height() < s + k_prime {
        return None;
    }
    if let Some(rows) = classify_contiguous_single_repair_from_metadata(row_isis, s, k_prime) {
        return Some(rows);
    }

    let mut systematic_rows = Vec::with_capacity(k_prime.saturating_sub(1));
    let mut expected_isi = 0usize;
    let mut missing_isi = None;
    let mut repair_matrix_row = None;

    for offset in 0..k_prime {
        let row = s + offset;
        match row_isis[row].map(|isi| isi as usize) {
            Some(isi) if isi == expected_isi => {
                systematic_rows.push((isi, row));
                expected_isi += 1;
            }
            Some(isi)
                if missing_isi.is_none()
                    && expected_isi + 1 < k_prime
                    && isi == expected_isi + 1 =>
            {
                missing_isi = Some(expected_isi);
                systematic_rows.push((isi, row));
                expected_isi += 2;
            }
            None => {
                if repair_matrix_row.is_some() {
                    return None;
                }
                if missing_isi.is_none() {
                    missing_isi = Some(expected_isi);
                    expected_isi += 1;
                }
                repair_matrix_row = Some(row);
            }
            Some(_) => return None,
        }
    }

    let missing_isi = missing_isi?;
    let repair_matrix_row = repair_matrix_row?;
    let source_rows_contiguous =
        repair_matrix_row == s + k_prime - 1 && systematic_rows.len() + 1 == k_prime;
    (expected_isi == k_prime && systematic_rows.len() + 1 == k_prime).then_some(
        SingleRepairSystematicRows {
            missing_isi,
            repair_matrix_row,
            repair_isi: None,
            systematic_rows: if source_rows_contiguous {
                SingleRepairSystematicRowLayout::Contiguous
            } else {
                SingleRepairSystematicRowLayout::Explicit(systematic_rows)
            },
        },
    )
}

#[cfg(feature = "std")]
fn classify_contiguous_single_repair_from_metadata(
    row_isis: &[Option<u32>],
    s: usize,
    k_prime: usize,
) -> Option<SingleRepairSystematicRows> {
    if k_prime == 0 || row_isis[s + k_prime - 1].is_some() {
        return None;
    }

    let mut missing_isi = None;
    for offset in 0..k_prime.saturating_sub(1) {
        let isi = row_isis[s + offset]? as usize;
        match missing_isi {
            None if isi == offset => {}
            None if isi == offset + 1 && offset + 1 < k_prime => {
                missing_isi = Some(offset);
            }
            Some(_) if isi == offset + 1 => {}
            _ => return None,
        }
    }

    Some(SingleRepairSystematicRows {
        missing_isi: missing_isi.unwrap_or(k_prime - 1),
        repair_matrix_row: s + k_prime - 1,
        repair_isi: None,
        systematic_rows: SingleRepairSystematicRowLayout::Contiguous,
    })
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
            let dependent_col = coefficient_col_index(dependent_col);
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
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut next_in_bucket = vec![NO_BUCKET_ROW; height];
    for (row, coefficients) in rows.iter().enumerate() {
        if let Some(&(col, _)) = coefficients.first() {
            push_row_bucket(
                &mut bucket_heads,
                &mut next_in_bucket,
                coefficient_col_index(col),
                row,
            );
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
                Some(coefficient_col(col))
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
                debug_assert!(coefficient_col_index(next_col) > col);
                push_row_bucket(
                    &mut bucket_heads,
                    &mut next_in_bucket,
                    coefficient_col_index(next_col),
                    row,
                );
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
            let dependent_col = coefficient_col_index(dependent_col);
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
            let col = coefficient_col_index(col);
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
    let col = coefficient_col(col);
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
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut next_in_bucket = vec![NO_BUCKET_ROW; height];
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
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    col: usize,
    row: usize,
) {
    debug_assert_eq!(next_in_bucket[row], NO_BUCKET_ROW);
    next_in_bucket[row] = bucket_heads[col];
    bucket_heads[col] = row;
}

fn push_counted_row_bucket(
    rows: &[CoefficientRow],
    bucket_heads: &mut [usize],
    singleton_bucket_heads: &mut [usize],
    bucket_counts: &mut [usize],
    next_in_bucket: &mut [usize],
    col: usize,
    row: usize,
) {
    debug_assert_eq!(
        rows[row].first().map(|&(entry_col, _)| entry_col),
        Some(coefficient_col(col))
    );
    if is_singleton_bucket_row(&rows[row], col) {
        push_row_bucket(singleton_bucket_heads, next_in_bucket, col, row);
    } else {
        push_row_bucket(bucket_heads, next_in_bucket, col, row);
    }
    bucket_counts[col] += 1;
}

fn is_singleton_bucket_row(row: &CoefficientRow, col: usize) -> bool {
    row.len() == 1 && row[0].0 == coefficient_col(col)
}

fn pop_row_bucket(
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    col: usize,
) -> Option<usize> {
    let row = bucket_heads[col];
    if row == NO_BUCKET_ROW {
        return None;
    }
    bucket_heads[col] = next_in_bucket[row];
    next_in_bucket[row] = NO_BUCKET_ROW;
    Some(row)
}

fn pop_counted_row_bucket(
    bucket_heads: &mut [usize],
    singleton_bucket_heads: &mut [usize],
    bucket_counts: &mut [usize],
    next_in_bucket: &mut [usize],
    col: usize,
) -> Option<usize> {
    let row = pop_row_bucket(singleton_bucket_heads, next_in_bucket, col)
        .or_else(|| pop_row_bucket(bucket_heads, next_in_bucket, col))?;
    debug_assert_ne!(bucket_counts[col], 0);
    bucket_counts[col] -= 1;
    Some(row)
}

fn pop_lightest_coefficient_row_bucket(
    rows: &[CoefficientRow],
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    col: usize,
) -> Option<(usize, Octet)> {
    pop_lightest_coefficient_row_bucket_with_min_suffix::<1>(
        rows,
        bucket_heads,
        next_in_bucket,
        col,
    )
}

fn pop_lightest_non_singleton_coefficient_row_bucket(
    rows: &[CoefficientRow],
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    col: usize,
) -> Option<(usize, Octet)> {
    pop_lightest_coefficient_row_bucket_with_min_suffix::<2>(
        rows,
        bucket_heads,
        next_in_bucket,
        col,
    )
}

fn pop_lightest_coefficient_row_bucket_with_min_suffix<const MIN_SUFFIX_LEN: usize>(
    rows: &[CoefficientRow],
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    col: usize,
) -> Option<(usize, Octet)> {
    let head = bucket_heads[col];
    if head == NO_BUCKET_ROW {
        return None;
    }
    debug_assert_eq!(
        rows[head].first().map(|&(entry_col, _)| entry_col),
        Some(coefficient_col(col))
    );
    debug_assert!(rows[head].len() >= MIN_SUFFIX_LEN);
    if next_in_bucket[head] == NO_BUCKET_ROW {
        bucket_heads[col] = NO_BUCKET_ROW;
        return Some((head, rows[head][0].1));
    }
    let head_value = rows[head][0].1;
    if rows[head].len() == MIN_SUFFIX_LEN && (MIN_SUFFIX_LEN == 1 || head_value == Octet::one()) {
        bucket_heads[col] = next_in_bucket[head];
        next_in_bucket[head] = NO_BUCKET_ROW;
        return Some((head, head_value));
    }

    let mut best = head;
    let mut best_previous = NO_BUCKET_ROW;
    let mut best_suffix_len = rows[head].len();
    let mut best_value = head_value;
    let mut previous = head;
    let mut current = next_in_bucket[head];

    while current != NO_BUCKET_ROW {
        let row = current;
        debug_assert_eq!(
            rows[row].first().map(|&(entry_col, _)| entry_col),
            Some(coefficient_col(col))
        );
        debug_assert!(rows[row].len() >= MIN_SUFFIX_LEN);
        let value = rows[row][0].1;
        let suffix_len = rows[row].len();
        if suffix_len < best_suffix_len
            || (suffix_len == best_suffix_len
                && value == Octet::one()
                && best_value != Octet::one())
        {
            best = row;
            best_previous = previous;
            best_suffix_len = suffix_len;
            best_value = value;
            if suffix_len == MIN_SUFFIX_LEN && (MIN_SUFFIX_LEN == 1 || best_value == Octet::one()) {
                break;
            }
        }
        previous = row;
        current = next_in_bucket[row];
    }

    if best_previous == NO_BUCKET_ROW {
        bucket_heads[col] = next_in_bucket[best];
    } else {
        next_in_bucket[best_previous] = next_in_bucket[best];
    }
    next_in_bucket[best] = NO_BUCKET_ROW;
    Some((best, best_value))
}

fn pop_lightest_counted_coefficient_row_bucket(
    rows: &[CoefficientRow],
    bucket_heads: &mut [usize],
    singleton_bucket_heads: &mut [usize],
    bucket_counts: &mut [usize],
    next_in_bucket: &mut [usize],
    col: usize,
) -> Option<(usize, Octet)> {
    let result = if let Some(row) = pop_row_bucket(singleton_bucket_heads, next_in_bucket, col) {
        debug_assert!(is_singleton_bucket_row(&rows[row], col));
        (row, rows[row][0].1)
    } else {
        pop_lightest_non_singleton_coefficient_row_bucket(rows, bucket_heads, next_in_bucket, col)?
    };
    debug_assert_ne!(bucket_counts[col], 0);
    bucket_counts[col] -= 1;
    Some(result)
}

fn pop_lightest_binary_row_bucket(
    rows: &PackedBinaryRows,
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    col: usize,
) -> Option<usize> {
    let head = bucket_heads[col];
    if head == NO_BUCKET_ROW {
        return None;
    }
    if next_in_bucket[head] == NO_BUCKET_ROW {
        bucket_heads[col] = NO_BUCKET_ROW;
        return Some(head);
    }

    let mut best = head;
    let mut best_previous = NO_BUCKET_ROW;
    let mut best_weight = rows.weight_at_or_after(head, col);
    let mut previous = head;
    let mut current = next_in_bucket[head];

    while current != NO_BUCKET_ROW {
        let row = current;
        let weight = rows.weight_at_or_after(row, col);
        if weight < best_weight {
            best = row;
            best_previous = previous;
            best_weight = weight;
            if weight == 1 {
                break;
            }
        }
        previous = row;
        current = next_in_bucket[row];
    }

    if best_previous == NO_BUCKET_ROW {
        bucket_heads[col] = next_in_bucket[best];
    } else {
        next_in_bucket[best_previous] = next_in_bucket[best];
    }
    next_in_bucket[best] = NO_BUCKET_ROW;
    Some(best)
}

#[cfg(feature = "std")]
fn pop_lightest_weighted_binary_row_bucket(
    row_weights: &[u32],
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    col: usize,
) -> Option<usize> {
    let head = bucket_heads[col];
    if head == NO_BUCKET_ROW {
        return None;
    }
    if next_in_bucket[head] == NO_BUCKET_ROW {
        bucket_heads[col] = NO_BUCKET_ROW;
        return Some(head);
    }

    let mut best = head;
    let mut best_previous = NO_BUCKET_ROW;
    let mut best_weight = row_weights[head];
    if best_weight == 1 {
        bucket_heads[col] = next_in_bucket[head];
        next_in_bucket[head] = NO_BUCKET_ROW;
        return Some(head);
    }
    let mut previous = head;
    let mut current = next_in_bucket[head];

    while current != NO_BUCKET_ROW {
        let row = current;
        let weight = row_weights[row];
        if weight < best_weight {
            best = row;
            best_previous = previous;
            best_weight = weight;
            if weight == 1 {
                break;
            }
        }
        previous = row;
        current = next_in_bucket[row];
    }

    if best_previous == NO_BUCKET_ROW {
        bucket_heads[col] = next_in_bucket[best];
    } else {
        next_in_bucket[best_previous] = next_in_bucket[best];
    }
    next_in_bucket[best] = NO_BUCKET_ROW;
    Some(best)
}

fn symbol_is_zero(symbol: &[u8]) -> bool {
    bytes_are_zero(symbol)
}

fn coefficient_at(row: &CoefficientRow, col: usize) -> Octet {
    let col = coefficient_col(col);
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
    let start_col = coefficient_col(start_col);
    let start = if row.first().is_some_and(|&(col, _)| col >= start_col) {
        0
    } else {
        row.partition_point(|&(col, _)| col < start_col)
    };
    let table = scalar.mul_table();
    for (_, value) in row[start..].iter_mut() {
        *value = multiply_with_table(*value, table);
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
                let src_col = coefficient_col(src_col);
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
                scratch.push((coefficient_col(src_col), scalar));
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
    let start_col = coefficient_col(start_col);
    let mut src_index = if src.first().is_some_and(|&(col, _)| col >= start_col) {
        0
    } else {
        src.partition_point(|&(col, _)| col < start_col)
    };
    let src_tail_len = src.len() - src_index;
    if use_sparse_source_merge(dest.len(), src_tail_len) {
        let dest_start = if dest.first().is_some_and(|&(col, _)| col >= start_col) {
            0
        } else {
            dest.partition_point(|&(col, _)| col < start_col)
        };
        add_scaled_sparse_source_matrix_row_tail(dest, &src[src_index..], scalar, dest_start);
        return;
    }
    let table = scalar.mul_table();
    scratch.clear();
    scratch.reserve(dest.len() + src_tail_len);

    while dest_index < dest.len() || src_index < src.len() {
        match (dest.get(dest_index), src.get(src_index)) {
            (Some(&(dest_col, dest_value)), Some(&(src_col, src_value))) => {
                if dest_col < src_col {
                    scratch.push((dest_col, dest_value));
                    dest_index += 1;
                } else if src_col < dest_col {
                    scratch.push((src_col, multiply_with_table(src_value, table)));
                    src_index += 1;
                } else {
                    let value = dest_value + multiply_with_table(src_value, table);
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
                scratch.push((src_col, multiply_with_table(src_value, table)));
                src_index += 1;
            }
            (None, None) => break,
        }
    }

    core::mem::swap(dest, scratch);
}

fn multiply_with_table(value: Octet, table: &[u8; 256]) -> Octet {
    Octet::new(table[value.value() as usize])
}

#[inline]
fn remove_matrix_row_entry_at(row: &mut CoefficientRow, index: usize) {
    let len = row.len();
    debug_assert!(index < len);
    if index + 1 < len {
        row.copy_within(index + 1..len, index);
    }
    row.truncate(len - 1);
}

#[inline]
fn insert_matrix_row_entry_at(
    row: &mut CoefficientRow,
    index: usize,
    entry: (CoefficientColumn, Octet),
) {
    let len = row.len();
    debug_assert!(index <= len);
    row.push(entry);
    if index < len {
        row.copy_within(index..len, index + 1);
        row[index] = entry;
    }
}

fn use_sparse_source_merge(dest_len: usize, src_tail_len: usize) -> bool {
    src_tail_len != 0
        && src_tail_len <= SPARSE_SOURCE_MERGE_MAX_SOURCE_LEN
        && dest_len / src_tail_len >= SPARSE_SOURCE_MERGE_DEST_FACTOR
}

fn add_scaled_sparse_source_matrix_row_tail(
    dest: &mut CoefficientRow,
    src_tail: &[(CoefficientColumn, Octet)],
    scalar: Octet,
    search_start: usize,
) {
    let mut search_start = search_start;
    if scalar == Octet::one() {
        for &(src_col, src_value) in src_tail {
            search_start =
                add_sparse_source_matrix_row_entry(dest, search_start, src_col, src_value);
        }
        return;
    }

    let table = scalar.mul_table();
    for &(src_col, src_value) in src_tail {
        search_start = add_sparse_source_matrix_row_entry(
            dest,
            search_start,
            src_col,
            multiply_with_table(src_value, table),
        );
    }
}

fn add_sparse_source_matrix_row_entry(
    dest: &mut CoefficientRow,
    mut search_start: usize,
    src_col: CoefficientColumn,
    scaled: Octet,
) -> usize {
    debug_assert!(search_start <= dest.len());
    if search_start == dest.len() || dest.last().is_some_and(|&(dest_col, _)| dest_col < src_col) {
        dest.push((src_col, scaled));
        return dest.len();
    }

    let mut remaining_linear = SPARSE_SOURCE_LINEAR_SCAN_LIMIT;
    while remaining_linear != 0 {
        let (dest_col, dest_value) = dest[search_start];
        if dest_col < src_col {
            search_start += 1;
            remaining_linear -= 1;
            continue;
        }
        if dest_col == src_col {
            let value = dest_value + scaled;
            if value.is_zero() {
                remove_matrix_row_entry_at(dest, search_start);
                return search_start;
            }

            dest[search_start].1 = value;
            return search_start + 1;
        }

        insert_matrix_row_entry_at(dest, search_start, (src_col, scaled));
        return search_start + 1;
    }

    let offset = dest[search_start..].partition_point(|&(dest_col, _)| dest_col < src_col);
    let index = search_start + offset;
    if dest
        .get(index)
        .is_some_and(|&(dest_col, _)| dest_col == src_col)
    {
        let value = dest[index].1 + scaled;
        if value.is_zero() {
            remove_matrix_row_entry_at(dest, index);
            index
        } else {
            dest[index].1 = value;
            index + 1
        }
    } else {
        insert_matrix_row_entry_at(dest, index, (src_col, scaled));
        index + 1
    }
}

fn add_short_matrix_row_entry(
    dest: &mut CoefficientRow,
    search_start: usize,
    src_col: CoefficientColumn,
    scaled: Octet,
) -> usize {
    debug_assert!(search_start <= dest.len());
    if search_start == dest.len() || dest.last().is_some_and(|&(dest_col, _)| dest_col < src_col) {
        dest.push((src_col, scaled));
        return dest.len();
    }

    let &(dest_col, dest_value) = &dest[search_start];
    if dest_col == src_col {
        let value = dest_value + scaled;
        if value.is_zero() {
            remove_matrix_row_entry_at(dest, search_start);
            return search_start;
        }

        dest[search_start].1 = value;
        return search_start + 1;
    }
    if dest_col > src_col {
        insert_matrix_row_entry_at(dest, search_start, (src_col, scaled));
        return search_start + 1;
    }

    match find_matrix_row_entry_from(dest, search_start + 1, src_col, SHORT_ROW_LINEAR_SCAN_LIMIT) {
        Ok(index) => {
            let value = dest[index].1 + scaled;
            if value.is_zero() {
                remove_matrix_row_entry_at(dest, index);
                index
            } else {
                dest[index].1 = value;
                index + 1
            }
        }
        Err(index) => {
            insert_matrix_row_entry_at(dest, index, (src_col, scaled));
            index + 1
        }
    }
}

#[inline]
fn find_matrix_row_entry_from(
    row: &CoefficientRow,
    mut search_start: usize,
    col: CoefficientColumn,
    mut linear_limit: usize,
) -> Result<usize, usize> {
    while search_start < row.len() && linear_limit != 0 {
        let row_col = row[search_start].0;
        if row_col < col {
            search_start += 1;
            linear_limit -= 1;
            continue;
        }
        return if row_col == col {
            Ok(search_start)
        } else {
            Err(search_start)
        };
    }

    let offset = row[search_start..].partition_point(|&(row_col, _)| row_col < col);
    let index = search_start + offset;
    if row.get(index).is_some_and(|&(row_col, _)| row_col == col) {
        Ok(index)
    } else {
        Err(index)
    }
}

#[cfg(all(test, feature = "std"))]
fn add_scaled_short_matrix_row(
    dest: &mut CoefficientRow,
    src: &CoefficientRow,
    start_col: usize,
    scalar: Octet,
) {
    let start_col = coefficient_col(start_col);
    let src_start = if src.first().is_some_and(|&(col, _)| col >= start_col) {
        0
    } else {
        src.partition_point(|&(col, _)| col < start_col)
    };
    let scalar_table = (scalar != Octet::one()).then(|| scalar.mul_table());
    let mut search_start = 0usize;

    for &(src_col, src_value) in &src[src_start..] {
        let scaled = match scalar_table.as_ref() {
            Some(table) => multiply_with_table(src_value, table),
            None => src_value,
        };
        let offset = dest[search_start..].partition_point(|&(dest_col, _)| dest_col < src_col);
        let index = search_start + offset;

        if dest
            .get(index)
            .is_some_and(|&(dest_col, _)| dest_col == src_col)
        {
            let value = dest[index].1 + scaled;
            if value.is_zero() {
                dest.remove(index);
                search_start = index;
            } else {
                dest[index].1 = value;
                search_start = index + 1;
            }
        } else {
            dest.insert(index, (src_col, scaled));
            search_start = index + 1;
        }
    }
}

fn disjoint_coefficient_rows_mut(
    rows: &mut [CoefficientRow],
    src: usize,
    dest: usize,
) -> (&CoefficientRow, &mut CoefficientRow) {
    assert_ne!(src, dest);
    if src < dest {
        let (before_dest, dest_and_after) = rows.split_at_mut(dest);
        (&before_dest[src], &mut dest_and_after[0])
    } else {
        let (before_src, src_and_after) = rows.split_at_mut(src);
        (&src_and_after[0], &mut before_src[dest])
    }
}

#[cfg(feature = "std")]
fn add_scaled_normalized_short_matrix_row(
    dest: &mut CoefficientRow,
    src: &CoefficientRow,
    start_col: usize,
    scalar: Octet,
) {
    let start_col = coefficient_col(start_col);
    debug_assert_eq!(src.first().copied(), Some((start_col, Octet::one())));
    debug_assert_eq!(dest.first().map(|&(col, _)| col), Some(start_col));
    debug_assert_eq!(dest.first().map(|&(_, value)| value), Some(scalar));

    let Some((&(first_col, first_value), remaining_src)) = src[1..].split_first() else {
        if dest.len() == 1 {
            dest.clear();
        } else {
            remove_matrix_row_entry_at(dest, 0);
        }
        return;
    };

    if scalar == Octet::one() {
        let search_start =
            add_short_matrix_row_entry_and_remove_pivot(dest, first_col, first_value);
        add_normalized_short_matrix_row_tail_from(dest, search_start, remaining_src);
    } else {
        let table = scalar.mul_table();
        let search_start = add_short_matrix_row_entry_and_remove_pivot(
            dest,
            first_col,
            multiply_with_table(first_value, table),
        );
        add_scaled_normalized_short_matrix_row_tail_from(dest, search_start, remaining_src, table);
    }
}

fn add_normalized_short_matrix_row_tail_from(
    dest: &mut CoefficientRow,
    search_start: usize,
    src_tail: &[(CoefficientColumn, Octet)],
) {
    let mut search_start = search_start;
    for &(src_col, src_value) in src_tail {
        search_start = add_short_matrix_row_entry(dest, search_start, src_col, src_value);
    }
}

fn add_scaled_normalized_short_matrix_row_tail_from(
    dest: &mut CoefficientRow,
    search_start: usize,
    src_tail: &[(CoefficientColumn, Octet)],
    table: &[u8; 256],
) {
    let mut search_start = search_start;
    for &(src_col, src_value) in src_tail {
        search_start = add_short_matrix_row_entry(
            dest,
            search_start,
            src_col,
            multiply_with_table(src_value, table),
        );
    }
}

#[cfg(feature = "std")]
fn add_normalized_binary_short_matrix_row(dest: &mut CoefficientRow, src: &CoefficientRow) {
    debug_assert!(!dest.is_empty());
    debug_assert!(!src.is_empty());
    debug_assert_eq!(dest[0], src[0]);
    debug_assert_eq!(src[0].1, Octet::one());

    let src_tail = &src[1..];
    if src_tail.is_empty() {
        remove_matrix_row_entry_at(dest, 0);
        return;
    }

    if dest.len() == 1 {
        dest.clear();
        dest.extend_from_slice(src_tail);
        return;
    }

    let mut search_start = add_binary_short_matrix_row_entry_and_remove_pivot(dest, src_tail[0].0);
    for &(src_col, _) in &src_tail[1..] {
        search_start = add_binary_short_matrix_row_entry_from(dest, src_col, search_start);
    }
}

#[cfg(feature = "std")]
#[inline]
fn add_binary_short_matrix_row_entry_from(
    dest: &mut CoefficientRow,
    src_col: CoefficientColumn,
    search_start: usize,
) -> usize {
    if search_start == dest.len() || dest.last().is_some_and(|&(dest_col, _)| dest_col < src_col) {
        dest.push((src_col, Octet::one()));
        return dest.len();
    }

    let dest_col = dest[search_start].0;
    if dest_col == src_col {
        remove_matrix_row_entry_at(dest, search_start);
        return search_start;
    }
    if dest_col > src_col {
        insert_matrix_row_entry_at(dest, search_start, (src_col, Octet::one()));
        return search_start + 1;
    }

    match find_matrix_row_entry_from(dest, search_start + 1, src_col, SHORT_ROW_LINEAR_SCAN_LIMIT) {
        Ok(index) => {
            remove_matrix_row_entry_at(dest, index);
            index
        }
        Err(index) => {
            insert_matrix_row_entry_at(dest, index, (src_col, Octet::one()));
            index + 1
        }
    }
}

#[cfg(feature = "std")]
#[inline]
fn add_binary_short_matrix_row_entry_and_remove_pivot(
    dest: &mut CoefficientRow,
    src_col: CoefficientColumn,
) -> usize {
    debug_assert!(!dest.is_empty());
    let len = dest.len();
    let offset = dest[1..].partition_point(|&(dest_col, _)| dest_col < src_col);
    let index = 1 + offset;

    if index < len && dest[index].0 == src_col {
        if index > 1 {
            dest.copy_within(1..index, 0);
        }
        if index + 1 < len {
            dest.copy_within(index + 1..len, index - 1);
        }
        dest.truncate(len - 2);
        index - 1
    } else {
        if index > 1 {
            dest.copy_within(1..index, 0);
        }
        dest[index - 1] = (src_col, Octet::one());
        index
    }
}

#[cfg(feature = "std")]
#[inline]
fn add_short_matrix_row_entry_and_remove_pivot(
    dest: &mut CoefficientRow,
    src_col: CoefficientColumn,
    scaled: Octet,
) -> usize {
    debug_assert!(!dest.is_empty());
    debug_assert!(dest[0].0 < src_col);

    // The first destination entry is the eliminated pivot; fold its removal
    // into the first tail merge so insertion/cancellation compacts once.
    let len = dest.len();
    if len == 1 {
        dest[0] = (src_col, scaled);
        return 1;
    }

    if dest[len - 1].0 < src_col {
        dest.copy_within(1..len, 0);
        dest[len - 1] = (src_col, scaled);
        return len;
    }

    match dest[1..].binary_search_by_key(&src_col, |&(dest_col, _)| dest_col) {
        Ok(offset) => {
            let index = 1 + offset;
            let value = dest[index].1 + scaled;
            if value.is_zero() {
                if index > 1 {
                    dest.copy_within(1..index, 0);
                }
                if index + 1 < len {
                    dest.copy_within(index + 1..len, index - 1);
                }
                dest.truncate(len - 2);
                index - 1
            } else {
                dest[index].1 = value;
                dest.copy_within(1..len, 0);
                dest.truncate(len - 1);
                index
            }
        }
        Err(offset) => {
            let index = 1 + offset;
            if index > 1 {
                dest.copy_within(1..index, 0);
            }
            dest[index - 1] = (src_col, scaled);
            index
        }
    }
}

fn add_unscaled_matrix_row(
    dest: &mut CoefficientRow,
    src: &CoefficientRow,
    start_col: usize,
    scratch: &mut CoefficientRow,
) {
    let mut dest_index = 0usize;
    let start_col = coefficient_col(start_col);
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
        let width = 64usize;
        let rows: Vec<CoefficientRow> = (0..width)
            .map(|col| vec![(coefficient_col(col), Octet::one())])
            .collect();
        let symbols = SymbolSlab::with_zeros(width, 1);

        let (decoded, ops) = solve(rows, width, symbols, OperationRecording::Record);

        assert!(decoded.is_some());
        assert!(ops.is_some());
    }

    #[test]
    fn triangular_recording_solves_and_replays_wide_system() {
        let width = TRIANGULAR_RECORDING_MIN_WIDTH;
        let mut rows: Vec<CoefficientRow> = (0..width)
            .map(|col| vec![(coefficient_col(col), Octet::one())])
            .collect();
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
                        acc + value * expected[coefficient_col_index(col)]
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
    use crate::systematic_constants::num_hdpc_symbols;

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
        let (rows, symbols) = prepared_plan_test_system();

        let (expected, ops) = solve(rows.clone(), 4, symbols.clone(), OperationRecording::Skip);
        let plan = prepare_cached_systematic_plan(rows, 4);
        let mut replayed = symbols;
        apply_prepared_systematic_plan(&plan, &mut replayed);

        assert_eq!(replayed, expected.unwrap());
        assert!(ops.is_none());
    }

    #[test]
    fn prepared_systematic_plan_handles_non_unit_singleton_pivot() {
        let rows = vec![
            vec![(0, Octet::new(7))],
            vec![(0, Octet::new(3)), (1, Octet::new(5))],
        ];
        let symbols = SymbolSlab::from_bytes(vec![11, 19], 1);

        let (expected, _) = solve(rows.clone(), 2, symbols.clone(), OperationRecording::Skip);
        let plan = prepare_cached_systematic_plan(rows, 2);
        let mut replayed = symbols;
        apply_prepared_systematic_plan(&plan, &mut replayed);

        assert_eq!(replayed, expected.unwrap());
    }

    #[test]
    fn direct_systematic_solve_matches_cached_plan_replay() {
        let source_symbols = 10;
        let width = num_intermediate_symbols(source_symbols) as usize;
        let s = num_ldpc_symbols(source_symbols) as usize;
        let h = num_hdpc_symbols(source_symbols) as usize;
        let symbol_size = 3;

        let mut cached = SymbolSlab::with_zeros(width, symbol_size);
        for isi in 0..source_symbols as usize {
            cached.get_mut(s + h + isi).copy_from_slice(&[
                isi as u8,
                (isi as u8).wrapping_mul(17).wrapping_add(3),
                (isi as u8) ^ 0x5a,
            ]);
        }
        let mut direct = cached.clone();

        apply_cached_systematic_plan(source_symbols, &mut cached);
        apply_direct_systematic_solve(source_symbols, &mut direct);

        assert_eq!(direct, cached);
    }

    #[test]
    fn direct_pivot_move_handles_cycles_and_hdpc_gaps() {
        let pivot_for_col = vec![
            coefficient_col(1),
            coefficient_col(2),
            coefficient_col(0),
            coefficient_col(3),
            NO_COEFFICIENT_COLUMN,
        ];
        let plan = DirectSystematicPlan {
            forward_steps: Vec::new(),
            forward_dests: DirectSystematicSlices {
                ranges: Vec::new(),
                entries: Vec::new(),
            },
            hdpc_update_pivots: Vec::new().into_boxed_slice(),
            hdpc_updates: CachedSystematicSlices {
                ranges: Vec::new(),
                entries: Vec::new(),
                unit_only: Vec::new(),
            },
            hdpc_free_rows: Vec::new(),
            free_cols: vec![coefficient_col(4)].into_boxed_slice(),
            pivot_symbol_moves: direct_pivot_symbol_moves(&pivot_for_col, 1, 1, 5),
            back_substitution: DirectSystematicSlices {
                ranges: Vec::new(),
                entries: Vec::new(),
            },
            width: 5,
            s: 1,
            h: 1,
        };
        let mut symbols = SymbolSlab::from_bytes(vec![10, 11, 12, 13, 14], 1);

        move_direct_pivot_symbols_to_columns(&plan, &mut symbols);

        assert_eq!(symbols.as_bytes(), &[12, 13, 10, 14, 14]);
    }

    #[test]
    fn non_singleton_bucket_preserves_unit_tie_break_at_min_suffix() {
        let rows = vec![
            vec![
                (coefficient_col(0), Octet::new(7)),
                (coefficient_col(2), Octet::one()),
            ],
            vec![
                (coefficient_col(0), Octet::one()),
                (coefficient_col(1), Octet::one()),
            ],
        ];
        let mut bucket_heads = vec![NO_BUCKET_ROW; 1];
        let mut next_in_bucket = vec![NO_BUCKET_ROW; rows.len()];

        push_row_bucket(&mut bucket_heads, &mut next_in_bucket, 0, 1);
        push_row_bucket(&mut bucket_heads, &mut next_in_bucket, 0, 0);

        assert_eq!(
            pop_lightest_non_singleton_coefficient_row_bucket(
                &rows,
                &mut bucket_heads,
                &mut next_in_bucket,
                0,
            ),
            Some((1, Octet::one()))
        );
    }

    #[test]
    fn weighted_binary_bucket_returns_head_singleton_without_scan() {
        let row_weights = vec![1, 3];
        let mut bucket_heads = vec![0];
        let mut next_in_bucket = vec![1, NO_BUCKET_ROW];

        assert_eq!(
            pop_lightest_weighted_binary_row_bucket(
                &row_weights,
                &mut bucket_heads,
                &mut next_in_bucket,
                0,
            ),
            Some(0)
        );
        assert_eq!(bucket_heads[0], 1);
        assert_eq!(next_in_bucket[0], NO_BUCKET_ROW);
    }

    fn prepared_plan_test_system() -> (Vec<CoefficientRow>, SymbolSlab) {
        (
            vec![
                vec![(0, Octet::new(2)), (1, Octet::new(5)), (3, Octet::new(7))],
                vec![(0, Octet::new(6)), (1, Octet::new(3)), (2, Octet::new(11))],
                vec![(1, Octet::new(9)), (2, Octet::new(5)), (3, Octet::new(13))],
                vec![(2, Octet::new(10)), (3, Octet::new(17))],
            ],
            SymbolSlab::from_bytes(
                vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 7, 11, 19, 31],
                4,
            ),
        )
    }

    #[test]
    fn large_plan_exercises_clone_free_elimination_and_batched_back_substitution() {
        let width = CLONE_FREE_PLAN_ELIMINATION_MIN_WIDTH.max(BATCHED_BACK_SUBSTITUTION_MIN_WIDTH);
        let mut rows: Vec<CoefficientRow> = (0..width)
            .map(|col| vec![(coefficient_col(col), Octet::one())])
            .collect::<Vec<_>>();
        rows[0] = vec![(0, Octet::one()), (1, Octet::one())];
        rows[1] = vec![(0, Octet::one()), (2, Octet::one())];

        let mut symbol_bytes = (0..width)
            .map(|index| (index as u8).wrapping_mul(31).wrapping_add(7))
            .collect::<Vec<_>>();
        symbol_bytes[0] = 0x11;
        symbol_bytes[1] = 0x22;
        symbol_bytes[2] = 0x44;
        let symbols = SymbolSlab::from_bytes(symbol_bytes.clone(), 1);

        let plan = prepare_cached_systematic_plan(rows, width);
        assert!(matches!(
            &plan.back_substitution,
            CachedSystematicBackSubstitution::Batches(_)
                | CachedSystematicBackSubstitution::FlatBatches(_)
        ));

        let mut replayed = symbols;
        apply_prepared_systematic_plan(&plan, &mut replayed);

        symbol_bytes[0] = 0x22 ^ 0x44;
        symbol_bytes[1] = 0x11 ^ 0x22 ^ 0x44;
        symbol_bytes[2] = 0x44;

        assert_eq!(replayed.as_bytes(), symbol_bytes.as_slice());
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
            .map(|&col| (coefficient_col(col), Octet::one()))
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

    #[test]
    fn short_matrix_row_merge_matches_generic_merge() {
        let src = vec![
            (0, Octet::one()),
            (2, Octet::new(5)),
            (5, Octet::new(9)),
            (8, Octet::new(11)),
        ];
        for scalar in [Octet::one(), Octet::new(7)] {
            let dest = vec![
                (0, scalar),
                (1, Octet::new(3)),
                (2, Octet::new(7)),
                (5, Octet::new(63)),
                (9, Octet::new(13)),
            ];
            let mut generic = dest.clone();
            let mut short = dest;
            let mut scratch = Vec::new();

            add_scaled_matrix_row(&mut generic, &src, 0, scalar, &mut scratch);
            add_scaled_short_matrix_row(&mut short, &src, 0, scalar);

            assert_eq!(short, generic);
        }
    }

    #[test]
    fn normalized_short_matrix_row_matches_generic_merge() {
        let src = vec![(2, Octet::one()), (4, Octet::new(17)), (8, Octet::new(91))];

        for scalar in [Octet::one(), Octet::new(11)] {
            let dest = vec![
                (2, scalar),
                (4, Octet::new(19)),
                (5, Octet::new(7)),
                (9, Octet::new(13)),
            ];
            let mut generic = dest.clone();
            let mut short = dest;
            let mut scratch = Vec::new();

            add_scaled_matrix_row(&mut generic, &src, 2, scalar, &mut scratch);
            add_scaled_normalized_short_matrix_row(&mut short, &src, 2, scalar);

            assert_eq!(short, generic);
        }

        let cases = [
            (
                vec![(2, Octet::one()), (3, Octet::new(3)), (8, Octet::new(9))],
                vec![(2, Octet::one()), (4, Octet::new(5))],
                Octet::one(),
            ),
            (
                vec![(2, Octet::new(11)), (3, Octet::new(3))],
                vec![(2, Octet::one()), (8, Octet::new(5))],
                Octet::new(11),
            ),
            (
                vec![(2, Octet::one()), (4, Octet::new(5)), (8, Octet::new(9))],
                vec![(2, Octet::one()), (4, Octet::new(5))],
                Octet::one(),
            ),
        ];

        for (dest, src, scalar) in cases {
            let mut generic = dest.clone();
            let mut short = dest;
            let mut scratch = Vec::new();

            add_scaled_matrix_row(&mut generic, &src, 2, scalar, &mut scratch);
            add_scaled_normalized_short_matrix_row(&mut short, &src, 2, scalar);

            assert_eq!(short, generic);
        }
    }

    #[test]
    fn sparse_source_matrix_row_merge_matches_dense_projection() {
        let scalar = Octet::new(7);
        let mut dest = (0..128)
            .map(|col| (coefficient_col(col), Octet::new((col % 251 + 1) as u8)))
            .collect::<Vec<_>>();
        let src = vec![
            (0, Octet::one()),
            (5, Octet::new(11)),
            (31, Octet::new(17)),
            (64, Octet::new(19)),
            (127, Octet::new(23)),
        ];
        let mut expected_coefficients = dest.iter().map(|&(_, value)| value).collect::<Vec<_>>();
        let mut scratch = Vec::new();

        add_scaled_matrix_row(&mut dest, &src, 0, scalar, &mut scratch);
        for &(col, value) in &src {
            let col = coefficient_col_index(col);
            expected_coefficients[col] += value * scalar;
        }
        let expected = expected_coefficients
            .into_iter()
            .enumerate()
            .filter_map(|(col, value)| (!value.is_zero()).then_some((coefficient_col(col), value)))
            .collect::<Vec<_>>();

        assert_eq!(dest, expected);
    }

    #[test]
    fn normalized_binary_matrix_row_matches_generic_merge() {
        let one = Octet::one();
        let cases = [
            (vec![(2, one)], vec![(2, one), (5, one)]),
            (vec![(2, one), (3, one)], vec![(2, one), (5, one)]),
            (vec![(2, one), (9, one)], vec![(2, one), (5, one)]),
            (
                vec![(2, one), (5, one), (7, one)],
                vec![(2, one), (5, one), (8, one)],
            ),
        ];

        for (src, dest) in cases {
            let mut generic = dest.clone();
            let mut binary_short = dest;
            let mut scratch = Vec::new();

            add_scaled_matrix_row(&mut generic, &src, 2, one, &mut scratch);
            add_normalized_binary_short_matrix_row(&mut binary_short, &src);

            assert_eq!(binary_short, generic);
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
            rows.push(vec![(coefficient_col(col), one)]);
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

    #[derive(Clone)]
    struct SystematicMetadataOnlyMatrix {
        height: usize,
        width: usize,
        row_isis: Vec<Option<u32>>,
    }

    impl BinaryMatrix for SystematicMetadataOnlyMatrix {
        fn new(_height: usize, _width: usize) -> SystematicMetadataOnlyMatrix {
            unimplemented!("test fixture supplies systematic row metadata directly")
        }

        fn height(&self) -> usize {
            self.height
        }

        fn width(&self) -> usize {
            self.width
        }

        fn systematic_row_isis(&self) -> Option<&[Option<u32>]> {
            Some(&self.row_isis)
        }

        fn get(&self, _row: usize, _col: usize) -> Octet {
            Octet::zero()
        }

        fn set(&mut self, _row: usize, _col: usize, _value: bool) {
            unimplemented!("test fixture is read-only")
        }
    }

    #[test]
    fn single_repair_classifier_uses_systematic_row_metadata() {
        let s = 3;
        let k_prime = 5;
        let row_isis = vec![None, None, None, Some(0), Some(1), Some(3), Some(4), None];
        let matrix = SystematicMetadataOnlyMatrix {
            height: s + k_prime,
            width: 12,
            row_isis,
        };

        let rows = classify_single_repair_systematic_rows(&matrix, k_prime as u32, s, k_prime)
            .expect("metadata should classify one missing systematic row");

        assert_eq!(rows.missing_isi, 2);
        assert_eq!(rows.repair_matrix_row, s + k_prime - 1);
        assert!(matches!(
            rows.systematic_rows,
            SingleRepairSystematicRowLayout::Contiguous
        ));
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
    fn square_hybrid_decode_handles_width_above_previous_cap() {
        let width = 32_768 + 1;
        let source_block_symbols = 10;
        let symbol_size = 1;
        let matrix = PackedOnlyMatrix::new(packed_identity_with_free_last_column(width));
        let mut hdpc_rows = DenseOctetMatrix::new(1, width);
        hdpc_rows.set(0, width - 1, Octet::one());
        let symbols = SymbolSlab::with_zeros(matrix.height() + hdpc_rows.height(), symbol_size);

        let (decoded, ops) =
            fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_block_symbols);

        assert_eq!(decoded.unwrap(), SymbolSlab::with_zeros(width, symbol_size));
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
    fn large_systematic_plan_uses_hybrid_direct_systematic_solve() {
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
            Some([SymbolOps::DirectSystematicSolve {
                source_block_symbols
            }]) if *source_block_symbols == k_prime
        ));
    }

    #[test]
    fn cached_hybrid_systematic_plan_replays_direct_hybrid_solve() {
        let source_block_symbols = extended_source_block_symbols(128);
        let symbol_size = 3;
        let indices: Vec<u32> = (0..source_block_symbols).collect();
        let (matrix, hdpc_rows) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_block_symbols, &indices);
        let mut symbols = SymbolSlab::with_zeros(matrix.height() + hdpc_rows.height(), symbol_size);
        for (index, byte) in symbols.as_mut_bytes().iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(37).wrapping_add(11);
        }

        let direct =
            try_hybrid_binary_hdpc_solve(&matrix, &hdpc_rows, &symbols, source_block_symbols)
                .unwrap();
        let plan = prepare_cached_hybrid_systematic_plan(source_block_symbols, &matrix, &hdpc_rows)
            .unwrap();

        let mut replayed = symbols.clone();
        apply_cached_hybrid_systematic_plan(&plan, &mut replayed);
        let mut binary_slab_replayed = symbols.clone();
        apply_cached_hybrid_systematic_plan_with_binary_slab(&plan, &mut binary_slab_replayed);
        let mut in_place_replayed = symbols;
        apply_cached_hybrid_systematic_plan_in_place(&plan, &mut in_place_replayed);

        assert_eq!(replayed, direct);
        assert_eq!(binary_slab_replayed, direct);
        assert_eq!(in_place_replayed, direct);
    }

    fn first_direct_systematic_source_symbols() -> u32 {
        (1..=SQUARE_HYBRID_MAX_WIDTH as u32)
            .find(|&source_symbols| {
                let width = num_intermediate_symbols(source_symbols) as usize;
                (DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH..=SQUARE_HYBRID_MAX_WIDTH).contains(&width)
            })
            .expect("direct systematic test threshold should be reachable")
    }

    #[test]
    fn threshold_systematic_plan_uses_direct_systematic_solve() {
        let source_symbols = first_direct_systematic_source_symbols();
        let k_prime = extended_source_block_symbols(source_symbols);
        let width = num_intermediate_symbols(source_symbols) as usize;
        let s = num_ldpc_symbols(source_symbols) as usize;
        let h = num_hdpc_symbols(source_symbols) as usize;
        let mut symbols = SymbolSlab::with_zeros(width, 1);
        for isi in 0..source_symbols as usize {
            symbols.get_mut(s + h + isi)[0] = (isi as u8).wrapping_mul(17).wrapping_add(3);
        }
        let original_symbols = symbols.clone();
        let indices: Vec<u32> = (0..k_prime).collect();
        let (matrix, hdpc_rows) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &indices);
        assert_eq!(matrix.width(), width);
        assert!(is_full_systematic_planning_matrix(&matrix, source_symbols));

        let (decoded, ops) = fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_symbols);
        let decoded = decoded.expect("wide direct systematic solve should decode");
        let ops = ops.expect("wide direct systematic solve should be recorded");

        assert!(matches!(
            ops.as_slice(),
            [SymbolOps::DirectSystematicSolve {
                source_block_symbols
            }] if *source_block_symbols == k_prime
        ));

        let mut replayed = original_symbols;
        for op in &ops {
            crate::operation_vector::perform_op(op, &mut replayed);
        }
        assert_eq!(replayed, decoded);
    }

    #[test]
    fn systematic_plan_cache_insert_evicts_old_entries() {
        let mut cache = SystematicPlanCache::default();

        for source_block_symbols in 0..=SYSTEMATIC_PLAN_CACHE_CAPACITY as u32 {
            let plan = std::sync::Arc::new(CachedSystematicPlan {
                forward_steps: Vec::new(),
                forward_dests: CachedSystematicSlices {
                    ranges: Vec::new(),
                    entries: Vec::new(),
                    unit_only: Vec::new(),
                },
                pivot_symbol_cycles: Vec::new(),
                back_substitution: CachedSystematicBackSubstitution::Rows(Vec::new()),
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
    fn cached_single_repair_basis_matches_direct_plan_replay() {
        let source_symbols = 10;
        let missing_isi = 3;
        let width = num_intermediate_symbols(source_symbols) as usize;
        let s = num_ldpc_symbols(source_symbols) as usize;
        let h = num_hdpc_symbols(source_symbols) as usize;

        let mut direct = SymbolSlab::with_zeros(width, 1);
        direct.get_mut(s + h + missing_isi)[0] = 1;
        apply_cached_systematic_plan(source_symbols, &mut direct);

        let cached = cached_single_repair_basis(source_symbols, missing_isi);
        let expected_nonzero_cols = direct
            .as_bytes()
            .iter()
            .enumerate()
            .filter_map(|(col, &coefficient)| (coefficient != 0).then_some(col))
            .collect::<Vec<_>>();

        assert_eq!(cached.coefficients.as_slice(), direct.as_bytes());
        assert_eq!(
            cached.nonzero_cols.as_ref(),
            expected_nonzero_cols.as_slice()
        );

        let cached_again = cached_single_repair_basis(source_symbols, missing_isi);
        assert!(std::sync::Arc::ptr_eq(&cached, &cached_again));
    }

    #[test]
    fn repair_source_coefficients_match_direct_basis_projection() {
        let source_symbols = 10;
        let width = num_intermediate_symbols(source_symbols) as usize;
        let repair_isi = extended_source_block_symbols(source_symbols);
        let plan = cached_systematic_plan(source_symbols);
        let repair_entries = systematic_constraint_row_entries(source_symbols, repair_isi);

        let coefficients = repair_source_coefficients_for_entries(&plan, &repair_entries);

        for source_col in 0..width {
            let basis = apply_prepared_systematic_plan_to_basis_coefficients(&plan, source_col);
            let expected = repair_entries
                .iter()
                .fold(0u8, |acc, &col| acc ^ basis[col]);
            assert_eq!(
                coefficients[source_col], expected,
                "source_col {source_col}"
            );
        }
    }

    #[test]
    fn repair_source_coefficients_handle_batched_back_substitution() {
        let plan = CachedSystematicPlan {
            forward_steps: Vec::new(),
            forward_dests: CachedSystematicSlices {
                ranges: Vec::new(),
                entries: Vec::new(),
                unit_only: Vec::new(),
            },
            pivot_symbol_cycles: Vec::new(),
            back_substitution: CachedSystematicBackSubstitution::Batches(vec![
                Vec::new().into_boxed_slice(),
                vec![(0, Octet::new(7))].into_boxed_slice(),
                vec![(1, Octet::new(3)), (0, Octet::new(5))].into_boxed_slice(),
            ]),
            width: 3,
        };

        let coefficients = repair_source_coefficients_for_entries(&plan, &[0]);

        assert_eq!(
            coefficients,
            vec![
                1,
                7,
                (Octet::new(5) + Octet::new(7) * Octet::new(3)).value()
            ]
        );
    }

    #[test]
    fn scaled_coefficient_add_ignores_zero_source() {
        let mut dest = 0x5a;

        add_scaled_coefficient(&mut dest, 0, Octet::new(7));

        assert_eq!(dest, 0x5a);
    }

    #[test]
    fn single_repair_systematic_decode_recovers_missing_source() {
        let source_symbols = 10;
        let k_prime = extended_source_block_symbols(source_symbols);
        let width = num_intermediate_symbols(source_symbols) as usize;
        let s = num_ldpc_symbols(source_symbols) as usize;
        let h = width - s - k_prime as usize;
        let symbol_size = 3;
        let missing_isi = 3usize;
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
            .filter(|&isi| isi != missing_isi as u32)
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
        let fast_decoded = fast_decoded.unwrap();
        assert_eq!(
            rebuilt_source_symbol(&fast_decoded, source_symbols, missing_isi),
            full_d.get(s + h + missing_isi)
        );

        let (decoded, ops) =
            fused_inverse_mul_symbols(matrix, hdpc_rows, decode_symbols, source_symbols);
        let decoded = decoded.unwrap();
        assert_eq!(
            rebuilt_source_symbol(&decoded, source_symbols, missing_isi),
            full_d.get(s + h + missing_isi)
        );
        assert!(ops.is_none());
    }

    fn rebuilt_source_symbol(
        intermediate_symbols: &SymbolSlab,
        source_symbols: u32,
        source_isi: usize,
    ) -> Vec<u8> {
        let mut rebuilt = vec![0u8; intermediate_symbols.symbol_size()];
        let entries = systematic_constraint_row_entries(source_symbols, source_isi as u32);
        rebuilt.copy_from_slice(intermediate_symbols.get(entries[0]));
        for &col in &entries[1..] {
            add_assign(&mut rebuilt, intermediate_symbols.get(col));
        }
        rebuilt
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

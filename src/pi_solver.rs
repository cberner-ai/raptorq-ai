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
    AddAssignFastPath, FusedAddAssignMulScalarFastPath, add_assign, bytes_are_zero,
    fused_addassign_mul_scalar, fused_mulassign_alpha_add_assign, mulassign_scalar,
};
use crate::operation_vector::SymbolOps;
#[cfg(feature = "std")]
use crate::operation_vector::fused_addassign_symbol_batch;
use crate::rng::RfcRand;
use crate::sparse_matrix::SparseBinaryMatrix;
use crate::symbol_slab::SymbolSlab;
use crate::systematic_constants::num_ldpc_symbols;
use crate::systematic_constants::{
    MAX_SUPPORTED_INTERMEDIATE_SYMBOLS, calculate_p1, extended_source_block_symbols,
    num_hdpc_symbols, num_intermediate_symbols, num_lt_symbols, num_pi_symbols, systematic_index,
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
const CACHED_SYSTEMATIC_PLAN_RECORDING_MIN_WIDTH: usize = 128;
#[cfg(not(feature = "std"))]
const CACHED_SYSTEMATIC_PLAN_RECORDING_MIN_WIDTH: usize = MAX_INLINE_RECORDED_SOLVER_WIDTH + 1;
#[cfg(feature = "std")]
const SYSTEMATIC_PLAN_FORWARD_DESTS_PER_COL_HINT: usize = 96;
const LIGHTEST_PIVOT_MIN_WIDTH: usize = 64;
const COEFFICIENT_BUCKET_SOLVER_MIN_WIDTH: usize = 512;
const SPARSE_SOURCE_MERGE_DEST_FACTOR: usize = 3;
const SPARSE_SOURCE_MERGE_MAX_SOURCE_LEN: usize = 512;
const SPARSE_SOURCE_LINEAR_SCAN_LIMIT: usize = 24;
const SHORT_ROW_LINEAR_SCAN_LIMIT: usize = 8;
const BINARY_FORWARD_SYMBOL_BATCH_MIN_WIDTH: usize = 512;
const BINARY_SOURCE_BATCH_16_MIN_WIDTH: usize = 10_000;
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
const LOW_DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH: usize = 128;
#[cfg(feature = "std")]
const LOW_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH: usize = 500;
#[cfg(feature = "std")]
const MID_DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH: usize = 500;
#[cfg(feature = "std")]
const MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH: usize = 1_500;
#[cfg(feature = "std")]
const TRUSTED_MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH: usize = 2_500;
#[cfg(feature = "std")]
const LOW_DIRECT_HYBRID_REPLAY_WARM_MAX_WIDTH: usize = LOW_DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH;
const DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH: usize = 5_000;
#[cfg(feature = "std")]
const DIRECT_SOURCE_BATCH_BACK_SUBSTITUTION_MIN_WIDTH: usize = 5_000;
#[cfg(feature = "std")]
const DIRECT_FORWARD_NO_ZERO_CHECK_MIN_WIDTH: usize = 10_000;
#[cfg(feature = "std")]
const DIRECT_SOURCE_BATCH_DIRECT_COLLECT_MIN_WIDTH: usize = 20_000;
#[cfg(feature = "std")]
const DIRECT_DECODE_SOURCE_BATCH_BACK_SUBSTITUTION_MIN_WIDTH: usize = 5_000;
#[cfg(feature = "std")]
const CACHED_HDPC_FREE_SOLVE_MAX_WIDTH: usize = 10_000;
#[cfg(feature = "std")]
const DIRECT_CACHED_HDPC_FREE_SOLVE_MAX_WIDTH: usize = HYBRID_MAX_WIDTH;
#[cfg(feature = "std")]
const DIRECT_SQUARE_HYBRID_DECODE_MIN_WIDTH: usize = 128;
#[cfg(all(feature = "std", not(test)))]
const SQUARE_HYBRID_DECODE_MIN_WIDTH: usize = DIRECT_SQUARE_HYBRID_DECODE_MIN_WIDTH;
#[cfg(all(feature = "std", test))]
const SQUARE_HYBRID_DECODE_MIN_WIDTH: usize = 64;
#[cfg(all(feature = "std", not(test)))]
const DIRECT_SINGLE_REPAIR_SYSTEMATIC_MIN_WIDTH: usize = DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH;
#[cfg(all(feature = "std", test))]
const DIRECT_SINGLE_REPAIR_SYSTEMATIC_MIN_WIDTH: usize = 1;
#[cfg(feature = "std")]
const SHORT_PIVOT_MERGE_MAX_LEN: usize = 64;
#[cfg(feature = "std")]
const REPAIR_SOURCE_COEFFICIENTS_CACHE_CAPACITY: usize = 16;
#[cfg(feature = "std")]
const DIRECT_SINGLE_REPAIR_COEFFICIENT_CACHE_CAPACITY: usize = 32;
#[cfg(feature = "std")]
const IN_PLACE_HYBRID_REPLAY_MIN_WIDTH: usize = 512;
#[cfg(feature = "std")]
const IN_PLACE_HYBRID_REPLAY_MAX_MID_WIDTH: usize = 768;
#[cfg(feature = "std")]
const LARGE_IN_PLACE_HYBRID_REPLAY_MIN_WIDTH: usize = 32_768;
#[cfg(not(test))]
const LARGE_BINARY_WEIGHTED_BUCKET_MIN_WIDTH: usize = 4_096;
#[cfg(test)]
const LARGE_BINARY_WEIGHTED_BUCKET_MIN_WIDTH: usize = 64;
const OVERDETERMINED_NO_HDPC_PREFIX_MIN_WIDTH: usize = 5_000;
const OVERDETERMINED_NO_HDPC_PREFIX_OWNED_MIN_WIDTH: usize = 20_000;
const OVERDETERMINED_NO_HDPC_PREFIX_METADATA_MIN_WIDTH: usize = 20_000;
const OVERDETERMINED_NO_HDPC_PREFIX_BACKSUB_BATCH4_MAX_WIDTH: usize = 32_768;
const COLUMN_MAJOR_HDPC_VERIFY_MIN_WIDTH: usize = 256;
#[cfg(feature = "std")]
const HDPC_VERIFY_ROW_PAIRS_CACHE_CAPACITY: usize = 16;
#[cfg(feature = "std")]
const HDPC_VERIFY_ROW_PAIRS_CACHE_MIN_GAMMA_WIDTH: usize = 512;
#[cfg(feature = "std")]
const HDPC_VERIFY_ROW_PAIRS_CACHE_MAX_GAMMA_WIDTH: usize = HYBRID_MAX_WIDTH;
const PLAN_SMALL_WEIGHT_BINARY_BUCKET_MAX: usize = 31;
const DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX: usize = 16;
#[cfg(all(test, feature = "std"))]
const SINGLE_REPAIR_BASIS_CACHE_CAPACITY: usize = 64;

#[cfg(feature = "std")]
#[inline]
fn use_direct_systematic_solve(width: usize) -> bool {
    width >= DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH
        || (LOW_DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH..LOW_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH)
            .contains(&width)
        || (MID_DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH..MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH)
            .contains(&width)
}

#[cfg(feature = "std")]
#[inline]
fn use_trusted_direct_systematic_solve(width: usize) -> bool {
    use_direct_systematic_solve(width)
        || (MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH..TRUSTED_MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH)
            .contains(&width)
}

#[cfg(feature = "std")]
#[inline]
fn use_direct_source_batch_back_substitution(width: usize) -> bool {
    width >= DIRECT_SOURCE_BATCH_BACK_SUBSTITUTION_MIN_WIDTH
        || (LOW_DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH..LOW_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH)
            .contains(&width)
        || (MID_DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH..MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH)
            .contains(&width)
}

#[cfg(feature = "std")]
#[inline]
fn use_trusted_direct_source_batch_back_substitution(width: usize) -> bool {
    width >= DIRECT_SOURCE_BATCH_BACK_SUBSTITUTION_MIN_WIDTH
        || (LOW_DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH..LOW_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH)
            .contains(&width)
        || (MID_DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH..TRUSTED_MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH)
            .contains(&width)
}

#[cfg(feature = "std")]
#[inline]
fn use_direct_forward_no_zero_check(width: usize) -> bool {
    width >= DIRECT_FORWARD_NO_ZERO_CHECK_MIN_WIDTH
}

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
    if recording == OperationRecording::Record
        && matrix.width() >= CACHED_SYSTEMATIC_PLAN_RECORDING_MIN_WIDTH
    {
        let source_block_symbols = extended_source_block_symbols(source_block_symbols);
        #[cfg(feature = "std")]
        {
            if !use_trusted_direct_systematic_solve(matrix.width()) {
                let width = matrix.width();
                if cached_hybrid_systematic_plan_from_matrix(
                    source_block_symbols,
                    &matrix,
                    &hdpc_rows,
                )
                .is_some()
                {
                    let op = SymbolOps::DirectSystematicSolve {
                        source_block_symbols,
                    };
                    if symbol_is_zero(symbols.as_bytes()) {
                        let decoded = SymbolSlab::with_zeros(width, symbols.symbol_size());
                        return (Some(decoded), Some(vec![op]));
                    }
                    let mut decoded = symbols.clone();
                    apply_direct_systematic_solve(source_block_symbols, &mut decoded);
                    return (Some(decoded), Some(vec![op]));
                }
                cached_systematic_plan_from_matrix(source_block_symbols, matrix, &hdpc_rows);
                let op = SymbolOps::ApplyCachedSystematicPlan {
                    source_block_symbols,
                };
                if symbol_is_zero(symbols.as_bytes()) {
                    let decoded = SymbolSlab::with_zeros(width, symbols.symbol_size());
                    return (Some(decoded), Some(vec![op]));
                }
                let mut decoded = symbols.clone();
                apply_cached_systematic_plan(source_block_symbols, &mut decoded);
                return (Some(decoded), Some(vec![op]));
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
        let width = matrix.width();
        let zero_symbols = symbol_is_zero(symbols.as_bytes());
        #[cfg(feature = "std")]
        {
            if zero_symbols
                && width < LOW_DIRECT_HYBRID_REPLAY_WARM_MAX_WIDTH
                && direct_systematic_plan_is_cached(source_block_symbols)
            {
                cached_hybrid_systematic_plan_from_matrix(
                    source_block_symbols,
                    &matrix,
                    &hdpc_rows,
                );
            }
            cached_direct_systematic_plan_from_matrix(source_block_symbols, &matrix, &hdpc_rows);
        }
        if zero_symbols {
            let decoded = SymbolSlab::with_zeros(width, symbols.symbol_size());
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

    #[cfg(feature = "std")]
    let mut symbols = symbols;

    #[cfg(feature = "std")]
    if recording == OperationRecording::Skip
        && square_hybrid_candidate
        && width >= SQUARE_HYBRID_DECODE_MIN_WIDTH
    {
        match try_square_hybrid_binary_hdpc_solve_owned(
            source_block_symbols,
            &matrix,
            &hdpc_rows,
            symbols,
        ) {
            SquareHybridDecodeResult::Decoded(decoded) => return (Some(decoded), None),
            SquareHybridDecodeResult::Failed => return (None, None),
            SquareHybridDecodeResult::Fallback(returned_symbols) => {
                symbols = returned_symbols;
            }
        }
    }

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
    binary_forward_dests: CachedBinarySlices,
    hdpc_symbol_steps: Vec<HybridHdpcSymbolStep>,
    free_cols: Box<[usize]>,
    free_rows: Vec<CoefficientRow>,
    free_solve: Option<CachedHdpcFreeSolve>,
    pivots: Box<[(usize, usize)]>,
    back_substitution: CachedBinarySlices,
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

#[cfg(feature = "std")]
enum SquareHybridDecodeResult {
    Decoded(SymbolSlab),
    Failed,
    Fallback(SymbolSlab),
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
type CachedBinarySliceParts = (
    Vec<(usize, usize)>,
    Vec<usize>,
    Vec<CoefficientColumn>,
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
    hdpc_free_solve: Option<CachedHdpcFreeSolve>,
    free_cols: Box<[CoefficientColumn]>,
    pivot_symbol_moves: Vec<Box<[usize]>>,
    back_substitution: DirectSystematicBackSubstitution,
    trust_source_batch_bounds: bool,
    width: usize,
    s: usize,
    h: usize,
}

#[cfg(feature = "std")]
enum DirectSystematicBackSubstitution {
    DestsBySource {
        slices: DirectSystematicSlices,
        non_empty_sources: Box<[CoefficientColumn]>,
    },
    SourcesByDest {
        slices: DirectSystematicSlices,
        non_empty_dests: Box<[CoefficientColumn]>,
    },
}

#[cfg(feature = "std")]
#[derive(Clone, Copy)]
enum DirectBackSubstitutionLayout {
    DestsBySource,
    SourcesByDest,
}

#[cfg(feature = "std")]
struct DirectSystematicForwardStep {
    pivot_symbol: CoefficientColumn,
}

#[cfg(feature = "std")]
struct CachedHdpcFreeSolve {
    ops: Vec<SymbolOps>,
    height: usize,
    free_count: usize,
}

#[cfg(feature = "std")]
struct DirectSystematicSlices {
    ranges: Vec<(usize, usize)>,
    entries: Vec<CoefficientColumn>,
}

#[cfg(feature = "std")]
struct CachedBinarySlices {
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
impl CachedBinarySlices {
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
#[derive(Default)]
struct HdpcVerifyRowPairsCache {
    pairs: HashMap<HdpcVerifyRowPairsKey, HdpcVerifyRowPairs>,
    insertion_order: VecDeque<HdpcVerifyRowPairsKey>,
}

#[cfg(feature = "std")]
type HdpcVerifyRowPairsCacheLock = Mutex<HdpcVerifyRowPairsCache>;
#[cfg(feature = "std")]
type HdpcVerifyRowPairsKey = (usize, usize);
#[cfg(feature = "std")]
type HdpcVerifyRowPair = (CoefficientColumn, CoefficientColumn);
#[cfg(feature = "std")]
type HdpcVerifyRowPairs = Arc<[HdpcVerifyRowPair]>;

#[cfg(feature = "std")]
fn hdpc_verify_row_pairs_cache() -> &'static HdpcVerifyRowPairsCacheLock {
    static CACHE: OnceLock<HdpcVerifyRowPairsCacheLock> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HdpcVerifyRowPairsCache::default()))
}

#[cfg(feature = "std")]
fn direct_systematic_plan_is_cached(source_block_symbols: u32) -> bool {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    let cache = direct_systematic_plan_cache();
    let guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.plans.contains_key(&source_block_symbols)
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
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct DirectSingleRepairCoefficientKey {
    source_block_symbols: u32,
    missing_isi: usize,
    repair_isi: u32,
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

#[cfg(feature = "std")]
#[derive(Default)]
struct DirectSingleRepairCoefficientCache {
    coefficients: HashMap<DirectSingleRepairCoefficientKey, Octet>,
    insertion_order: VecDeque<DirectSingleRepairCoefficientKey>,
}

#[cfg(feature = "std")]
type DirectSingleRepairCoefficientCacheLock = Mutex<DirectSingleRepairCoefficientCache>;

#[cfg(feature = "std")]
fn direct_single_repair_coefficient_cache() -> &'static DirectSingleRepairCoefficientCacheLock {
    static CACHE: OnceLock<DirectSingleRepairCoefficientCacheLock> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(DirectSingleRepairCoefficientCache::default()))
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

#[cfg(feature = "std")]
fn cached_direct_single_repair_coefficient(
    source_block_symbols: u32,
    missing_isi: usize,
    repair_isi: u32,
    repair_entries: &[usize],
) -> Octet {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    let key = DirectSingleRepairCoefficientKey {
        source_block_symbols,
        missing_isi,
        repair_isi,
    };
    {
        let cache = direct_single_repair_coefficient_cache();
        let guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(&coefficient) = guard.coefficients.get(&key) {
            return coefficient;
        }
    }

    let generated = generate_direct_single_repair_coefficient(
        source_block_symbols,
        missing_isi,
        repair_entries,
    );
    let cache = direct_single_repair_coefficient_cache();
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    insert_direct_single_repair_coefficient(&mut guard, key, generated)
}

#[cfg(feature = "std")]
fn insert_direct_single_repair_coefficient(
    cache: &mut DirectSingleRepairCoefficientCache,
    key: DirectSingleRepairCoefficientKey,
    generated: Octet,
) -> Octet {
    if let Some(&coefficient) = cache.coefficients.get(&key) {
        return coefficient;
    }

    if cache.coefficients.len() >= DIRECT_SINGLE_REPAIR_COEFFICIENT_CACHE_CAPACITY
        && let Some(evicted_key) = cache.insertion_order.pop_front()
    {
        cache.coefficients.remove(&evicted_key);
    }

    cache.insertion_order.push_back(key);
    cache.coefficients.insert(key, generated);
    generated
}

#[cfg(feature = "std")]
fn generate_direct_single_repair_coefficient(
    source_block_symbols: u32,
    missing_isi: usize,
    repair_entries: &[usize],
) -> Octet {
    let source_block_symbols = extended_source_block_symbols(source_block_symbols);
    let width = num_intermediate_symbols(source_block_symbols) as usize;
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let h = crate::systematic_constants::num_hdpc_symbols(source_block_symbols) as usize;

    let mut symbols = SymbolSlab::with_zeros(width, 1);
    symbols.get_mut(s + h + missing_isi)[0] = 1;
    let plan = cached_direct_systematic_plan(source_block_symbols);
    apply_prepared_direct_systematic_plan(&plan, &mut symbols);

    let mut coefficient = 0u8;
    for &entry in repair_entries {
        coefficient ^= symbols.get(entry)[0];
    }
    Octet::new(coefficient)
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
    prepare_direct_systematic_plan_with_small_weight_max::<PLAN_SMALL_WEIGHT_BINARY_BUCKET_MAX, M>(
        matrix,
        hdpc_rows,
        source_block_symbols,
        DirectBackSubstitutionLayout::SourcesByDest,
        true,
    )
}

#[cfg(feature = "std")]
fn prepare_direct_systematic_plan_for_decode<M: BinaryMatrix>(
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    source_block_symbols: u32,
) -> Option<DirectSystematicPlan> {
    prepare_direct_systematic_plan_with_small_weight_max::<DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX, M>(
        matrix,
        hdpc_rows,
        source_block_symbols,
        direct_decode_back_substitution_layout(matrix.width()),
        false,
    )
}

#[cfg(feature = "std")]
fn direct_decode_back_substitution_layout(width: usize) -> DirectBackSubstitutionLayout {
    if width >= DIRECT_DECODE_SOURCE_BATCH_BACK_SUBSTITUTION_MIN_WIDTH
        || (MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH..TRUSTED_MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH)
            .contains(&width)
    {
        DirectBackSubstitutionLayout::SourcesByDest
    } else {
        DirectBackSubstitutionLayout::DestsBySource
    }
}

#[cfg(feature = "std")]
#[inline]
fn use_direct_collect_sources_by_dest(
    width: usize,
    binary_height: usize,
    h: usize,
    layout: DirectBackSubstitutionLayout,
) -> bool {
    binary_height + h == width
        && width >= DIRECT_SOURCE_BATCH_DIRECT_COLLECT_MIN_WIDTH
        && matches!(layout, DirectBackSubstitutionLayout::SourcesByDest)
}

#[cfg(feature = "std")]
fn prepare_direct_systematic_plan_with_small_weight_max<
    const SMALL_WEIGHT_MAX: usize,
    M: BinaryMatrix,
>(
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    source_block_symbols: u32,
    back_substitution_layout: DirectBackSubstitutionLayout,
    trust_source_batch_bounds: bool,
) -> Option<DirectSystematicPlan> {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let h = hdpc_rows.height();
    let width = matrix.width();
    let binary_height = matrix.height();
    let source_batch_back_substitution = if trust_source_batch_bounds {
        use_trusted_direct_source_batch_back_substitution(width)
    } else {
        use_direct_source_batch_back_substitution(width)
            || (MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH
                ..TRUSTED_MID_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH)
                .contains(&width)
    };
    let back_substitution_layout = if matches!(
        back_substitution_layout,
        DirectBackSubstitutionLayout::SourcesByDest
    ) && source_batch_back_substitution
    {
        DirectBackSubstitutionLayout::SourcesByDest
    } else {
        DirectBackSubstitutionLayout::DestsBySource
    };
    assert!(width < NO_COEFFICIENT_COLUMN as usize);
    assert!(binary_height < NO_COEFFICIENT_COLUMN as usize);
    assert_eq!(hdpc_rows.width(), width);
    assert!(binary_height >= s);

    let use_weighted_buckets = width >= LARGE_BINARY_WEIGHTED_BUCKET_MIN_WIDTH;
    let (mut rows, mut row_weights, first_ones) = if use_weighted_buckets {
        matrix.packed_rows_with_row_weights_and_first_ones()
    } else {
        let (rows, first_ones) = matrix.packed_rows_with_first_ones();
        (rows, Vec::new(), first_ones)
    };
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut small_weight_buckets = SmallWeightBinaryBuckets::<u32>::new(
        if use_weighted_buckets { width } else { 0 },
        SMALL_WEIGHT_MAX,
    );
    let mut small_row_cache =
        SmallBinaryRowCache::<SMALL_WEIGHT_MAX>::new(if use_weighted_buckets {
            binary_height
        } else {
            0
        });
    let mut next_in_bucket = vec![NO_BUCKET_ROW; binary_height];
    for (row, first_one) in first_ones.into_iter().enumerate() {
        if let Some(col) = first_one {
            if use_weighted_buckets {
                push_weighted_binary_row_bucket(
                    &row_weights,
                    &mut bucket_heads,
                    &mut small_weight_buckets,
                    &mut next_in_bucket,
                    col,
                    row,
                );
            } else {
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
            }
        }
    }

    let mut pivot_for_col = vec![NO_COEFFICIENT_COLUMN; width];
    let mut is_pivot_row = vec![false; binary_height];
    let mut forward_steps = Vec::with_capacity(width);
    let mut forward_ranges = Vec::with_capacity(width);
    let mut forward_entries = if use_weighted_buckets {
        Vec::with_capacity(width.saturating_mul(SYSTEMATIC_PLAN_FORWARD_DESTS_PER_COL_HINT))
    } else {
        Vec::new()
    };
    for col in 0..width {
        let pivot = if use_weighted_buckets {
            pop_lightest_weighted_binary_row_bucket(
                &row_weights,
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                col,
            )
        } else {
            pop_lightest_binary_row_bucket(&rows, &mut bucket_heads, &mut next_in_bucket, col)
        };
        let Some(pivot) = pivot else {
            continue;
        };
        pivot_for_col[col] = coefficient_col(pivot);
        is_pivot_row[pivot] = true;
        let pivot_weight = if use_weighted_buckets {
            row_weights[pivot]
        } else {
            0
        };

        let dest_start = forward_entries.len();
        while let Some(row) = if use_weighted_buckets {
            pop_weighted_binary_row_bucket(
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                col,
            )
        } else {
            pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col)
        } {
            if use_weighted_buckets {
                let next_col = eliminate_weighted_binary_row::<SMALL_WEIGHT_MAX>(
                    &mut rows,
                    &mut row_weights,
                    &mut small_row_cache,
                    row,
                    pivot,
                    col,
                    pivot_weight,
                );
                if let Some(next_col) = next_col {
                    push_weighted_binary_row_bucket(
                        &row_weights,
                        &mut bucket_heads,
                        &mut small_weight_buckets,
                        &mut next_in_bucket,
                        next_col,
                        row,
                    );
                }
            } else {
                rows.xor_suffix(row, pivot, col);
                if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                    push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
                }
            }
            forward_entries.push(coefficient_col(direct_binary_symbol_index(row, s, h)));
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

    let mut free_cols = Vec::with_capacity(h);
    let mut back_substitution_capacity = 0usize;
    for (col, &pivot) in pivot_for_col.iter().enumerate() {
        if pivot == NO_COEFFICIENT_COLUMN {
            free_cols.push(coefficient_col(col));
        } else if use_weighted_buckets {
            back_substitution_capacity +=
                row_weights[coefficient_col_index(pivot)].saturating_sub(1) as usize;
        }
    }
    if free_cols.len() > h {
        return None;
    }

    let pivot_count = width - free_cols.len();

    let hdpc_coefficient_stride = direct_hdpc_coefficient_stride(h);
    let mut hdpc_coefficients =
        dense_hdpc_coefficient_values_column_major_padded(hdpc_rows, hdpc_coefficient_stride);
    let mut hdpc_update_pivots = Vec::with_capacity(pivot_count);
    let mut hdpc_update_ranges = Vec::with_capacity(pivot_count);
    let mut hdpc_update_entries = Vec::with_capacity(pivot_count.saturating_mul(h));
    let mut hdpc_update_unit_only = Vec::with_capacity(pivot_count);
    let direct_collect_sources_by_dest =
        use_direct_collect_sources_by_dest(width, binary_height, h, back_substitution_layout);
    let mut back_substitution_counts = if direct_collect_sources_by_dest {
        Vec::new()
    } else {
        vec![0usize; width]
    };
    let mut back_substitution_entries = if direct_collect_sources_by_dest {
        Vec::new()
    } else {
        Vec::with_capacity(back_substitution_capacity)
    };
    let mut direct_source_ranges = if direct_collect_sources_by_dest {
        Vec::with_capacity(width)
    } else {
        Vec::new()
    };
    let mut direct_source_entries = if direct_collect_sources_by_dest {
        Vec::with_capacity(back_substitution_capacity)
    } else {
        Vec::new()
    };
    for (col, &pivot) in pivot_for_col.iter().enumerate() {
        let direct_source_start = direct_source_entries.len();
        if pivot == NO_COEFFICIENT_COLUMN {
            if direct_collect_sources_by_dest {
                direct_source_ranges.push((direct_source_start, direct_source_start));
            }
            continue;
        }
        let pivot = coefficient_col_index(pivot);

        let (update_start, update_unit_only) =
            if use_weighted_buckets && (row_weights[pivot] as usize) <= SMALL_WEIGHT_MAX {
                let pivot_entries = small_row_cache.entries(&rows, pivot, col);
                eliminate_direct_hdpc_column_entries_and_collect_back_substitution(
                    pivot_entries,
                    &mut hdpc_coefficients,
                    h,
                    col,
                    direct_back_substitution_work(
                        back_substitution_layout,
                        direct_collect_sources_by_dest,
                        &mut back_substitution_counts,
                        &mut back_substitution_entries,
                        &mut direct_source_entries,
                    ),
                    &mut hdpc_update_entries,
                )
            } else {
                eliminate_direct_hdpc_column_and_collect_back_substitution(
                    &rows,
                    &mut hdpc_coefficients,
                    h,
                    col,
                    pivot,
                    direct_back_substitution_work(
                        back_substitution_layout,
                        direct_collect_sources_by_dest,
                        &mut back_substitution_counts,
                        &mut back_substitution_entries,
                        &mut direct_source_entries,
                    ),
                    &mut hdpc_update_entries,
                )
            };
        if direct_collect_sources_by_dest {
            direct_source_ranges.push((direct_source_start, direct_source_entries.len()));
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
    let hdpc_free_rows = direct_hdpc_free_rows(
        hdpc_coefficients,
        &free_cols,
        width,
        h,
        hdpc_coefficient_stride,
    )?;
    let hdpc_free_solve = (trust_source_batch_bounds
        && width <= DIRECT_CACHED_HDPC_FREE_SOLVE_MAX_WIDTH)
        .then(|| {
            prepare_cached_hdpc_free_solve_from_direct_rows(&hdpc_free_rows, free_cols.len(), 1)
        })
        .flatten();
    let back_substitution_slices = if direct_collect_sources_by_dest {
        debug_assert_eq!(direct_source_ranges.len(), width);
        DirectSystematicSlices {
            ranges: direct_source_ranges,
            entries: direct_source_entries,
        }
    } else {
        prepare_direct_back_substitution_batches(
            back_substitution_counts,
            back_substitution_entries,
        )
    };
    let non_empty_back_substitution_slices =
        direct_non_empty_slice_indices(&back_substitution_slices);
    let back_substitution = match back_substitution_layout {
        DirectBackSubstitutionLayout::DestsBySource => {
            DirectSystematicBackSubstitution::DestsBySource {
                slices: back_substitution_slices,
                non_empty_sources: non_empty_back_substitution_slices,
            }
        }
        DirectBackSubstitutionLayout::SourcesByDest => {
            DirectSystematicBackSubstitution::SourcesByDest {
                slices: back_substitution_slices,
                non_empty_dests: non_empty_back_substitution_slices,
            }
        }
    };
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
        hdpc_free_solve,
        free_cols: free_cols.into_boxed_slice(),
        pivot_symbol_moves,
        back_substitution,
        trust_source_batch_bounds,
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
    for (range_col, entry_col) in back_substitution_entries {
        let range_col = coefficient_col_index(range_col);
        let offset = offsets[range_col];
        debug_assert!(offset < entries_len);
        unsafe {
            entries.as_mut_ptr().add(offset).write(entry_col);
        }
        offsets[range_col] += 1;
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
fn direct_non_empty_slice_indices(slices: &DirectSystematicSlices) -> Box<[CoefficientColumn]> {
    slices
        .ranges
        .iter()
        .enumerate()
        .filter_map(|(index, &(start, end))| (start != end).then_some(coefficient_col(index)))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

#[cfg(feature = "std")]
enum DirectBackSubstitutionWork<'a> {
    Scatter {
        counts: &'a mut [usize],
        entries: &'a mut Vec<(CoefficientColumn, CoefficientColumn)>,
        layout: DirectBackSubstitutionLayout,
    },
    DirectSourcesByDest {
        entries: &'a mut Vec<CoefficientColumn>,
    },
}

#[cfg(feature = "std")]
impl DirectBackSubstitutionWork<'_> {
    fn push(&mut self, dest_col: usize, source_col: usize) {
        match self {
            DirectBackSubstitutionWork::Scatter {
                counts,
                entries,
                layout,
            } => match layout {
                DirectBackSubstitutionLayout::DestsBySource => {
                    counts[source_col] += 1;
                    entries.push((coefficient_col(source_col), coefficient_col(dest_col)));
                }
                DirectBackSubstitutionLayout::SourcesByDest => {
                    counts[dest_col] += 1;
                    entries.push((coefficient_col(dest_col), coefficient_col(source_col)));
                }
            },
            DirectBackSubstitutionWork::DirectSourcesByDest { entries } => {
                entries.push(coefficient_col(source_col));
            }
        }
    }
}

#[cfg(feature = "std")]
fn direct_back_substitution_work<'a>(
    layout: DirectBackSubstitutionLayout,
    direct_collect_sources_by_dest: bool,
    counts: &'a mut [usize],
    entries: &'a mut Vec<(CoefficientColumn, CoefficientColumn)>,
    direct_source_entries: &'a mut Vec<CoefficientColumn>,
) -> DirectBackSubstitutionWork<'a> {
    if direct_collect_sources_by_dest {
        debug_assert!(matches!(
            layout,
            DirectBackSubstitutionLayout::SourcesByDest
        ));
        DirectBackSubstitutionWork::DirectSourcesByDest {
            entries: direct_source_entries,
        }
    } else {
        DirectBackSubstitutionWork::Scatter {
            counts,
            entries,
            layout,
        }
    }
}

#[cfg(feature = "std")]
fn eliminate_direct_hdpc_column_and_collect_back_substitution(
    rows: &PackedBinaryRows,
    hdpc_coefficients: &mut [u8],
    h: usize,
    col: usize,
    pivot: usize,
    mut back_substitution: DirectBackSubstitutionWork<'_>,
    hdpc_update_entries: &mut Vec<(CoefficientColumn, Octet)>,
) -> (usize, bool) {
    let stride = direct_hdpc_coefficient_stride(h);
    let update_start = hdpc_update_entries.len();
    let col_start = col * stride;
    let mut factors = [0u8; 16];
    factors[..h].copy_from_slice(&hdpc_coefficients[col_start..col_start + h]);
    hdpc_coefficients[col_start..col_start + h].fill(0);
    let dependent_start_col = col + 1;

    if stride == 16 {
        let factor_block = u128::from_ne_bytes(factors);
        if factor_block != 0 {
            for (row, &factor) in factors.iter().enumerate() {
                if factor != 0 {
                    hdpc_update_entries.push((coefficient_col(row), Octet::new(factor)));
                }
            }
        }
        rows.visit_ones_at_or_after(pivot, dependent_start_col, |entry_col| {
            back_substitution.push(col, entry_col);
            if factor_block != 0 {
                let entry_start = entry_col * stride;
                let entry_ptr = unsafe { hdpc_coefficients.as_mut_ptr().add(entry_start) };
                let updated = unsafe { entry_ptr.cast::<u128>().read_unaligned() ^ factor_block };
                unsafe {
                    entry_ptr.cast::<u128>().write_unaligned(updated);
                }
            }
        });
    } else {
        let mut nonzero_factors = [(0usize, 0u8); 16];
        let mut nonzero_factor_count = 0usize;
        for (row, &factor) in factors[..h].iter().enumerate() {
            if factor == 0 {
                continue;
            }
            hdpc_update_entries.push((coefficient_col(row), Octet::new(factor)));
            nonzero_factors[nonzero_factor_count] = (row, factor);
            nonzero_factor_count += 1;
        }
        rows.visit_ones_at_or_after(pivot, dependent_start_col, |entry_col| {
            back_substitution.push(col, entry_col);
            for &(row, factor) in &nonzero_factors[..nonzero_factor_count] {
                hdpc_coefficients[entry_col * stride + row] ^= factor;
            }
        });
    }

    let update_unit_only = hdpc_update_entries[update_start..]
        .iter()
        .all(|&(_, factor)| factor == Octet::one());
    (update_start, update_unit_only)
}

#[cfg(feature = "std")]
fn eliminate_direct_hdpc_column_entries_and_collect_back_substitution(
    pivot_entries: &[CoefficientColumn],
    hdpc_coefficients: &mut [u8],
    h: usize,
    col: usize,
    mut back_substitution: DirectBackSubstitutionWork<'_>,
    hdpc_update_entries: &mut Vec<(CoefficientColumn, Octet)>,
) -> (usize, bool) {
    let stride = direct_hdpc_coefficient_stride(h);
    debug_assert_eq!(pivot_entries.first().copied(), Some(coefficient_col(col)));
    let update_start = hdpc_update_entries.len();
    let col_start = col * stride;
    let mut factors = [0u8; 16];
    factors[..h].copy_from_slice(&hdpc_coefficients[col_start..col_start + h]);
    hdpc_coefficients[col_start..col_start + h].fill(0);
    if stride == 16 {
        let factor_block = u128::from_ne_bytes(factors);
        if factor_block != 0 {
            for (row, &factor) in factors.iter().enumerate() {
                if factor != 0 {
                    hdpc_update_entries.push((coefficient_col(row), Octet::new(factor)));
                }
            }
        }
        for &entry_col in &pivot_entries[1..] {
            let entry_col = coefficient_col_index(entry_col);
            back_substitution.push(col, entry_col);
            if factor_block != 0 {
                let entry_start = entry_col * stride;
                let entry_ptr = unsafe { hdpc_coefficients.as_mut_ptr().add(entry_start) };
                let updated = unsafe { entry_ptr.cast::<u128>().read_unaligned() ^ factor_block };
                unsafe {
                    entry_ptr.cast::<u128>().write_unaligned(updated);
                }
            }
        }
    } else {
        let mut nonzero_factors = [(0usize, 0u8); 16];
        let mut nonzero_factor_count = 0usize;
        for (row, &factor) in factors[..h].iter().enumerate() {
            if factor == 0 {
                continue;
            }
            hdpc_update_entries.push((coefficient_col(row), Octet::new(factor)));
            nonzero_factors[nonzero_factor_count] = (row, factor);
            nonzero_factor_count += 1;
        }
        for &entry_col in &pivot_entries[1..] {
            let entry_col = coefficient_col_index(entry_col);
            back_substitution.push(col, entry_col);
            for &(row, factor) in &nonzero_factors[..nonzero_factor_count] {
                hdpc_coefficients[entry_col * stride + row] ^= factor;
            }
        }
    }

    let update_unit_only = hdpc_update_entries[update_start..]
        .iter()
        .all(|&(_, factor)| factor == Octet::one());
    (update_start, update_unit_only)
}

#[cfg(feature = "std")]
fn direct_hdpc_free_rows(
    hdpc_coefficients: Vec<u8>,
    free_cols: &[CoefficientColumn],
    width: usize,
    h: usize,
    stride: usize,
) -> Option<DirectSystematicFreeRows> {
    assert_eq!(hdpc_coefficients.len(), width * stride);

    let mut free_rows = Vec::with_capacity(h);
    for row in 0..h {
        let mut free_row = Vec::with_capacity(free_cols.len());
        for (free_index, &col) in free_cols.iter().enumerate() {
            let value = hdpc_coefficients[coefficient_col_index(col) * stride + row];
            if value == 0 {
                continue;
            }
            free_row.push((coefficient_col(free_index), Octet::new(value)));
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
    let mut offsets = Vec::with_capacity(counts.len());
    let mut next_start = 0usize;
    for count in counts {
        let start = next_start;
        offsets.push(start);
        next_start += count;
        ranges.push((start, next_start));
    }
    let entries = Vec::with_capacity(next_start);
    (ranges, offsets, entries, next_start)
}

#[cfg(feature = "std")]
fn binary_slices_from_counts(counts: Vec<usize>) -> CachedBinarySliceParts {
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
    assert!(
        try_apply_prepared_direct_systematic_plan(plan, symbols),
        "prepared direct systematic HDPC solve failed"
    );
}

#[cfg(feature = "std")]
fn try_apply_prepared_direct_systematic_plan(
    plan: &DirectSystematicPlan,
    symbols: &mut SymbolSlab,
) -> bool {
    assert_eq!(symbols.len(), plan.width);

    let symbol_size = symbols.symbol_size();
    let add_assign_path = AddAssignFastPath::new(symbol_size);
    let fused_mul_path = FusedAddAssignMulScalarFastPath::new(symbol_size);

    if use_direct_forward_no_zero_check(plan.width) {
        for (step_index, step) in plan.forward_steps.iter().enumerate() {
            addassign_direct_symbol_batch_no_zero_check(
                symbols,
                coefficient_col_index(step.pivot_symbol),
                plan.forward_dests.slice(step_index),
                add_assign_path,
            );
        }
    } else {
        for (step_index, step) in plan.forward_steps.iter().enumerate() {
            addassign_direct_symbol_batch(
                symbols,
                coefficient_col_index(step.pivot_symbol),
                plan.forward_dests.slice(step_index),
                add_assign_path,
            );
        }
    }

    let mut hdpc_symbols = SymbolSlab::with_zeros(plan.h, symbol_size);
    for row in 0..plan.h {
        hdpc_symbols
            .get_mut(row)
            .copy_from_slice(symbols.get(plan.s + row));
    }
    for (update_index, &pivot) in plan.hdpc_update_pivots.iter().enumerate() {
        let pivot_symbol = symbols.get(coefficient_col_index(pivot));
        let update_slice = plan.hdpc_updates.slice(update_index);
        if plan.hdpc_updates.unit_only[update_index] {
            addassign_external_symbol_batch(
                &mut hdpc_symbols,
                pivot_symbol,
                update_slice,
                add_assign_path,
            );
        } else {
            for &(row, factor) in update_slice {
                fused_mul_path.apply_nonzero(
                    hdpc_symbols.get_mut(coefficient_col_index(row)),
                    pivot_symbol,
                    &factor,
                );
            }
        }
    }

    let free_values = solve_prepared_hdpc_free_variables(
        &plan.hdpc_free_rows,
        plan.hdpc_free_solve.as_ref(),
        hdpc_symbols,
        plan.free_cols.len(),
        symbol_size,
    );
    let Some(free_values) = free_values else {
        return false;
    };

    move_direct_pivot_symbols_to_columns(plan, symbols);
    for (free_index, &col) in plan.free_cols.iter().enumerate() {
        symbols
            .get_mut(coefficient_col_index(col))
            .copy_from_slice(free_values.get(free_index));
    }
    let use_non_empty_back_substitution = use_direct_systematic_solve(plan.width);
    match &plan.back_substitution {
        DirectSystematicBackSubstitution::DestsBySource {
            slices,
            non_empty_sources,
        } => {
            if use_non_empty_back_substitution {
                for &src in non_empty_sources.iter().rev() {
                    let src = coefficient_col_index(src);
                    addassign_direct_symbol_batch_no_zero_check(
                        symbols,
                        src,
                        slices.slice(src),
                        add_assign_path,
                    );
                }
            } else {
                for src in (0..plan.width).rev() {
                    addassign_direct_symbol_batch_no_zero_check(
                        symbols,
                        src,
                        slices.slice(src),
                        add_assign_path,
                    );
                }
            }
        }
        DirectSystematicBackSubstitution::SourcesByDest {
            slices,
            non_empty_dests,
        } => {
            if use_non_empty_back_substitution {
                for &dest in non_empty_dests.iter().rev() {
                    let dest = coefficient_col_index(dest);
                    addassign_direct_symbol_source_batch_from_plan(
                        plan,
                        symbols,
                        dest,
                        slices.slice(dest),
                        add_assign_path,
                    );
                }
            } else {
                for dest in (0..plan.width).rev() {
                    addassign_direct_symbol_source_batch_from_plan(
                        plan,
                        symbols,
                        dest,
                        slices.slice(dest),
                        add_assign_path,
                    );
                }
            }
        }
    }
    true
}

#[cfg(feature = "std")]
fn addassign_direct_symbol_source_batch_from_plan(
    plan: &DirectSystematicPlan,
    symbols: &mut SymbolSlab,
    dest: usize,
    sources: &[CoefficientColumn],
    add_assign_path: AddAssignFastPath,
) {
    if plan.trust_source_batch_bounds {
        addassign_direct_symbol_source_batch_trusted(symbols, dest, sources, add_assign_path);
    } else {
        addassign_direct_symbol_source_batch(symbols, dest, sources, add_assign_path);
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
    cached_solve: Option<&CachedHdpcFreeSolve>,
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

    if let Some(cached_solve) = cached_solve {
        return apply_cached_hdpc_free_solve(cached_solve, hdpc_symbols);
    }

    let rows = rows.iter().map(|row| row.to_vec()).collect::<Vec<_>>();
    solve_without_recording(rows, free_count, hdpc_symbols).0
}

#[cfg(feature = "std")]
fn prepare_cached_hdpc_free_solve_from_rows(
    rows: &[CoefficientRow],
    free_count: usize,
    symbol_size: usize,
) -> Option<CachedHdpcFreeSolve> {
    if free_count == 0 {
        return None;
    }

    let height = rows.len();
    let symbols = SymbolSlab::with_zeros(height, symbol_size);
    let (_, ops) = solve(
        rows.to_vec(),
        free_count,
        symbols,
        OperationRecording::Record,
    );
    Some(CachedHdpcFreeSolve {
        ops: ops?,
        height,
        free_count,
    })
}

#[cfg(feature = "std")]
fn prepare_cached_hdpc_free_solve_from_direct_rows(
    rows: &DirectSystematicFreeRows,
    free_count: usize,
    symbol_size: usize,
) -> Option<CachedHdpcFreeSolve> {
    if free_count == 0 {
        return None;
    }

    let rows = rows.iter().map(|row| row.to_vec()).collect::<Vec<_>>();
    prepare_cached_hdpc_free_solve_from_rows(&rows, free_count, symbol_size)
}

#[cfg(feature = "std")]
fn apply_cached_hdpc_free_solve(
    cached_solve: &CachedHdpcFreeSolve,
    mut symbols: SymbolSlab,
) -> Option<SymbolSlab> {
    assert_eq!(symbols.len(), cached_solve.height);

    for op in &cached_solve.ops {
        apply_cached_hdpc_free_solve_op(op, &mut symbols);
    }
    for row in cached_solve.free_count..cached_solve.height {
        if !symbol_is_zero(symbols.get(row)) {
            return None;
        }
    }

    let mut free_values = SymbolSlab::with_zeros(cached_solve.free_count, symbols.symbol_size());
    for row in 0..cached_solve.free_count {
        free_values.get_mut(row).copy_from_slice(symbols.get(row));
    }
    Some(free_values)
}

#[cfg(feature = "std")]
fn apply_cached_hdpc_free_solve_op(op: &SymbolOps, symbols: &mut SymbolSlab) {
    match op {
        SymbolOps::Swap(a, b) => {
            symbols.swap_symbols(*a, *b);
        }
        SymbolOps::Scale(row, scalar) => {
            mulassign_scalar(symbols.get_mut(*row), scalar);
        }
        SymbolOps::FusedAdd { dest, src, scalar } => {
            let (src_symbol, dest_symbol) = symbols.get_disjoint_mut(*src, *dest);
            fused_addassign_mul_scalar(dest_symbol, src_symbol, scalar);
        }
        SymbolOps::FusedAddBatch { src, dests } => {
            for &(dest, factor) in dests.iter() {
                let (src_symbol, dest_symbol) = symbols.get_disjoint_mut(*src, dest);
                fused_addassign_mul_scalar(dest_symbol, src_symbol, &factor);
            }
        }
        #[cfg(feature = "std")]
        SymbolOps::ApplyCachedSystematicPlan { .. } | SymbolOps::DirectSystematicSolve { .. } => {
            unreachable!("HDPC free solve plan contains only row operations");
        }
    }
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
    addassign_direct_symbol_batch_impl::<true>(symbols, src, dests, add_assign_path);
}

#[cfg(feature = "std")]
fn addassign_direct_symbol_batch_no_zero_check(
    symbols: &mut SymbolSlab,
    src: usize,
    dests: &[CoefficientColumn],
    add_assign_path: AddAssignFastPath,
) {
    addassign_direct_symbol_batch_impl::<false>(symbols, src, dests, add_assign_path);
}

#[cfg(feature = "std")]
fn addassign_external_symbol_batch(
    symbols: &mut SymbolSlab,
    src_symbol: &[u8],
    dests: &[(CoefficientColumn, Octet)],
    add_assign_path: AddAssignFastPath,
) {
    if dests.is_empty() {
        return;
    }

    let symbol_size = symbols.symbol_size();
    assert_eq!(src_symbol.len(), symbol_size);
    let bytes = symbols.as_mut_bytes();
    let bytes_ptr = bytes.as_mut_ptr();
    let src_ptr = src_symbol.as_ptr();

    let mut dest_chunks = dests.chunks_exact(8);
    for chunk in dest_chunks.by_ref() {
        debug_assert!(chunk.iter().all(|&(_, factor)| factor == Octet::one()));
        let dest0 = coefficient_col_index(chunk[0].0);
        let dest1 = coefficient_col_index(chunk[1].0);
        let dest2 = coefficient_col_index(chunk[2].0);
        let dest3 = coefficient_col_index(chunk[3].0);
        let dest4 = coefficient_col_index(chunk[4].0);
        let dest5 = coefficient_col_index(chunk[5].0);
        let dest6 = coefficient_col_index(chunk[6].0);
        let dest7 = coefficient_col_index(chunk[7].0);
        let dest0_start = dest0 * symbol_size;
        let dest1_start = dest1 * symbol_size;
        let dest2_start = dest2 * symbol_size;
        let dest3_start = dest3 * symbol_size;
        let dest4_start = dest4 * symbol_size;
        let dest5_start = dest5 * symbol_size;
        let dest6_start = dest6 * symbol_size;
        let dest7_start = dest7 * symbol_size;
        assert!(dest0_start + symbol_size <= bytes.len());
        assert!(dest1_start + symbol_size <= bytes.len());
        assert!(dest2_start + symbol_size <= bytes.len());
        assert!(dest3_start + symbol_size <= bytes.len());
        assert!(dest4_start + symbol_size <= bytes.len());
        assert!(dest5_start + symbol_size <= bytes.len());
        assert!(dest6_start + symbol_size <= bytes.len());
        assert!(dest7_start + symbol_size <= bytes.len());
        unsafe {
            add_assign_path.apply_same_len_raw_8(
                [
                    bytes_ptr.add(dest0_start),
                    bytes_ptr.add(dest1_start),
                    bytes_ptr.add(dest2_start),
                    bytes_ptr.add(dest3_start),
                    bytes_ptr.add(dest4_start),
                    bytes_ptr.add(dest5_start),
                    bytes_ptr.add(dest6_start),
                    bytes_ptr.add(dest7_start),
                ],
                src_ptr,
                symbol_size,
            );
        }
    }

    let mut dest_chunks = dest_chunks.remainder().chunks_exact(4);
    for chunk in dest_chunks.by_ref() {
        debug_assert!(chunk.iter().all(|&(_, factor)| factor == Octet::one()));
        let dest0 = coefficient_col_index(chunk[0].0);
        let dest1 = coefficient_col_index(chunk[1].0);
        let dest2 = coefficient_col_index(chunk[2].0);
        let dest3 = coefficient_col_index(chunk[3].0);
        let dest0_start = dest0 * symbol_size;
        let dest1_start = dest1 * symbol_size;
        let dest2_start = dest2 * symbol_size;
        let dest3_start = dest3 * symbol_size;
        assert!(dest0_start + symbol_size <= bytes.len());
        assert!(dest1_start + symbol_size <= bytes.len());
        assert!(dest2_start + symbol_size <= bytes.len());
        assert!(dest3_start + symbol_size <= bytes.len());
        unsafe {
            add_assign_path.apply_same_len_raw_4(
                [
                    bytes_ptr.add(dest0_start),
                    bytes_ptr.add(dest1_start),
                    bytes_ptr.add(dest2_start),
                    bytes_ptr.add(dest3_start),
                ],
                src_ptr,
                symbol_size,
            );
        }
    }

    for &(dest, factor) in dest_chunks.remainder() {
        debug_assert_eq!(factor, Octet::one());
        let dest = coefficient_col_index(dest);
        let dest_start = dest * symbol_size;
        assert!(dest_start + symbol_size <= bytes.len());
        unsafe {
            let dest_symbol =
                core::slice::from_raw_parts_mut(bytes_ptr.add(dest_start), symbol_size);
            add_assign_path.apply_same_len(dest_symbol, src_symbol);
        }
    }
}

#[cfg(feature = "std")]
fn addassign_direct_symbol_source_batch(
    symbols: &mut SymbolSlab,
    dest: usize,
    sources: &[CoefficientColumn],
    add_assign_path: AddAssignFastPath,
) {
    if sources.is_empty() {
        return;
    }

    let symbol_size = symbols.symbol_size();
    let bytes = symbols.as_mut_bytes();
    let dest_start = dest * symbol_size;
    assert!(dest_start + symbol_size <= bytes.len());
    debug_assert_direct_symbol_source_batch_dest(dest, sources);
    let dest_ptr = unsafe { bytes.as_mut_ptr().add(dest_start) };
    addassign_symbol_sources_raw(
        dest_ptr,
        bytes.as_ptr(),
        bytes.len(),
        symbol_size,
        sources,
        add_assign_path,
    );
}

#[cfg(feature = "std")]
fn addassign_direct_symbol_source_batch_trusted(
    symbols: &mut SymbolSlab,
    dest: usize,
    sources: &[CoefficientColumn],
    add_assign_path: AddAssignFastPath,
) {
    if sources.is_empty() {
        return;
    }

    let symbol_size = symbols.symbol_size();
    let bytes = symbols.as_mut_bytes();
    let dest_start = dest * symbol_size;
    assert!(dest_start + symbol_size <= bytes.len());
    debug_assert_direct_symbol_source_batch_dest(dest, sources);
    let dest_ptr = unsafe { bytes.as_mut_ptr().add(dest_start) };
    addassign_symbol_sources_raw_trusted(
        dest_ptr,
        bytes.as_ptr(),
        bytes.len(),
        symbol_size,
        sources,
        add_assign_path,
    );
}

#[cfg(feature = "std")]
fn addassign_direct_symbol_batch_impl<const CHECK_ZERO_SOURCE: bool>(
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
    if CHECK_ZERO_SOURCE && bytes_are_zero(src_symbol) {
        return;
    }
    let bytes_ptr = bytes.as_mut_ptr();

    let mut dest_chunks = dests.chunks_exact(8);
    for chunk in dest_chunks.by_ref() {
        let dest0 = coefficient_col_index(chunk[0]);
        let dest1 = coefficient_col_index(chunk[1]);
        let dest2 = coefficient_col_index(chunk[2]);
        let dest3 = coefficient_col_index(chunk[3]);
        let dest4 = coefficient_col_index(chunk[4]);
        let dest5 = coefficient_col_index(chunk[5]);
        let dest6 = coefficient_col_index(chunk[6]);
        let dest7 = coefficient_col_index(chunk[7]);
        debug_assert_direct_batch_dests(
            src,
            &[dest0, dest1, dest2, dest3, dest4, dest5, dest6, dest7],
        );
        let dest0_start = dest0 * symbol_size;
        let dest1_start = dest1 * symbol_size;
        let dest2_start = dest2 * symbol_size;
        let dest3_start = dest3 * symbol_size;
        let dest4_start = dest4 * symbol_size;
        let dest5_start = dest5 * symbol_size;
        let dest6_start = dest6 * symbol_size;
        let dest7_start = dest7 * symbol_size;
        assert!(dest0_start + symbol_size <= bytes.len());
        assert!(dest1_start + symbol_size <= bytes.len());
        assert!(dest2_start + symbol_size <= bytes.len());
        assert!(dest3_start + symbol_size <= bytes.len());
        assert!(dest4_start + symbol_size <= bytes.len());
        assert!(dest5_start + symbol_size <= bytes.len());
        assert!(dest6_start + symbol_size <= bytes.len());
        assert!(dest7_start + symbol_size <= bytes.len());
        unsafe {
            add_assign_path.apply_same_len_raw_8(
                [
                    bytes_ptr.add(dest0_start),
                    bytes_ptr.add(dest1_start),
                    bytes_ptr.add(dest2_start),
                    bytes_ptr.add(dest3_start),
                    bytes_ptr.add(dest4_start),
                    bytes_ptr.add(dest5_start),
                    bytes_ptr.add(dest6_start),
                    bytes_ptr.add(dest7_start),
                ],
                src_ptr,
                symbol_size,
            );
        }
    }

    let mut dest_chunks = dest_chunks.remainder().chunks_exact(4);
    for chunk in dest_chunks.by_ref() {
        let dest0 = coefficient_col_index(chunk[0]);
        let dest1 = coefficient_col_index(chunk[1]);
        let dest2 = coefficient_col_index(chunk[2]);
        let dest3 = coefficient_col_index(chunk[3]);
        debug_assert_direct_batch_dests(src, &[dest0, dest1, dest2, dest3]);
        let dest0_start = dest0 * symbol_size;
        let dest1_start = dest1 * symbol_size;
        let dest2_start = dest2 * symbol_size;
        let dest3_start = dest3 * symbol_size;
        assert!(dest0_start + symbol_size <= bytes.len());
        assert!(dest1_start + symbol_size <= bytes.len());
        assert!(dest2_start + symbol_size <= bytes.len());
        assert!(dest3_start + symbol_size <= bytes.len());
        unsafe {
            add_assign_path.apply_same_len_raw_4(
                [
                    bytes_ptr.add(dest0_start),
                    bytes_ptr.add(dest1_start),
                    bytes_ptr.add(dest2_start),
                    bytes_ptr.add(dest3_start),
                ],
                src_ptr,
                symbol_size,
            );
        }
    }

    for &dest in dest_chunks.remainder() {
        let dest = coefficient_col_index(dest);
        assert_ne!(dest, src);
        let dest_start = dest * symbol_size;
        assert!(dest_start + symbol_size <= bytes.len());
        unsafe {
            let dest_symbol =
                core::slice::from_raw_parts_mut(bytes_ptr.add(dest_start), symbol_size);
            add_assign_path.apply_same_len(dest_symbol, src_symbol);
        }
    }
}

fn addassign_symbol_row_batch<const CHECK_ZERO_SOURCE: bool>(
    symbols: &mut SymbolSlab,
    src: usize,
    dests: &[usize],
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
    if CHECK_ZERO_SOURCE && bytes_are_zero(src_symbol) {
        return;
    }
    let bytes_ptr = bytes.as_mut_ptr();

    let mut dest_chunks = dests.chunks_exact(8);
    for chunk in dest_chunks.by_ref() {
        assert_symbol_row_batch_dests(src, chunk);
        let dest0_start = chunk[0] * symbol_size;
        let dest1_start = chunk[1] * symbol_size;
        let dest2_start = chunk[2] * symbol_size;
        let dest3_start = chunk[3] * symbol_size;
        let dest4_start = chunk[4] * symbol_size;
        let dest5_start = chunk[5] * symbol_size;
        let dest6_start = chunk[6] * symbol_size;
        let dest7_start = chunk[7] * symbol_size;
        assert!(dest0_start + symbol_size <= bytes.len());
        assert!(dest1_start + symbol_size <= bytes.len());
        assert!(dest2_start + symbol_size <= bytes.len());
        assert!(dest3_start + symbol_size <= bytes.len());
        assert!(dest4_start + symbol_size <= bytes.len());
        assert!(dest5_start + symbol_size <= bytes.len());
        assert!(dest6_start + symbol_size <= bytes.len());
        assert!(dest7_start + symbol_size <= bytes.len());
        unsafe {
            add_assign_path.apply_same_len_raw_8(
                [
                    bytes_ptr.add(dest0_start),
                    bytes_ptr.add(dest1_start),
                    bytes_ptr.add(dest2_start),
                    bytes_ptr.add(dest3_start),
                    bytes_ptr.add(dest4_start),
                    bytes_ptr.add(dest5_start),
                    bytes_ptr.add(dest6_start),
                    bytes_ptr.add(dest7_start),
                ],
                src_ptr,
                symbol_size,
            );
        }
    }

    let mut dest_chunks = dest_chunks.remainder().chunks_exact(4);
    for chunk in dest_chunks.by_ref() {
        assert_symbol_row_batch_dests(src, chunk);
        let dest0_start = chunk[0] * symbol_size;
        let dest1_start = chunk[1] * symbol_size;
        let dest2_start = chunk[2] * symbol_size;
        let dest3_start = chunk[3] * symbol_size;
        assert!(dest0_start + symbol_size <= bytes.len());
        assert!(dest1_start + symbol_size <= bytes.len());
        assert!(dest2_start + symbol_size <= bytes.len());
        assert!(dest3_start + symbol_size <= bytes.len());
        unsafe {
            add_assign_path.apply_same_len_raw_4(
                [
                    bytes_ptr.add(dest0_start),
                    bytes_ptr.add(dest1_start),
                    bytes_ptr.add(dest2_start),
                    bytes_ptr.add(dest3_start),
                ],
                src_ptr,
                symbol_size,
            );
        }
    }

    for &dest in dest_chunks.remainder() {
        assert_ne!(dest, src);
        let dest_start = dest * symbol_size;
        assert!(dest_start + symbol_size <= bytes.len());
        unsafe {
            let dest_symbol =
                core::slice::from_raw_parts_mut(bytes_ptr.add(dest_start), symbol_size);
            add_assign_path.apply_same_len(dest_symbol, src_symbol);
        }
    }
}

fn addassign_symbol_source_batch(
    symbols: &mut SymbolSlab,
    dest: usize,
    sources: &[usize],
    add_assign_path: AddAssignFastPath,
) {
    if sources.is_empty() {
        return;
    }

    let symbol_size = symbols.symbol_size();
    let bytes = symbols.as_mut_bytes();
    let dest_start = dest * symbol_size;
    assert!(dest_start + symbol_size <= bytes.len());
    assert_symbol_source_batch_dest(dest, sources);
    let dest_ptr = unsafe { bytes.as_mut_ptr().add(dest_start) };
    addassign_symbol_sources_raw(
        dest_ptr,
        bytes.as_ptr(),
        bytes.len(),
        symbol_size,
        sources,
        add_assign_path,
    );
}

fn addassign_symbol_sources_to_slice(
    dest: &mut [u8],
    symbols: &SymbolSlab,
    sources: &[usize],
    add_assign_path: AddAssignFastPath,
) {
    if sources.is_empty() {
        return;
    }

    let symbol_size = symbols.symbol_size();
    assert_eq!(dest.len(), symbol_size);
    addassign_symbol_sources_raw(
        dest.as_mut_ptr(),
        symbols.as_bytes().as_ptr(),
        symbols.as_bytes().len(),
        symbol_size,
        sources,
        add_assign_path,
    );
}

trait SymbolSourceIndex {
    fn symbol_source_index(self) -> usize;
}

impl SymbolSourceIndex for usize {
    fn symbol_source_index(self) -> usize {
        self
    }
}

impl SymbolSourceIndex for CoefficientColumn {
    fn symbol_source_index(self) -> usize {
        coefficient_col_index(self)
    }
}

fn addassign_symbol_sources_raw<T: Copy + SymbolSourceIndex>(
    dest_ptr: *mut u8,
    source_base: *const u8,
    source_len: usize,
    symbol_size: usize,
    sources: &[T],
    add_assign_path: AddAssignFastPath,
) {
    addassign_symbol_sources_raw_impl::<true, T>(
        dest_ptr,
        source_base,
        source_len,
        symbol_size,
        sources,
        add_assign_path,
    );
}

fn addassign_symbol_sources_raw_trusted<T: Copy + SymbolSourceIndex>(
    dest_ptr: *mut u8,
    source_base: *const u8,
    source_len: usize,
    symbol_size: usize,
    sources: &[T],
    add_assign_path: AddAssignFastPath,
) {
    addassign_symbol_sources_raw_impl::<false, T>(
        dest_ptr,
        source_base,
        source_len,
        symbol_size,
        sources,
        add_assign_path,
    );
}

fn addassign_symbol_sources_raw_impl<
    const CHECK_SOURCE_BOUNDS: bool,
    T: Copy + SymbolSourceIndex,
>(
    dest_ptr: *mut u8,
    source_base: *const u8,
    source_len: usize,
    symbol_size: usize,
    sources: &[T],
    add_assign_path: AddAssignFastPath,
) {
    let mut source_chunks = sources.chunks_exact(16);
    for chunk in source_chunks.by_ref() {
        let src0_start = chunk[0].symbol_source_index() * symbol_size;
        let src1_start = chunk[1].symbol_source_index() * symbol_size;
        let src2_start = chunk[2].symbol_source_index() * symbol_size;
        let src3_start = chunk[3].symbol_source_index() * symbol_size;
        let src4_start = chunk[4].symbol_source_index() * symbol_size;
        let src5_start = chunk[5].symbol_source_index() * symbol_size;
        let src6_start = chunk[6].symbol_source_index() * symbol_size;
        let src7_start = chunk[7].symbol_source_index() * symbol_size;
        let src8_start = chunk[8].symbol_source_index() * symbol_size;
        let src9_start = chunk[9].symbol_source_index() * symbol_size;
        let src10_start = chunk[10].symbol_source_index() * symbol_size;
        let src11_start = chunk[11].symbol_source_index() * symbol_size;
        let src12_start = chunk[12].symbol_source_index() * symbol_size;
        let src13_start = chunk[13].symbol_source_index() * symbol_size;
        let src14_start = chunk[14].symbol_source_index() * symbol_size;
        let src15_start = chunk[15].symbol_source_index() * symbol_size;
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src0_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src1_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src2_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src3_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src4_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src5_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src6_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src7_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src8_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src9_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src10_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src11_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src12_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src13_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src14_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src15_start, symbol_size, source_len);
        unsafe {
            add_assign_path.apply_sources_same_len_raw_16(
                dest_ptr,
                [
                    source_base.add(src0_start),
                    source_base.add(src1_start),
                    source_base.add(src2_start),
                    source_base.add(src3_start),
                    source_base.add(src4_start),
                    source_base.add(src5_start),
                    source_base.add(src6_start),
                    source_base.add(src7_start),
                    source_base.add(src8_start),
                    source_base.add(src9_start),
                    source_base.add(src10_start),
                    source_base.add(src11_start),
                    source_base.add(src12_start),
                    source_base.add(src13_start),
                    source_base.add(src14_start),
                    source_base.add(src15_start),
                ],
                symbol_size,
            );
        }
    }

    let mut source_chunks = source_chunks.remainder().chunks_exact(8);
    for chunk in source_chunks.by_ref() {
        let src0_start = chunk[0].symbol_source_index() * symbol_size;
        let src1_start = chunk[1].symbol_source_index() * symbol_size;
        let src2_start = chunk[2].symbol_source_index() * symbol_size;
        let src3_start = chunk[3].symbol_source_index() * symbol_size;
        let src4_start = chunk[4].symbol_source_index() * symbol_size;
        let src5_start = chunk[5].symbol_source_index() * symbol_size;
        let src6_start = chunk[6].symbol_source_index() * symbol_size;
        let src7_start = chunk[7].symbol_source_index() * symbol_size;
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src0_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src1_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src2_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src3_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src4_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src5_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src6_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src7_start, symbol_size, source_len);
        unsafe {
            add_assign_path.apply_sources_same_len_raw_8(
                dest_ptr,
                [
                    source_base.add(src0_start),
                    source_base.add(src1_start),
                    source_base.add(src2_start),
                    source_base.add(src3_start),
                    source_base.add(src4_start),
                    source_base.add(src5_start),
                    source_base.add(src6_start),
                    source_base.add(src7_start),
                ],
                symbol_size,
            );
        }
    }

    let mut source_chunks = source_chunks.remainder().chunks_exact(4);
    for chunk in source_chunks.by_ref() {
        let src0_start = chunk[0].symbol_source_index() * symbol_size;
        let src1_start = chunk[1].symbol_source_index() * symbol_size;
        let src2_start = chunk[2].symbol_source_index() * symbol_size;
        let src3_start = chunk[3].symbol_source_index() * symbol_size;
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src0_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src1_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src2_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src3_start, symbol_size, source_len);
        unsafe {
            add_assign_path.apply_sources_same_len_raw_4(
                dest_ptr,
                [
                    source_base.add(src0_start),
                    source_base.add(src1_start),
                    source_base.add(src2_start),
                    source_base.add(src3_start),
                ],
                symbol_size,
            );
        }
    }

    let mut source_chunks = source_chunks.remainder().chunks_exact(3);
    for chunk in source_chunks.by_ref() {
        let src0_start = chunk[0].symbol_source_index() * symbol_size;
        let src1_start = chunk[1].symbol_source_index() * symbol_size;
        let src2_start = chunk[2].symbol_source_index() * symbol_size;
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src0_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src1_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src2_start, symbol_size, source_len);
        unsafe {
            add_assign_path.apply_sources_same_len_raw_3(
                dest_ptr,
                [
                    source_base.add(src0_start),
                    source_base.add(src1_start),
                    source_base.add(src2_start),
                ],
                symbol_size,
            );
        }
    }

    let mut source_chunks = source_chunks.remainder().chunks_exact(2);
    for chunk in source_chunks.by_ref() {
        let src0_start = chunk[0].symbol_source_index() * symbol_size;
        let src1_start = chunk[1].symbol_source_index() * symbol_size;
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src0_start, symbol_size, source_len);
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(src1_start, symbol_size, source_len);
        unsafe {
            add_assign_path.apply_sources_same_len_raw_2(
                dest_ptr,
                [source_base.add(src0_start), source_base.add(src1_start)],
                symbol_size,
            );
        }
    }

    for &source in source_chunks.remainder() {
        let source_start = source.symbol_source_index() * symbol_size;
        check_source_bounds::<CHECK_SOURCE_BOUNDS>(source_start, symbol_size, source_len);
        unsafe {
            let dest_symbol = core::slice::from_raw_parts_mut(dest_ptr, symbol_size);
            let source_symbol =
                core::slice::from_raw_parts(source_base.add(source_start), symbol_size);
            add_assign_path.apply_same_len(dest_symbol, source_symbol);
        }
    }
}

#[inline(always)]
fn check_source_bounds<const CHECK_SOURCE_BOUNDS: bool>(
    source_start: usize,
    symbol_size: usize,
    source_len: usize,
) {
    if CHECK_SOURCE_BOUNDS {
        assert!(source_start + symbol_size <= source_len);
    } else {
        debug_assert!(source_start + symbol_size <= source_len);
    }
}

fn assert_symbol_source_batch_dest(dest: usize, sources: &[usize]) {
    for &source in sources {
        assert_ne!(dest, source);
    }
}

#[cfg(feature = "std")]
fn debug_assert_direct_symbol_source_batch_dest(dest: usize, sources: &[CoefficientColumn]) {
    for &source in sources {
        debug_assert_ne!(dest, coefficient_col_index(source));
    }
}

fn assert_symbol_row_batch_dests(src: usize, dests: &[usize]) {
    for (index, &dest) in dests.iter().enumerate() {
        assert_ne!(dest, src);
        for &other in &dests[..index] {
            assert_ne!(dest, other);
        }
    }
}

#[cfg(feature = "std")]
fn debug_assert_direct_batch_dests(src: usize, dests: &[usize]) {
    for (index, &dest) in dests.iter().enumerate() {
        debug_assert_ne!(dest, src);
        for &other in &dests[..index] {
            debug_assert_ne!(dest, other);
        }
    }
}

#[cfg(feature = "std")]
fn fused_addassign_cached_binary_symbol_batch(
    symbols: &mut SymbolSlab,
    src: usize,
    dests: &[CoefficientColumn],
    add_assign_path: AddAssignFastPath,
) {
    addassign_direct_symbol_batch(symbols, src, dests, add_assign_path);
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
    if use_in_place_hybrid_replay(plan.width) {
        apply_cached_hybrid_systematic_plan_in_place(plan, symbols);
    } else {
        apply_cached_hybrid_systematic_plan_with_binary_slab(plan, symbols);
    }
}

#[cfg(feature = "std")]
fn use_in_place_hybrid_replay(width: usize) -> bool {
    (IN_PLACE_HYBRID_REPLAY_MIN_WIDTH..IN_PLACE_HYBRID_REPLAY_MAX_MID_WIDTH).contains(&width)
        || width >= LARGE_IN_PLACE_HYBRID_REPLAY_MIN_WIDTH
}

#[cfg(feature = "std")]
fn apply_cached_hybrid_systematic_plan_with_binary_slab(
    plan: &CachedHybridSystematicPlan,
    symbols: &mut SymbolSlab,
) {
    assert_eq!(symbols.len(), plan.width);

    let symbol_size = symbols.symbol_size();
    let add_assign_path = AddAssignFastPath::new(symbol_size);
    let fused_mul_path = FusedAddAssignMulScalarFastPath::new(symbol_size);
    let binary_height = plan.width - plan.h;
    let mut binary_symbols = SymbolSlab::with_zeros(binary_height, symbol_size);
    let symbol_bytes = symbols.as_bytes();
    let low_binary_bytes = plan.s * symbol_size;
    binary_symbols.copy_block_from(0, &symbol_bytes[..low_binary_bytes]);
    let high_binary_start = (plan.s + plan.h) * symbol_size;
    binary_symbols.copy_block_from(plan.s, &symbol_bytes[high_binary_start..]);

    for (step_index, &(_, pivot)) in plan.pivots.iter().enumerate() {
        fused_addassign_cached_binary_symbol_batch(
            &mut binary_symbols,
            pivot,
            plan.binary_forward_dests.slice(step_index),
            add_assign_path,
        );
    }

    let mut hdpc_symbols = SymbolSlab::with_zeros(plan.h, symbol_size);
    let hdpc_start = plan.s * symbol_size;
    let hdpc_end = hdpc_start + plan.h * symbol_size;
    hdpc_symbols.copy_block_from(0, &symbol_bytes[hdpc_start..hdpc_end]);
    for step in &plan.hdpc_symbol_steps {
        fused_mul_path.apply_nonzero(
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
    } else if let Some(free_solve) = &plan.free_solve {
        apply_cached_hdpc_free_solve(free_solve, hdpc_symbols)
            .expect("cached hybrid systematic free-column solve failed")
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
        fused_addassign_cached_binary_symbol_batch(
            &mut decoded,
            src,
            plan.back_substitution.slice(src),
            add_assign_path,
        );
    }

    symbols.copy_block_from(0, decoded.as_bytes());
}

#[cfg(feature = "std")]
fn apply_cached_hybrid_systematic_plan_in_place(
    plan: &CachedHybridSystematicPlan,
    symbols: &mut SymbolSlab,
) {
    assert!(
        try_apply_cached_hybrid_systematic_plan_in_place(plan, symbols),
        "cached hybrid systematic in-place solve failed"
    );
}

#[cfg(feature = "std")]
fn try_apply_cached_hybrid_systematic_plan_in_place(
    plan: &CachedHybridSystematicPlan,
    symbols: &mut SymbolSlab,
) -> bool {
    assert_eq!(symbols.len(), plan.width);

    let symbol_size = symbols.symbol_size();
    let add_assign_path = AddAssignFastPath::new(symbol_size);
    let fused_mul_path = FusedAddAssignMulScalarFastPath::new(symbol_size);
    for (step_index, &(_, pivot)) in plan.pivots.iter().enumerate() {
        addassign_mapped_binary_symbol_batch(
            symbols,
            pivot,
            plan.binary_forward_dests.slice(step_index),
            plan.s,
            plan.h,
            add_assign_path,
        );
    }

    for step in &plan.hdpc_symbol_steps {
        let src = mapped_binary_symbol_row(step.pivot, plan.s, plan.h);
        let dest = plan.s + step.row;
        let (src_symbol, dest_symbol) = symbols.get_disjoint_mut(src, dest);
        fused_mul_path.apply_nonzero(dest_symbol, src_symbol, &step.factor);
    }

    let free_values = if plan.free_cols.is_empty() {
        if !(0..plan.h).all(|row| symbol_is_zero(symbols.get(plan.s + row))) {
            return false;
        }
        SymbolSlab::with_zeros(0, symbol_size)
    } else {
        let mut hdpc_symbols = SymbolSlab::with_zeros(plan.h, symbol_size);
        for row in 0..plan.h {
            hdpc_symbols
                .get_mut(row)
                .copy_from_slice(symbols.get(plan.s + row));
        }
        let Some(free_values) = (if let Some(free_solve) = &plan.free_solve {
            apply_cached_hdpc_free_solve(free_solve, hdpc_symbols)
        } else {
            solve_without_recording(plan.free_rows.clone(), plan.free_cols.len(), hdpc_symbols).0
        }) else {
            return false;
        };
        free_values
    };

    if let Some(output_symbol_cycles) = &plan.output_symbol_cycles {
        for (free_index, _) in plan.free_cols.iter().enumerate() {
            symbols
                .get_mut(plan.s + free_index)
                .copy_from_slice(free_values.get(free_index));
        }
        move_pivot_symbols_to_columns(symbols, output_symbol_cycles);
        for src in (0..plan.width).rev() {
            fused_addassign_cached_binary_symbol_batch(
                symbols,
                src,
                plan.back_substitution.slice(src),
                add_assign_path,
            );
        }
        return true;
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
        fused_addassign_cached_binary_symbol_batch(
            &mut decoded,
            src,
            plan.back_substitution.slice(src),
            add_assign_path,
        );
    }

    symbols.copy_block_from(0, decoded.as_bytes());
    true
}

#[cfg(feature = "std")]
fn mapped_binary_symbol_row(row: usize, s: usize, h: usize) -> usize {
    if row < s { row } else { row + h }
}

#[cfg(feature = "std")]
fn addassign_mapped_binary_symbol_batch(
    symbols: &mut SymbolSlab,
    src: usize,
    dests: &[CoefficientColumn],
    s: usize,
    h: usize,
    add_assign_path: AddAssignFastPath,
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

    let mut dest_chunks = dests.chunks_exact(8);
    for chunk in dest_chunks.by_ref() {
        let dest0 = mapped_binary_symbol_row(coefficient_col_index(chunk[0]), s, h);
        let dest1 = mapped_binary_symbol_row(coefficient_col_index(chunk[1]), s, h);
        let dest2 = mapped_binary_symbol_row(coefficient_col_index(chunk[2]), s, h);
        let dest3 = mapped_binary_symbol_row(coefficient_col_index(chunk[3]), s, h);
        let dest4 = mapped_binary_symbol_row(coefficient_col_index(chunk[4]), s, h);
        let dest5 = mapped_binary_symbol_row(coefficient_col_index(chunk[5]), s, h);
        let dest6 = mapped_binary_symbol_row(coefficient_col_index(chunk[6]), s, h);
        let dest7 = mapped_binary_symbol_row(coefficient_col_index(chunk[7]), s, h);
        debug_assert_direct_batch_dests(
            src,
            &[dest0, dest1, dest2, dest3, dest4, dest5, dest6, dest7],
        );
        let dest0_start = dest0 * symbol_size;
        let dest1_start = dest1 * symbol_size;
        let dest2_start = dest2 * symbol_size;
        let dest3_start = dest3 * symbol_size;
        let dest4_start = dest4 * symbol_size;
        let dest5_start = dest5 * symbol_size;
        let dest6_start = dest6 * symbol_size;
        let dest7_start = dest7 * symbol_size;
        assert!(dest0_start + symbol_size <= bytes.len());
        assert!(dest1_start + symbol_size <= bytes.len());
        assert!(dest2_start + symbol_size <= bytes.len());
        assert!(dest3_start + symbol_size <= bytes.len());
        assert!(dest4_start + symbol_size <= bytes.len());
        assert!(dest5_start + symbol_size <= bytes.len());
        assert!(dest6_start + symbol_size <= bytes.len());
        assert!(dest7_start + symbol_size <= bytes.len());
        unsafe {
            add_assign_path.apply_same_len_raw_8(
                [
                    bytes_ptr.add(dest0_start),
                    bytes_ptr.add(dest1_start),
                    bytes_ptr.add(dest2_start),
                    bytes_ptr.add(dest3_start),
                    bytes_ptr.add(dest4_start),
                    bytes_ptr.add(dest5_start),
                    bytes_ptr.add(dest6_start),
                    bytes_ptr.add(dest7_start),
                ],
                src_ptr,
                symbol_size,
            );
        }
    }

    let mut dest_chunks = dest_chunks.remainder().chunks_exact(4);
    for chunk in dest_chunks.by_ref() {
        let dest0 = mapped_binary_symbol_row(coefficient_col_index(chunk[0]), s, h);
        let dest1 = mapped_binary_symbol_row(coefficient_col_index(chunk[1]), s, h);
        let dest2 = mapped_binary_symbol_row(coefficient_col_index(chunk[2]), s, h);
        let dest3 = mapped_binary_symbol_row(coefficient_col_index(chunk[3]), s, h);
        debug_assert_direct_batch_dests(src, &[dest0, dest1, dest2, dest3]);
        let dest0_start = dest0 * symbol_size;
        let dest1_start = dest1 * symbol_size;
        let dest2_start = dest2 * symbol_size;
        let dest3_start = dest3 * symbol_size;
        assert!(dest0_start + symbol_size <= bytes.len());
        assert!(dest1_start + symbol_size <= bytes.len());
        assert!(dest2_start + symbol_size <= bytes.len());
        assert!(dest3_start + symbol_size <= bytes.len());
        unsafe {
            add_assign_path.apply_same_len_raw_4(
                [
                    bytes_ptr.add(dest0_start),
                    bytes_ptr.add(dest1_start),
                    bytes_ptr.add(dest2_start),
                    bytes_ptr.add(dest3_start),
                ],
                src_ptr,
                symbol_size,
            );
        }
    }

    for &dest in dest_chunks.remainder() {
        let dest = mapped_binary_symbol_row(coefficient_col_index(dest), s, h);
        assert_ne!(dest, src);
        let dest_start = dest * symbol_size;
        assert!(dest_start + symbol_size <= bytes.len());
        unsafe {
            let dest_symbol =
                core::slice::from_raw_parts_mut(bytes_ptr.add(dest_start), symbol_size);
            add_assign_path.apply_same_len(dest_symbol, src_symbol);
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
    mut symbols: SymbolSlab,
    source_block_symbols: u32,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    assert_eq!(symbols.len(), matrix.height());
    if matrix.width() >= OVERDETERMINED_NO_HDPC_PREFIX_MIN_WIDTH && matrix.height() > matrix.width()
    {
        if matrix.width() >= OVERDETERMINED_NO_HDPC_PREFIX_OWNED_MIN_WIDTH {
            match try_overdetermined_no_hdpc_prefix_solve_owned(
                &matrix,
                symbols,
                source_block_symbols,
            ) {
                OverdeterminedNoHdpcPrefixSolve::Decoded(decoded) => return (Some(decoded), None),
                OverdeterminedNoHdpcPrefixSolve::Fallback(returned_symbols) => {
                    symbols = returned_symbols;
                }
                OverdeterminedNoHdpcPrefixSolve::Failed => return (None, None),
            }
        } else if let Some(decoded) =
            try_overdetermined_no_hdpc_prefix_solve(&matrix, &symbols, source_block_symbols)
        {
            return (Some(decoded), None);
        }
    }

    let rows = matrix.packed_rows();

    let (decoded, ops) = solve_binary(rows, symbols);
    match decoded {
        Some(decoded) => (
            verify_no_hdpc_solution(
                decoded,
                source_block_symbols,
                matrix.height() > matrix.width(),
            ),
            ops,
        ),
        None => (None, ops),
    }
}

enum OverdeterminedNoHdpcPrefixSolve {
    Decoded(SymbolSlab),
    Fallback(SymbolSlab),
    Failed,
}

fn try_overdetermined_no_hdpc_prefix_solve_owned<M: BinaryMatrix>(
    matrix: &M,
    symbols: SymbolSlab,
    source_block_symbols: u32,
) -> OverdeterminedNoHdpcPrefixSolve {
    let width = matrix.width();
    let prefix_height = (width
        + crate::systematic_constants::num_hdpc_symbols(source_block_symbols) as usize)
        .min(matrix.height());
    if prefix_height == matrix.height() {
        return OverdeterminedNoHdpcPrefixSolve::Fallback(symbols);
    }
    let symbol_size = symbols.symbol_size();
    let mut prefix_bytes = symbols.into_bytes();
    let suffix_bytes = prefix_bytes.split_off(prefix_height * symbol_size);
    let prefix_symbols = SymbolSlab::from_bytes(prefix_bytes, symbol_size);
    let (decoded, suffix_bytes) = match solve_full_rank_binary_prefix_owned(
        matrix,
        prefix_height,
        prefix_symbols,
        suffix_bytes,
        symbol_size,
    ) {
        FullRankBinaryPrefixSolve::Decoded {
            decoded,
            suffix_bytes,
        } => (decoded, suffix_bytes),
        FullRankBinaryPrefixSolve::RankDeficient { restored_symbols } => {
            return OverdeterminedNoHdpcPrefixSolve::Fallback(restored_symbols);
        }
        FullRankBinaryPrefixSolve::Failed => return OverdeterminedNoHdpcPrefixSolve::Failed,
    };
    let Some(decoded) = verify_no_hdpc_solution(decoded, source_block_symbols, true) else {
        return OverdeterminedNoHdpcPrefixSolve::Failed;
    };
    let suffix_symbols = SymbolSlab::from_bytes(suffix_bytes, symbol_size);
    if binary_row_suffixes_satisfied(matrix, &decoded, &suffix_symbols, prefix_height) {
        OverdeterminedNoHdpcPrefixSolve::Decoded(decoded)
    } else {
        OverdeterminedNoHdpcPrefixSolve::Failed
    }
}

enum FullRankBinaryPrefixSolve {
    Decoded {
        decoded: SymbolSlab,
        suffix_bytes: Vec<u8>,
    },
    RankDeficient {
        restored_symbols: SymbolSlab,
    },
    Failed,
}

fn solve_full_rank_binary_prefix_owned<M: BinaryMatrix>(
    matrix: &M,
    prefix_height: usize,
    mut symbols: SymbolSlab,
    suffix_bytes: Vec<u8>,
    symbol_size: usize,
) -> FullRankBinaryPrefixSolve {
    let width = matrix.width();
    let use_weighted_buckets = width >= OVERDETERMINED_NO_HDPC_PREFIX_METADATA_MIN_WIDTH;
    let (mut rows, mut row_weights, first_ones) = if use_weighted_buckets {
        matrix.packed_row_prefix_with_row_weights_and_first_ones(prefix_height)
    } else {
        let rows = matrix.packed_row_prefix(prefix_height);
        let first_ones = (0..rows.height())
            .map(|row| rows.first_one_at_or_after(row, 0))
            .collect();
        (rows, Vec::new(), first_ones)
    };
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut small_weight_buckets = SmallWeightBinaryBuckets::<u16>::new(
        if use_weighted_buckets { width } else { 0 },
        DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX,
    );
    let mut small_row_cache = SmallBinaryRowCache::<DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX>::new(
        if use_weighted_buckets {
            prefix_height
        } else {
            0
        },
    );
    let mut next_in_bucket = vec![NO_BUCKET_ROW; prefix_height];
    for (row, first_one) in first_ones.into_iter().enumerate() {
        if let Some(col) = first_one {
            if use_weighted_buckets {
                push_weighted_binary_row_bucket(
                    &row_weights,
                    &mut bucket_heads,
                    &mut small_weight_buckets,
                    &mut next_in_bucket,
                    col,
                    row,
                );
            } else {
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
            }
        }
    }

    let mut pivot_for_col = vec![usize::MAX; width];
    let mut is_pivot_row = vec![false; prefix_height];
    let mut forward_ranges = Vec::with_capacity(width);
    let mut forward_entries = Vec::with_capacity(width.saturating_mul(4));
    let add_assign_path = AddAssignFastPath::new(symbol_size);
    let batch_forward_symbols = width >= BINARY_FORWARD_SYMBOL_BATCH_MIN_WIDTH;
    let mut forward_symbol_dests = Vec::new();
    for col in 0..width {
        let pivot = if use_weighted_buckets {
            pop_lightest_weighted_binary_row_bucket(
                &row_weights,
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                col,
            )
        } else {
            pop_lightest_binary_row_bucket(&rows, &mut bucket_heads, &mut next_in_bucket, col)
        };
        let Some(pivot) = pivot else {
            restore_binary_forward_symbols(
                &mut symbols,
                &pivot_for_col,
                &forward_ranges,
                &forward_entries,
                add_assign_path,
            );
            let mut bytes = symbols.into_bytes();
            bytes.extend_from_slice(&suffix_bytes);
            return FullRankBinaryPrefixSolve::RankDeficient {
                restored_symbols: SymbolSlab::from_bytes(bytes, symbol_size),
            };
        };
        pivot_for_col[col] = pivot;
        is_pivot_row[pivot] = true;
        let pivot_weight = if use_weighted_buckets {
            row_weights[pivot]
        } else {
            0
        };

        let dest_start = forward_entries.len();
        if batch_forward_symbols {
            forward_symbol_dests.clear();
        }
        while let Some(row) = if use_weighted_buckets {
            pop_weighted_binary_row_bucket(
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                col,
            )
        } else {
            pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col)
        } {
            if use_weighted_buckets {
                let next_col = eliminate_weighted_binary_row::<DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX>(
                    &mut rows,
                    &mut row_weights,
                    &mut small_row_cache,
                    row,
                    pivot,
                    col,
                    pivot_weight,
                );
                if let Some(next_col) = next_col {
                    push_weighted_binary_row_bucket(
                        &row_weights,
                        &mut bucket_heads,
                        &mut small_weight_buckets,
                        &mut next_in_bucket,
                        next_col,
                        row,
                    );
                }
            } else {
                rows.xor_suffix(row, pivot, col);
                if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                    push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
                }
            }
            forward_entries.push(coefficient_col(row));
            if batch_forward_symbols {
                forward_symbol_dests.push(row);
            } else {
                let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(pivot, row);
                add_assign_path.apply(dest_symbol, pivot_symbol);
            }
        }
        if batch_forward_symbols {
            addassign_symbol_row_batch::<true>(
                &mut symbols,
                pivot,
                &forward_symbol_dests,
                add_assign_path,
            );
        }
        forward_ranges.push((dest_start, forward_entries.len()));
    }

    for (row, is_pivot) in is_pivot_row.into_iter().enumerate() {
        if !is_pivot && (!rows.is_zero(row) || !symbol_is_zero(symbols.get(row))) {
            return FullRankBinaryPrefixSolve::Failed;
        }
    }

    let mut decoded = SymbolSlab::with_zeros(width, symbol_size);
    for col in (0..width).rev() {
        let pivot = pivot_for_col[col];
        decoded.get_mut(col).copy_from_slice(symbols.get(pivot));
        if width >= BINARY_SOURCE_BATCH_16_MIN_WIDTH && rows.height() == width {
            addassign_packed_row_sources_to_symbol::<16>(
                &rows,
                pivot,
                col + 1,
                &mut decoded,
                col,
                add_assign_path,
            );
        } else if rows.height() != width
            && width < OVERDETERMINED_NO_HDPC_PREFIX_BACKSUB_BATCH4_MAX_WIDTH
        {
            addassign_packed_row_sources_to_symbol::<4>(
                &rows,
                pivot,
                col + 1,
                &mut decoded,
                col,
                add_assign_path,
            );
        } else {
            addassign_packed_row_sources_to_symbol::<8>(
                &rows,
                pivot,
                col + 1,
                &mut decoded,
                col,
                add_assign_path,
            );
        }
    }

    FullRankBinaryPrefixSolve::Decoded {
        decoded,
        suffix_bytes,
    }
}

fn restore_binary_forward_symbols(
    symbols: &mut SymbolSlab,
    pivot_for_col: &[usize],
    forward_ranges: &[(usize, usize)],
    forward_entries: &[CoefficientColumn],
    add_assign_path: AddAssignFastPath,
) {
    for col in (0..forward_ranges.len()).rev() {
        let pivot = pivot_for_col[col];
        let (start, end) = forward_ranges[col];
        for &row in forward_entries[start..end].iter().rev() {
            let row = coefficient_col_index(row);
            let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(pivot, row);
            add_assign_path.apply(dest_symbol, pivot_symbol);
        }
    }
}

fn try_overdetermined_no_hdpc_prefix_solve<M: BinaryMatrix>(
    matrix: &M,
    symbols: &SymbolSlab,
    source_block_symbols: u32,
) -> Option<SymbolSlab> {
    let width = matrix.width();
    let prefix_height = (width
        + crate::systematic_constants::num_hdpc_symbols(source_block_symbols) as usize)
        .min(matrix.height());
    if prefix_height == matrix.height() {
        return None;
    }

    let symbol_size = symbols.symbol_size();
    let prefix_symbols = SymbolSlab::from_bytes(
        symbols.as_bytes()[..prefix_height * symbol_size].to_vec(),
        symbol_size,
    );
    let (decoded, _) = if width >= OVERDETERMINED_NO_HDPC_PREFIX_METADATA_MIN_WIDTH {
        let (prefix_rows, row_weights, first_ones) =
            matrix.packed_row_prefix_with_row_weights_and_first_ones(prefix_height);
        solve_binary_with_weighted_metadata(prefix_rows, prefix_symbols, row_weights, first_ones)
    } else {
        let prefix_rows = matrix.packed_row_prefix(prefix_height);
        solve_binary(prefix_rows, prefix_symbols)
    };
    let decoded = verify_no_hdpc_solution(decoded?, source_block_symbols, true)?;
    binary_rows_satisfied(matrix, &decoded, symbols, prefix_height).then_some(decoded)
}

fn binary_row_suffixes_satisfied<M: BinaryMatrix>(
    matrix: &M,
    decoded: &SymbolSlab,
    suffix_symbols: &SymbolSlab,
    start_row: usize,
) -> bool {
    let mut check = vec![0u8; decoded.symbol_size()];
    let add_assign_path = AddAssignFastPath::new(decoded.symbol_size());
    for row in start_row..matrix.height() {
        check.fill(0);
        addassign_binary_row_sources_to_slice::<8, M>(
            matrix,
            row,
            &mut check,
            decoded,
            add_assign_path,
        );
        if check.as_slice() != suffix_symbols.get(row - start_row) {
            return false;
        }
    }
    true
}

fn binary_rows_satisfied<M: BinaryMatrix>(
    matrix: &M,
    decoded: &SymbolSlab,
    symbols: &SymbolSlab,
    start_row: usize,
) -> bool {
    let mut check = vec![0u8; decoded.symbol_size()];
    let add_assign_path = AddAssignFastPath::new(decoded.symbol_size());
    for row in start_row..matrix.height() {
        check.fill(0);
        addassign_binary_row_sources_to_slice::<8, M>(
            matrix,
            row,
            &mut check,
            decoded,
            add_assign_path,
        );
        if check.as_slice() != symbols.get(row) {
            return false;
        }
    }
    true
}

fn addassign_binary_row_sources_to_slice<const BATCH: usize, M: BinaryMatrix>(
    matrix: &M,
    row: usize,
    check: &mut [u8],
    decoded: &SymbolSlab,
    add_assign_path: AddAssignFastPath,
) {
    let mut source_batch = [0usize; BATCH];
    let mut source_batch_len = 0usize;
    matrix.visit_row_entries(row, |col| {
        source_batch[source_batch_len] = col;
        source_batch_len += 1;
        if source_batch_len == source_batch.len() {
            addassign_symbol_sources_to_slice(check, decoded, &source_batch, add_assign_path);
            source_batch_len = 0;
        }
    });
    addassign_symbol_sources_to_slice(
        check,
        decoded,
        &source_batch[..source_batch_len],
        add_assign_path,
    );
}

fn verify_no_hdpc_solution(
    decoded: SymbolSlab,
    source_block_symbols: u32,
    cache_verify_row_pairs: bool,
) -> Option<SymbolSlab> {
    if decoded.len() != num_intermediate_symbols(source_block_symbols) as usize {
        return Some(decoded);
    }

    let h = num_hdpc_symbols(source_block_symbols) as usize;
    if is_rfc_hdpc_shape(decoded.len(), h, source_block_symbols)
        && decoded.len() >= COLUMN_MAJOR_HDPC_VERIFY_MIN_WIDTH
    {
        return rfc_hdpc_rows_satisfied_horner(&decoded, h, cache_verify_row_pairs)
            .then_some(decoded);
    }

    let hdpc_rows = generate_hdpc_rows(source_block_symbols);
    hdpc_rows_satisfied(&decoded, &hdpc_rows).then_some(decoded)
}

fn is_rfc_hdpc_shape(width: usize, h: usize, source_block_symbols: u32) -> bool {
    width == num_intermediate_symbols(source_block_symbols) as usize
        && h == num_hdpc_symbols(source_block_symbols) as usize
        && h > 1
}

#[inline]
fn rfc_hdpc_verify_rows_for_col(col: usize, h: usize) -> (usize, usize) {
    let random = RfcRand::new((col + 1) as u32);
    let row_a = random.get(6, h as u32) as usize;
    let row_b = (row_a + random.get(7, h as u32 - 1) as usize + 1) % h;
    (row_a, row_b)
}

#[cfg(feature = "std")]
fn cached_rfc_hdpc_verify_row_pairs_if_present(
    gamma_width: usize,
    h: usize,
) -> Option<HdpcVerifyRowPairs> {
    let key = (gamma_width, h);
    let cache = hdpc_verify_row_pairs_cache();
    let guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.pairs.get(&key).map(Arc::clone)
}

#[cfg(feature = "std")]
fn insert_rfc_hdpc_verify_row_pairs(
    gamma_width: usize,
    h: usize,
    generated: Vec<HdpcVerifyRowPair>,
) {
    let key = (gamma_width, h);
    let generated = Arc::from(generated.into_boxed_slice());
    let cache = hdpc_verify_row_pairs_cache();
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.pairs.contains_key(&key) {
        return;
    }
    if guard.pairs.len() >= HDPC_VERIFY_ROW_PAIRS_CACHE_CAPACITY
        && let Some(evicted) = guard.insertion_order.pop_front()
    {
        guard.pairs.remove(&evicted);
    }
    guard.insertion_order.push_back(key);
    guard.pairs.insert(key, generated);
}

fn rfc_hdpc_rows_satisfied_horner(
    decoded: &SymbolSlab,
    h: usize,
    cache_verify_row_pairs: bool,
) -> bool {
    #[cfg(not(feature = "std"))]
    let _ = cache_verify_row_pairs;

    let width = decoded.len();
    let Some(gamma_width) = width.checked_sub(h) else {
        return false;
    };
    if h <= 1 || gamma_width == 0 {
        return false;
    }

    let symbol_size = decoded.symbol_size();
    let add_assign_path = AddAssignFastPath::new(symbol_size);
    let fused_mul_path = FusedAddAssignMulScalarFastPath::new(symbol_size);
    let mut prefix = vec![0u8; symbol_size];
    let mut checks = SymbolSlab::with_zeros(h, symbol_size);
    #[cfg(feature = "std")]
    let cache_verify_row_pairs = cache_verify_row_pairs
        && (HDPC_VERIFY_ROW_PAIRS_CACHE_MIN_GAMMA_WIDTH
            ..=HDPC_VERIFY_ROW_PAIRS_CACHE_MAX_GAMMA_WIDTH)
            .contains(&gamma_width);
    #[cfg(feature = "std")]
    let cached_verify_row_pairs = cache_verify_row_pairs
        .then(|| cached_rfc_hdpc_verify_row_pairs_if_present(gamma_width, h))
        .flatten();
    #[cfg(feature = "std")]
    let mut generated_verify_row_pairs =
        if cache_verify_row_pairs && cached_verify_row_pairs.is_none() {
            Vec::with_capacity(gamma_width.saturating_sub(1))
        } else {
            Vec::new()
        };

    for col in 0..gamma_width {
        fused_mulassign_alpha_add_assign(&mut prefix, decoded.get(col));

        if col + 1 == gamma_width {
            for row in 0..h {
                let factor = Octet::alpha_pow(row);
                fused_mul_path.apply_nonzero(checks.get_mut(row), &prefix, &factor);
            }
        } else {
            #[cfg(feature = "std")]
            let (row_a, row_b) = if let Some(verify_row_pairs) = cached_verify_row_pairs.as_ref() {
                let (row_a, row_b) = verify_row_pairs[col];
                (coefficient_col_index(row_a), coefficient_col_index(row_b))
            } else {
                let (row_a, row_b) = rfc_hdpc_verify_rows_for_col(col, h);
                if cache_verify_row_pairs {
                    debug_assert!(CoefficientColumn::try_from(row_a).is_ok());
                    debug_assert!(CoefficientColumn::try_from(row_b).is_ok());
                    generated_verify_row_pairs
                        .push((coefficient_col(row_a), coefficient_col(row_b)));
                }
                (row_a, row_b)
            };
            #[cfg(not(feature = "std"))]
            let (row_a, row_b) = rfc_hdpc_verify_rows_for_col(col, h);
            addassign_hdpc_check_pair(&mut checks, row_a, row_b, &prefix, add_assign_path);
        }
    }

    #[cfg(feature = "std")]
    if cache_verify_row_pairs && cached_verify_row_pairs.is_none() {
        insert_rfc_hdpc_verify_row_pairs(gamma_width, h, generated_verify_row_pairs);
    }

    for row in 0..h {
        add_assign_path.apply(checks.get_mut(row), decoded.get(gamma_width + row));
    }

    checks
        .as_bytes()
        .chunks_exact(symbol_size)
        .all(bytes_are_zero)
}

fn addassign_hdpc_check_pair(
    checks: &mut SymbolSlab,
    row_a: usize,
    row_b: usize,
    prefix: &[u8],
    add_assign_path: AddAssignFastPath,
) {
    debug_assert_ne!(row_a, row_b);
    let symbol_size = checks.symbol_size();
    assert_eq!(prefix.len(), symbol_size);
    let bytes = checks.as_mut_bytes();
    let row_a_start = row_a * symbol_size;
    let row_b_start = row_b * symbol_size;
    assert!(row_a_start + symbol_size <= bytes.len());
    assert!(row_b_start + symbol_size <= bytes.len());
    let bytes_ptr = bytes.as_mut_ptr();
    unsafe {
        add_assign_path.apply_same_len_raw_2(
            [bytes_ptr.add(row_a_start), bytes_ptr.add(row_b_start)],
            prefix.as_ptr(),
            symbol_size,
        );
    }
}

fn hdpc_rows_satisfied(decoded: &SymbolSlab, hdpc_rows: &DenseOctetMatrix) -> bool {
    if hdpc_rows.width() >= COLUMN_MAJOR_HDPC_VERIFY_MIN_WIDTH && hdpc_rows.height() > 1 {
        return hdpc_rows_satisfied_column_major(decoded, hdpc_rows);
    }

    hdpc_rows_satisfied_row_major(decoded, hdpc_rows)
}

fn hdpc_rows_satisfied_row_major(decoded: &SymbolSlab, hdpc_rows: &DenseOctetMatrix) -> bool {
    let mut check = vec![0u8; decoded.symbol_size()];
    let fused_mul_path = FusedAddAssignMulScalarFastPath::new(decoded.symbol_size());
    for row in 0..hdpc_rows.height() {
        check.fill(0);
        for (col, &coefficient) in hdpc_rows.row(row).iter().enumerate() {
            if !coefficient.is_zero() {
                fused_mul_path.apply_nonzero(&mut check, decoded.get(col), &coefficient);
            }
        }
        if !symbol_is_zero(&check) {
            return false;
        }
    }
    true
}

fn hdpc_rows_satisfied_column_major(decoded: &SymbolSlab, hdpc_rows: &DenseOctetMatrix) -> bool {
    let symbol_size = decoded.symbol_size();
    let h = hdpc_rows.height();
    let coefficients = dense_hdpc_coefficient_values_column_major(hdpc_rows);
    let mut checks = vec![0u8; h * symbol_size];
    let fused_mul_path = FusedAddAssignMulScalarFastPath::new(symbol_size);

    for col in 0..hdpc_rows.width() {
        let coefficients = &coefficients[col * h..(col + 1) * h];
        unsafe {
            fused_mul_path.apply_column_coefficients(
                checks.as_mut_ptr(),
                symbol_size,
                decoded.get(col).as_ptr(),
                coefficients,
                symbol_size,
            );
        }
    }

    checks.chunks_exact(symbol_size).all(bytes_are_zero)
}

// Repair systems with at least L rows can often be reduced mostly over GF(2).
// Any remaining free columns form a small GF(256) system for the HDPC rows.
#[cfg(feature = "std")]
fn prepare_cached_hybrid_systematic_plan<M: BinaryMatrix>(
    source_block_symbols: u32,
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
) -> Option<CachedHybridSystematicPlan> {
    prepare_cached_hybrid_systematic_plan_with_small_weight_max::<
        PLAN_SMALL_WEIGHT_BINARY_BUCKET_MAX,
        M,
    >(source_block_symbols, matrix, hdpc_rows)
}

#[cfg(feature = "std")]
fn prepare_cached_hybrid_systematic_plan_for_decode<M: BinaryMatrix>(
    source_block_symbols: u32,
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
) -> Option<CachedHybridSystematicPlan> {
    prepare_cached_hybrid_systematic_plan_with_small_weight_max::<
        DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX,
        M,
    >(source_block_symbols, matrix, hdpc_rows)
}

#[cfg(feature = "std")]
fn prepare_cached_hybrid_systematic_plan_with_small_weight_max<
    const SMALL_WEIGHT_MAX: usize,
    M: BinaryMatrix,
>(
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

    let use_weighted_buckets = width >= LARGE_BINARY_WEIGHTED_BUCKET_MIN_WIDTH;
    let (mut rows, mut row_weights, first_ones) = if use_weighted_buckets {
        matrix.packed_rows_with_row_weights_and_first_ones()
    } else {
        let (rows, first_ones) = matrix.packed_rows_with_first_ones();
        (rows, Vec::new(), first_ones)
    };
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut small_weight_buckets = SmallWeightBinaryBuckets::<u32>::new(
        if use_weighted_buckets { width } else { 0 },
        SMALL_WEIGHT_MAX,
    );
    let mut small_row_cache =
        SmallBinaryRowCache::<SMALL_WEIGHT_MAX>::new(if use_weighted_buckets {
            binary_height
        } else {
            0
        });
    let mut next_in_bucket = vec![NO_BUCKET_ROW; binary_height];
    for (row, first_one) in first_ones.into_iter().enumerate() {
        if let Some(col) = first_one {
            if use_weighted_buckets {
                push_weighted_binary_row_bucket(
                    &row_weights,
                    &mut bucket_heads,
                    &mut small_weight_buckets,
                    &mut next_in_bucket,
                    col,
                    row,
                );
            } else {
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
            }
        }
    }

    let mut pivot_for_col = vec![None; width];
    let mut is_pivot_row = vec![false; binary_height];
    let mut pivots = Vec::with_capacity(width);
    let mut binary_forward_ranges = Vec::with_capacity(width);
    let mut binary_forward_entries = Vec::new();
    for col in 0..width {
        let pivot = if use_weighted_buckets {
            pop_lightest_weighted_binary_row_bucket(
                &row_weights,
                &mut bucket_heads,
                &mut small_weight_buckets,
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
        let pivot_weight = if use_weighted_buckets {
            row_weights[pivot]
        } else {
            0
        };

        let dest_start = binary_forward_entries.len();
        while let Some(row) = if use_weighted_buckets {
            pop_weighted_binary_row_bucket(
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                col,
            )
        } else {
            pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col)
        } {
            if use_weighted_buckets {
                let next_col = eliminate_weighted_binary_row::<SMALL_WEIGHT_MAX>(
                    &mut rows,
                    &mut row_weights,
                    &mut small_row_cache,
                    row,
                    pivot,
                    col,
                    pivot_weight,
                );
                if let Some(next_col) = next_col {
                    push_weighted_binary_row_bucket(
                        &row_weights,
                        &mut bucket_heads,
                        &mut small_weight_buckets,
                        &mut next_in_bucket,
                        next_col,
                        row,
                    );
                }
            } else {
                rows.xor_suffix(row, pivot, col);
                if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                    push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
                }
            }
            binary_forward_entries.push(coefficient_col(row));
        }
        binary_forward_ranges.push((dest_start, binary_forward_entries.len()));
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
    let mut hdpc_projection_rows = Vec::with_capacity(h);
    for &(col, pivot) in &pivots {
        hdpc_projection_rows.clear();
        for row in 0..h {
            let row_start = row * width;
            let factor = hdpc_coefficients[row_start + col];
            if factor.is_zero() {
                continue;
            }
            hdpc_symbol_steps.push(HybridHdpcSymbolStep { row, pivot, factor });
            hdpc_projection_rows.push((row_start, factor));
        }
        if hdpc_projection_rows.is_empty() {
            continue;
        }
        rows.visit_ones_at_or_after(pivot, col, |entry_col| {
            for &(row_start, factor) in &hdpc_projection_rows {
                hdpc_coefficients[row_start + entry_col] += factor;
            }
        });
    }

    let free_rows = hybrid_hdpc_free_rows(&hdpc_coefficients, &free_cols, width)?;
    let free_solve = (width <= CACHED_HDPC_FREE_SOLVE_MAX_WIDTH)
        .then(|| prepare_cached_hdpc_free_solve_from_rows(&free_rows, free_cols.len(), 1))
        .flatten();
    let back_substitution = prepare_binary_flat_back_substitution_batches(&rows, &pivots, width);
    let output_symbol_cycles = use_in_place_hybrid_replay(width)
        .then(|| hybrid_output_symbol_cycles(&pivots, &free_cols, s, h, width))
        .flatten();

    Some(CachedHybridSystematicPlan {
        binary_forward_dests: CachedBinarySlices {
            ranges: binary_forward_ranges,
            entries: binary_forward_entries,
        },
        hdpc_symbol_steps,
        free_cols: free_cols.into_boxed_slice(),
        free_rows,
        free_solve,
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
) -> CachedBinarySlices {
    let mut counts = vec![0usize; width];
    for &(col, pivot) in pivots {
        rows.visit_ones_at_or_after(pivot, col + 1, |dependent_col| {
            counts[dependent_col] += 1;
        });
    }

    let (ranges, mut offsets, mut entries, entries_len) = binary_slices_from_counts(counts);
    for &(col, pivot) in pivots {
        rows.visit_ones_at_or_after(pivot, col + 1, |dependent_col| {
            let offset = offsets[dependent_col];
            debug_assert!(offset < entries_len);
            // The first pass counted this slot, and each dependent column advances
            // monotonically inside its assigned range.
            unsafe {
                entries.as_mut_ptr().add(offset).write(coefficient_col(col));
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

    CachedBinarySlices { ranges, entries }
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
    let add_assign_path = AddAssignFastPath::new(symbol_size);
    let fused_mul_path = FusedAddAssignMulScalarFastPath::new(symbol_size);
    let overdetermined = binary_height + h > width;

    let mut binary_symbols = SymbolSlab::with_zeros(binary_height, symbol_size);
    let symbol_bytes = symbols.as_bytes();
    let low_binary_bytes = s * symbol_size;
    binary_symbols.copy_block_from(0, &symbol_bytes[..low_binary_bytes]);
    let high_binary_start = (s + h) * symbol_size;
    let high_binary_end = high_binary_start + (binary_height - s) * symbol_size;
    binary_symbols.copy_block_from(s, &symbol_bytes[high_binary_start..high_binary_end]);

    let use_weighted_buckets = width >= LARGE_BINARY_WEIGHTED_BUCKET_MIN_WIDTH;
    let (mut rows, mut row_weights, first_ones) = if use_weighted_buckets {
        matrix.packed_rows_with_row_weights_and_first_ones()
    } else {
        let (rows, first_ones) = matrix.packed_rows_with_first_ones();
        (rows, Vec::new(), first_ones)
    };
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut small_weight_buckets = SmallWeightBinaryBuckets::<u16>::new(
        if use_weighted_buckets { width } else { 0 },
        DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX,
    );
    let mut small_row_cache = SmallBinaryRowCache::<DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX>::new(
        if use_weighted_buckets {
            binary_height
        } else {
            0
        },
    );
    let mut next_in_bucket = vec![NO_BUCKET_ROW; binary_height];
    for (row, first_one) in first_ones.into_iter().enumerate() {
        if let Some(col) = first_one {
            if use_weighted_buckets {
                push_weighted_binary_row_bucket(
                    &row_weights,
                    &mut bucket_heads,
                    &mut small_weight_buckets,
                    &mut next_in_bucket,
                    col,
                    row,
                );
            } else {
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
            }
        }
    }

    let mut pivot_for_col = vec![None; width];
    let mut is_pivot_row = vec![false; binary_height];
    for col in 0..width {
        let pivot = if use_weighted_buckets {
            pop_lightest_weighted_binary_row_bucket(
                &row_weights,
                &mut bucket_heads,
                &mut small_weight_buckets,
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
        let pivot_weight = if use_weighted_buckets {
            row_weights[pivot]
        } else {
            0
        };

        while let Some(row) = if use_weighted_buckets {
            pop_weighted_binary_row_bucket(
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                col,
            )
        } else {
            pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col)
        } {
            if use_weighted_buckets {
                let next_col = eliminate_weighted_binary_row::<DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX>(
                    &mut rows,
                    &mut row_weights,
                    &mut small_row_cache,
                    row,
                    pivot,
                    col,
                    pivot_weight,
                );
                if let Some(next_col) = next_col {
                    push_weighted_binary_row_bucket(
                        &row_weights,
                        &mut bucket_heads,
                        &mut small_weight_buckets,
                        &mut next_in_bucket,
                        next_col,
                        row,
                    );
                }
            } else {
                rows.xor_suffix(row, pivot, col);
                if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                    push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
                }
            }
            let (pivot_symbol, dest_symbol) = binary_symbols.get_disjoint_mut(pivot, row);
            add_assign_path.apply(dest_symbol, pivot_symbol);
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
    let mut hdpc_projection_rows = Vec::with_capacity(h);

    for (col, pivot) in pivot_for_col.iter().copied().enumerate() {
        let Some(pivot) = pivot else {
            continue;
        };

        hdpc_projection_rows.clear();
        for row in 0..h {
            let row_start = row * width;
            let factor = hdpc_coefficients[row_start + col];
            if factor.is_zero() {
                continue;
            }
            hdpc_projection_rows.push((row_start, factor));
            fused_mul_path.apply_nonzero(
                hdpc_symbols.get_mut(row),
                binary_symbols.get(pivot),
                &factor,
            );
        }
        if hdpc_projection_rows.is_empty() {
            continue;
        }
        rows.visit_ones_at_or_after(pivot, col, |entry_col| {
            for &(row_start, factor) in &hdpc_projection_rows {
                hdpc_coefficients[row_start + entry_col] += factor;
            }
        });
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
                add_assign_path.apply(dest_symbol, dependent_symbol);
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
    let add_assign_path = AddAssignFastPath::new(symbol_size);
    let mut decoded = SymbolSlab::with_zeros(width, symbol_size);
    for col in (0..width).rev() {
        let pivot = pivot_for_col[col].expect("full-rank binary solve has every pivot");
        decoded
            .get_mut(col)
            .copy_from_slice(binary_symbols.get(pivot));
        rows.visit_ones_at_or_after(pivot, col + 1, |dependent_col| {
            let (dependent_symbol, dest_symbol) = decoded.get_disjoint_mut(dependent_col, col);
            add_assign_path.apply(dest_symbol, dependent_symbol);
        });
    }
    decoded
}

fn eliminate_weighted_binary_row<const SMALL_WEIGHT_MAX: usize>(
    rows: &mut PackedBinaryRows,
    row_weights: &mut [u32],
    small_row_cache: &mut SmallBinaryRowCache<SMALL_WEIGHT_MAX>,
    row: usize,
    pivot: usize,
    col: usize,
    pivot_weight: u32,
) -> Option<usize> {
    if (pivot_weight as usize) <= SMALL_WEIGHT_MAX {
        debug_assert_ne!(row_weights[row], 0);
        let pivot_entries = small_row_cache.entries(rows, pivot, col);
        let (weight, next_col) =
            rows.xor_u16_columns_update_weight_and_first_one(row, pivot_entries, row_weights[row]);
        row_weights[row] = weight;
        small_row_cache.invalidate(row);
        return next_col;
    }

    let (weight, next_col) = rows.xor_suffix_count_ones_and_first_one(row, pivot, col);
    row_weights[row] = weight;
    small_row_cache.invalidate(row);
    next_col
}

fn dense_hdpc_coefficients(matrix: &DenseOctetMatrix) -> Vec<Octet> {
    matrix.as_slice().to_vec()
}

fn dense_hdpc_coefficient_values_column_major(matrix: &DenseOctetMatrix) -> Vec<u8> {
    let mut coefficients = vec![0u8; matrix.width() * matrix.height()];
    for row in 0..matrix.height() {
        for (col, coefficient) in matrix.row(row).iter().enumerate() {
            coefficients[col * matrix.height() + row] = coefficient.value();
        }
    }
    coefficients
}

#[cfg(feature = "std")]
fn direct_hdpc_coefficient_stride(h: usize) -> usize {
    debug_assert!(h <= 16);
    16
}

#[cfg(feature = "std")]
fn dense_hdpc_coefficient_values_column_major_padded(
    matrix: &DenseOctetMatrix,
    stride: usize,
) -> Vec<u8> {
    debug_assert!(matrix.height() <= stride);
    let mut coefficients = vec![0u8; matrix.width() * stride];
    for row in 0..matrix.height() {
        for (col, coefficient) in matrix.row(row).iter().enumerate() {
            coefficients[col * stride + row] = coefficient.value();
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
            free_row.push((coefficient_col(free_index), value));
        }
        free_rows.push(free_row);
    }

    solve_without_recording(free_rows, free_cols.len(), hdpc_symbols).0
}

#[cfg(feature = "std")]
fn try_square_hybrid_binary_hdpc_solve_owned<M: BinaryMatrix>(
    source_block_symbols: u32,
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    mut symbols: SymbolSlab,
) -> SquareHybridDecodeResult {
    let width = matrix.width();
    if matrix.height() + hdpc_rows.height() != width || symbols.len() != width {
        return SquareHybridDecodeResult::Fallback(symbols);
    }

    if use_one_shot_square_hybrid_decode(width) {
        match try_square_hybrid_binary_hdpc_solve_one_shot(
            source_block_symbols,
            matrix,
            hdpc_rows,
            symbols,
        ) {
            SquareHybridDecodeResult::Fallback(returned_symbols) => {
                symbols = returned_symbols;
            }
            result => return result,
        }
    }

    if use_direct_square_hybrid_decode(width) {
        match try_direct_square_hybrid_decode(source_block_symbols, matrix, hdpc_rows, symbols) {
            SquareHybridDecodeResult::Fallback(returned_symbols) => {
                symbols = returned_symbols;
            }
            result => return result,
        }
    }

    let Some(plan) =
        prepare_cached_hybrid_systematic_plan_for_decode(source_block_symbols, matrix, hdpc_rows)
    else {
        return SquareHybridDecodeResult::Fallback(symbols);
    };

    if try_apply_cached_hybrid_systematic_plan_in_place(&plan, &mut symbols) {
        SquareHybridDecodeResult::Decoded(symbols)
    } else {
        SquareHybridDecodeResult::Failed
    }
}

#[cfg(feature = "std")]
fn try_direct_square_hybrid_decode<M: BinaryMatrix>(
    source_block_symbols: u32,
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    mut symbols: SymbolSlab,
) -> SquareHybridDecodeResult {
    let Some(plan) =
        prepare_direct_systematic_plan_for_decode(matrix, hdpc_rows, source_block_symbols)
    else {
        return SquareHybridDecodeResult::Fallback(symbols);
    };

    if try_apply_prepared_direct_systematic_plan(&plan, &mut symbols) {
        SquareHybridDecodeResult::Decoded(symbols)
    } else {
        SquareHybridDecodeResult::Failed
    }
}

#[cfg(all(feature = "std", not(test)))]
fn use_one_shot_square_hybrid_decode(_width: usize) -> bool {
    false
}

#[cfg(all(feature = "std", test))]
fn use_one_shot_square_hybrid_decode(width: usize) -> bool {
    width >= IN_PLACE_HYBRID_REPLAY_MIN_WIDTH
}

#[cfg(all(feature = "std", not(test)))]
fn use_direct_square_hybrid_decode(width: usize) -> bool {
    width >= SQUARE_HYBRID_DECODE_MIN_WIDTH
}

#[cfg(all(feature = "std", test))]
fn use_direct_square_hybrid_decode(_width: usize) -> bool {
    false
}

#[cfg(feature = "std")]
fn try_square_hybrid_binary_hdpc_solve_one_shot<M: BinaryMatrix>(
    source_block_symbols: u32,
    matrix: &M,
    hdpc_rows: &DenseOctetMatrix,
    mut symbols: SymbolSlab,
) -> SquareHybridDecodeResult {
    let s = num_ldpc_symbols(source_block_symbols) as usize;
    let h = hdpc_rows.height();
    let width = matrix.width();
    let binary_height = matrix.height();
    if binary_height + h != width
        || hdpc_rows.width() != width
        || binary_height < s
        || width >= NO_COEFFICIENT_COLUMN as usize
    {
        return SquareHybridDecodeResult::Fallback(symbols);
    }

    let symbol_size = symbols.symbol_size();
    let add_assign_path = AddAssignFastPath::new(symbol_size);
    let fused_mul_path = FusedAddAssignMulScalarFastPath::new(symbol_size);
    let use_weighted_buckets = width >= LARGE_BINARY_WEIGHTED_BUCKET_MIN_WIDTH;
    let (mut rows, mut row_weights, first_ones) = if use_weighted_buckets {
        matrix.packed_rows_with_row_weights_and_first_ones()
    } else {
        let (rows, first_ones) = matrix.packed_rows_with_first_ones();
        (rows, Vec::new(), first_ones)
    };
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut small_weight_buckets = SmallWeightBinaryBuckets::<u16>::new(
        if use_weighted_buckets { width } else { 0 },
        DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX,
    );
    let mut small_row_cache = SmallBinaryRowCache::<DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX>::new(
        if use_weighted_buckets {
            binary_height
        } else {
            0
        },
    );
    let mut next_in_bucket = vec![NO_BUCKET_ROW; binary_height];
    for (row, first_one) in first_ones.into_iter().enumerate() {
        if let Some(col) = first_one {
            if use_weighted_buckets {
                push_weighted_binary_row_bucket(
                    &row_weights,
                    &mut bucket_heads,
                    &mut small_weight_buckets,
                    &mut next_in_bucket,
                    col,
                    row,
                );
            } else {
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
            }
        }
    }

    let mut pivot_for_col = vec![None; width];
    let mut is_pivot_row = vec![false; binary_height];
    let mut pivots = Vec::with_capacity(binary_height);
    for col in 0..width {
        let pivot = if use_weighted_buckets {
            pop_lightest_weighted_binary_row_bucket(
                &row_weights,
                &mut bucket_heads,
                &mut small_weight_buckets,
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
        let pivot_weight = if use_weighted_buckets {
            row_weights[pivot]
        } else {
            0
        };

        let pivot_symbol = mapped_binary_symbol_row(pivot, s, h);
        while let Some(row) = if use_weighted_buckets {
            pop_weighted_binary_row_bucket(
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                col,
            )
        } else {
            pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col)
        } {
            if use_weighted_buckets {
                let next_col = eliminate_weighted_binary_row::<DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX>(
                    &mut rows,
                    &mut row_weights,
                    &mut small_row_cache,
                    row,
                    pivot,
                    col,
                    pivot_weight,
                );
                if let Some(next_col) = next_col {
                    push_weighted_binary_row_bucket(
                        &row_weights,
                        &mut bucket_heads,
                        &mut small_weight_buckets,
                        &mut next_in_bucket,
                        next_col,
                        row,
                    );
                }
            } else {
                rows.xor_suffix(row, pivot, col);
                if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                    push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
                }
            }
            let dest_symbol = mapped_binary_symbol_row(row, s, h);
            let (src, dest) = symbols.get_disjoint_mut(pivot_symbol, dest_symbol);
            add_assign_path.apply(dest, src);
        }
    }

    for (row, is_pivot) in is_pivot_row.into_iter().enumerate() {
        if !is_pivot
            && (!rows.is_zero(row)
                || !symbol_is_zero(symbols.get(mapped_binary_symbol_row(row, s, h))))
        {
            return SquareHybridDecodeResult::Failed;
        }
    }

    let free_cols = pivot_for_col
        .iter()
        .enumerate()
        .filter_map(|(col, pivot)| pivot.is_none().then_some(col))
        .collect::<Vec<_>>();
    if free_cols.len() > h {
        return SquareHybridDecodeResult::Failed;
    }

    let mut hdpc_coefficients = dense_hdpc_coefficients(hdpc_rows);
    let mut hdpc_projection_rows = Vec::with_capacity(h);
    let mut pivot_entries = Vec::new();
    let mut back_substitution_counts = vec![0usize; width];
    let mut back_substitution_entries = Vec::new();
    for &(col, pivot) in &pivots {
        pivot_entries.clear();
        rows.visit_ones_at_or_after(pivot, col + 1, |entry_col| {
            pivot_entries.push(entry_col);
        });
        for &dependent_col in &pivot_entries {
            back_substitution_counts[dependent_col] += 1;
            back_substitution_entries.push((coefficient_col(dependent_col), coefficient_col(col)));
        }

        hdpc_projection_rows.clear();
        let pivot_symbol = mapped_binary_symbol_row(pivot, s, h);
        for row in 0..h {
            let row_start = row * width;
            let factor = hdpc_coefficients[row_start + col];
            if factor.is_zero() {
                continue;
            }
            hdpc_coefficients[row_start + col] = Octet::zero();
            hdpc_projection_rows.push((row_start, factor));
            let hdpc_symbol = s + row;
            let (src, dest) = symbols.get_disjoint_mut(pivot_symbol, hdpc_symbol);
            fused_mul_path.apply_nonzero(dest, src, &factor);
        }
        if hdpc_projection_rows.is_empty() {
            continue;
        }
        for &entry_col in &pivot_entries {
            for &(row_start, factor) in &hdpc_projection_rows {
                hdpc_coefficients[row_start + entry_col] += factor;
            }
        }
    }

    let Some(free_rows) = hybrid_hdpc_free_rows(&hdpc_coefficients, &free_cols, width) else {
        return SquareHybridDecodeResult::Failed;
    };
    let mut hdpc_symbols = SymbolSlab::with_zeros(h, symbol_size);
    for row in 0..h {
        hdpc_symbols
            .get_mut(row)
            .copy_from_slice(symbols.get(s + row));
    }
    let Some(free_values) = solve_without_recording(free_rows, free_cols.len(), hdpc_symbols).0
    else {
        return SquareHybridDecodeResult::Failed;
    };

    let back_substitution = prepare_direct_back_substitution_batches(
        back_substitution_counts,
        back_substitution_entries,
    );
    if let Some(output_symbol_cycles) =
        hybrid_output_symbol_cycles(&pivots, &free_cols, s, h, width)
    {
        for (free_index, _) in free_cols.iter().enumerate() {
            symbols
                .get_mut(s + free_index)
                .copy_from_slice(free_values.get(free_index));
        }
        move_pivot_symbols_to_columns(&mut symbols, &output_symbol_cycles);
        for src in (0..width).rev() {
            addassign_direct_symbol_batch_no_zero_check(
                &mut symbols,
                src,
                back_substitution.slice(src),
                add_assign_path,
            );
        }
        return SquareHybridDecodeResult::Decoded(symbols);
    }

    let mut decoded = SymbolSlab::with_zeros(width, symbol_size);
    for (free_index, &col) in free_cols.iter().enumerate() {
        decoded
            .get_mut(col)
            .copy_from_slice(free_values.get(free_index));
    }
    for &(col, pivot) in &pivots {
        let pivot = mapped_binary_symbol_row(pivot, s, h);
        decoded.get_mut(col).copy_from_slice(symbols.get(pivot));
    }
    for src in (0..width).rev() {
        addassign_direct_symbol_batch_no_zero_check(
            &mut decoded,
            src,
            back_substitution.slice(src),
            add_assign_path,
        );
    }

    SquareHybridDecodeResult::Decoded(decoded)
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
    let repair_entries = if let Some(repair_isi) = rows.repair_isi {
        systematic_constraint_row_entries(source_block_symbols, repair_isi)
    } else {
        matrix.row_entries(rows.repair_matrix_row)
    };
    if repair_entries.is_empty() {
        return None;
    }

    if width >= DIRECT_SINGLE_REPAIR_SYSTEMATIC_MIN_WIDTH
        && let Some(decoded) = try_direct_single_repair_systematic_decode(
            source_block_symbols,
            &rows,
            &repair_entries,
            symbols,
            s,
            h,
        )
    {
        return Some(decoded);
    }

    let plan = cached_systematic_plan(k_prime as u32);
    let repair_coefficients = if let Some(repair_isi) = rows.repair_isi {
        cached_repair_source_coefficients(source_block_symbols, repair_isi, &plan, s, h)
    } else {
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
fn try_direct_single_repair_systematic_decode(
    source_block_symbols: u32,
    rows: &SingleRepairSystematicRows,
    repair_entries: &[usize],
    symbols: &SymbolSlab,
    s: usize,
    h: usize,
) -> Option<SymbolSlab> {
    let width = symbols.len();
    let symbol_size = symbols.symbol_size();
    let missing_coefficient = if let Some(repair_isi) = rows.repair_isi {
        cached_direct_single_repair_coefficient(
            source_block_symbols,
            rows.missing_isi,
            repair_isi,
            repair_entries,
        )
    } else {
        generate_direct_single_repair_coefficient(
            source_block_symbols,
            rows.missing_isi,
            repair_entries,
        )
    };
    if missing_coefficient.is_zero() {
        return None;
    }

    let mut known_symbols = SymbolSlab::with_zeros(width, symbol_size);
    match &rows.systematic_rows {
        SingleRepairSystematicRowLayout::Contiguous => {
            for isi in 0..source_block_symbols as usize {
                if isi == rows.missing_isi {
                    continue;
                }
                let matrix_row = s + if isi < rows.missing_isi { isi } else { isi - 1 };
                known_symbols
                    .get_mut(s + h + isi)
                    .copy_from_slice(symbols.get(matrix_row + h));
            }
        }
        SingleRepairSystematicRowLayout::Explicit(systematic_rows) => {
            for &(isi, matrix_row) in systematic_rows {
                if isi >= source_block_symbols as usize || isi == rows.missing_isi {
                    continue;
                }
                known_symbols
                    .get_mut(s + h + isi)
                    .copy_from_slice(symbols.get(matrix_row + h));
            }
        }
    }

    let plan = cached_direct_systematic_plan(source_block_symbols);
    apply_prepared_direct_systematic_plan(&plan, &mut known_symbols);

    let mut known_repair_symbol = vec![0u8; symbol_size];
    for &entry in repair_entries {
        add_assign(&mut known_repair_symbol, known_symbols.get(entry));
    }

    let mut missing_symbol = symbols.get(rows.repair_matrix_row + h).to_vec();
    add_assign(&mut missing_symbol, &known_repair_symbol);
    mulassign_scalar(&mut missing_symbol, &missing_coefficient.inverse());

    let missing_entries =
        systematic_constraint_row_entries(source_block_symbols, rows.missing_isi as u32);
    let &missing_entry = missing_entries.first()?;
    let mut decoded = SymbolSlab::with_zeros(width, symbol_size);
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
    rows: PackedBinaryRows,
    symbols: SymbolSlab,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let width = rows.width();
    let height = rows.height();
    let use_weighted_buckets = width >= LARGE_BINARY_WEIGHTED_BUCKET_MIN_WIDTH;
    let row_weights = if use_weighted_buckets {
        (0..height)
            .map(|row| rows.weight_at_or_after(row, 0))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let first_ones = (0..height)
        .map(|row| rows.first_one_at_or_after(row, 0))
        .collect::<Vec<_>>();
    solve_binary_with_initial_metadata(rows, symbols, row_weights, first_ones, use_weighted_buckets)
}

fn solve_binary_with_weighted_metadata(
    rows: PackedBinaryRows,
    symbols: SymbolSlab,
    row_weights: Vec<u32>,
    first_ones: Vec<Option<usize>>,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    debug_assert!(rows.width() >= LARGE_BINARY_WEIGHTED_BUCKET_MIN_WIDTH);
    solve_binary_with_initial_metadata(rows, symbols, row_weights, first_ones, true)
}

fn solve_binary_with_initial_metadata(
    mut rows: PackedBinaryRows,
    mut symbols: SymbolSlab,
    mut row_weights: Vec<u32>,
    first_ones: Vec<Option<usize>>,
    use_weighted_buckets: bool,
) -> (Option<SymbolSlab>, Option<Vec<SymbolOps>>) {
    let width = rows.width();
    let height = rows.height();
    assert_eq!(height, symbols.len());
    debug_assert_eq!(first_ones.len(), height);
    if use_weighted_buckets {
        debug_assert_eq!(row_weights.len(), height);
    } else {
        debug_assert!(row_weights.is_empty());
    }

    let add_assign_path = AddAssignFastPath::new(symbols.symbol_size());
    let batch_forward_symbols = width >= BINARY_FORWARD_SYMBOL_BATCH_MIN_WIDTH;
    let mut bucket_heads = vec![NO_BUCKET_ROW; width];
    let mut small_weight_buckets = SmallWeightBinaryBuckets::<u16>::new(
        if use_weighted_buckets { width } else { 0 },
        DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX,
    );
    let mut small_row_cache = SmallBinaryRowCache::<DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX>::new(
        if use_weighted_buckets { height } else { 0 },
    );
    let mut next_in_bucket = vec![NO_BUCKET_ROW; height];
    for (row, first_one) in first_ones.into_iter().enumerate() {
        if let Some(col) = first_one {
            if use_weighted_buckets {
                push_weighted_binary_row_bucket(
                    &row_weights,
                    &mut bucket_heads,
                    &mut small_weight_buckets,
                    &mut next_in_bucket,
                    col,
                    row,
                );
            } else {
                push_row_bucket(&mut bucket_heads, &mut next_in_bucket, col, row);
            }
        }
    }

    let mut pivot_for_col = vec![None; width];
    let mut is_pivot_row = vec![false; height];
    let mut forward_symbol_dests = Vec::new();

    for col in 0..width {
        let pivot = if use_weighted_buckets {
            pop_lightest_weighted_binary_row_bucket(
                &row_weights,
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                col,
            )
        } else {
            pop_lightest_binary_row_bucket(&rows, &mut bucket_heads, &mut next_in_bucket, col)
        };
        let Some(pivot) = pivot else {
            return (None, None);
        };
        pivot_for_col[col] = Some(pivot);
        is_pivot_row[pivot] = true;
        let pivot_weight = if use_weighted_buckets {
            row_weights[pivot]
        } else {
            0
        };

        if batch_forward_symbols {
            forward_symbol_dests.clear();
        }
        while let Some(row) = if use_weighted_buckets {
            pop_weighted_binary_row_bucket(
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                col,
            )
        } else {
            pop_row_bucket(&mut bucket_heads, &mut next_in_bucket, col)
        } {
            if use_weighted_buckets {
                let next_col = eliminate_weighted_binary_row::<DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX>(
                    &mut rows,
                    &mut row_weights,
                    &mut small_row_cache,
                    row,
                    pivot,
                    col,
                    pivot_weight,
                );
                if let Some(next_col) = next_col {
                    push_weighted_binary_row_bucket(
                        &row_weights,
                        &mut bucket_heads,
                        &mut small_weight_buckets,
                        &mut next_in_bucket,
                        next_col,
                        row,
                    );
                }
            } else {
                rows.xor_suffix(row, pivot, col);
                if let Some(next_col) = rows.first_one_at_or_after(row, col + 1) {
                    push_row_bucket(&mut bucket_heads, &mut next_in_bucket, next_col, row);
                }
            }
            if batch_forward_symbols {
                forward_symbol_dests.push(row);
            } else {
                let (pivot_symbol, dest_symbol) = symbols.get_disjoint_mut(pivot, row);
                add_assign_path.apply(dest_symbol, pivot_symbol);
            }
        }
        if batch_forward_symbols {
            addassign_symbol_row_batch::<true>(
                &mut symbols,
                pivot,
                &forward_symbol_dests,
                add_assign_path,
            );
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
        if width >= BINARY_SOURCE_BATCH_16_MIN_WIDTH && height == width {
            addassign_packed_row_sources_to_symbol::<16>(
                &rows,
                pivot,
                col + 1,
                &mut decoded,
                col,
                add_assign_path,
            );
        } else {
            addassign_packed_row_sources_to_symbol::<8>(
                &rows,
                pivot,
                col + 1,
                &mut decoded,
                col,
                add_assign_path,
            );
        }
    }

    (Some(decoded), None)
}

fn addassign_packed_row_sources_to_symbol<const BATCH: usize>(
    rows: &PackedBinaryRows,
    row: usize,
    start_col: usize,
    decoded: &mut SymbolSlab,
    dest: usize,
    add_assign_path: AddAssignFastPath,
) {
    let mut source_batch = [0usize; BATCH];
    let mut source_batch_len = 0usize;
    rows.visit_ones_at_or_after(row, start_col, |dependent_col| {
        source_batch[source_batch_len] = dependent_col;
        source_batch_len += 1;
        if source_batch_len == source_batch.len() {
            addassign_symbol_source_batch(decoded, dest, &source_batch, add_assign_path);
            source_batch_len = 0;
        }
    });
    addassign_symbol_source_batch(
        decoded,
        dest,
        &source_batch[..source_batch_len],
        add_assign_path,
    );
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

trait SmallWeightBucketMask: Copy {
    const BITS: usize;

    fn zero() -> Self;
    fn trailing_zeros(self) -> usize;
    fn set_bit(&mut self, bucket: usize);
    fn clear_bit(&mut self, bucket: usize);
}

impl SmallWeightBucketMask for u16 {
    const BITS: usize = u16::BITS as usize;

    fn zero() -> u16 {
        0
    }

    fn trailing_zeros(self) -> usize {
        u16::trailing_zeros(self) as usize
    }

    fn set_bit(&mut self, bucket: usize) {
        *self |= 1u16 << bucket;
    }

    fn clear_bit(&mut self, bucket: usize) {
        *self &= !(1u16 << bucket);
    }
}

impl SmallWeightBucketMask for u32 {
    const BITS: usize = u32::BITS as usize;

    fn zero() -> u32 {
        0
    }

    fn trailing_zeros(self) -> usize {
        u32::trailing_zeros(self) as usize
    }

    fn set_bit(&mut self, bucket: usize) {
        *self |= 1u32 << bucket;
    }

    fn clear_bit(&mut self, bucket: usize) {
        *self &= !(1u32 << bucket);
    }
}

struct SmallWeightBinaryBuckets<Mask: SmallWeightBucketMask> {
    heads: Vec<usize>,
    nonempty_masks: Vec<Mask>,
    width: usize,
    small_weight_max: usize,
}

impl<Mask: SmallWeightBucketMask> SmallWeightBinaryBuckets<Mask> {
    fn new(width: usize, small_weight_max: usize) -> SmallWeightBinaryBuckets<Mask> {
        debug_assert!(small_weight_max <= Mask::BITS);
        SmallWeightBinaryBuckets {
            heads: vec![NO_BUCKET_ROW; width * small_weight_max],
            nonempty_masks: vec![Mask::zero(); width],
            width,
            small_weight_max,
        }
    }

    fn push(&mut self, next_in_bucket: &mut [usize], col: usize, row: usize, weight: u32) -> bool {
        let Some(bucket) = small_weight_binary_bucket_index(weight, self.small_weight_max) else {
            return false;
        };
        debug_assert!(col < self.width);
        debug_assert_eq!(next_in_bucket[row], NO_BUCKET_ROW);

        let head_index = self.head_index(bucket, col);
        next_in_bucket[row] = self.heads[head_index];
        self.heads[head_index] = row;
        self.nonempty_masks[col].set_bit(bucket);
        true
    }

    fn pop_lightest(
        &mut self,
        next_in_bucket: &mut [usize],
        col: usize,
        row_weights: &[u32],
    ) -> Option<usize> {
        let bucket = self.nonempty_masks[col].trailing_zeros();
        if bucket >= self.small_weight_max {
            return None;
        }
        let row = self.pop_bucket(next_in_bucket, col, bucket)?;
        debug_assert_eq!(row_weights[row], bucket as u32 + 1);
        Some(row)
    }

    fn pop_any(&mut self, next_in_bucket: &mut [usize], col: usize) -> Option<usize> {
        let bucket = self.nonempty_masks[col].trailing_zeros();
        if bucket >= self.small_weight_max {
            return None;
        }
        self.pop_bucket(next_in_bucket, col, bucket)
    }

    fn pop_bucket(
        &mut self,
        next_in_bucket: &mut [usize],
        col: usize,
        bucket: usize,
    ) -> Option<usize> {
        let head_index = self.head_index(bucket, col);
        let row = self.heads[head_index];
        if row == NO_BUCKET_ROW {
            return None;
        }

        self.heads[head_index] = next_in_bucket[row];
        if self.heads[head_index] == NO_BUCKET_ROW {
            self.nonempty_masks[col].clear_bit(bucket);
        }
        next_in_bucket[row] = NO_BUCKET_ROW;
        Some(row)
    }

    fn head_index(&self, bucket: usize, col: usize) -> usize {
        debug_assert!(bucket < self.small_weight_max);
        bucket * self.width + col
    }
}

#[derive(Clone, Copy)]
struct SmallBinaryRowEntries<const CAPACITY: usize> {
    cols: [CoefficientColumn; CAPACITY],
    len: u8,
    valid: bool,
}

impl<const CAPACITY: usize> Default for SmallBinaryRowEntries<CAPACITY> {
    fn default() -> SmallBinaryRowEntries<CAPACITY> {
        SmallBinaryRowEntries {
            cols: [0; CAPACITY],
            len: 0,
            valid: false,
        }
    }
}

struct SmallBinaryRowCache<const CAPACITY: usize> {
    entries: Vec<SmallBinaryRowEntries<CAPACITY>>,
}

impl<const CAPACITY: usize> SmallBinaryRowCache<CAPACITY> {
    fn new(height: usize) -> SmallBinaryRowCache<CAPACITY> {
        debug_assert!(CAPACITY <= u32::BITS as usize);
        SmallBinaryRowCache {
            entries: vec![SmallBinaryRowEntries::default(); height],
        }
    }

    fn entries<'a>(
        &'a mut self,
        rows: &PackedBinaryRows,
        row: usize,
        start_col: usize,
    ) -> &'a [CoefficientColumn] {
        if !self.entries[row].valid {
            let entry = &mut self.entries[row];
            entry.len = 0;
            rows.visit_ones_at_or_after(row, start_col, |col| {
                let index = entry.len as usize;
                debug_assert!(index < entry.cols.len());
                entry.cols[index] = coefficient_col(col);
                entry.len += 1;
            });
            entry.valid = true;
        }
        let entry = &self.entries[row];
        &entry.cols[..entry.len as usize]
    }

    fn invalidate(&mut self, row: usize) {
        self.entries[row].valid = false;
    }
}

fn small_weight_binary_bucket_index(weight: u32, small_weight_max: usize) -> Option<usize> {
    let weight = weight as usize;
    (1..=small_weight_max)
        .contains(&weight)
        .then_some(weight - 1)
}

fn push_weighted_binary_row_bucket(
    row_weights: &[u32],
    bucket_heads: &mut [usize],
    small_weight_buckets: &mut SmallWeightBinaryBuckets<impl SmallWeightBucketMask>,
    next_in_bucket: &mut [usize],
    col: usize,
    row: usize,
) {
    let weight = row_weights[row];
    if !small_weight_buckets.push(next_in_bucket, col, row, weight) {
        push_row_bucket(bucket_heads, next_in_bucket, col, row);
    }
}

fn pop_weighted_binary_row_bucket(
    bucket_heads: &mut [usize],
    small_weight_buckets: &mut SmallWeightBinaryBuckets<impl SmallWeightBucketMask>,
    next_in_bucket: &mut [usize],
    col: usize,
) -> Option<usize> {
    small_weight_buckets
        .pop_any(next_in_bucket, col)
        .or_else(|| pop_row_bucket(bucket_heads, next_in_bucket, col))
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

fn pop_lightest_weighted_binary_row_bucket(
    row_weights: &[u32],
    bucket_heads: &mut [usize],
    small_weight_buckets: &mut SmallWeightBinaryBuckets<impl SmallWeightBucketMask>,
    next_in_bucket: &mut [usize],
    col: usize,
) -> Option<usize> {
    if let Some(row) = small_weight_buckets.pop_lightest(next_in_bucket, col, row_weights) {
        return Some(row);
    }

    let head = bucket_heads[col];
    if head == NO_BUCKET_ROW {
        return None;
    }
    let small_weight_max = small_weight_buckets.small_weight_max as u32;
    debug_assert!(row_weights[head] > small_weight_max);
    if next_in_bucket[head] == NO_BUCKET_ROW {
        bucket_heads[col] = NO_BUCKET_ROW;
        return Some(head);
    }
    let min_general_weight = small_weight_max + 1;
    if row_weights[head] == min_general_weight {
        bucket_heads[col] = next_in_bucket[head];
        next_in_bucket[head] = NO_BUCKET_ROW;
        return Some(head);
    }

    let mut best = head;
    let mut best_previous = NO_BUCKET_ROW;
    let mut best_weight = row_weights[head];
    let mut previous = head;
    let mut current = next_in_bucket[head];

    while current != NO_BUCKET_ROW {
        let row = current;
        let weight = row_weights[row];
        debug_assert!(weight > small_weight_max);
        if weight < best_weight {
            best = row;
            best_previous = previous;
            best_weight = weight;
            if weight == min_general_weight {
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
    fn column_major_hdpc_verification_matches_row_major() {
        let mut hdpc_rows = DenseOctetMatrix::new(3, 4);
        hdpc_rows.row_mut(0).copy_from_slice(&[
            Octet::new(2),
            Octet::zero(),
            Octet::new(9),
            Octet::one(),
        ]);
        hdpc_rows.row_mut(1).copy_from_slice(&[
            Octet::zero(),
            Octet::new(7),
            Octet::new(11),
            Octet::new(5),
        ]);
        hdpc_rows.row_mut(2).copy_from_slice(&[
            Octet::new(3),
            Octet::one(),
            Octet::zero(),
            Octet::new(13),
        ]);

        let decoded =
            SymbolSlab::from_bytes(vec![0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87], 2);

        assert_eq!(
            hdpc_rows_satisfied_column_major(&decoded, &hdpc_rows),
            hdpc_rows_satisfied_row_major(&decoded, &hdpc_rows)
        );
        assert!(!hdpc_rows_satisfied_column_major(&decoded, &hdpc_rows));

        let zero_decoded = SymbolSlab::with_zeros(4, 2);
        assert!(hdpc_rows_satisfied_column_major(&zero_decoded, &hdpc_rows));
    }

    #[test]
    fn horner_hdpc_verification_matches_generated_rows() {
        for source_symbols in [10, 100, 1000] {
            let hdpc_rows = generate_hdpc_rows(source_symbols);
            let symbol_size = 3usize;
            let mut bytes = vec![0u8; hdpc_rows.width() * symbol_size];
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = (index as u8).wrapping_mul(29).wrapping_add(17);
            }
            let decoded = SymbolSlab::from_bytes(bytes, symbol_size);

            assert_eq!(
                rfc_hdpc_rows_satisfied_horner(&decoded, hdpc_rows.height(), true),
                hdpc_rows_satisfied_row_major(&decoded, &hdpc_rows)
            );

            let zero_decoded = SymbolSlab::with_zeros(hdpc_rows.width(), symbol_size);
            assert!(rfc_hdpc_rows_satisfied_horner(
                &zero_decoded,
                hdpc_rows.height(),
                true
            ));
        }
    }

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
            hdpc_free_solve: None,
            free_cols: vec![coefficient_col(4)].into_boxed_slice(),
            pivot_symbol_moves: direct_pivot_symbol_moves(&pivot_for_col, 1, 1, 5),
            back_substitution: DirectSystematicBackSubstitution::SourcesByDest {
                slices: DirectSystematicSlices {
                    ranges: Vec::new(),
                    entries: Vec::new(),
                },
                non_empty_dests: Vec::new().into_boxed_slice(),
            },
            trust_source_batch_bounds: true,
            width: 5,
            s: 1,
            h: 1,
        };
        let mut symbols = SymbolSlab::from_bytes(vec![10, 11, 12, 13, 14], 1);

        move_direct_pivot_symbols_to_columns(&plan, &mut symbols);

        assert_eq!(symbols.as_bytes(), &[12, 13, 10, 14, 14]);
    }

    #[test]
    fn sources_by_dest_direct_back_substitution_replays_nonzero_chain() {
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
            hdpc_free_solve: None,
            free_cols: Vec::new().into_boxed_slice(),
            pivot_symbol_moves: Vec::new(),
            back_substitution: DirectSystematicBackSubstitution::SourcesByDest {
                slices: DirectSystematicSlices {
                    ranges: vec![(0, 2), (2, 3), (3, 4), (4, 4)],
                    entries: vec![
                        coefficient_col(1),
                        coefficient_col(3),
                        coefficient_col(2),
                        coefficient_col(3),
                    ],
                },
                non_empty_dests: vec![coefficient_col(0), coefficient_col(1), coefficient_col(2)]
                    .into_boxed_slice(),
            },
            trust_source_batch_bounds: true,
            width: 4,
            s: 0,
            h: 0,
        };
        let mut symbols = SymbolSlab::from_bytes(
            vec![
                0x10, 0x11, 0x12, 0x20, 0x21, 0x22, 0x40, 0x41, 0x42, 0x80, 0x81, 0x82,
            ],
            3,
        );

        apply_prepared_direct_systematic_plan(&plan, &mut symbols);

        assert_eq!(
            symbols.as_bytes(),
            &[
                0x70, 0x71, 0x72, 0xe0, 0xe1, 0xe2, 0xc0, 0xc0, 0xc0, 0x80, 0x81, 0x82
            ]
        );
    }

    #[test]
    fn large_sources_by_dest_direct_back_substitution_uses_non_empty_dests() {
        let width = DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH;
        let mut ranges = vec![(4, 4); width];
        ranges[0] = (0, 2);
        ranges[1] = (2, 3);
        ranges[2] = (3, 4);
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
            hdpc_free_solve: None,
            free_cols: Vec::new().into_boxed_slice(),
            pivot_symbol_moves: Vec::new(),
            back_substitution: DirectSystematicBackSubstitution::SourcesByDest {
                slices: DirectSystematicSlices {
                    ranges,
                    entries: vec![
                        coefficient_col(1),
                        coefficient_col(3),
                        coefficient_col(2),
                        coefficient_col(3),
                    ],
                },
                non_empty_dests: vec![coefficient_col(0), coefficient_col(1), coefficient_col(2)]
                    .into_boxed_slice(),
            },
            trust_source_batch_bounds: true,
            width,
            s: 0,
            h: 0,
        };
        let mut bytes = vec![0; width * 3];
        bytes[..12].copy_from_slice(&[
            0x10, 0x11, 0x12, 0x20, 0x21, 0x22, 0x40, 0x41, 0x42, 0x80, 0x81, 0x82,
        ]);
        let mut symbols = SymbolSlab::from_bytes(bytes, 3);

        apply_prepared_direct_systematic_plan(&plan, &mut symbols);

        assert_eq!(
            &symbols.as_bytes()[..12],
            &[
                0x70, 0x71, 0x72, 0xe0, 0xe1, 0xe2, 0xc0, 0xc0, 0xc0, 0x80, 0x81, 0x82
            ]
        );
        assert!(symbols.as_bytes()[12..].iter().all(|&byte| byte == 0));
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
        let mut bucket_heads = vec![1];
        let mut next_in_bucket = vec![NO_BUCKET_ROW, NO_BUCKET_ROW];
        let mut small_weight_buckets =
            SmallWeightBinaryBuckets::<u16>::new(1, DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX);
        assert!(small_weight_buckets.push(&mut next_in_bucket, 0, 0, row_weights[0]));

        assert_eq!(
            pop_lightest_weighted_binary_row_bucket(
                &row_weights,
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                0,
            ),
            Some(0)
        );
        assert_eq!(bucket_heads[0], 1);
        assert_eq!(small_weight_buckets.nonempty_masks[0], 0);
        assert_eq!(next_in_bucket[0], NO_BUCKET_ROW);
    }

    #[test]
    fn weighted_binary_bucket_returns_small_weight_before_general_chain() {
        let row_weights = vec![12, 7, 9];
        let mut bucket_heads = vec![2];
        let mut next_in_bucket = vec![NO_BUCKET_ROW, NO_BUCKET_ROW, NO_BUCKET_ROW];
        let mut small_weight_buckets =
            SmallWeightBinaryBuckets::<u16>::new(1, DECODE_SMALL_WEIGHT_BINARY_BUCKET_MAX);
        assert!(small_weight_buckets.push(&mut next_in_bucket, 0, 1, row_weights[1]));

        assert_eq!(
            pop_lightest_weighted_binary_row_bucket(
                &row_weights,
                &mut bucket_heads,
                &mut small_weight_buckets,
                &mut next_in_bucket,
                0,
            ),
            Some(1)
        );
        assert_eq!(bucket_heads[0], 2);
        assert_eq!(small_weight_buckets.nonempty_masks[0], 0);
        assert_eq!(next_in_bucket[1], NO_BUCKET_ROW);
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

        fn packed_row_prefix(&self, height: usize) -> PackedBinaryRows {
            packed_prefix_rows(&self.rows, height)
        }

        fn packed_row_prefix_with_row_weights_and_first_ones(
            &self,
            height: usize,
        ) -> (PackedBinaryRows, Vec<u32>, Vec<Option<usize>>) {
            let rows = packed_prefix_rows(&self.rows, height);
            let row_weights = (0..rows.height())
                .map(|row| rows.weight_at_or_after(row, 0))
                .collect();
            let first_ones = (0..rows.height())
                .map(|row| rows.first_one_at_or_after(row, 0))
                .collect();
            (rows, row_weights, first_ones)
        }

        fn packed_rows_with_first_ones(&self) -> (PackedBinaryRows, Vec<Option<usize>>) {
            let rows = self.rows.clone();
            let first_ones = (0..rows.height())
                .map(|row| rows.first_one_at_or_after(row, 0))
                .collect();
            (rows, first_ones)
        }

        fn packed_rows_with_row_weights_and_first_ones(
            &self,
        ) -> (PackedBinaryRows, Vec<u32>, Vec<Option<usize>>) {
            let rows = self.rows.clone();
            let row_weights = (0..rows.height())
                .map(|row| rows.weight_at_or_after(row, 0))
                .collect();
            let first_ones = (0..rows.height())
                .map(|row| rows.first_one_at_or_after(row, 0))
                .collect();
            (rows, row_weights, first_ones)
        }

        fn visit_row_entries<F>(&self, row: usize, _visit: F)
        where
            F: FnMut(usize),
        {
            assert!(row < self.height());
        }
    }

    fn packed_prefix_rows(rows: &PackedBinaryRows, height: usize) -> PackedBinaryRows {
        assert!(height <= rows.height());
        let mut prefix = PackedBinaryRows::new(height, rows.width());
        for row in 0..height {
            rows.visit_ones_at_or_after(row, 0, |col| prefix.set(row, col));
        }
        prefix
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
    fn direct_square_decode_rejects_unsatisfied_hdpc_without_panicking() {
        let width = IN_PLACE_HYBRID_REPLAY_MIN_WIDTH + 1;
        let source_block_symbols = 10;
        let s = num_ldpc_symbols(source_block_symbols) as usize;
        let symbol_size = 1;
        let matrix = PackedOnlyMatrix::new(packed_identity_with_free_last_column(width));
        let hdpc_rows = DenseOctetMatrix::new(1, width);
        let mut symbols = SymbolSlab::with_zeros(width, symbol_size);
        symbols.get_mut(s)[0] = 0x5a;

        let result =
            try_direct_square_hybrid_decode(source_block_symbols, &matrix, &hdpc_rows, symbols);

        match result {
            SquareHybridDecodeResult::Failed => {}
            SquareHybridDecodeResult::Decoded(_) => {
                panic!("inconsistent direct square decode unexpectedly succeeded")
            }
            SquareHybridDecodeResult::Fallback(_) => {
                panic!("direct square decode should build a plan for this fixture")
            }
        }
    }

    #[test]
    fn square_hybrid_owned_decode_solves_nonzero_free_dependency() {
        let width = IN_PLACE_HYBRID_REPLAY_MIN_WIDTH + 1;
        let source_block_symbols = 10;
        let s = num_ldpc_symbols(source_block_symbols) as usize;
        let h = 1;
        let symbol_size = 2;
        let free_col = width - 1;
        let binary_height = width - h;
        let mut rows = PackedBinaryRows::new(binary_height, width);
        rows.set(0, 0);
        rows.set(0, free_col);
        for col in 1..free_col {
            rows.set(col, col);
        }
        let matrix = PackedOnlyMatrix::new(rows);
        let mut hdpc_rows = DenseOctetMatrix::new(h, width);
        hdpc_rows.set(0, free_col, Octet::one());
        let mut symbols = SymbolSlab::with_zeros(width, symbol_size);
        for row in 0..binary_height {
            symbols
                .get_mut(mapped_binary_symbol_row(row, s, h))
                .copy_from_slice(&[row as u8, row.wrapping_mul(3) as u8]);
        }
        symbols.get_mut(0).copy_from_slice(&[0x11, 0x22]);
        symbols.get_mut(s).copy_from_slice(&[0x5a, 0xa5]);

        let (decoded, ops) =
            fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_block_symbols);
        let decoded = decoded.expect("square owned hybrid solve should succeed");

        assert!(ops.is_none());
        assert_eq!(decoded.get(0), &[0x4b, 0x87]);
        assert_eq!(decoded.get(1), &[1, 3]);
        assert_eq!(
            decoded.get(free_col - 1),
            &[(free_col - 1) as u8, (free_col - 1).wrapping_mul(3) as u8]
        );
        assert_eq!(decoded.get(free_col), &[0x5a, 0xa5]);
    }

    #[test]
    fn direct_square_plan_matches_hybrid_decode_for_free_dependency() {
        let width = 64;
        let source_block_symbols = 10;
        let s = num_ldpc_symbols(source_block_symbols) as usize;
        let h = 1;
        let symbol_size = 2;
        let free_col = width - 1;
        let binary_height = width - h;
        let mut rows = PackedBinaryRows::new(binary_height, width);
        rows.set(0, 0);
        rows.set(0, free_col);
        for col in 1..free_col {
            rows.set(col, col);
        }
        let matrix = PackedOnlyMatrix::new(rows);
        let mut hdpc_rows = DenseOctetMatrix::new(h, width);
        hdpc_rows.set(0, free_col, Octet::one());
        let mut symbols = SymbolSlab::with_zeros(width, symbol_size);
        for row in 0..binary_height {
            symbols
                .get_mut(mapped_binary_symbol_row(row, s, h))
                .copy_from_slice(&[row as u8, row.wrapping_mul(5) as u8]);
        }
        symbols.get_mut(0).copy_from_slice(&[0x33, 0x44]);
        symbols.get_mut(s).copy_from_slice(&[0x9a, 0xbc]);

        let mut direct = symbols.clone();
        let plan = prepare_direct_systematic_plan(&matrix, &hdpc_rows, source_block_symbols)
            .expect("square direct plan should solve the free HDPC dependency");
        apply_prepared_direct_systematic_plan(&plan, &mut direct);

        let hybrid =
            try_hybrid_binary_hdpc_solve(&matrix, &hdpc_rows, &symbols, source_block_symbols)
                .expect("hybrid solve should succeed");

        assert_eq!(direct, hybrid);
        assert_eq!(direct.get(0), &[0xa9, 0xf8]);
        assert_eq!(direct.get(free_col), &[0x9a, 0xbc]);
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
    fn in_place_hybrid_replay_is_limited_to_mid_width_500_symbol_band() {
        let width_for = |source_symbols| num_intermediate_symbols(source_symbols) as usize;

        assert!(!use_in_place_hybrid_replay(width_for(250)));
        assert!(use_in_place_hybrid_replay(width_for(500)));
        assert!(!use_in_place_hybrid_replay(width_for(1000)));
        assert!(!use_in_place_hybrid_replay(width_for(2000)));
    }

    #[test]
    fn direct_square_decode_threshold_covers_low_ci_exact_rows() {
        let width_for = |source_symbols| num_intermediate_symbols(source_symbols) as usize;

        assert!(width_for(10) < DIRECT_SQUARE_HYBRID_DECODE_MIN_WIDTH);
        assert!(width_for(100) >= DIRECT_SQUARE_HYBRID_DECODE_MIN_WIDTH);
        assert!(width_for(250) >= DIRECT_SQUARE_HYBRID_DECODE_MIN_WIDTH);
        assert!(width_for(500) >= DIRECT_SQUARE_HYBRID_DECODE_MIN_WIDTH);
    }

    #[test]
    fn column_major_hdpc_verify_threshold_covers_medium_overhead_rows() {
        let width_for = |source_symbols| num_intermediate_symbols(source_symbols) as usize;

        assert!(width_for(100) < COLUMN_MAJOR_HDPC_VERIFY_MIN_WIDTH);
        assert!(width_for(250) >= COLUMN_MAJOR_HDPC_VERIFY_MIN_WIDTH);
        assert!(width_for(500) >= COLUMN_MAJOR_HDPC_VERIFY_MIN_WIDTH);
        assert!(width_for(1_000) >= COLUMN_MAJOR_HDPC_VERIFY_MIN_WIDTH);
        assert!(width_for(2_000) >= COLUMN_MAJOR_HDPC_VERIFY_MIN_WIDTH);
    }

    #[test]
    fn direct_forward_no_zero_check_starts_after_5k_encode_row() {
        let width_for = |source_symbols| num_intermediate_symbols(source_symbols) as usize;

        assert!(!use_direct_forward_no_zero_check(width_for(5_000)));
        assert!(use_direct_forward_no_zero_check(width_for(10_000)));
        assert!(use_direct_forward_no_zero_check(width_for(20_000)));
        assert!(use_direct_forward_no_zero_check(width_for(50_000)));
    }

    #[test]
    fn large_systematic_plan_uses_hybrid_direct_systematic_solve() {
        let source_symbols = 4_000;
        let k_prime = extended_source_block_symbols(source_symbols);
        let symbols = SymbolSlab::with_zeros(num_intermediate_symbols(source_symbols) as usize, 1);
        let indices: Vec<u32> = (0..k_prime).collect();
        let (matrix, hdpc_rows) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &indices);

        let (decoded, ops) = fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_symbols);

        let decoded = decoded.expect("hybrid systematic solve should still return zero symbols");
        assert!(decoded.as_bytes().iter().all(|&byte| byte == 0));
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
                use_direct_systematic_solve(width)
            })
            .expect("direct systematic test threshold should be reachable")
    }

    fn first_cached_recording_source_symbols() -> u32 {
        (1..=MAX_INLINE_RECORDED_SOLVER_WIDTH as u32)
            .find(|&source_symbols| {
                let width = num_intermediate_symbols(source_symbols) as usize;
                (CACHED_SYSTEMATIC_PLAN_RECORDING_MIN_WIDTH..=MAX_INLINE_RECORDED_SOLVER_WIDTH)
                    .contains(&width)
            })
            .expect("cached recording test threshold should be reachable")
    }

    fn first_source_batched_direct_source_symbols() -> u32 {
        (1..=SQUARE_HYBRID_MAX_WIDTH as u32)
            .find(|&source_symbols| {
                let width = num_intermediate_symbols(source_symbols) as usize;
                (DIRECT_SOURCE_BATCH_BACK_SUBSTITUTION_MIN_WIDTH..=SQUARE_HYBRID_MAX_WIDTH)
                    .contains(&width)
            })
            .expect("source-batched direct test threshold should be reachable")
    }

    #[test]
    fn mid_width_systematic_plan_uses_cached_semantic_replay() {
        let source_symbols = first_cached_recording_source_symbols();
        let k_prime = extended_source_block_symbols(source_symbols);
        let width = num_intermediate_symbols(source_symbols) as usize;
        let s = num_ldpc_symbols(source_symbols) as usize;
        let h = num_hdpc_symbols(source_symbols) as usize;
        let mut symbols = SymbolSlab::with_zeros(width, 1);
        for isi in 0..source_symbols as usize {
            symbols.get_mut(s + h + isi)[0] = (isi as u8).wrapping_mul(11).wrapping_add(5);
        }
        let original_symbols = symbols.clone();
        let indices: Vec<u32> = (0..k_prime).collect();
        let (matrix, hdpc_rows) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &indices);
        let (expected_matrix, expected_hdpc_rows) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &indices);
        assert_eq!(matrix.width(), width);
        assert!(width <= MAX_INLINE_RECORDED_SOLVER_WIDTH);
        assert!(is_full_systematic_planning_matrix(&matrix, source_symbols));
        let (expected, _) = fused_inverse_mul_symbols_impl(
            expected_matrix,
            expected_hdpc_rows,
            original_symbols.clone(),
            source_symbols,
            OperationRecording::Skip,
        );
        let expected = expected.expect("mid-width systematic solve should decode");

        let (decoded, ops) = fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_symbols);
        assert_eq!(
            decoded.expect("mid-width systematic solve should decode"),
            expected
        );
        let ops = ops.expect("mid-width systematic solve should record a cached replay");

        match ops.as_slice() {
            [
                SymbolOps::DirectSystematicSolve {
                    source_block_symbols,
                }
                | SymbolOps::ApplyCachedSystematicPlan {
                    source_block_symbols,
                },
            ] => assert_eq!(*source_block_symbols, k_prime),
            _ => panic!("mid-width systematic solve should record one cached semantic op"),
        }

        let mut replayed = original_symbols;
        for op in &ops {
            crate::operation_vector::perform_op(op, &mut replayed);
        }
        assert_eq!(replayed, expected);
    }

    #[test]
    fn ci_mid_count_systematic_plans_record_cached_replay() {
        for source_symbols in [100, 250] {
            let k_prime = extended_source_block_symbols(source_symbols);
            let width = num_intermediate_symbols(source_symbols) as usize;
            assert!(width >= CACHED_SYSTEMATIC_PLAN_RECORDING_MIN_WIDTH);
            assert!(width < DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH);

            let symbols = SymbolSlab::with_zeros(width, 1);
            let indices: Vec<u32> = (0..k_prime).collect();
            let (matrix, hdpc_rows) =
                generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &indices);

            let (decoded, ops) =
                fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_symbols);
            let decoded = decoded.expect("systematic plan should still return decoded symbols");
            assert_eq!(decoded.len(), width);
            assert!(decoded.as_bytes().iter().all(|&byte| byte == 0));
            match ops
                .expect("systematic plan should record one cached replay")
                .as_slice()
            {
                [
                    SymbolOps::DirectSystematicSolve {
                        source_block_symbols,
                    }
                    | SymbolOps::ApplyCachedSystematicPlan {
                        source_block_symbols,
                    },
                ] => assert_eq!(*source_block_symbols, k_prime),
                _ => panic!("systematic plan should record one cached replay op"),
            }
        }
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
    fn ci_1000_systematic_plan_uses_direct_systematic_solve() {
        let source_symbols = 1_000;
        let k_prime = extended_source_block_symbols(source_symbols);
        let width = num_intermediate_symbols(source_symbols) as usize;
        assert!(use_direct_systematic_solve(width));

        let symbols = SymbolSlab::with_zeros(width, 1);
        let indices: Vec<u32> = (0..k_prime).collect();
        let (matrix, hdpc_rows) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &indices);

        let (decoded, ops) = fused_inverse_mul_symbols(matrix, hdpc_rows, symbols, source_symbols);
        assert_eq!(
            decoded
                .expect("1000-symbol direct systematic solve should decode")
                .len(),
            width
        );
        assert!(matches!(
            ops.expect("1000-symbol direct systematic solve should be recorded")
                .as_slice(),
            [SymbolOps::DirectSystematicSolve {
                source_block_symbols
            }] if *source_block_symbols == k_prime
        ));
    }

    #[test]
    fn ci_2000_systematic_plan_uses_trusted_direct_solve_and_source_batches() {
        let source_symbols = 2_000;
        let k_prime = extended_source_block_symbols(source_symbols);
        let width = num_intermediate_symbols(source_symbols) as usize;
        assert!(!use_direct_systematic_solve(width));
        assert!(use_trusted_direct_systematic_solve(width));
        assert!(matches!(
            direct_decode_back_substitution_layout(width),
            DirectBackSubstitutionLayout::SourcesByDest
        ));

        let symbols = SymbolSlab::with_zeros(width, 1);
        let indices: Vec<u32> = (0..k_prime).collect();
        let (matrix, hdpc_rows) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &indices);

        let (decoded, ops) =
            fused_inverse_mul_symbols(matrix.clone(), hdpc_rows.clone(), symbols, source_symbols);
        assert_eq!(
            decoded
                .expect("2000-symbol trusted direct systematic solve should decode")
                .len(),
            width
        );
        assert!(matches!(
            ops.expect("2000-symbol trusted direct systematic solve should be recorded")
                .as_slice(),
            [SymbolOps::DirectSystematicSolve {
                source_block_symbols
            }] if *source_block_symbols == k_prime
        ));

        let encode_plan = prepare_direct_systematic_plan(&matrix, &hdpc_rows, source_symbols)
            .expect("trusted 2000-symbol direct plan should build");
        assert!(matches!(
            &encode_plan.back_substitution,
            DirectSystematicBackSubstitution::SourcesByDest { .. }
        ));

        let decode_plan =
            prepare_direct_systematic_plan_for_decode(&matrix, &hdpc_rows, source_symbols)
                .expect("decode 2000-symbol direct plan should build");
        assert!(matches!(
            &decode_plan.back_substitution,
            DirectSystematicBackSubstitution::SourcesByDest { .. }
        ));
    }

    #[test]
    fn ci_100_and_250_systematic_plans_use_low_width_direct_solve() {
        for source_symbols in [100, 250] {
            let k_prime = extended_source_block_symbols(source_symbols);
            let width = num_intermediate_symbols(source_symbols) as usize;
            assert!(
                (LOW_DIRECT_SYSTEMATIC_SOLVE_MIN_WIDTH..LOW_DIRECT_SYSTEMATIC_SOLVE_MAX_WIDTH)
                    .contains(&width)
            );
            assert!(use_direct_systematic_solve(width));
            assert!(use_direct_source_batch_back_substitution(width));

            let indices: Vec<u32> = (0..k_prime).collect();
            let (matrix, hdpc_rows) =
                generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &indices);
            let plan = prepare_direct_systematic_plan(&matrix, &hdpc_rows, source_symbols)
                .expect("low-width direct plan should build");

            assert!(matches!(
                &plan.back_substitution,
                DirectSystematicBackSubstitution::SourcesByDest { .. }
            ));
        }
    }

    #[test]
    fn source_batched_direct_plan_starts_at_large_width() {
        let source_symbols = first_source_batched_direct_source_symbols();
        let k_prime = extended_source_block_symbols(source_symbols);
        let width = num_intermediate_symbols(source_symbols) as usize;
        let indices: Vec<u32> = (0..k_prime).collect();
        let (matrix, hdpc_rows) =
            generate_constraint_matrix::<SparseBinaryMatrix>(source_symbols, &indices);

        assert!(width >= DIRECT_SOURCE_BATCH_BACK_SUBSTITUTION_MIN_WIDTH);
        let plan = prepare_direct_systematic_plan(&matrix, &hdpc_rows, source_symbols)
            .expect("source-batched direct plan should build");

        assert!(matches!(
            &plan.back_substitution,
            DirectSystematicBackSubstitution::SourcesByDest { .. }
        ));
        if !plan.free_cols.is_empty() {
            assert!(plan.hdpc_free_solve.is_some());
        }

        let decode_plan =
            prepare_direct_systematic_plan_for_decode(&matrix, &hdpc_rows, source_symbols)
                .expect("direct decode plan should build");
        assert!(decode_plan.hdpc_free_solve.is_none());
    }

    #[test]
    fn source_batched_decode_plan_starts_at_large_width() {
        assert!(matches!(
            direct_decode_back_substitution_layout(
                DIRECT_DECODE_SOURCE_BATCH_BACK_SUBSTITUTION_MIN_WIDTH - 1
            ),
            DirectBackSubstitutionLayout::DestsBySource
        ));
        assert!(matches!(
            direct_decode_back_substitution_layout(
                DIRECT_DECODE_SOURCE_BATCH_BACK_SUBSTITUTION_MIN_WIDTH
            ),
            DirectBackSubstitutionLayout::SourcesByDest
        ));
    }

    #[test]
    fn direct_collect_source_batches_stays_on_square_source_by_dest_plans() {
        let width = DIRECT_SOURCE_BATCH_DIRECT_COLLECT_MIN_WIDTH;
        let h = 10;
        let square_binary_height = width - h;

        assert!(use_direct_collect_sources_by_dest(
            width,
            square_binary_height,
            h,
            DirectBackSubstitutionLayout::SourcesByDest
        ));
        assert!(!use_direct_collect_sources_by_dest(
            width - 1,
            square_binary_height - 1,
            h,
            DirectBackSubstitutionLayout::SourcesByDest
        ));
        assert!(!use_direct_collect_sources_by_dest(
            width,
            square_binary_height + 1,
            h,
            DirectBackSubstitutionLayout::SourcesByDest
        ));
        assert!(!use_direct_collect_sources_by_dest(
            width,
            square_binary_height,
            h,
            DirectBackSubstitutionLayout::DestsBySource
        ));
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
    fn cached_hdpc_free_solve_matches_existing_solver() {
        let rows = vec![
            vec![
                (coefficient_col(0), Octet::one()),
                (coefficient_col(1), Octet::one()),
            ],
            vec![(coefficient_col(1), Octet::one())],
            Vec::new(),
        ];
        let symbols = SymbolSlab::from_bytes(vec![3, 5, 7, 11, 0, 0], 2);
        let expected = solve_without_recording(rows.clone(), 2, symbols.clone())
            .0
            .expect("free solve should be consistent");
        let cached = prepare_cached_hdpc_free_solve_from_rows(&rows, 2, 1)
            .expect("free solve should record row operations");

        let actual = apply_cached_hdpc_free_solve(&cached, symbols)
            .expect("cached free solve should be consistent");

        assert_eq!(actual, expected);

        let inconsistent = SymbolSlab::from_bytes(vec![3, 5, 7, 11, 1, 0], 2);
        assert!(apply_cached_hdpc_free_solve(&cached, inconsistent).is_none());
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
    fn large_overdetermined_no_hdpc_system_decodes_above_legacy_cap() {
        let width = MAX_SUPPORTED_INTERMEDIATE_SYMBOLS as usize + 1;
        let mut rows = PackedBinaryRows::new(width + 1, width);
        for col in 0..width {
            rows.set(col, col);
        }
        rows.set(width, 0);
        let matrix = PackedOnlyMatrix::new(rows);

        let mut bytes = (0..width)
            .map(|col| (col as u8).wrapping_mul(17).wrapping_add(3))
            .collect::<Vec<_>>();
        bytes.push(bytes[0]);
        let expected = bytes[..width].to_vec();
        let symbols = SymbolSlab::from_bytes(bytes, 1);

        let (decoded, ops) = fused_inverse_mul_symbols_no_hdpc(matrix, symbols, 1);

        assert_eq!(decoded.unwrap().as_bytes(), expected.as_slice());
        assert!(ops.is_none());
    }

    #[test]
    fn binary_solve_accepts_precomputed_weight_metadata() {
        let width = LARGE_BINARY_WEIGHTED_BUCKET_MIN_WIDTH;
        let mut rows = PackedBinaryRows::new(width, width);
        let mut row_weights = Vec::with_capacity(width);
        let mut first_ones = Vec::with_capacity(width);
        for col in 0..width {
            rows.set(col, col);
            row_weights.push(1);
            first_ones.push(Some(col));
        }
        let expected = (0..width)
            .map(|col| (col as u8).wrapping_mul(19).wrapping_add(7))
            .collect::<Vec<_>>();
        let symbols = SymbolSlab::from_bytes(expected.clone(), 1);

        let (decoded, ops) =
            solve_binary_with_weighted_metadata(rows, symbols, row_weights, first_ones);

        assert_eq!(decoded.unwrap().as_bytes(), expected.as_slice());
        assert!(ops.is_none());
    }

    #[test]
    fn overdetermined_no_hdpc_prefix_accepts_verified_extra_rows() {
        let source_symbols = 1;
        let width = num_intermediate_symbols(source_symbols) as usize;
        let h = num_hdpc_symbols(source_symbols) as usize;
        let mut matrix = DenseBinaryMatrix::new(width + h + 1, width);
        for col in 0..width {
            matrix.set(col, col, true);
        }
        let symbols = SymbolSlab::with_zeros(matrix.height(), 1);

        let owned_decoded = match try_overdetermined_no_hdpc_prefix_solve_owned(
            &matrix,
            symbols.clone(),
            source_symbols,
        ) {
            OverdeterminedNoHdpcPrefixSolve::Decoded(decoded) => decoded,
            OverdeterminedNoHdpcPrefixSolve::Fallback(_) => panic!("owned prefix solve should run"),
            OverdeterminedNoHdpcPrefixSolve::Failed => panic!("owned prefix solve should decode"),
        };
        let decoded =
            try_overdetermined_no_hdpc_prefix_solve(&matrix, &symbols, source_symbols).unwrap();

        assert_eq!(owned_decoded, SymbolSlab::with_zeros(width, 1));
        assert_eq!(decoded, SymbolSlab::with_zeros(width, 1));
    }

    #[test]
    fn overdetermined_no_hdpc_prefix_falls_back_when_suffix_rows_complete_rank() {
        let source_symbols = 1;
        let width = 4;
        let h = num_hdpc_symbols(source_symbols) as usize;
        let prefix_height = width + h;
        let mut matrix = DenseBinaryMatrix::new(prefix_height + 1, width);
        matrix.set(0, 0, true);
        matrix.set(0, 1, true);
        matrix.set(1, 0, true);
        matrix.set(2, 2, true);
        matrix.set(prefix_height, width - 1, true);
        let expected = [0x11, 0x22, 0x33, 0x44];
        let mut symbols = SymbolSlab::with_zeros(matrix.height(), 1);
        symbols.get_mut(0)[0] = expected[0] ^ expected[1];
        symbols.get_mut(1)[0] = expected[0];
        symbols.get_mut(2)[0] = expected[2];
        symbols.get_mut(prefix_height)[0] = expected[3];

        let returned_symbols = match try_overdetermined_no_hdpc_prefix_solve_owned(
            &matrix,
            symbols.clone(),
            source_symbols,
        ) {
            OverdeterminedNoHdpcPrefixSolve::Decoded(_) => {
                panic!("rank-deficient prefix should not decode")
            }
            OverdeterminedNoHdpcPrefixSolve::Fallback(returned_symbols) => returned_symbols,
            OverdeterminedNoHdpcPrefixSolve::Failed => {
                panic!("rank-deficient prefix should fall back")
            }
        };

        assert_eq!(returned_symbols, symbols);
        let (decoded, ops) =
            fused_inverse_mul_symbols_no_hdpc(matrix, returned_symbols, source_symbols);

        assert_eq!(decoded.unwrap().as_bytes(), expected.as_slice());
        assert!(ops.is_none());
    }

    #[test]
    fn public_no_hdpc_decode_restores_owned_prefix_before_suffix_rank_fallback() {
        let source_symbols = 1;
        let width = OVERDETERMINED_NO_HDPC_PREFIX_OWNED_MIN_WIDTH;
        let h = num_hdpc_symbols(source_symbols) as usize;
        let prefix_height = width + h;
        let mut rows = PackedBinaryRows::new(prefix_height + 1, width);
        rows.set(0, 0);
        rows.set(0, 1);
        rows.set(1, 0);
        for col in 2..width - 1 {
            rows.set(col, col);
        }
        rows.set(prefix_height, width - 1);
        let matrix = PackedOnlyMatrix::new(rows);

        let symbol_size = 2;
        let col0 = [0x11, 0x22];
        let col1 = [0x33, 0x44];
        let col2 = [0x55, 0x66];
        let last = [0x77, 0x88];
        let mut expected = SymbolSlab::with_zeros(width, symbol_size);
        expected.get_mut(0).copy_from_slice(&col0);
        expected.get_mut(1).copy_from_slice(&col1);
        expected.get_mut(2).copy_from_slice(&col2);
        expected.get_mut(width - 1).copy_from_slice(&last);

        let mut symbols = SymbolSlab::with_zeros(prefix_height + 1, symbol_size);
        symbols
            .get_mut(0)
            .copy_from_slice(&[col0[0] ^ col1[0], col0[1] ^ col1[1]]);
        symbols.get_mut(1).copy_from_slice(&col0);
        symbols.get_mut(2).copy_from_slice(&col2);
        symbols.get_mut(prefix_height).copy_from_slice(&last);

        let (decoded, ops) = fused_inverse_mul_symbols_no_hdpc(matrix, symbols, source_symbols);

        assert_eq!(decoded.unwrap(), expected);
        assert!(ops.is_none());
    }

    #[test]
    fn overdetermined_no_hdpc_prefix_rejects_unverified_extra_rows() {
        let source_symbols = 1;
        let width = num_intermediate_symbols(source_symbols) as usize;
        let h = num_hdpc_symbols(source_symbols) as usize;
        let mut matrix = DenseBinaryMatrix::new(width + h + 1, width);
        for col in 0..width {
            matrix.set(col, col, true);
        }
        matrix.set(width + h, 0, true);
        let mut symbols = SymbolSlab::with_zeros(matrix.height(), 1);
        symbols.get_mut(width + h)[0] = 0x5a;

        let owned_decoded =
            try_overdetermined_no_hdpc_prefix_solve_owned(&matrix, symbols.clone(), source_symbols);
        let decoded = try_overdetermined_no_hdpc_prefix_solve(&matrix, &symbols, source_symbols);

        assert!(matches!(
            owned_decoded,
            OverdeterminedNoHdpcPrefixSolve::Failed
        ));
        assert!(decoded.is_none());
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

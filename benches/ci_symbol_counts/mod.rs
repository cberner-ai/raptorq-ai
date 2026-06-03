// Each benchmark binary includes this module separately and uses one of these subsets.
#[allow(dead_code)]
pub const CI_SYMBOL_COUNTS: [usize; 9] = [10, 100, 250, 500, 1000, 2000, 5000, 10000, 20000];

// Decode stays repair-only; higher overhead rows hit slow or unsupported solver paths.
#[allow(dead_code)]
pub const CI_DECODE_SYMBOL_COUNTS: [usize; 5] = [10, 100, 250, 500, 1000];

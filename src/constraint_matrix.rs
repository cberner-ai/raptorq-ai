use crate::base::deg;
use crate::matrix::BinaryMatrix;
use crate::octet::Octet;
use crate::octet_matrix::DenseOctetMatrix;
use crate::rng::rand;
use crate::systematic_constants::{
    calculate_p1, extended_source_block_symbols, num_hdpc_symbols, num_intermediate_symbols,
    num_ldpc_symbols, num_lt_symbols, num_pi_symbols, systematic_index,
};
#[cfg(feature = "std")]
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "std")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "std")]
const HDPC_ROWS_CACHE_CAPACITY: usize = 16;

pub fn generate_constraint_matrix<M: BinaryMatrix>(
    source_block_symbols: u32,
    encoded_isis: &[u32],
) -> (M, DenseOctetMatrix) {
    let matrix = generate_constraint_matrix_no_hdpc::<M>(source_block_symbols, encoded_isis);
    let hdpc = generate_hdpc_rows(source_block_symbols);
    (matrix, hdpc)
}

pub fn generate_constraint_matrix_no_hdpc<M: BinaryMatrix>(
    source_block_symbols: u32,
    encoded_isis: &[u32],
) -> M {
    let k_prime = extended_source_block_symbols(source_block_symbols);
    let s = num_ldpc_symbols(k_prime);
    let l = num_intermediate_symbols(k_prime);
    let mut matrix = M::new(s as usize + encoded_isis.len(), l as usize);

    fill_ldpc_rows(&mut matrix, k_prime);
    fill_encoded_rows(&mut matrix, s as usize, k_prime, encoded_isis);
    matrix.normalize_rows();

    matrix
}

fn fill_ldpc_rows<M: BinaryMatrix>(matrix: &mut M, k_prime: u32) {
    let s = num_ldpc_symbols(k_prime);
    let p = num_pi_symbols(k_prime);
    let w = num_lt_symbols(k_prime);
    let b = w - s;
    let reserve_per_row = (3 * b).div_ceil(s) as usize + 3;
    for row in 0..s {
        matrix.reserve_row_entries(row as usize, reserve_per_row);
    }

    for i in 0..b {
        let a = 1 + i / s;
        let mut row = i % s;
        xor_one(matrix, row as usize, i as usize);
        row = (row + a) % s;
        xor_one(matrix, row as usize, i as usize);
        row = (row + a) % s;
        xor_one(matrix, row as usize, i as usize);
    }

    for i in 0..s {
        let a = i % p;
        let b = (i + 1) % p;
        xor_one(matrix, i as usize, (w + a) as usize);
        xor_one(matrix, i as usize, (w + b) as usize);
        xor_one(matrix, i as usize, (w - s + i) as usize);
    }
}

fn fill_encoded_rows<M: BinaryMatrix>(
    matrix: &mut M,
    row_offset: usize,
    k_prime: u32,
    encoded_isis: &[u32],
) {
    let tuple_params = EncodingTupleParameters::new(k_prime);
    let mut previous_isi_and_y: Option<(u32, u32)> = None;

    for (row, isi) in encoded_isis.iter().enumerate() {
        let y = match previous_isi_and_y {
            Some((previous_isi, previous_y)) if *isi == previous_isi.wrapping_add(1) => {
                previous_y.wrapping_add(tuple_params.a)
            }
            _ => tuple_params.y(*isi),
        };
        previous_isi_and_y = Some((*isi, y));
        fill_encoded_row(
            matrix,
            row_offset + row,
            tuple_params,
            tuple_params.tuple_from_y(*isi, y),
        );
    }
}

#[derive(Clone, Copy)]
struct EncodingTupleParameters {
    lt_symbols: u32,
    pi_symbols: u32,
    p1: u32,
    a: u32,
    b: u32,
}

impl EncodingTupleParameters {
    #[inline]
    fn new(k_prime: u32) -> EncodingTupleParameters {
        let sys_index = systematic_index(k_prime);
        let mut a = 53591 + sys_index * 997;
        if a.is_multiple_of(2) {
            a += 1;
        }

        EncodingTupleParameters {
            lt_symbols: num_lt_symbols(k_prime),
            pi_symbols: num_pi_symbols(k_prime),
            p1: calculate_p1(k_prime),
            a,
            b: 10267 * (sys_index + 1),
        }
    }

    #[inline]
    fn y(self, internal_symbol_id: u32) -> u32 {
        self.b.wrapping_add(internal_symbol_id.wrapping_mul(self.a))
    }

    #[cfg(test)]
    #[inline]
    fn tuple(self, internal_symbol_id: u32) -> (u32, u32, u32, u32, u32, u32) {
        self.tuple_from_y(internal_symbol_id, self.y(internal_symbol_id))
    }

    #[inline]
    fn tuple_from_y(self, internal_symbol_id: u32, y: u32) -> (u32, u32, u32, u32, u32, u32) {
        let d = deg(rand(y, 0u32, 1048576), self.lt_symbols);
        let a = 1 + rand(y, 1u32, self.lt_symbols - 1);
        let b = rand(y, 2u32, self.lt_symbols);
        let d1 = if d < 4 {
            2 + rand(internal_symbol_id, 3u32, 2)
        } else {
            2
        };
        let a1 = 1 + rand(internal_symbol_id, 4u32, self.p1 - 1);
        let b1 = rand(internal_symbol_id, 5u32, self.p1);

        (d, a, b, d1, a1, b1)
    }
}

#[inline]
fn fill_encoded_row<M: BinaryMatrix>(
    matrix: &mut M,
    row: usize,
    params: EncodingTupleParameters,
    source_tuple: (u32, u32, u32, u32, u32, u32),
) {
    let (d, a, mut b, d1, a1, mut b1) = source_tuple;
    matrix.reserve_row_entries(row, (d + d1) as usize);

    xor_one(matrix, row, b as usize);
    for _ in 1..d {
        b = add_mod_once(b, a, params.lt_symbols);
        xor_one(matrix, row, b as usize);
    }

    while b1 >= params.pi_symbols {
        b1 = add_mod_once(b1, a1, params.p1);
    }
    xor_one(matrix, row, (params.lt_symbols + b1) as usize);

    for _ in 1..d1 {
        b1 = add_mod_once(b1, a1, params.p1);
        while b1 >= params.pi_symbols {
            b1 = add_mod_once(b1, a1, params.p1);
        }
        xor_one(matrix, row, (params.lt_symbols + b1) as usize);
    }
}

pub fn enc_indices<F>(
    source_tuple: (u32, u32, u32, u32, u32, u32),
    lt_symbols: u32,
    pi_symbols: u32,
    p1: u32,
    mut visit: F,
) where
    F: FnMut(usize),
{
    let (d, a, mut b, d1, a1, mut b1) = source_tuple;

    visit(b as usize);
    for _ in 1..d {
        b = add_mod_once(b, a, lt_symbols);
        visit(b as usize);
    }

    while b1 >= pi_symbols {
        b1 = add_mod_once(b1, a1, p1);
    }
    visit((lt_symbols + b1) as usize);

    for _ in 1..d1 {
        b1 = add_mod_once(b1, a1, p1);
        while b1 >= pi_symbols {
            b1 = add_mod_once(b1, a1, p1);
        }
        visit((lt_symbols + b1) as usize);
    }
}

#[inline]
fn add_mod_once(value: u32, addend: u32, modulus: u32) -> u32 {
    debug_assert!(value < modulus);
    debug_assert!(addend < modulus);
    let sum = value + addend;
    if sum >= modulus { sum - modulus } else { sum }
}

#[cfg(feature = "std")]
#[derive(Default)]
struct HdpcRowsCache {
    rows: HashMap<u32, DenseOctetMatrix>,
    insertion_order: VecDeque<u32>,
}

#[cfg(feature = "std")]
type HdpcRowsCacheLock = Mutex<HdpcRowsCache>;

#[cfg(feature = "std")]
fn hdpc_rows_cache() -> &'static HdpcRowsCacheLock {
    static CACHE: OnceLock<HdpcRowsCacheLock> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HdpcRowsCache::default()))
}

pub(crate) fn generate_hdpc_rows(source_block_symbols: u32) -> DenseOctetMatrix {
    let k_prime = extended_source_block_symbols(source_block_symbols);

    #[cfg(feature = "std")]
    {
        let cache = hdpc_rows_cache();
        {
            let guard = cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(rows) = guard.rows.get(&k_prime) {
                return rows.clone();
            }
        }

        let generated = generate_hdpc_rows_uncached(k_prime);
        let mut guard = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(rows) = guard.rows.get(&k_prime) {
            return rows.clone();
        }
        if guard.rows.len() >= HDPC_ROWS_CACHE_CAPACITY
            && let Some(evicted_source_block_symbols) = guard.insertion_order.pop_front()
        {
            guard.rows.remove(&evicted_source_block_symbols);
        }
        guard.insertion_order.push_back(k_prime);
        guard.rows.insert(k_prime, generated.clone());
        return generated;
    }

    #[cfg(not(feature = "std"))]
    {
        generate_hdpc_rows_uncached(k_prime)
    }
}

fn generate_hdpc_rows_uncached(k_prime: u32) -> DenseOctetMatrix {
    let s = num_ldpc_symbols(k_prime);
    let h = num_hdpc_symbols(k_prime);
    let l = num_intermediate_symbols(k_prime);
    let gamma_width = k_prime + s;
    let mut rows = DenseOctetMatrix::new(h as usize, l as usize);
    let mut mt = vec![Octet::zero(); h as usize * gamma_width as usize];

    for j in 0..(gamma_width - 1) {
        let row_a = rand(j + 1, 6u32, h);
        let row_b = (row_a + rand(j + 1, 7u32, h - 1) + 1) % h;
        mt[row_a as usize * gamma_width as usize + j as usize] += Octet::one();
        mt[row_b as usize * gamma_width as usize + j as usize] += Octet::one();
    }
    for row in 0..h {
        mt[row as usize * gamma_width as usize + (gamma_width - 1) as usize] =
            Octet::alpha_pow(row as usize);
    }

    for row in 0..h {
        let mut acc = Octet::zero();
        for col in (0..gamma_width).rev() {
            acc = acc * Octet::new(2) + mt[row as usize * gamma_width as usize + col as usize];
            if !acc.is_zero() {
                rows.set(row as usize, col as usize, acc);
            }
        }
    }

    for row in 0..h {
        rows.set(row as usize, (gamma_width + row) as usize, Octet::one());
    }

    rows
}

fn xor_one<M: BinaryMatrix>(matrix: &mut M, row: usize, col: usize) {
    matrix.toggle(row, col);
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    use super::*;
    use crate::base::intermediate_tuple;

    fn octet_row_entries(matrix: &DenseOctetMatrix, row: usize) -> Vec<(usize, u8)> {
        (0..matrix.width())
            .filter_map(|col| {
                let value = matrix.get(row, col).value();
                if value == 0 { None } else { Some((col, value)) }
            })
            .collect()
    }

    #[test]
    fn tuple_generator_matches_rfc_shape_vectors() {
        let cases = [
            (10, 0, (2, 4, 9, 2, 5, 1)),
            (10, 9, (6, 3, 16, 2, 6, 4)),
            (10, 10, (2, 15, 15, 2, 10, 7)),
            (101, 0, (2, 30, 4, 2, 5, 12)),
            (101, 100, (2, 90, 23, 2, 8, 12)),
            (1002, 1001, (23, 23, 516, 2, 44, 0)),
            (56403, 56403, (3, 11594, 17800, 3, 58, 66)),
        ];

        for (source_symbols, internal_symbol_id, expected) in cases {
            let tuple = intermediate_tuple(
                internal_symbol_id,
                num_lt_symbols(source_symbols),
                systematic_index(source_symbols),
                calculate_p1(source_symbols),
            );
            assert_eq!(tuple, expected);
            assert_eq!(
                EncodingTupleParameters::new(source_symbols).tuple(internal_symbol_id),
                expected
            );
        }
    }

    #[test]
    fn hdpc_rows_match_rfc_shape_vector_for_k10() {
        let rows = generate_hdpc_rows(10);
        assert_eq!(rows.height(), 10);
        assert_eq!(rows.width(), 27);

        assert_eq!(
            octet_row_entries(&rows, 0),
            vec![
                (0, 250),
                (1, 243),
                (2, 247),
                (3, 245),
                (4, 244),
                (5, 244),
                (6, 244),
                (7, 122),
                (8, 61),
                (9, 144),
                (10, 72),
                (11, 36),
                (12, 18),
                (13, 9),
                (14, 4),
                (15, 2),
                (16, 1),
                (17, 1),
            ]
        );
        assert_eq!(
            octet_row_entries(&rows, 9),
            vec![
                (0, 235),
                (1, 117),
                (2, 58),
                (3, 29),
                (4, 128),
                (5, 64),
                (6, 32),
                (7, 16),
                (8, 8),
                (9, 4),
                (10, 2),
                (11, 1),
                (12, 142),
                (13, 201),
                (14, 234),
                (15, 117),
                (16, 58),
                (26, 1),
            ]
        );

        for row in 0..rows.height() {
            assert_eq!(rows.get(row, 17 + row), Octet::one());
        }
    }
}

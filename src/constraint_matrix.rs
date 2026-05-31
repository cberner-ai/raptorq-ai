use crate::base::intermediate_tuple;
use crate::matrix::BinaryMatrix;
use crate::octet::Octet;
use crate::octet_matrix::DenseOctetMatrix;
use crate::rng::rand;
use crate::systematic_constants::{
    calculate_p1, extended_source_block_symbols, num_hdpc_symbols, num_intermediate_symbols,
    num_ldpc_symbols, num_lt_symbols, num_pi_symbols, systematic_index,
};

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

    matrix
}

fn fill_ldpc_rows<M: BinaryMatrix>(matrix: &mut M, k_prime: u32) {
    let s = num_ldpc_symbols(k_prime);
    let p = num_pi_symbols(k_prime);
    let w = num_lt_symbols(k_prime);
    let b = w - s;

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
    let lt_symbols = num_lt_symbols(k_prime);
    let pi_symbols = num_pi_symbols(k_prime);
    let sys_index = systematic_index(k_prime);
    let p1 = calculate_p1(k_prime);

    for (row, isi) in encoded_isis.iter().enumerate() {
        let tuple = intermediate_tuple(*isi, lt_symbols, sys_index, p1);
        enc_indices(tuple, lt_symbols, pi_symbols, p1, |col| {
            xor_one(matrix, row_offset + row, col);
        });
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
        b = (b + a) % lt_symbols;
        visit(b as usize);
    }

    while b1 >= pi_symbols {
        b1 = (b1 + a1) % p1;
    }
    visit((lt_symbols + b1) as usize);

    for _ in 1..d1 {
        b1 = (b1 + a1) % p1;
        while b1 >= pi_symbols {
            b1 = (b1 + a1) % p1;
        }
        visit((lt_symbols + b1) as usize);
    }
}

fn generate_hdpc_rows(source_block_symbols: u32) -> DenseOctetMatrix {
    let k_prime = extended_source_block_symbols(source_block_symbols);
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
    let next = matrix.get(row, col) == Octet::zero();
    matrix.set(row, col, next);
}

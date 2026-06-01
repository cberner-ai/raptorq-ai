use crate::octet::Octet;

pub fn add_assign(dest: &mut [u8], src: &[u8]) {
    assert_eq!(dest.len(), src.len());
    if try_add_assign_avx2(dest, src) {
        return;
    }
    add_assign_scalar(dest, src);
}

fn add_assign_scalar(dest: &mut [u8], src: &[u8]) {
    for (d, s) in dest.iter_mut().zip(src.iter()) {
        *d ^= *s;
    }
}

pub fn mulassign_scalar(dest: &mut [u8], scalar: &Octet) {
    if scalar.is_zero() {
        dest.fill(0);
    } else if *scalar != Octet::one() {
        if try_mulassign_scalar_avx2(dest, scalar) {
            return;
        }
        let table = scalar.mul_table();
        mulassign_table(dest, table);
    }
}

pub fn fused_addassign_mul_scalar(dest: &mut [u8], src: &[u8], scalar: &Octet) {
    assert_eq!(dest.len(), src.len());
    if scalar.is_zero() {
        return;
    }
    if *scalar == Octet::one() {
        add_assign(dest, src);
        return;
    }
    if try_fused_addassign_mul_scalar_avx2(dest, src, scalar) {
        return;
    }
    let table = scalar.mul_table();
    fused_addassign_table(dest, src, table);
}

fn mulassign_table(dest: &mut [u8], table: &[u8; 256]) {
    let mut chunks = dest.chunks_exact_mut(8);
    for chunk in chunks.by_ref() {
        chunk[0] = table[chunk[0] as usize];
        chunk[1] = table[chunk[1] as usize];
        chunk[2] = table[chunk[2] as usize];
        chunk[3] = table[chunk[3] as usize];
        chunk[4] = table[chunk[4] as usize];
        chunk[5] = table[chunk[5] as usize];
        chunk[6] = table[chunk[6] as usize];
        chunk[7] = table[chunk[7] as usize];
    }
    for d in chunks.into_remainder() {
        *d = table[*d as usize];
    }
}

fn fused_addassign_table(dest: &mut [u8], src: &[u8], table: &[u8; 256]) {
    let mut dest_chunks = dest.chunks_exact_mut(8);
    let mut src_chunks = src.chunks_exact(8);
    for (dest_chunk, src_chunk) in dest_chunks.by_ref().zip(src_chunks.by_ref()) {
        dest_chunk[0] ^= table[src_chunk[0] as usize];
        dest_chunk[1] ^= table[src_chunk[1] as usize];
        dest_chunk[2] ^= table[src_chunk[2] as usize];
        dest_chunk[3] ^= table[src_chunk[3] as usize];
        dest_chunk[4] ^= table[src_chunk[4] as usize];
        dest_chunk[5] ^= table[src_chunk[5] as usize];
        dest_chunk[6] ^= table[src_chunk[6] as usize];
        dest_chunk[7] ^= table[src_chunk[7] as usize];
    }
    for (d, s) in dest_chunks
        .into_remainder()
        .iter_mut()
        .zip(src_chunks.remainder())
    {
        *d ^= table[*s as usize];
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
fn try_add_assign_avx2(dest: &mut [u8], src: &[u8]) -> bool {
    if dest.len() < 64 || !std::arch::is_x86_feature_detected!("avx2") {
        return false;
    }

    unsafe {
        add_assign_avx2(dest, src);
    }
    true
}

#[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
fn try_add_assign_avx2(_dest: &mut [u8], _src: &[u8]) -> bool {
    false
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
fn try_mulassign_scalar_avx2(dest: &mut [u8], scalar: &Octet) -> bool {
    if dest.len() < 32 || !std::arch::is_x86_feature_detected!("avx2") {
        return false;
    }

    unsafe {
        mulassign_scalar_avx2(dest, scalar.value());
    }
    true
}

#[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
fn try_mulassign_scalar_avx2(_dest: &mut [u8], _scalar: &Octet) -> bool {
    false
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
fn try_fused_addassign_mul_scalar_avx2(dest: &mut [u8], src: &[u8], scalar: &Octet) -> bool {
    if dest.len() < 32 || !std::arch::is_x86_feature_detected!("avx2") {
        return false;
    }

    unsafe {
        fused_addassign_mul_scalar_avx2(dest, src, scalar.value());
    }
    true
}

#[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
fn try_fused_addassign_mul_scalar_avx2(_dest: &mut [u8], _src: &[u8], _scalar: &Octet) -> bool {
    false
}

#[cfg(all(feature = "std", target_arch = "x86"))]
use core::arch::x86::{
    __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_shuffle_epi8,
    _mm256_srli_epi16, _mm256_storeu_si256, _mm256_xor_si256,
};
#[cfg(all(feature = "std", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_shuffle_epi8,
    _mm256_srli_epi16, _mm256_storeu_si256, _mm256_xor_si256,
};

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn mulassign_scalar_avx2(dest: &mut [u8], scalar: u8) {
    let table = Octet::new(scalar).mul_table();
    let low_table = avx2_low_nibble_table(scalar);
    let high_table = avx2_high_nibble_table(scalar);
    let mask = _mm256_set1_epi8(0x0f);
    let mut offset = 0usize;
    let vector_len = dest.len() / 32 * 32;

    while offset < vector_len {
        unsafe {
            let value = _mm256_loadu_si256(dest.as_ptr().add(offset).cast::<__m256i>());
            let product = gf256_mul_vector(value, low_table, high_table, mask);
            _mm256_storeu_si256(dest.as_mut_ptr().add(offset).cast::<__m256i>(), product);
        }
        offset += 32;
    }

    for byte in &mut dest[offset..] {
        *byte = table[*byte as usize];
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn fused_addassign_mul_scalar_avx2(dest: &mut [u8], src: &[u8], scalar: u8) {
    let table = Octet::new(scalar).mul_table();
    let low_table = avx2_low_nibble_table(scalar);
    let high_table = avx2_high_nibble_table(scalar);
    let mask = _mm256_set1_epi8(0x0f);
    let mut offset = 0usize;
    let vector_len = src.len() / 32 * 32;

    while offset < vector_len {
        unsafe {
            let source = _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>());
            let product = gf256_mul_vector(source, low_table, high_table, mask);
            let current = _mm256_loadu_si256(dest.as_ptr().add(offset).cast::<__m256i>());
            let next = _mm256_xor_si256(current, product);
            _mm256_storeu_si256(dest.as_mut_ptr().add(offset).cast::<__m256i>(), next);
        }
        offset += 32;
    }

    for (d, s) in dest[offset..].iter_mut().zip(src[offset..].iter()) {
        *d ^= table[*s as usize];
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn add_assign_avx2(dest: &mut [u8], src: &[u8]) {
    let mut offset = 0usize;
    while offset + 128 <= dest.len() {
        // Pointers stay inside equal-length slices; unaligned loads handle arbitrary alignment.
        unsafe {
            xor_32(dest, src, offset);
            xor_32(dest, src, offset + 32);
            xor_32(dest, src, offset + 64);
            xor_32(dest, src, offset + 96);
        }
        offset += 128;
    }
    while offset + 32 <= dest.len() {
        unsafe {
            xor_32(dest, src, offset);
        }
        offset += 32;
    }
    add_assign_scalar(&mut dest[offset..], &src[offset..]);
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn xor_32(dest: &mut [u8], src: &[u8], offset: usize) {
    unsafe {
        let dest_ptr = dest.as_mut_ptr().add(offset).cast::<__m256i>();
        let src_ptr = src.as_ptr().add(offset).cast::<__m256i>();
        let updated = _mm256_xor_si256(
            _mm256_loadu_si256(dest_ptr.cast_const()),
            _mm256_loadu_si256(src_ptr),
        );
        _mm256_storeu_si256(dest_ptr, updated);
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
static AVX2_LOW_NIBBLE_TABLES: [[u8; 32]; 256] = generate_avx2_low_nibble_tables();

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
static AVX2_HIGH_NIBBLE_TABLES: [[u8; 32]; 256] = generate_avx2_high_nibble_tables();

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
const fn generate_avx2_low_nibble_tables() -> [[u8; 32]; 256] {
    let mut tables = [[0u8; 32]; 256];
    let mut scalar = 0usize;
    while scalar < 256 {
        let mut nibble = 0usize;
        while nibble < 16 {
            let value = crate::octet::gf_mul_slow(scalar as u8, nibble as u8);
            tables[scalar][nibble] = value;
            tables[scalar][nibble + 16] = value;
            nibble += 1;
        }
        scalar += 1;
    }
    tables
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
const fn generate_avx2_high_nibble_tables() -> [[u8; 32]; 256] {
    let mut tables = [[0u8; 32]; 256];
    let mut scalar = 0usize;
    while scalar < 256 {
        let mut nibble = 0usize;
        while nibble < 16 {
            let value = crate::octet::gf_mul_slow(scalar as u8, (nibble << 4) as u8);
            tables[scalar][nibble] = value;
            tables[scalar][nibble + 16] = value;
            nibble += 1;
        }
        scalar += 1;
    }
    tables
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
fn avx2_low_nibble_table(scalar: u8) -> __m256i {
    let low = &AVX2_LOW_NIBBLE_TABLES[scalar as usize];

    unsafe { _mm256_loadu_si256(low.as_ptr().cast::<__m256i>()) }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
fn avx2_high_nibble_table(scalar: u8) -> __m256i {
    let high = &AVX2_HIGH_NIBBLE_TABLES[scalar as usize];

    unsafe { _mm256_loadu_si256(high.as_ptr().cast::<__m256i>()) }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
fn gf256_mul_vector(
    value: __m256i,
    low_table: __m256i,
    high_table: __m256i,
    mask: __m256i,
) -> __m256i {
    let low_nibbles = _mm256_and_si256(value, mask);
    let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(value, 4), mask);
    _mm256_xor_si256(
        _mm256_shuffle_epi8(low_table, low_nibbles),
        _mm256_shuffle_epi8(high_table, high_nibbles),
    )
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn mulassign_scalar_matches_table() {
        for scalar in [0, 1, 2, 29, 255] {
            for len in [0usize, 1, 31, 32, 33, 64, 79] {
                let mut data = patterned_bytes(len);
                let mut expected = data.clone();
                let octet = Octet::new(scalar);
                let table = octet.mul_table();
                if scalar == 0 {
                    expected.fill(0);
                } else if scalar != 1 {
                    for byte in expected.iter_mut() {
                        *byte = table[*byte as usize];
                    }
                }

                mulassign_scalar(&mut data, &octet);
                assert_eq!(data, expected);
            }
        }
    }

    #[test]
    fn fused_addassign_mul_scalar_matches_table() {
        for scalar in [0, 1, 2, 29, 255] {
            for len in [0usize, 1, 31, 32, 33, 64, 79] {
                let mut dest = patterned_bytes(len);
                let src = patterned_bytes(len)
                    .into_iter()
                    .map(|byte| byte.rotate_left(1))
                    .collect::<Vec<_>>();
                let mut expected = dest.clone();
                let octet = Octet::new(scalar);
                let table = octet.mul_table();
                if scalar == 1 {
                    for (d, s) in expected.iter_mut().zip(src.iter()) {
                        *d ^= *s;
                    }
                } else if scalar != 0 {
                    for (d, s) in expected.iter_mut().zip(src.iter()) {
                        *d ^= table[*s as usize];
                    }
                }

                fused_addassign_mul_scalar(&mut dest, &src, &octet);
                assert_eq!(dest, expected);
            }
        }
    }

    #[test]
    fn add_assign_boundary_lengths_match_scalar_xor() {
        for len in [64usize, 95, 127, 128, 129] {
            let original = patterned_bytes(len);
            let src = patterned_bytes(len)
                .into_iter()
                .map(|byte| byte.rotate_left(3) ^ 0xa5)
                .collect::<Vec<_>>();
            let expected = scalar_xor(original.clone(), &src);

            let mut direct = original.clone();
            add_assign(&mut direct, &src);
            assert_eq!(direct, expected, "add_assign failed for length {len}");

            let mut symbol = Symbol::new(original.clone());
            let other = Symbol::new(src.clone());
            symbol += &other;
            assert_eq!(
                symbol.into_bytes(),
                expected,
                "Symbol += failed for length {len}"
            );

            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            if std::arch::is_x86_feature_detected!("avx2") {
                let mut avx2_direct = original;
                unsafe {
                    add_assign_avx2(&mut avx2_direct, &src);
                }
                assert_eq!(
                    avx2_direct, expected,
                    "direct AVX2 add_assign failed for length {len}"
                );
            }
        }
    }

    fn scalar_xor(mut dest: Vec<u8>, src: &[u8]) -> Vec<u8> {
        for (dest_byte, src_byte) in dest.iter_mut().zip(src.iter()) {
            *dest_byte ^= *src_byte;
        }
        dest
    }

    fn patterned_bytes(len: usize) -> Vec<u8> {
        (0..len)
            .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
            .collect()
    }
}

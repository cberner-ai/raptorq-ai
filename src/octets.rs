use crate::octet::Octet;

pub fn add_assign(dest: &mut [u8], src: &[u8]) {
    assert_eq!(dest.len(), src.len());
    for (d, s) in dest.iter_mut().zip(src.iter()) {
        *d ^= *s;
    }
}

pub fn mulassign_scalar(dest: &mut [u8], scalar: &Octet) {
    if scalar.is_zero() {
        dest.fill(0);
    } else if *scalar != Octet::one() {
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

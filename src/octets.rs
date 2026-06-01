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
        for d in dest.iter_mut() {
            *d = table[*d as usize];
        }
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
    for (d, s) in dest.iter_mut().zip(src.iter()) {
        *d ^= table[*s as usize];
    }
}

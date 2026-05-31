pub fn int_div_ceil(numerator: u64, denominator: u64) -> u32 {
    assert_ne!(denominator, 0);
    let quotient = numerator.div_ceil(denominator);
    u32::try_from(quotient).expect("integer division result exceeds u32")
}

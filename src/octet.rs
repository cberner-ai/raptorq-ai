use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Octet {
    value: u8,
}

impl Octet {
    pub const fn new(value: u8) -> Octet {
        Octet { value }
    }

    pub const fn zero() -> Octet {
        Octet { value: 0 }
    }

    pub const fn one() -> Octet {
        Octet { value: 1 }
    }

    pub const fn value(self) -> u8 {
        self.value
    }

    pub fn alpha_pow(power: usize) -> Octet {
        let mut result = Octet::one();
        let alpha = Octet::new(2);
        for _ in 0..(power % 255) {
            result *= alpha;
        }
        result
    }

    pub fn inverse(self) -> Octet {
        assert_ne!(self.value, 0);
        self.pow(254)
    }

    pub fn pow(self, mut power: usize) -> Octet {
        let mut base = self;
        let mut result = Octet::one();
        while power > 0 {
            if power & 1 == 1 {
                result *= base;
            }
            base *= base;
            power >>= 1;
        }
        result
    }

    pub const fn is_zero(self) -> bool {
        self.value == 0
    }

    pub(crate) fn mul_table(self) -> &'static [u8; 256] {
        &GF_MUL_TABLE[self.value as usize]
    }
}

static GF_MUL_TABLE: [[u8; 256]; 256] = generate_gf_mul_table();

const fn generate_gf_mul_table() -> [[u8; 256]; 256] {
    let mut table = [[0u8; 256]; 256];
    let mut a = 0usize;
    while a < 256 {
        let mut b = 0usize;
        while b < 256 {
            table[a][b] = gf_mul_slow(a as u8, b as u8);
            b += 1;
        }
        a += 1;
    }
    table
}

pub(crate) const fn gf_mul_slow(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0u8;
    while b != 0 {
        if b & 1 != 0 {
            product ^= a;
        }
        b >>= 1;
        let carry = a & 0x80;
        a <<= 1;
        if carry != 0 {
            a ^= 0x1d;
        }
    }
    product
}

#[inline]
fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        0
    } else if a == 1 {
        b
    } else if b == 1 {
        a
    } else {
        GF_MUL_TABLE[a as usize][b as usize]
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for Octet {
    type Output = Octet;

    fn add(self, rhs: Octet) -> Octet {
        Octet::new(self.value ^ rhs.value)
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl AddAssign for Octet {
    fn add_assign(&mut self, rhs: Octet) {
        self.value ^= rhs.value;
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Sub for Octet {
    type Output = Octet;

    fn sub(self, rhs: Octet) -> Octet {
        self + rhs
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl SubAssign for Octet {
    fn sub_assign(&mut self, rhs: Octet) {
        *self += rhs;
    }
}

impl Mul for Octet {
    type Output = Octet;

    fn mul(self, rhs: Octet) -> Octet {
        Octet::new(gf_mul(self.value, rhs.value))
    }
}

impl MulAssign for Octet {
    fn mul_assign(&mut self, rhs: Octet) {
        *self = *self * rhs;
    }
}

impl Div for Octet {
    type Output = Octet;

    fn div(self, rhs: Octet) -> Octet {
        assert_ne!(rhs.value, 0);
        if self.value == 0 {
            Octet::zero()
        } else {
            self * rhs.inverse()
        }
    }
}

impl DivAssign for Octet {
    fn div_assign(&mut self, rhs: Octet) {
        *self = *self / rhs;
    }
}

impl From<u8> for Octet {
    fn from(value: u8) -> Octet {
        Octet::new(value)
    }
}

impl From<Octet> for u8 {
    fn from(value: Octet) -> u8 {
        value.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gf_mul_table_matches_shift_multiply() {
        for a in u8::MIN..=u8::MAX {
            for b in u8::MIN..=u8::MAX {
                assert_eq!(gf_mul(a, b), gf_mul_slow(a, b));
            }
        }
    }

    #[test]
    fn nonzero_octets_have_multiplicative_inverses() {
        for value in 1..=u8::MAX {
            let octet = Octet::new(value);
            assert_eq!(octet * octet.inverse(), Octet::one());
        }
    }
}

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
}

#[inline]
fn gf_mul(mut a: u8, mut b: u8) -> u8 {
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

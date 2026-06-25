#[cfg(feature = "std")]
use std::sync::Arc;
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::octet::Octet;
#[cfg(feature = "serde_support")]
use serde::de;
#[cfg(feature = "serde_support")]
use serde::ser::SerializeStruct;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "std")]
#[derive(Debug)]
enum DenseOctetMatrixData {
    Owned(Vec<Octet>),
    Shared(Arc<[Octet]>),
}

#[cfg(not(feature = "std"))]
type DenseOctetMatrixData = Vec<Octet>;

#[cfg(feature = "std")]
const SHARED_DENSE_OCTET_MATRIX_MIN_LEN: usize = 100_000;

#[derive(Debug)]
pub struct DenseOctetMatrix {
    height: usize,
    width: usize,
    data: DenseOctetMatrixData,
}

impl PartialEq for DenseOctetMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.height == other.height
            && self.width == other.width
            && self.as_slice() == other.as_slice()
    }
}

impl Eq for DenseOctetMatrix {}

impl Clone for DenseOctetMatrix {
    fn clone(&self) -> DenseOctetMatrix {
        #[cfg(feature = "std")]
        {
            let data = match &self.data {
                DenseOctetMatrixData::Owned(data)
                    if data.len() >= SHARED_DENSE_OCTET_MATRIX_MIN_LEN =>
                {
                    DenseOctetMatrixData::Shared(Arc::from(data.as_slice()))
                }
                DenseOctetMatrixData::Owned(data) => DenseOctetMatrixData::Owned(data.clone()),
                DenseOctetMatrixData::Shared(data)
                    if data.len() >= SHARED_DENSE_OCTET_MATRIX_MIN_LEN =>
                {
                    DenseOctetMatrixData::Shared(Arc::clone(data))
                }
                DenseOctetMatrixData::Shared(data) => {
                    DenseOctetMatrixData::Owned(data.as_ref().to_vec())
                }
            };
            DenseOctetMatrix {
                height: self.height,
                width: self.width,
                data,
            }
        }

        #[cfg(not(feature = "std"))]
        {
            DenseOctetMatrix {
                height: self.height,
                width: self.width,
                data: self.data.clone(),
            }
        }
    }
}

impl DenseOctetMatrix {
    pub fn new(height: usize, width: usize) -> DenseOctetMatrix {
        DenseOctetMatrix {
            height,
            width,
            data: dense_octet_matrix_data_from_vec(vec![Octet::zero(); height * width]),
        }
    }

    #[cfg(feature = "serde_support")]
    fn from_vec(height: usize, width: usize, data: Vec<Octet>) -> DenseOctetMatrix {
        assert_eq!(data.len(), height * width);
        DenseOctetMatrix {
            height,
            width,
            data: dense_octet_matrix_data_from_vec(data),
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn into_shared_if_large(self) -> DenseOctetMatrix {
        #[cfg(feature = "std")]
        {
            let DenseOctetMatrix {
                height,
                width,
                data,
            } = self;
            let data = match data {
                DenseOctetMatrixData::Owned(data)
                    if data.len() >= SHARED_DENSE_OCTET_MATRIX_MIN_LEN =>
                {
                    DenseOctetMatrixData::Shared(Arc::from(data.into_boxed_slice()))
                }
                data => data,
            };
            DenseOctetMatrix {
                height,
                width,
                data,
            }
        }

        #[cfg(not(feature = "std"))]
        {
            self
        }
    }

    pub(crate) fn row(&self, row: usize) -> &[Octet] {
        assert!(row < self.height);
        let start = row * self.width;
        &self.as_slice()[start..start + self.width]
    }

    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [Octet] {
        assert!(row < self.height);
        let start = row * self.width;
        let end = start + self.width;
        let data = self.data_mut();
        &mut data[start..end]
    }

    pub(crate) fn as_slice(&self) -> &[Octet] {
        #[cfg(feature = "std")]
        {
            match &self.data {
                DenseOctetMatrixData::Owned(data) => data.as_slice(),
                DenseOctetMatrixData::Shared(data) => data.as_ref(),
            }
        }

        #[cfg(not(feature = "std"))]
        {
            self.data.as_slice()
        }
    }

    fn data_mut(&mut self) -> &mut [Octet] {
        #[cfg(feature = "std")]
        {
            match &mut self.data {
                DenseOctetMatrixData::Owned(data) => data.as_mut_slice(),
                DenseOctetMatrixData::Shared(data) => Arc::make_mut(data),
            }
        }

        #[cfg(not(feature = "std"))]
        {
            self.data.as_mut_slice()
        }
    }

    #[allow(dead_code)]
    pub fn get(&self, row: usize, col: usize) -> Octet {
        assert!(row < self.height);
        assert!(col < self.width);
        self.as_slice()[row * self.width + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: Octet) {
        assert!(row < self.height);
        assert!(col < self.width);
        let index = row * self.width + col;
        self.data_mut()[index] = value;
    }
}

#[cfg(feature = "std")]
fn dense_octet_matrix_data_from_vec(data: Vec<Octet>) -> DenseOctetMatrixData {
    DenseOctetMatrixData::Owned(data)
}

#[cfg(not(feature = "std"))]
fn dense_octet_matrix_data_from_vec(data: Vec<Octet>) -> DenseOctetMatrixData {
    data
}

#[cfg(feature = "serde_support")]
impl Serialize for DenseOctetMatrix {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("DenseOctetMatrix", 3)?;
        state.serialize_field("height", &self.height)?;
        state.serialize_field("width", &self.width)?;
        state.serialize_field("data", self.as_slice())?;
        state.end()
    }
}

#[cfg(feature = "serde_support")]
impl<'de> Deserialize<'de> for DenseOctetMatrix {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DenseOctetMatrixFields {
            height: usize,
            width: usize,
            data: Vec<Octet>,
        }

        let fields = DenseOctetMatrixFields::deserialize(deserializer)?;
        dense_octet_matrix_from_deserialized_parts(fields.height, fields.width, fields.data)
            .map_err(de::Error::custom)
    }
}

#[cfg(feature = "serde_support")]
fn dense_octet_matrix_from_deserialized_parts(
    height: usize,
    width: usize,
    data: Vec<Octet>,
) -> Result<DenseOctetMatrix, &'static str> {
    let Some(expected_len) = height.checked_mul(width) else {
        return Err("DenseOctetMatrix data length does not match dimensions");
    };
    if data.len() != expected_len {
        return Err("DenseOctetMatrix data length does not match dimensions");
    }
    Ok(DenseOctetMatrix::from_vec(height, width, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_accessors_expose_contiguous_storage() {
        let mut matrix = DenseOctetMatrix::new(2, 3);
        matrix
            .row_mut(1)
            .copy_from_slice(&[Octet::new(4), Octet::new(5), Octet::new(6)]);

        assert_eq!(
            matrix.row(0),
            &[Octet::zero(), Octet::zero(), Octet::zero()]
        );
        assert_eq!(
            matrix.as_slice(),
            &[
                Octet::zero(),
                Octet::zero(),
                Octet::zero(),
                Octet::new(4),
                Octet::new(5),
                Octet::new(6)
            ]
        );
        assert_eq!(matrix.get(1, 1), Octet::new(5));
    }

    #[test]
    fn cloned_matrix_mutation_does_not_change_original() {
        let mut original = DenseOctetMatrix::new(2, 3);
        original.set(0, 1, Octet::new(7));

        let mut cloned = original.clone();
        cloned.set(0, 1, Octet::new(9));
        cloned
            .row_mut(1)
            .copy_from_slice(&[Octet::new(1), Octet::new(2), Octet::new(3)]);

        assert_eq!(original.get(0, 1), Octet::new(7));
        assert_eq!(
            original.row(1),
            &[Octet::zero(), Octet::zero(), Octet::zero()]
        );
        assert_eq!(cloned.get(0, 1), Octet::new(9));
        assert_eq!(
            cloned.row(1),
            &[Octet::new(1), Octet::new(2), Octet::new(3)]
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn large_cloned_matrix_mutation_does_not_change_original() {
        let width = SHARED_DENSE_OCTET_MATRIX_MIN_LEN + 1;
        let mut original = DenseOctetMatrix::new(1, width);
        original.set(0, width - 1, Octet::new(7));

        let mut cloned = original.clone();
        cloned.set(0, width - 1, Octet::new(9));

        assert_eq!(original.get(0, width - 1), Octet::new(7));
        assert_eq!(cloned.get(0, width - 1), Octet::new(9));
    }

    #[cfg(feature = "std")]
    #[test]
    fn large_cloned_matrix_compares_by_contents_not_storage() {
        let width = SHARED_DENSE_OCTET_MATRIX_MIN_LEN + 1;
        let mut original = DenseOctetMatrix::new(1, width);
        original.set(0, width - 1, Octet::new(7));

        let mut cloned = original.clone();
        assert_eq!(original, cloned);

        cloned.set(0, width - 1, Octet::new(9));
        assert_ne!(original, cloned);
    }

    #[cfg(feature = "std")]
    #[test]
    fn large_matrix_can_be_shared_before_repeated_clones() {
        let width = SHARED_DENSE_OCTET_MATRIX_MIN_LEN + 1;
        let mut original = DenseOctetMatrix::new(1, width);
        original.set(0, width - 1, Octet::new(7));

        let shared = original.into_shared_if_large();
        let mut cloned = shared.clone();
        cloned.set(0, width - 1, Octet::new(9));

        assert_eq!(shared.get(0, width - 1), Octet::new(7));
        assert_eq!(cloned.get(0, width - 1), Octet::new(9));
    }

    #[cfg(feature = "serde_support")]
    #[test]
    fn deserialized_matrix_rejects_overflowing_dimensions() {
        let result = dense_octet_matrix_from_deserialized_parts(usize::MAX, 2, Vec::<Octet>::new());

        assert_eq!(
            result.unwrap_err(),
            "DenseOctetMatrix data length does not match dimensions"
        );
    }
}

use crate::octet::Octet;

#[derive(Clone, Copy)]
pub(crate) struct AddAssignFastPath {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    use_avx2: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct FusedAddAssignMulScalarFastPath {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    use_avx2: bool,
}

impl AddAssignFastPath {
    pub(crate) fn new(symbol_len: usize) -> AddAssignFastPath {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            return AddAssignFastPath {
                use_avx2: symbol_len >= 64 && std::arch::is_x86_feature_detected!("avx2"),
            };
        }

        #[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
        {
            let _ = symbol_len;
            AddAssignFastPath {}
        }
    }

    pub(crate) fn apply(self, dest: &mut [u8], src: &[u8]) {
        assert_eq!(dest.len(), src.len());
        self.apply_same_len(dest, src);
    }

    pub(crate) fn apply_same_len(self, dest: &mut [u8], src: &[u8]) {
        debug_assert_eq!(dest.len(), src.len());
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if self.use_avx2 {
            unsafe {
                add_assign_avx2(dest, src);
            }
            return;
        }

        add_assign_scalar(dest, src);
    }

    /// Applies one source slice into four disjoint destination slices.
    ///
    /// # Safety
    ///
    /// `src` and every destination pointer must be valid for `len` bytes. Destinations must be
    /// writable and must not overlap each other or `src`.
    pub(crate) unsafe fn apply_same_len_raw_4(
        self,
        dests: [*mut u8; 4],
        src: *const u8,
        len: usize,
    ) {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if self.use_avx2 {
            unsafe {
                add_assign_4_avx2(dests, src, len);
            }
            return;
        }

        for dest in dests {
            let dest = unsafe { core::slice::from_raw_parts_mut(dest, len) };
            let src = unsafe { core::slice::from_raw_parts(src, len) };
            add_assign_scalar(dest, src);
        }
    }

    /// Applies one source slice into eight disjoint destination slices.
    ///
    /// # Safety
    ///
    /// `src` and every destination pointer must be valid for `len` bytes. Destinations must be
    /// writable and must not overlap each other or `src`.
    pub(crate) unsafe fn apply_same_len_raw_8(
        self,
        dests: [*mut u8; 8],
        src: *const u8,
        len: usize,
    ) {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if self.use_avx2 {
            unsafe {
                add_assign_8_avx2(dests, src, len);
            }
            return;
        }

        for dest in dests {
            let dest = unsafe { core::slice::from_raw_parts_mut(dest, len) };
            let src = unsafe { core::slice::from_raw_parts(src, len) };
            add_assign_scalar(dest, src);
        }
    }

    /// Applies four source slices into one destination slice.
    ///
    /// # Safety
    ///
    /// `dest` and every source pointer must be valid for `len` bytes. The destination must be
    /// writable and must not overlap any source.
    pub(crate) unsafe fn apply_sources_same_len_raw_4(
        self,
        dest: *mut u8,
        srcs: [*const u8; 4],
        len: usize,
    ) {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if self.use_avx2 {
            unsafe {
                add_assign_sources_4_avx2(dest, srcs, len);
            }
            return;
        }

        let dest = unsafe { core::slice::from_raw_parts_mut(dest, len) };
        for src in srcs {
            let src = unsafe { core::slice::from_raw_parts(src, len) };
            add_assign_scalar(dest, src);
        }
    }

    /// Applies eight source slices into one destination slice.
    ///
    /// # Safety
    ///
    /// `dest` and every source pointer must be valid for `len` bytes. The destination must be
    /// writable and must not overlap any source.
    pub(crate) unsafe fn apply_sources_same_len_raw_8(
        self,
        dest: *mut u8,
        srcs: [*const u8; 8],
        len: usize,
    ) {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if self.use_avx2 {
            unsafe {
                add_assign_sources_8_avx2(dest, srcs, len);
            }
            return;
        }

        let dest = unsafe { core::slice::from_raw_parts_mut(dest, len) };
        for src in srcs {
            let src = unsafe { core::slice::from_raw_parts(src, len) };
            add_assign_scalar(dest, src);
        }
    }

    /// Applies sixteen source slices into one destination slice.
    ///
    /// # Safety
    ///
    /// `dest` and every source pointer must be valid for `len` bytes. The destination must be
    /// writable and must not overlap any source.
    pub(crate) unsafe fn apply_sources_same_len_raw_16(
        self,
        dest: *mut u8,
        srcs: [*const u8; 16],
        len: usize,
    ) {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if self.use_avx2 {
            unsafe {
                add_assign_sources_16_avx2(dest, srcs, len);
            }
            return;
        }

        let dest = unsafe { core::slice::from_raw_parts_mut(dest, len) };
        for src in srcs {
            let src = unsafe { core::slice::from_raw_parts(src, len) };
            add_assign_scalar(dest, src);
        }
    }
}

impl FusedAddAssignMulScalarFastPath {
    pub(crate) fn new(symbol_len: usize) -> FusedAddAssignMulScalarFastPath {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            return FusedAddAssignMulScalarFastPath {
                use_avx2: symbol_len >= 32 && std::arch::is_x86_feature_detected!("avx2"),
            };
        }

        #[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
        {
            let _ = symbol_len;
            FusedAddAssignMulScalarFastPath {}
        }
    }

    pub(crate) fn apply(self, dest: &mut [u8], src: &[u8], scalar: &Octet) {
        assert_eq!(dest.len(), src.len());
        if scalar.is_zero() {
            return;
        }
        self.apply_nonzero(dest, src, scalar);
    }

    pub(crate) fn apply_nonzero(self, dest: &mut [u8], src: &[u8], scalar: &Octet) {
        debug_assert!(!scalar.is_zero());
        assert_eq!(dest.len(), src.len());
        if *scalar == Octet::one() {
            AddAssignFastPath {
                #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
                use_avx2: self.use_avx2,
            }
            .apply(dest, src);
            return;
        }

        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if self.use_avx2 {
            unsafe {
                fused_addassign_mul_scalar_avx2(dest, src, scalar.value());
            }
            return;
        }

        let table = scalar.mul_table();
        fused_addassign_table(dest, src, table);
    }

    /// Applies one source slice into a strided set of destination slices, using one coefficient
    /// per destination.
    ///
    /// # Safety
    ///
    /// `dests` must point to `coefficients.len()` writable slices of `len` bytes separated by
    /// `dest_stride`. Each destination slice must be disjoint from every other destination and
    /// from `src`, and `src` must be valid for `len` bytes.
    pub(crate) unsafe fn apply_column_coefficients(
        self,
        dests: *mut u8,
        dest_stride: usize,
        src: *const u8,
        coefficients: &[u8],
        len: usize,
    ) {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if self.use_avx2 {
            unsafe {
                fused_addassign_mul_column_coefficients_avx2(
                    dests,
                    dest_stride,
                    src,
                    coefficients,
                    len,
                );
            }
            return;
        }

        unsafe {
            fused_addassign_mul_column_coefficients_scalar(
                dests,
                dest_stride,
                src,
                coefficients,
                len,
            );
        }
    }
}

pub fn add_assign(dest: &mut [u8], src: &[u8]) {
    AddAssignFastPath::new(dest.len()).apply(dest, src);
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
        if bytes_are_zero(dest) {
            return;
        }
        if try_mulassign_scalar_avx2(dest, scalar) {
            return;
        }
        let table = scalar.mul_table();
        mulassign_table(dest, table);
    }
}

pub fn fused_addassign_mul_scalar(dest: &mut [u8], src: &[u8], scalar: &Octet) {
    FusedAddAssignMulScalarFastPath::new(dest.len()).apply(dest, src, scalar);
}

pub(crate) fn bytes_are_zero(bytes: &[u8]) -> bool {
    let prefix_len = bytes.len().min(16);
    if !bytes_are_zero_scalar(&bytes[..prefix_len]) {
        return false;
    }

    let rest = &bytes[prefix_len..];
    if let Some(result) = try_bytes_are_zero_avx2(rest) {
        return result;
    }
    bytes_are_zero_scalar(rest)
}

fn bytes_are_zero_scalar(bytes: &[u8]) -> bool {
    bytes.iter().all(|&byte| byte == 0)
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
fn try_bytes_are_zero_avx2(bytes: &[u8]) -> Option<bool> {
    if bytes.len() < 64 || !std::arch::is_x86_feature_detected!("avx2") {
        return None;
    }

    Some(unsafe { bytes_are_zero_avx2(bytes) })
}

#[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
fn try_bytes_are_zero_avx2(_bytes: &[u8]) -> Option<bool> {
    None
}

#[cfg(all(feature = "std", target_arch = "x86"))]
use core::arch::x86::{
    __m256i, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
    _mm256_or_si256, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_srli_epi16, _mm256_storeu_si256, _mm256_xor_si256,
};
#[cfg(all(feature = "std", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
    _mm256_or_si256, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_srli_epi16, _mm256_storeu_si256, _mm256_xor_si256,
};

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn mulassign_scalar_avx2(dest: &mut [u8], scalar: u8) {
    let low_table = avx2_low_nibble_table(scalar);
    let high_table = avx2_high_nibble_table(scalar);
    let mask = _mm256_set1_epi8(0x0f);
    let mut offset = 0usize;
    let vector_len = dest.len() / 32 * 32;

    while offset + 128 <= vector_len {
        unsafe {
            mulassign_32(dest, offset, low_table, high_table, mask);
            mulassign_32(dest, offset + 32, low_table, high_table, mask);
            mulassign_32(dest, offset + 64, low_table, high_table, mask);
            mulassign_32(dest, offset + 96, low_table, high_table, mask);
        }
        offset += 128;
    }
    while offset < vector_len {
        unsafe {
            mulassign_32(dest, offset, low_table, high_table, mask);
        }
        offset += 32;
    }

    if offset < dest.len() {
        let table = Octet::new(scalar).mul_table();
        for byte in &mut dest[offset..] {
            *byte = table[*byte as usize];
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn fused_addassign_mul_scalar_avx2(dest: &mut [u8], src: &[u8], scalar: u8) {
    let low_table = avx2_low_nibble_table(scalar);
    let high_table = avx2_high_nibble_table(scalar);
    let mask = _mm256_set1_epi8(0x0f);
    let mut offset = 0usize;
    let vector_len = src.len() / 32 * 32;

    while offset + 128 <= vector_len {
        unsafe {
            fused_addassign_mul_32(dest, src, offset, low_table, high_table, mask);
            fused_addassign_mul_32(dest, src, offset + 32, low_table, high_table, mask);
            fused_addassign_mul_32(dest, src, offset + 64, low_table, high_table, mask);
            fused_addassign_mul_32(dest, src, offset + 96, low_table, high_table, mask);
        }
        offset += 128;
    }
    while offset < vector_len {
        unsafe {
            fused_addassign_mul_32(dest, src, offset, low_table, high_table, mask);
        }
        offset += 32;
    }

    if offset < src.len() {
        let table = Octet::new(scalar).mul_table();
        for (d, s) in dest[offset..].iter_mut().zip(src[offset..].iter()) {
            *d ^= table[*s as usize];
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
const COLUMN_COEFFICIENT_AVX2_STACK_MAX: usize = 64;

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn fused_addassign_mul_column_coefficients_avx2(
    dests: *mut u8,
    dest_stride: usize,
    src: *const u8,
    coefficients: &[u8],
    len: usize,
) {
    if coefficients.len() > COLUMN_COEFFICIENT_AVX2_STACK_MAX {
        unsafe {
            fused_addassign_mul_column_coefficients_scalar(
                dests,
                dest_stride,
                src,
                coefficients,
                len,
            );
        }
        return;
    }

    let zero = _mm256_setzero_si256();
    let mut rows = [0usize; COLUMN_COEFFICIENT_AVX2_STACK_MAX];
    let mut low_tables = [zero; COLUMN_COEFFICIENT_AVX2_STACK_MAX];
    let mut high_tables = [zero; COLUMN_COEFFICIENT_AVX2_STACK_MAX];
    let mut table_count = 0usize;
    for (row, &coefficient) in coefficients.iter().enumerate() {
        if coefficient == 0 {
            continue;
        }
        rows[table_count] = row;
        low_tables[table_count] = avx2_low_nibble_table(coefficient);
        high_tables[table_count] = avx2_high_nibble_table(coefficient);
        table_count += 1;
    }

    let mask = _mm256_set1_epi8(0x0f);
    let tables = ColumnCoefficientAvx2Tables {
        rows: &rows[..table_count],
        low_tables: &low_tables[..table_count],
        high_tables: &high_tables[..table_count],
        mask,
    };
    let mut offset = 0usize;

    while offset + 128 <= len {
        unsafe {
            fused_addassign_mul_column_coefficients_32(dests, dest_stride, src, &tables, offset);
            fused_addassign_mul_column_coefficients_32(
                dests,
                dest_stride,
                src,
                &tables,
                offset + 32,
            );
            fused_addassign_mul_column_coefficients_32(
                dests,
                dest_stride,
                src,
                &tables,
                offset + 64,
            );
            fused_addassign_mul_column_coefficients_32(
                dests,
                dest_stride,
                src,
                &tables,
                offset + 96,
            );
        }
        offset += 128;
    }
    while offset + 32 <= len {
        unsafe {
            fused_addassign_mul_column_coefficients_32(dests, dest_stride, src, &tables, offset);
        }
        offset += 32;
    }

    if offset < len {
        unsafe {
            fused_addassign_mul_column_coefficients_tail(
                dests,
                dest_stride,
                src.add(offset),
                coefficients,
                len - offset,
                offset,
            );
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
struct ColumnCoefficientAvx2Tables<'a> {
    rows: &'a [usize],
    low_tables: &'a [__m256i],
    high_tables: &'a [__m256i],
    mask: __m256i,
}

unsafe fn fused_addassign_mul_column_coefficients_scalar(
    dests: *mut u8,
    dest_stride: usize,
    src: *const u8,
    coefficients: &[u8],
    len: usize,
) {
    unsafe {
        fused_addassign_mul_column_coefficients_tail(dests, dest_stride, src, coefficients, len, 0);
    }
}

unsafe fn fused_addassign_mul_column_coefficients_tail(
    dests: *mut u8,
    dest_stride: usize,
    src: *const u8,
    coefficients: &[u8],
    len: usize,
    dest_offset: usize,
) {
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for (row, &coefficient) in coefficients.iter().enumerate() {
        if coefficient == 0 {
            continue;
        }
        let dest = unsafe {
            core::slice::from_raw_parts_mut(dests.add(row * dest_stride + dest_offset), len)
        };
        if coefficient == 1 {
            add_assign_scalar(dest, src);
        } else {
            let table = Octet::new(coefficient).mul_table();
            fused_addassign_table(dest, src, table);
        }
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
unsafe fn add_assign_4_avx2(dests: [*mut u8; 4], src: *const u8, len: usize) {
    let mut offset = 0usize;
    while offset + 128 <= len {
        unsafe {
            xor_4_32(dests, src, offset);
            xor_4_32(dests, src, offset + 32);
            xor_4_32(dests, src, offset + 64);
            xor_4_32(dests, src, offset + 96);
        }
        offset += 128;
    }
    while offset + 32 <= len {
        unsafe {
            xor_4_32(dests, src, offset);
        }
        offset += 32;
    }

    if offset < len {
        let src = unsafe { core::slice::from_raw_parts(src.add(offset), len - offset) };
        for dest in dests {
            let dest = unsafe { core::slice::from_raw_parts_mut(dest.add(offset), len - offset) };
            add_assign_scalar(dest, src);
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn add_assign_8_avx2(dests: [*mut u8; 8], src: *const u8, len: usize) {
    let mut offset = 0usize;
    while offset + 128 <= len {
        unsafe {
            xor_8_32(dests, src, offset);
            xor_8_32(dests, src, offset + 32);
            xor_8_32(dests, src, offset + 64);
            xor_8_32(dests, src, offset + 96);
        }
        offset += 128;
    }
    while offset + 32 <= len {
        unsafe {
            xor_8_32(dests, src, offset);
        }
        offset += 32;
    }

    if offset < len {
        let src = unsafe { core::slice::from_raw_parts(src.add(offset), len - offset) };
        for dest in dests {
            let dest = unsafe { core::slice::from_raw_parts_mut(dest.add(offset), len - offset) };
            add_assign_scalar(dest, src);
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn add_assign_sources_4_avx2(dest: *mut u8, srcs: [*const u8; 4], len: usize) {
    let mut offset = 0usize;
    while offset + 128 <= len {
        unsafe {
            xor_sources_4_32(dest, srcs, offset);
            xor_sources_4_32(dest, srcs, offset + 32);
            xor_sources_4_32(dest, srcs, offset + 64);
            xor_sources_4_32(dest, srcs, offset + 96);
        }
        offset += 128;
    }
    while offset + 32 <= len {
        unsafe {
            xor_sources_4_32(dest, srcs, offset);
        }
        offset += 32;
    }

    if offset < len {
        let dest = unsafe { core::slice::from_raw_parts_mut(dest.add(offset), len - offset) };
        for src in srcs {
            let src = unsafe { core::slice::from_raw_parts(src.add(offset), len - offset) };
            add_assign_scalar(dest, src);
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn add_assign_sources_8_avx2(dest: *mut u8, srcs: [*const u8; 8], len: usize) {
    let mut offset = 0usize;
    while offset + 128 <= len {
        unsafe {
            xor_sources_8_32(dest, srcs, offset);
            xor_sources_8_32(dest, srcs, offset + 32);
            xor_sources_8_32(dest, srcs, offset + 64);
            xor_sources_8_32(dest, srcs, offset + 96);
        }
        offset += 128;
    }
    while offset + 32 <= len {
        unsafe {
            xor_sources_8_32(dest, srcs, offset);
        }
        offset += 32;
    }

    if offset < len {
        let dest = unsafe { core::slice::from_raw_parts_mut(dest.add(offset), len - offset) };
        for src in srcs {
            let src = unsafe { core::slice::from_raw_parts(src.add(offset), len - offset) };
            add_assign_scalar(dest, src);
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn add_assign_sources_16_avx2(dest: *mut u8, srcs: [*const u8; 16], len: usize) {
    let mut offset = 0usize;
    while offset + 128 <= len {
        unsafe {
            xor_sources_16_32(dest, srcs, offset);
            xor_sources_16_32(dest, srcs, offset + 32);
            xor_sources_16_32(dest, srcs, offset + 64);
            xor_sources_16_32(dest, srcs, offset + 96);
        }
        offset += 128;
    }
    while offset + 32 <= len {
        unsafe {
            xor_sources_16_32(dest, srcs, offset);
        }
        offset += 32;
    }

    if offset < len {
        let dest = unsafe { core::slice::from_raw_parts_mut(dest.add(offset), len - offset) };
        for src in srcs {
            let src = unsafe { core::slice::from_raw_parts(src.add(offset), len - offset) };
            add_assign_scalar(dest, src);
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn bytes_are_zero_avx2(bytes: &[u8]) -> bool {
    let zero = _mm256_setzero_si256();
    let mut offset = 0usize;

    while offset + 128 <= bytes.len() {
        let combined = unsafe {
            let first = _mm256_or_si256(
                _mm256_loadu_si256(bytes.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(bytes.as_ptr().add(offset + 32).cast::<__m256i>()),
            );
            let second = _mm256_or_si256(
                _mm256_loadu_si256(bytes.as_ptr().add(offset + 64).cast::<__m256i>()),
                _mm256_loadu_si256(bytes.as_ptr().add(offset + 96).cast::<__m256i>()),
            );
            _mm256_or_si256(first, second)
        };
        if _mm256_movemask_epi8(_mm256_cmpeq_epi8(combined, zero)) != -1 {
            return false;
        }
        offset += 128;
    }

    while offset + 32 <= bytes.len() {
        let chunk = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(offset).cast::<__m256i>()) };
        if _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, zero)) != -1 {
            return false;
        }
        offset += 32;
    }

    bytes_are_zero_scalar(&bytes[offset..])
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
#[target_feature(enable = "avx2")]
unsafe fn xor_4_32(dests: [*mut u8; 4], src: *const u8, offset: usize) {
    let source = unsafe { _mm256_loadu_si256(src.add(offset).cast::<__m256i>()) };
    for dest in dests {
        let dest = unsafe { dest.add(offset).cast::<__m256i>() };
        let updated = unsafe { _mm256_xor_si256(_mm256_loadu_si256(dest.cast_const()), source) };
        unsafe {
            _mm256_storeu_si256(dest, updated);
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn xor_8_32(dests: [*mut u8; 8], src: *const u8, offset: usize) {
    let source = unsafe { _mm256_loadu_si256(src.add(offset).cast::<__m256i>()) };
    for dest in dests {
        let dest = unsafe { dest.add(offset).cast::<__m256i>() };
        let updated = unsafe { _mm256_xor_si256(_mm256_loadu_si256(dest.cast_const()), source) };
        unsafe {
            _mm256_storeu_si256(dest, updated);
        }
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn xor_sources_4_32(dest: *mut u8, srcs: [*const u8; 4], offset: usize) {
    let dest = unsafe { dest.add(offset).cast::<__m256i>() };
    let mut updated = unsafe { _mm256_loadu_si256(dest.cast_const()) };
    for src in srcs {
        let source = unsafe { _mm256_loadu_si256(src.add(offset).cast::<__m256i>()) };
        updated = _mm256_xor_si256(updated, source);
    }
    unsafe {
        _mm256_storeu_si256(dest, updated);
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn xor_sources_8_32(dest: *mut u8, srcs: [*const u8; 8], offset: usize) {
    let dest = unsafe { dest.add(offset).cast::<__m256i>() };
    let mut updated = unsafe { _mm256_loadu_si256(dest.cast_const()) };
    for src in srcs {
        let source = unsafe { _mm256_loadu_si256(src.add(offset).cast::<__m256i>()) };
        updated = _mm256_xor_si256(updated, source);
    }
    unsafe {
        _mm256_storeu_si256(dest, updated);
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn xor_sources_16_32(dest: *mut u8, srcs: [*const u8; 16], offset: usize) {
    let dest = unsafe { dest.add(offset).cast::<__m256i>() };
    let mut updated = unsafe { _mm256_loadu_si256(dest.cast_const()) };
    for src in srcs {
        let source = unsafe { _mm256_loadu_si256(src.add(offset).cast::<__m256i>()) };
        updated = _mm256_xor_si256(updated, source);
    }
    unsafe {
        _mm256_storeu_si256(dest, updated);
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn mulassign_32(
    dest: &mut [u8],
    offset: usize,
    low_table: __m256i,
    high_table: __m256i,
    mask: __m256i,
) {
    unsafe {
        let dest_ptr = dest.as_mut_ptr().add(offset).cast::<__m256i>();
        let value = _mm256_loadu_si256(dest_ptr.cast_const());
        let product = gf256_mul_vector(value, low_table, high_table, mask);
        _mm256_storeu_si256(dest_ptr, product);
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn fused_addassign_mul_32(
    dest: &mut [u8],
    src: &[u8],
    offset: usize,
    low_table: __m256i,
    high_table: __m256i,
    mask: __m256i,
) {
    unsafe {
        let source = _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>());
        let product = gf256_mul_vector(source, low_table, high_table, mask);
        let dest_ptr = dest.as_mut_ptr().add(offset).cast::<__m256i>();
        let current = _mm256_loadu_si256(dest_ptr.cast_const());
        let next = _mm256_xor_si256(current, product);
        _mm256_storeu_si256(dest_ptr, next);
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn fused_addassign_mul_column_coefficients_32(
    dests: *mut u8,
    dest_stride: usize,
    src: *const u8,
    tables: &ColumnCoefficientAvx2Tables<'_>,
    offset: usize,
) {
    let source = unsafe { _mm256_loadu_si256(src.add(offset).cast::<__m256i>()) };
    for ((&row, &low_table), &high_table) in tables
        .rows
        .iter()
        .zip(tables.low_tables.iter())
        .zip(tables.high_tables.iter())
    {
        let product = gf256_mul_vector(source, low_table, high_table, tables.mask);
        let dest = unsafe { dests.add(row * dest_stride + offset).cast::<__m256i>() };
        let current = unsafe { _mm256_loadu_si256(dest.cast_const()) };
        let next = _mm256_xor_si256(current, product);
        unsafe {
            _mm256_storeu_si256(dest, next);
        }
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
    fn mulassign_scalar_keeps_zero_symbol_zero() {
        let mut data = vec![0u8; 129];

        mulassign_scalar(&mut data, &Octet::new(7));

        assert_eq!(data, vec![0u8; 129]);
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

                let mut fast_path_dest = patterned_bytes(len);
                FusedAddAssignMulScalarFastPath::new(len).apply(&mut fast_path_dest, &src, &octet);
                assert_eq!(fast_path_dest, expected);
            }
        }
    }

    #[test]
    fn fused_column_coefficients_match_scalar_tables() {
        for len in [0usize, 1, 31, 32, 33, 64, 79] {
            let src = patterned_bytes(len)
                .into_iter()
                .map(|byte| byte.rotate_left(2) ^ 0x39)
                .collect::<Vec<_>>();
            let coefficients = [0u8, 1, 2, 29, 255];
            let mut dests = Vec::new();
            let mut expected = Vec::new();

            for (row, &coefficient) in coefficients.iter().enumerate() {
                let original = patterned_bytes(len)
                    .into_iter()
                    .map(|byte| byte.wrapping_add(row as u8))
                    .collect::<Vec<_>>();
                let mut row_expected = original.clone();
                if coefficient == 1 {
                    for (dest, src) in row_expected.iter_mut().zip(src.iter()) {
                        *dest ^= *src;
                    }
                } else if coefficient != 0 {
                    let table = Octet::new(coefficient).mul_table();
                    for (dest, src) in row_expected.iter_mut().zip(src.iter()) {
                        *dest ^= table[*src as usize];
                    }
                }
                dests.extend_from_slice(&original);
                expected.extend_from_slice(&row_expected);
            }

            unsafe {
                FusedAddAssignMulScalarFastPath::new(len).apply_column_coefficients(
                    dests.as_mut_ptr(),
                    len,
                    src.as_ptr(),
                    &coefficients,
                    len,
                );
            }
            assert_eq!(dests, expected);
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

            let mut fast_path_direct = original.clone();
            AddAssignFastPath::new(len).apply(&mut fast_path_direct, &src);
            assert_eq!(
                fast_path_direct, expected,
                "AddAssignFastPath failed for length {len}"
            );

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

    #[test]
    fn add_assign_raw_4_matches_scalar_xor() {
        for len in [31usize, 32, 64, 128, 129] {
            let src = patterned_bytes(len)
                .into_iter()
                .map(|byte| byte.rotate_left(2) ^ 0x5a)
                .collect::<Vec<_>>();
            let mut bytes = vec![0u8; len * 5];
            bytes[..len].copy_from_slice(&src);
            let mut expected = bytes.clone();

            for symbol in 1..5 {
                let start = symbol * len;
                let dest = patterned_bytes(len)
                    .into_iter()
                    .map(|byte| byte.wrapping_add(symbol as u8))
                    .collect::<Vec<_>>();
                bytes[start..start + len].copy_from_slice(&dest);
                expected[start..start + len].copy_from_slice(&scalar_xor(dest, &src));
            }

            let ptr = bytes.as_mut_ptr();
            unsafe {
                AddAssignFastPath::new(len).apply_same_len_raw_4(
                    [
                        ptr.add(len),
                        ptr.add(len * 2),
                        ptr.add(len * 3),
                        ptr.add(len * 4),
                    ],
                    ptr.cast_const(),
                    len,
                );
            }

            assert_eq!(
                bytes, expected,
                "raw 4-way add_assign failed for length {len}"
            );
        }
    }

    #[test]
    fn add_assign_raw_8_matches_scalar_xor() {
        for len in [31usize, 32, 64, 128, 129] {
            let src = patterned_bytes(len)
                .into_iter()
                .map(|byte| byte.rotate_left(2) ^ 0x5a)
                .collect::<Vec<_>>();
            let mut bytes = vec![0u8; len * 9];
            bytes[..len].copy_from_slice(&src);
            let mut expected = bytes.clone();

            for symbol in 1..9 {
                let start = symbol * len;
                let dest = patterned_bytes(len)
                    .into_iter()
                    .map(|byte| byte.wrapping_add(symbol as u8))
                    .collect::<Vec<_>>();
                bytes[start..start + len].copy_from_slice(&dest);
                expected[start..start + len].copy_from_slice(&scalar_xor(dest, &src));
            }

            let ptr = bytes.as_mut_ptr();
            unsafe {
                AddAssignFastPath::new(len).apply_same_len_raw_8(
                    [
                        ptr.add(len),
                        ptr.add(len * 2),
                        ptr.add(len * 3),
                        ptr.add(len * 4),
                        ptr.add(len * 5),
                        ptr.add(len * 6),
                        ptr.add(len * 7),
                        ptr.add(len * 8),
                    ],
                    ptr.cast_const(),
                    len,
                );
            }

            assert_eq!(
                bytes, expected,
                "raw 8-way add_assign failed for length {len}"
            );
        }
    }

    #[test]
    fn add_assign_sources_raw_4_matches_scalar_xor() {
        for len in [31usize, 32, 64, 128, 129] {
            let mut bytes = vec![0u8; len * 5];
            let dest = patterned_bytes(len)
                .into_iter()
                .map(|byte| byte ^ 0xc3)
                .collect::<Vec<_>>();
            bytes[..len].copy_from_slice(&dest);

            let mut expected = dest;
            for symbol in 1..5 {
                let start = symbol * len;
                let src = patterned_bytes(len)
                    .into_iter()
                    .map(|byte| byte.rotate_left(symbol as u32) ^ symbol as u8)
                    .collect::<Vec<_>>();
                bytes[start..start + len].copy_from_slice(&src);
                expected = scalar_xor(expected, &src);
            }

            let ptr = bytes.as_mut_ptr();
            unsafe {
                AddAssignFastPath::new(len).apply_sources_same_len_raw_4(
                    ptr,
                    [
                        ptr.add(len).cast_const(),
                        ptr.add(len * 2).cast_const(),
                        ptr.add(len * 3).cast_const(),
                        ptr.add(len * 4).cast_const(),
                    ],
                    len,
                );
            }

            assert_eq!(
                &bytes[..len],
                expected.as_slice(),
                "raw 4-source add_assign failed for length {len}"
            );
        }
    }

    #[test]
    fn add_assign_sources_raw_8_matches_scalar_xor() {
        for len in [31usize, 32, 64, 128, 129] {
            let mut bytes = vec![0u8; len * 9];
            let dest = patterned_bytes(len)
                .into_iter()
                .map(|byte| byte ^ 0x3c)
                .collect::<Vec<_>>();
            bytes[..len].copy_from_slice(&dest);

            let mut expected = dest;
            for symbol in 1..9 {
                let start = symbol * len;
                let src = patterned_bytes(len)
                    .into_iter()
                    .map(|byte| byte.rotate_left(symbol as u32) ^ symbol as u8)
                    .collect::<Vec<_>>();
                bytes[start..start + len].copy_from_slice(&src);
                expected = scalar_xor(expected, &src);
            }

            let ptr = bytes.as_mut_ptr();
            unsafe {
                AddAssignFastPath::new(len).apply_sources_same_len_raw_8(
                    ptr,
                    [
                        ptr.add(len).cast_const(),
                        ptr.add(len * 2).cast_const(),
                        ptr.add(len * 3).cast_const(),
                        ptr.add(len * 4).cast_const(),
                        ptr.add(len * 5).cast_const(),
                        ptr.add(len * 6).cast_const(),
                        ptr.add(len * 7).cast_const(),
                        ptr.add(len * 8).cast_const(),
                    ],
                    len,
                );
            }

            assert_eq!(
                &bytes[..len],
                expected.as_slice(),
                "raw 8-source add_assign failed for length {len}"
            );
        }
    }

    #[test]
    fn add_assign_sources_raw_16_matches_scalar_xor() {
        for len in [31usize, 32, 64, 128, 129] {
            let mut bytes = vec![0u8; len * 17];
            let dest = patterned_bytes(len)
                .into_iter()
                .map(|byte| byte ^ 0x5a)
                .collect::<Vec<_>>();
            bytes[..len].copy_from_slice(&dest);

            let mut expected = dest;
            for symbol in 1..17 {
                let start = symbol * len;
                let src = patterned_bytes(len)
                    .into_iter()
                    .map(|byte| byte.rotate_left(symbol as u32) ^ symbol as u8)
                    .collect::<Vec<_>>();
                bytes[start..start + len].copy_from_slice(&src);
                expected = scalar_xor(expected, &src);
            }

            let ptr = bytes.as_mut_ptr();
            unsafe {
                AddAssignFastPath::new(len).apply_sources_same_len_raw_16(
                    ptr,
                    [
                        ptr.add(len).cast_const(),
                        ptr.add(len * 2).cast_const(),
                        ptr.add(len * 3).cast_const(),
                        ptr.add(len * 4).cast_const(),
                        ptr.add(len * 5).cast_const(),
                        ptr.add(len * 6).cast_const(),
                        ptr.add(len * 7).cast_const(),
                        ptr.add(len * 8).cast_const(),
                        ptr.add(len * 9).cast_const(),
                        ptr.add(len * 10).cast_const(),
                        ptr.add(len * 11).cast_const(),
                        ptr.add(len * 12).cast_const(),
                        ptr.add(len * 13).cast_const(),
                        ptr.add(len * 14).cast_const(),
                        ptr.add(len * 15).cast_const(),
                        ptr.add(len * 16).cast_const(),
                    ],
                    len,
                );
            }

            assert_eq!(
                &bytes[..len],
                expected.as_slice(),
                "raw 16-source add_assign failed for length {len}"
            );
        }
    }

    #[test]
    fn bytes_are_zero_matches_scalar_for_boundary_lengths() {
        for len in [0usize, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129] {
            let zeros = vec![0u8; len];
            assert!(bytes_are_zero(&zeros), "zero check failed for length {len}");

            for index in [0usize, len.saturating_sub(1), len / 2] {
                if index >= len {
                    continue;
                }
                let mut bytes = zeros.clone();
                bytes[index] = 1;
                assert!(
                    !bytes_are_zero(&bytes),
                    "nonzero byte at {index} was missed for length {len}"
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

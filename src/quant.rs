#![allow(dead_code)] // Bagian dari API yang dipakai tahap berikutnya.

//! Dekuantisasi bobot GGUF/GGML — implementasi mandiri, tanpa dependency.
//!
//! Format blok mengikuti spesifikasi GGML (ggml-common.h / ggml-quants.c).
//! Fungsi di sini hanya membaca data; tidak ada kode dari proyek lain yang
//! dipakai saat runtime.

use std::convert::TryInto;

// ---------------------------------------------------------------------------
// Tipe data GGML
// ---------------------------------------------------------------------------

pub const TYPE_F32: u32 = 0;
pub const TYPE_F16: u32 = 1;
pub const TYPE_Q4_0: u32 = 2;
pub const TYPE_Q4_1: u32 = 3;
pub const TYPE_Q5_0: u32 = 6;
pub const TYPE_Q5_1: u32 = 7;
pub const TYPE_Q8_0: u32 = 8;
pub const TYPE_Q8_1: u32 = 9;
pub const TYPE_Q2_K: u32 = 10;
pub const TYPE_Q3_K: u32 = 11;
pub const TYPE_Q4_K: u32 = 12;
pub const TYPE_Q5_K: u32 = 13;
pub const TYPE_Q6_K: u32 = 14;
pub const TYPE_Q8_K: u32 = 15;
pub const TYPE_BF16: u32 = 30;

/// (jumlah elemen per blok, ukuran blok dalam byte)
pub fn block_layout(dtype: u32) -> Option<(usize, usize)> {
    Some(match dtype {
        TYPE_F32 => (1, 4),
        TYPE_F16 => (1, 2),
        TYPE_Q4_0 => (32, 18),
        TYPE_Q4_1 => (32, 20),
        TYPE_Q5_0 => (32, 22),
        TYPE_Q5_1 => (32, 24),
        TYPE_Q8_0 => (32, 34),
        TYPE_Q8_1 => (32, 36),
        TYPE_Q2_K => (256, 84),
        TYPE_Q3_K => (256, 110),
        TYPE_Q4_K => (256, 144),
        TYPE_Q5_K => (256, 176),
        TYPE_Q6_K => (256, 210),
        TYPE_Q8_K => (256, 292),
        TYPE_BF16 => (1, 2),
        _ => return None,
    })
}

pub fn type_name(dtype: u32) -> &'static str {
    match dtype {
        TYPE_F32 => "F32",
        TYPE_F16 => "F16",
        TYPE_Q4_0 => "Q4_0",
        TYPE_Q4_1 => "Q4_1",
        TYPE_Q5_0 => "Q5_0",
        TYPE_Q5_1 => "Q5_1",
        TYPE_Q8_0 => "Q8_0",
        TYPE_Q8_1 => "Q8_1",
        TYPE_Q2_K => "Q2_K",
        TYPE_Q3_K => "Q3_K",
        TYPE_Q4_K => "Q4_K",
        TYPE_Q5_K => "Q5_K",
        TYPE_Q6_K => "Q6_K",
        TYPE_Q8_K => "Q8_K",
        TYPE_BF16 => "BF16",
        _ => "UNKNOWN",
    }
}

/// Ukuran data tensor dalam byte dari bentuk (dims) dan tipe.
pub fn tensor_nbytes(dtype: u32, dims: &[u64]) -> Option<u64> {
    let (bk_elems, bk_bytes) = block_layout(dtype)?;
    let elems = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d))?;
    if elems % bk_elems as u64 != 0 {
        return None;
    }
    (elems / bk_elems as u64).checked_mul(bk_bytes as u64)
}

// ---------------------------------------------------------------------------
// Konversi floating point
// ---------------------------------------------------------------------------

/// IEEE 754 half precision (u16) -> f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        // subnormal: nilai = mant * 2^-24
        sign * (mant as f32) * 2.0f32.powi(-24)
    } else if exp == 31 {
        if mant == 0 { sign * f32::INFINITY } else { f32::NAN }
    } else {
        sign * (1.0f32 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15)
    }
}

/// Google brain float16 -> f32 (cukup geser 16 bit).
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f32 -> f16 (dipakai pembuat fixture test; pembulatan sederhana).
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        if exp < -10 { return sign; }
        let shifted = (mant | 0x80_0000) >> (1 - exp);
        let mut m = shifted >> 13;
        if (shifted >> 12) & 1 != 0 && (shifted & 0xfff != 0 || m & 1 != 0) { m += 1; }
        sign | m as u16
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        let mut m = mant >> 13;
        if (mant >> 12) & 1 != 0 && (mant & 0xfff != 0 || m & 1 != 0) { m += 1; }
        if m >= 1024 {
            let e = exp + 1;
            return if e >= 31 { sign | 0x7c00 } else { sign | ((e as u16) << 10) };
        }
        sign | ((exp as u16) << 10) | m as u16
    }
}

#[inline]
fn u16le(b: &[u8]) -> u16 { u16::from_le_bytes(b[..2].try_into().unwrap()) }
#[inline]
fn u32le(b: &[u8]) -> u32 { u32::from_le_bytes(b[..4].try_into().unwrap()) }

// ---------------------------------------------------------------------------
// Helper K-quant
// ---------------------------------------------------------------------------

/// Ambil skala (d) dan minimum (m) 6-bit dari array `scales[12]`.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

// ---------------------------------------------------------------------------
// Dekuantisasi per blok
// ---------------------------------------------------------------------------

fn deq_q4_0(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16le(&src[..2]));
    for j in 0..16 {
        let b = src[2 + j];
        out[j] = ((b & 0x0f) as i32 - 8) as f32 * d;
        out[j + 16] = ((b >> 4) as i32 - 8) as f32 * d;
    }
}

fn deq_q4_1(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16le(&src[..2]));
    let m = f16_to_f32(u16le(&src[2..4]));
    for j in 0..16 {
        let b = src[4 + j];
        out[j] = (b & 0x0f) as f32 * d + m;
        out[j + 16] = (b >> 4) as f32 * d + m;
    }
}

fn deq_q5_0(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16le(&src[..2]));
    let qh = u32::from_le_bytes(src[18..22].try_into().unwrap());
    for j in 0..16 {
        let xh0 = ((qh >> (j + 0)) << 4) & 0x10;
        let xh1 = (qh >> (j + 12)) & 0x10;
        let x0 = ((src[2 + j] & 0x0f) as u32 | xh0) as i32 - 16;
        let x1 = ((src[2 + j] >> 4) as u32 | xh1) as i32 - 16;
        out[j] = x0 as f32 * d;
        out[j + 16] = x1 as f32 * d;
    }
}

fn deq_q5_1(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16le(&src[..2]));
    let m = f16_to_f32(u16le(&src[2..4]));
    let qh = u32::from_le_bytes(src[20..24].try_into().unwrap());
    for j in 0..16 {
        let xh0 = ((qh >> (j + 0)) << 4) & 0x10;
        let xh1 = (qh >> (j + 12)) & 0x10;
        let x0 = (src[4 + j] & 0x0f) as u32 | xh0;
        let x1 = (src[4 + j] >> 4) as u32 | xh1;
        out[j] = x0 as f32 * d + m;
        out[j + 16] = x1 as f32 * d + m;
    }
}

fn deq_q8_0(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16le(&src[..2]));
    for j in 0..32 {
        out[j] = (src[2 + j] as i8) as f32 * d;
    }
}

fn deq_q8_1(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16le(&src[..2]));
    for j in 0..32 {
        out[j] = (src[4 + j] as i8) as f32 * d;
    }
}

/// q2_K: `qs` 64 byte untuk 256 elemen, dibaca per grup 128.
fn deq_q2_k(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16le(&src[..2]));
    let min = f16_to_f32(u16le(&src[2..4]));
    let scales = &src[4..20];
    let mut q = &src[20..84];
    let mut is = 0usize;
    let mut o = 0usize;
    for _n in 0..2 {
        let mut shift = 0u32;
        for _j in 0..4 {
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0x0f) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                out[o] = dl * (((q[l] >> shift) & 3) as i8) as f32 - ml;
                o += 1;
            }
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0x0f) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                out[o] = dl * (((q[l + 16] >> shift) & 3) as i8) as f32 - ml;
                o += 1;
            }
            shift += 2;
        }
        q = &q[32..];
    }
}

fn deq_q3_k(src: &[u8], out: &mut [f32]) {
    let d_all = f16_to_f32(u16le(&src[..2]));
    let scales_raw = &src[2..14];
    let hmask = &src[14..46];
    let mut q = &src[46..110]; // qs = QK_K/4 = 64 byte

    let mut aux = [0u32; 4];
    for i in 0..3 {
        aux[i] = u32::from_le_bytes(scales_raw[i * 4..i * 4 + 4].try_into().unwrap());
    }
    let kmask1 = 0x0303_0303u32;
    let kmask2 = 0x0f0f_0f0fu32;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let sc_bytes: Vec<u8> = aux.iter().flat_map(|a| a.to_le_bytes()).collect();
    let sc: Vec<i8> = sc_bytes.iter().map(|&b| b as i8).collect();

    let mut is = 0usize;
    let mut m = 1u8;
    let mut o = 0usize;
    for _n in 0..2 {
        let mut shift = 0u32;
        for _j in 0..4 {
            let dl = d_all * (sc[is] as f32 - 32.0);
            is += 1;
            for l in 0..16 {
                let v = (((q[l] >> shift) & 3) as i8) - if hmask[l] & m != 0 { 0 } else { 4 };
                out[o] = dl * v as f32;
                o += 1;
            }
            let dl = d_all * (sc[is] as f32 - 32.0);
            is += 1;
            for l in 0..16 {
                let v = (((q[l + 16] >> shift) & 3) as i8) - if hmask[l + 16] & m != 0 { 0 } else { 4 };
                out[o] = dl * v as f32;
                o += 1;
            }
            shift += 2;
            m <<= 1;
        }
        q = &q[32..];
    }
}

fn deq_q4_k(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16le(&src[..2]));
    let min = f16_to_f32(u16le(&src[2..4]));
    let scales = &src[4..16];
    let mut q = &src[16..144];
    let mut is = 0usize;
    let mut o = 0usize;
    for _j in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for l in 0..32 { out[o + l] = d1 * (q[l] & 0x0f) as f32 - m1; }
        for l in 0..32 { out[o + 32 + l] = d2 * (q[l] >> 4) as f32 - m2; }
        o += 64;
        q = &q[32..];
        is += 2;
    }
}

fn deq_q5_k(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16le(&src[..2]));
    let min = f16_to_f32(u16le(&src[2..4]));
    let scales = &src[4..16];
    let qh = &src[16..48];
    let mut ql = &src[48..176];
    let mut is = 0usize;
    let mut o = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    for _j in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for l in 0..32 {
            let v1 = (ql[l] & 0x0f) as u32 + if qh[l] & u1 != 0 { 16 } else { 0 };
            let v2 = (ql[l] >> 4) as u32 + if qh[l] & u2 != 0 { 16 } else { 0 };
            out[o + l] = d1 * v1 as f32 - m1;
            out[o + 32 + l] = d2 * v2 as f32 - m2;
        }
        o += 64;
        ql = &ql[32..];
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn deq_q6_k(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16le(&src[..2]));
    let mut ql = &src[2..130];
    let mut qh = &src[130..194];
    let mut sc = &src[194..210];
    for _n in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0f) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) as i16 - 32;
            let q2 = ((ql[l + 32] & 0x0f) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) as i16 - 32;
            let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) as i16 - 32;
            let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) as i16 - 32;
            out[l] = d * sc[is] as i8 as f32 * q1 as f32;
            out[l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
            out[l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
            out[l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
        }
        ql = &ql[64..];
        qh = &qh[32..];
        sc = &sc[8..];
    }
}

/// Dekuantisasi `n` elemen dari `src` ke `out`.
pub fn dequantize(dtype: u32, src: &[u8], n: usize, out: &mut [f32]) -> Result<(), String> {
    if out.len() != n {
        return Err(format!("dequantize: output {} != {}", out.len(), n));
    }
    let (bk_elems, bk_bytes) = block_layout(dtype)
        .ok_or_else(|| format!("tipe GGML {} belum didukung", dtype))?;
    if n % bk_elems != 0 {
        return Err(format!("jumlah elemen {} bukan kelipatan blok {}", n, bk_elems));
    }
    let need = n / bk_elems * bk_bytes;
    if src.len() < need {
        return Err(format!("data kurang: {} < {}", src.len(), need));
    }
    // Tipe tidak terkuantisasi: konversi langsung (sekali jalan).
    match dtype {
        TYPE_F32 => {
            for i in 0..n { out[i] = f32::from_bits(u32le(&src[i * 4..])); }
            return Ok(());
        }
        TYPE_F16 => {
            for i in 0..n { out[i] = f16_to_f32(u16le(&src[i * 2..])); }
            return Ok(());
        }
        TYPE_BF16 => {
            for i in 0..n { out[i] = bf16_to_f32(u16le(&src[i * 2..])); }
            return Ok(());
        }
        _ => {}
    }

    for bi in 0..n / bk_elems {
        let s = &src[bi * bk_bytes..(bi + 1) * bk_bytes];
        let o = &mut out[bi * bk_elems..(bi + 1) * bk_elems];
        match dtype {
            TYPE_Q4_0 => deq_q4_0(s, o),
            TYPE_Q4_1 => deq_q4_1(s, o),
            TYPE_Q5_0 => deq_q5_0(s, o),
            TYPE_Q5_1 => deq_q5_1(s, o),
            TYPE_Q8_0 => deq_q8_0(s, o),
            TYPE_Q8_1 => deq_q8_1(s, o),
            TYPE_Q2_K => deq_q2_k(s, o),
            TYPE_Q3_K => deq_q3_k(s, o),
            TYPE_Q4_K => deq_q4_k(s, o),
            TYPE_Q5_K => deq_q5_k(s, o),
            TYPE_Q6_K => deq_q6_k(s, o),
            _ => return Err(format!("tipe GGML {} belum didukung", dtype)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_basic() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 2.0, -3.25] {
            let h = f32_to_f16(v);
            assert!((f16_to_f32(h) - v).abs() < 1e-3, "{v} -> {h}");
        }
    }

    #[test]
    fn deq_q8_0_simple() {
        let mut block = vec![0u8; 34];
        block[..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        block[2..34].iter_mut().for_each(|b| *b = 2);
        let mut out = vec![0f32; 32];
        dequantize(TYPE_Q8_0, &block, 32, &mut out).unwrap();
        assert!(out.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn deq_k_quants_zero_blocks() {
        for (t, bytes) in [
            (TYPE_Q2_K, 84), (TYPE_Q3_K, 110), (TYPE_Q4_K, 144),
            (TYPE_Q5_K, 176), (TYPE_Q6_K, 210),
        ] {
            let src = vec![0u8; bytes];
            let mut out = vec![0f32; 256];
            dequantize(t, &src, 256, &mut out).unwrap();
            assert!(out.iter().all(|&x| x == 0.0), "tipe {t} menghasilkan nilai nonzero");
        }
    }

    #[test]
    fn block_layouts_known() {
        assert_eq!(block_layout(TYPE_Q4_K), Some((256, 144)));
        assert_eq!(block_layout(TYPE_Q5_K), Some((256, 176)));
        assert_eq!(block_layout(TYPE_Q6_K), Some((256, 210)));
        assert_eq!(block_layout(TYPE_Q2_K), Some((256, 84)));
        assert_eq!(block_layout(TYPE_Q3_K), Some((256, 110)));
    }
}

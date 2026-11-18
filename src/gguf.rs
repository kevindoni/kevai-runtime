#![allow(dead_code)] // Bagian dari API yang dipakai tahap berikutnya.

//! Parser GGUF v2/v3 — ditulis dari nol dengan Rust std saja.
//!
//! Struktur file: header -> metadata key/value -> info tensor -> (padding) -> data tensor.

use crate::quant::{self, tensor_nbytes, type_name};
use std::fs;
use std::path::Path;

const MAGIC: &[u8; 4] = b"GGUF";

// Tipe nilai metadata GGUF
const V_UINT8: u32 = 0;
const V_INT8: u32 = 1;
const V_UINT16: u32 = 2;
const V_INT16: u32 = 3;
const V_UINT32: u32 = 4;
const V_INT32: u32 = 5;
const V_FLOAT32: u32 = 6;
const V_BOOL: u32 = 7;
const V_STRING: u32 = 8;
const V_ARRAY: u32 = 9;
const V_UINT64: u32 = 10;
const V_INT64: u32 = 11;
const V_FLOAT64: u32 = 12;

#[derive(Debug, Clone)]
pub enum Value {
    Uint8(u8), Int8(i8), Uint16(u16), Int16(i16),
    Uint32(u32), Int32(i32), Uint64(u64), Int64(i64),
    Float32(f32), Float64(f64), Bool(bool),
    String(String), Array(Vec<Value>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self { Value::String(s) => Some(s), _ => None }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint8(v) => Some(*v as u64), Value::Int8(v) => Some(*v as u64),
            Value::Uint16(v) => Some(*v as u64), Value::Int16(v) => Some(*v as u64),
            Value::Uint32(v) => Some(*v as u64), Value::Int32(v) => Some(*v as u64),
            Value::Uint64(v) => Some(*v), Value::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::Float32(v) => Some(*v), Value::Float64(v) => Some(*v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self { Value::Bool(v) => Some(*v), _ => None }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self { Value::Array(v) => Some(v), _ => None }
    }
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Uint8(_) => "uint8", Value::Int8(_) => "int8",
            Value::Uint16(_) => "uint16", Value::Int16(_) => "int16",
            Value::Uint32(_) => "uint32", Value::Int32(_) => "int32",
            Value::Uint64(_) => "uint64", Value::Int64(_) => "int64",
            Value::Float32(_) => "float32", Value::Float64(_) => "float64",
            Value::Bool(_) => "bool", Value::String(_) => "string",
            Value::Array(_) => "array",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: u32,
    /// Offset relatif terhadap awal region data tensor.
    pub offset: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> u64 { self.dims.iter().product() }
    pub fn nbytes(&self) -> Result<u64, String> {
        tensor_nbytes(self.dtype, &self.dims)
            .ok_or_else(|| format!("bentuk tensor {} tidak valid untuk {}", self.name, type_name(self.dtype)))
    }
    pub fn dtype_name(&self) -> &'static str { type_name(self.dtype) }
}

pub struct Gguf {
    pub version: u32,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    /// Posisi absolut di file tempat data tensor dimulai.
    pub data_offset: usize,
    bytes: Vec<u8>,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self { Self { b, p: 0 } }
    fn pos(&self) -> usize { self.p }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("overflow saat membaca GGUF")?;
        if end > self.b.len() {
            return Err(format!("GGUF terpotong: butuh {} byte di offset {}, file hanya {}", n, self.p, self.b.len()));
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> { Ok(self.take(1)?[0]) }
    fn i8(&mut self) -> Result<i8, String> { Ok(self.u8()? as i8) }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16, String> { Ok(self.u16()? as i16) }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, String> { Ok(self.u32()? as i32) }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, String> { Ok(self.u64()? as i64) }
    fn f32(&mut self) -> Result<f32, String> { Ok(f32::from_bits(self.u32()?)) }
    fn f64(&mut self) -> Result<f64, String> { Ok(f64::from_bits(self.u64()?)) }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        let s = self.take(n)?;
        String::from_utf8(s.to_vec()).map_err(|e| format!("string UTF-8 invalid di metadata: {e}"))
    }
    fn value(&mut self, typ: u32, depth: usize) -> Result<Value, String> {
        if depth > 16 { return Err("metadata GGUF terlalu dalam".into()); }
        Ok(match typ {
            V_UINT8 => Value::Uint8(self.u8()?),
            V_INT8 => Value::Int8(self.i8()?),
            V_UINT16 => Value::Uint16(self.u16()?),
            V_INT16 => Value::Int16(self.i16()?),
            V_UINT32 => Value::Uint32(self.u32()?),
            V_INT32 => Value::Int32(self.i32()?),
            V_UINT64 => Value::Uint64(self.u64()?),
            V_INT64 => Value::Int64(self.i64()?),
            V_FLOAT32 => Value::Float32(self.f32()?),
            V_FLOAT64 => Value::Float64(self.f64()?),
            V_BOOL => Value::Bool(self.u8()? != 0),
            V_STRING => Value::String(self.string()?),
            V_ARRAY => {
                let item_type = self.u32()?;
                let n = self.u64()? as usize;
                if n > 10_000_000 { return Err("array metadata terlalu besar".into()); }
                let mut v = Vec::with_capacity(n.min(1024));
                for _ in 0..n { v.push(self.value(item_type, depth + 1)?); }
                Value::Array(v)
            }
            _ => return Err(format!("tipe metadata GGUF tidak dikenal: {typ}")),
        })
    }
}

impl Gguf {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("gagal membaca {}: {e}", path.display()))?;
        Self::parse(bytes)
    }

    pub fn parse(bytes: Vec<u8>) -> Result<Self, String> {
        let mut r = Reader::new(&bytes);
        if r.take(4)? != MAGIC {
            return Err("bukan file GGUF (magic tidak cocok)".into());
        }
        let version = r.u32()?;
        if version != 2 && version != 3 {
            return Err(format!("versi GGUF {version} belum didukung (butuh 2 atau 3)"));
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()? as usize;
        if tensor_count > 100_000 || kv_count > 100_000 {
            return Err("jumlah tensor/metadata tidak wajar".into());
        }

        let mut metadata = Vec::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = r.string()?;
            let typ = r.u32()?;
            let val = r.value(typ, 0)?;
            metadata.push((key, val));
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let ndim = r.u32()? as usize;
            if ndim == 0 || ndim > 4 {
                return Err(format!("jumlah dimensi tensor tidak valid: {ndim}"));
            }
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim { dims.push(r.u64()?); }
            let dtype = r.u32()?;
            let offset = r.u64()?;
            tensors.push(TensorInfo { name, dims, dtype, offset });
        }

        let align = metadata.iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
            .unwrap_or(32)
            .max(1) as usize;
        let pad = (align - (r.pos() % align)) % align;
        r.take(pad)?;
        let data_offset = r.pos();
        let data_len = bytes.len().saturating_sub(data_offset);

        for t in &tensors {
            let nb = t.nbytes()?;
            let end = t.offset.checked_add(nb).ok_or("offset tensor overflow")?;
            if end > data_len as u64 {
                return Err(format!("tensor {} melewati akhir file (offset {}, {} byte, data {} byte)",
                    t.name, t.offset, nb, data_len));
            }
        }

        Ok(Self { version, metadata, tensors, data_offset, bytes })
    }

    pub fn meta(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn meta_str(&self, key: &str) -> Option<&str> { self.meta(key).and_then(Value::as_str) }
    pub fn meta_u64(&self, key: &str) -> Option<u64> { self.meta(key).and_then(Value::as_u64) }
    pub fn meta_f32(&self, key: &str) -> Option<f32> { self.meta(key).and_then(Value::as_f32) }
    pub fn meta_bool(&self, key: &str) -> Option<bool> { self.meta(key).and_then(Value::as_bool) }
    pub fn meta_string_array(&self, key: &str) -> Option<Vec<String>> {
        self.meta(key).and_then(Value::as_array).map(|a| {
            a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
        })
    }
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn architecture(&self) -> &str { self.meta_str("general.architecture").unwrap_or("unknown") }

    /// Baca tensor dan kembalikan sebagai f32 (dekuantisasi jika perlu).
    pub fn tensor_f32(&self, t: &TensorInfo) -> Result<Vec<f32>, String> {
        let n = t.numel() as usize;
        let nb = t.nbytes()? as usize;
        let start = self.data_offset
            .checked_add(t.offset as usize)
            .ok_or("offset tensor overflow")?;
        let end = start.checked_add(nb).ok_or("ukuran tensor overflow")?;
        if end > self.bytes.len() { return Err("data tensor melewati akhir file".into()); }
        let mut out = vec![0f32; n];
        quant::dequantize(t.dtype, &self.bytes[start..end], n, &mut out)?;
        Ok(out)
    }
}

/// File GGUF sintetis kecil untuk mencoba parser tanpa model asli.
pub fn demo_fixture() -> Vec<u8> {
    fn push_str(v: &mut Vec<u8>, s: &str) {
        v.extend_from_slice(&(s.len() as u64).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
    }
    {
        let mut f = Vec::new();
        f.extend_from_slice(MAGIC);
        f.extend_from_slice(&2u32.to_le_bytes());   // version
        f.extend_from_slice(&2u64.to_le_bytes());   // tensor_count
        f.extend_from_slice(&3u64.to_le_bytes());   // kv_count

        // metadata
        push_str(&mut f, "general.architecture");
        f.extend_from_slice(&V_STRING.to_le_bytes());
        push_str(&mut f, "qwen2");

        push_str(&mut f, "general.alignment");
        f.extend_from_slice(&V_UINT32.to_le_bytes());
        f.extend_from_slice(&32u32.to_le_bytes());

        push_str(&mut f, "tokenizer.ggml.tokens");
        f.extend_from_slice(&V_ARRAY.to_le_bytes());
        f.extend_from_slice(&V_STRING.to_le_bytes());
        f.extend_from_slice(&2u64.to_le_bytes());
        push_str(&mut f, "hello");
        push_str(&mut f, "world");

        // tensor 1: F32 [4, 2] = 8 elemen = 32 byte, offset 0
        push_str(&mut f, "tok.weight");
        f.extend_from_slice(&2u32.to_le_bytes());
        f.extend_from_slice(&4u64.to_le_bytes());
        f.extend_from_slice(&2u64.to_le_bytes());
        f.extend_from_slice(&0u32.to_le_bytes()); // F32
        f.extend_from_slice(&0u64.to_le_bytes());

        // tensor 2: Q4_K [256] = 144 byte, offset 32
        push_str(&mut f, "blk.weight");
        f.extend_from_slice(&1u32.to_le_bytes());
        f.extend_from_slice(&256u64.to_le_bytes());
        f.extend_from_slice(&12u32.to_le_bytes()); // Q4_K
        f.extend_from_slice(&32u64.to_le_bytes());

        // padding ke 32
        while f.len() % 32 != 0 { f.push(0); }
        for i in 0..8u32 { f.extend_from_slice(&(i as f32).to_le_bytes()); }
        f.extend(std::iter::repeat(0u8).take(144));
        f
    }
}

/// GGUF Qwen2 mini (F32) untuk menguji tokenizer -> forward -> server end-to-end.
pub fn tiny_fixture() -> Vec<u8> {
    fn ps(v: &mut Vec<u8>, s: &str) {
        v.extend_from_slice(&(s.len() as u64).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
    }

    const N_EMBD: u64 = 8;
    const N_FF: u64 = 16;
    const N_HEADS: u64 = 2;
    const N_KV: u64 = 1;

    // Vocabulary: semua karakter byte-level BPE + token khusus.
    let mut bs: Vec<u32> = (0x21..=0x7e).chain(0xa1..=0xac).chain(0xae..=0xff).collect();
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) { bs.push(b); cs.push(256 + n); n += 1; }
    }
    let mut tokens: Vec<String> = cs.iter().map(|&c| char::from_u32(c).unwrap().to_string()).collect();
    for s in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "assistant", "system", "user"] {
        tokens.push(s.to_string());
    }
    let vocab = tokens.len() as u64;

    // Susun data tensor dulu supaya offset diketahui.
    let specs: Vec<(&str, Vec<u64>)> = vec![
        ("token_embd.weight", vec![N_EMBD, vocab]),
        ("blk.0.attn_norm.weight", vec![N_EMBD]),
        ("blk.0.attn_q.weight", vec![N_EMBD, N_EMBD]),
        ("blk.0.attn_k.weight", vec![N_EMBD, N_KV * (N_EMBD / N_HEADS)]),
        ("blk.0.attn_v.weight", vec![N_EMBD, N_KV * (N_EMBD / N_HEADS)]),
        ("blk.0.attn_output.weight", vec![N_EMBD, N_EMBD]),
        ("blk.0.ffn_norm.weight", vec![N_EMBD]),
        ("blk.0.ffn_gate.weight", vec![N_EMBD, N_FF]),
        ("blk.0.ffn_up.weight", vec![N_EMBD, N_FF]),
        ("blk.0.ffn_down.weight", vec![N_FF, N_EMBD]),
        ("output_norm.weight", vec![N_EMBD]),
        ("output.weight", vec![N_EMBD, vocab]),
    ];
    let mut data: Vec<u8> = Vec::new();
    let mut infos: Vec<(&str, Vec<u64>, u64)> = Vec::new();
    let mut seed = 0u32;
    for (name, dims) in specs {
        while data.len() % 32 != 0 { data.push(0); }
        let offset = data.len() as u64;
        let elems: usize = dims.iter().map(|&d| d as usize).product();
        for _ in 0..elems {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((seed >> 16) as f32 / 32768.0 - 1.0) * 0.05;
            data.extend_from_slice(&v.to_le_bytes());
        }
        infos.push((name, dims, offset));
    }

    let mut f = Vec::new();
    f.extend_from_slice(MAGIC);
    f.extend_from_slice(&2u32.to_le_bytes());
    f.extend_from_slice(&(infos.len() as u64).to_le_bytes());
    f.extend_from_slice(&12u64.to_le_bytes());

    ps(&mut f, "general.architecture"); f.extend_from_slice(&V_STRING.to_le_bytes()); ps(&mut f, "qwen2");
    ps(&mut f, "general.alignment"); f.extend_from_slice(&V_UINT32.to_le_bytes()); f.extend_from_slice(&32u32.to_le_bytes());
    ps(&mut f, "qwen2.block_count"); f.extend_from_slice(&V_UINT32.to_le_bytes()); f.extend_from_slice(&1u32.to_le_bytes());
    ps(&mut f, "qwen2.embedding_length"); f.extend_from_slice(&V_UINT32.to_le_bytes()); f.extend_from_slice(&(N_EMBD as u32).to_le_bytes());
    ps(&mut f, "qwen2.attention.head_count"); f.extend_from_slice(&V_UINT32.to_le_bytes()); f.extend_from_slice(&(N_HEADS as u32).to_le_bytes());
    ps(&mut f, "qwen2.attention.head_count_kv"); f.extend_from_slice(&V_UINT32.to_le_bytes()); f.extend_from_slice(&(N_KV as u32).to_le_bytes());
    ps(&mut f, "qwen2.feed_forward_length"); f.extend_from_slice(&V_UINT32.to_le_bytes()); f.extend_from_slice(&(N_FF as u32).to_le_bytes());
    ps(&mut f, "qwen2.context_length"); f.extend_from_slice(&V_UINT32.to_le_bytes()); f.extend_from_slice(&512u32.to_le_bytes());
    ps(&mut f, "qwen2.rope.freq_base"); f.extend_from_slice(&V_FLOAT32.to_le_bytes()); f.extend_from_slice(&10000f32.to_le_bytes());
    ps(&mut f, "qwen2.attention.layer_norm_rms_epsilon"); f.extend_from_slice(&V_FLOAT32.to_le_bytes()); f.extend_from_slice(&1e-6f32.to_le_bytes());
    ps(&mut f, "tokenizer.ggml.tokens");
    f.extend_from_slice(&V_ARRAY.to_le_bytes()); f.extend_from_slice(&V_STRING.to_le_bytes());
    f.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for t in &tokens { ps(&mut f, t); }
    ps(&mut f, "tokenizer.ggml.merges");
    f.extend_from_slice(&V_ARRAY.to_le_bytes()); f.extend_from_slice(&V_STRING.to_le_bytes());
    f.extend_from_slice(&0u64.to_le_bytes());

    for (name, dims, offset) in infos {
        ps(&mut f, name);
        f.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims { f.extend_from_slice(&d.to_le_bytes()); }
        f.extend_from_slice(&0u32.to_le_bytes()); // F32
        f.extend_from_slice(&offset.to_le_bytes());
    }
    while f.len() % 32 != 0 { f.push(0); }
    f.extend_from_slice(&data);
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_header_metadata_and_tensors() {
        let g = Gguf::parse(demo_fixture()).unwrap();
        assert_eq!(g.version, 2);
        assert_eq!(g.architecture(), "qwen2");
        assert_eq!(g.meta_u64("general.alignment"), Some(32));
        assert_eq!(g.meta_string_array("tokenizer.ggml.tokens").unwrap(), vec!["hello", "world"]);
        assert_eq!(g.tensors.len(), 2);
        assert_eq!(g.tensors[1].dtype_name(), "Q4_K");
        assert_eq!(g.tensors[1].nbytes().unwrap(), 144);
        assert_eq!(g.data_offset % 32, 0);
    }

    #[test]
    fn read_tensor_values() {
        let g = Gguf::parse(demo_fixture()).unwrap();
        let t = g.find_tensor("tok.weight").unwrap();
        let v = g.tensor_f32(t).unwrap();
        assert_eq!(v.len(), 8);
        for (i, &x) in v.iter().enumerate() { assert_eq!(x, i as f32); }
        let t2 = g.find_tensor("blk.weight").unwrap();
        assert_eq!(g.tensor_f32(t2).unwrap().len(), 256);
    }

    #[test]
    fn reject_bad_magic() {
        assert!(Gguf::parse(b"NOPE".to_vec()).is_err());
    }
}

//! Forward pass Qwen2 / Llama (dense) murni Rust.
//! Dua mode bobot:
//!  - default: semua tensor di-dekuantisasi ke f32 saat load (cepat, RAM ~4x file)
//!  - low_mem: bobot tetap terkuantisasi (RAM ~ukuran file), dekuantisasi per kolom saat matmul

use crate::gguf::Gguf;
use crate::quant::{self, block_layout};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Config {
    pub arch: String,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub ctx_len: usize,
    pub freq_base: f32,
    pub rms_eps: f32,
}

fn meta_any_u64(g: &Gguf, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| g.meta_u64(k))
}
fn meta_any_f32(g: &Gguf, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|k| g.meta_f32(k))
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let arch = g.architecture().to_string();
        let n_layers = meta_any_u64(g, &["qwen2.block_count", "llama.block_count"])
            .ok_or("metadata block_count tidak ada")? as usize;
        let n_embd = meta_any_u64(g, &["qwen2.embedding_length", "llama.embedding_length"])
            .ok_or("metadata embedding_length tidak ada")? as usize;
        let n_heads = meta_any_u64(g, &["qwen2.attention.head_count", "llama.attention.head_count"])
            .ok_or("metadata head_count tidak ada")? as usize;
        let n_kv_heads = meta_any_u64(g, &[
            "qwen2.attention.head_count_kv", "llama.attention.head_count_kv",
        ]).unwrap_or(n_heads as u64) as usize;
        let n_ff = meta_any_u64(g, &["qwen2.feed_forward_length", "llama.feed_forward_length"])
            .ok_or("metadata feed_forward_length tidak ada")? as usize;
        let n_vocab = meta_any_u64(g, &["qwen2.vocab_size", "llama.vocab_size"])
            .unwrap_or(0) as usize;
        let ctx_len = meta_any_u64(g, &[
            "qwen2.context_length", "llama.context_length",
        ]).unwrap_or(2048) as usize;
        let freq_base = meta_any_f32(g, &[
            "qwen2.rope.freq_base", "llama.rope.freq_base",
        ]).unwrap_or(1_000_000.0);
        let rms_eps = meta_any_f32(g, &[
            "qwen2.attention.layer_norm_rms_epsilon",
            "llama.attention.layer_norm_rms_epsilon",
        ]).unwrap_or(1e-6);

        if n_heads == 0 || n_embd == 0 || n_embd % n_heads != 0 {
            return Err("konfigurasi attention tidak valid".into());
        }
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err("head_count_kv harus membagi head_count".into());
        }
        let vocab = if n_vocab > 0 { n_vocab } else {
            g.meta_string_array("tokenizer.ggml.tokens").map(|v| v.len())
                .ok_or("ukuran vocabulary tidak diketahui")?
        };
        Ok(Self { arch, n_layers, n_heads, n_kv_heads, n_embd, n_ff, n_vocab: vocab, ctx_len, freq_base, rms_eps })
    }

    pub fn head_dim(&self) -> usize { self.n_embd / self.n_heads }
}

// ---------------------------------------------------------------------------
// Penyimpanan bobot
// ---------------------------------------------------------------------------

pub struct QTensor {
    dtype: u32,
    data: Arc<Vec<u8>>,
    start: usize,
    end: usize,
    numel: usize,
}

impl QTensor {
    fn dequant_all(&self) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0f32; self.numel];
        quant::dequantize(self.dtype, &self.data[self.start..self.end], self.numel, &mut out)?;
        Ok(out)
    }

    fn matmul(&self, x: &[f32], n_in: usize, n_out: usize) -> Result<Vec<f32>, String> {
        let (bk_elems, bk_bytes) = block_layout(self.dtype)
            .ok_or_else(|| format!("tipe {} belum didukung", self.dtype))?;
        if n_in % bk_elems != 0 {
            return Err(format!("dimensi input {n_in} bukan kelipatan blok {bk_elems}"));
        }
        let col_bytes = n_in / bk_elems * bk_bytes;
        if self.end - self.start < n_out * col_bytes {
            return Err("data bobot tidak cukup untuk matmul".into());
        }
        let mut y = vec![0.0f32; n_out];
        let mut buf = vec![0.0f32; n_in];
        for o in 0..n_out {
            let s = self.start + o * col_bytes;
            quant::dequantize(self.dtype, &self.data[s..s + col_bytes], n_in, &mut buf)?;
            let mut acc = 0.0f32;
            for k in 0..n_in { acc += buf[k] * x[k]; }
            y[o] = acc;
        }
        Ok(y)
    }
}

pub enum Weight {
    F32(Vec<f32>),
    Q(QTensor),
}

impl Weight {
    fn as_f32(&self) -> Result<Cow<'_, [f32]>, String> {
        match self {
            Weight::F32(v) => Ok(Cow::Borrowed(v)),
            Weight::Q(q) => Ok(Cow::Owned(q.dequant_all()?)),
        }
    }

    fn row(&self, idx: usize, len: usize) -> Result<Vec<f32>, String> {
        match self {
            Weight::F32(v) => {
                let s = idx.checked_mul(len).ok_or("row overflow")?;
                Ok(v.get(s..s + len).ok_or("row di luar bobot")?.to_vec())
            }
            Weight::Q(q) => {
                let (bk_elems, bk_bytes) = block_layout(q.dtype)
                    .ok_or_else(|| format!("tipe {} belum didukung", q.dtype))?;
                let row_bytes = len / bk_elems * bk_bytes;
                let s = q.start + idx * row_bytes;
                let mut out = vec![0.0f32; len];
                quant::dequantize(q.dtype, &q.data[s..s + row_bytes], len, &mut out)?;
                Ok(out)
            }
        }
    }

    fn matmul(&self, x: &[f32], n_in: usize, n_out: usize) -> Result<Vec<f32>, String> {
        match self {
            Weight::F32(v) => Ok(matmul(v, x, n_in, n_out)),
            Weight::Q(q) => q.matmul(x, n_in, n_out),
        }
    }
}

// ---------------------------------------------------------------------------
// Operasi matematika
// ---------------------------------------------------------------------------

fn matmul(w: &[f32], x: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), n_in * n_out);
    debug_assert_eq!(x.len(), n_in);
    let mut y = vec![0.0f32; n_out];
    for o in 0..n_out {
        let base = o * n_in;
        let mut s = 0.0f32;
        // Loop paling panas — sengaja tanpa iterator rumit.
        for k in 0..n_in { s += w[base + k] * x[k]; }
        y[o] = s;
    }
    y
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * scale * b).collect()
}

fn rope(v: &mut [f32], head_dim: usize, pos: usize, base: f32) {
    let half = head_dim / 2;
    for i in 0..half {
        let inv_freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * inv_freq;
        let (c, s) = (angle.cos(), angle.sin());
        let a = v[i];
        let b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = b * c + a * s;
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for v in x.iter_mut() { *v = (*v - max).exp(); }
    let sum: f32 = x.iter().sum();
    if sum > 0.0 { for v in x.iter_mut() { *v /= sum; } }
}

pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(cfg: &Config) -> Self {
        let per_layer = cfg.ctx_len * cfg.n_kv_heads * cfg.head_dim();
        Self {
            k: vec![vec![0.0; per_layer]; cfg.n_layers],
            v: vec![vec![0.0; per_layer]; cfg.n_layers],
        }
    }
}

pub struct Model {
    pub cfg: Config,
    w: HashMap<String, Weight>,
    pub low_mem: bool,
}

impl Model {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> { Self::from_gguf_opts(g, false) }

    pub fn from_gguf_opts(g: &Gguf, low_mem: bool) -> Result<Self, String> {
        let cfg = Config::from_gguf(g)?;
        let mut w = HashMap::new();
        for t in &g.tensors {
            let weight = if low_mem {
                let (data, start, end) = g.tensor_raw(t)?;
                Weight::Q(QTensor {
                    dtype: t.dtype, data, start, end,
                    numel: t.numel() as usize,
                })
            } else {
                Weight::F32(g.tensor_f32(t)?)
            };
            w.insert(t.name.clone(), weight);
        }
        Ok(Self { cfg, w, low_mem })
    }

    fn weight(&self, names: &[&str]) -> Result<&Weight, String> {
        for n in names {
            if let Some(v) = self.w.get(*n) { return Ok(v); }
        }
        Err(format!("tensor tidak ditemukan: {}", names.join(" / ")))
    }

    fn vec(&self, names: &[&str]) -> Result<Cow<'_, [f32]>, String> {
        self.weight(names)?.as_f32()
    }

    /// Jalankan satu token, kembalikan logits vocabulary.
    pub fn forward(&self, token: usize, pos: usize, cache: &mut KvCache) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        if pos >= c.ctx_len { return Err(format!("posisi {pos} melebihi context length {}", c.ctx_len)); }
        let hd = c.head_dim();
        let eps = c.rms_eps;

        let emb = self.weight(&["token_embd.weight", "tok_embeddings.weight"])?;
        let mut x = emb.row(token, c.n_embd)?;

        for l in 0..c.n_layers {
            // ---- Attention ----
            let norm1 = self.vec(&[
                &format!("blk.{l}.attn_norm.weight"),
                &format!("blk.{l}.ln1.weight"),
            ])?;
            let xn = rms_norm(&x, &norm1, eps);

            let qw = self.weight(&[&format!("blk.{l}.attn_q.weight")])?;
            let kw = self.weight(&[&format!("blk.{l}.attn_k.weight")])?;
            let vw = self.weight(&[&format!("blk.{l}.attn_v.weight")])?;
            let mut q = qw.matmul(&xn, c.n_embd, c.n_embd)?;
            let mut k = kw.matmul(&xn, c.n_embd, c.n_kv_heads * hd)?;
            let v = vw.matmul(&xn, c.n_embd, c.n_kv_heads * hd)?;

            for h in 0..c.n_heads { rope(&mut q[h * hd..(h + 1) * hd], hd, pos, c.freq_base); }
            for h in 0..c.n_kv_heads { rope(&mut k[h * hd..(h + 1) * hd], hd, pos, c.freq_base); }

            for h in 0..c.n_kv_heads {
                let dst = (pos * c.n_kv_heads + h) * hd;
                cache.k[l][dst..dst + hd].copy_from_slice(&k[h * hd..(h + 1) * hd]);
                cache.v[l][dst..dst + hd].copy_from_slice(&v[h * hd..(h + 1) * hd]);
            }

            let mut attn_out = vec![0.0f32; c.n_embd];
            let scale = 1.0 / (hd as f32).sqrt();
            let groups = c.n_heads / c.n_kv_heads;
            for h in 0..c.n_heads {
                let kv_h = h / groups;
                let qh = &q[h * hd..(h + 1) * hd];
                let mut scores = vec![0.0f32; pos + 1];
                for t in 0..=pos {
                    let kt = &cache.k[l][(t * c.n_kv_heads + kv_h) * hd..(t * c.n_kv_heads + kv_h + 1) * hd];
                    let mut s = 0.0f32;
                    for d in 0..hd { s += qh[d] * kt[d]; }
                    scores[t] = s * scale;
                }
                softmax(&mut scores);
                let out_h = &mut attn_out[h * hd..(h + 1) * hd];
                for d in 0..hd { out_h[d] = 0.0; }
                for t in 0..=pos {
                    let vt = &cache.v[l][(t * c.n_kv_heads + kv_h) * hd..(t * c.n_kv_heads + kv_h + 1) * hd];
                    let p = scores[t];
                    for d in 0..hd { out_h[d] += p * vt[d]; }
                }
            }

            let ow = self.weight(&[&format!("blk.{l}.attn_output.weight")])?;
            let da = ow.matmul(&attn_out, c.n_embd, c.n_embd)?;
            for i in 0..c.n_embd { x[i] += da[i]; }

            // ---- Feed forward (SwiGLU) ----
            let norm2 = self.vec(&[&format!("blk.{l}.ffn_norm.weight")])?;
            let xn2 = rms_norm(&x, &norm2, eps);
            let gw = self.weight(&[&format!("blk.{l}.ffn_gate.weight")])?;
            let uw = self.weight(&[&format!("blk.{l}.ffn_up.weight")])?;
            let dw = self.weight(&[&format!("blk.{l}.ffn_down.weight")])?;
            let gate = gw.matmul(&xn2, c.n_embd, c.n_ff)?;
            let up = uw.matmul(&xn2, c.n_embd, c.n_ff)?;
            let mut act = vec![0.0f32; c.n_ff];
            for i in 0..c.n_ff {
                let g = gate[i];
                act[i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            let dff = dw.matmul(&act, c.n_ff, c.n_embd)?;
            for i in 0..c.n_embd { x[i] += dff[i]; }
        }

        let final_norm = self.vec(&["output_norm.weight", "model.norm.weight"])?;
        x = rms_norm(&x, &final_norm, eps);
        let out_w = self.weight(&["output.weight", "lm_head.weight"])?;
        out_w.matmul(&x, c.n_embd, c.n_vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tiny_fixture;

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1] && v[1] > v[0]);
    }

    #[test]
    fn rope_magnitude_preserved() {
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0];
        rope(&mut v, 4, 3, 10000.0);
        let mag: f32 = v.iter().map(|x| x * x).sum();
        assert!((mag - 30.0).abs() < 1e-4, "{mag}");
    }

    #[test]
    fn matmul_shape() {
        let w: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let y = matmul(&w, &[1.0, 2.0], 2, 3);
        assert_eq!(y, vec![2.0, 8.0, 14.0]);
    }

    #[test]
    fn low_mem_matches_f32_logits() {
        let bytes = tiny_fixture();
        let g = Gguf::parse(bytes).unwrap();
        let fast = Model::from_gguf(&g).unwrap();
        let lean = Model::from_gguf_opts(&g, true).unwrap();
        let mut c1 = KvCache::new(&fast.cfg);
        let mut c2 = KvCache::new(&lean.cfg);
        let a = fast.forward(1, 0, &mut c1).unwrap();
        let b = lean.forward(1, 0, &mut c2).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-4, "fast={x}, low_mem={y}");
        }
    }
}

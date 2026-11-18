//! Forward pass Qwen2 (dense) murni Rust: RMSNorm, RoPE, GQA attention, SwiGLU.
//! Bobot di-dekuantisasi sekali saat load, lalu disimpan sebagai f32.

use crate::gguf::Gguf;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Config {
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
        Ok(Self { n_layers, n_heads, n_kv_heads, n_embd, n_ff, n_vocab: vocab, ctx_len, freq_base, rms_eps })
    }

    pub fn head_dim(&self) -> usize { self.n_embd / self.n_heads }
}

pub struct Model {
    pub cfg: Config,
    w: HashMap<String, Vec<f32>>,
}

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

impl Model {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let cfg = Config::from_gguf(g)?;
        let mut w = HashMap::new();
        for t in &g.tensors {
            w.insert(t.name.clone(), g.tensor_f32(t)?);
        }
        Ok(Self { cfg, w })
    }

    fn tensor(&self, names: &[&str]) -> Result<&[f32], String> {
        for n in names {
            if let Some(v) = self.w.get(*n) { return Ok(v); }
        }
        Err(format!("tensor tidak ditemukan: {}", names.join(" / ")))
    }

    /// Jalankan satu token, kembalikan logits vocabulary.
    pub fn forward(&self, token: usize, pos: usize, cache: &mut KvCache) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        if pos >= c.ctx_len { return Err(format!("posisi {pos} melebihi context length {}", c.ctx_len)); }
        let hd = c.head_dim();
        let eps = c.rms_eps;

        let emb = self.tensor(&["token_embd.weight", "tok_embeddings.weight"])?;
        let mut x = emb[token * c.n_embd..(token + 1) * c.n_embd].to_vec();

        for l in 0..c.n_layers {
            // ---- Attention ----
            let norm1 = self.tensor(&[
                &format!("blk.{l}.attn_norm.weight"),
                &format!("blk.{l}.ln1.weight"),
            ])?;
            let xn = rms_norm(&x, norm1, eps);

            let qw = self.tensor(&[&format!("blk.{l}.attn_q.weight"), &format!("blk.{l}.attn_q.w")])?;
            let kw = self.tensor(&[&format!("blk.{l}.attn_k.weight"), &format!("blk.{l}.attn_k.w")])?;
            let vw = self.tensor(&[&format!("blk.{l}.attn_v.weight"), &format!("blk.{l}.attn_v.w")])?;
            let mut q = matmul(qw, &xn, c.n_embd, c.n_embd);
            let mut k = matmul(kw, &xn, c.n_embd, c.n_kv_heads * hd);
            let v = matmul(vw, &xn, c.n_embd, c.n_kv_heads * hd);

            for h in 0..c.n_heads { rope(&mut q[h * hd..(h + 1) * hd], hd, pos, c.freq_base); }
            for h in 0..c.n_kv_heads { rope(&mut k[h * hd..(h + 1) * hd], hd, pos, c.freq_base); }

            // Tulis K/V ke cache.
            for h in 0..c.n_kv_heads {
                let dst = (pos * c.n_kv_heads + h) * hd;
                cache.k[l][dst..dst + hd].copy_from_slice(&k[h * hd..(h + 1) * hd]);
                cache.v[l][dst..dst + hd].copy_from_slice(&v[h * hd..(h + 1) * hd]);
            }

            // Attention per head (GQA: beberapa q-head berbagi satu kv-head).
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

            let ow = self.tensor(&[&format!("blk.{l}.attn_output.weight")])?;
            let da = matmul(ow, &attn_out, c.n_embd, c.n_embd);
            for i in 0..c.n_embd { x[i] += da[i]; }

            // ---- Feed forward (SwiGLU) ----
            let norm2 = self.tensor(&[&format!("blk.{l}.ffn_norm.weight")])?;
            let xn2 = rms_norm(&x, norm2, eps);
            let gw = self.tensor(&[&format!("blk.{l}.ffn_gate.weight")])?;
            let uw = self.tensor(&[&format!("blk.{l}.ffn_up.weight")])?;
            let dw = self.tensor(&[&format!("blk.{l}.ffn_down.weight")])?;
            let gate = matmul(gw, &xn2, c.n_embd, c.n_ff);
            let up = matmul(uw, &xn2, c.n_embd, c.n_ff);
            let mut act = vec![0.0f32; c.n_ff];
            for i in 0..c.n_ff {
                let g = gate[i];
                act[i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            let dff = matmul(dw, &act, c.n_ff, c.n_embd);
            for i in 0..c.n_embd { x[i] += dff[i]; }
        }

        let final_norm = self.tensor(&["output_norm.weight", "model.norm.weight"])?;
        x = rms_norm(&x, final_norm, eps);
        let out_w = self.tensor(&["output.weight", "lm_head.weight"])?;
        Ok(matmul(out_w, &x, c.n_embd, c.n_vocab))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let w: Vec<f32> = (0..6).map(|x| x as f32).collect(); // [2 input, 3 output]
        let y = matmul(&w, &[1.0, 2.0], 2, 3);
        assert_eq!(y, vec![2.0, 8.0, 14.0]);
    }
}

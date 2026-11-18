#![allow(dead_code)] // Bagian dari API yang dipakai server/IDE nanti.

//! Tokenizer byte-level BPE untuk Qwen2 dari metadata GGUF.
//! Tidak memakai tokenizer.json / library eksternal.

use crate::gguf::Gguf;
use std::collections::HashMap;

pub struct Tokenizer {
    tokens: Vec<String>,
    id_of: HashMap<String, usize>,
    ranks: HashMap<(String, String), usize>,
    /// Pemetaan byte (0..255) -> karakter unicode seperti GPT-2.
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub eos_id: usize,
}

/// bytes_to_unicode dari GPT-2 (dipakai Qwen2).
fn byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(0x21..=0x7e);   // ! .. ~
    bs.extend(0xa1..=0xac);   // ¡ .. ¬
    bs.extend(0xae..=0xff);   // ® .. ÿ
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut b2c = ['\0'; 256];
    let mut c2b = HashMap::new();
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).unwrap_or('\u{fffd}');
        b2c[b as usize] = ch;
        c2b.insert(ch, b as u8);
    }
    (b2c, c2b)
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let tokens = g.meta_string_array("tokenizer.ggml.tokens")
            .ok_or("metadata tokenizer.ggml.tokens tidak ada")?;
        if tokens.is_empty() { return Err("daftar token kosong".into()); }
        let mut id_of = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() { id_of.insert(t.clone(), i); }

        let mut ranks = HashMap::new();
        if let Some(merges) = g.meta_string_array("tokenizer.ggml.merges") {
            for (rank, m) in merges.iter().enumerate() {
                let mut it = m.split_whitespace();
                if let (Some(a), Some(b)) = (it.next(), it.next()) {
                    ranks.insert((a.to_string(), b.to_string()), rank);
                }
            }
        }

        let (byte_to_char, char_to_byte) = byte_maps();
        let eos_id = ["<|endoftext|>", "<|im_end|>"]
            .iter().find_map(|s| id_of.get(*s).copied())
            .unwrap_or(tokens.len() - 1);

        Ok(Self { tokens, id_of, ranks, byte_to_char, char_to_byte, eos_id })
    }

    pub fn vocab_size(&self) -> usize { self.tokens.len() }

    /// Id dari token khusus (mis. `<|im_start|>`).
    pub fn special(&self, s: &str) -> Result<usize, String> {
        self.id_of.get(s).copied().ok_or_else(|| format!("token khusus tidak ada: {s}"))
    }

    /// Ubah teks biasa (bukan token khusus) jadi id via byte-BPE.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mapped: String = text.as_bytes().iter()
            .map(|&b| self.byte_to_char[b as usize])
            .collect();
        let mut parts: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map_or(true, |(br, _)| r < br) { best = Some((r, i)); }
                }
            }
            match best {
                Some((_, i)) => {
                    let merged = format!("{}{}", parts[i], parts[i + 1]);
                    parts[i] = merged;
                    parts.remove(i + 1);
                }
                None => break,
            }
        }
        parts.iter().filter_map(|p| self.id_of.get(p).copied()).collect()
    }

    /// Id -> potongan teks (dipakai untuk decode output).
    pub fn piece(&self, id: usize) -> &str {
        self.tokens.get(id).map(String::as_str).unwrap_or("")
    }

    /// Decode deretan id menjadi teks UTF-8, membalik pemetaan byte BPE.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let s = self.piece(id);
            if s.starts_with("<|") && s.ends_with("|>") { continue; }
            for ch in s.chars() {
                if let Some(&b) = self.char_to_byte.get(&ch) {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Bangun prompt ChatML seperti format pelatihan KevAI.
    pub fn chat_prompt(&self, system: &str, history: &[(String, String)]) -> Result<Vec<usize>, String> {
        let start = self.special("<|im_start|>")?;
        let end = self.special("<|im_end|>")?;
        let mut ids = Vec::new();
        let push_text = |ids: &mut Vec<usize>, role: &str, text: &str| {
            ids.push(start);
            ids.extend(self.encode(role));
            ids.push(self.newline());
            ids.extend(self.encode(text));
            ids.push(end);
            ids.push(self.newline());
        };
        push_text(&mut ids, "system", system);
        for (u, a) in history {
            push_text(&mut ids, "user", u);
            push_text(&mut ids, "assistant", a);
        }
        // Mulai giliran assistant berikutnya.
        ids.push(start);
        ids.extend(self.encode("assistant"));
        ids.push(self.newline());
        Ok(ids)
    }

    fn newline(&self) -> usize {
        self.encode("\n").first().copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_maps_are_inverse() {
        let (b2c, c2b) = byte_maps();
        for b in 0u16..256 {
            let b = b as u8;
            assert_eq!(c2b[&b2c[b as usize]], b);
        }
    }
}

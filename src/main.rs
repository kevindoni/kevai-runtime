//! KevAI Runtime — mesin inferensi GGUF mandiri (tanpa Ollama / llama.cpp).
//!
//! Tahap saat ini: parser GGUF + dekuantisasi + inspeksi model.
//! Tahap berikutnya: tokenizer, forward Qwen2, sampling, server OpenAI-compatible.

mod gguf;
mod json;
mod model;
mod quant;
mod sampler;
mod server;
mod tokenizer;

use gguf::{demo_fixture, tiny_fixture, Gguf, Value};
use model::{KvCache, Model};
use sampler::Sampler;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use tokenizer::Tokenizer;

fn human_bytes(n: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut x = n as f64;
    let mut i = 0;
    while x >= 1024.0 && i < U.len() - 1 { x /= 1024.0; i += 1; }
    format!("{:.2} {}", x, U[i])
}

fn fmt_value(v: &Value) -> String {
    match v {
        Value::Array(a) => {
            let preview: Vec<String> = a.iter().take(5).map(fmt_value).collect();
            let more = if a.len() > 5 { format!(", … (+{})", a.len() - 5) } else { String::new() };
            format!("[{} items: {}{}]", a.len(), preview.join(", "), more)
        }
        Value::String(s) if s.len() > 160 => format!("{}…", &s[..160]),
        Value::String(s) => s.clone(),
        _ => format!("{:?}", v),
    }
}

fn inspect(path: &Path, show_tensors: bool) -> Result<(), String> {
    let g = Gguf::from_file(path)?;
    println!("=== KevAI GGUF Inspector ===");
    println!("file       : {}", path.display());
    println!("versi GGUF : {}", g.version);
    println!("arsitektur : {}", g.architecture());
    println!("data start : {} ({:#x})", g.data_offset, g.data_offset);

    if let Some(n) = g.meta_u64("general.file_type") {
        println!("file type  : {n}");
    }
    for key in [
        "qwen2.block_count", "qwen2.attention.head_count",
        "qwen2.attention.head_count_kv", "qwen2.embedding_length",
        "qwen2.feed_forward_length", "qwen2.context_length",
        "qwen2.rope.freq_base", "qwen2.attention.layer_norm_rms_epsilon",
    ] {
        if let Some(v) = g.meta(key) {
            println!("{key:<38}: {}", fmt_value(v));
        }
    }

    println!("\n--- Metadata ({} entri) ---", g.metadata.len());
    let mut entries: Vec<_> = g.metadata.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, v) in entries {
        //Tokenizer bisa sangat besar; ringkas kecuali diminta.
        if matches!(v, Value::Array(a) if a.len() > 8) {
            println!("{k:<38}: [{} items: {}]", v.as_array().unwrap().len(),
                v.as_array().unwrap().iter().take(3).map(fmt_value).collect::<Vec<_>>().join(", "));
        } else {
            println!("{k:<38}: {}", fmt_value(v));
        }
    }

    println!("\n--- Tensor ({} buah) ---", g.tensors.len());
    let mut total_elems = 0u64;
    let mut total_bytes = 0u64;
    let mut per_type: std::collections::BTreeMap<&str, (u64, u64)> = std::collections::BTreeMap::new();
    for t in &g.tensors {
        let nb = t.nbytes()?;
        total_elems += t.numel();
        total_bytes += nb;
        let e = per_type.entry(t.dtype_name()).or_insert((0, 0));
        e.0 += t.numel();
        e.1 += nb;
        if show_tensors {
            println!("{:<52} {:<6} {:>8}  {:>14}", t.name, t.dtype_name(), t.numel(), human_bytes(nb));
        }
    }
    if !show_tensors {
        println!("(gunakan --tensors untuk daftar lengkap)");
    }
    println!("\nparam total   : {} ({:.3} M)", total_elems, total_elems as f64 / 1e6);
    println!("ukuran bobot  : {}", human_bytes(total_bytes));
    for (ty, (n, b)) in per_type {
        println!("  {:<8} {:>12} elemen  {:>12}", ty, n, human_bytes(b));
    }
    Ok(())
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned()
}

fn chat(path: &Path, system: &str, max_tokens: usize, temp: f32, low_mem: bool) -> Result<(), String> {
    println!("[1/3] Membaca GGUF: {}", path.display());
    let g = Gguf::from_file(path)?;
    println!("[2/3] Menyiapkan tokenizer & model...");
    let tok = Tokenizer::from_gguf(&g)?;
    let model = Model::from_gguf_opts(&g, low_mem)?;
    let mut cache = KvCache::new(&model.cfg);
    let mut sampler = Sampler::new(0x4b45564149);
    sampler.temperature = temp;
    println!("[3/3] Siap — {} {}, {} layer, {} head, {} vocab, ctx {}.{}",
        model.cfg.arch, if model.low_mem { "(low-mem)" } else { "" },
        model.cfg.n_layers, model.cfg.n_heads, model.cfg.n_vocab, model.cfg.ctx_len,
        if model.low_mem { " Bobot tetap terkuantisasi di RAM." } else { "" });
    println!("Ketik 'exit' untuk keluar. System: \"{system}\"\n");

    let stdin = io::stdin();
    let end_id = tok.special("<|im_end|>").ok();
    let mut history: Vec<(String, String)> = Vec::new();

    loop {
        print!("Kamu> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 { break; }
        let user = line.trim().to_string();
        if user.eq_ignore_ascii_case("exit") || user.eq_ignore_ascii_case("quit") { break; }
        if user.is_empty() { continue; }

        let prompt = tok.chat_prompt(system, &history)?;
        if prompt.len() + max_tokens > model.cfg.ctx_len {
            return Err(format!("prompt {} token + {} max melebihi ctx {}", prompt.len(), max_tokens, model.cfg.ctx_len));
        }

        let mut logits = Vec::new();
        for (i, &id) in prompt.iter().enumerate() {
            logits = model.forward(id, i, &mut cache)?;
        }

        print!("KevAI> ");
        io::stdout().flush().ok();
        let mut out_ids: Vec<usize> = Vec::new();
        let mut printed_chars = 0usize;
        for _ in 0..max_tokens {
            let id = sampler.sample(&logits);
            if Some(id) == end_id || id == tok.eos_id { break; }
            out_ids.push(id);
            let text = tok.decode(&out_ids);
            let chars: Vec<char> = text.chars().collect();
            if chars.len() > printed_chars {
                let baru: String = chars[printed_chars..].iter().collect();
                print!("{baru}");
                io::stdout().flush().ok();
                printed_chars = chars.len();
            }
            let pos = prompt.len() + out_ids.len() - 1;
            if pos >= model.cfg.ctx_len { break; }
            logits = model.forward(id, pos, &mut cache)?;
        }
        println!();
        history.push((user, tok.decode(&out_ids)));
    }
    Ok(())
}

fn help() {
    println!("KevAI Runtime v{} — inferensi GGUF mandiri", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Pemakaian:");
    println!("  kevai-runtime inspect <model.gguf> [--tensors]");
    println!("  kevai-runtime fixture <out.gguf>       # contoh parser (bukan model)");
    println!("  kevai-runtime tiny <out.gguf>          # model Qwen2 mini untuk test pipeline");
    println!("  kevai-runtime chat <model.gguf> [--system teks] [--max-tokens N] [--temp F] [--low-mem]");
    println!("  kevai-runtime serve <model.gguf> [--addr 127.0.0.1:8080] [--name kevai] [--low-mem]");
    println!("  kevai-runtime --help");
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        help();
        return;
    }
    let result = match args[0].as_str() {
        "inspect" => {
            if args.len() < 2 {
                Err("inspect butuh path file .gguf".to_string())
            } else {
                inspect(Path::new(&args[1]), args.iter().any(|a| a == "--tensors"))
            }
        }
        "fixture" => {
            if args.len() < 2 {
                Err("fixture butuh path tujuan".to_string())
            } else {
                fs::write(&args[1], demo_fixture())
                    .map_err(|e| format!("gagal menulis {}: {e}", args[1]))
                    .map(|_| println!("file contoh ditulis: {}", args[1]))
            }
        }
        "tiny" => {
            if args.len() < 2 {
                Err("tiny butuh path tujuan".to_string())
            } else {
                fs::write(&args[1], tiny_fixture())
                    .map_err(|e| format!("gagal menulis {}: {e}", args[1]))
                    .map(|_| println!("model Qwen2 mini ditulis: {}", args[1]))
            }
        }
        "chat" => {
            if args.len() < 2 {
                Err("chat butuh path file .gguf".to_string())
            } else {
                let system = arg_value(&args, "--system")
                    .unwrap_or_else(|| "Anda adalah KevAI, asisten pribadi Kevin. Jawab dalam Bahasa Indonesia yang santai dan jelas.".into());
                let max_tokens = arg_value(&args, "--max-tokens")
                    .and_then(|v| v.parse().ok()).unwrap_or(256usize);
                let temp = arg_value(&args, "--temp")
                    .and_then(|v| v.parse().ok()).unwrap_or(0.8f32);
                let low_mem = args.iter().any(|a| a == "--low-mem");
                chat(Path::new(&args[1]), &system, max_tokens, temp, low_mem)
            }
        }
        "serve" => {
            if args.len() < 2 {
                Err("serve butuh path file .gguf".to_string())
            } else {
                let addr = arg_value(&args, "--addr").unwrap_or_else(|| "127.0.0.1:8080".into());
                let name = arg_value(&args, "--name").unwrap_or_else(|| "kevai".into());
                let system = arg_value(&args, "--system")
                    .unwrap_or_else(|| "Anda adalah KevAI, asisten pribadi Kevin. Jawab dalam Bahasa Indonesia yang santai dan jelas.".into());
                let low_mem = args.iter().any(|a| a == "--low-mem");
                server::serve(Path::new(&args[1]), &addr, system, name, low_mem)
            }
        }
        other => Err(format!("perintah tidak dikenal: {other}")),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

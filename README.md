# KevAI Runtime

Mesin inferensi GGUF mandiri untuk KevAI — **tanpa Ollama, tanpa llama.cpp saat runtime, tanpa crate pihak ketiga**.
Ditulis dengan Rust std saja (`cargo build` tidak mengunduh dependency apa pun).

## Kenapa ini ada

Repo `kevindoni/kevai-open` sudah melatih adapter LoRA di atas Qwen2.5-1.5B-Instruct dan
berencana menghasilkan file GGUF. KevAI Runtime mengambil file GGUF itu dan menjalankannya
langsung: parser GGUF, dekuantisasi, tokenizer byte-BPE, forward Qwen2, sampling, dan server
OpenAI-compatible — semuanya kode sendiri.

## Build

```powershell
cd D:\PALIKASIKU\AiKoding\kevai-runtime
cargo build --release
```

Binary: `target\release\kevai-runtime.exe`.

> Catatan: README AiKoding IDE menyarankan toolchain MSVC untuk proyek Tauri. Crate ini
> berdiri sendiri dan sudah teruji dengan Rust GNU (EnvKit) maupun MSVC.

## Pemakaian

```powershell
# Inspeksi model (metadata, config, daftar tensor, ukuran bobot)
kevai-runtime inspect model.gguf --tensors

# Buat fixture untuk mencoba tanpa model asli
kevai-runtime tiny tiny.gguf
kevai-runtime chat tiny.gguf --max-tokens 32

# Chat dengan model asli (cepat, butuh RAM ±4x ukuran file)
kevai-runtime chat kevai-edge-Q4_K_M.gguf --max-tokens 256 --temp 0.8

# Mode hemat memori: bobot tetap terkuantisasi (RAM ≈ ukuran file, lebih lambat)
kevai-runtime chat kevai-edge-Q4_K_M.gguf --low-mem --max-tokens 256

# Server OpenAI-compatible
kevai-runtime serve kevai-edge-Q4_K_M.gguf --addr 127.0.0.1:8080 --name kevai
kevai-runtime serve kevai-edge-Q4_K_M.gguf --addr 127.0.0.1:8080 --low-mem
```

Endpoint server:

| Method | Path | Fungsi |
|---|---|---|
| GET | `/v1/models` | daftar model |
| GET | `/health` | health check |
| POST | `/v1/chat/completions` | chat (streaming SSE didukung) |

Contoh request:

```powershell
$body = @{
  messages = @(@{ role = "user"; content = "ringkas 3 poin: hujan" })
  max_tokens = 128
  temperature = 0.8
  stream = $true
} | ConvertTo-Json -Depth 5

curl.exe -s -N -H "Content-Type: application/json" -d $body `
  http://127.0.0.1:8080/v1/chat/completions
```

## Pasang ke AiKoding IDE

Panel AI AiKoding sudah mendukung endpoint OpenAI-compatible:

1. Jalankan `kevai-runtime serve ...` (biarkan jalan di latar belakang).
2. Di AiKoding IDE, buka pengaturan AI / endpoint.
3. Set base URL: `http://127.0.0.1:8080/v1`
4. Set model: `kevai` (atau nilai `--name` yang dipakai).
5. API key bebas diisi apa saja (server tidak memeriksa).

Sekarang panel AI memakai model KevAI lokal — tidak ada Ollama, tidak ada API cloud.

## Arsitektur kode

| File | Isi |
|---|---|
| `src/gguf.rs` | parser GGUF v2/v3: header, metadata, info tensor, validasi offset |
| `src/quant.rs` | dekuantisasi F32/F16/BF16, Q4_0/1, Q5_0/1, Q8_0/1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K |
| `src/tokenizer.rs` | byte-level BPE dari metadata GGUF + template ChatML |
| `src/model.rs` | RMSNorm, RoPE, attention GQA + KV cache, SwiGLU, logits |
| `src/sampler.rs` | greedy / temperature / top-k / top-p (RNG xorshift64) |
| `src/json.rs` | parser JSON minimal untuk request chat |
| `src/server.rs` | HTTP + SSE dengan `std::net::TcpListener`, multithread |

## Batasan saat ini

- Arsitektur **Qwen2 dense dan Llama dense** (belum MoE / Gemma).
- Dua mode memori:
  - **default** — bobot di-dekuantisasi ke `f32` saat load: paling cepat, RAM ±4× ukuran file
    (1.5B Q4_K ≈ 1 GB file → ±6 GB RAM).
  - **`--low-mem`** — bobot tetap terkuantisasi, dekuantisasi per kolom saat matmul:
    RAM ≈ ukuran file GGUF, tapi inferensi lebih lambat. Cocok untuk laptop RAM 8 GB.
  - Keduanya sudah diuji menghasilkan logits identik (tes `low_mem_matches_f32_logits`).
- Inferensi CPU single-thread per permintaan (belum SIMD/rayon/GPU).
- `tokenizer.chat_template` (Jinja) belum dieksekusi; template ChatML mengikuti format
  pelatihan `train_lora_local.py`.

## Roadmap

- [x] Parser GGUF + dekuantisasi
- [x] Tokenizer + forward Qwen2 + sampling + chat CLI
- [x] Server OpenAI-compatible + SSE
- [ ] Integrasi ke panel AI AiKoding IDE (pengaturan endpoint)
- [x] Bobot quantized di memory (`--low-mem`)
- [x] Workflow GitHub Actions: training -> merge -> GGUF artifact
- [x] Dukungan arsitektur Llama dense
- [ ] matmul SIMD / dukungan Qwen2-MoE

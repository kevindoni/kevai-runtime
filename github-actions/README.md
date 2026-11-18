# Workflow GitHub Actions untuk `kevindoni/kevai-open`

Folder ini berisi versi revisi dari workflow training repo KevAI. Saya belum bisa push
langsung ke GitHub (butuh kredensial Anda), jadi salin file-file ini ke repo `kevai-open`:

```
kevai-open/
├─ train_lora_local.py        <- ganti dengan github-actions/train_lora_local.py
└─ .github/workflows/
   └─ train.yml               <- ganti dengan github-actions/train.yml
```

## Apa yang diperbaiki

1. **Dataset 514, bukan 387.** README lama menyebut target `387/387`, tapi
   `model-latihan.jsonl` sekarang berisi **514 entri**. Skrip baru menghitung jumlah
   entri dari file dan menulisnya ke `work/dataset_count.txt`. Perbarui angka di README
   repo supaya tidak menyesatkan.
2. **`use_cache=False` saat gradient checkpointing.** Tanpa ini, Trainer bisa error
   atau boros memori: `model.config.use_cache = False` sebelum
   `gradient_checkpointing_enable()`, lalu dikembalikan ke `True` sebelum merge.
3. **Gate DONE benar-benar dipakai.** `work/DONE` hanya ditulis setelah
   `save_pretrained()` berhasil. Run berikutnya yang melihat DONE + `merged/config.json`
   langsung keluar (tidak latih ulang).
4. **Konversi GGUF otomatis.** Begitu DONE, workflow meng-clone llama.cpp, menjalankan
   `convert_hf_to_gguf.py --outtype q4_k_m`, lalu upload artifact bernama
   **`kevai-edge-gguf`** — sesuai janji README lama yang belum terwujud.
5. **Chain 24/7 lebih sederhana & andal.** `workflow_run` tidak bisa memicu
   `workflow_run` lagi (pembatasan GitHub), jadi dipakai `schedule` tiap 5 jam +
   `actions/cache` untuk resume. Job timeout 350 menit, jeda 10 menit, lalu run
   berikutnya mengambil checkpoint terakhir.

## Catatan penting

- `llama.cpp` dipakai **hanya di CI sebagai alat konversi**, bukan sebagai runtime.
  Runtime Anda tetap `kevai-runtime` (Rust murni, tanpa dependency).
- Setelah artifact `kevai-edge-gguf` tersedia, unduh `.gguf`-nya lalu:

  ```powershell
  kevai-runtime inspect kevai-edge-Q4_K_M.gguf --tensors
  kevai-runtime serve kevai-edge-Q4_K_M.gguf --addr 127.0.0.1:8080
  ```

- Kalau ingin benar-benar tanpa llama.cpp juga di CI, langkah berikutnya adalah
  menulis konverter HF safetensors -> GGUF sendiri (bisa ditambahkan ke crate Rust).

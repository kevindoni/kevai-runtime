# KevAI-OPEN

Training LoRA otak kecil KevAI 24/7 di GitHub Actions (ubuntu CPU gratis, repo public = menit tanpa batas).

## Rantai 24/7
- Job hingga 350 menit: training + checkpoint tiap 60 step (`actions/cache`, resume otomatis).
- Job berikutnya dijadwalkan tiap 5 jam (cron) dan melanjutkan dari checkpoint terakhir.
  (`workflow_run` tidak bisa memicu `workflow_run` lagi, jadi dipakai `schedule`.)
- Setelah seluruh dataset tuntas: penanda `work/DONE` ditulis -> merge adapter ->
  konversi GGUF `q4_k_m` -> artifact **`kevai-edge-gguf`**.
- Penanda DONE juga jadi gate: run berikutnya tidak akan melatih ulang.

## Dataset
`model-latihan.jsonl` sekarang berisi **514 entri** (README lama menyebut 387 — sudah
diperbarui). Jumlah entri dihitung otomatis oleh trainer dan ditulis ke
`work/dataset_count.txt`.

## Privasi
Dataset & kepribadian KevAI PUBLIC di repo ini — keputusan pemilik proyek.

## Runtime mandiri
Model GGUF dari repo ini dijalankan oleh **[kevai-runtime](https://github.com/kevindoni/kevai-runtime)**:
runtime Rust murni tanpa Ollama / llama.cpp saat inferensi, lengkap dengan server
OpenAI-compatible untuk panel AI AiKoding IDE.

```powershell
kevai-runtime serve kevai-edge-Q4_K_M.gguf --addr 127.0.0.1:8080
# lalu arahkan AiKoding IDE ke http://127.0.0.1:8080/v1
```

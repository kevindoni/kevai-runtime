# ================================================================
# KevAI-OPEN TRAINER v2 — LoRA CPU di GitHub Actions
# Perubahan dari v1:
#  - keluar anggun kalau work/DONE sudah ada (tuntas)
#  - model.config.use_cache = False saat gradient checkpointing
#  - jumlah entri dataset dihitung dari file (tidak hardcode 387)
#  - penanda DONE ditulis SETELAH model berhasil disimpan
# ================================================================
import json
import os
import sys

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

BASE = os.environ.get("KEVAI_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATA = "model-latihan.jsonl"
WORK = os.environ.get("KEVAI_WORK", "work")
ADAPTER_DIR = os.path.join(WORK, "adapter")
MERGED_DIR = os.path.join(WORK, "merged")
DONE = os.path.join(WORK, "DONE")
EPOCHS = 3

# [0/6] Gate: kalau sudah tuntas, jangan latih ulang.
if os.path.isfile(DONE) and os.path.isfile(os.path.join(MERGED_DIR, "config.json")):
    print("[0/6] DONE sudah ada dan model merged valid — skip training.")
    sys.exit(0)

os.makedirs(WORK, exist_ok=True)

# [1/6] dataset
print("[1/6] Memuat dataset:", DATA)
data = []
with open(DATA, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line)["messages"])
print("     entri:", len(data))
if not data:
    raise SystemExit("Dataset kosong")

with open(os.path.join(WORK, "dataset_count.txt"), "w", encoding="utf-8") as f:
    f.write(str(len(data)))


def to_text(msgs):
    t = ""
    for m in msgs:
        role = m["role"]
        if role == "system":
            t += "<|im_start|>system\n" + m["content"] + "<|im_end|>\n"
        elif role == "user":
            t += "<|im_start|>user\n" + m["content"] + "<|im_end|>\n"
        else:
            t += "<|im_start|>assistant\n" + m["content"] + "<|im_end|>\n"
    return t + "<|im_start|>assistant\n"


texts = [to_text(m) for m in data]

# [2/6] tokenizer + base model
print("[2/6] Memuat tokenizer + base model:", BASE)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    device_map="cpu",
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

enc = tok(texts, truncation=True, max_length=512, padding=False)
ds = Dataset.from_dict({"input_ids": enc["input_ids"]})


def add_labels(batch):
    batch["labels"] = [
        [(tid if tid != tok.pad_token_id else -100) for tid in seq]
        for seq in batch["input_ids"]
    ]
    return batch


ds = ds.map(add_labels, batched=True)

# [3/6] LoRA adapter
print("[3/6] Memasang LoRA adapter...")
lcfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
model.enable_input_require_grads()
model.config.use_cache = False          # wajib saat gradient checkpointing
model.gradient_checkpointing_enable()

args = TrainingArguments(
    output_dir=ADAPTER_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=30,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=60,
    save_total_limit=2,
    optim="adamw_torch",
    report_to=[],
    seed=42,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tok, padding=True),
)

# [4/6] training dengan auto-resume dari checkpoint
print("[4/6] TRAINING — resume otomatis dari checkpoint...")
ckpt = None
if os.path.isdir(ADAPTER_DIR):
    cands = [d for d in os.listdir(ADAPTER_DIR) if d.startswith("checkpoint-")]
    if cands:
        ckpt = os.path.join(ADAPTER_DIR, max(cands, key=lambda x: int(x.split("-")[-1])))
        print("     lanjut dari:", ckpt)
trainer.train(resume_from_checkpoint=ckpt)

# [5/6] merge + simpan
print("[5/6] Merge adapter + simpan model KevAI-Edge...")
model.config.use_cache = True
model = model.merge_and_unload()
model = model.to(torch.float16)
os.makedirs(MERGED_DIR, exist_ok=True)
model.save_pretrained(MERGED_DIR, safe_serialization=True)
tok.save_pretrained(MERGED_DIR, safe_serialization=True)

# [6/6] penanda DONE — ditulis paling akhir
with open(DONE, "w", encoding="utf-8") as f:
    f.write("done")

print("SELESAI — model KevAI-Edge siap dikonversi ke GGUF")

"""
=============================================================================
Myanmar NER - XLM-RoBERTa Training Script
=============================================================================
Key differences from BiLSTM train.py:
  • Layer-wise LR: encoder=2e-5, CRF head=1e-3
  • Linear warmup (10%) + cosine decay
  • compute_loss / decode receive word_tag_ids and word_lengths
    (required after the first-piece extraction fix in xlmr_model.py)
=============================================================================
"""

import os
import json
import time
import math
import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from seqeval.metrics import classification_report, f1_score

from xlmr_dataset_loader import get_xlmr_dataloaders, IGNORE_IDX
from xlmr_model import build_xlmr_model


# LR scheduler: linear warmup + cosine decay
def get_warmup_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# Evaluation
@torch.no_grad()
def evaluate(model, loader, idx_to_tag, device):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_ids      = batch["label_ids"].to(device)
        word_tag_ids   = batch["word_tag_ids"]       # CPU
        word_lengths   = batch["word_lengths"]       # CPU

        pred_ids = model.decode(input_ids, attention_mask, label_ids,
                                word_lengths.to(device))

        for i in range(len(pred_ids)):
            length    = word_lengths[i].item()
            gold_ids  = word_tag_ids[i, :length].tolist()
            gold_strs = [idx_to_tag.get(t, "O") for t in gold_ids]
            pred_strs = [idx_to_tag.get(t, "O") for t in pred_ids[i]]
            min_len   = min(len(gold_strs), len(pred_strs))
            all_labels.append(gold_strs[:min_len])
            all_preds.append(pred_strs[:min_len])

    f1     = f1_score(all_labels, all_preds, zero_division=0)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    return f1, report


# Training loop
def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Model name : {config['model_name']}")

    torch.manual_seed(config.get("seed", 42))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.get("seed", 42))

    # ── Vocabulary
    data_dir   = config["data_dir"]
    tag_vocab  = json.load(open(f"{data_dir}/tag_vocab.json", encoding="utf-8"))
    idx_to_tag = {v: k for k, v in tag_vocab.items()}
    config["num_tags"] = len(tag_vocab)
    print(f"NER tags   : {len(tag_vocab)}")

    # ── DataLoaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, _ = get_xlmr_dataloaders(
        data_dir       = data_dir,
        tag_vocab      = tag_vocab,
        tokenizer_name = config["model_name"],
        max_length     = config.get("max_length", 256),
        batch_size     = config["batch_size"],
    )

    # ── Model
    print("\nLoading XLM-RoBERTa...")
    model = build_xlmr_model(config).to(device)

    n_params    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_params  = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Trainable params : {n_params:,}  (encoder {enc_params:,} | head+CRF {n_params-enc_params:,})")

    # ── Layer-wise learning rates
    encoder_lr = config.get("encoder_lr", 2e-5)
    head_lr    = config.get("head_lr",    1e-3)

    optimizer = optim.AdamW(
        [
            {"params": model.encoder.parameters(),    "lr": encoder_lr},
            {"params": model.classifier.parameters(), "lr": head_lr},
            {"params": model.crf.parameters(),        "lr": head_lr},
        ],
        weight_decay=config.get("weight_decay", 1e-2),
    )

    epochs       = config["epochs"]
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    scheduler    = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    print(f"Total steps      : {total_steps:,}  (warmup {warmup_steps})")

    # Training loop
    out_dir    = config.get("output_dir", "checkpoints/xlmr_crf")
    os.makedirs(out_dir, exist_ok=True)
    best_f1    = 0.0
    best_state = None
    no_improve = 0
    patience   = config.get("patience", 5)
    clip       = config.get("grad_clip", 1.0)

    print(f"\n{'─'*60}")
    print(f"  Training for up to {epochs} epochs  (patience={patience})")
    print(f"{'─'*60}")

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids      = batch["label_ids"].to(device)
            word_tag_ids   = batch["word_tag_ids"].to(device)
            word_lengths   = batch["word_lengths"].to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(input_ids, attention_mask,
                                      label_ids, word_tag_ids, word_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        # Validation 
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label_ids      = batch["label_ids"].to(device)
                word_tag_ids   = batch["word_tag_ids"].to(device)
                word_lengths   = batch["word_lengths"].to(device)
                val_loss += model.compute_loss(
                    input_ids, attention_mask, label_ids,
                    word_tag_ids, word_lengths
                ).item()

        val_f1, _ = evaluate(model, val_loader, idx_to_tag, device)

        avg_train = epoch_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        elapsed   = time.time() - t0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_f1"].append(val_f1)

        enc_lr_cur  = optimizer.param_groups[0]["lr"]
        head_lr_cur = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR enc: {enc_lr_cur:.2e} head: {head_lr_cur:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = deepcopy(model.state_dict())
            ckpt_path  = f"{out_dir}/best_model.pt"
            torch.save({"model_state": best_state, "config": config}, ckpt_path)
            print(f"  ✓ New best F1: {best_f1:.4f}  → saved to {ckpt_path}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping (no improvement for {patience} epochs).")
                break

    total_time = time.time() - total_start
    print(f"\n{'─'*60}")
    print(f"Training complete.  Best val F1: {best_f1:.4f}  |  Total time: {total_time:.1f}s")

    # Test evaluation 
    print(f"\n{'='*60}\n  TEST SET EVALUATION\n{'='*60}")
    model.load_state_dict(best_state)
    test_f1, test_report = evaluate(model, test_loader, idx_to_tag, device)
    print(f"\nTest F1: {test_f1:.4f}\n")
    print(test_report)

    json.dump(history, open(f"{out_dir}/history.json", "w"), indent=2)
    with open(f"{out_dir}/test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Test F1: {test_f1:.4f}\n\n{test_report}")
    print(f"\n✓ Results saved to: {out_dir}/")

    return model, history


# Configurations
CONFIGS = {

    "xlmr_base": dict(
        model_name     = "FacebookAI/xlm-roberta-base",
        data_dir       = "data/processed",
        output_dir     = "checkpoints/xlmr_base",
        dropout        = 0.1,
        freeze_encoder = False,
        encoder_lr     = 2e-5,
        head_lr        = 1e-3,
        weight_decay   = 1e-2,
        warmup_ratio   = 0.1,
        batch_size     = 16,
        max_length     = 256,
        grad_clip      = 1.0,
        epochs         = 20,
        patience       = 5,
        seed           = 42,
    ),

    "xlmr_base_frozen": dict(
        model_name     = "FacebookAI/xlm-roberta-base",
        data_dir       = "data/processed",
        output_dir     = "checkpoints/xlmr_base_frozen",
        dropout        = 0.1,
        freeze_encoder = True,
        encoder_lr     = 0.0,
        head_lr        = 1e-3,
        weight_decay   = 1e-2,
        warmup_ratio   = 0.0,
        batch_size     = 32,
        max_length     = 256,
        grad_clip      = 1.0,
        epochs         = 20,
        patience       = 5,
        seed           = 42,
    ),
}


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Myanmar NER with XLM-RoBERTa")
    parser.add_argument("--model", default="xlmr_base", choices=list(CONFIGS.keys()))
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--encoder_lr", type=float, default=None)
    parser.add_argument("--freeze",     action="store_true")
    args = parser.parse_args()

    def _apply(cfg):
        for k in ("data_dir", "output_dir", "epochs", "batch_size", "encoder_lr"):
            v = getattr(args, k)
            if v is not None:
                cfg[k] = v
        if args.freeze:
            cfg["freeze_encoder"] = True
        return cfg

    if args.all:
        results = {}
        for idx, name in enumerate(CONFIGS, 1):
            print(f"\n{'─'*62}\n  [{idx}/{len(CONFIGS)}] {name}\n{'─'*62}")
            cfg = _apply(CONFIGS[name].copy())
            t0 = time.time()
            try:
                _, history = train(cfg)
                best_f1 = max(history["val_f1"]) if history["val_f1"] else 0.0
                results[name] = {"status": "ok", "best_val_f1": round(best_f1, 4),
                                 "elapsed_s": round(time.time()-t0, 1)}
            except Exception as exc:
                results[name] = {"status": f"FAILED: {exc}", "best_val_f1": None,
                                 "elapsed_s": round(time.time()-t0, 1)}
                print(f"\n  ✗ {name} FAILED: {exc}")

        print(f"\n{'='*62}\n  SUMMARY\n{'='*62}")
        for name, r in results.items():
            f1_str = f"{r['best_val_f1']:.4f}" if r["best_val_f1"] else "  —"
            print(f"  {name:<30} F1={f1_str}  {r.get('elapsed_s',0):.0f}s  {r['status']}")
        os.makedirs("checkpoints", exist_ok=True)
        json.dump(results, open("checkpoints/xlmr_run_summary.json", "w"), indent=2)
    else:
        train(_apply(CONFIGS[args.model].copy()))


#python xlmr_train.py --all
#python xlmr_train.py --model xlmr_base_frozen
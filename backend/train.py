"""
=============================================================================
Myanmar NER - STEP 4: Training Script
=============================================================================
Features:
  • Supports all 3 model types (BiLSTM-Softmax, BiLSTM-CRF, BiLSTM-CRF+Char)
  • Optional fastText pretrained embeddings (frozen or fine-tuned)
  • Learning-rate scheduling with ReduceLROnPlateau
  • Early stopping (patience-based)
  • Gradient clipping
  • Best-model checkpointing
  • Full NER evaluation (entity-level F1 via seqeval)
=============================================================================
"""

import os
import json
import time
import argparse
from pathlib import Path
from copy import deepcopy

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from seqeval.metrics import classification_report, f1_score

# local modules
from dataset_loader import (
    MyanmarNERDataset,
    build_char_vocab,
    get_dataloaders,
    load_fasttext_embeddings,
)
from model import build_model


# Evaluation helpers
@torch.no_grad()
def evaluate(
    model,
    loader,
    idx_to_tag: dict,
    device: torch.device,
    use_chars: bool = False,
) -> tuple[float, str]:
    """
    Returns (entity_f1, full_classification_report_string).

    seqeval expects:
        all_preds  : list of sentences, each sentence = list of tag STRINGS
        all_labels : same shape

    Common bugs avoided here:
        - PAD tag (id=0) must be stripped from gold labels using actual length
        - pred[i] from CRF.decode() is already the right length (no padding)
        - batch["lengths"] contains tensors; use .item() before slicing
    """
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        x    = batch["word_ids"].to(device)
        tags = batch["tag_ids"]           # keep on CPU for .tolist()
        lens = batch["lengths"]           # (B,) tensor of actual word counts

        if use_chars:
            xc   = batch["char_ids"].to(device)
            pred_ids = model.decode(x, xc)
        else:
            pred_ids = model.decode(x)    # list[list[int]], already length-trimmed by CRF

        for i in range(len(pred_ids)):
            length = lens[i].item()       # int — actual sentence length (no PAD)

            # Gold: slice to real length, convert ids → tag strings
            gold_ids  = tags[i, :length].tolist()
            gold_strs = [idx_to_tag.get(t, "O") for t in gold_ids]

            # Pred: CRF.decode already returns exactly  tags
            pred_strs = [idx_to_tag.get(t, "O") for t in pred_ids[i]]

            # Safety: if somehow lengths differ, truncate to shorter
            min_len = min(len(gold_strs), len(pred_strs))
            all_labels.append(gold_strs[:min_len])
            all_preds.append(pred_strs[:min_len])

    f1      = f1_score(all_labels, all_preds, zero_division=0)
    report  = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    return f1, report


# Training loop
def train(config: dict):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_chars = config["model_type"] == "bilstm_crf_char"
    print(f"\nDevice : {device}")
    print(f"Model  : {config['model_type']}")
    print(f"Chars  : {use_chars}")

    # ── Load vocabulary
    data_dir = config["data_dir"]
    vocab     = json.load(open(f"{data_dir}/vocab.json",    encoding="utf-8"))
    tag_vocab = json.load(open(f"{data_dir}/tag_vocab.json",encoding="utf-8"))
    idx_to_tag = {v: k for k, v in tag_vocab.items()}

    print(f"\n  Vocabulary     : {len(vocab):,} words")
    print(f"  NER tag set    : {len(tag_vocab)} tags")
    print(f"  Tags           : {sorted(tag_vocab.keys())}")

    # ── Character vocabulary
    char_vocab = None
    if use_chars:
        train_ds_tmp = MyanmarNERDataset(
            f"{data_dir}/train.conll", vocab, tag_vocab
        )
        char_vocab = build_char_vocab(train_ds_tmp)
        print(f"  Char vocabulary: {len(char_vocab)} chars")
        config["char_vocab_size"] = len(char_vocab)

    # ── DataLoaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_dir, vocab, tag_vocab,
        char_vocab  = char_vocab,
        batch_size  = config["batch_size"],
        max_char_len = config.get("max_char_len", 30),
    )

    # ── Class weights (up-weight ORG and PER to fix class imbalance) ─────────
    # Computed from training tag counts using inverse-frequency weighting.
    # Tags listed in `boosted_tags` get an additional multiplier on top.
    if config.get("use_class_weights", True) and config["model_type"] != "bilstm_softmax":
        from collections import Counter as _Counter
        train_ds = _  # datasets dict returned by get_dataloaders — but we need raw tags
        # Re-read train tag counts directly from the dataset loader's sentences
        tag_counter: dict = _Counter()
        train_path = f"{data_dir}/train.conll"
        with open(train_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.rstrip("\n\r")
                if _line.strip() and "\t" in _line:
                    _tag = _line.split("\t")[-1]
                    tag_counter[_tag] += 1

        num_tags   = len(tag_vocab)
        counts     = torch.ones(num_tags, dtype=torch.float)
        for tag_str, idx in tag_vocab.items():
            if tag_str in tag_counter:
                counts[idx] = tag_counter[tag_str]

        # Inverse-frequency weights, normalised so median weight = 1
        inv_freq   = 1.0 / counts
        inv_freq   = inv_freq / inv_freq.median()

        # Extra boost multiplier for specific hard classes
        boost_tags = config.get("boost_tags", {"B-ORG", "I-ORG", "E-ORG", "S-ORG",
                                                "B-PER", "I-PER", "E-PER", "S-PER"})
        boost_factor = config.get("boost_factor", 3.0)
        for tag_str, idx in tag_vocab.items():
            if tag_str in boost_tags:
                inv_freq[idx] *= boost_factor

        inv_freq[0] = 0.0   # PAD tag gets zero weight
        config["class_weights"] = inv_freq
        print(f"\n  Class weights computed (boost x{boost_factor} for: {boost_tags})")
        for tag_str, idx in sorted(tag_vocab.items(), key=lambda x: -inv_freq[x[1]].item())[:8]:
            print(f"    {tag_str:<20} weight={inv_freq[idx]:.3f}")
    else:
        config["class_weights"] = None

    # ── Pretrained embeddings
    pretrained = None
    if config.get("fasttext_path"):
        print(f"\nLoading fastText from: {config['fasttext_path']}")
        pretrained = load_fasttext_embeddings(
            config["fasttext_path"], vocab,
            embedding_dim=config["embedding_dim"]
        ).to(device)

    # ── Build model 
    config["vocab_size"] = len(vocab)
    config["num_tags"]   = len(tag_vocab)
    if config.get("class_weights") is not None:
        config["class_weights"] = config["class_weights"].to(device)
    model = build_model(config, pretrained_embeddings=pretrained).to(device)

    # Optionally freeze embeddings
    if config.get("freeze_embeddings") and pretrained is not None:
        if hasattr(model, "embedding"):
            model.embedding.weight.requires_grad_(False)
        if hasattr(model, "word_embed"):
            model.word_embed.weight.requires_grad_(False)
        print("  Embeddings frozen.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")

    # ── Optimizer & scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    # ── Training
    out_dir = config.get("output_dir", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    best_f1        = 0.0
    best_state     = None
    no_improve     = 0
    patience       = config.get("patience", 5)
    epochs         = config["epochs"]
    clip           = config.get("grad_clip", 5.0)
    total_start    = time.time()

    print(f"\n{'─'*60}")
    print(f"  Training for up to {epochs} epochs  (patience={patience})")
    print(f"{'─'*60}")

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            x    = batch["word_ids"].to(device)
            tags = batch["tag_ids"].to(device)
            optimizer.zero_grad()

            if use_chars:
                xc   = batch["char_ids"].to(device)
                loss = model.compute_loss(x, xc, tags)
            else:
                loss = model.compute_loss(x, tags)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

        # ── Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x    = batch["word_ids"].to(device)
                tags = batch["tag_ids"].to(device)
                if use_chars:
                    xc  = batch["char_ids"].to(device)
                    val_loss += model.compute_loss(x, xc, tags).item()
                else:
                    val_loss += model.compute_loss(x, tags).item()

        val_f1, _ = evaluate(model, val_loader, idx_to_tag, device, use_chars)
        scheduler.step(val_f1)

        avg_train = epoch_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        elapsed   = time.time() - t0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_f1"].append(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # ── Checkpoint best
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
    print(f"Training complete.  Best val F1: {best_f1:.4f}")
    print(f"Total time: {total_time:.1f}s")

    # ── Test evaluation
    print(f"\n{'='*60}")
    print("  TEST SET EVALUATION")
    print(f"{'='*60}")
    model.load_state_dict(best_state)
    test_f1, test_report = evaluate(model, test_loader, idx_to_tag, device, use_chars)
    print(f"\nTest F1: {test_f1:.4f}\n")
    print(test_report)

    # Save history and report
    json.dump(history, open(f"{out_dir}/history.json", "w"), indent=2)
    with open(f"{out_dir}/test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Test F1: {test_f1:.4f}\n\n")
        f.write(test_report)
    print(f"\n✓ Results saved to: {out_dir}/")

    return model, history


# Default configurations
CONFIGS = {

    # ── Baseline
    "bilstm_softmax": dict(
        model_type        = "bilstm_softmax",
        data_dir          = "data/processed",
        output_dir        = "checkpoints/bilstm_softmax",
        embedding_dim     = 256,
        hidden_dim        = 256,
        num_layers        = 2,
        dropout           = 0.3,
        batch_size        = 32,
        lr                = 1e-3,
        weight_decay      = 1e-4,
        grad_clip         = 5.0,
        epochs            = 30,
        patience          = 5,
        fasttext_path     = None,
        freeze_embeddings = False,
        use_class_weights = False,
    ),

    # ── BiLSTM-CRF (random init)
    "bilstm_crf": dict(
        model_type        = "bilstm_crf",
        data_dir          = "data/processed",
        output_dir        = "checkpoints/bilstm_crf",
        embedding_dim     = 256,
        hidden_dim        = 256,
        num_layers        = 2,
        dropout           = 0.3,
        batch_size        = 32,
        lr                = 1e-3,
        weight_decay      = 1e-4,
        grad_clip         = 5.0,
        epochs            = 30,
        patience          = 5,
        fasttext_path     = None,
        freeze_embeddings = False,
        use_class_weights = True,
        boost_factor      = 1.5,
    ),

    # # ── BiLSTM-CRF + fastText (frozen)
    "bilstm_crf_fasttext_frozen": dict(
        model_type      = "bilstm_crf",
        data_dir        = "data/processed",
        output_dir      = "checkpoints/bilstm_crf_ft_frozen",
        embedding_dim   = 300,          # must match fastText dim
        hidden_dim      = 256,
        num_layers      = 2,
        dropout         = 0.3,
        batch_size      = 32,
        lr              = 1e-3,
        weight_decay    = 1e-4,
        grad_clip       = 5.0,
        epochs          = 30,
        patience        = 5,
        fasttext_path   = "embeddings/cc.my.300.bin",   # ← update path
        freeze_embeddings = True,
    ),

    # # ── BiLSTM-CRF + fastText (fine-tuned)
    "bilstm_crf_fasttext_finetune": dict(
        model_type      = "bilstm_crf",
        data_dir        = "data/processed",
        output_dir      = "checkpoints/bilstm_crf_ft_finetune",
        embedding_dim   = 300,
        hidden_dim      = 256,
        num_layers      = 2,
        dropout         = 0.3,
        batch_size      = 32,
        lr              = 5e-4,       
        weight_decay    = 1e-4,
        grad_clip       = 5.0,
        epochs          = 30,
        patience        = 5,
        fasttext_path   = "embeddings/cc.my.300.bin",
        freeze_embeddings = False,
    ),

    # ── BiLSTM-CRF + CharCNN (best for OOV)
    "bilstm_crf_char": dict(
        model_type        = "bilstm_crf_char",
        data_dir          = "data/processed",
        output_dir        = "checkpoints/bilstm_crf_char",
        embedding_dim     = 128,
        hidden_dim        = 256,
        num_layers        = 2,
        dropout           = 0.5,         
        char_embed_dim    = 30,
        char_num_filters  = 50,
        char_kernel_sizes = (2, 3, 4, 5),
        max_char_len      = 30,
        batch_size        = 32,
        lr                = 1e-3,
        weight_decay      = 1e-3,      
        grad_clip         = 5.0,
        epochs            = 30,
        patience          = 5,
        fasttext_path     = None,
        freeze_embeddings = False,
        # ── Class weighting for ORG / PER
        use_class_weights = True,
        boost_tags        = {
            "B-ORG", "I-ORG", "E-ORG", "S-ORG",
            "B-PER", "I-PER", "E-PER", "S-PER",
        },
        boost_factor      = 3.0,
    ),
}

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Myanmar NER")
    parser.add_argument(
        "--model", default="bilstm_crf",
        choices=list(CONFIGS.keys()),
        help="Which model config to use (ignored when --all is set)"
    )
    parser.add_argument("--all",           action="store_true",
                        help="Train every config in CONFIGS sequentially")
    parser.add_argument("--skip",          nargs="*", default=[],
                        choices=list(CONFIGS.keys()),
                        help="Model name(s) to skip when using --all")
    parser.add_argument("--data_dir",      default=None)
    parser.add_argument("--output_dir",    default=None)
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--fasttext_path", default=None)
    args = parser.parse_args()

    override_keys = ("data_dir", "output_dir", "epochs", "batch_size", "lr", "fasttext_path")

    def _apply_overrides(cfg: dict) -> dict:
        for k in override_keys:
            v = getattr(args, k)
            if v is not None:
                cfg[k] = v
        return cfg

    if args.all:
        FASTTEXT_BIN    = "embeddings/cc.my.300.bin"
        FASTTEXT_MODELS = {"bilstm_crf_fasttext_frozen", "bilstm_crf_fasttext_finetune"}

        models_to_run = [n for n in CONFIGS if n not in args.skip]
        results = {}

        print("=" * 62)
        print(f"  Running all {len(models_to_run)} model(s)")
        print("=" * 62)

        for idx, name in enumerate(models_to_run, 1):
            print(f"\n{'─'*62}")
            print(f"  [{idx}/{len(models_to_run)}] {name}")
            print(f"{'─'*62}")

            if name in FASTTEXT_MODELS and not Path(FASTTEXT_BIN).exists():
                print(f"  ⚠  Skipping '{name}': '{FASTTEXT_BIN}' not found.")
                print(f"     Download: https://fasttext.cc/docs/en/crawl-vectors.html")
                results[name] = {"status": "skipped (no fasttext)", "best_val_f1": None}
                continue

            config = _apply_overrides(CONFIGS[name].copy())
            t0 = time.time()
            try:
                _, history = train(config)
                best_f1 = max(history["val_f1"]) if history["val_f1"] else 0.0
                results[name] = {
                    "status":      "ok",
                    "best_val_f1": round(best_f1, 4),
                    "elapsed_s":   round(time.time() - t0, 1),
                }
                print(f"\n  ✓ {name}  |  best val F1: {best_f1:.4f}")
            except Exception as exc:
                results[name] = {
                    "status":      f"FAILED: {exc}",
                    "best_val_f1": None,
                    "elapsed_s":   round(time.time() - t0, 1),
                }
                print(f"\n  ✗ {name} FAILED: {exc}")

        # Summary table
        print(f"\n{'='*62}\n  SUMMARY\n{'='*62}")
        col = 36
        print(f"  {'Model':<{col}} {'Val F1':>8}  {'Time':>7}  Status")
        print(f"  {'-'*col}  {'─'*8}  {'─'*7}  {'─'*16}")
        for name, r in results.items():
            f1_str   = f"{r['best_val_f1']:.4f}" if r["best_val_f1"] is not None else "     —"
            time_str = f"{r.get('elapsed_s', 0):.0f}s"
            print(f"  {name:<{col}} {f1_str:>8}  {time_str:>7}  {r['status']}")
        print("=" * 62)

        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/run_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n  Summary saved → checkpoints/run_summary.json")

    else:
        config = _apply_overrides(CONFIGS[args.model].copy())
        train(config)



#python train.py --all
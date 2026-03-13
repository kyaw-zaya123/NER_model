#!/usr/bin/env python3
"""
split_conll.py — Split a CoNLL file into train / val / test sets
                 and build word / POS / NER-tag vocabularies.
"""

import json
import random
from collections import Counter
from pathlib import Path

# ── Configure here ────────────────────────────────────────────────────────────
INPUT_FILE = r"train1_dedup.conll"

OUTPUT_TRAIN = r"data/processed/train.conll"
OUTPUT_VAL   = r"data/processed/val.conll"
OUTPUT_TEST  = r"data/processed/test.conll"

OUTPUT_VOCAB     = r"data/processed/vocab.json"
OUTPUT_POS_VOCAB = r"data/processed/pos_vocab.json"
OUTPUT_TAG_VOCAB = r"data/processed/tag_vocab.json"

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def read_conll(path: str) -> list[list[tuple]]:
    """Read CoNLL file → list of sentences (each sentence = list of (word, pos, ner))."""
    text = Path(path).read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sentences, sent = [], []
    for line in text.split("\n"):
        line = line.strip()
        if line == "":
            if sent:
                sentences.append(sent)
                sent = []
        else:
            parts = line.split("\t")
            if len(parts) == 3:
                sent.append((parts[0], parts[1], parts[2]))
            elif len(parts) == 2:
                sent.append((parts[0], "n", parts[1]))
    if sent:
        sentences.append(sent)
    return sentences


def write_conll(path: str, sentences: list[list[tuple]]) -> None:
    """Write sentences to CoNLL format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for word, pos, ner in sent:
                f.write(f"{word}\t{pos}\t{ner}\n")
            f.write("\n")


def build_vocab(train_sentences: list[list[tuple]]) -> tuple[dict, dict, dict]:
    """
    Build vocabularies from training sentences only.

    Returns:
        vocab     : word  → index  (includes <PAD>=0, <UNK>=1)
        pos_vocab : POS   → index  (includes <PAD>=0)
        tag_vocab : NER   → index  (includes <PAD>=0)
    """
    vocab:     dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    pos_vocab: dict[str, int] = {"<PAD>": 0}
    tag_vocab: dict[str, int] = {"<PAD>": 0}

    for sent in train_sentences:
        for word, pos, tag in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
            if pos not in pos_vocab:
                pos_vocab[pos] = len(pos_vocab)
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)

    return vocab, pos_vocab, tag_vocab


def save_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def print_stats(label: str, sentences: list[list[tuple]]) -> None:
    total_tokens = sum(len(s) for s in sentences)
    tag_counts: Counter = Counter()
    for sent in sentences:
        for _, _, tag in sent:
            tag_counts[tag] += 1
    entity_tokens = sum(v for k, v in tag_counts.items() if k != "O")
    print(f"  {label:<8}: {len(sentences):>7,} sentences | "
          f"{total_tokens:>8,} tokens | "
          f"{entity_tokens:>7,} entity tokens "
          f"({entity_tokens/total_tokens:.1%})")


def main():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, \
        "Ratios must sum to 1.0"

    # ── Load ──────────────────────────────────────────────────────────────────
    sentences = read_conll(INPUT_FILE)
    total     = len(sentences)
    print(f"Loaded  : {total:,} sentences from {INPUT_FILE}\n")

    # ── Shuffle ───────────────────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    random.shuffle(sentences)

    # ── Split ─────────────────────────────────────────────────────────────────
    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)

    train = sentences[:n_train]
    val   = sentences[n_train : n_train + n_val]
    test  = sentences[n_train + n_val :]

    # ── Write CoNLL files ─────────────────────────────────────────────────────
    write_conll(OUTPUT_TRAIN, train)
    write_conll(OUTPUT_VAL,   val)
    write_conll(OUTPUT_TEST,  test)

    # ── Build & save vocabularies (from train only) ───────────────────────────
    vocab, pos_vocab, tag_vocab = build_vocab(train)
    save_json(vocab,     OUTPUT_VOCAB)
    save_json(pos_vocab, OUTPUT_POS_VOCAB)
    save_json(tag_vocab, OUTPUT_TAG_VOCAB)

    # ── Report ────────────────────────────────────────────────────────────────
    print("Split complete:")
    print_stats("Train", train)
    print_stats("Val",   val)
    print_stats("Test",  test)
    print(f"\nVocabularies (built from train only):")
    print(f"  vocab.json     : {len(vocab):,} words       → {OUTPUT_VOCAB}")
    print(f"  pos_vocab.json : {len(pos_vocab):,} POS tags    → {OUTPUT_POS_VOCAB}")
    print(f"  tag_vocab.json : {len(tag_vocab):,} NER tags    → {OUTPUT_TAG_VOCAB}")
    print(f"\nCoNLL files:")
    print(f"  {OUTPUT_TRAIN}")
    print(f"  {OUTPUT_VAL}")
    print(f"  {OUTPUT_TEST}")


if __name__ == "__main__":
    main()
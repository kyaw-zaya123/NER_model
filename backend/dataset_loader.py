"""
=============================================================================
Myanmar NER - STEP 3: PyTorch Dataset & DataLoader
=============================================================================
Handles:
  • Word-level tokenisation with vocabulary lookup
  • Character-level tokenisation for CharCNN model
  • Dynamic padding via custom collate_fn
  • Optional fastText vector loading
=============================================================================
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np


# CoNLL Dataset
class MyanmarNERDataset(Dataset):
    """
    Reads a processed CoNLL file (word TAB pos TAB ner) and converts to
    integer tensors.

    Supports optional character-level encoding for BiLSTMCRFCharCNN.
    """

    def __init__(
        self,
        conll_path: str,
        vocab: dict,
        tag_vocab: dict,
        pos_vocab: Optional[dict] = None,
        char_vocab: Optional[dict] = None,
        max_char_len: int = 30,
    ):
        self.vocab        = vocab
        self.tag_vocab    = tag_vocab
        self.pos_vocab    = pos_vocab
        self.char_vocab   = char_vocab
        self.max_char_len = max_char_len
        self.use_chars    = char_vocab is not None
        self.use_pos      = pos_vocab is not None

        self.sentences, self.pos_seqs, self.tag_seqs = self._load(conll_path)

    def _load(self, path):
        sentences, pos_seqs, tag_seqs = [], [], []
        with open(path, encoding="utf-8") as f:
            words, poss, tags = [], [], []
            for line in f:
                line = line.rstrip("\n\r")
                if line.strip() == "":
                    if words:
                        sentences.append(words)
                        pos_seqs.append(poss)
                        tag_seqs.append(tags)
                        words, poss, tags = [], [], []
                else:
                    parts = line.split("\t")
                    w = parts[0]
                    p = parts[1] if len(parts) > 2 else "n"
                    t = parts[-1]
                    words.append(w)
                    poss.append(p)
                    tags.append(t)
            if words:
                sentences.append(words)
                pos_seqs.append(poss)
                tag_seqs.append(tags)
        return sentences, pos_seqs, tag_seqs

    def _word_ids(self, words):
        unk = self.vocab.get("<UNK>", 1)
        return [self.vocab.get(w, unk) for w in words]

    def _tag_ids(self, tags):
        return [self.tag_vocab.get(t, 0) for t in tags]

    def _pos_ids(self, poss):
        unk = self.pos_vocab.get("<UNK>", 0)
        return [self.pos_vocab.get(p, unk) for p in poss]

    def _char_ids(self, words):
        """
        Returns a list of lists: for each word, the list of character ids
        (truncated / padded to max_char_len).
        """
        result = []
        for w in words:
            chars = list(w)[:self.max_char_len]
            ids   = [self.char_vocab.get(c, 1) for c in chars]  
            ids  += [0] * (self.max_char_len - len(ids))
            result.append(ids)
        return result

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        poss  = self.pos_seqs[idx]
        tags  = self.tag_seqs[idx]
        item  = {
            "words"   : words,
            "word_ids": torch.tensor(self._word_ids(words),  dtype=torch.long),
            "tag_ids" : torch.tensor(self._tag_ids(tags),    dtype=torch.long),
            "length"  : len(words),
        }
        if self.use_pos:
            item["pos_ids"] = torch.tensor(self._pos_ids(poss), dtype=torch.long)
        if self.use_chars:
            item["char_ids"] = torch.tensor(
                self._char_ids(words), dtype=torch.long
            )  # (S, C)
        return item


# Collate function (handles variable-length sentences)
def collate_fn(batch: list[dict]) -> dict:
    """
    Pad word_ids, pos_ids, tag_ids (and optionally char_ids) to the longest
    sentence in the batch.
    """
    max_len = max(item["length"] for item in batch)

    word_ids_batch = []
    pos_ids_batch  = []
    tag_ids_batch  = []
    char_ids_batch = []
    lengths        = []
    has_chars = "char_ids" in batch[0]
    has_pos   = "pos_ids"  in batch[0]

    for item in batch:
        pad_len = max_len - item["length"]
        word_ids_batch.append(
            torch.cat([item["word_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        tag_ids_batch.append(
            torch.cat([item["tag_ids"],  torch.zeros(pad_len, dtype=torch.long)])
        )
        lengths.append(item["length"])
        if has_pos:
            pos_ids_batch.append(
                torch.cat([item["pos_ids"], torch.zeros(pad_len, dtype=torch.long)])
            )
        if has_chars:
            C = item["char_ids"].shape[-1]
            pad_chars = torch.zeros(pad_len, C, dtype=torch.long)
            char_ids_batch.append(torch.cat([item["char_ids"], pad_chars], dim=0))

    result = {
        "word_ids": torch.stack(word_ids_batch),   # (B, S)
        "tag_ids" : torch.stack(tag_ids_batch),    # (B, S)
        "lengths" : torch.tensor(lengths),
    }
    if has_pos:
        result["pos_ids"] = torch.stack(pos_ids_batch)   # (B, S)
    if has_chars:
        result["char_ids"] = torch.stack(char_ids_batch)  # (B, S, C)
    return result


# Build character vocabulary from training data
def build_char_vocab(train_dataset: MyanmarNERDataset) -> dict:
    """
    Builds a char → id mapping from training sentences.
    0 = <PAD_CHAR>, 1 = <UNK_CHAR>
    """
    char_vocab = {"<PAD_CHAR>": 0, "<UNK_CHAR>": 1}
    for words in train_dataset.sentences:
        for w in words:
            for c in w:
                if c not in char_vocab:
                    char_vocab[c] = len(char_vocab)
    return char_vocab


# fastText embedding loader
def load_fasttext_embeddings(
    fasttext_bin_path: str,
    vocab: dict,
    embedding_dim: int = 300,
) -> torch.Tensor:
    """
    Load fastText embeddings (.bin or .vec) aligned to our vocabulary.

    For .bin files: uses the fasttext Python library.
    For .vec files: loads plain text vectors.

    Returns a (vocab_size, embedding_dim) float tensor.
    """
    vocab_size  = len(vocab)
    matrix      = np.random.normal(0, 0.1, (vocab_size, embedding_dim)).astype(np.float32)
    matrix[0]   = 0.0   # PAD stays zero

    ext = fasttext_bin_path.rsplit(".", 1)[-1].lower()

    if ext == "bin":
        try:
            import fasttext
        except ImportError:
            raise ImportError("pip install fasttext")
        ft_model = fasttext.load_model(fasttext_bin_path)
        for word, idx in vocab.items():
            if word in ("<PAD>", "<UNK>"):
                continue
            vec = ft_model.get_word_vector(word)
            if vec.shape[0] == embedding_dim:
                matrix[idx] = vec
        print(f"✓ Loaded fastText (.bin) for {vocab_size:,} words")

    elif ext == "vec":
        found = 0
        with open(fasttext_bin_path, encoding="utf-8") as f:
            header = f.readline()               # skip "n_words dim" header
            for line in f:
                parts = line.rstrip().split(" ")
                word  = parts[0]
                if word in vocab and len(parts) == embedding_dim + 1:
                    matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                    found += 1
        print(f"✓ Loaded fastText (.vec): {found:,}/{vocab_size:,} words covered")

    else:
        raise ValueError(f"Unknown fastText format: .{ext}  (expected .bin or .vec)")

    return torch.tensor(matrix)


# DataLoader factory
def get_dataloaders(
    data_dir: str,
    vocab: dict,
    tag_vocab: dict,
    pos_vocab: Optional[dict] = None,
    char_vocab: Optional[dict] = None,
    batch_size: int = 32,
    max_char_len: int = 30,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build train / val / test DataLoaders from processed CoNLL files.

    Returns: (train_loader, val_loader, test_loader, datasets_dict)
    """
    datasets = {}
    for split in ("train", "val", "test"):
        path = f"{data_dir}/{split}.conll"
        datasets[split] = MyanmarNERDataset(
            path, vocab, tag_vocab, pos_vocab, char_vocab, max_char_len
        )
        print(f"  {split:5s}: {len(datasets[split]):,} sentences")

    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size,
        shuffle=True,  collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, datasets


# Sanity check
if __name__ == "__main__":
    import json
    vocab     = json.load(open("data/processed/vocab.json", encoding="utf-8"))
    tag_vocab = json.load(open("data/processed/tag_vocab.json", encoding="utf-8"))
    pos_vocab = json.load(open("data/processed/pos_vocab.json", encoding="utf-8"))

    print("Building train dataset...")
    train_ds = MyanmarNERDataset("data/processed/train.conll", vocab, tag_vocab, pos_vocab)
    char_vocab = build_char_vocab(train_ds)
    print(f"  char_vocab size: {len(char_vocab)}")

    train_ds = MyanmarNERDataset(
        "data/processed/train.conll", vocab, tag_vocab,
        pos_vocab=pos_vocab, char_vocab=char_vocab
    )
    loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
    batch  = next(iter(loader))
    print(f"\nSample batch shapes:")
    print(f"  word_ids : {batch['word_ids'].shape}")
    print(f"  tag_ids  : {batch['tag_ids'].shape}")
    if "pos_ids" in batch:
        print(f"  pos_ids  : {batch['pos_ids'].shape}")
    if "char_ids" in batch:
        print(f"  char_ids : {batch['char_ids'].shape}")
    print(f"  lengths  : {batch['lengths'].tolist()}")
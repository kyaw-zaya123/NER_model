import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from transformers import AutoTokenizer

IGNORE_IDX = -100   # CrossEntropyLoss / CRF mask sentinel


# Dataset
class XLMRNERDataset(Dataset):
    """
    Reads a CoNLL file (word TAB pos TAB ner) and produces subword-aligned
    tensors for XLM-RoBERTa fine-tuning.
    """

    def __init__(
        self,
        conll_path: str,
        tag_vocab: dict,
        tokenizer_name: str = "FacebookAI/xlm-roberta-base",
        max_length: int = 256,
        tokenizer: Optional[object] = None,
    ):
        self.tag_vocab   = tag_vocab
        self.max_length  = max_length
        self.tokenizer   = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)

        self.sentences, self.pos_seqs, self.tag_seqs = self._load_conll(conll_path)

    # ── CoNLL reader

    def _load_conll(self, path: str):
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
                    words.append(parts[0])
                    poss.append(parts[1] if len(parts) >= 3 else "n")
                    tags.append(parts[-1])
            if words:
                sentences.append(words)
                pos_seqs.append(poss)
                tag_seqs.append(tags)
        return sentences, pos_seqs, tag_seqs

    # ── Subword alignment

    def _encode(self, words: list[str], tags: list[str]) -> dict:
        """
        Tokenise a word-level sentence and align NER labels to subword pieces.

        Returns a dict with:
            input_ids      : (S,)  subword ids including [CLS]/[SEP]
            attention_mask : (S,)
            label_ids      : (S,)  first-piece = real tag id; others = IGNORE_IDX
            word_tag_ids   : (W,)  original word-level tag ids (for eval)
            num_words      : int
        """
        enc = self.tokenizer(
            words,
            is_split_into_words=True,   # treat list as pre-tokenised words
            max_length=self.max_length,
            truncation=True,
            padding=False,              # collate_fn handles padding
            return_tensors=None,        # return plain lists
        )

        word_ids    = enc.word_ids()    # list: subword index → word index (None for special)
        label_ids   = []
        seen_words  = set()

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(IGNORE_IDX)
            elif word_idx not in seen_words:
                tag_str = tags[word_idx] if word_idx < len(tags) else "O"
                label_ids.append(self.tag_vocab.get(tag_str, 0))
                seen_words.add(word_idx)
            else:
                label_ids.append(IGNORE_IDX)

        word_tag_ids = [self.tag_vocab.get(t, 0) for t in tags]

        return {
            "input_ids"    : enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label_ids"    : label_ids,
            "word_tag_ids" : word_tag_ids,   # word-level gold (for seqeval)
            "num_words"    : len(words),
        }

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        item = self._encode(self.sentences[idx], self.tag_seqs[idx])
        return {k: torch.tensor(v, dtype=torch.long)
                if isinstance(v, list) else v
                for k, v in item.items()}


# Collate function
def xlmr_collate_fn(batch: list[dict]) -> dict:
    """
    Pad subword sequences (input_ids, attention_mask, label_ids) and
    word-level tag sequences (word_tag_ids) to their respective max lengths.
    """
    # Subword max length (for transformer input)
    max_subword = max(item["input_ids"].shape[0] for item in batch)
    # Word max length (for evaluation tensors)
    max_words   = max(item["num_words"] for item in batch)

    pad_token_id = 1   # XLM-R pad token id

    input_ids_batch      = []
    attention_mask_batch = []
    label_ids_batch      = []
    word_tag_ids_batch   = []
    word_lengths         = []

    for item in batch:
        sw_len   = item["input_ids"].shape[0]
        sw_pad   = max_subword - sw_len
        w_len    = item["num_words"]
        w_pad    = max_words - w_len

        input_ids_batch.append(torch.cat([
            item["input_ids"],
            torch.full((sw_pad,), pad_token_id, dtype=torch.long),
        ]))
        attention_mask_batch.append(torch.cat([
            item["attention_mask"],
            torch.zeros(sw_pad, dtype=torch.long),
        ]))
        label_ids_batch.append(torch.cat([
            item["label_ids"],
            torch.full((sw_pad,), IGNORE_IDX, dtype=torch.long),
        ]))
        word_tag_ids_batch.append(torch.cat([
            item["word_tag_ids"],
            torch.zeros(w_pad, dtype=torch.long),   # 0 = <PAD> tag
        ]))
        word_lengths.append(w_len)

    return {
        "input_ids"      : torch.stack(input_ids_batch),        # (B, Ssw)
        "attention_mask" : torch.stack(attention_mask_batch),   # (B, Ssw)
        "label_ids"      : torch.stack(label_ids_batch),        # (B, Ssw)
        "word_tag_ids"   : torch.stack(word_tag_ids_batch),     # (B, W)
        "word_lengths"   : torch.tensor(word_lengths),          # (B,)
    }


# DataLoader factory
def get_xlmr_dataloaders(
    data_dir: str,
    tag_vocab: dict,
    tokenizer_name: str = "FacebookAI/xlm-roberta-base",
    max_length: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build train / val / test DataLoaders.

    Shares one tokenizer instance across all splits for efficiency.

    Returns: (train_loader, val_loader, test_loader, datasets_dict)
    """
    print(f"  Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    datasets = {}
    for split in ("train", "val", "test"):
        path = f"{data_dir}/{split}.conll"
        datasets[split] = XLMRNERDataset(
            path, tag_vocab,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        print(f"  {split:5s}: {len(datasets[split]):,} sentences")

    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size,
        shuffle=True,  collate_fn=xlmr_collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        datasets["val"],   batch_size=batch_size,
        shuffle=False, collate_fn=xlmr_collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        datasets["test"],  batch_size=batch_size,
        shuffle=False, collate_fn=xlmr_collate_fn, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, datasets


# Sanity check
if __name__ == "__main__":
    tag_vocab = json.load(open("data/processed/tag_vocab.json", encoding="utf-8"))

    train_loader, val_loader, test_loader, _ = get_xlmr_dataloaders(
        data_dir   = "data/processed",
        tag_vocab  = tag_vocab,
        batch_size = 4,
    )
    batch = next(iter(train_loader))
    print("\nSample batch shapes:")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k:20s}: {tuple(v.shape)}")
    n_real = (batch["label_ids"] != IGNORE_IDX).sum(dim=1)
    print(f"\n  Real labels per sentence : {n_real.tolist()}")
    print(f"  Word lengths             : {batch['word_lengths'].tolist()}")
    print("  ✓ Counts match           :", (n_real == batch["word_lengths"]).all().item())
"""
=============================================================================
Myanmar NER - XLM-RoBERTa + Token CRF Model
=============================================================================
Architecture:
    XLM-RoBERTa (frozen or fine-tuned)
        ↓  last_hidden_state  (B, S_subword, 768)
    _extract_word_emissions()          ← KEY FIX
        ↓  first-piece rows only  (B, W, 768)
    Dropout → Linear(768 → num_tags)
        ↓  emission scores  (B, W, T)
    Token CRF  — dense sequence, no masked gaps
        ↓
    Word-level predictions  (B, W)

Root cause of previous negative-loss bug:
    Passing the full subword sequence (B, S_subword, T) to torchcrf with a
    sparse mask (most positions False) causes the CRF partition function to
    grow unboundedly — log-likelihood explodes positive, negated loss goes
    increasingly negative, model learns nothing.

Fix:
    Extract only the first-piece hidden states BEFORE the linear head, giving
    the CRF a clean dense (B, W, T) tensor.  No masked gaps, no explosion.
=============================================================================
"""

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel
from typing import Optional

IGNORE_IDX = -100


class XLMRobertaNERCRF(nn.Module):
    """
    XLM-RoBERTa encoder → first-piece extraction → Linear → Token CRF.
    All CRF operations work on word-level (dense) sequences only.
    """

    def __init__(
        self,
        num_tags: int,
        model_name: str = "FacebookAI/xlm-roberta-base",
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        pretrained_model: Optional[object] = None,
    ):
        super().__init__()
        self.num_tags = num_tags

        # ── Encoder
        self.encoder = pretrained_model or AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size   # 768 for base, 1024 for large

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        # ── Head
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)

        # ── CRF — operates on word-level dense sequences 
        self.crf = CRF(num_tags, batch_first=True)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ── Core: extract first-piece hidden states 

    @staticmethod
    def _extract_word_emissions(
        hidden: torch.Tensor,       # (B, S_subword, H)
        label_ids: torch.Tensor,    # (B, S_subword)  IGNORE_IDX except first pieces
        max_words: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather the hidden state at each word's FIRST subword piece.

        Returns:
            word_hidden  : (B, W, H)   — dense word-level hidden states
            word_mask    : (B, W) bool — True for real words, False for padding
                           (needed because different sentences have different W)

        This is the fix for the CRF log-likelihood explosion:
        instead of passing (B, S_subword, T) with a sparse mask to torchcrf,
        we pass a clean dense (B, W, T) tensor where every position is a
        real word — no masked gaps for the partition function to exploit.
        """
        B, S, H = hidden.shape
        first_piece_mask = (label_ids != IGNORE_IDX)   # (B, S)

        word_hidden = hidden.new_zeros(B, max_words, H)   # (B, W, H)
        word_mask   = hidden.new_zeros(B, max_words, dtype=torch.bool)

        for b in range(B):
            indices = first_piece_mask[b].nonzero(as_tuple=True)[0]  # positions of first pieces
            n = min(len(indices), max_words)
            if n > 0:
                word_hidden[b, :n] = hidden[b, indices[:n]]
                word_mask[b, :n]   = True

        return word_hidden, word_mask

    # ── Encode subwords → word-level emissions

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_ids: torch.Tensor,
        max_words: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run XLM-R, extract first-piece hidden states, project to tag space.

        Returns:
            emissions : (B, W, num_tags)
            word_mask : (B, W) bool
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden  = self.dropout(outputs.last_hidden_state)          # (B, S, H)
        word_h, word_mask = self._extract_word_emissions(hidden, label_ids, max_words)
        emissions = self.classifier(word_h)                        # (B, W, T)
        return emissions, word_mask

    # ── Training 

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_ids: torch.Tensor,
        word_tag_ids: torch.Tensor,     # (B, W)  word-level gold tag ids
        word_lengths: torch.Tensor,     # (B,)    number of words per sentence
    ) -> torch.Tensor:
        """
        CRF negative log-likelihood on word-level dense sequences.

        word_tag_ids  : (B, W)  — gold tag id for each word (0=PAD for padding)
        word_lengths  : (B,)    — actual word counts (for building CRF mask)
        """
        max_words = word_tag_ids.shape[1]
        emissions, word_mask = self._encode(input_ids, attention_mask, label_ids, max_words)

        # CRF mask: True for real words only (matches word_mask exactly)
        # word_tag_ids already has 0 for padding positions — safe for CRF
        return -self.crf(emissions, word_tag_ids, mask=word_mask, reduction="mean")

    # ── Inference 

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_ids: torch.Tensor,
        word_lengths: torch.Tensor,     # (B,)
    ) -> list[list[int]]:
        """
        Decode to word-level tag id sequences.

        Returns: list[list[int]], one list per sentence, length = word_lengths[i]
        """
        max_words = word_lengths.max().item()
        emissions, word_mask = self._encode(input_ids, attention_mask, label_ids, max_words)
        return self.crf.decode(emissions, mask=word_mask)


# Model factory
def build_xlmr_model(config: dict) -> XLMRobertaNERCRF:
    return XLMRobertaNERCRF(
        num_tags       = config["num_tags"],
        model_name     = config.get("model_name", "FacebookAI/xlm-roberta-base"),
        dropout        = config.get("dropout", 0.1),
        freeze_encoder = config.get("freeze_encoder", False),
    )


# Sanity check
if __name__ == "__main__":
    import json
    B, W, T = 3, 12, 52   # batch, words, tags

    # Simulate a batch: each sentence has W real words
    # Subword sequence is longer: ~2x words on average for Myanmar
    S = W * 2 + 2   # +2 for [CLS] and [SEP]

    input_ids      = torch.randint(0, 250002, (B, S))
    attention_mask = torch.ones(B, S, dtype=torch.long)

    label_ids    = torch.full((B, S), IGNORE_IDX, dtype=torch.long)
    word_tag_ids = torch.zeros(B, W, dtype=torch.long)
    for b in range(B):
        for w in range(W):
            subword_pos = 1 + w * 2   # position 1,3,5,... (skip [CLS] at 0)
            tag = torch.randint(1, T, (1,)).item()
            label_ids[b, subword_pos]  = tag
            word_tag_ids[b, w]         = tag
    word_lengths = torch.full((B,), W, dtype=torch.long)

    model = XLMRobertaNERCRF(num_tags=T)
    model.eval()

    loss  = model.compute_loss(input_ids, attention_mask, label_ids, word_tag_ids, word_lengths)
    preds = model.decode(input_ids, attention_mask, label_ids, word_lengths)

    print(f"✓ Loss (should be positive): {loss.item():.4f}")
    assert loss.item() > 0, "Loss must be positive!"
    print(f"✓ Pred lengths (should be {W} each): {[len(p) for p in preds]}")
    assert all(len(p) == W for p in preds), "Pred length mismatch!"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable params: {n_params:,}")
    print("All assertions passed.")
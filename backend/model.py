"""
=============================================================================
Myanmar NER - STEP 2: Model Architecture
=============================================================================
Three model variants, each in its own class:

  1. BiLSTM-Softmax   : fast baseline, word embeddings �?BiLSTM �?Linear
  2. BiLSTM-CRF       : adds CRF layer for structured prediction
  3. BiLSTM-CRF+Char  : adds character-level CNN encoder (best for OOV)

All models share the same inference interface: .decode(tokens) �?tag list.
=============================================================================
"""

import torch
import torch.nn as nn
try:
    from torchcrf import CRF
except ModuleNotFoundError:
    from TorchCRF import CRF
from typing import Optional

# Utility: char-level CNN encoder
class CharCNNEncoder(nn.Module):
    """
    Encodes each word into a fixed-size vector via character-level CNN.

    Input : (batch, seq_len, max_char_len)  �?integer char ids
    Output: (batch, seq_len, char_out_dim)
    """
    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int = 30,
        num_filters: int = 50,
        kernel_sizes: tuple = (2, 3, 4),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = num_filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C)  �? treat each (B*S) word independently
        B, S, C = x.shape
        x = x.view(B * S, C)                         # (B*S, C)
        emb = self.char_embed(x)                      # (B*S, C, D)
        emb = emb.permute(0, 2, 1)                    # (B*S, D, C)  for Conv1d
        emb = self.dropout(emb)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))                 # (B*S, F, C-k+1)
            c = c.max(dim=-1).values                  # (B*S, F)
            pooled.append(c)
        out = torch.cat(pooled, dim=-1)               # (B*S, F*num_kernels)
        out = self.dropout(out)
        return out.view(B, S, -1)                     # (B, S, out_dim)


# Highway layer (improves gradient flow in deep models)

class Highway(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.H = nn.Linear(size, size)
        self.T = nn.Linear(size, size)
        nn.init.constant_(self.T.bias, -1.0)   # bias gate toward carry

    def forward(self, x):
        t = torch.sigmoid(self.T(x))
        return t * torch.relu(self.H(x)) + (1 - t) * x


# Model 1 �?BiLSTM + Softmax (baseline)
class BiLSTMSoftmax(nn.Module):
    """
    Simple BiLSTM with linear output + cross-entropy loss.
    Fast to train; good as a baseline.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout   = nn.Dropout(dropout)
        self.bilstm    = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.highway = Highway(hidden_dim * 2)
        self.fc      = nn.Linear(hidden_dim * 2, num_tags)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))      # (B, S, E)
        out, _ = self.bilstm(emb)                  # (B, S, 2H)
        out = self.highway(out)
        logits = self.fc(self.dropout(out))        # (B, S, T)
        return logits

    def compute_loss(self, x, tags):
        logits = self.forward(x)                   # (B, S, T)
        B, S, T = logits.shape
        return self.loss_fn(logits.view(B * S, T), tags.view(B * S))

    def decode(self, x: torch.Tensor) -> list[list[int]]:
        logits = self.forward(x)                   # (B, S, T)
        preds  = logits.argmax(dim=-1)             # (B, S)
        mask   = x != self.pad_idx
        result = []
        for i, pred_row in enumerate(preds):
            length = mask[i].sum().item()
            result.append(pred_row[:length].tolist())
        return result


# Model 2 �?BiLSTM-CRF  (recommended default)
class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF: the workhorse of sequence labelling.
    CRF layer enforces valid BIOES tag transitions.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)

        self.dropout   = nn.Dropout(dropout)
        self.bilstm    = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.highway   = Highway(hidden_dim * 2)
        self.fc        = nn.Linear(hidden_dim * 2, num_tags)
        self.crf       = CRF(num_tags, batch_first=True)
        # Weighted CE on emissions (auxiliary loss to up-weight rare classes)
        self.aux_loss  = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=pad_idx
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        emb      = self.dropout(self.embedding(x))
        out, _   = self.bilstm(emb)
        out      = self.highway(out)
        logits   = self.fc(self.dropout(out))
        return logits

    def compute_loss(self, x: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        mask   = (x != self.pad_idx)
        logits = self._encode(x)
        crf_loss = -self.crf(logits, tags, mask=mask, reduction="mean")
        # Auxiliary weighted CE steers emissions toward rare classes (ORG, PER)
        # without breaking CRF transition learning.
        B, S, T = logits.shape
        aux_loss = self.aux_loss(logits.view(B * S, T), tags.view(B * S))
        return crf_loss + 0.1 * aux_loss

    def decode(self, x: torch.Tensor) -> list[list[int]]:
        mask   = (x != self.pad_idx)
        logits = self._encode(x)
        return self.crf.decode(logits, mask=mask)


# Model 3 �?BiLSTM-CRF + Character CNN  (best accuracy)
class BiLSTMCRFCharCNN(nn.Module):
    """
    BiLSTM-CRF with character-level CNN embeddings concatenated to word
    embeddings.  Handles OOV Myanmar syllables much better than word-only
    models.

    Input:
        x       : (B, S)          word ids
        x_chars : (B, S, C)       character ids per token
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        char_vocab_size: int,
        char_embed_dim: int = 30,
        char_num_filters: int = 50,
        char_kernel_sizes: tuple = (2, 3, 4),
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.pad_idx   = pad_idx

        # Word embeddings
        self.word_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.word_embed.weight = nn.Parameter(pretrained_embeddings)

        # Char CNN
        self.char_cnn = CharCNNEncoder(
            char_vocab_size, char_embed_dim,
            char_num_filters, char_kernel_sizes, dropout
        )

        self.dropout = nn.Dropout(dropout)
        lstm_input   = embedding_dim + self.char_cnn.out_dim
        self.bilstm  = nn.LSTM(
            lstm_input, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.highway  = Highway(hidden_dim * 2)
        self.fc       = nn.Linear(hidden_dim * 2, num_tags)
        self.crf      = CRF(num_tags, batch_first=True)
        self.aux_loss = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=pad_idx
        )

    def _encode(self, x: torch.Tensor, x_chars: torch.Tensor) -> torch.Tensor:
        word_emb = self.dropout(self.word_embed(x))    # (B, S, E)
        char_emb = self.char_cnn(x_chars)              # (B, S, F)
        combined = torch.cat([word_emb, char_emb], -1) # (B, S, E+F)
        out, _   = self.bilstm(combined)               # (B, S, 2H)
        out      = self.highway(out)
        logits   = self.fc(self.dropout(out))          # (B, S, T)
        return logits

    def compute_loss(
        self,
        x: torch.Tensor,
        x_chars: torch.Tensor,
        tags: torch.Tensor,
    ) -> torch.Tensor:
        mask   = (x != self.pad_idx)
        logits = self._encode(x, x_chars)
        crf_loss = -self.crf(logits, tags, mask=mask, reduction="mean")
        B, S, T  = logits.shape
        aux_loss = self.aux_loss(logits.view(B * S, T), tags.view(B * S))
        return crf_loss + 0.1 * aux_loss

    def decode(
        self,
        x: torch.Tensor,
        x_chars: torch.Tensor,
    ) -> list[list[int]]:
        mask   = (x != self.pad_idx)
        logits = self._encode(x, x_chars)
        return self.crf.decode(logits, mask=mask)


# Model factory
def build_model(config: dict, pretrained_embeddings=None):
    """
    config keys:
        model_type      : "bilstm_softmax" | "bilstm_crf" | "bilstm_crf_char"
        vocab_size      : int
        embedding_dim   : int
        hidden_dim      : int
        num_tags        : int
        num_layers      : int
        dropout         : float
        char_vocab_size : int  (only for bilstm_crf_char)
        char_embed_dim  : int
        char_num_filters: int
    """
    mtype = config["model_type"]
    common = dict(
        vocab_size    = config["vocab_size"],
        embedding_dim = config["embedding_dim"],
        hidden_dim    = config["hidden_dim"],
        num_tags      = config["num_tags"],
        num_layers    = config.get("num_layers", 2),
        dropout       = config.get("dropout", 0.3),
    )
    if pretrained_embeddings is not None:
        common["pretrained_embeddings"] = pretrained_embeddings

    class_weights = config.get("class_weights", None)  

    if mtype == "bilstm_softmax":
        return BiLSTMSoftmax(**common)
    elif mtype == "bilstm_crf":
        return BiLSTMCRF(**common, class_weights=class_weights)
    elif mtype == "bilstm_crf_char":
        return BiLSTMCRFCharCNN(
            **common,
            char_vocab_size  = config["char_vocab_size"],
            char_embed_dim   = config.get("char_embed_dim", 30),
            char_num_filters = config.get("char_num_filters", 50),
            class_weights    = class_weights,
        )
    else:
        raise ValueError(f"Unknown model_type: {mtype}")


if __name__ == "__main__":
    # Quick sanity check
    B, S, C = 4, 20, 10
    T = 25   # num tags

    x       = torch.randint(2, 100, (B, S))
    x[:, 15:] = 0         
    x_chars = torch.randint(1, 50, (B, S, C))
    tags    = torch.randint(1, T, (B, S))
    tags[:, 15:] = 0

    for mtype in ["bilstm_softmax", "bilstm_crf", "bilstm_crf_char"]:
        cfg = dict(
            model_type="bilstm_crf_char" if mtype == "bilstm_crf_char" else mtype,
            vocab_size=200, embedding_dim=64, hidden_dim=64,
            num_tags=T, num_layers=2, dropout=0.3,
            char_vocab_size=80,
        )
        m = build_model(cfg)
        m.eval()
        if mtype == "bilstm_crf_char":
            out = m.decode(x, x_chars)
        else:
            out = m.decode(x)
        print(f"�?{mtype:25s}  output lengths: {[len(o) for o in out]}")

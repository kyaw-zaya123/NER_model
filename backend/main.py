"""
Myanmar NER — FastAPI inference server
Supports: bilstm_softmax | bilstm_crf | bilstm_crf_char | xlmr_base
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── paths (relative to backend/)
BASE_DIR     = Path(__file__).parent
CKPT_DIR     = BASE_DIR / "checkpoints"
DATA_DIR     = BASE_DIR / "processed"
MODEL_NAME   = os.getenv("NER_MODEL", "bilstm_crf")   # override via env var

# ── colour palette sent to the frontend
ENTITY_COLORS: dict[str, str] = {
    "PER":          "#f87171",
    "ORG":          "#34d399",
    "DATE":         "#fbbf24",
    "TIME":         "#fb923c",
    "NUM":          "#c084fc",
    "LOC":          "#60a5fa",
    "LOC-COUNTRY":  "#3b82f6",
    "LOC-STATE":    "#6ee7b7",
    "LOC-DISTRICT": "#5eead4",
    "LOC-TOWNSHIP": "#38bdf8",
    "LOC-CITY":     "#7dd3fc",
    "LOC-VILLAGE":  "#86efac",
    "LOC-WARD":     "#93c5fd",
}


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

class NERModel:
    """Wraps any checkpoint (BiLSTM or XLM-R) behind a unified .predict() API."""

    def __init__(self, model_name: str):
        self.model_name  = model_name
        self.device      = torch.device("cpu")
        self.model       = None
        self.vocab:       dict[str, int] = {}
        self.tag_vocab:   dict[str, int] = {}
        self.char_vocab:  dict[str, int] = {}
        self.idx_to_tag:  dict[int, str] = {}
        self.config:      dict            = {}
        self.is_xlmr      = model_name.startswith("xlmr")
        self.tokenizer    = None
        self._load()

    def _load(self):
        ckpt_path = CKPT_DIR / self.model_name / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint      = torch.load(ckpt_path, map_location="cpu")
        self.config     = checkpoint["config"]
        state_dict      = checkpoint["model_state"]

        # Vocabularies
        with open(DATA_DIR / "tag_vocab.json", encoding="utf-8") as f:
            self.tag_vocab = json.load(f)
        self.idx_to_tag = {v: k for k, v in self.tag_vocab.items()}

        if self.is_xlmr:
            from xlmr_model import build_xlmr_model
            from transformers import AutoTokenizer
            self.config["num_tags"] = len(self.tag_vocab)
            self.model = build_xlmr_model(self.config)
            self.model.load_state_dict(state_dict)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.get("model_name", "FacebookAI/xlm-roberta-base")
            )
        else:
            from model import build_model
            with open(DATA_DIR / "vocab.json", encoding="utf-8") as f:
                self.vocab = json.load(f)
            if self.config.get("model_type") == "bilstm_crf_char":
                char_path = DATA_DIR / "char_vocab.json"
                if char_path.exists():
                    with open(char_path, encoding="utf-8") as f:
                        self.char_vocab = json.load(f)
            self.config["num_tags"]   = len(self.tag_vocab)
            self.config["vocab_size"] = len(self.vocab)
            # Strip non-serialisable tensors that were saved into config
            self.config.pop("class_weights", None)
            self.config.pop("boost_tags", None)
            self.model = build_model(self.config)
            self.model.load_state_dict(state_dict)

        self.model.eval()
        print(f"✓ Loaded model '{self.model_name}' ({self.config.get('model_type', 'xlmr')})")

    # ── tokenise Myanmar text into whitespace-split words
    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return text.strip().split()

    def _predict_bilstm(self, words: list[str]) -> list[str]:
        unk      = self.vocab.get("<UNK>", 1)
        word_ids = torch.tensor([[self.vocab.get(w, unk) for w in words]])

        with torch.no_grad():
            if self.config.get("model_type") == "bilstm_crf_char":
                max_char = self.config.get("max_char_len", 30)
                unk_c    = self.char_vocab.get("<UNK_CHAR>", 1)
                char_ids = []
                for w in words:
                    ids  = [self.char_vocab.get(c, unk_c) for c in list(w)[:max_char]]
                    ids += [0] * (max_char - len(ids))
                    char_ids.append(ids)
                x_chars  = torch.tensor([char_ids])
                pred_ids = self.model.decode(word_ids, x_chars)
            else:
                pred_ids = self.model.decode(word_ids)

        return [self.idx_to_tag.get(i, "O") for i in pred_ids[0]]

    def _predict_xlmr(self, words: list[str]) -> list[str]:
        from xlmr_dataset_loader import IGNORE_IDX

        max_length = self.config.get("max_length", 256)
        enc        = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        word_ids_enc = enc.word_ids()

        # Build label_ids: placeholder tag id at first piece, IGNORE_IDX elsewhere
        label_ids = []
        seen: set[int] = set()
        for wid in word_ids_enc:
            if wid is None:
                label_ids.append(IGNORE_IDX)
            elif wid not in seen:
                label_ids.append(1)   # placeholder — only position matters
                seen.add(wid)
            else:
                label_ids.append(IGNORE_IDX)

        label_ids_t  = torch.tensor([label_ids])
        word_lengths = torch.tensor([len(words)])

        with torch.no_grad():
            pred_ids = self.model.decode(
                enc["input_ids"], enc["attention_mask"],
                label_ids_t, word_lengths,
            )
        return [self.idx_to_tag.get(i, "O") for i in pred_ids[0]]

    def predict(self, text: str) -> list[dict]:
        """
        Returns a list of token dicts:
            { word, tag, entity_type, color, is_entity }
        """
        words = self._tokenise(text)
        if not words:
            return []

        tags  = self._predict_bilstm(words) if not self.is_xlmr else self._predict_xlmr(words)

        result = []
        for word, tag in zip(words, tags):
            is_entity   = tag != "O"
            entity_type = tag.split("-", 1)[1] if is_entity and "-" in tag else None
            result.append({
                "word":        word,
                "tag":         tag,
                "entity_type": entity_type,
                "color":       ENTITY_COLORS.get(entity_type or "", "#94a3b8"),
                "is_entity":   is_entity,
            })
        return result


# ─────────────────────────────────────────────
# App + lifespan
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_ner_model() -> NERModel:
    return NERModel(MODEL_NAME)


app = FastAPI(
    title="Myanmar NER API",
    description="Named Entity Recognition for Burmese text",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description="Space-separated Myanmar words to tag")

class TokenResult(BaseModel):
    word:        str
    tag:         str
    entity_type: Optional[str]
    color:       str
    is_entity:   bool

class PredictResponse(BaseModel):
    tokens:      list[TokenResult]
    model_name:  str
    num_tokens:  int
    num_entities: int
    latency_ms:  float
    entity_counts: dict[str, int]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/entity-colors")
def entity_colors():
    return ENTITY_COLORS


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        ner    = get_ner_model()
        t0     = time.perf_counter()
        tokens = ner.predict(req.text)
        ms     = (time.perf_counter() - t0) * 1000

        entity_counts: dict[str, int] = {}
        for tok in tokens:
            if tok["is_entity"] and tok["tag"].startswith("S-") or tok["tag"].startswith("B-"):
                et = tok["entity_type"] or "LOC"
                entity_counts[et] = entity_counts.get(et, 0) + 1

        return PredictResponse(
            tokens        = tokens,
            model_name    = ner.model_name,
            num_tokens    = len(tokens),
            num_entities  = sum(entity_counts.values()),
            latency_ms    = round(ms, 2),
            entity_counts = entity_counts,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

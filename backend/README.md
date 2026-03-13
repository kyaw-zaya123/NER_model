# Myanmar NER — Backend

FastAPI inference server for the Myanmar NER model.

## Setup

```bash
cd backend

# Copy your trained artifacts here
cp -r ../checkpoints ./checkpoints
cp -r ../data        ./data
cp    ../model.py    .
cp    ../xlmr_model.py .
cp    ../xlmr_dataset_loader.py .

# Install dependencies
pip install -r requirements.txt

# Run (defaults to bilstm_crf)
uvicorn main:app --reload --port 8000

# Run with a different model
NER_MODEL=bilstm_crf_char uvicorn main:app --reload --port 8000
NER_MODEL=xlmr_base       uvicorn main:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/entity-colors` | Colour map for entity types |
| POST | `/predict` | Run NER on input text |

### POST /predict

**Request:**
```json
{ "text": "ဘူးသီးတောင်မြို့နယ် မှ ဒေသခံ တစ်ဦး" }
```

**Response:**
```json
{
  "tokens": [
    { "word": "ဘူးသီးတောင်မြို့နယ်", "tag": "S-LOC-TOWNSHIP",
      "entity_type": "LOC-TOWNSHIP", "color": "#38bdf8", "is_entity": true },
    { "word": "မှ", "tag": "O", "entity_type": null, "color": "#94a3b8", "is_entity": false }
  ],
  "model_name": "bilstm_crf",
  "num_tokens": 4,
  "num_entities": 1,
  "latency_ms": 12.4,
  "entity_counts": { "LOC-TOWNSHIP": 1 }
}
```

## Directory layout expected

```
backend/
├── main.py
├── model.py                  ← copy from project root
├── xlmr_model.py             ← copy from project root
├── xlmr_dataset_loader.py    ← copy from project root
├── requirements.txt
├── checkpoints/
│   └── bilstm_crf/
│       └── best_model.pt
└── data/
    └── processed/
        ├── vocab.json
        ├── tag_vocab.json
        ├── char_vocab.json   ← only for bilstm_crf_char
        └── pos_vocab.json
```

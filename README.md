# Myanmar NER — Web Application

A full-stack web app for Myanmar Named Entity Recognition, built with **React** (GitHub Pages) and **FastAPI** (self-hosted or any cloud VM).

```
Frontend  →  GitHub Pages       (free, auto-deployed via GitHub Actions)
Backend   →  Your server / VM   (FastAPI + your best_model.pt)
```

---

## Architecture

```
┌─────────────────────────────────┐       ┌──────────────────────────────┐
│  GitHub Pages                   │       │  Your server (port 8000)     │
│                                 │  HTTP │                              │
│  React frontend                 │──────▶│  FastAPI                     │
│  • textarea input               │  POST │  • loads best_model.pt       │
│  • colour-coded entity spans    │◀──────│  • returns token + tag JSON  │
│  • click-to-inspect tokens      │       │                              │
└─────────────────────────────────┘       └──────────────────────────────┘
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/myanmar-ner.git
cd myanmar-ner
```

### 2. Set up the backend

```bash
cd backend

# Copy your trained artifacts
cp -r ../checkpoints ./checkpoints      # contains best_model.pt
cp -r ../data        ./data             # contains vocab JSON files
cp    ../model.py                  .
cp    ../xlmr_model.py             .
cp    ../xlmr_dataset_loader.py    .

# Install dependencies
pip install -r requirements.txt

# Run (defaults to bilstm_crf)
uvicorn main:app --host 0.0.0.0 --port 8000

# Or choose a different model
NER_MODEL=bilstm_crf_char uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be live at `http://your-server-ip:8000`.  
Swagger docs at `http://your-server-ip:8000/docs`.

### 3. Set up the frontend locally

```bash
cd frontend
cp .env.example .env.local
# Edit .env.local and set REACT_APP_API_URL=http://your-server-ip:8000

npm install
npm start      # http://localhost:3000
```

---

## Deploy the Frontend to GitHub Pages

### Step 1 — Enable GitHub Pages

In your repository: **Settings → Pages → Source → GitHub Actions**

### Step 2 — Add your backend URL as a secret

**Settings → Secrets and variables → Actions → New repository secret**

```
Name:  REACT_APP_API_URL
Value: http://your-server-ip:8000
```

### Step 3 — Push to main

```bash
git add .
git commit -m "deploy"
git push origin main
```

The GitHub Action (`.github/workflows/deploy-frontend.yml`) builds the React app and publishes it automatically. Your site will be live at:

```
https://your-username.github.io/myanmar-ner/
```

---

## Backend: required file layout

```
backend/
├── main.py
├── model.py                   ← from project root
├── xlmr_model.py              ← from project root
├── xlmr_dataset_loader.py     ← from project root
├── requirements.txt
├── checkpoints/
│   └── bilstm_crf/            ← or bilstm_crf_char / xlmr_base
│       └── best_model.pt
└── data/
    └── processed/
        ├── vocab.json
        ├── tag_vocab.json
        └── char_vocab.json    ← only needed for bilstm_crf_char
```

### Supported models

| `NER_MODEL` env var | Architecture |
|---|---|
| `bilstm_softmax` | BiLSTM + Softmax |
| `bilstm_crf` | BiLSTM + CRF (default) |
| `bilstm_crf_char` | BiLSTM + CharCNN + CRF |
| `xlmr_base` | XLM-RoBERTa + CRF |
| `xlmr_base_frozen` | XLM-RoBERTa (frozen) + CRF |

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "ဘူးသီးတောင်မြို့နယ် မှ မှိုင်းဝေ သည် PDF ကို ဆန့်ကျင်ခဲ့သည်"}'
```

**Response:**
```json
{
  "tokens": [
    { "word": "ဘူးသီးတောင်မြို့နယ်", "tag": "S-LOC-TOWNSHIP",
      "entity_type": "LOC-TOWNSHIP", "color": "#38bdf8", "is_entity": true },
    { "word": "မှ",   "tag": "O", "entity_type": null, "color": "#94a3b8", "is_entity": false },
    { "word": "မှိုင်းဝေ", "tag": "S-PER",
      "entity_type": "PER", "color": "#f87171", "is_entity": true }
  ],
  "model_name":    "bilstm_crf",
  "num_tokens":    7,
  "num_entities":  3,
  "latency_ms":    14.2,
  "entity_counts": { "LOC-TOWNSHIP": 1, "PER": 1, "ORG": 1 }
}
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/entity-colors` | GET | Colour map for all entity types |
| `/predict` | POST | Run NER on input text |
| `/docs` | GET | Swagger UI |

---

## Entity Types

| Tag | Colour | Description |
|---|---|---|
| `PER` | 🔴 | Person name |
| `ORG` | 🟢 | Organisation |
| `DATE` | 🟡 | Date expression |
| `TIME` | 🟠 | Time expression |
| `NUM` | 🟣 | Number |
| `LOC` | 🔵 | Generic location |
| `LOC-COUNTRY` | 🔵 | Country |
| `LOC-STATE` | 💚 | State / Region |
| `LOC-DISTRICT` | 🩵 | District |
| `LOC-TOWNSHIP` | 🔵 | Township |
| `LOC-CITY` | 🩵 | City |
| `LOC-VILLAGE` | 🟢 | Village |
| `LOC-WARD` | 🔵 | Ward |

---

## Project Structure

```
myanmar-ner/
├── .github/workflows/
│   ├── deploy-frontend.yml   ← auto-deploy React to GitHub Pages on push
│   └── backend-ci.yml        ← lint + smoke-test backend on every PR
├── frontend/
│   ├── public/index.html
│   ├── src/
│   │   ├── App.js
│   │   ├── index.js / index.css
│   │   ├── hooks/useNER.js
│   │   └── components/
│   │       ├── TokenDisplay.js
│   │       ├── EntityLegend.js
│   │       └── StatsBar.js
│   ├── .env.example
│   └── package.json
├── backend/
│   ├── main.py
│   └── requirements.txt
└── .gitignore
```

# F1Goat 🐐🏁

An interactive Formula 1 GOAT (Greatest Of All Time) ranking engine.

This project builds career-level driver features from historical F1 data, scores drivers across multiple dimensions, and serves an interactive ranking UI via a FastAPI backend and Next.js frontend.

---

## Features

- 🏆 Career-based GOAT scoring system
- 📈 Adjustable weight sliders (Career, Peak, Context, Longevity, Qualifying)
- 🔄 Optional era normalization
- 📊 Minimum starts filter
- 🚀 FastAPI backend
- 💻 Next.js frontend
- 🗂 Parquet-based data pipeline for fast analytics

---

## Scoring Model (High Level)

Each driver receives five sub-scores:

- Career
- Peak
- Context (car strength / overachieve proxy)
- Longevity
- Qualifying

Final GOAT score:

GOAT = (career*w1 + peak*w2 + context*w3 + longevity*w4 + quali*w5)

Weights are automatically normalized so they always sum to 1.

An optional era multiplier can be applied if era normalization is enabled.

---

## Project Structure

```
F1Goat/
│
├── archive/                        # Raw CSV data
├── parquet_out/                    # Generated parquet files
│
├── build_parquet.py                # Converts CSV → parquet
├── backend_goat.py                 # Feature engineering + scoring logic
├── main.py                         # FastAPI app
│
├── frontend/                       # Next.js frontend
│   ├── app/
│   ├── package.json
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## Backend Setup (Python)

### 1) Create a virtual environment

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3) Build parquet tables

```bash
python build_parquet.py
```

This creates processed parquet files inside:

```
parquet_out/
```

---

### 4) Build driver career features

```bash
python backend_goat.py
```

This generates:

```
parquet_out/driver_career_features.parquet
```

---

### 5) Run the API

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Available endpoints:

- GET /health
- POST /rank

API will run at:

```
http://127.0.0.1:8000
```

---

## Frontend Setup (Next.js)

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```
http://localhost:3000
```

---

## Example API Request

```bash
curl -X POST http://127.0.0.1:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "weights": {
      "career": 0.30,
      "peak": 0.25,
      "context": 0.20,
      "longevity": 0.15,
      "quali": 0.10
    },
    "era_normalize": true,
    "min_starts": 30,
    "top_n": 50
  }'
```

---

## Troubleshooting

### If the API fails on startup

Make sure you ran:

```bash
python build_parquet.py
python backend_goat.py
```

and that `parquet_out/driver_career_features.parquet` exists.

---

### If the frontend cannot connect to backend

- Confirm API is running on:
  ```
  http://127.0.0.1:8000
  ```

- Ensure CORS in `main.py` allows:
  ```
  http://localhost:3000
  ```

---

## Notes

- This is a customizable scoring engine — the GOAT ranking depends on how you weight each category.
- Era normalization can significantly change rankings.
- Data processing is done once via parquet for performance.

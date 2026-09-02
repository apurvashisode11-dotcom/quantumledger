# QuantumLedger

Real-time financial fraud detection using a Graph Neural Network, built end-to-end: streaming ingestion, GNN-based detection, a persistent database, a REST + WebSocket API, and a live React dashboard.

Solo learning project — built to deeply understand every layer of a real-time ML system, not just train a model in a notebook.

## What it does

Detects two common forms of market manipulation in a live stream of trades:
- **Spoofing** — a trader placing abnormally large orders to create false market pressure
- **Wash trading** — two traders repeatedly trading with each other to fake volume/activity

Trades flow continuously through the system; each one is scored in real time by a trained Graph Neural Network, and results are stored, served over an API, and streamed live to a dashboard.

## Architecture

```
trade_generator.py (Kafka producer, synthetic trades + labeled fraud)
        │
        ▼
   Kafka topic: raw-trades
        │
        ▼
  live_detector.py (consumer)
   - builds a live trader graph from a rolling window of recent trades
   - runs the trained GNN for fraud scoring
   - writes trades + predictions to TimescaleDB
        │
        ▼
  TimescaleDB (trades, fraud_alerts tables)
        │
        ▼
   FastAPI (api.py)
   - REST: /trades/recent, /alerts, /alerts/stats, /trader/{id}
   - WebSocket: /ws/live — pushes new fraud alerts in real time
        │
        ▼
   React dashboard (dashboard/)
   - live trade feed, live fraud alerts, live accuracy stats
```

## Key technical decisions & findings

**Label leakage.** An early version of the GNN included a `fraud_ratio` node feature that was derived from the labels being predicted. Removing it and replacing it with a legitimate behavioral feature (trade volume standard deviation) improved F1 from 0.64 to 0.71 — the leaked feature was actually a shortcut that hurt generalization.

**Dataset scaling.** Growing the training set from 500 → 15,000 trades closed the train/test generalization gap from 3.5 points down to under 1 point, and pushed recall to 95.7% on held-out static test data.

**Offline vs. live distribution shift.** The model trained on one static graph performed far worse in live streaming conditions (68.5% accuracy) than in offline evaluation (86.1%). Root cause: node features are computed relative to whatever's in the current rolling window, and live windows have different statistical compositions than the fixed training graph. Fixed by retraining directly on hundreds of overlapping sliding-window graph snapshots (mirroring production conditions), with a time-based train/test split to avoid leakage between overlapping windows. This closed the gap — live accuracy improved to 88–94% across multiple sustained sessions.

**Live-validated results** (largest session, ~9,000 predictions): 88.1% accuracy, 77.9% precision, 90.1% recall, F1 0.84.

## Tech stack

- **Streaming:** Apache Kafka (KRaft mode, no Zookeeper)
- **ML:** PyTorch, PyTorch Geometric (GraphSAGE-based edge classifier), CUDA (RTX 3070 Ti)
- **Database:** TimescaleDB (PostgreSQL + time-series hypertables)
- **Backend:** FastAPI, psycopg2, WebSockets
- **Frontend:** React + Vite
- **Infra:** Docker Compose, WSL2

## Project structure

```
trade_generator.py        # Kafka producer: continuous trades + injected fraud
bulk_generator.py         # Fast, no-delay version for bulk dataset generation
feature_processor.py      # Phase 2: sliding-window feature extraction
collect_trades.py         # Pulls a batch of trades from Kafka into a CSV
build_graph.py            # CSV -> PyTorch Geometric graph
gnn_model.py               # FraudGNN model definition (GraphSAGE + edge classifier)
train_gnn.py               # Static-graph training
train_gnn_windowed.py      # Sliding-window training (fixes distribution shift)
evaluate_gnn.py             # Offline evaluation (precision/recall/F1)
live_detector.py            # Live Kafka consumer -> GNN inference -> DB writes
setup_database.py           # Creates trades / fraud_alerts tables
api.py                      # FastAPI REST + WebSocket backend
dashboard/                  # React frontend
```

## Running it locally

Requires Docker, Python 3.11, Node.js.

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Set up the database (first time only)
python3 setup_database.py

# 3. Terminal 1 — trade generator
source venv/bin/activate
python3 trade_generator.py

# 4. Terminal 2 — live detector (GNN inference + DB writes)
source venv311/bin/activate
python3 live_detector.py

# 5. Terminal 3 — API server
source venv311/bin/activate
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# 6. Terminal 4 — dashboard
cd dashboard
npm install
npm run dev
```

Dashboard: http://localhost:5173
API docs: http://localhost:8000/docs

## Known limitations / future work

- WebSocket bridge uses database polling (1s interval) rather than a message broker (Redis pub/sub or direct Kafka consumption) — a reasonable trade-off at this scale, but wouldn't hold up at high throughput
- Precision (~72–82%) has room to improve — some false positives on legitimate traders who coincidentally trade with the same counterparty a few times in one window
- No formal throughput/latency load testing yet against original design targets
- SHAP-style explainability (why a specific trade was flagged) not yet implemented
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import psycopg2
import psycopg2.extras
from psycopg2 import pool
from datetime import datetime
from contextlib import contextmanager
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json

app = FastAPI(title="QuantumLedger API")

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "quantumledger",
    "user": "quantumledger",
    "password": "quantumledger_dev",
}

# minconn=2, maxconn=10 — small pool is plenty for a solo/demo project;
# tune maxconn up only if you actually see "pool exhausted" errors under load
connection_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=2,
    maxconn=10,
    **DB_CONFIG,
)

@contextmanager
def get_conn():
    """Borrow a connection from the pool, always return it — even on error."""
    conn = connection_pool.getconn()
    try:
        yield conn
    finally:
        connection_pool.putconn(conn)


@app.get("/")
def root():
    return {"status": "ok", "service": "QuantumLedger API"}


# ---------- Response models ----------

class Trade(BaseModel):
    id: int
    trade_id: str
    trader_id: str
    counterparty_id: str
    price: float
    volume: float
    timestamp: datetime
    is_fraud_actual: Optional[bool]
    fraud_type: Optional[str]

class Alert(BaseModel):
    id: int
    trade_id: str
    trader_id: str
    counterparty_id: str
    volume: float
    predicted_fraud: bool
    confidence: float
    detected_at: datetime

class AlertStats(BaseModel):
    total_alerts: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1: float


# ---------- Endpoints ----------

@app.get("/trades/recent", response_model=list[Trade])
def recent_trades(limit: int = 50):
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, trade_id, trader_id, counterparty_id, price, volume,
                   timestamp, is_fraud_actual, fraud_type
            FROM trades
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return rows


@app.get("/alerts", response_model=list[Alert])
def recent_alerts(limit: int = 50, fraud_only: bool = False):
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query = """
            SELECT id, trade_id, trader_id, counterparty_id, volume,
                   predicted_fraud, confidence, detected_at
            FROM fraud_alerts
        """
        params = []
        if fraud_only:
            query += " WHERE predicted_fraud = TRUE"
        query += " ORDER BY detected_at DESC LIMIT %s"
        params.append(limit)

        cur.execute(query, params)
        rows = cur.fetchall()
    return rows


@app.get("/alerts/stats", response_model=AlertStats)
def alert_stats():
    """
    Joins fraud_alerts back to trades on trade_id to compare
    predicted_fraud (model output) against is_fraud_actual (ground truth),
    then computes a live confusion matrix + precision/recall/F1.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                SUM(CASE WHEN a.predicted_fraud = TRUE AND t.is_fraud_actual = TRUE THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN a.predicted_fraud = TRUE AND t.is_fraud_actual = FALSE THEN 1 ELSE 0 END) AS fp,
                SUM(CASE WHEN a.predicted_fraud = FALSE AND t.is_fraud_actual = FALSE THEN 1 ELSE 0 END) AS tn,
                SUM(CASE WHEN a.predicted_fraud = FALSE AND t.is_fraud_actual = TRUE THEN 1 ELSE 0 END) AS fn,
                COUNT(*) AS total
            FROM fraud_alerts a
            JOIN trades t ON a.trade_id = t.trade_id
            """
        )
        tp, fp, tn, fn, total = cur.fetchone()
        tp, fp, tn, fn, total = (tp or 0), (fp or 0), (tn or 0), (fn or 0), (total or 0)

    if total == 0:
        raise HTTPException(status_code=404, detail="No alert data yet")

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return AlertStats(
        total_alerts=total,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


@app.get("/trader/{trader_id}")
def trader_history(trader_id: str, limit: int = 50):
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, trade_id, trader_id, counterparty_id, price, volume,
                   timestamp, is_fraud_actual, fraud_type
            FROM trades
            WHERE trader_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (trader_id, limit),
        )
        trades = cur.fetchall()

        if not trades:
            raise HTTPException(status_code=404, detail=f"No trades found for trader {trader_id}")

        cur.execute(
            """
            SELECT COUNT(*) FROM fraud_alerts WHERE trader_id = %s AND predicted_fraud = TRUE
            """,
            (trader_id,),
        )
        flagged_count = cur.fetchone()[0]

    return {
        "trader_id": trader_id,
        "trade_count": len(trades),
        "flagged_alert_count": flagged_count,
        "trades": trades,
    }


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    last_alert_id = 0

    # On connect, find the current max alert id so we only stream NEW data forward
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(id), 0) FROM fraud_alerts")
        last_alert_id = cur.fetchone()[0]

    try:
        while True:
            await asyncio.sleep(1)  # poll interval

            with get_conn() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(
                    """
                    SELECT id, trade_id, trader_id, counterparty_id, volume,
                           predicted_fraud, confidence, detected_at
                    FROM fraud_alerts
                    WHERE id > %s
                    ORDER BY id ASC
                    """,
                    (last_alert_id,),
                )
                new_alerts = cur.fetchall()

            if new_alerts:
                for alert in new_alerts:
                    alert_dict = dict(alert)
                    alert_dict['detected_at'] = alert_dict['detected_at'].isoformat()
                    alert_dict['confidence'] = float(alert_dict['confidence'])  # Decimal -> float
                    alert_dict['volume'] = float(alert_dict['volume'])  # in case volume is also Decimal/numeric
                    await websocket.send_text(json.dumps(alert_dict))
                last_alert_id = new_alerts[-1]['id']

    except WebSocketDisconnect:
        print("Client disconnected from /ws/live")
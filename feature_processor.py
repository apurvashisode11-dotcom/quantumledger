from kafka import KafkaConsumer, KafkaProducer
import json
from datetime import datetime, UTC
from collections import deque, defaultdict

consumer = KafkaConsumer(
    'raw-trades',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='latest'
)

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

WINDOW_SECONDS = 30

# Each trader gets their own rolling history of recent trades
trader_windows = defaultdict(deque)

def clean_old_trades(trader_id, now):
    """Remove trades older than WINDOW_SECONDS from this trader's history"""
    window = trader_windows[trader_id]
    while window and (now - window[0]['ts']).total_seconds() > WINDOW_SECONDS:
        window.popleft()

def compute_features(trader_id, current_trade, now):
    window = trader_windows[trader_id]

    trade_count = len(window)
    avg_volume = sum(t['volume'] for t in window) / trade_count if trade_count > 0 else current_trade['volume']
    volume_spike_ratio = round(current_trade['volume'] / avg_volume, 2) if avg_volume > 0 else 1.0
    unique_counterparties = len(set(t['counterparty'] for t in window))

    return {
        "trader_id": trader_id,
        "trade_id": current_trade['trade_id'],
        "trade_count_30s": trade_count,
        "avg_volume_30s": round(avg_volume, 2),
        "volume_spike_ratio": volume_spike_ratio,
        "unique_counterparties_30s": unique_counterparties,
        "current_price": current_trade['price'],
        "current_volume": current_trade['volume'],
        "timestamp": now.isoformat(),
        "is_fraud": current_trade['is_fraud'],
        "fraud_type": current_trade['fraud_type']
    }

print(f"Processing trades with {WINDOW_SECONDS}s sliding window... (Ctrl+C to stop)")

try:
    for message in consumer:
        trade = message.value
        trader_id = trade['trader_id']
        now = datetime.now(UTC)

        clean_old_trades(trader_id, now)

        features = compute_features(trader_id, trade, now)

        producer.send('trade-features', value=features)

        flag = "🚩" if trade['is_fraud'] else "  "
        print(f"{flag} {trader_id} | trades_30s={features['trade_count_30s']} "
              f"avg_vol={features['avg_volume_30s']} spike_ratio={features['volume_spike_ratio']} "
              f"unique_cp={features['unique_counterparties_30s']}")

        # Add current trade to window AFTER computing features (so features reflect history, not including itself)
        trader_windows[trader_id].append({
            "ts": now,
            "volume": trade['volume'],
            "counterparty": trade['counterparty_id']
        })

except KeyboardInterrupt:
    print("\nStopped feature processor.")
    producer.flush()
    producer.close()

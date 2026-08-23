from kafka import KafkaProducer
import json
import random
import time
from datetime import datetime, UTC

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_normal_trade():
    return {
        "trade_id": random.randint(100000, 999999),
        "trader_id": f"T{random.randint(1, 50):03d}",
        "counterparty_id": f"T{random.randint(1, 50):03d}",
        "price": round(random.uniform(100, 500), 2),
        "volume": random.randint(1, 1000),
        "timestamp": datetime.now(UTC).isoformat(),
        "is_fraud": False,
        "fraud_type": None
    }

def generate_wash_trade_burst():
    trader_a = f"T{random.randint(1, 50):03d}"
    trader_b = f"T{random.randint(1, 50):03d}"
    base_price = round(random.uniform(100, 500), 2)
    burst = []
    for i in range(5):
        buyer, seller = (trader_a, trader_b) if i % 2 == 0 else (trader_b, trader_a)
        burst.append({
            "trade_id": random.randint(100000, 999999),
            "trader_id": buyer,
            "counterparty_id": seller,
            "price": round(base_price + random.uniform(-0.5, 0.5), 2),
            "volume": random.randint(400, 600),
            "timestamp": datetime.now(UTC).isoformat(),
            "is_fraud": True,
            "fraud_type": "wash_trading"
        })
    return burst

def generate_spoofing_burst():
    trader = f"T{random.randint(1, 50):03d}"
    burst = []
    for i in range(4):
        burst.append({
            "trade_id": random.randint(100000, 999999),
            "trader_id": trader,
            "counterparty_id": f"T{random.randint(1, 50):03d}",
            "price": round(random.uniform(100, 500), 2),
            "volume": random.randint(5000, 10000),
            "timestamp": datetime.now(UTC).isoformat(),
            "is_fraud": True,
            "fraud_type": "spoofing"
        })
    return burst

TARGET = 10000
count = 0
trades_until_fraud = random.randint(15, 20)

print(f"Bulk generating {TARGET} trades (no delay, for dataset building)...")

while count < TARGET:
    if count > 0 and count % trades_until_fraud == 0:
        burst = random.choice([generate_wash_trade_burst, generate_spoofing_burst])()
        for trade in burst:
            producer.send('raw-trades', value=trade)
            count += 1
        trades_until_fraud = random.randint(15, 20)
    else:
        producer.send('raw-trades', value=generate_normal_trade())
        count += 1

    if count % 500 == 0:
        print(f"Generated {count}/{TARGET}")

producer.flush()
producer.close()
print(f"Done. Total sent: {count}")

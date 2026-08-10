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
    trade = {
        "trade_id": random.randint(100000, 999999),
        "trader_id": f"T{random.randint(1, 50):03d}",
        "counterparty_id": f"T{random.randint(1, 50):03d}",
        "price": round(random.uniform(100, 500), 2),
        "volume": random.randint(1, 1000),
        "timestamp": datetime.now(UTC).isoformat(),
        "is_fraud": False,
        "fraud_type": None
    }
    return trade

def generate_wash_trade_burst():
    """Same two traders trading back and forth rapidly - fake volume"""
    trader_a = f"T{random.randint(1, 50):03d}"
    trader_b = f"T{random.randint(1, 50):03d}"
    base_price = round(random.uniform(100, 500), 2)
    burst = []
    for i in range(5):
        buyer, seller = (trader_a, trader_b) if i % 2 == 0 else (trader_b, trader_a)
        trade = {
            "trade_id": random.randint(100000, 999999),
            "trader_id": buyer,
            "counterparty_id": seller,
            "price": round(base_price + random.uniform(-0.5, 0.5), 2),
            "volume": random.randint(400, 600),
            "timestamp": datetime.now(UTC).isoformat(),
            "is_fraud": True,
            "fraud_type": "wash_trading"
        }
        burst.append(trade)
    return burst

def generate_spoofing_burst():
    """One trader places abnormally large orders in quick succession"""
    trader = f"T{random.randint(1, 50):03d}"
    burst = []
    for i in range(4):
        trade = {
            "trade_id": random.randint(100000, 999999),
            "trader_id": trader,
            "counterparty_id": f"T{random.randint(1, 50):03d}",
            "price": round(random.uniform(100, 500), 2),
            "volume": random.randint(5000, 10000),  # abnormally large
            "timestamp": datetime.now(UTC).isoformat(),
            "is_fraud": True,
            "fraud_type": "spoofing"
        }
        burst.append(trade)
    return burst

print("Starting continuous trade generation with fraud injection... (Ctrl+C to stop)")
count = 0
trades_until_fraud = random.randint(15, 20)

try:
    while True:
        if count > 0 and count % trades_until_fraud == 0:
            fraud_type = random.choice(["wash", "spoof"])
            burst = generate_wash_trade_burst() if fraud_type == "wash" else generate_spoofing_burst()
            print(f"\n*** INJECTING {burst[0]['fraud_type'].upper()} PATTERN ***")
            for trade in burst:
                producer.send('raw-trades', value=trade)
                count += 1
                print(f"[{count}] FRAUD Sent: {trade}")
                time.sleep(0.3)
            print("*** END FRAUD PATTERN ***\n")
            trades_until_fraud = random.randint(15, 20)
        else:
            trade = generate_normal_trade()
            producer.send('raw-trades', value=trade)
            count += 1
            print(f"[{count}] Sent: {trade}")
            time.sleep(1)
except KeyboardInterrupt:
    print(f"\nStopped. Total trades sent: {count}")
    producer.flush()
    producer.close()

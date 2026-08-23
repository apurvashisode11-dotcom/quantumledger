from kafka import KafkaConsumer
import json
import pandas as pd

TARGET_COUNT = 15000

consumer = KafkaConsumer(
    'raw-trades',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest'
)

trades = []
print(f"Collecting {TARGET_COUNT} trades from 'raw-trades'...")

for message in consumer:
    trades.append(message.value)
    print(f"Collected {len(trades)}/{TARGET_COUNT}", end='\r')
    if len(trades) >= TARGET_COUNT:
        break

consumer.close()

df = pd.DataFrame(trades)
df.to_csv('trades_dataset.csv', index=False)

print(f"\nSaved {len(df)} trades to trades_dataset.csv")
print(f"Fraud trades: {df['is_fraud'].sum()}")
print(f"Normal trades: {(~df['is_fraud']).sum()}")

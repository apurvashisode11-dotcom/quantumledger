import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="quantumledger",
    user="quantumledger",
    password="quantumledger_dev"
)
conn.autocommit = True
cur = conn.cursor()

print("Connected to TimescaleDB successfully.")

# Drop tables if they exist, to start clean
cur.execute("DROP TABLE IF EXISTS trades CASCADE;")
cur.execute("DROP TABLE IF EXISTS fraud_alerts CASCADE;")

# Create trades table - primary key includes timestamp, required for hypertables
cur.execute("""
CREATE TABLE trades (
    id SERIAL,
    trade_id BIGINT NOT NULL,
    trader_id TEXT NOT NULL,
    counterparty_id TEXT NOT NULL,
    price NUMERIC NOT NULL,
    volume INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    is_fraud_actual BOOLEAN,
    fraud_type TEXT,
    PRIMARY KEY (id, timestamp)
);
""")
print("Created 'trades' table.")

# Create fraud_alerts table (no need for hypertable here, smaller/simpler table)
cur.execute("""
CREATE TABLE fraud_alerts (
    id SERIAL PRIMARY KEY,
    trade_id BIGINT NOT NULL,
    trader_id TEXT NOT NULL,
    counterparty_id TEXT NOT NULL,
    volume INTEGER NOT NULL,
    predicted_fraud BOOLEAN NOT NULL,
    confidence NUMERIC NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""")
print("Created 'fraud_alerts' table.")

# Turn 'trades' into a TimescaleDB hypertable
cur.execute("""
SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);
""")
print("Converted 'trades' to a TimescaleDB hypertable.")

cur.close()
conn.close()
print("\nDatabase setup complete.")

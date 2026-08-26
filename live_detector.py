from kafka import KafkaConsumer
import json
import torch
import pandas as pd
from collections import deque
from datetime import datetime, UTC
from torch_geometric.data import Data
from gnn_model import FraudGNN

GRAPH_REBUILD_INTERVAL = 20
HISTORY_SIZE = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = FraudGNN(node_feature_dim=5, edge_feature_dim=2).to(device)
model.load_state_dict(torch.load('fraud_gnn_windowed.pt', weights_only=True))
model.eval()
print("Loaded trained GNN model.")

trade_history = deque(maxlen=HISTORY_SIZE)
results_log = []

def build_live_graph(trades):
    df = pd.DataFrame(trades)
    all_traders = pd.concat([df['trader_id'], df['counterparty_id']]).unique()
    trader_to_idx = {t: i for i, t in enumerate(all_traders)}

    source_nodes = df['trader_id'].map(trader_to_idx).tolist()
    target_nodes = df['counterparty_id'].map(trader_to_idx).tolist()
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_features = torch.tensor(df[['price', 'volume']].values, dtype=torch.float)

    node_features_list = []
    for trader in all_traders:
        as_trader = df[df['trader_id'] == trader]
        as_counterparty = df[df['counterparty_id'] == trader]
        all_involved = pd.concat([as_trader, as_counterparty])
        total_trades = len(all_involved)
        avg_volume = all_involved['volume'].mean() if total_trades > 0 else 0
        avg_price = all_involved['price'].mean() if total_trades > 0 else 0
        counterparties = pd.concat([as_trader['counterparty_id'], as_counterparty['trader_id']]).unique()
        unique_counterparties = len(counterparties)
        volume_std = all_involved['volume'].std() if total_trades > 1 else 0
        node_features_list.append([total_trades, avg_volume, avg_price, unique_counterparties, volume_std])

    node_features = torch.tensor(node_features_list, dtype=torch.float)
    node_features = torch.nan_to_num(node_features, nan=0.0)

    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    return graph.to(device), df, trader_to_idx

def print_summary():
    if not results_log:
        print("No predictions logged yet.")
        return
    df = pd.DataFrame(results_log)
    tp = ((df['pred'] == 1) & (df['actual'] == 1)).sum()
    fp = ((df['pred'] == 1) & (df['actual'] == 0)).sum()
    tn = ((df['pred'] == 0) & (df['actual'] == 0)).sum()
    fn = ((df['pred'] == 0) & (df['actual'] == 1)).sum()

    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*50}")
    print(f"LIVE SESSION SUMMARY - {len(df)} predictions logged")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"{'='*50}\n")

consumer = KafkaConsumer(
    'raw-trades',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='latest'
)

print(f"\nLive fraud detection started. Rebuilding graph every {GRAPH_REBUILD_INTERVAL} trades.")
print("Press Ctrl+C to stop and see summary.\n")

trade_count = 0

try:
    for message in consumer:
        trade = message.value
        trade_history.append(trade)
        trade_count += 1

        if trade_count % GRAPH_REBUILD_INTERVAL == 0 and len(trade_history) >= 10:
            current_graph, current_df, trader_map = build_live_graph(list(trade_history))
            with torch.no_grad():
                out = model(current_graph.x, current_graph.edge_index, current_graph.edge_attr)
                preds = out.argmax(dim=1)
                probs = torch.softmax(out, dim=1)[:, 1]

            recent_n = min(GRAPH_REBUILD_INTERVAL, len(current_df))
            recent_preds = preds[-recent_n:]
            recent_probs = probs[-recent_n:]
            recent_rows = current_df.iloc[-recent_n:]

            for (idx, row), pred, prob in zip(recent_rows.iterrows(), recent_preds, recent_probs):
                actual_label = 1 if row['is_fraud'] else 0
                pred_label = pred.item()

                results_log.append({'pred': pred_label, 'actual': actual_label})

                if pred_label == 1 and actual_label == 1:
                    print(f"FRAUD FLAGGED | {row['trader_id']} -> {row['counterparty_id']} | vol={row['volume']} | confidence={prob.item():.2f} | CORRECT")
                elif pred_label == 1 and actual_label == 0:
                    print(f"FALSE ALARM | {row['trader_id']} -> {row['counterparty_id']} | vol={row['volume']} | confidence={prob.item():.2f}")
                elif pred_label == 0 and actual_label == 1:
                    print(f"MISSED FRAUD | {row['trader_id']} -> {row['counterparty_id']} | vol={row['volume']}")

            if trade_count % 200 == 0:
                print_summary()

except KeyboardInterrupt:
    print("\n\nStopped by user.")
    print_summary()
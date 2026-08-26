import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gnn_model import FraudGNN
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

df = pd.read_csv('trades_dataset.csv')
print(f"Loaded {len(df)} trades")

WINDOW_SIZE = 500
STEP = 20

def build_window_graph(window_df):
    all_traders = pd.concat([window_df['trader_id'], window_df['counterparty_id']]).unique()
    trader_to_idx = {t: i for i, t in enumerate(all_traders)}

    source_nodes = window_df['trader_id'].map(trader_to_idx).tolist()
    target_nodes = window_df['counterparty_id'].map(trader_to_idx).tolist()
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_features = torch.tensor(window_df[['price', 'volume']].values, dtype=torch.float)
    edge_labels = torch.tensor(window_df['is_fraud'].astype(int).values, dtype=torch.long)

    node_features_list = []
    for trader in all_traders:
        as_trader = window_df[window_df['trader_id'] == trader]
        as_counterparty = window_df[window_df['counterparty_id'] == trader]
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

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=edge_labels)

print(f"Building windowed graphs (window={WINDOW_SIZE}, step={STEP})...")
window_graphs = []
for start in range(0, len(df) - WINDOW_SIZE, STEP):
    window_df = df.iloc[start:start + WINDOW_SIZE]
    graph = build_window_graph(window_df)
    window_graphs.append(graph)

print(f"Built {len(window_graphs)} windowed graphs")

split_point = int(0.8 * len(window_graphs))
train_graphs = window_graphs[:split_point]
test_graphs = window_graphs[split_point:]
print(f"Train windows: {len(train_graphs)}, Test windows: {len(test_graphs)}")

train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)

model = FraudGNN(node_feature_dim=5, edge_feature_dim=2).to(device)

all_train_labels = torch.cat([g.y for g in train_graphs])
fraud_count = all_train_labels.sum().item()
normal_count = len(all_train_labels) - fraud_count
class_weights = torch.tensor([1.0, normal_count / fraud_count], dtype=torch.float).to(device)
print(f"Class weights: {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

EPOCHS = 15
print(f"\nTraining for {EPOCHS} epochs across {len(train_graphs)} windowed graphs...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    if (epoch + 1) % 3 == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                preds = out.argmax(dim=1)
                correct += (preds == batch.y).sum().item()
                total += len(batch.y)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | Train Accuracy: {correct/total:.4f}")

print("\nEvaluating on held-out (future) windows...")
model.eval()
tp = fp = tn = fn = 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        preds = out.argmax(dim=1)
        tp += ((preds == 1) & (batch.y == 1)).sum().item()
        fp += ((preds == 1) & (batch.y == 0)).sum().item()
        tn += ((preds == 0) & (batch.y == 0)).sum().item()
        fn += ((preds == 0) & (batch.y == 1)).sum().item()

accuracy = (tp + tn) / (tp + fp + tn + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n--- Held-out Windows Test Results ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"TP={tp} FP={fp} TN={tn} FN={fn}")

torch.save(model.state_dict(), 'fraud_gnn_windowed.pt')
print("\nSaved windowed model to fraud_gnn_windowed.pt")

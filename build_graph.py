import pandas as pd
import torch
from torch_geometric.data import Data

df = pd.read_csv('trades_dataset.csv')
print(f"Loaded {len(df)} trades")

all_traders = pd.concat([df['trader_id'], df['counterparty_id']]).unique()
trader_to_idx = {trader: idx for idx, trader in enumerate(all_traders)}
num_nodes = len(all_traders)
print(f"Found {num_nodes} unique traders")

source_nodes = df['trader_id'].map(trader_to_idx).tolist()
target_nodes = df['counterparty_id'].map(trader_to_idx).tolist()
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

edge_features = torch.tensor(df[['price', 'volume']].values, dtype=torch.float)
edge_labels = torch.tensor(df['is_fraud'].astype(int).values, dtype=torch.long)


node_features_list = []
for trader in all_traders:
    as_trader = df[df['trader_id'] == trader]
    as_counterparty = df[df['counterparty_id'] == trader]
    all_involved = pd.concat([as_trader, as_counterparty])

    total_trades = len(all_involved)
    avg_volume = all_involved['volume'].mean() if total_trades > 0 else 0
    avg_price = all_involved['price'].mean() if total_trades > 0 else 0
    counterparties = pd.concat([
        as_trader['counterparty_id'],
        as_counterparty['trader_id']
    ]).unique()
    unique_counterparties = len(counterparties)

 
    volume_std = all_involved['volume'].std() if total_trades > 1 else 0

    node_features_list.append([total_trades, avg_volume, avg_price, unique_counterparties, volume_std])

node_features = torch.tensor(node_features_list, dtype=torch.float)
node_features = torch.nan_to_num(node_features, nan=0.0)  

print(f"\nnode_features shape: {node_features.shape}")
print(f"Example - first trader's features: {node_features[0]}")
print(f"  (total_trades, avg_volume, avg_price, unique_counterparties, volume_std)")

graph_data = Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_features,
    y=edge_labels
)

print(f"\nFull graph object:\n{graph_data}")

torch.save(graph_data, 'trade_graph.pt')
print("\nSaved graph to trade_graph.pt")

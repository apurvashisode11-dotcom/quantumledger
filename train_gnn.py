import torch
import torch.nn as nn
from torch_geometric.data import Data
from gnn_model import FraudGNN
import random

# Load our graph
graph_data = torch.load('trade_graph.pt', weights_only=False)
print(f"Loaded graph: {graph_data}")

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
graph_data = graph_data.to(device)

# Step 1: Create train/test split on EDGES (trades)
num_edges = graph_data.edge_index.shape[1]
indices = list(range(num_edges))
random.seed(42)  # reproducible shuffling
random.shuffle(indices)

split_point = int(0.8 * num_edges)
train_idx = torch.tensor(indices[:split_point], dtype=torch.long).to(device)
test_idx = torch.tensor(indices[split_point:], dtype=torch.long).to(device)

print(f"Train edges: {len(train_idx)}, Test edges: {len(test_idx)}")

# Step 2: Initialize model
model = FraudGNN(
    node_feature_dim=graph_data.x.shape[1],
    edge_feature_dim=graph_data.edge_attr.shape[1]
).to(device)

# Step 3: Set up loss function with class weighting
fraud_count = graph_data.y.sum().item()
normal_count = len(graph_data.y) - fraud_count
class_weights = torch.tensor([1.0, normal_count / fraud_count], dtype=torch.float).to(device)
print(f"Class weights [not_fraud, fraud]: {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 4: Training loop
EPOCHS = 100
print(f"\nTraining for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    # Forward pass - predict on ALL edges (message passing needs full graph structure)
    out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

    # But only compute loss on TRAINING edges
    loss = criterion(out[train_idx], graph_data.y[train_idx])

    # Backward pass - this is where the model actually learns
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        # Quick check on training accuracy
        model.eval()
        with torch.no_grad():
            preds = out[train_idx].argmax(dim=1)
            train_acc = (preds == graph_data.y[train_idx]).float().mean().item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | Train Accuracy: {train_acc:.4f}")

print("\nTraining complete. Saving model...")
torch.save(model.state_dict(), 'fraud_gnn_trained.pt')
torch.save({'train_idx': train_idx, 'test_idx': test_idx}, 'train_test_split.pt')
print("Saved fraud_gnn_trained.pt and train_test_split.pt")

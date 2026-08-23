import torch
from gnn_model import FraudGNN

# Load graph and model
graph_data = torch.load('trade_graph.pt', weights_only=False)
split = torch.load('train_test_split.pt', weights_only=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph_data = graph_data.to(device)
train_idx = split['train_idx'].to(device)
test_idx = split['test_idx'].to(device)

model = FraudGNN(
    node_feature_dim=graph_data.x.shape[1],
    edge_feature_dim=graph_data.edge_attr.shape[1]
).to(device)
model.load_state_dict(torch.load('fraud_gnn_trained.pt', weights_only=True))
model.eval()

with torch.no_grad():
    out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    preds = out.argmax(dim=1)

def evaluate(idx, name):
    y_true = graph_data.y[idx]
    y_pred = preds[idx]

    true_positives = ((y_pred == 1) & (y_true == 1)).sum().item()
    false_positives = ((y_pred == 1) & (y_true == 0)).sum().item()
    true_negatives = ((y_pred == 0) & (y_true == 0)).sum().item()
    false_negatives = ((y_pred == 0) & (y_true == 1)).sum().item()

    accuracy = (y_pred == y_true).float().mean().item()
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n--- {name} Set ---")
    print(f"Total: {len(idx)} | Actual fraud: {y_true.sum().item()} | Predicted fraud: {y_pred.sum().item()}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (of predicted fraud, how much was real)")
    print(f"Recall:    {recall:.4f}  (of real fraud, how much we caught)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion: TP={true_positives} FP={false_positives} TN={true_negatives} FN={false_negatives}")

evaluate(train_idx, "Train")
evaluate(test_idx, "Test")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FraudGNN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=32):
        super().__init__()

        # Message passing layers - these refine node representations
        # by pulling in info from neighbors, layer by layer
        self.conv1 = SAGEConv(node_feature_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # Edge classifier - takes [source_node_embedding, target_node_embedding, edge_features]
        # and outputs a fraud/not-fraud score
        classifier_input_dim = (hidden_dim * 2) + edge_feature_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 classes: not fraud, fraud
        )

    def forward(self, x, edge_index, edge_attr):
        # Step 1: Message passing - refine node representations
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        # h now contains "context-aware" embeddings for every node

        # Step 2: For each edge, gather its source and target node embeddings
        source_idx, target_idx = edge_index
        source_embeddings = h[source_idx]
        target_embeddings = h[target_idx]

        # Step 3: Combine source + target + edge features, then classify
        edge_input = torch.cat([source_embeddings, target_embeddings, edge_attr], dim=1)
        out = self.edge_classifier(edge_input)

        return out

# Quick test - load our saved graph and run one forward pass (untrained, just checking shapes)
if __name__ == "__main__":
    graph_data = torch.load('trade_graph.pt', weights_only=False)

    model = FraudGNN(
        node_feature_dim=graph_data.x.shape[1],
        edge_feature_dim=graph_data.edge_attr.shape[1]
    )

    output = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    print(f"Model output shape: {output.shape}")
    print(f"Expected: [num_edges, 2] since we predict 2 classes per edge")
    print(f"\nFirst 3 predictions (raw scores, untrained so meaningless yet):")
    print(output[:3])

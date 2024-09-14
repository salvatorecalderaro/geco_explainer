import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from explainer import GECo

# Simple GNN Model: 2-layer GCN for demonstration purposes
class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_dummy_graph():
    num_nodes = 20
    
    # Create random edges for the graph
    edge_index = torch.randint(0, num_nodes, (2, 30), dtype=torch.long)  # 300 edges
    
    # Create random node features (3 features per node)
    x = torch.rand((num_nodes, 3), dtype=torch.float)  # 100 nodes with 3 features each
    
    # Create the graph
    graph = Data(x=x, edge_index=edge_index)
    return graph

# Instantiate GNN and GECo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = SimpleGNN()
geco = GECo(device, gnn)

# Create dummy graph
dummy_graph = create_dummy_graph()

# Predict and explain for the dummy graph
pred, _ = geco.predict(dummy_graph)  # Get a prediction
pred=0
explanation = geco.explain(dummy_graph, pred, visualize=True)  # Explain the prediction
print(f"Predicted class: {pred}")
print(f"Explanation (important nodes): {explanation}")
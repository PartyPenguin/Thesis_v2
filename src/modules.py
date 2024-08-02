import torch as th
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv, GATConv, RGATConv, SAGEConv, GATv2Conv
from torch_geometric.nn import Linear
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import MeanAggregation
import yaml


device = "cuda" if th.cuda.is_available() else "cpu"

# Load config from params.yaml
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)


class GCNPolicy(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super().__init__()

        # Define the GCN layers
        self.gcn_conv1 = GCNConv(obs_dims, 128)
        self.gcn_conv2 = GCNConv(128, 128)
        self.gcn_conv3 = GCNConv(128, 128)

        # Define the linear layer
        self.lin = Linear(128, act_dims)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply the GCN layers
        x = self.gcn_conv1(x, edge_index).relu()
        x = self.gcn_conv2(x, edge_index).relu()
        # x = self.gcn_conv3(x, edge_index).relu()

        # Apply the linear layer
        x = self.lin(x)

        # Apply the tanh activation function because the actions are in the range [-1, 1]
        x = th.tanh(x)

        # Apply global mean pooling
        x = global_mean_pool(x, batch)

        return x


class GATPolicy(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super().__init__()
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        num_heads = config["train"]["model_params"]["num_heads"]
        hidden_dim = config["train"]["model_params"]["hidden_dim"]
        dropout = config["train"]["model_params"]["dropout"]
        # Define the GAT layers
        self.gat_conv1 = GATConv(
            obs_dims, hidden_dim, edge_dim=1, heads=num_heads, dropout=dropout
        )
        self.gat_conv2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            edge_dim=1,
            heads=num_heads,
            dropout=dropout,
        )
        self.gat_conv3 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            edge_dim=1,
            heads=num_heads,
            dropout=dropout,
        )

        # Define the linear layer
        self.lin = Linear(hidden_dim * num_heads, act_dims)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Apply the GAT layers
        x = self.gat_conv1(x, edge_index, edge_attr).relu()
        x = self.gat_conv2(x, edge_index, edge_attr).relu()
        x = self.gat_conv3(x, edge_index, edge_attr).relu()

        # Apply global mean pooling
        x = global_mean_pool(x, batch)

        # Apply the linear layer
        x = self.lin(x)
        # Apply the tanh activation function because the actions are in the range [-1, 1]
        x = th.tanh(x)

        return x


class GATv2Policy(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super().__init__()
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        num_heads = config["train"]["model_params"]["num_heads"]
        hidden_dim = config["train"]["model_params"]["hidden_dim"]
        dropout = config["train"]["model_params"]["dropout"]
        # Define the GAT layers
        self.gat_conv1 = GATv2Conv(
            obs_dims, hidden_dim, edge_dim=1, heads=num_heads, dropout=dropout
        )
        self.gat_conv2 = GATv2Conv(
            hidden_dim * num_heads,
            hidden_dim,
            edge_dim=1,
            heads=num_heads,
            dropout=dropout,
        )
        self.gat_conv3 = GATv2Conv(
            hidden_dim * num_heads,
            hidden_dim,
            edge_dim=1,
            heads=num_heads,
            dropout=dropout,
        )
        self.gat_conv4 = GATv2Conv(
            hidden_dim * num_heads,
            hidden_dim,
            edge_dim=1,
            heads=num_heads,
            dropout=dropout,
        )

        # Define the linear layer
        self.lin = Linear(hidden_dim * num_heads, act_dims)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Apply the GAT layers
        x = self.gat_conv1(x, edge_index, edge_attr).relu()
        x = self.gat_conv2(x, edge_index, edge_attr).relu()
        x = self.gat_conv3(x, edge_index, edge_attr).relu()

        # Apply global mean pooling
        x = global_mean_pool(x, batch)

        # Apply the linear layer
        x = self.lin(x)
        # Apply the tanh activation function because the actions are in the range [-1, 1]
        x = th.tanh(x)

        return x


class GripperClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_layer = nn.Sequential(
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classification_layer(x)


class GRURegression(nn.Module):
    def __init__(self, obs_dims, output_dim):
        super().__init__()
        self.obs_dims = obs_dims
        self.act_dims = output_dim

        self.hidden_dim = config["train"]["model_params"]["hidden_dim"]
        self.dropout = config["train"]["model_params"]["dropout"]
        self.num_layers = config["train"]["model_params"]["num_layers"]

        # Define the linear layers
        self.gru1 = nn.GRU(
            obs_dims,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.lin = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        h0 = th.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = th.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # Forward pass through the LSTM
        x, hn = self.gru1(x, h0)
        x = x[:, -1, :]
        x = self.lin(x)
        x = th.tanh(x)
        return x


class BaselineModel(nn.Module):
    # Combines the GripperClassifier and GRURegression models
    def __init__(self, obs_dims, output_dim):
        super().__init__()
        self.obs_dims = obs_dims
        self.act_dims = output_dim
        self.gripper_classifier = GripperClassifier()
        self.gru_regression = GRURegression(obs_dims, output_dim)

    def forward(self, x):
        regression_output = self.gru_regression(x)
        gripper_output = self.gripper_classifier(x[:, -1, :])
        return regression_output, gripper_output

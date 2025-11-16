from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn.modules.container import ModuleList
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        local_layers: int = 3,
        dropout: float = 0.5,
        norm: Literal["batch", "layer"] | None = None
    ):
        super(GCN, self).__init__()
        self.dropout: float = dropout

        convs = [GCNConv(num_node_features, hidden_channels)]
        convs += [GCNConv(hidden_channels, hidden_channels) for _ in range(local_layers - 1)]
        self.local_convs: ModuleList = torch.nn.ModuleList(convs)

        def _make_norm_layer(norm_type: str | None, channels: int) -> torch.nn.Module | None:
            if norm_type == "batch":
                return BatchNorm1d(channels)
            if norm_type == "layer":
                return torch.nn.LayerNorm(channels)
            return None

        self.norm: str | None = norm
        # Always create a ModuleList for norm_layers so indexing is safe in forward
        norm_layers = [_make_norm_layer(norm, hidden_channels) for _ in range(local_layers)]
        self.norm_layers: ModuleList = torch.nn.ModuleList(
            [nl if nl is not None else torch.nn.Identity() for nl in norm_layers]
        )

        self.linear = torch.nn.Linear(hidden_channels, 1)
    
    def forward(self, data: Any) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        for i, local_conv in enumerate(self.local_convs):
            x = local_conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.norm_layers[i](x)

            if i < len(self.local_convs) - 1:
                x = F.relu(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int = 64, local_layers: int = 2, dropout: float = 0.5, norm: Literal["batch", "layer"] | None = None):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout

        convs = [SAGEConv(num_node_features, hidden_channels)]
        convs += [SAGEConv(hidden_channels, hidden_channels) for _ in range(local_layers - 1)]
        self.local_convs: ModuleList = torch.nn.ModuleList(convs)

        def _make_norm_layer(norm_type: str | None, channels: int) -> torch.nn.Module | None:
            if norm_type == "batch":
                return BatchNorm1d(channels)
            if norm_type == "layer":
                return torch.nn.LayerNorm(channels)
            return None

        norm_layers = [_make_norm_layer(norm, hidden_channels) for _ in range(local_layers)]
        self.norm_layers: ModuleList = torch.nn.ModuleList([nl if nl is not None else torch.nn.Identity() for nl in norm_layers])

        self.linear: Linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.local_convs):
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.norm_layers[i](x)
            if i < len(self.local_convs) - 1:
                x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x


class GATv2(torch.nn.Module):
    def __init__(
        self, num_node_features: int, hidden_channels: int = 64, heads: int = 4, local_layers: int = 2, dropout: float = 0.5, norm: Literal["batch", "layer"] | None = None
    ):
        super(GATv2, self).__init__()

        # First layer uses heads; subsequent layers use heads=1 and adjust channels accordingly
        convs = [GATv2Conv(num_node_features, hidden_channels, heads=heads)]
        for _ in range(local_layers - 1):
            convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=1))
        self.local_convs: ModuleList = torch.nn.ModuleList(convs)

        self.dropout = dropout

        def _make_norm_layer(norm_type: str | None, channels: int) -> torch.nn.Module | None:
            if norm_type == "batch":
                return BatchNorm1d(channels)
            if norm_type == "layer":
                return torch.nn.LayerNorm(channels)
            return None

        norm_layers = [_make_norm_layer(norm, hidden_channels) for _ in range(local_layers)]
        self.norm_layers: ModuleList = torch.nn.ModuleList([nl if nl is not None else torch.nn.Identity() for nl in norm_layers])

        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.local_convs):
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.norm_layers[i](x)
            if i < len(self.local_convs) - 1:
                x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

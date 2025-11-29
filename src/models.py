from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch.nn.modules.linear import Linear
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GCNConv,
    GINConv,
    GraphNorm,
    JumpingKnowledge,
    LayerNorm,
    SAGEConv,
    global_mean_pool,
)


def _make_norm_layer(norm_type: str | None, channels: int) -> torch.nn.Module | None:
    if norm_type == "batch":
        return BatchNorm(channels)
    if norm_type == "layer":
        return LayerNorm(channels)
    if norm_type == 'graph':
        return GraphNorm(channels)
    return None


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        local_layers: int = 3,
        dropout: float = 0.5,
        norm: Literal["batch", "layer", "graph"] | None = None,
        jk: Literal["max", "cat", "lstm"] | None = None,
        res: bool = False,
    ):
        super(GCN, self).__init__()
        jk = None if jk == 'None' else jk
        self.dropout: float = dropout

        convs = [GCNConv(num_node_features, hidden_channels)]
        convs += [
            GCNConv(hidden_channels, hidden_channels) for _ in range(local_layers - 1)
        ]
        self.local_convs: ModuleList = torch.nn.ModuleList(convs)

        # Residual linear projections (map input dim to conv output dim)
        self.res = res
        res_linears = [torch.nn.Linear(num_node_features, hidden_channels)]
        res_linears += [
            torch.nn.Linear(hidden_channels, hidden_channels)
            for _ in range(local_layers - 1)
        ]
        self.res_linears: ModuleList = torch.nn.ModuleList(res_linears)

        self.norm: str | None = norm
        # Always create a ModuleList for norm_layers so indexing is safe in forward
        norm_layers = [
            _make_norm_layer(norm, hidden_channels) for _ in range(local_layers)
        ]
        self.norm_layers: ModuleList = torch.nn.ModuleList(
            [nl if nl is not None else torch.nn.Identity() for nl in norm_layers]
        )

        self.jk_mode = jk
        if jk is not None:
            self.jk = JumpingKnowledge(jk, hidden_channels, local_layers)
            jk_channels = (
                hidden_channels * local_layers if jk == "cat" else hidden_channels
            )
        else:
            self.jk = None
            jk_channels = hidden_channels

        self.linear = torch.nn.Linear(jk_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        jk_inputs: list[torch.Tensor] = []
        for i, conv in enumerate(self.local_convs):
            if self.res:
                x = conv(x, edge_index) + self.res_linears[i](x)
            else:
                x = conv(x, edge_index)

            if i < len(self.local_convs) - 1 or self.jk is not None:
                x = self.norm_layers[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.jk is not None:
                    jk_inputs.append(x)

        if self.jk is not None:
            x = self.jk(jk_inputs)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        local_layers: int = 2,
        dropout: float = 0.5,
        norm: Literal["batch", "layer", "graph"] | None = None,
        jk: Literal["max", "cat", "lstm"] | None = None,
        res: bool = False,
    ):
        super(GraphSAGE, self).__init__()
        jk = None if jk == 'None' else jk
        self.dropout = dropout

        convs = [SAGEConv(num_node_features, hidden_channels)]
        convs += [
            SAGEConv(hidden_channels, hidden_channels) for _ in range(local_layers - 1)
        ]
        self.local_convs: ModuleList = torch.nn.ModuleList(convs)

        # Residual linears for SAGE
        self.res = res
        res_linears = [torch.nn.Linear(num_node_features, hidden_channels)]
        res_linears += [
            torch.nn.Linear(hidden_channels, hidden_channels)
            for _ in range(local_layers - 1)
        ]
        self.res_linears: ModuleList = torch.nn.ModuleList(res_linears)

        norm_layers = [
            _make_norm_layer(norm, hidden_channels) for _ in range(local_layers)
        ]
        self.norm_layers: ModuleList = torch.nn.ModuleList(
            [nl if nl is not None else torch.nn.Identity() for nl in norm_layers]
        )

        self.jk_mode = jk
        if jk is not None:
            self.jk = JumpingKnowledge(jk, hidden_channels, local_layers)
            jk_channels = (
                hidden_channels * local_layers if jk == "cat" else hidden_channels
            )
        else:
            self.jk = None
            jk_channels = hidden_channels

        self.linear: Linear = torch.nn.Linear(jk_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        jk_inputs: list[torch.Tensor] = []
        for i, conv in enumerate(self.local_convs):
            if self.res:
                x = conv(x, edge_index) + self.res_linears[i](x)
            else:
                x = conv(x, edge_index)

            if i < len(self.local_convs) - 1 or self.jk is not None:
                x = self.norm_layers[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.jk is not None:
                    jk_inputs.append(x)

        if self.jk is not None:
            x = self.jk(jk_inputs)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x


class GATv2(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        heads: int = 1,
        local_layers: int = 2,
        dropout: float = 0.5,
        norm: Literal["batch", "layer", "graph"] | None = None,
        jk: Literal["max", "cat", "lstm"] | None = None,
        res: bool = False,
    ):
        super(GATv2, self).__init__()
        jk = None if jk == 'None' else jk

        if hidden_channels % heads != 0:
            raise ValueError(
                f"Ensure that the number of output channels of "
                + f"'GATConv' (got '{hidden_channels}') is divisible "
                + f"by the number of heads (got '{heads}')"
            )

        # First layer uses heads; subsequent layers use heads=1 and adjust channels ac“cordingly
        convs = [
            GATv2Conv(
                num_node_features, hidden_channels // heads, heads=heads, residual=res
            )
        ]
        for _ in range(local_layers - 1):
            convs.append(
                GATv2Conv(
                    hidden_channels, hidden_channels // heads, heads=heads, residual=res
                )
            )
        self.local_convs: ModuleList = torch.nn.ModuleList(convs)

        self.dropout: float = dropout

        norm_layers = [
            _make_norm_layer(norm, hidden_channels) for _ in range(local_layers)
        ]
        self.norm_layers: ModuleList = torch.nn.ModuleList(
            [nl if nl is not None else torch.nn.Identity() for nl in norm_layers]
        )

        base_channels = hidden_channels
        self.jk_mode: Literal["max", "cat", "lstm"] | None = jk
        if jk is not None:
            self.jk = JumpingKnowledge(jk, base_channels, local_layers)
            jk_channels = base_channels * local_layers if jk == "cat" else base_channels
        else:
            self.jk = None
            jk_channels = base_channels

        self.linear: Linear = torch.nn.Linear(jk_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        jk_inputs: list[torch.Tensor] = []
        for i, conv in enumerate(self.local_convs):
            # Residual passed as arg to GatConv layer
            x = conv(x, edge_index)

            if i < len(self.local_convs) - 1 or self.jk is not None:
                x = self.norm_layers[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.jk is not None:
                    jk_inputs.append(x)

        if self.jk is not None:
            x = self.jk(jk_inputs)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x


class GIN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        local_layers: int = 3,
        dropout: float = 0.5,
        norm: Literal["batch", "layer", "graph"] | None = None,
        jk: Literal["max", "cat", "lstm"] | None = None,
        res: bool = False,
    ):
        super(GIN, self).__init__()
        jk = None if jk == 'None' else jk
        self.dropout = dropout

        def make_mlp(in_dim: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )

        convs = [GINConv(make_mlp(num_node_features))]
        convs += [GINConv(make_mlp(hidden_channels)) for _ in range(local_layers - 1)]
        self.local_convs: ModuleList = torch.nn.ModuleList(convs)

        self.res = res
        res_linears = [torch.nn.Linear(num_node_features, hidden_channels)]
        res_linears += [
            torch.nn.Linear(hidden_channels, hidden_channels)
            for _ in range(local_layers - 1)
        ]
        self.res_linears: ModuleList = torch.nn.ModuleList(res_linears)

        norm_layers = [
            _make_norm_layer(norm, hidden_channels) for _ in range(local_layers)
        ]
        self.norm_layers: ModuleList = torch.nn.ModuleList(
            [nl if nl is not None else torch.nn.Identity() for nl in norm_layers]
        )

        self.jk_mode = jk
        if jk is not None:
            self.jk = JumpingKnowledge(jk, hidden_channels, local_layers)
            jk_channels = (
                hidden_channels * local_layers if jk == "cat" else hidden_channels
            )
        else:
            self.jk = None
            jk_channels = hidden_channels

        self.linear = torch.nn.Linear(jk_channels, 1)

    def forward(self, data, mixup: bool = False, target: torch.Tensor | None = None, alpha: float = 1.0):
        """
        Public forward method that dispatches to Normal GNN or FCN Mixup logic.
        """
        if mixup:
            # We strictly need targets for Manifold Mixup
            assert target is not None, "Target must be provided for mixup training"
            return self._forward_fcn(data, target, alpha)
        else:
            return self._forward_normal(data)

    def _forward_normal(self, data) -> torch.Tensor:
        """
        Standard GIN Forward Pass (Uses Graph Structure)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        jk_inputs: list[torch.Tensor] = []
        for i, conv in enumerate(self.local_convs):
            if self.res:
                x = conv(x, edge_index) + self.res_linears[i](x)
            else:
                x = conv(x, edge_index)

            if i < len(self.local_convs) - 1 or self.jk is not None:
                x = self.norm_layers[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.jk is not None:
                    jk_inputs.append(x)

        if self.jk is not None:
            x = self.jk(jk_inputs)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

    def _forward_fcn(self, data, target: torch.Tensor, alpha: float):
        """
        FCN Forward Pass with Manifold Mixup (Ignores Graph Structure)
        """
        x, batch = data.x, data.batch
        
        # 1. Create Empty Edges (FCN View)
        # By passing empty edges, GINConv acts like a standard MLP on node features
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

        jk_inputs: list[torch.Tensor] = []
        for i, conv in enumerate(self.local_convs):
            if self.res:
                x = conv(x, empty_edge_index) + self.res_linears[i](x)
            else:
                x = conv(x, empty_edge_index)

            if i < len(self.local_convs) - 1 or self.jk is not None:
                x = self.norm_layers[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.jk is not None:
                    jk_inputs.append(x)

        if self.jk is not None:
            x = self.jk(jk_inputs)

        # 2. Pooling
        # We pool the "node-level FCN" features into a graph embedding
        x = global_mean_pool(x, batch)

        # 3. Manifold Mixup (Applied at the Embedding Layer)
        # We mix the pooled graph embeddings and the graph-level targets
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        # Permute the batch (graphs), not the nodes
        batch_size = x.size(0)
        perm = torch.randperm(batch_size, device=x.device)

        # Mix Hidden States (Embeddings)
        x = lam * x + (1 - lam) * x[perm]
        
        # Prepare Mixed Targets
        target_a = target
        target_b = target[perm]

        # 4. Final Projection
        x = self.linear(x)

        return x, target_a, target_b, lam

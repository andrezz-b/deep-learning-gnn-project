from copy import deepcopy
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
        norm: Literal["batch", "layer"] | None = None,
        res: bool = False,
    ):
        super(GCN, self).__init__()
        self.dropout: float = dropout

        convs = [GCNConv(num_node_features, hidden_channels)]
        convs += [GCNConv(hidden_channels, hidden_channels) for _ in range(local_layers - 1)]
        self.local_convs: ModuleList = torch.nn.ModuleList(convs)

        # Residual linear projections (map input dim to conv output dim)
        self.res = res
        res_linears = [torch.nn.Linear(num_node_features, hidden_channels)]
        res_linears += [torch.nn.Linear(hidden_channels, hidden_channels) for _ in range(local_layers - 1)]
        self.res_linears: ModuleList = torch.nn.ModuleList(res_linears)

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
            if self.res:
                x = local_conv(x, edge_index) + self.res_linears[i](x)
            else:
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
    def __init__(self, num_node_features: int, hidden_channels: int = 64, local_layers: int = 2, dropout: float = 0.5, norm: Literal["batch", "layer"] | None = None, res: bool = False):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout

        convs = [SAGEConv(num_node_features, hidden_channels)]
        convs += [SAGEConv(hidden_channels, hidden_channels) for _ in range(local_layers - 1)]
        self.local_convs: ModuleList = torch.nn.ModuleList(convs)

        # Residual linears for SAGE
        self.res = res
        res_linears = [torch.nn.Linear(num_node_features, hidden_channels)]
        res_linears += [torch.nn.Linear(hidden_channels, hidden_channels) for _ in range(local_layers - 1)]
        self.res_linears: ModuleList = torch.nn.ModuleList(res_linears)

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
            if self.res:
                x = conv(x, edge_index) + self.res_linears[i](x)
            else:
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
        self, num_node_features: int, hidden_channels: int = 64, heads: int = 4, local_layers: int = 2, dropout: float = 0.5, norm: Literal["batch", "layer"] | None = None, res: bool = False
    ):
        super(GATv2, self).__init__()

        # First layer uses heads; subsequent layers use heads=1 and adjust channels accordingly
        convs = [GATv2Conv(num_node_features, hidden_channels, heads=heads)]
        for _ in range(local_layers - 1):
            convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=1))
        self.local_convs: ModuleList = torch.nn.ModuleList(convs)

        self.dropout = dropout

        # Residual linears for GATv2: sizes depend on heads and layer position
        self.res = res
        res_linears = []
        prev_dim = num_node_features
        for i in range(local_layers):
            if i == 0:
                out_dim = hidden_channels * heads
            else:
                out_dim = hidden_channels
            res_linears.append(torch.nn.Linear(prev_dim, out_dim))
            prev_dim = out_dim
        self.res_linears: ModuleList = torch.nn.ModuleList(res_linears)

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
            if self.res:
                x = conv(x, edge_index) + self.res_linears[i](x)
            else:
                x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.norm_layers[i](x)
            if i < len(self.local_convs) - 1:
                x = F.relu(x)

        x = global_mean_pool(x, batch) 
        x = self.linear(x)
        return x


class MeanTeacher(torch.nn.Module):
    """
    forward(..., use_teacher=False) for training if we want to use the student model as usual,
    and use_teacher=True if we want the teacher's predictions
    update_teacher() 
    """

    def __init__(self, student_model: torch.nn.Module, ema_decay: float = 0.999, update_every: int = 1):
        super().__init__()
        self.student = student_model
        self.teacher = deepcopy(student_model) # cloning the student model to teacher model 
        self.teacher.requires_grad_(False) # goes through all params and sets the grad flag to False
        # we dont need to calculate and store gradients in teacher model, so autograd won't track operations on it

        self.ema_decay = ema_decay #exponential moving average
        '''
        θ_teacher = ema_decay * θ_teacher + (1 - ema_decay) * θ_student,
        so this param closer to 1.0 make the teacher change slowly,
        while smaller values make it follow the student more
        '''

        self.update_every = update_every #skip teacher updates to save time, check out update_teacher

        # here we attach a tensor to the module that isn’t a trainable parameter
        self.register_buffer("_update_counter", torch.tensor(0, dtype=torch.long), persistent=False)

    def forward(self, data: Any, use_teacher: bool = False) -> torch.Tensor:
        '''
        Decides which model to run (student/teacher) and in what autograd mode.

        - we run student forward to get predictions that are part of the loss; gradients flow through this path.
        - then the teacher forward (use_teacher=True) to produce targets (a vector with probabilities for each class).
          (this forward is inside torch.no_grad(), so it stays out of autograd, but the resulting predictions
          still enter the consistency loss computation)
        - combining losses, doing backpropogation, and updating the student weights
        - after we call model.update_teacher() so the teacher weights move toward the updated student weights
        '''

        if use_teacher:
            self.teacher.eval() #sets model to evaluation mode, so regularization behaves accordingly
            with torch.no_grad():
                return self.teacher(data) # we get the EMA output (for consistency loss)
        return self.student(data) # if use_teacher=False (default), we move to self.student(data), so training behaves like a normal model

    @torch.no_grad() #tells autograd not to track any operations inside this function while it runs, just in case
    def update_teacher(self) -> None:
        self._update_counter += 1
        if self._update_counter % self.update_every != 0:
            return # when the check succeeds (!= 0) the return exits the function and the loop is skipped entirely
        
        # if the counter is divisible (% = 0):
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()): #teacher/student parameter pairs (same ordering thanks to deepcopy)
            teacher_param.data.mul_(self.ema_decay)
            teacher_param.data.add_((1.0 - self.ema_decay) * student_param.data)
            # we have updated the teacher weights

    @torch.no_grad()
    def reset_teacher(self) -> None:
        """Hard reset so the teacher and student share weights again."""
        self.teacher.load_state_dict(self.student.state_dict())


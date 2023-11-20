import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import mean_nodes
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, SAGEConv, GATConv, GATv2Conv


EMBEDDINGS = {
    "l2n": (300, 200),
    "n2l": (200, 300),
}


class GraphConvModel(nn.Module):
    def __init__(
        self,
        num_conv_layers=3,
        num_fc_layers=1,
        conv_dim=128,
        fc_dim=128,
        conv_norm="both",
        conv_activation=F.elu,
        fc_activation=F.relu,
        norm=nn.LayerNorm,
    ):
        super(GraphConvModel, self).__init__()

        self.num_fc_layers = num_fc_layers
        self.conv_activation = conv_activation
        self.fc_activation = fc_activation

        self.norms = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            HeteroGraphConv(
                {
                    k: GraphConv(v[0], conv_dim, norm=conv_norm)
                    for k, v in EMBEDDINGS.items()
                },
                aggregate="mean",
            )
        )
        self.norms.append(nn.ModuleDict({k: norm(conv_dim) for k in ("n", "l")}))

        for i in range(1, num_conv_layers):
            self.conv_layers.append(
                HeteroGraphConv(
                    {
                        k: GraphConv(conv_dim, conv_dim, norm=conv_norm)
                        for k, v in EMBEDDINGS.items()
                    },
                    aggregate="mean",
                )
            )
            self.norms.append(nn.ModuleDict({k: norm(conv_dim) for k in ("n", "l")}))
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(conv_dim, fc_dim))
        for l in range(1, num_fc_layers):
            self.fc_layers.append(nn.Linear(fc_dim, fc_dim))
        self.out = nn.Linear(fc_dim, 1)

    def forward(self, g):
        with g.local_scope():
            h = g.ndata["feat"]

            for i in range(len(self.conv_layers)):
                h = self.conv_layers[i](g, (h, h))
                h = {k: self.norms[i][k](v) for k, v in h.items()}
                h = {k: self.conv_activation(v) for k, v in h.items()}
            g.ndata["h"] = h
            hg = 0

            for ntype in g.ntypes:
                hg = hg + mean_nodes(g, "h", ntype=ntype)
            for k in range(self.num_fc_layers):
                hg = self.fc_layers[k](hg)
                hg = self.fc_activation(hg)
            return self.out(hg)


class SAGEConvModel(nn.Module):
    def __init__(
        self,
        num_conv_layers=3,
        num_fc_layers=1,
        conv_dim=128,
        fc_dim=128,
        agg_type="pool",
        feat_drop=0.0,
        conv_activation=F.elu,
        fc_activation=F.relu,
        norm=nn.LayerNorm,
    ):
        super(SAGEConvModel, self).__init__()

        self.num_fc_layers = num_fc_layers
        self.conv_activation = conv_activation
        self.fc_activation = fc_activation

        self.norms = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            HeteroGraphConv(
                {
                    k: SAGEConv(
                        v, conv_dim, aggregator_type=agg_type, feat_drop=feat_drop
                    )
                    for k, v in EMBEDDINGS.items()
                },
                aggregate="mean",
            )
        )
        self.norms.append(nn.ModuleDict({k: norm(conv_dim) for k in ("n", "l")}))

        for _ in range(1, num_conv_layers):
            self.conv_layers.append(
                HeteroGraphConv(
                    {
                        k: SAGEConv(
                            conv_dim,
                            conv_dim,
                            aggregator_type=agg_type,
                            feat_drop=feat_drop,
                        )
                        for k, v in EMBEDDINGS.items()
                    },
                    aggregate="mean",
                )
            )
            self.norms.append(nn.ModuleDict({k: norm(conv_dim) for k in ("n", "l")}))
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(conv_dim, fc_dim))
        for _ in range(1, num_fc_layers):
            self.fc_layers.append(nn.Linear(fc_dim, fc_dim))
        self.out = nn.Linear(fc_dim, 1)

    def forward(self, g):
        with g.local_scope():
            h = g.ndata["feat"]

            for i in range(len(self.conv_layers)):
                h = self.conv_layers[i](g, (h, h))
                h = {k: self.norms[i][k](v) for k, v in h.items()}
                h = {k: self.conv_activation(v) for k, v in h.items()}
            g.ndata["h"] = h
            hg = 0

            for ntype in g.ntypes:
                hg = hg + mean_nodes(g, "h", ntype=ntype)
            for k in range(self.num_fc_layers):
                hg = self.fc_layers[k](hg)
                hg = self.fc_activation(hg)
            return self.out(hg)


class GATConvModel(nn.Module):
    def __init__(
        self,
        num_conv_layers=3,
        num_fc_layers=1,
        conv_dim=128,
        fc_dim=128,
        num_heads=8,
        lrelu_slope=0.2,
        conv_activation=F.elu,
        fc_activation=F.relu,
        norm=nn.LayerNorm,
    ):
        super(GATConvModel, self).__init__()

        self.num_fc_layers = num_fc_layers
        self.conv_activation = conv_activation
        self.fc_activation = fc_activation

        self.norms = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            HeteroGraphConv(
                {
                    k: GATConv(
                        v,
                        conv_dim,
                        num_heads,
                        negative_slope=lrelu_slope,
                        residual=True,
                    )
                    for k, v in EMBEDDINGS.items()
                },
                aggregate="mean",
            )
        )
        self.norms.append(
            nn.ModuleDict({k: norm(conv_dim * num_heads) for k in ("n", "l")})
        )

        for i in range(1, num_conv_layers):
            self.conv_layers.append(
                HeteroGraphConv(
                    {
                        k: GATConv(
                            conv_dim * num_heads,
                            conv_dim,
                            num_heads,
                            negative_slope=lrelu_slope,
                            residual=True,
                        )
                        for k, v in EMBEDDINGS.items()
                    },
                    aggregate="mean",
                )
            )
            if i == num_conv_layers - 1:
                self.norms.append(
                    nn.ModuleDict({k: norm(conv_dim) for k in ("n", "l")})
                )
            else:
                self.norms.append(
                    nn.ModuleDict({k: norm(conv_dim * num_heads) for k in ("n", "l")})
                )
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(conv_dim, fc_dim))
        for _ in range(1, num_fc_layers):
            self.fc_layers.append(nn.Linear(fc_dim, fc_dim))
        self.out = nn.Linear(fc_dim, 1)

    def forward(self, g):
        with g.local_scope():
            h = g.ndata["feat"]

            for i in range(len(self.conv_layers) - 1):
                h = self.conv_layers[i](g, (h, h))
                h = {k: v.flatten(1) for k, v in h.items()}
                h = {k: self.norms[i][k](v) for k, v in h.items()}
                h = {k: self.conv_activation(v) for k, v in h.items()}
            h = self.conv_layers[-1](g, (h, h))
            h = {k: v.mean(1) for k, v in h.items()}
            h = {k: self.norms[-1][k](v) for k, v in h.items()}
            h = {k: self.conv_activation(v) for k, v in h.items()}

            g.ndata["h"] = h
            hg = 0

            for ntype in g.ntypes:
                hg = hg + mean_nodes(g, "h", ntype=ntype)
            for k in range(self.num_fc_layers):
                hg = self.fc_layers[k](hg)
                hg = self.fc_activation(hg)
            return self.out(hg)


class GATv2ConvModel(nn.Module):
    def __init__(
        self,
        num_conv_layers=3,
        num_fc_layers=1,
        conv_dim=128,
        fc_dim=128,
        num_heads=8,
        lrelu_slope=0.2,
        conv_activation=F.elu,
        fc_activation=F.relu,
        norm=nn.LayerNorm,
    ):
        super(GATv2ConvModel, self).__init__()

        self.num_fc_layers = num_fc_layers
        self.conv_activation = conv_activation
        self.fc_activation = fc_activation

        self.norms = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            HeteroGraphConv(
                {
                    k: GATv2Conv(
                        v,
                        conv_dim,
                        num_heads,
                        negative_slope=lrelu_slope,
                        residual=True,
                    )
                    for k, v in EMBEDDINGS.items()
                },
                aggregate="mean",
            )
        )
        self.norms.append(
            nn.ModuleDict({k: norm(conv_dim * num_heads) for k in ("n", "l")})
        )

        for i in range(1, num_conv_layers):
            self.conv_layers.append(
                HeteroGraphConv(
                    {
                        k: GATv2Conv(
                            conv_dim * num_heads,
                            conv_dim,
                            num_heads,
                            negative_slope=lrelu_slope,
                            residual=True,
                        )
                        for k, v in EMBEDDINGS.items()
                    },
                    aggregate="mean",
                )
            )
            if i == num_conv_layers - 1:
                self.norms.append(
                    nn.ModuleDict({k: norm(conv_dim) for k in ("n", "l")})
                )
            else:
                self.norms.append(
                    nn.ModuleDict({k: norm(conv_dim * num_heads) for k in ("n", "l")})
                )
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(conv_dim, fc_dim))
        for _ in range(1, num_fc_layers):
            self.fc_layers.append(nn.Linear(fc_dim, fc_dim))
        self.out = nn.Linear(fc_dim, 1)

    def forward(self, g):
        with g.local_scope():
            h = g.ndata["feat"]

            for i in range(len(self.conv_layers) - 1):
                h = self.conv_layers[i](g, (h, h))
                h = {k: v.flatten(1) for k, v in h.items()}
                h = {k: self.norms[i][k](v) for k, v in h.items()}
                h = {k: self.conv_activation(v) for k, v in h.items()}
            h = self.conv_layers[-1](g, (h, h))
            h = {k: v.mean(1) for k, v in h.items()}
            h = {k: self.norms[-1][k](v) for k, v in h.items()}
            h = {k: self.conv_activation(v) for k, v in h.items()}

            g.ndata["h"] = h
            hg = 0

            for ntype in g.ntypes:
                hg = hg + mean_nodes(g, "h", ntype=ntype)
            for k in range(self.num_fc_layers):
                hg = self.fc_layers[k](hg)
                hg = self.fc_activation(hg)
            return self.out(hg)


def train_step(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * X.batch_size
            print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss / num_batches


def valid_step(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
    total_loss /= num_batches
    print(f"valid loss: {total_loss:>8f} \n")

    return total_loss


def predict_dataloader(model, dataloader, device):
    model.eval()
    y_true = torch.tensor([], dtype=torch.float32, device=device)
    y_pred = torch.tensor([], dtype=torch.float32, device=device)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_true = torch.cat((y_true, y), 0)
            y_pred = torch.cat((y_pred, model(X)), 0)
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    return y_true, y_pred

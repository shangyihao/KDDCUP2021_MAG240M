import os
import time
import glob
import argparse
import math
import os.path as osp
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from typing import Optional, List, NamedTuple, Union, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv, MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.data import NeighborSampler
import torch.multiprocessing as mp

from ogb.lsc import MAG240MEvaluator
from dataset import MAG240MDataset
ROOT = '/data/shangyihao/'


class Temporal_emd_5(MessagePassing):
    def __init__(self, in_channels):
        super(Temporal_emd_5, self).__init__()
        self.in_channels = in_channels
        self.att1 = torch.nn.Parameter(torch.Tensor(1, self.in_channels))
        self.att2 = torch.nn.Parameter(torch.Tensor(1, self.in_channels))
        self.lin1 = Linear(self.in_channels, self.in_channels)
        self.lin2 = Linear(self.in_channels, self.in_channels)
        self.negative_slope = 0.2
        self.gap = None
        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        glorot(self.att1)
        glorot(self.att2)

    def forward(self, years, edge_index, x):
        year1 = years[0].float()
        year2 = years[1].float()
        self.gap = torch.abs(year1 - year2)

        x1, x2 = x[0], x[1]
        x1 = self.lin1(x1).view(-1, self.in_channels)
        x2 = self.lin2(x2).view(-1, self.in_channels)
        alpha_l = (x1 * self.att1).unsqueeze(-1).sum(dim=-1)
        alpha_r = (x2 * self.att2).unsqueeze(-1).sum(dim=-1)

        out = self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r), years=years)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, years):
        year1 = years[0].float()
        year2 = years[1].float()

        alpha = alpha_j + alpha_i
        gap = torch.abs(year1 - year2).exp_().pow_(-1).unsqueeze(-1).type_as(x_j)
        alpha = (alpha * gap.type_as(x_j))
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr)
        alpha = alpha.sum(dim=-1)
        alpha = F.dropout(alpha, p=0.5, training=self.training)
        self._alpha = alpha

        return x_j * alpha.unsqueeze(-1)


class RGAT(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int, num_relations: int, num_layers: int,
                 heads: int = 4, dropout: float = 0.5):
        super(RGAT, self).__init__()
        dataset = MAG240MDataset(2, 0.1, 0)
        self.years = torch.from_numpy(dataset.all_paper_year)
        self.num_relations = num_relations
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()
        # self.emb = ModuleList()

        self.convs.append(
            ModuleList([
                GATConv(in_channels, hidden_channels // heads, heads,
                        add_self_loops=False) for _ in range(num_relations)
            ]))

        for _ in range(num_layers - 1):
            self.convs.append(
                ModuleList([
                    GATConv(hidden_channels, hidden_channels // heads,
                            heads, add_self_loops=False)
                    for _ in range(num_relations)
                ]))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        # self.emb.append(Temporal_emd_5(in_channels))
        # self.emb.append(Temporal_emd_5(hidden_channels))

        self.skips.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.skips.append(Linear(hidden_channels, hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:

        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]

            out = self.skips[i](x_target)

            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                # if j == 0:
                #     row, col, _ = subadj_t.coo()
                #     year1 = years[row]
                #     year_target = years[col]
                #     x_target = self.emb[i]((year1, year_target), subadj_t, (x, x_target))

                if subadj_t.nnz() > 0:
                    out += self.convs[i][j]((x, x_target), subadj_t)

            x = self.norms[i](out)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


class MagDataset(object):

    def __init__(self, batch_size, size):
        self.batch_size = batch_size
        self.sizes = size
        self.ratio = 0.1
        self.dataset = MAG240MDataset(2, self.ratio, 0)
        self.years = torch.from_numpy(self.dataset.all_paper_year)
        # self.world_size = world_size
        # self.rank = rank

        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = self.dataset

        self.train_idx = dataset.get_idx_split('train')
        self.val_idx = dataset.get_idx_split('valid')
        self.test_idx = dataset.get_idx_split('test')

        self.x = torch.from_numpy(dataset.paper_feat)
        # self.x = torch.from_numpy(np.memmap(f'/data/shangyihao/mag240m/RGNN/full_feat{self.ratio}.npy', dtype=np.float16,
        #                    mode='r', shape=(N, self.num_features)))
        self.y = torch.from_numpy(dataset.all_paper_label)

        path = f'/data/shangyihao/mag240m/newedges{self.ratio}/adj_same_author.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    @property
    def num_features(self) -> int:
        return self.dataset.num_paper_features

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    @property
    def num_relations(self) -> int:
        return 5

    def train_dataloader(self):
        # train_sampler = torch.utils.data.distributed.DistributedSampler(self, num_replicas=self.world_size,
        #                                                                 rank=self.rank)
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=2)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch, shuffle=True,
                               batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=3)

    def convert_batch(self, batch_size, n_id, adjs):
        x = self.x[n_id.numpy()].to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)

        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])


def run(args):

    devices = f'cuda:{args.devices}' if torch.cuda.is_available() else 'cpu'

    # dist.init_process_group('nccl', rank=rank, world_size=n_gpus)

    dataset = MagDataset(args.batch_size, args.sizes)
    # train_idx = dataset.train_idx
    # valid_idx = dataset.val_idx
    # test_idx = dataset.test_idx

    train_loader = dataset.train_dataloader()
    evaluator = MAG240MEvaluator()

    val_loader = dataset.val_dataloader()

    torch.manual_seed(12345)
    # torch.cuda.set_device(args.local_rank)
    model = RGAT(dataset.num_features, dataset.num_classes, args.hidden_channels,
                 dataset.num_relations, num_layers=len(args.sizes), dropout=args.dropout).to(devices)
    print(f'#Params {sum([p.numel() for p in model.parameters()])}')
    # model = DistributedDataParallel(model, device_ids=[devices])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

    best_valid_acc = 0
    for epoch in tqdm(range(args.epochs)):
        # train_loader.sampler.set_epoch(epoch)
        model.train()

        total_loss = 0
        y_pred, y_true = [], []
        for data in tqdm(train_loader):
            data = data.to(devices)
            # years = dataset.years
            y_hat = model(data.x, data.adjs_t)
            train_loss = F.cross_entropy(y_hat, data.y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                y_pred.append(y_hat.argmax(dim=-1))
                y_true.append(data.y)

            total_loss += float(train_loss) * y_hat.size(0)
        loss = total_loss / len(train_loader.dataset)

        # y_pred, y_true = [], []
        # for data in train_loader:
        #     data = data.to(devices)
        #     years = dataset.years
        #     y_hat = model(data.x, data.adjs_t, years)
        #     with torch.no_grad():
        #         model.eval()
        #         y_pred.append(y_hat.argmax(dim=-1))
        #         y_true.append(data.y)

        train_acc = evaluator.eval({'y_true': torch.cat(y_true, dim=0),
                                    'y_pred': torch.cat(y_pred, dim=0)})['acc']

        model.eval()
        y_pred, y_true = [], []
        for data in tqdm(val_loader):
            data = data.to(devices)
            # years = dataset.years
            y_hat = model(data.x, data.adjs_t)
            y_pred.append(y_hat.argmax(dim=-1))
            y_true.append(data.y)

        valid_acc = evaluator.eval({'y_true': torch.cat(y_true, dim=0),
                                    'y_pred': torch.cat(y_pred, dim=0)})['acc']

        scheduler.step(valid_acc)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
              f'Train_acc: {train_acc:.4f}, Valid_acc: {valid_acc:.4f}, '
              f'Best_valid_acc: {best_valid_acc:.4f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--devices', type=str, default='6')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    # devices = list(map(int, args.devices.split(',')))
    # n_gpus = len(devices)
    torch.manual_seed(12345)

    run(args)



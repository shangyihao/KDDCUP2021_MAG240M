import os
import time
import glob
import argparse
import math
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Optional, List, NamedTuple, Union, Tuple
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv, MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.data import NeighborSampler

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

        out = self.propagate(edge_index, x=(x1, x2), alpha=(alpha_l, alpha_r), years=years)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, years):
        year1 = years[0].float()
        year2 = years[1].float()

        alpha = alpha_j + alpha_i
        gap = torch.abs(year1 - year2).exp_().pow_(-1).unsqueeze(-1).to(device=x_j.device)
        alpha = (alpha * gap.to(device=x_j.device))
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr)
        alpha = alpha.sum(dim=-1)
        alpha = F.dropout(alpha, p=0.5, training=self.training)
        self._alpha = alpha

        return x_j * alpha.unsqueeze(-1)


class Temporal_emd_6(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Temporal_emd_6, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att1 = torch.nn.Parameter(torch.Tensor(1, self.out_channels))
        self.att2 = torch.nn.Parameter(torch.Tensor(1, self.out_channels))
        self.lin1 = Linear(self.in_channels, self.out_channels)
        self.lin2 = Linear(self.in_channels, self.out_channels)
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
        x1 = self.lin1(x1).view(-1, self.out_channels)
        x2 = self.lin2(x2).view(-1, self.out_channels)
        alpha_l = (x1 * self.att1).unsqueeze(-1).sum(dim=-1)
        alpha_r = (x2 * self.att2).unsqueeze(-1).sum(dim=-1)

        out = self.propagate(edge_index, x=(x1, x2), alpha=(alpha_l, alpha_r), years=years)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, years):
        year1 = years[0].float()
        year2 = years[1].float()

        alpha = alpha_j + alpha_i
        gap = torch.abs(year1 - year2).exp_().pow_(-1).unsqueeze(-1).to(device=x_j.device)
        alpha = (alpha * gap.to(device=x_j.device))
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr)
        alpha = alpha.sum(dim=-1)
        alpha = F.dropout(alpha, p=0.5, training=self.training)
        self._alpha = alpha

        return x_j * alpha.unsqueeze(-1)


class Temporal_emd(MessagePassing):

    def __init__(self, in_channels):
        super(Temporal_emd, self).__init__()
        self.in_channels = in_channels
        self.att = torch.nn.Parameter(torch.Tensor(1, self.in_channels))
        self.lin = Linear(self.in_channels, self.in_channels)
        self.negative_slope = 0.2
        self.dropout = 0.5

    def forward(self, years, edge_index, x):
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        x = self.lin(x).view(-1, self.in_channels)
        alpha_l = (x * self.att).unsqueeze(-1)
        alpha_l = alpha_l.sum(dim=-1)
        return self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r), years=years)

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, index: Tensor, ptr: OptTensor, years):
        year1 = years[0].float()
        year2 = years[1].float()
        # year_gap = torch.abs(year1 - year2).numpy()
        gap = torch.abs(year1 - year2).exp_().pow_(-1).unsqueeze(-1).to(device=x_j.device)
        alpha = (alpha_j * gap.to(device=x_j.device))
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha, index, ptr)
        alpha = alpha.sum(dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)


class Temporal_emd_7(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Temporal_emd_7, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att1 = torch.nn.Parameter(torch.Tensor(1, self.out_channels))
        self.att2 = torch.nn.Parameter(torch.Tensor(1, self.out_channels))
        self.lin1 = Linear(self.in_channels, self.out_channels)
        self.lin2 = Linear(self.in_channels, self.out_channels)
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
        size = None
        self.gap = torch.abs(year1 - year2)

        x1, x2 = x[0], x[1]
        x1 = self.lin1(x1).view(-1, self.out_channels)
        x2 = self.lin2(x2).view(-1, self.out_channels)
        alpha_l = (x1 * self.att1)
        alpha_r = (x2 * self.att2)

        out = self.propagate(edge_index, x=(x1, x2), alpha=(alpha_l, alpha_r), size=size,
                             years=years)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, years):
        year1 = years[0].float()
        year2 = years[1].float()

        alpha = alpha_j + alpha_i
        gap = torch.abs(year1 - year2).exp_().pow_(-1).unsqueeze(-1).to(device=x_j.device)
        # alpha = (alpha * gap.to(device=x_j.device)).sum(dim=-1)
        alpha = (alpha * gap.to(device=x_j.device))
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr)
        # alpha = alpha.sum(dim=-1)
        alpha = F.dropout(alpha, p=0.5, training=self.training)
        self._alpha = alpha

        return x_j * alpha


class Temporal_emd_8(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Temporal_emd_8, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att1 = torch.nn.Parameter(torch.Tensor(1, self.out_channels))
        self.att2 = torch.nn.Parameter(torch.Tensor(1, self.out_channels))
        self.lin1 = Linear(self.in_channels, self.out_channels)
        self.lin2 = Linear(self.in_channels, self.out_channels)
        self.negative_slope = 0.2
        self.gap = None
        self._alpha = None
        self.sqrt_dk = math.sqrt(out_channels)

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
        x1 = self.lin1(x1).view(-1, self.out_channels)
        x2 = self.lin2(x2).view(-1, self.out_channels)
        alpha_l = (x1 * self.att1)
        alpha_r = (x2 * self.att2)

        out = self.propagate(edge_index, x=(x1, x2), alpha=(alpha_l, alpha_r), years=years)
        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, years):
        year1 = years[0].float()
        year2 = years[1].float()

        alpha = alpha_j + alpha_i
        gap = torch.abs(year1 - year2).exp_().pow_(-1).unsqueeze(-1).to(device=x_j.device)
        alpha = (alpha * gap.to(device=x_j.device))
        alpha = (alpha / self.sqrt_dk).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr)
        alpha = F.dropout(alpha, p=0.5, training=self.training)
        self._alpha = alpha

        return x_j * alpha.unsqueeze(-1)


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    years: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            years=self.years.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)


def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]


class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.ratio = 0.1
        self.dataset = MAG240MDataset(2, self.ratio, 0)

    @property
    def num_features(self) -> int:
        return self.dataset.num_paper_features

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    @property
    def num_relations(self) -> int:
        return 5

    def prepare_data(self):

        path = f'/data/shangyihao/mag240m/RGNN/paper_to_paper_symmetric{self.ratio}.pt'
        if not osp.exists(path):  # Will take approximately 5 minutes...
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = self.dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(self.dataset.num_papers, self.dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'/data/shangyihao/mag240m/RGNN/full_adj_t{self.ratio}.pt'
        if not osp.exists(path):  # Will take approximately 16 minutes...
            t = time.perf_counter()
            print('Merging adjacency matrices...', end=' ', flush=True)

            row, col, _ = torch.load(
                f'/data/shangyihao/mag240m/RGNN/paper_to_paper_symmetric{self.raio}.pt').coo()
            rows, cols = [row], [col]

            edge_index = self.dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            row += self.dataset.num_papers
            rows += [row, col]
            cols += [col, row]

            edge_index = self.dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            row += self.dataset.num_papers
            col += self.dataset.num_papers + self.dataset.num_authors
            rows += [row, col]
            cols += [col, row]

            edge_types = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]

            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols

            N = (self.dataset.num_papers + self.dataset.num_authors +
                 self.dataset.num_institutions)

            perm = (N * row).add_(col).numpy().argsort()
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]

            edge_type = torch.cat(edge_types, dim=0)[perm]
            del edge_types

            full_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                      sparse_sizes=(N, N), is_sorted=True)

            torch.save(full_adj_t, path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'/data/shangyihao/mag240m/RGNN/full_feat{self.ratio}.npy'
        done_flag_path = f'/data/shangyihao/mag240m/RGNN/full_feat_done{self.ratio}.txt'
        if not osp.exists(done_flag_path):  # Will take ~3 hours...
            t = time.perf_counter()
            print('Generating full feature matrix...')

            node_chunk_size = 100000
            dim_chunk_size = 64
            N = (self.dataset.num_papers + self.dataset.num_authors +
                 self.dataset.num_institutions)

            paper_feat = self.dataset.paper_feat
            x = np.memmap(path, dtype=np.float16, mode='w+',
                          shape=(N, self.num_features))

            print('Copying paper features...')
            for i in tqdm(range(0, self.dataset.num_papers, node_chunk_size)):
                j = min(i + node_chunk_size, self.dataset.num_papers)
                x[i:j] = paper_feat[i:j]

            edge_index = self.dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=row, col=col,
                sparse_sizes=(self.dataset.num_authors, self.dataset.num_papers),
                is_sorted=True)

            # Processing 64-dim subfeatures at a time for memory efficiency.
            print('Generating author features...')
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(paper_feat, start_row_idx=0,
                                       end_row_idx=self.dataset.num_papers,
                                       start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x, start_row_idx=self.dataset.num_papers,
                    end_row_idx=self.dataset.num_papers + self.dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                del outputs

            edge_index = self.dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=col, col=row,
                sparse_sizes=(self.dataset.num_institutions, self.dataset.num_authors),
                is_sorted=False)

            print('Generating institution features...')
            # Processing 64-dim subfeatures at a time for memory efficiency.
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(
                    x, start_row_idx=self.dataset.num_papers,
                    end_row_idx=self.dataset.num_papers + self.dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x,
                    start_row_idx=self.dataset.num_papers + self.dataset.num_authors,
                    end_row_idx=N, start_col_idx=i, end_col_idx=j)
                del outputs

            x.flush()
            del x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            with open(done_flag_path, 'w') as f:
                f.write('done')

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = self.dataset

        # train_path = '/data/shangyihao/mag240m/new_feat(train_valid)0.1_0.7/type_4/train_index.pt'
        # self.train_idx = torch.load(train_path)
        self.train_idx = dataset.get_idx_split('train')
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        # valid_path = '/data/shangyihao/mag240m/new_feat(train_valid)0.1_0.7/type_5/valid_index.pt'
        # self.val_idx = torch.load(valid_path)
        self.val_idx = dataset.get_idx_split('valid')
        self.val_idx.share_memory_()
        self.test_idx = dataset.get_idx_split('test')
        self.test_idx.share_memory_()

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        self.x = np.memmap(f'/data/shangyihao/mag240m/RGNN/full_feat{self.ratio}.npy', dtype=np.float16,
                           mode='r', shape=(N, self.num_features))

        year_path = '/data/shangyihao/mag240m/RGNN/all_year.npy'
        self.years = np.load(year_path)
        self.y = torch.from_numpy(dataset.all_paper_label)

        path = f'/data/shangyihao/mag240m/RGNN/full_adj_t{self.ratio}.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=4)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
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
        x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        years = torch.from_numpy(self.years[n_id.numpy()])
        y = self.y[n_id[:batch_size]].to(torch.long)

        return Batch(x=x, y=y, years=years, adjs_t=[adj_t for adj_t, _, _ in adjs])


class RGNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_relations: int, num_layers: int,
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        dataset = MAG240MDataset(2, 0.1, 0)
        self.save_hyperparameters()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()
        self.emb = ModuleList()

        if self.model == 'rgat':
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

        elif self.model == 'rgraphsage':
            self.convs.append(
                ModuleList([
                    SAGEConv(in_channels, hidden_channels, root_weight=False)
                    for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        SAGEConv(hidden_channels, hidden_channels,
                                 root_weight=False)
                        for _ in range(num_relations)
                    ]))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.emb.append(Temporal_emd_8(in_channels, in_channels))
        self.emb.append(Temporal_emd_8(hidden_channels, hidden_channels))

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

        self.acc = Accuracy()

    def forward(self, x: Tensor, adjs_t: List[SparseTensor], years) -> Tensor:

        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]
            out = self.skips[i](x_target)
            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                if subadj_t.nnz() > 0:
                    if j == 0:
                        row, col, _ = subadj_t.coo()
                        year1 = years[row]
                        year_target = years[col]
                        x_target = self.emb[i]((year1, year_target), subadj_t, (x, x_target))
                        out += self.convs[i][j]((x, x_target), subadj_t)
                    else:
                        out += self.convs[i][j]((x, x_target), subadj_t)

            x = self.norms[i](out)
            x = F.elu(x) if self.model == 'rgat' else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t, batch.years)
        train_loss = F.cross_entropy(y_hat, batch.y)
        train_acc = self.acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t, batch.years)
        val_acc = self.acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return val_acc

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t, batch.years)
        test_acc = self.acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return test_acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--device', type=str, default='6')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes)

    if not args.evaluate:
        model = RGNN(args.model, datamodule.num_features,
                     datamodule.num_classes, args.hidden_channels,
                     datamodule.num_relations, num_layers=len(args.sizes),
                     dropout=args.dropout)
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1,
                                              filename='epoch={epoch}-validacc={val_acc:.4f}')
        trainer = Trainer(gpus=args.device, max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          default_root_dir=f'logs/{args.model}_year')
        trainer.fit(model, datamodule=datamodule)

    if args.evaluate:
        dirs = glob.glob(f'logs/{args.model}_try/lightning_logs/*')
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = f'logs/{args.model}_try/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]

        trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)
        model = RGNN.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

        trainer.test(model=model, datamodule=datamodule)

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()

        model.eval()
        y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(int(args.device))
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        # evaluator.save_test_submission(res, f'results/{args.model}')

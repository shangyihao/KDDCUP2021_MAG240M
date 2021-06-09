import os
import time
import glob
import argparse
import os.path as osp
from tqdm import tqdm

from typing import Optional, List, NamedTuple

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
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT


dataset = MAG240MDataset(ROOT)

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


def prepare_data():

    path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
    if not osp.exists(path):  # Will take approximately 5 minutes...
        t = time.perf_counter()
        print('Converting adjacency matrix...', end=' ', flush=True)
        edge_index = dataset.edge_index('paper', 'cites', 'paper')
        edge_index = torch.from_numpy(edge_index)
        adj_t = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(dataset.num_papers, dataset.num_papers),
            is_sorted=True)
        torch.save(adj_t.to_symmetric(), path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    path = f'{dataset.dir}/full_adj_t.pt'
    if not osp.exists(path):  # Will take approximately 16 minutes...
        t = time.perf_counter()
        print('Merging adjacency matrices...', end=' ', flush=True)

        row, col, _ = torch.load(
            f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()
        rows, cols = [row], [col]

        edge_index = dataset.edge_index('author', 'writes', 'paper')
        row, col = torch.from_numpy(edge_index)
        row += dataset.num_papers
        rows += [row, col]
        cols += [col, row]

        edge_index = dataset.edge_index('author', 'institution')
        row, col = torch.from_numpy(edge_index)
        row += dataset.num_papers
        col += dataset.num_papers + dataset.num_authors
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

        N = (dataset.num_papers + dataset.num_authors +
             dataset.num_institutions)

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

    path = f'{dataset.dir}/full_feat.npy'
    done_flag_path = f'{dataset.dir}/full_feat_done.txt'
    if not osp.exists(done_flag_path):  # Will take ~3 hours...
        t = time.perf_counter()
        print('Generating full feature matrix...')

        node_chunk_size = 100000
        dim_chunk_size = 64
        N = (dataset.num_papers + dataset.num_authors +
             dataset.num_institutions)

        paper_feat = dataset.paper_feat
        x = np.memmap(path, dtype=np.float16, mode='w+',
                      shape=(N, dataset.num_paper_features))

        print('Copying paper features...')
        for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
            j = min(i + node_chunk_size, dataset.num_papers)
            x[i:j] = paper_feat[i:j]

        edge_index = dataset.edge_index('author', 'writes', 'paper')
        row, col = torch.from_numpy(edge_index)
        adj_t = SparseTensor(
            row=row, col=col,
            sparse_sizes=(dataset.num_authors, dataset.num_papers),
            is_sorted=True)

        # Processing 64-dim subfeatures at a time for memory efficiency.
        print('Generating author features...')
        for i in tqdm(range(0, dataset.num_paper_features, dim_chunk_size)):
            j = min(i + dim_chunk_size, dataset.num_paper_features)
            inputs = get_col_slice(paper_feat, start_row_idx=0,
                                   end_row_idx=dataset.num_papers,
                                   start_col_idx=i, end_col_idx=j)
            inputs = torch.from_numpy(inputs)
            outputs = adj_t.matmul(inputs, reduce='mean').numpy()
            del inputs
            save_col_slice(
                x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
                end_row_idx=dataset.num_papers + dataset.num_authors,
                start_col_idx=i, end_col_idx=j)
            del outputs

        edge_index = dataset.edge_index('author', 'institution')
        row, col = torch.from_numpy(edge_index)
        adj_t = SparseTensor(
            row=col, col=row,
            sparse_sizes=(dataset.num_institutions, dataset.num_authors),
            is_sorted=False)

        print('Generating institution features...')
        # Processing 64-dim subfeatures at a time for memory efficiency.
        for i in tqdm(range(0, dataset.num_paper_features, dim_chunk_size)):
            j = min(i + dim_chunk_size, dataset.num_paper_features)
            inputs = get_col_slice(
                x, start_row_idx=dataset.num_papers,
                end_row_idx=dataset.num_papers + dataset.num_authors,
                start_col_idx=i, end_col_idx=j)
            inputs = torch.from_numpy(inputs)
            outputs = adj_t.matmul(inputs, reduce='mean').numpy()
            del inputs
            save_col_slice(
                x_src=outputs, x_dst=x,
                start_row_idx=dataset.num_papers + dataset.num_authors,
                end_row_idx=N, start_col_idx=i, end_col_idx=j)
            del outputs

        x.flush()
        del x
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

        with open(done_flag_path, 'w') as f:
            f.write('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    seed_everything(42)
    prepare_data()

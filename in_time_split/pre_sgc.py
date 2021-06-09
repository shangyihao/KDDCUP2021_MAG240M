# NOTE: 128-256GB CPU memory required to run this script.

import os
import time
import argparse
import os.path as osp
from tqdm import tqdm

import torch
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.lsc import MAG240MDataset
ROOT = '/data/shangyihao/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=3),
    args = parser.parse_args()
    print(args)

    dataset = MAG240MDataset(ROOT)
    t = time.perf_counter()
    print('Reading adjacency matrix...', end=' ', flush=True)
    path = f'{dataset.dir}/process_shang/time_split/paper_to_paper_train_symmetric_gcn.pt'
    if osp.exists(path):
        adj_t = torch.load(path)
    else:
        path_sym = f'{dataset.dir}/process_shang/time_split/paper_to_paper_train_symmetric.pt'
        if osp.exists(path):
            adj_t = torch.load(path_sym)
        else:
            num = dataset.num_papers
            edge_file = f'{dataset.dir}/process_shang/split_graph_dict.npz'
            edge_idx = np.load(edge_file)['train']
            print(edge_idx.shape)
            edge_idx = edge_idx.T
            print(edge_idx.shape)
            edge_idx = torch.from_numpy(edge_idx)
            adj_t = SparseTensor(
                row=edge_idx[0], col=edge_idx[1],
                sparse_sizes=(num, num), is_sorted=True)
            adj_t = adj_t.to_symmetric()
            torch.save(adj_t, path_sym)
        adj_t = gcn_norm(adj_t, add_self_loops=True)
        torch.save(adj_t, path)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    split_file = f'{dataset.dir}/process_shang/split_labels_dict.npy'

    split = np.load(split_file, allow_pickle=True).item()
    train_idx = split['train']
    valid_idx = split['valid']
    test_idx = split['test']
    num_features = dataset.num_paper_features

    pbar = tqdm(total=args.num_layers * (num_features // 128))
    pbar.set_description('Pre-processing node features')

    for j in range(0, num_features, 128):  # Run spmm in column-wise chunks...
        x = dataset.paper_feat[:, j:min(j + 128, num_features)]
        x = torch.from_numpy(x.astype(np.float32))

        for i in range(1, args.num_layers + 1):
            x = adj_t @ x
            np.save(f'{dataset.dir}/process_shang/time_split/x_train_{i}_{j}.npy', x[train_idx].numpy())
            np.save(f'{dataset.dir}/process_shang/time_split/x_valid_{i}_{j}.npy', x[valid_idx].numpy())
            np.save(f'{dataset.dir}/process_shang/time_split/x_test_{i}_{j}.npy', x[test_idx].numpy())
            pbar.update(1)
    pbar.close()

    t = time.perf_counter()
    print('Merging node features...', end=' ', flush=True)
    for i in range(1, args.num_layers + 1):
        x_train, x_valid, x_test = [], [], []
        for j in range(0, num_features, 128):
            x_train += [np.load(f'{dataset.dir}/process_shang/time_split/x_train_{i}_{j}.npy')]
            x_valid += [np.load(f'{dataset.dir}/process_shang/time_split/x_valid_{i}_{j}.npy')]
            x_test += [np.load(f'{dataset.dir}/process_shang/time_split/x_test_{i}_{j}.npy')]
        x_train = np.concatenate(x_train, axis=-1)
        x_valid = np.concatenate(x_valid, axis=-1)
        x_test = np.concatenate(x_test, axis=-1)
        np.save(f'{dataset.dir}/process_shang/time_split/x_train_{i}.npy', x_train)
        np.save(f'{dataset.dir}/process_shang/time_split/x_valid_{i}.npy', x_valid)
        np.save(f'{dataset.dir}/process_shang/time_split/x_test_{i}.npy', x_test)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    print('Cleaning up...', end=' ', flush=True)
    for i in range(1, args.num_layers + 1):
        for j in range(0, num_features, 128):
            os.remove(f'{dataset.dir}/process_shang/time_split/x_train_{i}_{j}.npy')
            os.remove(f'{dataset.dir}/process_shang/time_split/x_valid_{i}_{j}.npy')
            os.remove(f'{dataset.dir}/process_shang/time_split/x_test_{i}_{j}.npy')
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

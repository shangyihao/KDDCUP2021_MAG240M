# NOTE: 256GB CPU memory required to run this script.

import os.path as osp
import time
import argparse

import torch
import numpy as np
from torch_sparse import SparseTensor
from CorrectAndSmooth import CorrectAndSmooth
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
ROOT = '/data/shangyihao/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_correction_layers', type=int, default=3)
    parser.add_argument('--correction_alpha', type=float, default=1.0)
    parser.add_argument('--num_smoothing_layers', type=int, default=2)
    parser.add_argument('--smoothing_alpha', type=float, default=0.8)
    args = parser.parse_args()
    print(args)

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    t = time.perf_counter()
    print('Reading graph-agnostic predictions...', end=' ', flush=True)
    y_pred = torch.from_numpy(np.load('results/cs/pred.npy'))
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    print('Reading adjacency matrix...', end=' ', flush=True)
    path = f'{dataset.dir}/process_shang/time_split/paper_to_paper_train_symmetric_gcn.pt'
    if osp.exists(path):
        adj_t = torch.load(path)
    else:
        path_sym = f'{dataset.dir}/process_shang/time_split/paper_to_paper_train_symmetric.pt'
        if osp.exists(path_sym):
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

    y_train = torch.from_numpy(dataset.paper_label[train_idx]).to(torch.long)
    y_valid = torch.from_numpy(dataset.paper_label[valid_idx]).to(torch.long)
    train_idx = np.asarray(train_idx)
    train_idx = torch.from_numpy(train_idx)

    model = CorrectAndSmooth(args.num_correction_layers, args.correction_alpha,
                             args.num_smoothing_layers, args.smoothing_alpha,
                             autoscale=True)

    t = time.perf_counter()
    print('Correcting predictions...', end=' ', flush=True)
    y_pred = model.correct(y_pred, y_train, train_idx, adj_t)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    print('Smoothing predictions...', end=' ', flush=True)
    y_pred = model.smooth(y_pred, y_train, train_idx, adj_t)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    train_acc = evaluator.eval({
        'y_true': y_train,
        'y_pred': y_pred[train_idx].argmax(dim=-1)
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_valid,
        'y_pred': y_pred[valid_idx].argmax(dim=-1)
    })['acc']
    print(f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}')

    res = {'y_pred': y_pred[test_idx].argmax(dim=-1)}
    # evaluator.save_test_submission(res, 'results/cs')
    y_test = torch.from_numpy(dataset.paper_label[test_idx])
    y_test = y_test.to(torch.long)
    res['y_true'] = y_test
    acc = evaluator.eval(res)['acc']
    print(f'Test_acc: {acc:.4f} ')
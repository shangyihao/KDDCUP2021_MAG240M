# NOTE: 256GB CPU memory required to run this script.

import os.path as osp
import time
import argparse
import numpy as np
import torch
from torch_sparse import SparseTensor
from LabelPropagation import LabelPropagation
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=3),
    parser.add_argument('--alpha', type=float, default=0.9),
    args = parser.parse_args()
    print(args)

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    t = time.perf_counter()
    print('Reading adjacency matrix...', end=' ', flush=True)
    path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
    if osp.exists(path):
        adj_t = torch.load(path)
    else:
        edge_index = dataset.edge_index('paper', 'cites', 'paper')
        edge_index = torch.from_numpy(edge_index)
        adj_t = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(dataset.num_papers, dataset.num_papers),
            is_sorted=True)
        adj_t = adj_t.to_symmetric()
        torch.save(adj_t, path)
    adj_t = gcn_norm(adj_t, add_self_loops=False)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    split_file = f'{dataset.dir}/process_shang/split_labels_dict.npy'

    split = np.load(split_file, allow_pickle=True).item()
    train_idx = split['train']
    valid_idx = split['valid']
    test_idx = split['test']

    y_train = torch.from_numpy(dataset.paper_label[train_idx]).to(torch.long)
    y_valid = torch.from_numpy(dataset.paper_label[valid_idx]).to(torch.long)
    y_test = torch.from_numpy(dataset.paper_label[test_idx]).to(torch.long)

    model = LabelPropagation(args.num_layers, args.alpha)

    t = time.perf_counter()
    print('Propagating labels...', end=' ', flush=True)
    y_pred = model(y_train, adj_t, train_idx).argmax(dim=-1)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    train_acc = evaluator.eval({
        'y_true': y_train,
        'y_pred': y_pred[train_idx]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_valid,
        'y_pred': y_pred[valid_idx]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_test,
        'y_pred': y_pred[test_idx]
    })['acc']
    print(f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, Test:{test_acc:.4f}')

    # res = {'y_pred': y_pred[test_idx]}
    # evaluator.save_test_submission(res, 'results/label_prop')

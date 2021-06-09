from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import os.path as osp
import torch
import pickle
import argparse
import time
import numpy as np
from tqdm import tqdm
ROOT = '/data/shangyihao/'

dataset = MAG240MDataset(ROOT)


def prep_data(id_sub):
    subgraph_file = f'{dataset.dir}/process_shang/subsample/subgraph_k_hop_0.1_{id_sub}'
    '''
        Node index operation
    '''
    t = time.perf_counter()
    arxiv_file = osp.join(subgraph_file, 'arxiv_dict.pt')
    arxiv_dict = torch.load(arxiv_file)

    node_file = osp.join(subgraph_file, 'sampled_idx.pt')
    node_dict = torch.load(node_file)

    num_nodes = {key: len(node_dict[key].tolist()) for key in node_dict}
    print(num_nodes.values())
    node_map = {}
    for _type in node_dict:
        i = 0
        _map = {}
        for idx in node_dict[_type]:
            idx = idx.item()
            _map[idx] = i
            i += 1
        node_map[_type] = _map
    file = osp.join(subgraph_file, 'node_map.txt')
    with open(file, 'wb') as f:
        pickle.dump(node_map, f)
    print('saved node_map! ')
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    split_dict = {}
    paper_map = node_map['paper']
    for key in arxiv_dict:
        key_set = arxiv_dict[key]
        new_list = []
        for idx in key_set:
            new_list.append(paper_map[idx.item()])
        new_list.sort()
        split_dict[key] = new_list
    file = osp.join(subgraph_file, 'split_dict.txt')
    with open(file, 'wb') as f:
        pickle.dump(split_dict, f)
    print('saved split_dict! ')
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    '''
        Edge index operation
    '''
    t = time.perf_counter()
    edge_file = osp.join(subgraph_file, 'alledge_index.pt')
    all_edge_dict = torch.load(edge_file)
    edge_index_dict = {}
    edge_key = [('paper', 'cites', 'paper'), ('author', 'writes', 'paper'),
                ('author', 'affiliated_with', 'institution')]

    rows, cols = all_edge_dict['cites'].numpy()
    paper_map = node_map['paper']
    new_edges = []
    for i in tqdm(range(len(rows))):
        e = [rows[i], cols[i]]
        new_e = [paper_map[e[0]], paper_map[e[1]]]
        new_edges.append(new_e)
    new_edges = (np.asarray(new_edges)).T
    new_edges = torch.from_numpy(new_edges)
    edge_index_dict[edge_key[0]] = new_edges

    rows, cols = all_edge_dict['writes'].numpy()
    author_map = node_map['author']
    new_edges = []
    for i in tqdm(range(len(rows))):
        e = [rows[i], cols[i]]
        new_e = [author_map[e[0]], paper_map[e[1]]]
        new_edges.append(new_e)
    new_edges = (np.asarray(new_edges)).T
    new_edges = torch.from_numpy(new_edges)
    edge_index_dict[edge_key[1]] = new_edges

    rows, cols = all_edge_dict['affs'].numpy()
    insti_map = node_map['institution']
    new_edges = []
    for i in tqdm(range(len(rows))):
        e = [rows[i], cols[i]]
        new_e = [author_map[e[1]], insti_map[e[0]]]
        new_edges.append(new_e)
    new_edges = (np.asarray(new_edges)).T
    new_edges = torch.from_numpy(new_edges)
    edge_index_dict[edge_key[2]] = new_edges

    file = osp.join(subgraph_file, 'edge_index.pt')
    torch.save(edge_index_dict, file)
    print('saved edge_index_dict! ')
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    '''
        Generate new year map
    '''
    t = time.perf_counter()
    years = dataset.paper_year
    new_years = np.zeros(num_nodes['paper'], dtype=int)
    for idx in node_dict['paper']:
        idx = idx.item()
        loc = paper_map[idx]
        new_years[loc] = years[idx]
    file = osp.join(subgraph_file, 'years.npy')
    np.save(file, new_years)
    print('saved year map! ')
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    '''
        Generate new paper label and feature
    '''
    t = time.perf_counter()
    labels = dataset.paper_label
    y = np.zeros(num_nodes['paper'], dtype=int)
    feat_file = osp.join(subgraph_file, 'paper_feat.npy')
    new_feat = np.memmap(feat_file, dtype=np.float, mode='w+', shape=(num_nodes['paper'], dataset.num_paper_features))
    pbar = tqdm(total=num_nodes['paper'])
    for idx in node_dict['paper']:
        idx = idx.item()
        loc = paper_map[idx]
        new_feat[loc] = dataset.paper_feat[idx]
        if labels[idx] >= 0:
            y[loc] = labels[idx]
        else:
            y[loc] = -1
        pbar.update(1)
    pbar.close()
    file = osp.join(subgraph_file, 'labels.npy')
    np.save(file, y)
    print('saved paper label and features! ')
    print(f'Done! [{time.perf_counter() - t:.2f}s]')


for id_sub in range(1, 6):
    prep_data(id_sub)
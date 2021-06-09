import time
import os.path as osp
from os import makedirs
from ogb.lsc import MAG240MDataset
from tqdm import tqdm
import numpy as np
from torch_geometric.utils import k_hop_subgraph
import torch
import pickle
import argparse


def load_alledge(path):

    if not osp.exists(path):
        edge_index = dataset.edge_index('paper', 'paper')
        rows, cols = torch.from_numpy(edge_index)

        edge_index = dataset.edge_index('author', 'writes', 'paper')
        row, col = torch.from_numpy(edge_index)
        row += dataset.num_papers  # let author_idx start from dataset.num_papers
        rows = torch.cat([rows, row], dim=0)
        cols = torch.cat([cols, col], dim=0)

        edge_index = dataset.edge_index('author', 'institution')
        col, row = torch.from_numpy(edge_index)  # let institution as source_node
        row += (dataset.num_papers + dataset.num_authors)  # let institution_idx start after papers+authors
        col += dataset.num_papers
        rows = torch.cat([rows, row], dim=0)
        cols = torch.cat([cols, col], dim=0)

        edge_index = torch.stack([rows, cols], dim=0)
        torch.save(edge_index, path)
        return edge_index

    else:
        edge_index = torch.load(path)
        return edge_index


def subgraph(sample_depth, ratio, i):  # need about half an hour per i
    path = f'{dataset.dir}/process_shang/subsample/subgraph_{sample_depth}_hop_{ratio}_{i}'
    makedirs(path)

    edge_path = f'{dataset.dir}/process_shang/all_graph_split/all_edge.pt'
    edge_index = load_alledge(edge_path)

    split_dict = dataset.get_idx_split()  # sample according to the given dataset
    train_idx = split_dict['train']
    train_idx = np.random.choice(train_idx, int(ratio * len(train_idx)), replace=False)
    valid_idx = split_dict['valid']
    valid_idx = np.random.choice(valid_idx, int(ratio * len(valid_idx)), replace=False)
    test_idx = split_dict['test']
    test_idx = np.random.choice(test_idx, int(ratio * len(test_idx)), replace=False)

    train_idx = torch.from_numpy(train_idx)
    valid_idx = torch.from_numpy(valid_idx)
    test_idx = torch.from_numpy(test_idx)
    node_index = torch.cat([train_idx, valid_idx, test_idx], dim=0)

    '''
    divide dataset for model testing
    '''
    t = time.perf_counter()
    print('dividing dataset ...', end=' ', flush=True)
    test_set = valid_idx
    year = dataset.paper_year
    train_set = []
    valid_set = []
    for idx in train_idx:
        paper = idx.item()
        if year[paper] == 2018:
            valid_set.append(paper)
        else:
            train_set.append(paper)
    train_set = torch.from_numpy(np.asarray(train_set))
    valid_set = torch.from_numpy(np.asarray(valid_set))
    arxiv_dict = {'train': train_set, 'valid': valid_set, 'test': test_set}

    dict_path = osp.join(path, 'arxiv_dict.pt')
    torch.save(arxiv_dict, dict_path)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    num_hops = sample_depth
    N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
    flow = 'source_to_target'

    t = time.perf_counter()
    print('sampling ...', end=' ', flush=True)
    sampled_node, sampled_edge, map_node, edge_mask = k_hop_subgraph(
        node_idx=node_index, num_hops=num_hops, edge_index=edge_index, flow=flow, num_nodes=N
    )
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    print(f'sampled node size: {sampled_node.shape} ')
    print(f'sampled edge size: {sampled_edge.shape} ')
    print(f'map_node size: {map_node.shape} ')
    print(f'edge_mask size: {edge_mask.shape} ')

    sub_graph = {'node': sampled_node, 'edge': sampled_edge, 'map': map_node, 'mask': edge_mask}
    file = osp.join(path, 'sub_graph.pt')
    torch.save(sub_graph, file)
    '''
    save sampled_node_index per node_type
    '''
    paper_node = []
    author_node = []
    insti_node = []

    for node_id in sampled_node.numpy():
        if node_id < dataset.num_papers:
            paper_node.append(node_id)
        elif node_id < dataset.num_papers + dataset.num_authors:
            author_node.append(node_id)
        else:
            insti_node.append(node_id)
    paper_node = torch.from_numpy(np.asarray(paper_node))
    author_node = torch.from_numpy(np.asarray(author_node))
    insti_node = torch.from_numpy(np.asarray(insti_node))
    sampled_idx = {'paper': paper_node, 'author': author_node, 'institution': insti_node}
    sampled_idx_file = osp.join(path, 'sampled_idx.pt')
    torch.save(sampled_idx, sampled_idx_file)
    '''
    save edge_index per edge_type
    '''
    rows, cols = sampled_edge.numpy()
    num = len(rows)
    edge_cites = []
    edge_writes = []
    edge_affs = []
    for j in tqdm(range(num)):
        e = [rows[j], cols[j]]
        if e[1] in paper_set:
            if e[0] in paper_set:
                edge_cites.append(e)
            else:
                edge_writes.append(e)
        else:
            edge_affs.append(e)

    edge_cites = (np.asarray(edge_cites)).T
    edge_cites = torch.from_numpy(edge_cites)
    edge_writes = (np.asarray(edge_writes)).T
    edge_writes = torch.from_numpy(edge_writes)
    edge_affs = (np.asarray(edge_affs)).T
    edge_affs = torch.from_numpy(edge_affs)

    edge_dict = {'all': sampled_edge, 'cites': edge_cites, 'writes': edge_writes, 'affs': edge_affs}
    edge_file = osp.join(path, 'alledge_index.pt')
    torch.save(edge_dict, edge_file)

    def prep_data():
        '''
            Node index operation
        '''
        t = time.perf_counter()
        node_dict = sampled_idx
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
        file = osp.join(path, 'node_map.txt')
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
        file = osp.join(path, 'split_dict.txt')
        with open(file, 'wb') as f:
            pickle.dump(split_dict, f)
        print('saved split_dict! ')
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

        '''
            Edge index operation
        '''
        t = time.perf_counter()
        all_edge_dict = edge_dict
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

        file = osp.join(path, 'edge_index.pt')
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
        file = osp.join(path, 'years.npy')
        np.save(file, new_years)
        print('saved year map! ')
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

        '''
            Generate new paper label and feature
        '''
        t = time.perf_counter()
        labels = dataset.paper_label
        y = np.zeros(num_nodes['paper'], dtype=int)
        feat_file = osp.join(path, 'paper_feat.npy')
        new_feat = np.memmap(feat_file, dtype=np.float, mode='w+',
                             shape=(num_nodes['paper'], dataset.num_paper_features))
        pbar = tqdm(total=num_nodes['paper'])
        for idx in node_dict['paper']:
            idx = idx.item()
            loc = paper_map[idx]
            new_feat[loc] = dataset.paper_feat[idx]
            if labels[idx] > 0:
                y[loc] = labels[idx]
            pbar.update(1)
        pbar.close()
        file = osp.join(path, 'labels.npy')
        np.save(file, y)
        print('saved paper label and features! ')
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    prep_data()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_depth', type=int, default=3,
                        help='How many numbers to sample the graph')
    parser.add_argument('--batch_ratio', type=float, default=0.6,
                        help='ratio of arxiv papers for sampled')
    args = parser.parse_args()

    ROOT = '/data/shangyihao/'
    dataset = MAG240MDataset(ROOT)
    np.random.seed()

    paper_set = set(range(dataset.num_papers))
    # author_set = set(range(dataset.num_papers, (dataset.num_papers + dataset.num_authors)))
    # insti_set = set(range((dataset.num_papers + dataset.num_authors),
    #                         (dataset.num_papers + dataset.num_authors + dataset.num_institutions)))

    for id in tqdm(range(1, 6)):
        subgraph(args.sample_depth, args.batch_ratio, id)


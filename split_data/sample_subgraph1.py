import time
import os.path as osp
from ogb.lsc import MAG240MDataset
from tqdm import tqdm
from collections import defaultdict
import pickle
import numpy as np
import argparse

np.random.seed(2333)


def create_map(ets):
    _map = defaultdict(  # edge_type:'cites','writes','affiliated with'
        lambda: defaultdict(  # target_id
            lambda: []  # [source_id, target_id]
        ))


    def add_map(_map, edge_type):
        t = time.perf_counter()
        print('load edge_index...', end=' ', flush=True)
        if edge_type == 'cites':
            edge_index = dataset.edge_index('paper', 'paper')
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
        elif edge_type == 'writes':
            edge_index = dataset.edge_index('author', 'paper')
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
        else:
            edge_index = dataset.edge_index('author', 'institution')
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        row, col = edge_index
        del edge_index
        _len = len(row)
        t = time.perf_counter()
        print('add specific map for the given edge_type...', end=' ', flush=True)
        for i in tqdm(range(_len)):
            e = [row[i], col[i]]
            _map[edge_type][col[i]].append(e)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
        del row
        del col


    for _type in ets:
        t = time.perf_counter()
        print(f'add map_{_type}...', end=' ', flush=True)
        add_map(_map, _type)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    # path = f'{dataset.dir}/process_shang/map.txt'
    # if not osp.exists(path):
    #     t = time.perf_counter()
    #     print(f'save map.txt...', end=' ', flush=True)
    #     with open(path,  'wb')as outfile:
    #         pickle.dump(_map, outfile)
    #     print(f'Done! [{time.perf_counter() - t:.2f}s]')
    #
    #     t = time.perf_counter()
    #     print('Cleaning up...', end=' ', flush=True)
    #     for _type in edge_types:
    #         os.remove(f'{dataset.dir}/process_shang/map_{_type}.json')
    #     print(f'Done! [{time.perf_counter() - t:.2f}s]')
    return _map


def sample_subgraph(_map, sampled_num=8, sampled_depth=2, inp=None):

    sample_data = defaultdict(  # target_type : 'paper', 'author', 'institution'
                        lambda: set()  # sampled_node
    )
    budget = defaultdict(  # source_type
                        lambda: defaultdict(  # source_id
                                        lambda: 0.  # sampled_score
        ))

    def add_budget(target_id, ets=edge_types, sd=sample_data, bd=budget):
        for et in ets:
            if et == ets[0]:
                source_type = 'paper'
            else:
                source_type = 'author'
            edge_index = _map[et][target_id]
            nb_list = list(map(lambda i: edge_index[i][0], range(len(edge_index))))
            if len(nb_list) < sampled_num:
                sampled_ids = nb_list
            else:
                sampled_ids = np.random.choice(nb_list, sampled_num, replace=False)
            for source_id in sampled_ids:
                if source_id in sd[source_type]:
                    continue
                bd[source_type][source_id] = 1. / len(sampled_ids)
                # Add candidate node s to budget B with target node t ’s normalized degree.

    for _type in inp:
        for idx in inp[_type]:
            sample_data[_type].add(idx)
    '''
        First adding the sampled nodes then updating budget.
    '''
    for _type in inp:
        if _type == 'paper':
            eg_tps = ['cites', 'writes']
        elif _type == 'author':
            eg_tps = ['writes', 'affiliated with']
        else:
            eg_tps = ['affiliated with']
        for idx in inp[_type]:
            add_budget(idx, ets=eg_tps, sd=sample_data, bd=budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            source_idx = np.array(list(budget[source_type].keys()))
            if sampled_num > len(source_idx):  # directly sample all the nodes
                sampled_ids = np.arange(len(source_idx))
            else:
                score = np.array(list(budget[source_type].values())) ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_num,
                                               p=score, replace=False)
            sampled_keys = source_idx[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for k in sampled_keys:
                sample_data[source_type].add(k)
            for k in sampled_keys:
                add_budget(k, ets=edge_types, sd=sample_data, bd=budget)
                budget[source_type].pop(k)

    def sampled_save():

        node_types = list(sample_data.keys())
        for nt in node_types:
            node_set = list(sample_data[nt]).sort()
            print(f'save sampled node_{nt}...', end=' ', flush=True)
            t = time.perf_counter()
            path = f'{dataset.dir}/process_shang/subgraph/node_{nt}.npy'
            np.save(path, node_set)
            print(f'num of node_{nt}: {len(node_set)}')
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        for edge_type in edge_types:
            t = time.perf_counter()
            print('load edge_index...', end=' ', flush=True)
            if edge_type == 'cites':
                edge_index = dataset.edge_index('paper', 'paper')
                set1 = sample_data['paper']
                set2 = sample_data['paper']
                print(f'Done! [{time.perf_counter() - t:.2f}s]')
            elif edge_type == 'writes':
                edge_index = dataset.edge_index('author', 'paper')
                set1 = sample_data['author']
                set2 = sample_data['paper']
                print(f'Done! [{time.perf_counter() - t:.2f}s]')
            else:
                edge_index = dataset.edge_index('author', 'institution')
                set1 = sample_data['author']
                set2 = sample_data['institution']
                print(f'Done! [{time.perf_counter() - t:.2f}s]')

            row, col = edge_index
            del edge_index
            _len = len(row)
            sampled_edge = []
            t = time.perf_counter()
            print(f'create sampled edge_index_{edge_type}...', end=' ', flush=True)
            for i in tqdm(range(_len)):
                e = [row[i], col[i]]
                if e[0] in set1 and e[1] in set2:
                    sampled_edge.append(e)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
            print(f'num of edge_{edge_type}: {len(sampled_edge)} ')
            del row
            del col

            print(f'save sampled edge_index_{edge_type}...', end=' ', flush=True)
            t = time.perf_counter()
            path = f'{dataset.dir}/process_shang/subgraph/edge_index_{edge_type}.npy'
            np.save(path, sampled_edge)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
            del sampled_edge

    sampled_save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_depth', type=int, default=2,
                        help='How many numbers to sample the graph')
    parser.add_argument('--sample_width', type=int, default=200,
                        help='How many nodes to be sampled per layer per type')
    parser.add_argument('--batch_size', type=int, default=100000,
                        help='Number of papers for sampled')
    parser.add_argument('--batch_ratio', type=float, default=0.005,
                        help='ratio of authors or institutions for sampled')

    args = parser.parse_args()
    ROOT = '/data/shangyihao/'
    dataset = MAG240MDataset(ROOT)
    edge_types = ['cites', 'writes', 'affiliated with']
    _map = create_map(edge_types)

    split_dict = dataset.get_idx_split()
    train_idx = list(split_dict['train'])
    valid_idx = list(split_dict['valid'])
    test_idx = list(split_dict['test'])
    arxiv_idx = train_idx + valid_idx
    arxiv_idx.extend(test_idx)
    arxiv_idx.sort()
    del train_idx
    del valid_idx
    del test_idx

    samp_papers = np.random.choice(arxiv_idx, args.batch_size, replace=False)
    #
    # authors_idx = range(dataset.num_authors)
    # institutions_idx = range(dataset.num_institutions)
    # num_sub_authors = int(dataset.num_authors * args.batch_ratio)


    inp = {'paper': samp_papers}
    sample_subgraph(_map, sampled_num=args.sample_width, sampled_depth=args.sample_depth, inp=inp)



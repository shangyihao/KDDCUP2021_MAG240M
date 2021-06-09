from pyHGT.data import *
from pyHGT.utils import *
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import os.path as osp
import torch
import argparse
import time
import pickle
import numpy as np
import dill
import joblib
from tqdm import tqdm
ROOT = '/data/shangyihao/'

dataset = MAG240MDataset(ROOT)
evaluator = MAG240MEvaluator()

parser = argparse.ArgumentParser(description='Preprocess mag240m graph')

args = parser.parse_args()

output_dir = f'{dataset.dir}/process_shang/subgraph/3/OGB_MAG.pk'  # The address to output the preprocessed graph

subgraph_file = f'{dataset.dir}/process_shang/subsample/subgraph_2_hop_0.01_1'

node_file = osp.join(subgraph_file, 'sampled_idx.pt')
node_dict = torch.load(node_file)
num_nodes = {key: len(node_dict[key].tolist()) for key in node_dict}
print(num_nodes.values())

file = osp.join(subgraph_file, 'split_dict.txt')
with open(file, 'rb') as f:
    split_dict = pickle.load(f)

file = osp.join(subgraph_file, 'edge_index.pt')
edge_index_dict = torch.load(file)

file = osp.join(subgraph_file, 'years.npy')
new_years = np.load(file)

file = osp.join(subgraph_file, 'paper_feat.npy')
paper_feat = np.memmap(file, dtype=np.float, mode='c', shape=(num_nodes['paper'], dataset.num_paper_features))

file = osp.join(subgraph_file, 'labels.npy')
labels = np.load(file)


graph = Graph()
edg = graph.edge_list
years = new_years
del new_years

t = time.perf_counter()
print('seting paper year ...', end=' ', flush=True)
for key in edge_index_dict:
    print(key)
    edges = edge_index_dict[key]
    s_type, r_type, t_type = key[0], key[1], key[2]
    elist = edg[t_type][s_type][r_type]
    rlist = edg[s_type][t_type]['rev_' + r_type]
    for s_id, t_id in edges.t().tolist():
        year = None
        if s_type == 'paper':
            year = years[s_id]
        elif t_type == 'paper':
            year = years[t_id]
        elist[t_id][s_id] = year
        rlist[s_id][t_id] = year
print(f'Done! [{time.perf_counter() - t:.2f}s]')

edg = {}
deg = {key : np.zeros(num_nodes[key]) for key in num_nodes}

t = time.perf_counter()
print('calculating degree ...', end=' ', flush=True)
for k1 in graph.edge_list:
    if k1 not in edg:  # target_type
        edg[k1] = {}
    for k2 in graph.edge_list[k1]:  # source_type
        if k2 not in edg[k1]:
            edg[k1][k2] = {}
        for k3 in graph.edge_list[k1][k2]:  # relation_type
            if k3 not in edg[k1][k2]:
                edg[k1][k2][k3] = {}
            for e1 in graph.edge_list[k1][k2][k3]:  # target_idx
                if len(graph.edge_list[k1][k2][k3][e1]) == 0:
                    continue  # the node no such type edge
                
                edg[k1][k2][k3][e1] = {}
                for e2 in graph.edge_list[k1][k2][k3][e1]:
                    edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]
                deg[k1][e1] += len(edg[k1][k2][k3][e1])
            print(k1, k2, k3, len(edg[k1][k2][k3]))
print(f'Done! [{time.perf_counter() - t:.2f}s]')
graph.edge_list = edg

t = time.perf_counter()
print('generating paper feature ...', end=' ', flush=True)
cv = paper_feat
graph.node_feature['paper'] = np.concatenate((cv, np.log10(deg['paper'].reshape(-1, 1))), axis=-1)
print(f'Done! [{time.perf_counter() - t:.2f}s]')

t = time.perf_counter()
print('generating author feature ...', end=' ', flush=True)
for _type in num_nodes:
    print(_type)
    if _type not in ['paper', 'institution']:
        i = []
        for _rel in graph.edge_list[_type]['paper']:
            for t in graph.edge_list[_type]['paper'][_rel]:
                for s in graph.edge_list[_type]['paper'][_rel][t]:
                    i += [[t, s]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(sp.coo_matrix((v, i), shape=(num_nodes[_type], num_nodes['paper'])))
        out = m.dot(cv)
        graph.node_feature[_type] = np.concatenate((out, np.log10(deg[_type].reshape(-1, 1))), axis=-1)
print(f'Done! [{time.perf_counter() - t:.2f}s]')

t = time.perf_counter()
print('generating institution feature ...', end=' ', flush=True)
cv = graph.node_feature['author'][:, :-1]
i = []
for _rel in graph.edge_list['institution']['author']:
    for j in graph.edge_list['institution']['author'][_rel]:
        for t in graph.edge_list['institution']['author'][_rel][j]:
            i += [[j, t]]
i = np.array(i).T
v = np.ones(i.shape[1])
m = normalize(sp.coo_matrix((v, i), shape=(num_nodes['institution'], num_nodes['author'])))
out = m.dot(cv)
graph.node_feature['institution'] = np.concatenate((out, np.log10(deg['institution'].reshape(-1, 1))), axis=-1)
print(f'Done! [{time.perf_counter() - t:.2f}s]')


train_paper = np.asarray(split_dict['train'])
valid_paper = np.asarray(split_dict['valid'])
test_paper  = np.asarray(split_dict['test'])


graph.y = labels
graph.train_paper = train_paper
graph.valid_paper = valid_paper
graph.test_paper  = test_paper
graph.years       = years
del labels, years

graph.train_mask = np.zeros(num_nodes['paper'], dtype=bool)
graph.train_mask[graph.train_paper] = True

graph.valid_mask = np.zeros(num_nodes['paper'], dtype=bool)
graph.valid_mask[graph.valid_paper] = True

graph.test_mask = np.zeros(num_nodes['paper'],  dtype=bool)
graph.test_mask[graph.test_paper] = True

t = time.perf_counter()
print('saving graph class...', end=' ', flush=True)
joblib.dump(graph, output_dir)
print(f'Done! [{time.perf_counter() - t:.2f}s]')

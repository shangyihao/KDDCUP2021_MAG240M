import numpy as np
import torch
import time
import os
import os.path as osp
import networkx as nx
from tqdm import tqdm
from dataset import MAG240MDataset
from torch_sparse import SparseTensor
from networkx.algorithms.bipartite.generators import complete_bipartite_graph

ratio = 0.1
dataset = MAG240MDataset(2, ratio, 0)
save_root = f'/data/shangyihao/mag240m/newedges{ratio}'
os.mkdir(save_root)

write_edge = dataset.edge_index('author', 'paper')
aff_edge = dataset.edge_index('author', 'institution')
authors, papers = write_edge
num = len(authors)

_map_author = {}  # paper_ids per author
_map_paper = {}  # author_ids per paper
for i in tqdm(range(num)):
    author = authors[i]
    paper = papers[i]
    if author not in _map_author:
        _map_author[author] = []
    _map_author[author].append(paper)
    if paper not in _map_paper:
        _map_paper[paper] = []
    _map_paper[paper].append(author)
del authors, papers

authors, institutions = aff_edge
num = len(authors)
i2a = {}  # author_ids per institution
for i in tqdm(range(num)):
    author = authors[i]
    institution = institutions[i]
    if institution not in i2a:
        i2a[institution] = []
    i2a[institution].append(author)
del authors, institutions

i2p = {}  # paper_ids per institution
for key, value in tqdm(i2a.items()):
    if key not in i2p:
        i2p[key] = []
    for item in value:
        i2p[key] += _map_author[item]  # base on per author in institution, add papers of the author
    i2p[key] = set(i2p[key])

paper2ins = []  # create edge from institution to paper
for key, value in tqdm(i2p.items()):
    g = complete_bipartite_graph([key], value)
    paper2ins += list(g.edges)
paper_ins_edge = np.asarray(paper2ins).T
file = osp.join(save_root, 'institution_paper.npy')

same_author = []  # create paper connection via same author
for key, value in tqdm(_map_author.items()):
    g = nx.complete_graph(value)
    same_author += list(g.edges)
same_author_edge = np.asarray(same_author).T  # turn to edge_index type
print(f'same_author_edges.shape: {same_author_edge.shape}')
file = osp.join(save_root, 'same_author.npy')
np.save(file, same_author_edge)
print('Saved !')


together_write = []  # create author connection via writing together
for key, value in tqdm(_map_paper.items()):
    g = nx.complete_graph(value)
    together_write += list(g.edges)
together_write_edge = np.asarray(together_write).T
print(f'together_write_edges.shape: {together_write_edge.shape}')

# same_institution = []  # create paper connection via same institution
# for key, value in tqdm(i2p.items()):
#     g = nx.complete_graph(value)
#     same_institution += list(g.edges)
# same_institution_edge = np.asarray(same_institution).T
# print(f'same_institution_edges.shape: {same_institution_edge.shape}')
# file = osp.join(save_root, 'same_institution.npy')
# np.save(file, same_institution_edge)
# print('Saved !')

row, col = together_write_edge.tolist()
rows = row + col
cols = col + row
num =len(rows)

a2a = {}  # co_author_ids per author
for i in tqdm(range(num)):
    author1 = rows[i]
    author2 = cols[i]
    if author1 not in a2a:
        a2a[author1] = []
    a2a[author1].append(author2)
del row, col, rows, cols

a_p_co_p = {}  # paper with papers of cooperators per author
for key, value in tqdm(a2a.items()):
    if key not in a_p_co_p:
        a_p_co_p[key] = [_map_author[key], []]  # [paper],[co_paper]
    for item in value:
        a_p_co_p[key][1] += _map_author[item]

paper_co_paper = []  # create connection between paper with papers of cooperators
for key, value in tqdm(a_p_co_p.items()):
    g = complete_bipartite_graph(value[0], value[1])
    paper_co_paper += list(g.edges)
paper_co_paper_edge = np.asarray(paper_co_paper).T
print(f'paper_co_paper_edges.shape: {paper_co_paper_edge.shape}')
file = osp.join(save_root, 'paper_copaper.npy')
np.save(file, paper_co_paper_edge)
print('Saved !')


def prepare_data():
    path = osp.join(save_root, 'full_adj.pt')
    if not osp.exists(path):
        t = time.perf_counter()
        print('Merging adjacency matrices...', end=' ', flush=True)

        edge_index = dataset.edge_index('paper', 'paper')
        row, col = torch.from_numpy(edge_index)
        rows, cols = [row], [col]

        row, col = torch.from_numpy(same_author_edge)
        rows += [row, col]
        cols += [col, row]

        # row, col = torch.from_numpy(same_institution_edge)
        # rows += [row, col]
        # cols += [col, row]

        row, col = torch.from_numpy(paper_co_paper_edge)
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

        N = dataset.num_papers
        perm = (N * row).add_(col).numpy().argsort()
        perm = torch.from_numpy(perm)
        row = row[perm]
        col = col[perm]
        edge_type = torch.cat(edge_types, dim=0)[perm]
        del edge_types
        print(f'edge_type: {edge_type}')

        full_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                  sparse_sizes=(N, N), is_sorted=True)

        torch.save(full_adj_t, path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')


prepare_data()

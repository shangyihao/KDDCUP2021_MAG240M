import os
import os.path as osp
from dataset import MAG240MDataset
import numpy as np
import torch
from tqdm import tqdm

dataset = MAG240MDataset(2, 0.1, 0)
write_edge = dataset.edge_index('author', 'paper')
authors, papers = write_edge
num = len(authors)
_map_author = {}  # paper_ids per author
for i in tqdm(range(num)):
    author = authors[i]
    paper = papers[i]
    if author not in _map_author:
        _map_author[author] = []
    _map_author[author].append(paper)

aff_edge = dataset.edge_index('author', 'institution')
authors, institutions = aff_edge
num = len(authors)
i2a = {}  # author_ids per institution
for i in tqdm(range(num)):
    author = authors[i]
    institution = institutions[i]
    if institution not in i2a:
        i2a[institution] = []
    i2a[institution].append(author)

train_idx = dataset.get_idx_split('train').numpy()
valid_idx = dataset.get_idx_split('valid').numpy()
paper_year = dataset.paper_year
paper_label = dataset.paper_label

cite_edge = dataset.edge_index('paper', 'paper')
rows, cols = cite_edge
num = len(rows)
p2p = {}  # neighbor paper per paper
for i in tqdm(range(num)):
    row = rows[i]
    col = cols[i]
    if row not in p2p:
        p2p[row] = set()
    if col not in p2p:
        p2p[col] = set()
    p2p[row].add(col)
    p2p[col].add(row)

year_count = {}
for idx in tqdm(train_idx):
    year = paper_year[idx]
    if year not in year_count:
        year_count[year] = [0, []]  # [count, idxs in year]
    year_count[year][0] += 1
    year_count[year][1].append(idx)

train_rand_idx = []
for year, [count, idxs] in tqdm(year_count.items()):
    print(f'year {year} has {count} papers')
    chose_idxs = np.random.choice(idxs, int(0.7 * count), replace=False)
    train_rand_idx.append(chose_idxs)
train_rand_idx = np.concatenate(train_rand_idx, axis=0)
valid_rand_idx = np.random.choice(valid_idx, int(0.7 * len(valid_idx)), replace=False)

save_root = '/data/shangyihao/mag240m/new_feat(train_valid)0.1_0.7'
os.makedirs(save_root)
train_index_path = osp.join(save_root, 'train_index.pt')
valid_index_path = osp.join(save_root, 'valid_index.pt')
torch.save(torch.from_numpy(train_rand_idx), train_index_path)
torch.save(torch.from_numpy(valid_rand_idx), valid_index_path)

train_idx = set(train_idx)
valid_idx = set(valid_idx)
train_rand_idx = set(train_rand_idx)
print(f'train_idx size: {len(train_rand_idx)}')
valid_rand_idx = set(valid_rand_idx)
print(f'valid_idx size: {len(valid_rand_idx)}')
prefer_idx = (train_idx - train_rand_idx) | (valid_idx - valid_rand_idx)
print(f'prefer_idx size: {len(prefer_idx)}')

paper_prefer_path = osp.join(save_root, 'paper_prefer.npy')
paper_prefer = np.memmap(paper_prefer_path, dtype=np.float16, mode='w+',
                         shape=(dataset.num_papers, dataset.num_classes))
for idx, paper_ids in tqdm(p2p.items()):
    feat = np.zeros(dataset.num_classes, dtype=int)
    for paper_id in paper_ids:
        if paper_id in prefer_idx:
            loc = int(paper_label[paper_id])
            feat[loc] += 1
    paper_prefer[idx] = feat

# type 1
author_path = osp.join(save_root, 'author_feat.npy')
author_feat = np.memmap(author_path, dtype=np.float16, mode='w+',
                        shape=(dataset.num_authors, dataset.num_classes))
for author_id, paper_ids in tqdm(_map_author.items()):
    feat = np.zeros(dataset.num_classes, dtype=int)
    for paper_id in paper_ids:
        if paper_id in prefer_idx:
            loc = int(paper_label[paper_id])
            feat[loc] += 1
    author_feat[author_id] = feat

# type 2
author_path = osp.join(save_root, 'author_feat.npy')
author_feat = np.memmap(author_path, dtype=np.float16, mode='w+',
                        shape=(dataset.num_authors, dataset.num_classes))
for author_id, paper_ids in tqdm(_map_author.items()):
    for paper_id in paper_ids:
        author_feat[author_id] += paper_prefer[paper_id].copy()

institution_path = osp.join(save_root, 'institution_feat.npy')
institution_feat = np.memmap(institution_path, dtype=np.float16, mode='w+',
                             shape=(dataset.num_institutions, dataset.num_classes))
for institution_id, author_ids in tqdm(i2a.items()):
    for author_id in author_ids:
        institution_feat[institution_id] += author_feat[author_id].copy()

full_feat_path = osp.join(save_root, 'full_feat.npy')
N = (dataset.num_papers + dataset.num_authors + dataset.num_institutions)
node_chunk_size = 100000
dim_chunk_size = 64
# x = np.memmap(full_feat_path, dtype=np.float16, mode='w+',
#               shape=(N, dataset.num_paper_features))
#
# paper_feat = dataset.paper_feat
# for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
#     j = min(i + node_chunk_size, dataset.num_papers)
#     x[i:j] = paper_feat[i:j]
#
# for i in tqdm(range(0, dataset.num_authors, node_chunk_size)):
#     start = i + dataset.num_papers
#     j = min(i + node_chunk_size, dataset.num_authors)
#     end = j + dataset.num_papers
#     x[start:end, :dataset.num_classes] = author_feat[i:j]
#
# start = dataset.num_papers + dataset.num_authors
# x[start:, :dataset.num_classes] = institution_feat[:]

feat_prefer_path = osp.join(save_root, 'feat_prefer.npy')
feat_prefer = np.memmap(feat_prefer_path, dtype=np.float16, mode='w+',
                        shape=(N, dataset.num_classes))
print('copy paper preference feature...')
for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
    j = min(i + node_chunk_size, dataset.num_papers)
    feat_prefer[i:j] = paper_prefer[i:j]
print('copy author preference feature...')
for i in tqdm(range(0, dataset.num_authors, node_chunk_size)):
    start = i + dataset.num_papers
    j = min(i + node_chunk_size, dataset.num_authors)
    end = j + dataset.num_papers
    feat_prefer[start:end] = author_feat[i:j]
print('copy institution preference feature...')
start = dataset.num_papers + dataset.num_authors
feat_prefer[start:] = institution_feat[:]

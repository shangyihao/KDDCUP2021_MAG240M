from ogb.lsc import MAG240MDataset
import numpy as np
import time
from tqdm import tqdm
import os.path as osp
ROOT = '/data/shangyihao/'

dataset = MAG240MDataset(root=ROOT)

# split_file = f'{dataset.dir}/process_shang/split_labels_dict.npy'
# split = np.load(split_file, allow_pickle=True).item()
# train_idx = split['train']
# valid_idx = split['valid']
# test_idx = split['test']
#
# train_set = set()
# valid_set = set()
# test_set = set()
# for idx1 in train_idx:
#     train_set.add(idx1)
# for idx in valid_idx:
#     valid_set.add(idx)
# for idx in test_idx:
#     test_set.add(idx)

split_file = f'{dataset.dir}/process_shang/split_dict.npy'
split = np.load(split_file, allow_pickle=True).item()
train_allpaperidx = split['train']
valid_allpaperidx = split['valid']
test_allpaperidx = split['test']

train_allpaper_set = set()
valid_allpaper_set = set()
test_allpaper_set = set()
for idx1 in train_allpaperidx:
    train_allpaper_set.add(idx1)
for idx in valid_allpaperidx:
    valid_allpaper_set.add(idx)
for idx in test_allpaperidx:
    test_allpaper_set.add(idx)


edge_idx_cites = dataset.edge_index('paper', 'paper')
edge_cite_num = edge_idx_cites[0].size
# edge_file = f'{dataset.dir}/process_shang/split_edge_dict.npy'


# def split():
#     t = time.perf_counter()
#     edge_train = []
#     edge_valid = []
#     edge_test = []
#     for i in tqdm(range(edge_cite_num)):
#         e = [edge_idx_cites[0][i], edge_idx_cites[1][i]]
#         if e[0] in train_set and e[1] in train_set:
#             edge_train.append(e)
#         if e[0] in valid_set and e[1] in valid_set:
#             edge_valid.append(e)
#         if e[0] in test_set and e[1] in test_set:
#             edge_test.append(e)
#     np.savez(edge_file, train=edge_train, valid=edge_valid, test=edge_test)
#     print(f'Done ! : use time {(time.perf_counter() - t):.4f}s')
#
#     print(f'num of edge in trainset: {len(edge_train)}')  # 1378243
#     print(f'num of edge in validset: {len(edge_valid)}')  # 37885
#     print(f'num of edge in testset: {len(edge_test)}')  # 53725
#
#
# split()


def split_allpaper(filename):
    t = time.perf_counter()
    edge_train = []
    edge_valid = []
    edge_test = []
    for i in tqdm(range(edge_cite_num)):
        e = [edge_idx_cites[0][i], edge_idx_cites[1][i]]
        if e[0] in train_allpaper_set and e[1] in train_allpaper_set:
            edge_train.append(e)
        if e[0] in valid_allpaper_set and e[1] in valid_allpaper_set:
            edge_valid.append(e)
        if e[0] in test_allpaper_set and e[1] in test_allpaper_set:
            edge_test.append(e)
    np.savez(filename, train=edge_train, valid=edge_valid, test=edge_test)
    print(f'Done ! : use time {(time.perf_counter() - t):.4f}s')

    print(f'num of edge in trainset: {len(edge_train)}')  # num of edge in trainset: 1037704532
    print(f'num of edge in validset: {len(edge_valid)}')  # num of edge in validset: 1670817
    print(f'num of edge in testset: {len(edge_test)}')  # num of edge in testset: 2087294


split_edge = f'{dataset.dir}/process_shang/split_graph_dict.npz'
split_allpaper(split_edge)
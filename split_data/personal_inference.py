from ogb.lsc import MAG240MDataset
from root import ROOT
import math
import numpy as np
import torch
from torch_sparse import SparseTensor
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import time

dataset = MAG240MDataset(root=ROOT)

edge_idx_cites = dataset.edge_index('paper', 'paper')
edge_cite_num = edge_idx_cites[0].size

split_file = f'{dataset.dir}/process_shang/split_labels_dict.npy'

split = np.load(split_file, allow_pickle=True).item()
train_idx = split['train']
valid_idx = split['valid']
test_idx = split['test']
print(f'num_trainset: {len(train_idx)}')
print(f'num_validset: {len(valid_idx)}')
print(f'num_testset: {len(test_idx)}')
arxiv_idx = train_idx + valid_idx
arxiv_idx.extend(test_idx)
arxiv_idx.sort()
print(f'num_arxivset: {len(arxiv_idx)}')
# paper_label = dataset.paper_label
# paper_year = dataset.paper_year


def static_degree(set_idx, name):
    total_t = time.perf_counter()
    stat_list = []
    mark = 0
    t = time.perf_counter()
    paper_edge = list(set(set_idx).intersection(set(edge_idx_cites[0])))
    paper_edge.sort()
    print(f'交集处理完毕: {(time.perf_counter() - t):.4f}')
    print(f'num of paper with edge: {len(paper_edge)}')
    print(paper_edge[3])
    t = time.perf_counter()
    paper_noedge = list(set(set_idx).difference(set(paper_edge)))
    paper_noedge.sort()
    print(f'差集处理完毕: {(time.perf_counter() - t):.4f}')
    print(f'num of paper without edge: {len(paper_noedge)}')

    count = 0
    t = time.perf_counter()
    for i in paper_edge:  # 遍历每篇有引用的paper
        count += 1
        # t = time.perf_counter()
        degree = 0
        label_count = 0
        flag = False
        for j in range(mark, edge_cite_num):
            if i == edge_idx_cites[0][j]:  # 该paper有去引用其他paper
                degree += 1
                target_node_idx = edge_idx_cites[1][j]
                if not flag:  # 这个paper第一次被遍历到
                    flag = True
                if target_node_idx in arxiv_idx:
                    label_count += 1
            if flag and i != edge_idx_cites[0][j]:  # 与这个paper相关的引用遍历结束
                mark = j
                break
        concat = {'index': i, 'degree': degree, 'ratio': label_count/degree}  # 记录每个idx对应的度数和有label的比重
        stat_list.append(concat)
        if count % 100 == 0:
            print(f'chunk {count / 100} time used: {(time.perf_counter() - t):.4f}s')
            t = time.perf_counter()

    count = 0
    t = time.perf_counter()
    for i in paper_noedge:
        count += 1
        degree = 0
        ratio = 0.
        concat = {'index': i, 'degree': degree, 'ratio': ratio}
        stat_list.append(concat)
        if count % 100 == 0:
            print(f'chunk {count / 100} time used: {(time.perf_counter() - t):.4f}s')
            t = time.perf_counter()

    stat_list.sort(key=lambda s: s['index'])
    print(f'size of stat_list = {len(stat_list)}')

    if name == 'train':
        file = f'{dataset.dir}/process_shang/train_idx_degree.npy'
    elif name == 'valid':
        file = f'{dataset.dir}/process_shang/valid_idx_degree.npy'
    else:
        file = f'{dataset.dir}/process_shang/test_idx_degree.npy'
    np.save(file, stat_list)

    print(f'Done ! : use time {(time.perf_counter() - total_t):.4f}s')


static_degree(valid_idx, 'valid')
static_degree(test_idx, 'test')
static_degree(train_idx, 'train')


# edge_index = torch.from_numpy(edge_idx_cites)
# adj_t = SparseTensor(
#         row=edge_index[0], col=edge_index[1],
#         sparse_sizes=(dataset.num_papers, dataset.num_papers),
#         is_sorted=True)
# adj_t = adj_t.to_torch_sparse_coo_tensor()
# indices = np.array(adj_t._indices())
# print(indices.size)
# print(indices.shape)

# indices_len = int(indices.size/2)

# for i in range(indices_len):
#     if indices[0][i] == 113:
#         print(i)


# def statistic_degree(set_idx, name):
#     stat_list = []
#     mark = 0
#     for i in set_idx:
#         t = time.perf_counter()
#         degree = 0
#         label_count = 0
#         flag = False
#         for j in range(mark, indices_len, 1):
#             if i == indices[0][j]:  # 该paper有去引用其他paper
#                 degree += 1
#                 target_node_idx = indices[1][j]
#                 if not flag:  # 这个paper第一次被遍历到
#                     flag = True
#                 if target_node_idx in arxiv_idx:
#                     label_count += 1
#             if i != indices[0][j] and flag:  # 与这个paper相关的引用遍历结束
#                 mark = j
#                 flag = False
#                 break
#         if degree == 0:
#             ratio = 0
#         else:
#             ratio = label_count/degree
#         concat = {'index': i, 'degree': degree, 'ratio': ratio }  # 记录每个idx对应的度数和有label的比重
#         stat_list.append(concat)
#         print(f'node {i} time used: {(time.perf_counter() - t):.4f}')
#     if name == 'train':
#         file = f'{dataset.dir}/process_shang/train_idx_degree.npy'
#     elif name == 'valid':
#         file = f'{dataset.dir}/process_shang/valid_idx_degree.npy'
#     else:
#         file = f'{dataset.dir}/process_shang/test_idx_degree.npy'
#     np.save(file, stat_list)


# statistic_degree(train_idx, 'train')
# statistic_degree(valid_idx, 'valid')
# statistic_degree(test_idx, 'test')

# train_path = f'{dataset.dir}/process_shang/train_idx_degree.npy'
# valid_path = f'{dataset.dir}/process_shang/valid_idx_degree.npy'
# test_path = f'{dataset.dir}/process_shang/test_idx_degree.npy'
#
# train_list = np.load(train_path, allow_pickle=True).item()
# valid_list = np.load(valid_path, allow_pickle=True).item()
# test_list = np.load(test_path, allow_pickle=True).item()
#
#
# def show_statistic(stat_list: list, name: str):
#     degree = []
#     ratio = []
#     for i in stat_list:
#         degree.append(i['degree'])
#         ratio.append(i['ratio'])
#     degree_max = max(degree)
#     degree_min = min(degree)
#     degree_mean = np.mean(degree)
#     degree_var = np.var(degree)
#     print(f'{name}的度数最大值为{degree_max},最小值为{degree_min}, 均值为{degree_mean}, 方差为{degree_var} ')
#     return degree, ratio
#
#
# train_degree, train_ratio = show_statistic(train_list, 'train_set')
# valid_degree, valid_ratio = show_statistic(valid_list, 'valid_set')
# test_degree, test_ratio = show_statistic(valid_list, 'test_set')
#
#
# def ratio_draw(train_ratio, valid_ratio, test_ratio):
#     plt.figure(dpi=120)
#     sns.set(style='dark')
#     sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
#     sns.kdeplot(train_ratio, kde=True, label="train")
#     sns.kdeplot(valid_ratio, kde=True, label="valid")
#     sns.kdeplot(test_ratio, kde=True, label="test")
#     plt.legend()
#     plt.show()




# for i in train_idx:
#     degree = 0
#     mark = 0
#     label_count = 0
#     flag = False
#     for j in range(mark, len, 1):
#         if i == indices[0][j]:  # 该paper有去引用其他paper
#             degree += 1
#             target_node_idx = indices[1][j]
#             if not flag :  # 这个paper第一次被遍历到
#                 flag = True
#             if target_node_idx in arxiv_idx:
#                 label_count += 1
#         if i != indices[0][j] and flag:  # 与这个paper相关的引用遍历结束
#             mark = j
#             flag = False
#             break


# size = edge_idx_cites.size
# print(edge_idx_cites.shape)
#
# for i in train_idx:
#     temp_edge_idx = edge_idx_cites[i]
#     for edge in temp_edge_idx:


# print(paper_label[valid_idx])
# for i in train_idx:
#     if math.isnan(paper_label[i]):
#         print('训练集中存在non_arXig_paper.')
#         break

# count = 0
# for i in range(dataset.num_papers):
#     if paper_year[i] == 2020:
#         count += 1
#     if i % 500000 == 0:
#         print(f'count = {count}')
#     if i == dataset.num_papers - 1:
#         print(f'num_papers in 2020 is {count}')

# for i in train_idx:
#     if paper_year[i] > 2018:
#         print('训练集中存在年份大于2018年的paper')
#         break

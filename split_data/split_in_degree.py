from ogb.lsc import MAG240MDataset
from root import ROOT
import numpy as np
import time
from tqdm import tqdm
import os.path as osp

dataset = MAG240MDataset(root=ROOT)

edge_idx_cites = dataset.edge_index('paper', 'paper')
edge_cite_num = edge_idx_cites[0].size

split_file = f'{dataset.dir}/process_shang/split_labels_dict.npy'

split = np.load(split_file, allow_pickle=True).item()
train_idx = split['train']
valid_idx = split['valid']
test_idx = split['test']
print(f'num_train_set: {len(train_idx)}')
print(f'num_valid_set: {len(valid_idx)}')
print(f'num_test_set: {len(test_idx)}')
arxiv_idx = train_idx + valid_idx
arxiv_idx.extend(test_idx)
arxiv_idx.sort()
label_idx = set()
for idx in arxiv_idx:
    label_idx.add(idx)
print(f'num_arxiv_set: {len(arxiv_idx)}')


def statistic_degree(set_idx, name):
    total_t = time.perf_counter()
    stat_list = []
    file = f'{dataset.dir}/process_shang/{name}indegree_mp_dict.npy'

    if not osp.exists(file):
        idx = set()
        mp = {}  # 记录每个节点所拥有的边集
        for i in set_idx:
            idx.add(i)
        for i in tqdm(range(edge_cite_num)):  # 遍历每条边
            e = [edge_idx_cites[0][i], edge_idx_cites[1][i]]  # 将每条边记录成[source_idx, target_idx]的形式
            if e[1] in idx:  # 该paper在这个数据集中
                if e[1] not in mp:
                    mp[e[1]] = [e]
                else:
                    mp[e[1]].append(e)
            time.sleep(0)
        print(f'num of paper with cites: {len(mp)}')
        np.save(file, mp)
    else:
        mp = np.load(file, allow_pickle=True).item()

    file = f'{dataset.dir}/process_shang/{name}indegree_degree.npz'

    if not osp.exists(file):
        pbar = tqdm(total=len(set_idx))
        pbar.set_description('Pre-processing paper_dict')
        degree_list = []  # 依次记录数据集中每个节点的度数
        ratio = []  # 依次记录数据集中每个节点周围带有label的节点个数
        for i in set_idx:
            label_count = 0
            edge = mp.setdefault(i, [])
            degree = len(edge)
            for e in edge:
                if e[0] in label_idx:
                    label_count += 1
            if degree != 0:
                label_ratio = label_count / degree
            else:
                label_ratio = 0.
            degree_list.append(degree)
            ratio.append(label_ratio)
            # paper_dict = {'index': i, 'degree': degree, 'ratio': label_ratio}
            # stat_list.append(paper_dict)
            pbar.update(1)
        pbar.close()
        np.savez(file, degree=degree_list, ratio=ratio)
        print(f'Done ! : use time {(time.perf_counter() - total_t):.4f}s')


statistic_degree(train_idx, 'train')  # 有入边节点484500， 用时1834s
statistic_degree(valid_idx, 'valid')  # 有入边节点54068， 用时1966s
statistic_degree(test_idx, 'test')  # 有入边节点56820， 用时1642s





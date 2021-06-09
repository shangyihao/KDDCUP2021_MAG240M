import time
import os.path as osp
import numpy as np
from ogb.lsc import MAG240MDataset
from root import ROOT


def Split(path, dataset):
    if not osp.exists(path):
        year = dataset.paper_year
        train_idx = []
        valid_idx = []
        test_idx = []

        t = time.perf_counter()
        for i in range(dataset.num_papers):
            if year[i] != 2020:
                if year[i] == 2019:
                    test_idx.append(i)
                elif year[i] == 2018:
                    valid_idx.append(i)
                else:
                    train_idx.append(i)

        split = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        print(f'Split Done! [{time.perf_counter() - t:.2f}s]')
        print(f'num_trainset: {len(train_idx)}')
        print(f'num_validset: {len(valid_idx)}')
        print(f'num_testset: {len(test_idx)}')

        np.save(path, split)


def Split_labels(path):
    if not osp.exists(path):
        year = dataset.paper_year
        presplit_train_idx = dataset.get_idx_split('train')
        presplit_valid_idx = dataset.get_idx_split('valid')
        train_idx = []  # year until 2017
        valid_idx = []  # year = 2018
        test_idx = presplit_valid_idx  # year = 2019

        t = time.perf_counter()
        for i in presplit_train_idx:
            if year[i] == 2018:
                valid_idx.append(i)
            else:
                train_idx.append(i)

        split_labels = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        print(f'Split Done! [{time.perf_counter() - t:.2f}s]')
        print(f'num_trainset: {len(train_idx)}')
        print(f'num_validset: {len(valid_idx)}')
        print(f'num_testset: {len(test_idx)}')

        np.save(path, split_labels)


if __name__ == '__main__':
    dataset = MAG240MDataset(root=ROOT)

    split_file = f'{dataset.dir}/process_shang/split_dict.npy'
    Split(split_file, dataset)

    split_labels_file = f'{dataset.dir}/process_shang/split_labels_dict.npy'
    Split_labels(split_labels_file)


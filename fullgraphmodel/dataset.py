from typing import Optional, Union, Dict
import os.path as osp
import torch
import numpy as np


class MAG240MDataset(object):

    version = 1
    ROOT = '/data/tushihao/mag240m/subgraph/'

    __rels__ = {
        ('author', 'paper'): 'writes',
        ('author', 'institution'): 'affiliated_with',
        ('paper', 'paper'): 'cites',
    }

    def __init__(self, hop, sample_ratio, version):
        """
        文件夹格式: subgraph_{hop}_hop_{sample_ratio}_{version}
        hop: 数据集采样的跳数
        sample_ratio: 采样率（当采样率为1时，其他参数无效，使用全部数据集）
        version: 版本（相同hop与sample_ratio下进行多次采样）
        """
        if sample_ratio == 1:
            self.dir = '/data/shangyihao/mag240m_kddcup2021'
        else:
            self.dir = osp.join(MAG240MDataset.ROOT, f'subgraph_{hop}_hops_{sample_ratio}_{version}')
        self.__meta__ = torch.load(osp.join(self.dir, 'meta.pt'))
        self.__split__ = torch.load(osp.join(self.dir, 'split_dict.pt'))

    @property
    def num_papers(self) -> int:
        return self.__meta__['paper']

    @property
    def num_authors(self) -> int:
        return self.__meta__['author']

    @property
    def num_institutions(self) -> int:
        return self.__meta__['institution']

    @property
    def num_paper_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return self.__meta__['num_classes']

    def get_idx_split(
        self, split: Optional[str] = None
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        return self.__split__ if split is None else self.__split__[split]

    @property
    def paper_feat(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_feat.npy')
        return np.memmap(path, dtype=np.float16, mode='c', shape=(self.num_papers, self.num_paper_features))

    @property
    def all_paper_feat(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_feat.npy')
        return np.load(path)

    @property
    def paper_label(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_label.npy')
        return np.load(path, mmap_mode='r')

    @property
    def all_paper_label(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_label.npy')
        return np.load(path)

    @property
    def paper_year(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_year.npy')
        return np.load(path, mmap_mode='r')

    @property
    def all_paper_year(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_year.npy')
        return np.load(path)

    def edge_index(self, id1: str, id2: str,
                   id3: Optional[str] = None) -> np.ndarray:
        src = id1
        rel, dst = (id3, id2) if id3 is None else (id2, id3)
        rel = self.__rels__[(src, dst)] if rel is None else rel
        name = f'{src}___{rel}___{dst}'
        path = osp.join(self.dir, 'processed', name, 'edge_index.npy')
        return np.load(path)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
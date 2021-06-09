import time
import torch
from ogb.lsc import MAG240MDataset
ROOT = '/data/shangyihao/'

dataset = MAG240MDataset(root=ROOT)

x = dataset.paper_feat
idx1 = torch.randint(0, dataset.paper_feat.shape[0], (200, )).long().numpy()
idx2 = torch.randint(0, dataset.paper_feat.shape[0], (200, )).long().numpy()
t = time.perf_counter()
x[idx1]
print(time.perf_counter() - t)
t = time.perf_counter()
x[idx2]
print(time.perf_counter() - t)


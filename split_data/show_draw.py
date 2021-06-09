from ogb.lsc import MAG240MDataset
from root import ROOT
import math
import numpy as np
import torch
from torch_sparse import SparseTensor
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import time

dataset = MAG240MDataset(root=ROOT)
train_path = f'{dataset.dir}/process_shang/trainindegree_degree.npz'
valid_path = f'{dataset.dir}/process_shang/validindegree_degree.npz'
test_path = f'{dataset.dir}/process_shang/testindegree_degree.npz'

train_list = np.load(train_path)
valid_list = np.load(valid_path)
test_list = np.load(test_path)

train_degree, train_ratio = train_list['degree'], train_list['ratio']
valid_degree, valid_ratio = valid_list['degree'], valid_list['ratio']
test_degree, test_ratio = test_list['degree'], test_list['ratio']


def show_statistic(degree, ratio, name: str):
    degree_max = np.max(degree)
    degree_mean = np.mean(degree)
    degree_var = np.var(degree)
    ratio_max = np.max(ratio)
    ratio_mean = np.mean(ratio)
    ratio_var = np.var(ratio)
    print(f'{name}的度数最大值为{degree_max},均值为{degree_mean:.5f},方差为{degree_var:.5f} ')
    print(f'{name}的带标签比例的最大值为{ratio_max*100.:.3f}%, 均值为{ratio_mean*100.:.3f}%, 方差为{ratio_var:.5f} ')


show_statistic(train_degree, train_ratio, 'train_set')
show_statistic(valid_degree, valid_ratio, 'valid_set')
show_statistic(test_degree, test_ratio, 'test_set')


def ratio_draw(ratio1, ratio2, ratio3):

    plt.figure(dpi=120)
    sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
    sns.displot(ratio1, kind='kde', color='r', label='train')
    sns.displot(ratio2, kind='kde', color='b', label='valid')
    sns.displot(ratio3, kind='kde', color='g', label='test')
    plt.title('ratio distribution of sets')
    plt.legend()
    plt.show()
    # file = f'{dataset.dir}/process_shang/ratio.png'
    # plt.savefig(file)


ratio_draw(train_ratio, valid_ratio, test_ratio)


def degree_draw(degree1, degree2, degree3):

    plt.figure(dpi=120)
    sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
    sns.displot(degree1, kind='kde', color='r', label='train')
    sns.displot(degree3, kind='kde', color='g', label='valid')
    sns.displot(degree2, kind='kde', color='b', label='test')
    plt.title('degree distribution of sets')
    plt.legend()
    plt.show()
    # file = f'{dataset.dir}/process_shang/degree.png'
    # plt.savefig(file)


def degree_lower20(degree):
    new = []
    for i in degree:
        if i < 20:
            new.append(i)
    return new


train_degree = degree_lower20(train_degree)
valid_degree = degree_lower20(valid_degree)
test_degree = degree_lower20(test_degree)
degree_draw(train_degree, valid_degree, test_degree)
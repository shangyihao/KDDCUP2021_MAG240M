# Track List

## 1. Relational Sample

## 2. Relational BatchNorm

## 3. Label add to feature

1. Masked Label Prediction
2. 直接把label加进去

## 4. year  add to feature

1. year as position ensemble
2. 直接把year加进去

## 5. feature preprocess

1. metapath2vec
2. PCA 降维
3. 用SGC进行预处理paper结点的feature

## 6. create edge feature,then use personalized MPNN 

## 7. 用了BGRL(bootstrapped graph latents):

BGRL bootstraps the GNN to make a node’s embeddings be predictive of its embeddings from another view, under a target GNN. The target network’s parameters are always set to an exponential moving average (EMA) of the GNN parameters. 、

Paper：

[Bootstrapped Representation Learning on Graphs](http://arxiv.org/abs/2102.06514)

这个是针对unlabeled nodes

## 8. 用EMDE和Cleora结合

这个用的不是GNN-based的模型，有点难以理解我再看看

对于Cleora：用的是马尔科夫转移矩阵

we consider all possible random-walk transitions in a Markov transition matrix. Multiplication of vertex embeddings by this matrix allows to perform all possible random walks in parallel, with one large step

不过由于不用卷积所以训练时间很短：

**Training a single model takes around 42 minutes, and inference takes around 7 minutes**

Paper：

[EMDE: An efficient manifold density estimator for all recommendation systems](https://arxiv.org/abs/2006.01894)

[Cleora: A simple, strong and scalable graph embedding scheme](https://arxiv.org/abs/2102.02302)

## 9. Metapath-based Label Propagation

concatenate all types created feature

- R-GAT emb
- paper-focus-based subgraph embs
- several label feature embs

## 10.  Transfer Learning Strategies

1. 先在训练集训练，验证集上测验训练好参数
2. 再将参数拷贝到同样的另一个模型上，固定所有GNNlayer的参数，根据 5-fold cross validation on validation set调整最后的MLP层的参数

## 11. generate author labels

在pre-process中生成author labels, 

训练

根据trained-obtained author labels和paper-labels再来post-process

从trained-obtained author labels再转回到了soft-paper-labels

最后combine这两种paper-labels
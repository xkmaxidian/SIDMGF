import os
import sys
import torch
import tempfile
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import subprocess

from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from typing import Optional
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score, homogeneity_score


def spatial_construct_graph(adata,
                            radius: Optional[float] = 150.0,
                            k: Optional[int] = 8,
                            include_self: bool = False,
                            metric: str = "euclidean", ):
    if radius is None and k is None:
        raise ValueError("You must provide either `radius` or `k`.")

    coords = adata.obsm['spatial']
    n = coords.shape[0]

    # 拟合邻居搜索器
    if radius is not None:
        nbrs = NearestNeighbors(radius=radius, metric=metric).fit(coords)
        distances, indices = nbrs.radius_neighbors(coords, return_distance=True)
        row = np.repeat(np.arange(n), [len(idxs) for idxs in indices])
        col = np.concatenate(indices) if len(indices) > 0 else np.array([], dtype=int)
    else:
        k_eff = k + (1 if include_self else 0)
        nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric).fit(coords)
        distances, indices = nbrs.kneighbors(coords, return_distance=True)
        row = np.repeat(np.arange(n), indices.shape[1])
        col = indices.reshape(-1)

    if not include_self:
        keep = row != col
        row, col = row[keep], col[keep]

    data = np.ones_like(row, dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float32)
    adj = adj.maximum(adj.T)

    # ensure no self-loop
    adj.setdiag(0)
    adj.eliminate_zeros()

    # 统计信息（边按无向计数）
    num_edges_undirected = adj.nnz // 2
    avg_deg = adj.sum(axis=1).A1.mean()
    print(f"The graph contains {num_edges_undirected} undirected edges, {n} cells.")
    print(f"{avg_deg:.4f} neighbors per cell on average.")

    A_dense = adj.toarray().astype(np.float32)
    graph_nei = torch.from_numpy(A_dense)

    sadj = adj.tocoo().astype(np.float32)
    return sadj, graph_nei


def features_construct_graph(features, k=15, mode="connectivity", metric="cosine"):
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    return fadj


def get_signal(adata,
               prior_file,
               path,
               r_path="C:/Program Files/R/R-4.3.1/bin/x64/Rscript.exe",
               r_script="GSVA_scores.R",
               threads=80):
    if os.path.exists(path + '/pathway_activity.csv'):
        pathway_activity = pd.read_csv(path + '/pathway_activity.csv', index_col=0)
        print('Using existing signal activity file...')
        return pathway_activity

    expr_df = pd.DataFrame(
        adata.X.toarray().T if sp.issparse(adata.X) else adata.X.T,
        index=adata.var_names.astype(str),
        columns=adata.obs_names.astype(str)
    )
    tf_expr = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    expr_df.to_csv(tf_expr.name, sep="\t")
    tf_expr.close()
    tf_out = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    tf_out.close()

    cmd = [
        r_path, r_script,
        "--expr", tf_expr.name,
        "--gmt", prior_file,
        "--out", tf_out.name,
        "--threads", str(threads)
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        os.remove(tf_expr.name)

    pathway_activity = pd.read_csv(tf_out.name, sep="\t", index_col=0)
    os.remove(tf_out.name)
    pathway_activity = pathway_activity.loc[adata.obs_names].astype("float32")
    pathway_activity.to_csv(path + 'pathway_activity.csv')
    return pathway_activity


def normalize_sparse_matrix(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def Hungarian(A):
    _, col_ind = linear_sum_assignment(A)
    # Cost can be found as A[row_ind, col_ind].sum()
    return col_ind


def BestMap(L1, L2):
    L1 = L1.flatten(order='F').astype(float)
    L2 = L2.flatten(order='F').astype(float)
    if L1.size != L2.size:
        sys.exit('size(L1) must == size(L2)')
    Label1 = np.unique(L1)
    nClass1 = Label1.size
    Label2 = np.unique(L2)
    nClass2 = Label2.size
    nClass = max(nClass1, nClass2)

    # For Hungarian - Label2 are Workers, Label1 are Tasks.
    G = np.zeros([nClass, nClass]).astype(float)
    for i in range(0, nClass2):
        for j in range(0, nClass1):
            G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

    c = Hungarian(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(0, nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def performance_eval(truth_label, predict_label):
    acc = accuracy_score(truth_label, predict_label)
    ari = adjusted_rand_score(truth_label, predict_label)
    ami = adjusted_mutual_info_score(truth_label, predict_label)
    nmi = normalized_mutual_info_score(truth_label, predict_label)
    hs = homogeneity_score(truth_label, predict_label)

    return acc, ari, ami, nmi, hs


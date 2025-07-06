import numpy as np
import scipy.spatial
import scipy
import igraph as ig
from sklearn.decomposition import TruncatedSVD

from numba import cuda
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import random, math
from sklearn.neighbors import NearestNeighbors
import torch

# -----------------------GENERAL FUNCTIONS-----------------------

def snn(X, n_neighbors=20, min_weight=1/15, metric='cosine'):
    # graph = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric).fit(X)
    indices = nbrs.kneighbors(X,return_distance=False)
    indices = indices[:, 1:]

    n_samples = indices.shape[0]
    edges = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            edges.append((i,neighbor))
    
    g = ig.Graph(n=n_samples,edges=edges,directed=False)
    weights = np.array(g.similarity_jaccard(pairs=g.get_edgelist()))
    g.es['weight'] = weights
    
    edges_to_delete = [i for i, w in enumerate(weights) if w < min_weight]
    g.delete_edges(edges_to_delete)
    
    return g


def find_clusters(X, 
                n_neighbors=20, 
                min_weight=1/15, 
                metric='cosine',
                res=1.2,
                n_iterations=-1):
    
    G = snn(X, n_neighbors=n_neighbors, min_weight=min_weight, metric=metric)
    partition = G.community_leiden(objective_function='modularity',weights=G.es['weight'],n_iterations=n_iterations,resolution=res)
    labels = np.zeros(X.shape[0])
    for i, cluster in enumerate(partition):
        for element in cluster:
            labels[element] = i + 1
    
    return labels


def leiden_clusters(G, 
                res=1.2,
                n_iterations=-1):
    
    partition = G.community_leiden(objective_function='modularity',weights=G.es['weight'],n_iterations=n_iterations,resolution=res)
    labels = np.zeros(G.vcount())
    for i, cluster in enumerate(partition):
        for element in cluster:
            labels[element] = i + 1
    
    return labels


def construct_sample_clusters(X,
                              filler=-1,
                              reps=100,
                              size=0.8,
                              res=1.2,
                              n_jobs=None,
                              metric='cosine',
                              batch_size=20,
                              **kwargs):
    """
    Creates clusterings based on a subset of the dataset
    """
    
    clusters = []

    if reps is None:
        # if not isinstance(res, (list, tuple, np.ndarray)):
        #     res_list = [res]
        # else:
        #     res_list = res

        # total_tasks = len(res_list)
        # for batch_start in tqdm(range(0, total_tasks, batch_size), desc='Batched Sampling', **kwargs):
        #     batch_end = min(batch_start + batch_size, total_tasks)
        #     batch_res_list = res_list[batch_start:batch_end]

        #     with tqdm_joblib(desc='Constructing samples', total=len(batch_res_list), **kwargs):
        #         parallel = Parallel(n_jobs=n_jobs)
        #         batch_clusters = parallel(
        #             delayed(sample_cluster)(X, k=k, res=res_i, filler=filler, sample=False, metric=metric)
        #             for res_i in batch_res_list
        #         )
        #     clusters.extend(batch_clusters)
        G = snn(X, metric=metric)
        if not isinstance(res, (list, tuple, np.ndarray)):
            res_list = [res]
        else:
            res_list = res

        total_tasks = len(res_list)
        for batch_start in tqdm(range(0, total_tasks, batch_size), desc='Batched Sampling', **kwargs):
            batch_end = min(batch_start + batch_size, total_tasks)
            batch_res_list = res_list[batch_start:batch_end]

            with tqdm_joblib(desc='Constructing samples', total=len(batch_res_list), **kwargs):
                parallel = Parallel(n_jobs=n_jobs)
                batch_clusters = parallel(
                    delayed(leiden_clusters)(G,res=res_i)
                    for res_i in batch_res_list
                )
            clusters.extend(batch_clusters)
    else:
        k = int(X.shape[0] * size)
        total_tasks = reps
        for batch_start in tqdm(range(0, total_tasks, batch_size), desc='Batched Sampling', **kwargs):
            batch_end = min(batch_start + batch_size, total_tasks)
            batch_reps = batch_end - batch_start

            with tqdm_joblib(desc='Constructing samples', total=batch_reps, **kwargs):
                parallel = Parallel(n_jobs=n_jobs)
                batch_clusters = parallel(
                    delayed(sample_cluster)(X, k=k, res=res, filler=filler, metric=metric)
                    for _ in range(batch_reps)
                )
            clusters.extend(batch_clusters)

    return clusters

def sample_cluster(X, k, res=1.2, filler=-1, sample=True, metric='cosine'):
    """
    Sample and cluster data
    """
    if not sample:
        cls = find_clusters(X, res=res, metric=metric)
        return cls
    
    row = np.zeros(X.shape[0])
    row.fill(filler)
    sample = random.sample(range(X.shape[0]), k)
    cls = find_clusters(X[sample], res=res, metric=metric)
    np.put(row, sample, cls)
    return row

def calculate_score(clusters, n, reps, device='cpu'):
    if device == 'gpu':
        if cuda.is_available():
            return calculate_score_gpu(clusters, n, reps)
        else:
            print('GPU is not available, function will be run in CPU')
            return calculate_score_cpu(clusters, n, reps)
    elif device == 'cpu':
        return calculate_score_cpu(clusters, n, reps)
    else:
        raise Exception("Device not recognized. Please choose one of 'cpu' or 'gpu'")

def calculate_score_gpu(clusters, n, reps):
    """
    Score calculation on GPU
    """
    score = np.zeros((n, n), dtype=np.csingle)
    score_device = cuda.to_device(score)

    threadsPerBlock = (16, 16)
    blocksPerGrid_x = math.ceil(n / threadsPerBlock[0])
    blocksPerGrid_y = math.ceil(n / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
    
    for row in clusters:
        x_device = cuda.to_device(row)
        outer_equality_kernel[blocksPerGrid, threadsPerBlock](x_device, score_device)
        del x_device
        cuda.current_context().memory_manager.deallocations.clear()
    
    score = score_device.copy_to_host()
    score = np.where(score.real > 0, percent_match(score, reps), 0)
    
    del score_device
    cuda.current_context().memory_manager.deallocations.clear()
    return score

@cuda.jit
def outer_equality_kernel(x, out):
    """
    GPU kernel score calculation algorithm
    """
    tx, ty = cuda.grid(2)

    if tx < x.shape[0] and ty < x.shape[0]:
        if x[tx] == -1 or x[ty] == -1:
            out[tx, ty] += 1j
        elif x[tx] == x[ty]:
            out[tx, ty] += 1

def calculate_score_cpu(clusters, n, reps, n_jobs=None):
    """
    Calculate score on CPU
    """
    score = np.zeros((n, n), dtype=np.csingle)

    for row in clusters:
        parallel = Parallel(n_jobs=n_jobs)
        parallel(outer_equality(row, idx, score) for idx in range(row.shape[0]))
    
    score = np.where(score.real > 0, percent_match(score, reps), 0)
    return score

@delayed
def outer_equality(x, idx, out):
    """
    CPU score calculation algorithm
    """
    if x[idx] == -1:
        out[:, idx] += 1j
        return
    
    for i in range(x.shape[0]):
        if x[i] == x[idx]:
            out[i, idx] += 1
        elif x[i] == -1:
            out[i, idx] += 1j

# def truncated_svd(X, nPC):
#     # X = preprocess(X)
#     svd = TruncatedSVD(n_components=nPC,algorithm="arpack")
#     # U, S, V = torch.svd_lowrank(A, q=10)  # 10차원 저랭크 근사
#     return svd.fit_transform(X)

def truncated_svd(X_np: np.ndarray, nPC: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    if X_np.dtype != np.float32:
        X_np = X_np.astype(np.float32)
    X_tensor = torch.from_numpy(X_np).to(device)
    U, S, V = torch.svd_lowrank(X_tensor, q=nPC)
    transformed_X = U * S

    result_np = transformed_X.cpu().numpy()
    return result_np

def truncated_pca(X,nPC=30):
    Y = np.transpose(X.T - np.mean(X, axis=1))
    return truncated_svd(Y,nPC=nPC)

# -----------------------CHOOSER FUNCTIONS-----------------------
            
def group_silhouette(sil, labels):
    """
    Computes average per-cluster silhouette score 
    """
    sil_grp = list()
    for cls in set(labels):
        idx = np.where(labels == cls)
        sil_grp.append(np.mean(sil[idx]))
    return sil_grp   

def percent_match(x, reps):
    """
    Percentage of co-clustering
    """
    return np.divide(x.real, (reps - x.imag), where=x.imag!=reps)

# -----------------------MULTIK FUNCTIONS-----------------------

def rPAC(consensus, x1=0.1, x2=0.9):
    """"""
    consensus[consensus == 0] = -1
    consensus = np.ravel(np.tril(consensus))
    consensus = consensus[consensus != 0]
    consensus[consensus == -1] = 0

    cdf = scipy.stats.ecdf(consensus).cdf
    pac = cdf.evaluate(x2) - cdf.evaluate(x1)
    zeros = np.sum(consensus == 0) / consensus.shape[0]
    return pac / (1 - zeros)

@delayed
def calculate_one_minus_rpac(cluster, n, x1, x2, device='gpu'):
    n_k = cluster.shape[0]
    
    consensus = calculate_score(cluster, n, n_k, device=device)

    res = 1 - rPAC(consensus, x1, x2)
    return res
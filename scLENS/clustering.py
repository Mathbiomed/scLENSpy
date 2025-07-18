import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.metrics import silhouette_samples
from resample.bootstrap import confidence_interval
from joblib import Parallel
import contextlib
import io

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .cluster_utils import find_clusters, construct_sample_clusters, calculate_score, \
    group_silhouette, calculate_one_minus_rpac, truncated_svd
from .scLENS import scLENS

from collections import Counter
import random

def chooseR(X,
            reps=100,
            size=0.8,
            resolutions=None,
            device='gpu',
            n_jobs=None,
            silent=False,
            batch_size = 20,
            metric='cosine'):
    """
    Chooses the best resolution from a set of possible resolutions
    by repeatedly subsampling and clustering the data,
    calculating the silhouette score for each clustering,
    and selecting the lowest resolution whose median score passes a threshold

    Parameters
    ----------
    X: np.ndarray
        Data to be clustered
    reps: int
        Number of sampling repetitions for each resolution
    size: float
        Portion of data to be subsampled. Must be between 0 and 1
    resolutions: list of float
        Possible resolutions to choose from
    device: One of ['cpu', 'gpu']
        Device to run the scoring on. 'cpu' will run scoring on CPU with n_jobs parallel jobs
    n_jobs: int or None
        joblib n_jobs; Number of CPU jobs to run in parallel

    Returns
    -------
    float
        The chosen best resolution
    """
    if resolutions is None:
        # resolutions = [0.3, 0.5, 0.8, 1, 1.2, 1.6, 2, 4, 6, 8]
        resolutions = np.arange(0.05, 2.05, 0.05)
    resolutions = set(resolutions)
    stats = list()
    for res in tqdm(resolutions,
                    desc='ChooseR', 
                    total=len(resolutions), 
                    disable=silent):
        stats_row = [res]
        cls = find_clusters(X, res=res,metric=metric)
        stats_row.append(len(np.unique(cls)))
        
        clusters = construct_sample_clusters(X, 
                                            reps=reps, 
                                            size=size, 
                                            res=res, 
                                            n_jobs=n_jobs,
                                            metric=metric,
                                            batch_size=batch_size,
                                            disable=True)
        score = calculate_score(clusters, X.shape[0], reps, device=device)
        
        score = 1 - score
        np.fill_diagonal(score, 0)

        # sil = silhouette_samples(score, cls, metric='precomputed')
        try:
            sil = silhouette_samples(score, cls, metric='precomputed')
        except:
            sil = np.zeros(X.shape[0])
            
        sil_grp = group_silhouette(sil, cls)

        stats_row.append(confidence_interval(np.median, sil_grp)[0])
        stats_row.append(np.median(sil_grp))

        stats.append(stats_row)
    
    stats = pd.DataFrame(stats, columns=['res', 'n_clusters', 'low_med', 'med']).sort_values(by=['n_clusters'], ascending=False)
    threshold = max(stats['low_med'])
    filtered_stats = stats[stats['med'] >= threshold]

    return filtered_stats

# X = sclens._raw
# resolutions=None;
# reps=100;
# size=0.8;
# x1=0.1; x2=0.9;
# metric='cosine';
# reduce_func=None;
# nPC=None;
# device='gpu';
# n_jobs=None;
# batch_size=20;
# silent=False;
# plot_flag=False;
def multiK(X,
           resolutions=None,
           reps=100,
           size=0.8,
           x1=0.1, x2=0.9,
           metric='cosine',
           reduce_func=None,
           nPC=None,
           device='gpu',
           n_jobs=None,
           # old_preprocessing=False,
           batch_size=20,
           silent=False,
           plot_flag=False,
           **kwargs):
    """
    Predicts the optimal number of clusters k by 
    testing a range of different resolution parameters,
    and scoring the results based on the observed frequency 
    and rPAC for each k
    
    Parameters
    ----------
    X:  np.ndarray
        Data to be clustered
    resolutions: list of float
        Range of resolutions to be tested
    reps: int
        Number of sampling repetitions for each resolution
    size: float
        Portion of data to be subsampled. Must be between 0 and 1
    x1: float
        Argument for evaluating rPAC. Must be between 0 and 1
    x2: float
        Argument for evaluating rPAC. Must be between 0 and 1
    metric: function or None,
        **
    device: One of ['cpu', 'gpu']
        Device to run the scoring on. 'cpu' will run scoring on CPU with n_jobs parallel jobs
    n_jobs: int or None
        joblib n_jobs; Number of CPU jobs to run in parallel
    
    Returns
    -------
    float
        The chosen best resolution
    """
    if reduce_func is not None and nPC is None:
        raise ValueError('nPC must be specified when using a custom reduce_func')
    if nPC is not None and reduce_func is None:
        raise ValueError('reduce_func must be specified when using a custom nPC')
    
    if resolutions is None:
        resolutions = np.arange(0.05, 2.05, 0.05)
    else:
        resolutions = np.array(resolutions)
    
    n = len(resolutions) * reps
    clusters = np.zeros((n, X.shape[0]))
    ks = np.zeros(n)
    f = io.StringIO()
    for i in tqdm(range(reps)):
        offset = i * len(resolutions)
        k = int(X.shape[0] * size)
        sample = random.sample(range(X.shape[0]), k)

        X_sample = X[sample].copy()
        if reduce_func is not None:
            X_sample = reduce_func(X_sample, nPC)
        else:
            if nPC is None:
                scl = scLENS(device=torch.device('cuda') if device == 'gpu' else torch.device('cpu'))
                scl.preprocess(X_sample,verb=False)
                X_sample = scl.fit_transform()
                nPC = X_sample.shape[1]
            else:
                with contextlib.redirect_stdout(f):
                    scl.preprocess(X_sample,verb=False)
                X_sample = truncated_svd(scl.X, nPC)
                
        sample_cls = construct_sample_clusters(X_sample,
                                               reps=None, 
                                               res=resolutions, 
                                               n_jobs=n_jobs,
                                               metric=metric,
                                               batch_size=batch_size,
                                               disable=True)

        full_cls = np.zeros((len(resolutions), X.shape[0])) - 1
        full_cls[:, sample] = sample_cls
        
        clusters[offset:offset+len(resolutions)] = full_cls
        ks[offset:offset+len(resolutions)] = [len(np.unique(cls)) - 1 for cls in full_cls] # accomodate for label of dropped data
    
    k_runs = [x[1] for x in sorted(Counter(ks).items())]
    k_unique = np.unique(ks)
    
    parallel = Parallel(n_jobs=n_jobs)
    one_minus_rpac = parallel(calculate_one_minus_rpac(clusters[ks==k], 
                                                        X.shape[0],
                                                        x1,
                                                        x2,
                                                        device=device)
                                                        for k in k_unique)

    points = np.array(list(zip(one_minus_rpac, k_runs)))

    if len(points) < 3:
        hull_points = points
        hull_k = k_unique
    else:
        chull = scipy.spatial.ConvexHull(points)
        hull_points = points[chull.vertices]
        hull_k = k_unique[chull.vertices]
    best_rpac = np.argmax(hull_points[:, 0])
    best_run = np.argmax(hull_points[:, 1])

    if best_rpac == best_run:
        opt_points = np.array([hull_points[best_rpac]])
        opt_k = np.array([hull_k[best_rpac]])
    elif best_rpac < best_run:
        opt_points = hull_points[best_rpac:best_run + 1]
        opt_k = hull_k[best_rpac:best_run + 1]
    else:
        opt_points = np.concatenate([hull_points[:best_run + 1], hull_points[best_rpac:]])
        opt_k = np.concatenate([hull_k[:best_run + 1], hull_k[best_rpac:]])

    if plot_flag:
        plt.plot(points[:, 0], points[:, 1], 'ko')
        plt.plot(opt_points[:, 0], opt_points[:, 1], 'ro')
        for i, k in enumerate(opt_k):
            plt.annotate(str(k), (opt_points[i, 0], opt_points[i, 1]))
        plt.xlabel("1-rPAC")
        plt.ylabel("Number of clustering runs")
        plt.show()

    result = list()
    for k in opt_k:
        idx = np.nonzero(ks == k)[0] // reps
        res = Counter(list(resolutions[idx])).most_common(1)[0][0]
        result.append(res)
    
    print(f'Optimal resolutions: {result}')
    return result

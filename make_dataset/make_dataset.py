import numpy as np
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

#from utils import *

def _learn_metric(X, Y, method):
    if method == 'euclidean':
        M = np.eye(X.shape[1])
    elif method == 'kissme':
        num = len(Y)
        X1, X2 = np.meshgrid(np.arange(0, num), np.arange(0, num))
        X1, X2 = X1[X1 < X2], X2[X1 < X2]
        matches = (Y[X1] == Y[X2])
        num_matches = matches.sum()
        num_non_matches = len(matches) - num_matches
        idxa = X1[matches]
        idxb = X2[matches]
        S = X[idxa] - X[idxb]
        C1 = S.transpose().dot(S) / num_matches
        p = np.random.choice(num_non_matches, num_matches, replace=False)
        idxa = X1[matches == False]
        idxb = X2[matches == False]
        idxa = idxa[p]
        idxb = idxb[p]
        S = X[idxa] - X[idxb]
        C0 = S.transpose().dot(S) / num_matches
        M = np.linalg.inv(C1) - np.linalg.inv(C0)
    return M

def _eval_cmc(PX, PY, GX, GY, M):
    #D = pairwise_distances(GX, PX, metric='mahalanobis', VI=M, n_jobs=-2)
    D = pairwise_distances(GX, PX, metric='euclidean', n_jobs=-2)
    print "D is a",np.shape(D)
    C = cmc(D, GY, PY)
    return C
def _cmc_core(D, G, P):
    m, n = D.shape
    order = np.argsort(D, axis=0)
    match = (G[order] == P)
    return (match.sum(axis=1) * 1.0 / n).cumsum()


def cmc(distmat, glabels=None, plabels=None, ds=None, repeat=None):
    """Compute the Cumulative Match Characteristic (CMC)
    This function assumes that gallery labels have no duplication. If there are
    duplications, random downsampling will be performed on gallery labels, and
    the computation will be repeated to get an average result.
    Parameters
    ----------
    distmat : numpy.ndarray
        The distance matrix. ``distmat[i, j]`` is the distance between i-th
        gallery sample and j-th probe sample.
    glabels : numpy.ndarray or None, optional
    plabels : numpy.ndarray or None, optional
        If None, then gallery and probe labels are assumed to have no
        duplications. Otherwise, they represent the vector of gallery and probe
        labels. Default is None.
    ds : int or None, optional
        If None, then no downsampling on gallery labels will be performed.
        Otherwise, it represents the number of gallery labels to be randomly
        selected. Default is None.
    repeat : int or None, optional
        If None, then the function will repeat the computation for 100 times
        when downsampling is performed. Otherwise, it specifies the number of
        repetition. Default is None.
    Returns
    -------
    out : numpy.ndarray
        The rank-1 to rank-m accuracy, where m is the number of (downsampled)
        gallery labels.
    """
    m, n = distmat.shape
    print "m =",m
    print "n=",n
    if glabels is None and plabels is None:
        glabels = np.arange(0, m)
        plabels = np.arange(0, n)
        
    print "glabels=",np.shape(glabels)
    
    if isinstance(glabels, list):
        glabels = np.asarray(glabels)
    if isinstance(plabels, list):
        plabels = np.asarray(plabels)
        
    print "glabels=",np.shape(glabels)
    
    ug = np.unique(glabels)
    print (ug)
    if ds is None:
        ds = ug.size
    if repeat is None:
        if ds == ug.size and ug.size == len(glabels):
            repeat = 1
        else:
            repeat = 100
    print"finish set"

    ret = 0
    for __ in xrange(repeat):
        # Randomly select gallery labels
        G = np.random.choice(ug, ds, replace=False)
        
        
        # Select corresponding probe samples
        p_inds = [i for i in xrange(len(plabels)) if plabels[i] in G]
        
        P = plabels[p_inds]
        
        # Randomly select one gallery sample per label selected
        D = np.zeros((ds, P.size))
        
        for i, g in enumerate(G):
            samples = np.where(glabels == g)[0]
            j = np.random.choice(samples)
            D[i, :] = distmat[j, p_inds]
        # Compute CMC
        
        ret += _cmc_core(D, G, P)
        print "repeat"
    return ret / repeat



PX = np.load("/home/nikoong/Algorithm_test/test_temp/841/camb_feat.npy")
PY = np.load("/home/nikoong/Algorithm_test/test_temp/841/probe_id.npy")
GX = np.load("/home/nikoong/Algorithm_test/test_temp/841/cama_feat.npy")
GY = np.load("/home/nikoong/Algorithm_test/test_temp/841/gallery_id.npy")
print "set data"
M = _learn_metric(PX, PY, 'euclidean')
C = _eval_cmc(PX, PY, GX, GY, M)
print C


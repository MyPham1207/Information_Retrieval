from featureExtractor import FeatureExtractor
import numpy as np
from scipy.spatial.distance import cdist


def cosine_distance(query, doc):
    query_norm = np.sqrt(np.sum(query**2))
    doc_norm = np.sqrt(np.sum(doc**2))
    cosine = np.sum(doc*query)/(query_norm*doc_norm + np.finfo(float).eps)
    return cosine

def search(codeword, pqcode, query):
    M, Ks, Ds = codeword.shape

    dist_table = np.empty((M, Ks), np.float32)

    for m in range(M):
        query_sub = query[m * Ds: (m + 1) * Ds]
        dist_table[m, :] = cdist([query_sub], codeword[m], 'cosine')[0]

    dist = np.sum(dist_table[range(M), pqcode], axis=1)

    return dist

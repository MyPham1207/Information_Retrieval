from featureExtractor import FeatureExtractor
import numpy as np

def cosine_distance(query, doc):
    query_norm = np.sqrt(np.sum(query**2))
    doc_norm = np.sqrt(np.sum(doc**2))
    cosine = np.sum(doc*query)/(query_norm*doc_norm + np.finfo(float).eps)
    return cosine

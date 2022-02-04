# source code tham khảo từ [Paper Explained] Product Quantization for Approximate Nearest Neighbor Search, Phan Huy Hoàng
# https://viblo.asia/p/paper-explained-product-quantization-for-approximate-nearest-neighbor-search-yMnKMA6gK7P
from featureExtractor import FeatureExtractor
import os
import numpy as np
import pickle
from scipy.cluster.vq import vq, kmeans2

def train(vec, M, Ks=256):
    '''
    :param M: số lượng sub-vectors của từng vector
    :param Ks: số cluster áp dụng trên từng tập sub-vectors
    '''
    Ds = int(vec.shape[1] / M)  # số chiều 1 sub-vector
    # tạo M codebooks
    # mỗi codebook gồm Ks codewords
    codeword = np.empty((M, Ks, Ds), np.float32)

    for m in range(M):
        vec_sub = vec[:, m * Ds: (m + 1) * Ds]
        # thực hiện k-means trên từng tập sub-vector thứ m
        centroids, labels = kmeans2(vec_sub, Ks)
        # centroids: (Ks x Ds)
        # labels: vec.shape[0]
        codeword[m] = centroids

    return codeword


def encode(codeword, vec):
    # M sub-vectors
    # Ks clusters ứng với từng tập sub-vector 
    # Ds: số chiều 1 sub-vector
    M, Ks, Ds = codeword.shape

    # tạo pq-code cho n samples (với n = vec.shape[0])
    # mỗi pq-code gồm M giá trị
    pqcode = np.empty((vec.shape[0], M), np.uint8)

    for m in range(M):
        vec_sub = vec[:, m * Ds: (m + 1) * Ds]
        # codes: 1 mảng gồm n phần tử (n = vec.shape[0]), lưu giữ cluster index gần nhất của sub-vector thứ m của từng vector
        # distances: 1 mảng gồm n phần từ (n = vec.shape[0]), lưu giữ khoảng cách giữa sub-vector thứ m của từng vector với centroid gần nhất
        codes, distances = vq(vec_sub, codeword[m])
        # codes: vec.shape[0]
        # distances: vec.shape[0]
        pqcode[:, m] = codes

    return pqcode



if __name__ == '__main__':
    N, D = 6337, 2048
    vec = np.load('./static/featureVector/Feature_Vector.npy', allow_pickle=True)
    M = 8  # chia query-vector thành thành M sub-vector
    codeword = train(vec, M)
    pqcode = encode(codeword, vec)  # tạo pqcode lấy tập dữ liệu training
    np.save('./static/featureVector/codeword.npy', codeword, allow_pickle=True)
    np.save('./static/featureVector/pqcode.npy', pqcode, allow_pickle=True)

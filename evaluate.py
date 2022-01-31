import os
import numpy as np
from featureExtractor import FeatureExtractor
from distance import cosine_distance, search


fe = FeatureExtractor()
featuresVec = []
img_paths = []

featuresVec = np.load('./static/featureVector/Feature_Vector.npy', allow_pickle=True)
img_paths = np.load('./static/featureVector/Name_Vector.npy', allow_pickle=True)
codeword = np.load('./static/featureVector/codeword.npy', allow_pickle=True)
pqcode = np.load('./static/featureVector/pqcode.npy', allow_pickle=True)

def get_sort(path_query):
    query = fe.get_feature(path_query)
    dists = []
    for fv in featuresVec:
        cosine = cosine_distance(query, fv)
        dists.append(cosine)
    ids = np.argsort(dists)
    ids = np.flip(ids)
    return ids

def get_sort_pq(path_query):
    query = fe.get_feature(path_query)
    dists = search(codeword, pqcode, query)
    ids = np.argsort(dists)
    return ids

def get_junk(path_junk):  
  junk = open(path_junk)
  junk_list=[]
  while True:
    line = junk.readline()
    if not line:
      break
    junk_list.append(line.replace('\n',''))
  return junk_list

def get_positive(path_positive):
  positive = open(path_positive)
  positive_list=[]
  while True:
    line = positive.readline()
    if not line:
      break
    positive_list.append(line.replace('\n',''))
  return positive_list

def get_result(sort, k):
    result = []
    for i in range(k):
        rs = img_paths[sort[i]].replace('.jpg','')
        rs = rs.split('/')[4]
        result.append(rs)
    return result

def get_ap(result, junk_list, positive_list):
    sum = 0
    ts = 0
    ms = 0
    count = 0
    for i in range(len(result)):
        if result[i] not in junk_list:
            count +=1
            if result[i] in positive_list:
                ts += 1
                ms += 1
            else:
                ms +=1
            t = ts/ms
            sum +=t
    ap = sum/count
    return ap

def evaluation():
    path_query = 'static/query'
    path_groundtruth = 'static/ground truth'
    groundtruth = os.listdir(path_groundtruth)
    count = 0
    sum = 0
    sum_pq = 0
    for i in range(len(groundtruth)):
        if 'query' in groundtruth[i]:
            count +=1
            domain = groundtruth[i].split('_')[0] +'_'+ groundtruth[i].split('_')[1]
            junk = os.path.join(path_groundtruth, domain) + '_junk.txt'
            junk = get_junk(junk)
            positive = os.path.join(path_groundtruth, domain) + '_ok.txt'
            positive = get_positive(positive)
            f = open(os.path.join(path_groundtruth, groundtruth[i]))
            data = f.read().split(' ')
            query = data[0]
            query = os.path.join(path_query, query) + '.jpg'
            sort = get_sort(query)
            sort_pq = get_sort_pq(query)
            result = get_result(sort, len(positive))
            result_pq = get_result(sort_pq, len(positive))
            ap = get_ap(result, junk, positive)
            ap_pq = get_ap(result_pq, junk, positive)
            sum += ap
            sum_pq += ap_pq
    map = sum/count
    map_pq = sum_pq/count
    return map, map_pq

if __name__ == '__main__':
    map, map_pq = evaluation()
    print(map)
    print(map_pq)
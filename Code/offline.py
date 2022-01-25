from featureExtractor import FeatureExtractor
import os
import numpy as np
import pickle

featuresVec = FeatureExtractor()
vector_feature = []
vector_name = []
path = 'F:/Information Retrieval/Project/Data/Data/paris/'
folder = os.listdir(path)

for i in range(len(folder)):
  folderPath = os.path.join(path, folder[i])
  imageList = os.listdir(folderPath)

  for j in range(len(imageList)):
    imgPath = os.path.join(folderPath, imageList[j])
    print('Processing: ', imageList[j])
    feature = featuresVec.get_feature(imgPath)
    vector_feature.append(feature)
    vector_name.append(imageList[j])

fv = open('F:/Information Retrieval/Project/Data/featureVector/' + 'Feature_Vector.py', 'wb')
pickle.dump(vectors, fv)
fv.close()
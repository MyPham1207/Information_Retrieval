from featureExtractor import FeatureExtractor
import os
import numpy as np
import pickle

featuresVec = FeatureExtractor()
vector_feature = []
vector_name = []
path = 'static/Data/paris/'
folder = os.listdir(path)

for i in range(len(folder)):
  folderPath = os.path.join(path, folder[i])
  imageList = os.listdir(folderPath)

  for j in range(len(imageList)):
    imgPath = os.path.join(folderPath, imageList[j])
    print('Processing: ', imageList[j])
    feature = featuresVec.get_feature(imgPath)
    vector_feature.append(feature)
    vector_name.append('static/Data/paris/{}/{}'.format(folder[i], imageList[j]))

vector_feature = np.array(vector_feature)
np.save('./static/featureVector/Feature_Vector.npy', vector_feature, allow_pickle=True)

vector_name = np.array(vector_name)
np.save('./static/featureVector/Name_Vector.npy', vector_name, allow_pickle=True)
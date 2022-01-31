import imp
import os
import numpy as np 

path = 'static/featureVector/Feature_Vector.npy'
fv = np.load(path, allow_pickle=True)
print(fv.shape)
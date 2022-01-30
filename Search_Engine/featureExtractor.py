import cv2
import tensorflow.keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input

class FeatureExtractor:
    def __init__(self):
        base_model = ResNet101(weights='imagenet', include_top=False)
        x = base_model.output
        x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
        self.model = Model(inputs = base_model.input, outputs= x)

    def get_feature(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis = 0)
        img = preprocess_input(img)

        vector = self.model.predict(img)[0]

        return vector / np.linalg.norm(vector)
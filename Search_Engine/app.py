from flask import Flask, request, render_template
from PIL import Image
from featureExtractor import FeatureExtractor
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
from distance import cosine_distance
import time

app = Flask(__name__)

# Read img feature
fe = FeatureExtractor()
featuresVec = []
img_paths = []

featuresVec = np.load('./static/featureVector/Feature_Vector.npy', allow_pickle=True)
img_paths = np.load('./static/featureVector/Name_Vector.npy', allow_pickle=True)

@app.route("/", methods=["GET", "POST"])
def page():
    if request.method == "POST":
        file = request.files["query_img"]

        img = Image.open(file.stream)
        img = img.convert("RGB")
        uploaded_img_path = "static/uploaded/"+ file.filename
        img.save(uploaded_img_path)

        start = time.time()
        print(start)

        query = fe.get_feature(uploaded_img_path)
        #dists = np.linalg.norm(featuresVec - query, axis=1)
        dists = []
        for fv in featuresVec:
            cosine = cosine_distance(query, fv)
            dists.append(cosine)
        ids = np.argsort(dists)
        ids = np.flip(ids)[:30]
        scores = [(dists[id], img_paths[id]) for id in ids]

        end = time.time()
        print(end)

        print(scores)
        print(end-start)

        return render_template("test.html", query_path=uploaded_img_path, scores=scores)

    else:
        return render_template("test.html")
    
    
app.run()
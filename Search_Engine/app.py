from flask import Flask, request, render_template
from PIL import Image
from featureExtractor import FeatureExtractor
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
from distance import search
import time

app = Flask(__name__)

# Read img feature
fe = FeatureExtractor()
featuresVec = []
img_paths = []

codeword = np.load('./static/featureVector/codeword.npy', allow_pickle=True)
pqcode = np.load('./static/featureVector/pqcode.npy', allow_pickle=True)
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
        dists = search(codeword, pqcode, query)
        ids = np.argsort(dists)[:30]
        scores = [(dists[id], img_paths[id]) for id in ids]

        end = time.time()
        print(end)

        print(scores)
        print(end-start)

        return render_template("page.html", query_path=uploaded_img_path, scores=scores)

    else:
        return render_template("page.html")
    
    
app.run()
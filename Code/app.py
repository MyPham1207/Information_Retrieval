from flask import Flask, request, render_template
from PIL import Image
from featureExtractor import FeatureExtractor
from datetime import datetime
from pathlib import Path
import numpy as np

app = Flask(__name__)

# Read img feature
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path(".featureVector/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
feature = np.array(features)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["query_img"]

        img = Image.open(file.stream)
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        query = fe.extract(img)
        dists = np.linalg.norm(feature - query, axis=1)
        ids = np.argsort(dists)[:30]
        scores = [(dists[id], img_paths[id]) for id in ids]

        print(scores)

        return render_template("page.html", query_path=uploaded_img_path, scores=scores)
    else:
        return render_template("page.html")
    
app.run()
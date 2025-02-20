from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

app = Flask(__name__)

def cluster_image_rgb_values(image, n_colors=10):
    """Cluster image RGB values using KMeans."""
    image = image.convert("RGB")
    image_array = np.array(image).reshape(-1, 3)  # Flatten to (num_pixels, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=1, n_init=10)
    kmeans.fit(image_array)
    clusters = kmeans.cluster_centers_.astype(int)
    return clusters

@app.route("/")
def home():
    """Serve the index.html file."""
    return render_template("index.html")

@app.route("/process_image", methods=["POST"])
def process_image():
    """Process the uploaded image and return color clusters."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    n_colors = int(request.form.get("n_colors", 10))

    try:
        image = Image.open(file)
        color_clusters = cluster_image_rgb_values(image, n_colors=n_colors)
        return jsonify({"colors": color_clusters.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
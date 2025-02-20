import streamlit as st
import numpy as np
import pandas as pd
import PIL
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def cluster_image_rgb_values(
    image: PIL.Image,
    n_colors: int = 10,
    n_init: int = 10,
    random_state: int = 1,
):
    # Convert the image to RGB
    image = image.convert("RGB")
    image_array = np.array(image).reshape(-1, 3)  # Flatten to (num_pixels, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=1, n_init=10)
    kmeans.fit(image_array)

    # Get the clustered colors
    clusters = kmeans.cluster_centers_.astype(int)

    return clusters


# Streamlit UI
st.title("Image RGB Palette App")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    n_colors = st.slider("Number of Colors", 2, 20, 10)

    if st.button("Extract Colors"):
        color_clusters = cluster_image_rgb_values(image, n_colors=n_colors)
        color_clusters_norm = color_clusters / 256

        # Show colors
        st.write("Extracted Colors:")
        color_df = pd.DataFrame(color_clusters, columns=["R", "G", "B"])
        st.dataframe(color_df)

        # Plot colors
        fig, ax = plt.subplots(figsize=(8, 2))
        for i, color in enumerate(color_clusters):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color) / 255))
        ax.set_xlim(0, len(color_clusters))
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
        
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
    
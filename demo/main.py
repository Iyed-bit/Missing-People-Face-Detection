import os
import json
import streamlit as st
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# Cache model to avoid reloading it each time
@st.cache_resource
def app_model():
    app = FaceAnalysis()
    app.prepare(ctx_id=-1)
    return app

# Load stored embeddings and person information
@st.cache_resource
def load_stored_embeddings(directory="../subscribed-people/"):
    embeddings = {}
    app = app_model()

    for folder_name in os.listdir(directory):
        person_folder_path = os.path.join(directory, folder_name)

        if os.path.isdir(person_folder_path):
            for filename in os.listdir(person_folder_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif')):
                    filepath = os.path.join(person_folder_path, filename)
                    image = np.asarray(Image.open(filepath))
                    faces = app.get(image)

                    if faces:
                        embedding = faces[0]['embedding'] / norm(faces[0]['embedding'])
                        info = {}
                        info_path = os.path.join(person_folder_path, "info.json")
                        if os.path.exists(info_path):
                            with open(info_path, "r") as f:
                                info = json.load(f)

                        embeddings[folder_name] = (embedding, info)
                    break
    return embeddings

# Find best match based on cosine similarity
def find_best_match(uploaded_embedding, stored_embeddings, threshold=0.45):
    best_match, best_score, best_info = None, 0, None

    for folder_name, (stored_embedding, info) in stored_embeddings.items():
        score = np.dot(uploaded_embedding, stored_embedding)
        if score > best_score and score >= threshold:
            best_match, best_score, best_info = folder_name, score, info

    return best_match, best_score, best_info

# Streamlit UI
st.title("Face Recognition App")

# Load stored embeddings
stored_embeddings = load_stored_embeddings()

# File uploader
uploaded_file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg", "jfif", "webp"])

if uploaded_file:
    st.image(uploaded_file)

# Process uploaded file
                st.metric("Similarity Score", f"{score:.2f}")

                # Display matched image
                matched_image_path = os.path.join("../subscribed-people", best_match)
                for file in os.listdir(matched_image_path):
                    if file.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif')):
                        st.image(os.path.join(matched_image_path, file), caption="Matched Image")
                        break
            else:
                st.warning("No matches found with similarity score above 50%.")
        else:
            st.warning("No face detected in the uploaded image.")

import os
import json
import streamlit as st
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from model import process_image_with_collection

@st.cache_resource
def app_model():
    app = FaceAnalysis()
    app.prepare(ctx_id=-1)
    return app

@st.cache_resource
def load_stored_images(directory="../subscribed-people/"):
    embeddings = {}
    app = app_model()
    
    # Iterate through each folder (representing a person)
    for person_folder in os.listdir(directory):
        person_folder_path = os.path.join(directory, person_folder)
        
        if os.path.isdir(person_folder_path):
            # Load all images within this folder
            for filename in os.listdir(person_folder_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif')):
                    filepath = os.path.join(person_folder_path, filename)
                    image = np.asarray(Image.open(filepath))
                    
                    embedding = app.get(image)[0]['embedding']
                    embeddings[filename] = embedding  
    
    return embeddings

# Load person metadata (name, age, etc...)
def load_person_info(person_folder):
    """
    Loads the information of a person from the info.json file in their folder.

    Args:
        person_folder: The folder where the person's images and info.json are stored.

    Returns:
        dict: The data from the info.json file or None if not found.
    """
    # Path to the person's info.json file
    info_file_path = os.path.join(person_folder, "info.json")
    
    if os.path.exists(info_file_path):
        with open(info_file_path, "r") as f:
            return json.load(f)
    return None

st.title("Missing People Face Detection")

stored_embeddings = load_stored_images()

# Upload a single image for comparison
uploaded_file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg", "jfif"])

if uploaded_file:
    st.image(uploaded_file)

if st.button("Predict"):
    if not uploaded_file:
        st.error("Please upload an image file.")
    else:
        # Convert uploaded image to numpy array
        uploaded_image = np.asarray(Image.open(uploaded_file))
        
        # Process the uploaded image against the stored embeddings
        matches = process_image_with_collection(uploaded_image, stored_embeddings, app_model())
        
        if matches:
            # Get the best match
            best_match, score = matches[0]
            st.success("Missing person detected!")
            
            # Extract the folder name (remove file extension from best match)
            person_folder_name = os.path.splitext(best_match)[0]  # Get the folder name (without .jpg or extension)
            person_folder = os.path.join("../subscribed-people", person_folder_name)
            
            # Load additional details from the person's info.json file
            person_info = load_person_info(person_folder)
            
            if person_info:
                st.subheader("Person Information:")
                st.markdown(f"**Name**: {person_info['name']}")
                st.write(f"**Age**: {person_info['age']}")
                st.write(f"**Last known location:** {person_info['last_known_location']}")
                st.write(f"**Missing since:** {person_info['missing_since']}")
                st.write(f"**Additional details:** {person_info['other_details']}")
                st.write(f"**Contact:** {person_info['contact']}")

                matched_image_path = os.path.join(person_folder, best_match)  
                if os.path.exists(matched_image_path):
                    st.image(matched_image_path, caption="Matched Person", use_container_width=True)
                else:
                    st.warning("Image not found for this person.")
            else:
                st.warning("No additional information available for this person.")
        else:
            st.warning("No matches found.")

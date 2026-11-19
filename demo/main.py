
import os
import json
import html
import streamlit as st
import numpy as np

from PIL import Image
from insightface.app import FaceAnalysis
from numpy.linalg import norm


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Missing People Face Detection",
    page_icon="🔎",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .profile-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.2);
    }

    .person-name {
        font-size: 27px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .match-label {
        color: #16a34a;
        font-weight: 600;
        font-size: 14px;
    }

    .info-label {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 3px;
    }

    .info-value {
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 14px;
        overflow-wrap: anywhere;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SUBSCRIBED_PEOPLE_DIR = "../subscribed-people"

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".jfif"
)

SIMILARITY_THRESHOLD = 0.45


# --------------------------------------------------
# LOAD INSIGHTFACE MODEL
# --------------------------------------------------

@st.cache_resource
def app_model():

    app = FaceAnalysis()

    # CPU execution
    app.prepare(ctx_id=-1)

    return app


# --------------------------------------------------
# LOAD STORED EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def load_stored_embeddings(directory=SUBSCRIBED_PEOPLE_DIR):

    embeddings = {}

    app = app_model()

    if not os.path.isdir(directory):
        st.error(f"Directory not found: {directory}")
        return embeddings

    for folder_name in os.listdir(directory):

        person_folder_path = os.path.join(
            directory,
            folder_name
        )

        if not os.path.isdir(person_folder_path):
            continue

        for filename in os.listdir(person_folder_path):

            if filename.lower().endswith(IMAGE_EXTENSIONS):

                filepath = os.path.join(
                    person_folder_path,
                    filename
                )

                try:

                    image = np.asarray(
                        Image.open(filepath).convert("RGB")
                    )

                    faces = app.get(image)

                    if faces:

                        embedding = faces[0]["embedding"]

                        embedding = embedding / norm(embedding)

                        info = {}

                        info_path = os.path.join(
                            person_folder_path,
                            "info.json"
                        )

                        if os.path.exists(info_path):

                            with open(
                                info_path,
                                "r",
                                encoding="utf-8"
                            ) as f:

                                info = json.load(f)

                        embeddings[folder_name] = (
                            embedding,
                            info
                        )

                    # Use the first image in the folder
                    break

                except Exception as e:

                    st.warning(
                        f"Could not process {filepath}: {e}"
                    )

                    break

    return embeddings


# --------------------------------------------------
# FIND BEST MATCH
# --------------------------------------------------

def find_best_match(
    uploaded_embedding,
    stored_embeddings,
    threshold=SIMILARITY_THRESHOLD
):

    best_match = None
    best_score = 0
    best_info = None

    for folder_name, (
        stored_embedding,
        info
    ) in stored_embeddings.items():

        score = np.dot(
            uploaded_embedding,
            stored_embedding
        )

        if score > best_score and score >= threshold:

            best_match = folder_name
            best_score = score
            best_info = info

    return best_match, best_score, best_info


# --------------------------------------------------
# GET PERSON IMAGE
# --------------------------------------------------

def get_person_image(person_folder_path):

    if not os.path.isdir(person_folder_path):
        return None

    for filename in os.listdir(person_folder_path):

        if filename.lower().endswith(IMAGE_EXTENSIONS):

            return os.path.join(
                person_folder_path,
                filename
            )

    return None


# --------------------------------------------------
# DISPLAY PERSON PROFILE
# --------------------------------------------------

def display_person_profile(
    person_name,
    person_info,
    similarity_score,
    person_image_path
):

    st.divider()

    st.markdown(
        '<div class="section-title">Person Profile</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        if person_image_path:

            st.image(
                person_image_path,
                use_container_width=True
            )

        else:

            st.info("No profile image available.")

    with col2:

        display_name = person_info.get(
            "name",
            person_name
        )

        safe_name = html.escape(str(display_name))

        st.markdown(
            f'<div class="person-name">{safe_name}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="match-label">● Potential database match</div>',
            unsafe_allow_html=True
        )

        st.write("")

        st.metric(
            "Similarity Score",
            f"{similarity_score:.2%}"
        )

        st.progress(
            min(max(float(similarity_score), 0.0), 1.0)
        )

        # Display information except name
        for key, value in person_info.items():

            if key == "name":
                continue

            label = html.escape(
                str(key).replace("_", " ").title()
            )

            safe_value = html.escape(str(value))

            st.markdown(
                f"""
                <div class="info-label">{label}</div>
                <div class="info-value">{safe_value}</div>
                """,
                unsafe_allow_html=True
            )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔎 Missing People Face Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Search for potential matches using facial recognition.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# LOAD DATABASE
# --------------------------------------------------

with st.spinner("Loading face recognition database..."):

    stored_embeddings = load_stored_embeddings()

st.caption(
    f"Database contains {len(stored_embeddings)} registered people."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Search for a person</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "jfif", "webp"]
)


# --------------------------------------------------
# PROCESS UPLOADED IMAGE
# --------------------------------------------------

if uploaded_file:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        uploaded_image = np.asarray(image)

        col1, col2 = st.columns([1, 2])

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            st.markdown("### Image Analysis")

            with st.spinner("Analyzing image..."):

                app = app_model()

                faces = app.get(uploaded_image)

            if not faces:

                st.warning(
                    "No face detected in the uploaded image."
                )

            else:

                st.success(
                    f"{len(faces)} face(s) detected."
                )

                if len(faces) > 1:

                    st.info(
                        "Multiple faces detected. "
                        "The first detected face will be used."
                    )

                uploaded_embedding = faces[0]["embedding"]

                uploaded_embedding = (
                    uploaded_embedding / norm(uploaded_embedding)
                )

                with st.spinner("Searching the database..."):

                    best_match, score, best_info = find_best_match(
                        uploaded_embedding,
                        stored_embeddings
                    )

                if best_match:

                    st.success("Potential match found!")

                    matched_image_path = os.path.join(
                        SUBSCRIBED_PEOPLE_DIR,
                        best_match
                    )

                    person_image_path = get_person_image(
                        matched_image_path
                    )

                    display_person_profile(
                        person_name=best_match,
                        person_info=best_info,
                        similarity_score=score,
                        person_image_path=person_image_path
                    )

                else:

                    st.warning(
                        "No potential matches found in the database."
                    )

                    st.caption(
                        f"Minimum similarity threshold: "
                        f"{SIMILARITY_THRESHOLD:.0%}"
                    )

    except Exception as e:

        st.error(
            f"An error occurred while processing the image: {e}"
        )


else:

    st.info(
        "Upload an image to start searching for a potential match."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Missing People Face Detection | Powered by InsightFace"
)

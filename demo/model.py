import numpy as np
from insightface.app import FaceAnalysis


def process_image_with_collection(uploaded_img, stored_embeddings, app):
    """
    Compare the uploaded image against a collection of stored images.

    Args:
        uploaded_img: numpy array of the uploaded image.
        stored_embeddings: dict of {filename: embedding} for stored images.
        app: InsightFace app instance.

    Returns:
        List of tuples (filename, similarity_score), sorted by similarity in descending order.
    """
    uploaded_embedding = app.get(uploaded_img)[0]['embedding']
    results = []
    for filename, stored_embedding in stored_embeddings.items():
        score = np.dot(uploaded_embedding, stored_embedding)  # Cosine similarity
        results.append((filename, score))
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
    return results

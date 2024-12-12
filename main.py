# main.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import requests

# ------------------------------
# Google Drive Links for Model Files
# ------------------------------

MODEL_FILES = {
    'TfidfVectorizer.joblib': 'https://drive.google.com/uc?export=download&id=1_okuE11hoEQBQ-GTlhfsJNHvn24HtHPX',
    'categorical_tfidf.joblib': 'https://drive.google.com/uc?export=download&id=1rXZ-6YkwuK7rVH96r1v08H8bYGG6XVQr',
    'knn_model.joblib': 'https://drive.google.com/uc?export=download&id=1agA2gdSNNXnXjOp7aU8amTrnZEr6moCs',
    'combined_features_normalized.npy': 'https://drive.google.com/uc?export=download&id=166yw8uYQYgiVo5X6SjO184GpbdGxF5HU',
    'movies_with_embeddings.csv': 'https://drive.google.com/uc?export=download&id=1RjGo00bKSQ5ryOlzKEPTrTwjEQ6q9_Sc'
}

# ------------------------------
# Helper Function to Download Files
# ------------------------------

def download_file(file_name, url):
    """
    Download a file from a given URL and save it locally if it does not already exist.

    Parameters:
        file_name (str): Name of the file to save.
        url (str): Download URL.

    Returns:
        str: Path to the downloaded or existing file.
    """
    if not os.path.exists(file_name):
        with st.spinner(f"Downloading {file_name}..."):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors
                with open(file_name, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
                st.success(f"{file_name} downloaded successfully.")
            except Exception as e:
                st.error(f"Failed to download {file_name}: {e}")
                raise e
    else:
        st.info(f"{file_name} is already available locally.")
    return file_name

# ------------------------------
# Load Models and Data
# ------------------------------

# ------------------------------
# Load Models and Data
# ------------------------------

def load_models():
    """
    Load the TF-IDF vectorizer, KNN model, combined features, movies DataFrame, and Sentence Transformer model.

    Returns:
        tfidf (TfidfVectorizer): Loaded TF-IDF vectorizer.
        knn (NearestNeighbors): Loaded KNN model.
        combined_features (np.ndarray): Combined feature vectors.
        df (pd.DataFrame): Movies DataFrame.
        sentence_model (SentenceTransformer): Loaded Sentence Transformer model.
    """
    try:
        tfidf = joblib.load('TfidfVectorizer.joblib')  # Path to TF-IDF Vectorizer
        knn = joblib.load('knn_model.joblib')  # Path to KNN Model
        combined_features = np.load('combined_features_normalized.npy')  # Path to Combined Features
        df = pd.read_csv('movies_with_embeddings.csv')  # Path to Movies DataFrame

        # Load Sentence Transformer Model
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different model if preferred

        return tfidf, knn, combined_features, df, sentence_model
    except Exception as e:
        st.error(f"Error loading models or data: {e}")
        return None, None, None, None, None

tfidf, knn, combined_features, df, sentence_model = load_models()

if tfidf is None:
    st.stop()  # Stop the app if models/data failed to load

# ------------------------------
# Helper Functions
# ------------------------------

def extract_unique_terms(df, category):
    """
    Extract unique terms from a specific category in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing movie data.
        category (str): The category to extract terms from ('genres', 'actor', 'actress', 'director').

    Returns:
        List[str]: Sorted list of unique preprocessed terms.
    """
    terms = set()
    for entries in df[category].dropna():
        for entry in entries.split(' '):
            term = entry.strip().lower()
            if term:
                terms.add(term)
    return sorted(list(terms))

def preprocess_selection(selection):
    """
    Convert user input into the preprocessed feature name.

    Parameters:
        selection (str): The user input (e.g., "Ryan Reynolds").

    Returns:
        str: The preprocessed feature name (e.g., "ryanreynolds").
    """
    return selection.lower().replace(' ', '')

def recommend_based_on_movies(selected_movies, df, combined_features, knn_model, top_n=10):
    """
    Recommend movies based on selected favorite movies.

    Parameters:
        selected_movies (List[str]): List of favorite movie titles.
        df (pd.DataFrame): DataFrame containing movie data.
        combined_features (np.ndarray): Combined feature vectors.
        knn_model (NearestNeighbors): Fitted NearestNeighbors model.
        top_n (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: DataFrame of recommended movies.
    """
    indices = []
    for movie in selected_movies:
        # Find all indices of the movie in the dataframe (in case of duplicates)
        idx_list = df[df['title'].str.lower() == movie.lower()].index.tolist()
        if not idx_list:
            st.warning(f"Movie '{movie}' not found in the dataset.")
        else:
            indices.extend(idx_list)

    if not indices:
        st.error("No valid movies selected.")
        return pd.DataFrame()

    # Calculate the mean vector of the selected movies
    mean_vector = combined_features[indices].mean(axis=0)

    # Normalize the mean vector
    mean_vector_normalized = normalize(mean_vector.reshape(1, -1), norm='l2', axis=1)

    # Find nearest neighbors
    n_neighbors = top_n * 10  # Increase the number of neighbors to ensure enough unique titles
    distances, indices = knn_model.kneighbors(mean_vector_normalized, n_neighbors=n_neighbors)

    # Retrieve movie titles, excluding the selected movies and duplicates
    recommended_titles = []
    selected_movies_lower = [movie.lower() for movie in selected_movies]
    seen_titles = set(selected_movies_lower)
    for idx in indices[0]:
        title = df.iloc[idx]['title']
        title_lower = title.lower()
        if title_lower not in seen_titles:
            recommended_titles.append(title)
            seen_titles.add(title_lower)
        if len(recommended_titles) == top_n:
            break

    # Create DataFrame of recommendations
    final_recommendations = df[df['title'].isin(recommended_titles)]
    final_recommendations = final_recommendations.drop_duplicates(subset='title', keep='first')
    final_recommendations = final_recommendations.head(top_n)

    return final_recommendations

def recommend_based_on_text(user_description, sentence_model, combined_features, df, top_n=10):
    """
    Recommend movies based on user-provided textual description using Sentence Transformers.

    Parameters:
        user_description (str): User's description of desired movie.
        sentence_model (SentenceTransformer): Loaded Sentence Transformer model.
        combined_features (np.ndarray): Combined feature vectors.
        df (pd.DataFrame): Movies DataFrame.
        top_n (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: DataFrame of recommended movies.
    """
    if not user_description.strip():
        st.error("Please enter a description of the movie you want to watch.")
        return pd.DataFrame()

    # Encode user description
    user_vector = sentence_model.encode(user_description)

    # Normalize the user vector
    user_vector_normalized = normalize(user_vector.reshape(1, -1), norm='l2', axis=1)

    # Compute cosine similarity with movie overviews
    similarities = cosine_similarity(user_vector_normalized, combined_features[:, :384])
    similarities = similarities.flatten()

    # Get top indices
    top_indices = similarities.argsort()[::-1]

    # Retrieve movie titles, ensuring no duplicates
    recommended_titles = []
    seen_titles = set()
    for idx in top_indices:
        title = df.iloc[idx]['title']
        title_lower = title.lower()
        if title_lower not in seen_titles:
            recommended_titles.append(title)
            seen_titles.add(title_lower)
        if len(recommended_titles) == top_n:
            break

    # Create DataFrame of recommendations
    final_recommendations = df[df['title'].isin(recommended_titles)]
    final_recommendations = final_recommendations.drop_duplicates(subset='title', keep='first')
    final_recommendations = final_recommendations.head(top_n)

    return final_recommendations

def display_recommendations(recommended_df):
    """
    Display recommended movies.

    Parameters:
        recommended_df (pd.DataFrame): DataFrame containing recommended movies with 'title', 'overview', etc.
    """
    for idx, row in recommended_df.reset_index(drop=True).iterrows():
        st.markdown(f"### {idx + 1}. {row['title']}")
        if 'overview' in row and not pd.isna(row['overview']):
            with st.expander("See overview"):
                st.write(row['overview'])
        st.write("---")

# ------------------------------
# Extract Unique Terms for Dropdowns
# ------------------------------

unique_genres = extract_unique_terms(df, 'genres')
actors = extract_unique_terms(df, 'actor')
actresses = extract_unique_terms(df, 'actress')
unique_cast = sorted(list(set(actors) | set(actresses)))
unique_directors = extract_unique_terms(df, 'director')

# Extract unique movie titles for dropdowns, removing duplicates based on title
unique_movie_titles = sorted(df['title'].drop_duplicates().unique())
unique_movie_titles_with_prompt = ["Select a movie"] + unique_movie_titles

# ------------------------------
# Streamlit App Layout
# ------------------------------

# Use a custom title with emojis
st.markdown("<h1 style='text-align: center;'>🎥 Movie Recommendation System 🍿</h1>", unsafe_allow_html=True)

# Add a horizontal divider
st.markdown("---")

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.write("Welcome to the **Movie Recommendation System**!")
    st.write("Get personalized movie recommendations based on your favorite movies or a description of what you'd like to watch.")
    st.write("Navigate through the tabs to choose a recommendation method.")
    st.write("Enjoy your movie journey! 🎬")

# Create tabs for different recommendation functionalities
tab1, tab2 = st.tabs(["⭐ Favorite Movies", "💬 Describe a Movie"])

with tab1:
    st.header("Recommend Based on Favorite Movies")
    st.write("Select **three** of your favorite movies to receive similar movie recommendations.")

    # User Inputs for Favorite Movies using Dropdowns
    col1, col2, col3 = st.columns(3)

    with col1:
        favorite_movie_1 = st.selectbox(
            "First favorite movie:",
            unique_movie_titles_with_prompt,
            index=0  # Default selection
        )

    with col2:
        favorite_movie_2 = st.selectbox(
            "Second favorite movie:",
            unique_movie_titles_with_prompt,
            index=0  # Default selection
        )

    with col3:
        favorite_movie_3 = st.selectbox(
            "Third favorite movie:",
            unique_movie_titles_with_prompt,
            index=0  # Default selection
        )

    # Button to trigger movie-based recommendations
    if st.button("Get Recommendations", key='fav_movies'):
        # Check if all movies are selected
        if (favorite_movie_1 == "Select a movie" or
            favorite_movie_2 == "Select a movie" or
            favorite_movie_3 == "Select a movie"):
            st.error("Please select all three favorite movies.")
        else:
            # Ensure that the three selected movies are not the same
            if len(set([favorite_movie_1, favorite_movie_2, favorite_movie_3])) < 3:
                st.error("Please select three different movies.")
            else:
                with st.spinner('Generating recommendations...'):
                    # Simulate progress
                    time.sleep(1)  # Simulate a delay
                    recommendations = recommend_based_on_movies(
                        [favorite_movie_1, favorite_movie_2, favorite_movie_3],
                        df,
                        combined_features,
                        knn,
                        top_n=20
                    )
                if recommendations is not None and not recommendations.empty:
                    st.success("Here are your recommended movies:")
                    display_recommendations(recommendations.reset_index(drop=True))
                    st.balloons()
                else:
                    st.write("No recommendations found based on your selections.")

with tab2:
    st.header("Recommend Based on Description")
    st.write("Describe the kind of movie you want to watch, and we'll recommend movies based on your description.")

    # User Input
    user_description = st.text_area(
        "Enter your movie preferences:",
        height=150,
        placeholder="e.g., A thrilling adventure in space with unexpected twists."
    )

    # Button to Generate Recommendations
    if st.button("Get Recommendations", key='description'):
        if not user_description.strip():
            st.error("Please enter a description of the movie you want to watch.")
        else:
            with st.spinner('Generating recommendations...'):
                # Simulate progress
                time.sleep(1)  # Simulate a delay
                recommendations = recommend_based_on_text(
                    user_description,
                    sentence_model,
                    combined_features,
                    df,
                    top_n=20
                )
            if recommendations is not None and not recommendations.empty:
                st.success("Here are your recommended movies:")
                display_recommendations(recommendations.reset_index(drop=True))
                st.balloons()
            else:
                st.write("No recommendations found based on your description.")

# Add a footer
st.markdown("---")
st.markdown("<h4 style='text-align: center;'>© 2023 Movie Recommendation System</h4>", unsafe_allow_html=True)

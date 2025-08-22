import streamlit as st
import pandas as pd
from recommender import Recommender

# Initialize recommender
@st.cache_resource
def load_recommender():
    return Recommender()

recommender = load_recommender()

st.title("🎬 Movie Recommendation System")

# Sidebar to select mode
mode = st.sidebar.radio("Choose Recommendation Mode:", ["Cold Start", "Hybrid (CF + CB)"])

if mode == "Cold Start":
    st.header("Cold Start Recommendation")

    # Multi-select genres
    selected_genres = st.multiselect("Select genres you like:", recommender.all_genres)

    top_n = st.number_input("Number of recommendations", min_value=5, max_value=50, value=10, step=1)

    if st.button("Recommend"):
        if not selected_genres:
            st.warning("Please select at least one genre.")
        else:
            df = recommender.cold_start_recommend(selected_genres, top_n=top_n)
            st.subheader("Recommended Movies 🎥")
            for _, row in df.iterrows():
                st.markdown(f"**{row['title']}**  \nGenres: {', '.join(row['genres'])}")

elif mode == "Hybrid (CF + CB)":
    st.header("Hybrid Recommendation")

    st.markdown("Enter the IMDb IDs of movies you’ve watched with your ratings (1-5).")

    imdb_ids_input = st.text_area("IMDb IDs (comma separated, e.g., tt0468569, tt0372784):")
    ratings_input = st.text_area("Ratings (comma separated, same order, e.g., 4, 5):")

    n_recommendation = st.number_input("Number of recommendations", min_value=5, max_value=50, value=10, step=1)

    if st.button("Recommend"):
        if not imdb_ids_input.strip() or not ratings_input.strip():
            st.warning("Please provide IMDb IDs and Ratings.")
        else:
            imdb_ids = [x.strip() for x in imdb_ids_input.split(",")]
            ratings = [float(x.strip()) for x in ratings_input.split(",")]

            if len(imdb_ids) != len(ratings):
                st.error("Number of IMDb IDs and ratings must match.")
            else:
                movie_ids = []
                invalid_ids = []
                for imdb in imdb_ids:
                    movie_id = recommender.imdb_to_movie_id(imdb)
                    if movie_id is None:
                        invalid_ids.append(imdb)
                    else:
                        movie_ids.append(movie_id)

                if invalid_ids:
                    st.error(f"Invalid IMDb IDs: {', '.join(invalid_ids)}")
                else:
                    recommendations, _ = recommender.recommend(movie_ids, ratings, n_recommendation)

                    st.subheader("Recommended Movies 🎥")
                    for _, row in recommendations.iterrows():
                        st.markdown(
                            f"**{row['title']}**  \n"
                            f"Genres: {', '.join(row['genres'])}  \n"
                            f"🔹 Hybrid Similarity: `{row['hybrid']:.3f}`  \n"
                            f"🔹 Genres Similarity: `{row['genres_similarity']:.3f}`  \n"
                            f"🔹 Keyword Similarity: `{row['keyword_similarity']:.3f}`  \n"
                            f"🔹 Overview Similarity: `{row['overview_similarity']:.3f}`"
                        )

from content_based import ContentBased
from collabrative_filtering import CollabrativeFiltering
import pandas as pd

class Recommender:
    def __init__(self):
        credits = pd.read_csv('../dataset/credits.csv')
        keywords = pd.read_csv('../dataset/keywords.csv')
        links = pd.read_csv('../dataset/links_small.csv')
        md = pd.read_csv('../dataset/movies_metadata.csv')
        ratings = pd.read_csv('../dataset/ratings_small.csv')

        self.collabrative_filtering = CollabrativeFiltering(
            credits_df=credits,
            keywords_df=keywords,
            links_df=links,
            md_df=md,
            ratings_df=ratings,
        )

        self.content_based = ContentBased(
            credits_df=credits,
            keywords_df=keywords,
            links_df=links,
            md_df=md,
            ratings_df=ratings,
        )

    def recommend(self, movie_ids, ratings, n_recommendation, alpha=0.25):
        # Raw recommendation uses for metrics like precision and etc...
        if len(movie_ids) != len(ratings):
            raise ValueError('number of movie ids and ratings are not the same !')

        cb_recommendation = self.content_based.recommend(movie_ids, ratings)
        cf_recommendation = self.collabrative_filtering.recommend(movie_ids, ratings, cb_recommendation.index.tolist())

        merged_recommendations = cb_recommendation.merge(cf_recommendation, left_index=True, right_on='movieId')

        merged_recommendations['hybrid'] = (merged_recommendations['rating'] / 5) * alpha + merged_recommendations['final_similarity'] * (1 - alpha)
        merged_recommendations.sort_values(by='hybrid', ascending=False, inplace=True)

        raw_recommendations = merged_recommendations.copy()

        # Remove movies that user rated
        repetitive_movie_ids = merged_recommendations[merged_recommendations['movieId'].isin(movie_ids)].index
        merged_recommendations.drop(repetitive_movie_ids, inplace=True)

        recommendations = merged_recommendations[:n_recommendation].sort_values(by='wr', ascending=False)

        return recommendations, raw_recommendations
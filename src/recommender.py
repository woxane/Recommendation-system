from src.content_based import ContentBased
from src.collabrative_filtering import CollabrativeFiltering
import pandas as pd
from ast import literal_eval

class Recommender:
    def __init__(self):
        credits = pd.read_csv('dataset/credits.csv')
        keywords = pd.read_csv('dataset/keywords.csv')
        links = pd.read_csv('dataset/links_small.csv')
        self.md = pd.read_csv('dataset/movies_metadata.csv')
        ratings = pd.read_csv('dataset/ratings_small.csv')
        md_copy = self.md.copy()

        self.collabrative_filtering = CollabrativeFiltering(
            credits_df=credits,
            keywords_df=keywords,
            links_df=links,
            md_df=md_copy,
            ratings_df=ratings,
        )

        self.content_based = ContentBased(
            credits_df=credits,
            keywords_df=keywords,
            links_df=links,
            md_df=md_copy,
            ratings_df=ratings,
        )

        self.md['genres'] = self.md['genres'].fillna('[]').apply(literal_eval)
        self.md['genres'] = self.md['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

        self.all_genres = self.md['genres'].explode().unique()

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
    

    def cold_start_recommend(self, genres, top_n=50, percentile=0.85):
        filtered_row_ids = []
        
        for idx, row in self.md.iterrows():
            if all(genre in row['genres'] for genre in genres):
                filtered_row_ids.append(idx)
        
        filtered_row_ids = list(set(filtered_row_ids))
        
        df = self.md.loc[filtered_row_ids]
        
        vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)
        
        qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['id', 'title', 'genres', 'vote_count', 'vote_average']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        
        qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(top_n)
        
        return qualified
    

    def imdb_to_movie_id(self, imdb_id):
        movie_id = self.md[self.md['imdb_id'] == imdb_id]['id'].values
        if len(movie_id) == 0:
            return None
        return int(movie_id[0])

    def get_title_by_movie_id(self, movie_id):
        title =  self.md[self.md['id'] == str(movie_id)]['title'].values
        if len(title) == 0:
            return None
        return title[0]
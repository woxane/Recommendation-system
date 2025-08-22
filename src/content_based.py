import pandas as pd
import seaborn as snas
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval # use literal eval instead of eval
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer

class ContentBased:
    def __init__(
            self,
            credits_df,
            keywords_df,
            links_df,
            md_df,
            ratings_df,
            vote_count_cut_off_quantile = 0.8
    ):
        self._vote_count_cut_off_quantile = vote_count_cut_off_quantile

        # Those ends with *_df are dataset that located in dataset directory from root
        self._embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self._credits_df = credits_df
        self._keywords_df = keywords_df
        self._links_df = links_df
        self._md_df = md_df
        self._ratings_df = ratings_df

        # Clean dataset
        self._df = self._df_clean()

        # Dataset data for weighted rating
        self._m = self._md_df['vote_count'].quantile(self._vote_count_cut_off_quantile)
        self._C = self._md_df['vote_average'].mean()


        self._all_genres = self._md_df['genres'].explode().unique()

        self._all_supported_casts, self._cast_index_map = self._get_supported_casts()
        self._all_supported_directors, self._director_index_map = self._get_supported_directors()


        self._movies_latent_feature = self._get_movie_latent_vector(3, self._df)

    def recommend(self, movie_ids, ratings):
        if len(movie_ids) != len(ratings):
            raise ValueError('number of movie ids and ratings are not the same !')
        
        user_id = len(movie_ids) * [1]
        test_user = pd.DataFrame({
            'userId': user_id,
            'movieId': movie_ids,
            'rating': ratings,
        })

        top_k_casts = 3
        user_latent_feature = self._get_user_latent_vector(top_k_casts, test_user)
        user_sim_matrix = np.array([self._get_similarity(user_latent_feature.iloc[0], movie_features)[0] for _, movie_features in self._movies_latent_feature.iterrows()])

        rmi = np.argsort(user_sim_matrix)[::-1]

        similarities = user_sim_matrix[rmi]

        results_df = self._df.iloc[rmi][['title', 'genres', 'wr', 'id']].copy()
        results_df.set_index(results_df['id'], inplace=True)
        results_df.drop('id', axis=1, inplace=True)
        results_df['similarity'] = similarities

        # TODO: OKAY THIS CONSTANT
        alpha = 0.25
        results_df['ranking_metrics'] = (results_df['wr'] / 10) * alpha + results_df['similarity'] * (1 - alpha)
        return results_df

    def _get_user_latent_vector(self, top_k, user_df):
        user = user_df.copy()
        # Check user movie id rates be in movie metadata
        user = user[user['movieId'].isin(self._df['id'])].reset_index(drop=True)
        
        user_mean_rates = user['rating'].mean()

        # Get movie latent features for user rates
        user_movie_rates = self._df[self._df['id'].isin(user['movieId'])]
        user_movie_latents = self._get_movie_latent_vector(top_k, user_movie_rates)

        # If all of the rates are equal don't get them a weight
        if user['rating'].std() != 0:
            for idx, row in user.iterrows():
                movie_id = row['movieId']
                # user_movie_latents.loc[movie_id, :] *= row['rating'] - user_mean_rates
                # I don't want to use the user_mean_rates normalization
                user_movie_latents.loc[movie_id, :] *= row['rating']

        n_movie_rates = user_movie_latents.shape[0]
        
        columns_to_aggregate = [
            'genres', 'cast', 'crew', 'keywords_embedding', 'overview_embedding'
        ]
        
        profile_dict = {
            col: [user_movie_latents[col].sum() / n_movie_rates]
            for col in columns_to_aggregate
        }
        
        profile = pd.DataFrame(profile_dict)
        return profile

    def _get_similarity(self, row1, row2):
        # Genres
        genres1 = np.expand_dims(row1['genres'], axis=0)
        genres2 = np.expand_dims(row2['genres'], axis=0)
        genres_similarity = cosine_similarity(genres1, genres2)[0, 0]


        # As our cast and crew is csr matrix, getting cosine similary for complete 0 would lead us to -1 as a placeholder for undefined cosine
        # So i would check for -1 
        # Cast
        cast1 = row1['cast']
        cast2 = row2['cast']
        cast_similarity = cosine_similarity(cast1, cast2)[0, 0]
        cast_similarity = 0 if cast_similarity == -1 else cast_similarity
            
        # Crew
        crew1 = row1['crew']
        crew2 = row2['crew']
        crew_similarity = cosine_similarity(crew1, crew2)[0, 0]
        crew_similarity = 0 if crew_similarity == -1 else crew_similarity


        # Keyword
        keyword1 = np.expand_dims(row1['keywords_embedding'], axis=0)
        keyword2 = np.expand_dims(row2['keywords_embedding'], axis=0)
        keyword_similarity = cosine_similarity(keyword1, keyword2)[0, 0]

        # Overview
        overview1 = np.expand_dims(row1['overview_embedding'], axis=0)
        overview2 = np.expand_dims(row2['overview_embedding'], axis=0)
        overview_similarity = cosine_similarity(overview1, overview2)[0, 0]

        # Weights:
        genre_w = 0.25
        cast_w = 0.05
        crew_w = 0.05
        keyword_w = 0.30
        overview_w = 0.35

        final_similarity = genre_w * genres_similarity + cast_w * cast_similarity + crew_w * crew_similarity + keyword_w * keyword_similarity + overview_w * overview_similarity

        return final_similarity, genres_similarity, cast_similarity, crew_similarity, keyword_similarity, overview_similarity

    def _get_movie_latent_vector(self, movie_dataframe, top_k=3):
        # Genres
        genres_df = self._encode_genres(movie_dataframe['genres'])

        # Credits
        cast_df = movie_dataframe['cast'].apply(self._encode_casts)
        director_df = movie_dataframe['crew'].apply(self._encode_directors)

        # Keywords (Textual)
        keywords_df = self._get_top_k_keywords(10, movie_dataframe['keywords'])
        keywords_embedding_df = keywords_df.apply(self._embed_texts)

        # Movie Textual Data (Textual)
        movie_textual_df = self._get_movie_textual_data(movie_dataframe['overview'], movie_dataframe['tagline'])
        movie_textual_embedding_df = movie_textual_df.apply(self._embed_texts)

        result = pd.concat([genres_df, cast_df, director_df, keywords_embedding_df, movie_textual_embedding_df], axis=1)
        result.index = movie_dataframe['id']
        result = result.rename({0: 'overview_embedding', 'keywords': 'keywords_embedding'}, axis=1)
        return result

    def _weighted_rating(self, row):
        v = row['vote_count']
        R = row['vote_average']
        return (v/(v+self._m) * R) + (self._m/(self._m+v) * self._C)
    

    def _get_top_k_keywords(top_k, keyword_df):
        keyword_df = keyword_df.copy()
        # Add underscore between same group keyword to they treated as one token
        keyword_df = keyword_df.apply(lambda x: " ".join(['_'.join(i['name'].split()) for i in x][:top_k]))
        return keyword_df
    
    def get_movie_textual_data(overview_df, tagline_df):
        result_df = overview_df.fillna(' ') + ' ' + tagline_df.fillna(' ')
        return result_df

    def _encode_directors(self, crew_list, top_k_directors=1):
        rows, cols, data = [], [], []
        director_ids = [d['id'] for d in crew_list if d.get('job') == 'Director']
        # Just one director
        for did in director_ids[:top_k_directors]:
            if did in self._director_index_map:
                col_idx = self._director_index_map[did]
            else: 
                col_idx = self._director_index_map[0]  # all others go here

            rows.append(0)
            cols.append(col_idx)
            data.append(1)

        return csr_matrix((data, (rows, cols)), shape=(1, len(self._all_supported_directors)))

    def _get_supported_directors(self, directors_n_movies_quantile=0.995):
        # This function is like the supported casts
        directors_n_movies = credits['crew'].apply(lambda x: [i['id'] for i in x if i['job'] == 'Director']).explode().value_counts()

        director_movies_cut_off = directors_n_movies.quantile(directors_n_movies_quantile)

        all_supported_directors = directors_n_movies[directors_n_movies >= director_movies_cut_off].index
        # Add a index 0 for other directors
        all_supported_directors = np.append(all_supported_directors, 0)
        director_index_map = {did: idx for idx, did in enumerate(all_supported_directors)}

        return all_supported_directors, director_index_map

    def _encode_casts(self, cast_list, top_k_casts=3):
        rows, cols, data = [], [], []

        for member in cast_list[:top_k_casts]:
            cid = member["id"]
            if cid in self._cast_index_map:
                col_idx = self._cast_index_map[cid]
            else:
                col_idx = self._cast_index_map[0]  # all others go here

            rows.append(0)
            cols.append(col_idx)
            data.append(1)

        return csr_matrix((data, (rows, cols)), shape=(1, len(self._all_supported_casts)))
    
    def _get_supported_casts(self, top_k_casts=3, cast_played_movies_quantile=0.995):
        # This function would find a very little quantile of 0.995 
        # and from that choose those casts that have played movie more than that
        # Also we would just get top 3 casts in a movie

        casts_n_played_movies = self._credits_df['cast'].apply(lambda cast_list: [i['name'] for i in cast_list][:top_k_casts]).explode().value_counts()

        cast_n_play_cut_off = casts_n_played_movies.quantile(cast_played_movies_quantile)
        all_supported_casts = casts_n_played_movies[casts_n_played_movies >= cast_n_play_cut_off].index

        # Add a index 0 for other casts
        all_supported_casts = np.append(all_supported_casts, 0)
        cast_index_map = {cid: idx for idx, cid in enumerate(all_supported_casts)}

        return all_supported_casts, cast_index_map
    
    def _encode_genres(self, genres_df):
        genres_df = genres_df.copy()

        genres_df = genres_df.apply(
            lambda genres_list: np.array([genre in genres_list for genre in self._all_genres], dtype=int)
        )

        return genres_df

    def _df_clean(self):
        self._md_clean()
        self._credits_clean()
        self._keywords_clean()

        self._md_df['wr'] = self._md.apply(self._weighted_rating, axis=1)

        df = pd.merge(self._md_df, self._credits_df).merge(self._keywords_df)

        vote_count_cut_off = df['vote_count'].quantile(0.80)
        df = df[df['vote_count'] >= vote_count_cut_off]

        return df

    def _md_clean(self):
        eval_columns = ['belongs_to_collection', 'production_companies', 'production_countries', 'spoken_languages', 'genres']
        for eval_column in eval_columns:
            self._self._md_df[eval_column] = self._self._md_df[eval_column].fillna('[]').apply(literal_eval)
            self._md_df[eval_column] = self._md_df[eval_column].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

        bad_data = self._md_df[self._md_df['iself._md_dfb_id'] == '0'].index
        self._md_df.drop(bad_data, inplace=True)

        cols_to_float = ['revenue', 'vote_count', 'vote_average', 'budget', 'popularity']
        self._md_df[cols_to_float] = self._md_df[cols_to_float].astype(float)

        self._md_df['id'] = self._md_df['id'].astype(int)

    def _credits_clean(self):
        eval_columns = ['cast', 'crew']

        for eval_column in eval_columns:
            self._credits_df[eval_column] = self._credits_df[eval_column].fillna('[]').apply(literal_eval)

        self._credits_df['id'] = self._credits_df['id'].astype(int)

    def _keywords_clean(self):
        eval_columns = ['keywords']

        for eval_column in eval_columns:
            self._keywords_df[eval_column] = self._keywords_df[eval_column].fillna('[]').apply(literal_eval)

        self._keywords_df['id'] = self._keywords_df['id'].astype(int)

    def _embed_texts(self, sentences):
        embeddings = self._embedding_model.encode(sentences, normalize_embeddings=True)
        return embeddings 
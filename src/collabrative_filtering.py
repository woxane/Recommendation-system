import pandas as pd
from surprise import Reader, SVD, Dataset


class CollabrativeFiltering:
    def __init__(
        self,
        credits_df,
        keywords_df,
        links_df,
        md_df,
        ratings_df,
    ):
        self._credits_df = credits_df
        self._keywords_df = keywords_df
        self._links_df = links_df
        self._md_df = md_df
        self._ratings_df = ratings_df

        # Get from CF jupyter notebook
        self._MODEL_BEST_PARAMS = {'n_epochs': 10, 'lr_all': 0.01}

    def _train_model(self, ratings):
        # Support again training
        reader = Reader(rating_scale=(1, 5))

        data = Dataset.load_from_df(
            ratings[['userId', 'movieId', 'rating']],
            reader
        )

        svd_model = SVD(**self._MODEL_BEST_PARAMS)
        data = data.build_full_trainset()
        svd_model.fit(data)
        return svd_model

    def recommend(self, movie_ids, user_ratings, recommended_movie_ids=None):
        """
        if recommended_movie_ids passes we would iterate over all of the movies and sort based on that.
        if not (like movie ids get from CB, we would use those) 
        """
        if recommended_movie_ids is None:
            recommended_movie_ids = self._md_df['id'].values

        if len(movie_ids) != len(user_ratings):
            raise ValueError('number of movie ids and user ratings are not the same !')
        
        user_id = len(movie_ids) * [1]
        test_user = pd.DataFrame({
            'userId': user_id,
            'movieId': movie_ids,
            'rating': user_ratings,
        })

        ratings = pd.concat((self._ratings_df, test_user))
        svd_model = self._train_model(ratings)

        result_data = []
        for idx in recommended_movie_ids:
            _, iid, _, est, _ = svd_model.predict(uid=800, iid=idx)
            result_data.append((iid, est))

        result_df = pd.DataFrame(result_data).sort_values(by=1, ascending=False).rename({0: 'movieId', 1: 'rating'}, axis=1)
        
        return result_df
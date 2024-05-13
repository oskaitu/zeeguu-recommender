from datetime import datetime
from enum import Enum
import os
import numpy as np
from api.zeeguu.core.model.article import Article
from api.zeeguu.core.model.user_article import UserArticle
from recommender.cf_model import CFModel
from recommender.mapper import Mapper
from recommender.utils.tensor_utils import build_liked_sparse_tensor
from recommender.utils.train_utils import Measure, train
from recommender.utils.recommender_utils import filter_article_embeddings, get_recommendable_articles, setup_df_rs
from recommender.utils.train_utils import user_embeddings_path, article_embeddings_path
import pandas as pd
from typing import Callable
from IPython import display
from recommender.mock.tensor_utils_mock import build_mock_sparse_tensor
from recommender.mock.generators_mock import generate_articles_with_titles
from recommender.visualization.model_visualizer import ModelVisualizer
from recommender.utils.elastic_utils import find_articles_like

class RecommenderSystem:
    visualizer = ModelVisualizer()

    def __init__(
        self,
        sessions : pd.DataFrame,
        mapper: Mapper,
        num_users: int,
        num_items: int,
        data_since: datetime = None,
        embedding_dim : int =20,
        generator_function: Callable=None, #function type
        stddev=0.1,
    ):
        self.mapper = mapper
        self.test=generator_function is not None
        if(self.test):
            print("Warning! Running in test mode")
            self.sessions = generator_function(num_users, num_items)
            self.articles = generate_articles_with_titles(num_items)
        else:
            self.sessions = sessions
            articles = get_recommendable_articles(since_date=data_since)
            self.articles = mapper.map_articles(articles)
        self.cf_model = CFModel(self.sessions, num_users, num_items, embedding_dim, self.test, stddev)

    def compute_scores(self, query_embedding, item_embeddings, measure=Measure.DOT):
        """Computes the scores of the candidates given a query.
        Args:
            query_embedding: a vector of shape [k], representing the query embedding.
            item_embeddings: a matrix of shape [N, k], such that row i is the embedding
            of item i.
            measure: a string specifying the similarity measure to be used. Can be
            either DOT or COSINE.
        Returns:
            scores: a vector of shape [N], such that scores[i] is the score of item i.
        """
        u = query_embedding
        V = item_embeddings
        if measure == Measure.COSINE:
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            u = u / np.linalg.norm(u)
        scores = u.dot(V.T)
        return scores
    
    def user_recommendations(self, user_id: int, language_id: int, measure=Measure.DOT, exclude_read: bool=True, k=None, more_like_this=True):
        if self.test:
            user_order = user_id
        else:
            user_order = self.mapper.user_id_to_order.get(user_id)
        if self.sessions is not None:
            user_likes = self.sessions[self.sessions["user_id"] == user_order]['article_id'].values
            if not self.test:
                user_likes = [self.mapper.article_order_to_id.get(l) for l in user_likes]
            print(f"User likes: {sorted(user_likes)}")

        user_embeddings = self.cf_model.embeddings["user_id"]
        article_embeddings = self.cf_model.embeddings["article_id"]

        should_recommend = True
        if should_recommend:
            valid_articles = self.articles[self.articles['language_id'] == language_id]
            valid_article_embeddings = filter_article_embeddings(article_embeddings, valid_articles['id'])
            scores = self.compute_scores(
                user_embeddings[user_order], valid_article_embeddings, measure)
            score_key = str(measure) + ' score'
            df = pd.DataFrame({
                score_key: list(scores),
                'article_id': valid_articles['id'],
                'language_id': valid_articles['language_id'],
                #'titles': valid_articles['title'],
            })#.dropna(subset=["titles"]) # dopna no longer needed because we filter in the articles that we save in the RecommenderSystem itself.
            if not self.test:
                df['article_id'] = df['article_id'].map(self.mapper.article_order_to_id)
            #df = df.sort_values([score_key], ascending=False)
            df = df.iloc[df[score_key].apply(lambda x: abs(x - 1)).argsort()]
            print("Top articles with own likes: ")
            display.display(df.head(len(df) if k is None else k))

            if exclude_read:
                own_likes = UserArticle.all_articles_of_user(user_id)
                list_of_own_likes = [id.article.id for id in own_likes]
                df_without_own_likes = df[~df['article_id'].isin(list_of_own_likes)]
                print("Top articles without own likes: ")
                display.display(df_without_own_likes.head(20 if k is None else k))

            top_results = df_without_own_likes['article_id'].head(20).values if exclude_read else df['article_id'].head(20).values

            articles_to_recommend = []
            if more_like_this:
                print("With more like this \n")
                articles_to_recommend = find_articles_like(top_results, 20, 50, language_id)
            else:
                print("Only CF \n")
                for article_id in top_results:
                    articles_to_recommend.append(Article.find_by_id(article_id))
            return articles_to_recommend
        else:
            # Possibly do elastic stuff to just give some random recommendations
            return

    def previous_likes(self, user_id: int, language_id: int):
        query = UserArticle.all_liked_articles_of_user_by_id(user_id)
        
        user_likes = []
        for article in query:
            if article.article.language_id == language_id:
                user_likes.append(article.article_id)
        
        articles_to_recommend = find_articles_like(user_likes, 20, 50, language_id)
        return articles_to_recommend

    def article_neighbors(self, article_id, measure=Measure.DOT, k=10):
        scores = self.compute_scores(
            self.cf_model.embeddings["article_id"][article_id], self.cf_model.embeddings["article_id"],
            measure)
        score_key = str(measure) + ' score'
        df = pd.DataFrame({
            score_key: list(scores),
            'article_id': self.articles['id'],
            'titles': self.articles['title'],
        })
        display.display(df.sort_values([score_key], ascending=False).head(k))

    def visualize_article_embeddings(self, marked_articles=[]):
        #TODO Fix for small test cases. Right now, the function crashes with low user/article count.
        self.visualizer.visualize_tsne_article_embeddings(self.cf_model, self.articles, marked_articles)
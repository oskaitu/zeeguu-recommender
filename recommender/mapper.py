from datetime import datetime, timedelta
import os

from pandas import DataFrame
from api.zeeguu.core.model.article import Article
from api.zeeguu.core.model.user import User
import pickle
from recommender.utils.train_utils import mappings_path

user_order_to_id_path = f"{mappings_path}user_order_mapping.pkl"
user_id_to_order_path = f"{mappings_path}user_id_mapping.pkl"
article_order_to_id_path = f"{mappings_path}article_order_mapping.pkl"
article_id_to_order_path = f"{mappings_path}article_id_mapping.pkl"

class Mapper:
    num_users = 0
    num_articles = 0
    
    def __init__(self, data_since: datetime = None):
        self.user_order_to_id = {}
        self.user_id_to_order = {}
        self.article_order_to_id = {}
        self.article_id_to_order = {}
        self.data_since = data_since

        self.__set_user_order_to_id()
        self.__set_article_order_to_id()

    def __set_article_order_to_id(self):
        if os.path.exists(article_order_to_id_path) and os.path.exists(article_id_to_order_path):
            print("Loading article mappings from files.")
            self.article_id_to_order = pickle.load(open(article_id_to_order_path, 'rb'))
            self.article_order_to_id = pickle.load(open(article_order_to_id_path, 'rb'))
            self.num_articles = len(self.article_order_to_id)
        else:
            print("No article mappings found. Building new mappings.")
            article_query = (
                Article
                    .query
                    .filter(Article.broken != 1)
                    .order_by(Article.id)
            )
            if self.data_since:
                article_query = article_query.filter(Article.published_time >= self.data_since)
            articles = article_query.all()
            index = 0
            for article in articles:
                self.article_order_to_id[index] = article.id
                self.article_id_to_order[article.id] = index
                index += 1
            self.num_articles = index

            if not (os.path.exists(mappings_path)):
                os.makedirs(mappings_path)
            with open(article_order_to_id_path, 'wb') as f:
                pickle.dump(self.article_order_to_id, f)
            with open(article_id_to_order_path, 'wb') as f:
                pickle.dump(self.article_id_to_order, f)

    def __set_user_order_to_id(self):
        if os.path.exists(user_order_to_id_path) and os.path.exists(user_id_to_order_path):
            print("Loading user mappings from files.")
            self.user_id_to_order = pickle.load(open(user_id_to_order_path, 'rb'))
            self.user_order_to_id = pickle.load(open(user_order_to_id_path, 'rb'))
            self.num_users = len(self.user_order_to_id)
        else:
            print("No user mappings found. Building new mappings.")
            users = User.query.filter(User.is_dev != True).all()
            index = 0
            for user in users:
                self.user_order_to_id[index] = user.id
                self.user_id_to_order[user.id] = index
                index += 1
            self.num_users = index
            
            if not (os.path.exists(mappings_path)):
                os.makedirs(mappings_path)
            with open(user_order_to_id_path, 'wb') as f:
                pickle.dump(self.user_order_to_id, f)
            with open(user_id_to_order_path, 'wb') as f:
                pickle.dump(self.user_id_to_order, f)

    def map_articles(self, articles: DataFrame):
        articles['id'] = articles['id'].map(self.article_id_to_order)
        return articles
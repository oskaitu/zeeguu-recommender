import collections
import os
from typing import Callable
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf

from recommender.utils.train_utils import article_embeddings_path, gravity, user_embeddings_path
from recommender.mock.tensor_utils_mock import build_mock_sparse_tensor
from recommender.utils.tensor_utils import build_liked_sparse_tensor
from recommender.utils.train_utils import Measure, train
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

class CFModel():
    embeddings = None

    def __init__(
        self,
        sessions : pd.DataFrame,
        num_users: int,
        num_items: int,
        embedding_dim,
        test,
        stddev,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.sessions = sessions
        self.embedding_dim = embedding_dim
        self.test=test
        self.stddev=stddev
        self.set_embeddings()

    def split_dataframe(self, df: pd.DataFrame, holdout_fraction : float =0.05   ):
        """Splits a DataFrame into training and test sets.
        Args:
            df: a dataframe.
            holdout_fraction: fraction of dataframe rows to use in the test set.
        Returns:
            train: dataframe for training
            test: dataframe for testing
        """
        test = df.sample(frac=holdout_fraction, replace=False)
        train = df[~df.index.isin(test.index)]
        return train, test

    def sparse_mean_square_error(self, sparse_sessions, user_embeddings, article_embeddings):
        """
        Args:
            sparse_sessions: A SparseTensor session matrix, of dense_shape [N, M]
            user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
            dimension, such that U_i is the embedding of user i.
            article_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
            dimension, such that V_j is the embedding of movie j.
        Returns:
            A scalar Tensor representing the MSE between the true ratings and the
            model's predictions.
        """
        predictions = tf.reduce_sum(
            tf.gather(user_embeddings, sparse_sessions.indices[:, 0]) *
            tf.gather(article_embeddings, sparse_sessions.indices[:, 1]),
            axis=1)
        loss = tf.losses.mean_squared_error(sparse_sessions.values, predictions)
        return loss

    def build_loss(self, regularization_coeff=0.7, gravity_coeff=1.):
        """
        Args:
            ratings: the DataFrame of movie ratings.
            embedding_dim: The dimension of the embedding space.
            regularization_coeff: The regularization coefficient lambda.
            gravity_coeff: The gravity regularization coefficient lambda_g.
        Returns:
            A CFModel object that uses a regularized loss.
        """

        train_sessions, test_sessions = self.split_dataframe(self.sessions)
        if(self.test):
            A_train = build_mock_sparse_tensor(train_sessions, "train", self.num_users, self.num_items)
            A_test = build_mock_sparse_tensor(test_sessions, "test", self.num_users, self.num_items)
        else:
            A_train = build_liked_sparse_tensor(train_sessions, self.num_users, self.num_items)
            A_test = build_liked_sparse_tensor(test_sessions, self.num_users, self.num_items)

        tf_embeddings = self.get_tf_embeddings()
        user_embeddings = tf_embeddings["user_id"]
        article_embeddings = tf_embeddings["article_id"]

        error_train = self.sparse_mean_square_error(A_train, user_embeddings, article_embeddings)
        error_test = self.sparse_mean_square_error(A_test, user_embeddings, article_embeddings)
        gravity_loss = gravity_coeff * gravity(user_embeddings, article_embeddings)
        regularization_loss = regularization_coeff * (
            # The Colab notebook just summed the values of each embedding vector. Normally, the norm of a vector is calculated using the formula for Euclidian norm.
            tf.reduce_sum(user_embeddings * user_embeddings) / user_embeddings.shape[0].value + tf.reduce_sum(article_embeddings * article_embeddings) / article_embeddings.shape[0].value)
            #tf.norm(user_embeddings*user_embeddings)/user_embeddings.shape[0].value + tf.norm(article_embeddings*article_embeddings)/article_embeddings.shape[0].value)
        total_loss = error_train + regularization_loss + gravity_loss

        losses = {
            'train_error': error_train,
            'test_error': error_test
        }
        loss_components = {
            'observed_loss': error_train,
            'regularization_loss': regularization_loss,
            'gravity_loss': gravity_loss,
        }

        return tf_embeddings, total_loss, [losses, loss_components]

    def set_embeddings(self):
        if os.path.exists(user_embeddings_path) and os.path.exists(article_embeddings_path):
            print("Attempting to load embeddings from files.")
            user_embeddings = np.load(user_embeddings_path)
            article_embeddings = np.load(article_embeddings_path)
            embeddings = {
                "user_id": user_embeddings,
                "article_id": article_embeddings
            }

            self.embeddings = embeddings
        else:
            print("No embeddings found. Building new model.")
            user_embeddings = tf.Variable(tf.random_normal(
                [self.num_users, self.embedding_dim], stddev=self.stddev))
            article_embeddings = tf.Variable(tf.random_normal(
                [self.num_items, self.embedding_dim], stddev=self.stddev))

            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                embeddings = {
                    "user_id": user_embeddings.eval(),
                    "article_id": article_embeddings.eval()
                }

            self.embeddings = embeddings

    def train_model(self, num_iterations=1000, learning_rate=0.1, plot_results=False, optimizer=tf.train.GradientDescentOptimizer):
        tf_embeddings, total_loss, metrics = self.build_loss()

        self.embeddings = train(tf_embeddings, total_loss, metrics, num_iterations, learning_rate, plot_results, optimizer)

    def get_tf_embeddings(self):
        tf_user_embeddings = tf.Variable(self.embeddings["user_id"], dtype=tf.float32)
        tf_article_embeddings = tf.Variable(self.embeddings["article_id"], dtype=tf.float32)
        tf_embeddings = {
            "user_id": tf_user_embeddings,
            "article_id": tf_article_embeddings
        }
        return tf_embeddings
import random
from api.zeeguu.core.model.article import Article
from api.zeeguu.core.model.article_difficulty_feedback import ArticleDifficultyFeedback
from api.zeeguu.core.model.user import User
from api.zeeguu.core.model.user_activitiy_data import UserActivityData
from api.zeeguu.core.model.user_article import UserArticle
from recommender.mapper import Mapper
from recommender.utils.tensor_utils import build_liked_sparse_tensor
from recommender.utils.recommender_utils import get_expected_reading_time, lower_bound_reading_speed, upper_bound_reading_speed, ShowData, get_difficulty_adjustment, get_user_reading_sessions, get_sum_of_translation_from_user_activity_data, get_all_user_language_levels
from datetime import datetime
import pandas as pd
from collections import Counter
from recommender.visualization.session_visualizer import SessionVisualizer
from api.zeeguu.core.model import db
from sqlalchemy import or_, and_

import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

class FeedbackMatrixSession:
    def __init__(self, user_id, article_id, session_duration, language_id, difficulty, word_count, article_topic_list, expected_read, liked, difficulty_feedback, days_since):
        self.user_id = user_id
        self.article_id = article_id
        self.session_duration = session_duration
        self.original_session_duration = session_duration
        self.language_id = language_id
        self.difficulty = difficulty
        self.word_count = word_count
        self.article_topic_list = article_topic_list
        self.expected_read = expected_read
        self.original_expected_read = expected_read
        self.liked = liked
        self.difficulty_feedback = difficulty_feedback
        self.days_since = days_since

class AdjustmentConfig:
    '''Adjustments made to the expected active duration of a session depending on 
    1. Variance in user fk and article fk
    2. Number of translated words in the article'''
    def __init__(self, difficulty_weight : int = 1, translation_adjustment_value: int = 3):
        self.difficulty_weight = difficulty_weight
        self.translation_adjustment_value = translation_adjustment_value

class FeedbackMatrixConfig:
    def __init__(self, adjustment_config: AdjustmentConfig = AdjustmentConfig(), show_data: list[ShowData] =[], test_tensor: bool = False, data_since: datetime = None):
        self.adjustment_config = adjustment_config
        self.show_data = show_data
        self.data_since = data_since
        self.test_tensor = test_tensor

class FeedbackMatrix:
    liked_tensor = None
    sessions_df = None
    liked_sessions_df = None
    have_read_sessions = None

    def __init__(self, config: FeedbackMatrixConfig, mapper: Mapper, num_users: int, num_articles: int):
        self.config = config
        self.mapper = mapper
        self.num_of_users = num_users
        self.num_of_articles = num_articles
        self.visualizer = SessionVisualizer()

    def __get_sessions(self):
        '''Gets all user reading sessions with respect to the given config'''
        print("Getting sessions")
        sessions: dict[tuple[int, int], FeedbackMatrixSession] = {}
        query_data, liked_data, difficulty_feedback_data = get_user_reading_sessions(self.config.data_since, self.config.show_data)

        for session in query_data:
            article_id = session.article_id
            user_id = session.user_id
            article = session.article
            session_duration = int(session.duration) / 1000 # in seconds
            if (user_id, article_id) in liked_data:
                liked_value = liked_data[(user_id, article_id)]
            else:
                liked_value = 0
            
            if (user_id, article_id) in difficulty_feedback_data:
                difficulty_feedback = difficulty_feedback_data[(user_id, article_id)]
            else:
                difficulty_feedback = 0
            
            article_topic = article.topics
            article_topic_list = []
            if len(article_topic) > 0:
                for topic in article_topic:
                    article_topic_list.append(topic.title)

            if (user_id, article_id) not in sessions:
                sessions[(user_id, article_id)] = self.create_feedback_matrix_session(session, article, session_duration, liked_value, difficulty_feedback, article_topic_list)
            else:
                sessions[(user_id, article_id)].session_duration += session_duration

        return self.__get_sessions_data(sessions)

    def create_feedback_matrix_session(self, session, article, session_duration, liked_value, difficulty_feedback_value, article_topic_list):
        return FeedbackMatrixSession(
            user_id=session.user_id,
            article_id=session.article_id,
            session_duration=session_duration,
            language_id=article.language_id,
            difficulty=article.fk_difficulty,
            word_count=article.word_count,
            article_topic_list=article_topic_list,
            expected_read=0,
            liked=liked_value,
            difficulty_feedback=difficulty_feedback_value,
            days_since=(datetime.now() - session.start_time).days,
        )
    
    def __get_sessions_data(self, sessions: 'dict[tuple[int, int], FeedbackMatrixSession]'):
        '''Manipulate data for each session in the sessions dict, according to the parameters given in the config.'''
        liked_sessions = []
        have_read_sessions = 0
        translate_data = get_sum_of_translation_from_user_activity_data(self.config.data_since)

        if self.config.adjustment_config is None:
            self.config.adjustment_config = AdjustmentConfig(difficulty_weight=self.default_difficulty_weight, translation_adjustment_value=self.default_translation_adjustment_value)

        user_language_levels = get_all_user_language_levels()
        
        for session in sessions.keys():
            if (sessions[session].user_id, sessions[session].article_id) in translate_data:
                sessions[session].session_duration -= translate_data[(sessions[session].user_id, sessions[session].article_id)]['count'] * self.config.adjustment_config.translation_adjustment_value

            if (sessions[session].user_id, sessions[session].language_id) in user_language_levels:
                sessions[session].session_duration = get_difficulty_adjustment(sessions[session], self.config.adjustment_config.difficulty_weight, user_language_levels[(sessions[session].user_id, sessions[session].language_id)]['cefr_level'])

            should_spend_reading_lower_bound = get_expected_reading_time(sessions[session].word_count, upper_bound_reading_speed)
            should_spend_reading_upper_bound = get_expected_reading_time(sessions[session].word_count, lower_bound_reading_speed)

            if self.duration_is_within_bounds(sessions[session].session_duration, should_spend_reading_lower_bound, should_spend_reading_upper_bound):
                have_read_sessions += 1
                sessions[session].expected_read = 1
                liked_sessions.append(sessions[session])
            if self.duration_is_within_bounds(sessions[session].original_session_duration, should_spend_reading_lower_bound, should_spend_reading_upper_bound):
                sessions[session].original_expected_read = 1
        
        negative_sampling_sessions = []
        for user_id in self.mapper.user_id_to_order.keys():
            random_article_ids = random.sample(self.mapper.article_id_to_order.keys(), 2)
            valid_random_article_ids = [article_id for article_id in random_article_ids if (user_id, article_id) not in sessions]
            negative_sampling_sessions = negative_sampling_sessions + [
                FeedbackMatrixSession(
                    user_id=user_id,
                    article_id=article_id,
                    article_topic_list=[],
                    session_duration=0,
                    days_since=0,
                    liked=0,
                    expected_read=-1,
                    difficulty_feedback=0,
                    language_id=0,
                    difficulty=0,
                    word_count=0
                ) for article_id in valid_random_article_ids
            ]

        return sessions, liked_sessions, have_read_sessions, negative_sampling_sessions

    def get_translation_adjustment(self, session: FeedbackMatrixSession, adjustment_value):
        timesTranslated = UserActivityData.translated_words_for_article(session.user_id, session.article_id)
        return session.session_duration - (timesTranslated * adjustment_value)

    def duration_is_within_bounds(self, duration, lower, upper):
        return duration <= upper and duration >= lower

    def generate_dfs(self):
        sessions, liked_sessions, have_read_sessions, negative_sampling_sessions = self.__get_sessions()

        df = self.__session_map_to_df(sessions)
        if self.config.test_tensor:
            liked_df = self.__session_list_to_df([FeedbackMatrixSession(1, 1, 1, 1, 1, 1, [1], 1, 1, 1, 1), FeedbackMatrixSession(1, 5, 1, 1, 1, 1, [1], 1, 1, 1, 1), FeedbackMatrixSession(2, 5, 100, 5, 5, 100, [1], 1, 1, 1, 20)])
            negative_sampling_df = self.__session_list_to_df([FeedbackMatrixSession(1, 2, 1, 1, 1, 1, [1], 1, 1, 1, 1), FeedbackMatrixSession(2, 3, 1, 1, 1, 1, [1], 1, 1, 1, 1)])
        else:
            for i in range(len(liked_sessions)):
                liked_sessions[i].user_id = self.mapper.user_id_to_order.get(liked_sessions[i].user_id)
                liked_sessions[i].article_id = self.mapper.article_id_to_order.get(liked_sessions[i].article_id)

            liked_df = self.__session_list_to_df(liked_sessions)

            for i in range(len(negative_sampling_sessions)):
                negative_sampling_sessions[i].user_id = self.mapper.user_id_to_order.get(negative_sampling_sessions[i].user_id)
                negative_sampling_sessions[i].article_id = self.mapper.article_id_to_order.get(negative_sampling_sessions[i].article_id)

            negative_sampling_df = self.__session_list_to_df(negative_sampling_sessions)

        self.sessions_df = df
        self.liked_sessions_df = liked_df
        self.have_read_sessions = have_read_sessions
        self.negative_sampling_df = negative_sampling_df

    def __session_map_to_df(self, sessions: 'dict[tuple[int, int], FeedbackMatrixSession]'):
        data = {index: vars(session) for index, session in sessions.items()}
        df = pd.DataFrame.from_dict(data, orient='index')
        return df

    def __session_list_to_df(self, sessions: 'list[FeedbackMatrixSession]'):
        # Pretty weird logic. We convert a list to a dict and then to a dataframe. Should be changed.
        data = {index: vars(session) for index, session in enumerate(sessions)}
        df = pd.DataFrame.from_dict(data, orient='index')
        return df

    def build_sparse_tensor(self, force=False):
        # This function is not run in the constructor because it takes such a long time to run.
        print("Building sparse tensor")
        if (self.liked_sessions_df is None or self.sessions_df is None or self.have_read_sessions is None) or force:
            self.generate_dfs()

        self.liked_tensor = build_liked_sparse_tensor(self.liked_sessions_df, self.num_of_users, self.num_of_articles)

    def plot_sessions_df(self, name):
        print("Plotting sessions. Saving to file: " + name + ".png")
        self.visualizer.plot_urs_with_duration_and_word_count(self.sessions_df, self.have_read_sessions, name, self.config.show_data, self.config.data_since)

    def visualize_tensor(self, file_name='tensor'):
        print("Visualizing tensor")

        if self.liked_tensor is None:
            print("Tensor is None. Building tensor first")
            self.build_sparse_tensor()

        self.visualizer.visualize_tensor(self.liked_tensor, file_name)
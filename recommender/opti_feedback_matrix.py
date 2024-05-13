

import datetime
from zeeguu.core.model.article import Article
from zeeguu.core.model.user import User
from recommender.utils.recommender_utils import ShowData, get_all_user_language_levels, get_dataframe_user_reading_sessions, get_difficulty_adjustment_opti, get_sum_of_translation_from_user_activity_data, get_expected_reading_time, lower_bound_reading_speed, upper_bound_reading_speed
from recommender.visualization.session_visualizer import SessionVisualizer
import pandas as pd

class OptiAdjustmentConfig:
    def __init__(self, difficulty_weight : int, translation_adjustment_value: int):
        self.difficulty_weight = difficulty_weight
        self.translation_adjustment_value = translation_adjustment_value

class OptiFeedbackMatrixConfig:
    def __init__(self, show_data: list[ShowData], adjustment_config: OptiAdjustmentConfig, test_tensor: bool = False, data_since: datetime = None):
        self.show_data = show_data
        self.data_since = data_since
        self.adjustment_config = adjustment_config
        self.test_tensor = test_tensor

class OptiFeedbackMatrix:
    default_difficulty_weight = 1
    default_translation_adjustment_value = 3
    
    tensor = None
    sessions_df = None
    liked_sessions_df = None
    have_read_sessions = None
    feedback_diff_list_toprint = None
    feedback_counter = 0

    def __init__(self, config: OptiFeedbackMatrixConfig):
        self.config = config
        self.num_of_users = User.num_of_users()
        self.num_of_articles = Article.num_of_articles()
        self.visualizer = SessionVisualizer()
        self.max_article_id = Article.query.filter(Article.broken == 0).order_by(Article.id.desc()).first().id
        self.max_user_id = User.query.filter(User.is_dev == False).order_by(User.id.desc()).first().id

    def generate_opti_dfs(self):
        self.have_read_sessions = 0


        user_reading_df = get_dataframe_user_reading_sessions(self.config.data_since)

        self.liked_sessions_df = pd.DataFrame(columns=user_reading_df.columns)

        if self.config.adjustment_config is None:
            self.config.adjustment_config = OptiAdjustmentConfig(difficulty_weight=self.default_difficulty_weight, translation_adjustment_value=self.default_translation_adjustment_value)
        
        translate_data = get_sum_of_translation_from_user_activity_data(self.config.data_since)
        user_language_levels = get_all_user_language_levels()

        for index, row in user_reading_df.iterrows():
            user_id = row['user_id']
            article_id = row['article_id']
            language_id = row['language_id']
            duration = row['duration']
            difficulty = row['fk_difficulty']
            word_count = row['word_count']

            #print(f"Duration before divide: {user_reading_df.at[index, 'duration']}")
            user_reading_df.at[index, 'duration'] = duration / 1000
            #print(f"Duration after divide: {user_reading_df.at[index, 'duration']}")
            if (user_id, article_id) in translate_data:
                count = translate_data[(user_id, article_id)]['count']
                user_reading_df.at[index, 'duration'] += count * self.config.adjustment_config.translation_adjustment_value
            #print(f"Duration after translate: {user_reading_df.at[index, 'duration']}")
            #print(f"Duration before difficulty: {user_reading_df.at[index, 'duration']}")
            if (user_id, language_id) in user_language_levels:
                user_reading_df.at[index, 'duration'] = get_difficulty_adjustment_opti(difficulty, duration, self.config.adjustment_config.difficulty_weight, user_language_levels[(user_id, language_id)]['cefr_level'])
            #print(f"Duration after difficulty: {user_reading_df.at[index, 'duration']}")

            #print(f"Word count: {word_count}")
            should_spend_reading_lower_bound = get_expected_reading_time(word_count, upper_bound_reading_speed)
            should_spend_reading_upper_bound = get_expected_reading_time(word_count, lower_bound_reading_speed)


            
            if self.duration_is_within_bounds(user_reading_df.at[index, 'duration'], should_spend_reading_lower_bound, should_spend_reading_upper_bound):
                #print("Inside the function")
                self.have_read_sessions += 1
                #self.liked_sessions_df = self.liked_sessions_df.append(row, ignore_index=True)
        
        self.sessions_df = user_reading_df

    def duration_is_within_bounds(self, duration, lower, upper):
            return duration <= upper and duration >= lower

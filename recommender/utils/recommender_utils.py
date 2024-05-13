from datetime import datetime
from operator import or_
import os
from enum import Enum, auto
import zeeguu
from zeeguu.core.model.article_difficulty_feedback import ArticleDifficultyFeedback
from zeeguu.core.model.user_activitiy_data import UserActivityData
from zeeguu.core.model.user_article import UserArticle
from zeeguu.core.model.user_language import UserLanguage
from zeeguu.core.model.user_reading_session import UserReadingSession
from zeeguu.core.model.user import User
from zeeguu.core.model.article import Article
import pandas as pd
from zeeguu.core.model import db

def import_tf():
    import tensorflow as tf
    tf = tf.compat.v1
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    return tf

from zeeguu.core.model import db
import sqlalchemy as database

tf = import_tf()

resource_path = './zeeguu/recommender/resources/'

average_reading_speed = 70
upper_bound_reading_speed = 45
lower_bound_reading_speed = -35
user_level_dict = None

accurate_duration_date = datetime(day=30, month=1, year=2024)

class ShowData(Enum):
    '''If no ShowData is chosen, all data will be retrieved and shown.'''
    LIKED = auto()
    RATED_DIFFICULTY = auto()

def get_resource_path():
    if not os.path.exists(resource_path):
        os.makedirs(resource_path)
        print(f"Folder '{resource_path}' created successfully.")
    return resource_path

def get_expected_reading_time(word_count, offset):
    ''' The higher the offset is, the higher we want the WPM to be. When WPM is larger, the user is expected to be able to read faster.
     Thus, high offset/WPM = low expected reading time. '''
    return (word_count / (average_reading_speed + offset)) * 60

def cefr_to_fk_difficulty(number):
    result = 0

    if 0 <= number <= 20:
        result = 1
    elif 21 <= number <= 40:
        result = 2
    elif 41 <= number <= 60:
        result = 3
    elif 61 <= number <= 80:
        result = 4
    elif 81 <= number <= 100:
        result = 5

    # This implementation matches the information that Oscar found online. This gives some weird results because a lot of articles are above 50.
    '''if 0 <= number <= 10:
        result = 1
    elif 11 <= number <= 20:
        result = 2
    elif 21 <= number <= 30:
        result = 3
    elif 31 <= number <= 40:
        result = 4
    elif 41 <= number <= 50:
        result = 5
    elif 51 <= number <= 100:
        result = 6'''

    return result

def get_diff_in_article_and_user_level(article_diff, user_level, weight):
    if article_diff > user_level:
        diff = 1 + (((article_diff - user_level) / 100) * weight)
    elif article_diff < user_level:
        diff = 1 - (((user_level - article_diff) / 100) * weight)
    else:
        diff = 1

    return diff

def days_since_normalizer(days_since):
        if days_since < 365 * 1/4:
            return 1
        elif days_since < 365 * 2/4:
            return 0.75
        elif days_since < 365 * 3/4:
            return 0.5
        return 0.25

def add_filters_to_query(query, show_data: 'list[ShowData]'):
    or_filters = []
    if ShowData.LIKED in show_data:
        query = (
            query.join(UserArticle, (UserArticle.article_id == UserReadingSession.article_id) & (UserArticle.user_id == UserReadingSession.user_id), isouter=True)
        )
        or_filters.append(UserArticle.liked == True)
    if ShowData.RATED_DIFFICULTY in show_data:
        query = (
            query.join(ArticleDifficultyFeedback, (ArticleDifficultyFeedback.article_id == UserReadingSession.article_id) & (ArticleDifficultyFeedback.user_id == UserReadingSession.user_id), isouter=True)
        )
        or_filters.append(ArticleDifficultyFeedback.difficulty_feedback.isnot(None))
    if len(or_filters) > 0:
        if len(or_filters) == 1:
            query = query.filter(or_filters[0])
        else:
            query = query.filter(or_(*or_filters))
    return query

def get_user_reading_sessions(data_since: datetime, show_data: 'list[ShowData]' = []):
    print("Getting all user reading sessions")
    liked_dict = {}
    feedback_dict = {}
    query = (
        UserReadingSession.query
            .join(User, User.id == UserReadingSession.user_id)
            .join(Article, Article.id == UserReadingSession.article_id)
            .filter(Article.broken != 1)
            .filter(User.is_dev != True)
            .filter(UserReadingSession.article_id.isnot(None))
            .order_by(UserReadingSession.user_id.asc())
            #.filter(User.is_dev != True)
            #.filter(UserReadingSession.duration >= 30000) # 30 seconds
            #.filter(UserReadingSession.duration <= 3600000) # 1 hour
    )
    if data_since:
        query = (
            query
            .filter(UserReadingSession.start_time > data_since)
            .filter(Article.published_time > data_since)
        )

    if ShowData.LIKED in show_data:
        liked_dict = get_user_article_information(data_since)
    if ShowData.RATED_DIFFICULTY in show_data:
        feedback_dict = get_all_article_difficulty_feedback(data_since)

    return query.all(), liked_dict, feedback_dict 

def get_sum_of_translation_from_user_activity_data(data_since: datetime):
    count_dict = {}
    query = (
        UserActivityData.query
            .filter(UserActivityData.event.like('%TRANSLATE TEXT%'))
    )
    if data_since:
        query = query.filter(UserActivityData.time >= data_since)
    for row in query.all():
        if (row.user_id, row.article_id) not in count_dict:
            count_dict[(row.user_id, row.article_id)] = {
                'count': 1,
            }
        else:
            count_dict[(row.user_id, row.article_id)]['count'] += 1
    
    return count_dict

def get_user_article_information(data_since: datetime):
    liked_dict = {}
    query = (
        UserArticle.query
            .filter(UserArticle.opened.isnot(None))
            .filter(UserArticle.liked == True)
    )
    if data_since:
        query = query.filter(UserArticle.opened >= data_since)
    
    for row in query.all():
        liked_dict[(row.user_id, row.article_id)] = 1

    return liked_dict

def get_all_article_difficulty_feedback(data_since: datetime):
    feedback_dict = {}
    query = (
        ArticleDifficultyFeedback.query
            .filter(ArticleDifficultyFeedback.difficulty_feedback.isnot(None))
    )
    if data_since:
        query = query.filter(ArticleDifficultyFeedback.date >= data_since)
    for row in query.all():
        if (row.user_id, row.article_id) not in feedback_dict:
            feedback_dict[(row.user_id, row.article_id)] = row.difficulty_feedback
        else:
            feedback_dict[(row.user_id, row.article_id)] = row.difficulty_feedback

    return feedback_dict

def get_all_user_language_levels():
    user_level_dict = {}
    query = (
        UserLanguage.query
            .filter(UserLanguage.cefr_level.isnot(None))
    )
    for row in query.all():
        if (row.user_id, row.language_id) not in user_level_dict:
            user_level_dict[(row.user_id, row.language_id)] = { 'cefr_level': row.cefr_level }
        else:
            user_level_dict[(row.user_id, row.language_id)]['cefr_level'] = row.cefr_level
    
    """ for row in user_level_dict:
        print(row)
        print(user_level_dict[row]) """

    return user_level_dict  


def get_difficulty_adjustment(session, weight, user_level_query):
    

    """ user_level_query = (
        UserLanguage.query
            .filter_by(user_id = session.user_id, language_id=session.language_id)
            .filter(UserLanguage.cefr_level.isnot(None))
            .with_entities(UserLanguage.cefr_level)
            .first()
    ) """
    
    if user_level_query is None:
        return session.session_duration
    user_level = user_level_query
    difficulty = session.difficulty
    fk_difficulty = cefr_to_fk_difficulty(difficulty)
    return session.session_duration * get_diff_in_article_and_user_level(fk_difficulty, user_level, weight)

def get_difficulty_adjustment_opti(difficulty, duration, weight, user_level_query):
    
    if user_level_query is None:
        return duration
    user_level = user_level_query
    fk_difficulty = cefr_to_fk_difficulty(difficulty)
    return duration * get_diff_in_article_and_user_level(fk_difficulty, user_level, weight)

def setup_df_rs(num_items : int) -> pd.DataFrame:
    '''fetches all articles and fills out the space between them
    using fillna to avoid index out of bounds.
    this is done so we cant kick them out later'''
    articles = pd.read_sql_query("Select id, title from article", db.engine)
    all_null_df = pd.DataFrame({'id': range(1, num_items+1)})
    all_null_df.fillna(0, inplace=True)
    articles = pd.merge(all_null_df, articles, on='id', how='left', validate="many_to_many")
    return articles

def get_recommendable_articles(since_date: datetime=None, lowest_id: int=None) -> pd.DataFrame:
    '''Fetches all the valid articles that a user can be recommended'''
    query = f"""
        Select distinct a.id, a.title, a.language_id, a.published_time
        from article a
        join user_article ua on a.id = ua.article_id
        where broken != 1"""
    if since_date:
        query += f" and a.published_time > '{since_date.strftime('%Y-%m-%dT%H:%M:%S')}'"
    if lowest_id:
        query += f" and a.id > {lowest_id}"
    articles = pd.read_sql_query(query, db.engine)
    return articles

def filter_article_embeddings(embeddings, article_ids):
    '''Filters the article embeddings to only include the articles that are in the articles dataframe'''
    '''with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        embeddings_result = tf.nn.embedding_lookup(embeddings, [5])
        
        embeddings_array = embeddings_result.eval()'''
    return embeddings[article_ids]

def setup_df_correct(num_items : int) -> pd.DataFrame:
    
    article_query ="SELECT id, title FROM article ORDER BY id ASC"
    article_list = pd.read_sql(article_query, db.engine)

    all_null_df = pd.DataFrame({'id': range(1, num_items+1)})
    articles = pd.merge(all_null_df, article_list, on='id', how='left', validate="many_to_many")
    articles = articles.dropna(subset=['title'])
    print("printing articles")
    print(articles)
    print(len(articles))

    return article_list

def get_dataframe_user_reading_sessions(data_since: datetime):
    DB_URI = zeeguu.core.app.config["SQLALCHEMY_DATABASE_URI"]
    engine = database.create_engine(DB_URI)
    date_since = data_since.strftime('%Y-%m-%d')

    print(date_since)

    user_reading_query = f"""
        SELECT urs.user_id, urs.article_id, urs.start_time, urs.duration, a.*, ua.liked
        FROM user_reading_session AS urs 
        JOIN user AS u ON u.id = urs.user_id 
        JOIN article AS a ON a.id = urs.article_id 
        LEFT JOIN user_article as ua on ua.user_id = urs.user_id and ua.article_id = urs.article_id
        WHERE a.broken = 0 
        AND u.is_dev = FALSE 
        AND urs.article_id IS NOT NULL 
        AND urs.start_time >= '2023-03-26'
        AND urs.duration >= 30000 
        AND urs.duration <= 3600000 
        ORDER BY urs.user_id ASC
    """

    df = pd.read_sql(user_reading_query, engine)

    aggregated_df = df.groupby(['user_id', 'article_id']).agg({'duration': 'sum'}).reset_index()
    merged_df = pd.merge(aggregated_df, df.drop(columns=['duration']), on=['user_id', 'article_id'], how='left')
    merged_df = merged_df.drop_duplicates(subset=['user_id', 'article_id'], keep='first').reset_index()
    return merged_df
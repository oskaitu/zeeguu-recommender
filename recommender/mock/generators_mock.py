import random
from typing import Union
import pandas as pd

def __initial_session_setup(size) -> Union[pd.DataFrame, list]:
    '''
    Intial setup for all setup_session_... functions
    it returns a Union of users with id and name an empty list of sessions_data 
    '''
    users = pd.DataFrame()
    users['id'] = range(0, size)
    users['name'] = [f'user_{i}' for i in range(0, size)]
    sessions_data = []
    return users, sessions_data

def generate_articles_with_titles(size) -> pd.DataFrame:
    '''This mocks n articles in a DataFrame, only used by recommender.py when testing'''
    articles = pd.DataFrame()
    articles['id'] = range(0, size)
    articles['title'] = [f'article_{i}' for i in range(0, size)]
    articles['language_id'] = 1
    return articles

def setup_session_5_likes_range(num_users, num_items) -> pd.DataFrame:
    '''
    This function mocks a session where everyone likes 
    all articles within a range of 5 of their user ID
    Expect article 50, it is the worst article ever'''
    users, sessions_data = __initial_session_setup(num_users)
    for user_id in users['id']:
        min_article_id = user_id - 5
        if(min_article_id < 0):
            min_article_id = 0
        max_article_id = user_id + 5
        if(max_article_id > num_items-1):
            max_article_id = num_items
        liked_articles = [article_id for article_id in range(min_article_id, max_article_id)]        
        for article_id in liked_articles:
            if(article_id != 50):
                sessions_data.append({'user_id': user_id, 'article_id': article_id, 'expected_read': 1.0})
    sessions = pd.DataFrame(sessions_data)
    return sessions

def setup_session_2_categories(num_users,num_items) -> pd.DataFrame:
    '''
    This function mocks a session where everyone 
    likes articles within two different categories
    '''
    users, sessions_data = __initial_session_setup(num_users)
    split = int(num_items/2)
    for user_id in users['id']:
        if(user_id < int(num_users/2)):
            liked_articles = random.sample(range(0, split),3)
            for article_id in liked_articles:
                sessions_data.append({'user_id': user_id, 'article_id': article_id, 'expected_read': 1.0})
        else: 
            liked_articles = random.sample(range(split, num_items),3)
            for article_id in liked_articles:
                sessions_data.append({'user_id': user_id, 'article_id': article_id, 'expected_read': 1.0})
    sessions = pd.DataFrame(sessions_data)
    return sessions

def setup_sessions_4_categories_with_noise(num_users, num_items) -> pd.DataFrame:
    '''
    This function mocks a session where everyone likes articles within four different categories
    and some random spread 
    '''
    r = range(1,5)
    categories = 4 
    articlesplit = int(num_items/categories)
    m_max = 0
    m_less = 0
    users, sessions_data = __initial_session_setup(num_users)
    for n in r: 
        for user_id in users['id']:
            m_max= int(num_users/categories*n)
            if(user_id >= m_less and user_id < m_max):
                if(random.random() > 0.8 or user_id == 1):
                    liked_article_noise = random.sample(range(0, num_items),50)
                    if(liked_article_noise[0] not in range(articlesplit*n-articlesplit, articlesplit*n)):
                        sessions_data.append({'user_id': user_id, 'article_id': liked_article_noise[0], 'expected_read': 1.0})
                liked_articles = random.sample(range(articlesplit*n-articlesplit, articlesplit*n),random.randint(25,100))
                for article_id in liked_articles:
                    sessions_data.append({'user_id': user_id, 'article_id': article_id, 'expected_read': 1.0}) 
        m_less = m_max
    sessions = pd.DataFrame(sessions_data)
    return sessions


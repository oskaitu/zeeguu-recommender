from datetime import datetime, timedelta
from .mapper import Mapper
from .recommender_system import (
    RecommenderSystem,
)



class RecommenderSystemSingleton:
    __instance = None

    def get_recommender(self):
        data_since = datetime.now() - timedelta(days=365)
        if self.__instance is None:
            self.mapper = Mapper(data_since=data_since)
            self.num_users = self.mapper.num_users
            self.num_items = self.mapper.num_articles
            self.__instance = RecommenderSystem(
                sessions=None,
                num_users=self.num_users,
                num_items=self.num_items,
                mapper=self.mapper,
                data_since=data_since,
            )
        
        return self.__instance

    

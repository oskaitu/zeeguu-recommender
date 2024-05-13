from datetime import timedelta, datetime
from api.zeeguu.core.model import Language
from api.zeeguu.core.model import Article
from api.zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from elasticsearch import Elasticsearch
from api.zeeguu.core.content_recommender.elastic_recommender import _to_articles_from_ES_hits

def find_articles_like(recommended_articles_ids: 'list[int]', limit: int, article_age: int, language_id: int) -> 'list[Article]':
    es = Elasticsearch(ES_CONN_STRING)
    fields = ["content", "title"]
    language = Language.find_by_id(language_id)
    like_documents = [
        {"_index": ES_ZINDEX, "_id": str(doc_id) } for doc_id in recommended_articles_ids
    ]

    cutoff_date = datetime.now() - timedelta(days=article_age)

    mlt_query = {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [
                            {'match': {'language': language.name}}
                        ],
                        "should": {
                            "more_like_this": {
                                "fields": fields,
                                "like": like_documents,
                                "min_term_freq": 2,
                                "max_query_terms": 25,
                                "min_doc_freq": 5,
                                "min_word_length": 3
                            }
                        },
                        "filter": {
                            "bool": {
                                "must": [
                                    {
                                        "range": {
                                            "published_time": {
                                                "gte": cutoff_date.strftime('%Y-%m-%dT%H:%M:%S'),
                                                "lte": "now"
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }, "functions": [
                        {"gauss": {
                            "published_time": {
                                "origin": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                                "scale": "10d",
                                "offset": "4h",
                                "decay": 0.9
                            }
                        }}
                ],
                "score_mode": "sum"
            }
        }
    }

    res = es.search(index=ES_ZINDEX, body=mlt_query, size=limit)
    articles = _to_articles_from_ES_hits(res["hits"]["hits"])
    articles = [a for a in articles if a.broken == 0]
    return articles
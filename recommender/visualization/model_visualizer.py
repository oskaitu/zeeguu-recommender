import altair as alt
import sklearn
import sklearn.manifold
from recommender.utils.recommender_utils import filter_article_embeddings, get_resource_path
from typing import List

class ModelVisualizer:
    def __visualize_article_embeddings(self, model, articles, x, y, marked_articles):
        data = self.__tsne_article_embeddings(model, articles)

        data['is_marked'] = data['id'].isin(marked_articles)

        nearest = alt.selection_point(
            encodings=['x', 'y'], on='mouseover', nearest=True, empty=True)
        base = alt.Chart().mark_circle().encode(
            x=x,
            y=y,
            color=alt.condition('datum.is_marked', alt.value('red'), alt.value('blue')),
        ).properties(
            width=600,
            height=600,
        ).add_params(nearest)
        text = alt.Chart().mark_text(align='left', dx=5, dy=-5).encode(
            x=x,
            y=y,
            text=alt.condition(nearest, 'title', alt.value('')))
        return alt.hconcat(alt.layer(base, text), data=data)
    
    def __tsne_article_embeddings(self, model, articles):
        """Visualizes the article embeddings, projected using t-SNE with Cosine measure.
        Args:
            model: A MFModel object.
        """
        tsne = sklearn.manifold.TSNE(
            n_components=2, perplexity=40, metric='cosine', early_exaggeration=10.0,
            init='pca', verbose=True, n_iter=400)

        print('Running t-SNE...')
        filtered_embeddings = filter_article_embeddings(model.embeddings["article_id"], articles['id'])
        V_proj = tsne.fit_transform(filtered_embeddings)
        articles.loc[:,'x'] = V_proj[:, 0]
        articles.loc[:,'y'] = V_proj[:, 1]
        return articles
    
    def visualize_tsne_article_embeddings(self, model, articles, marked_articles):
        return self.__visualize_article_embeddings(model, articles, 'x', 'y', marked_articles).save(get_resource_path() + "article_embeddings.json")

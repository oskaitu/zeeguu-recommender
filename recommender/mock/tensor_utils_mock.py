import tensorflow as tf
import recommender.visualization.tensor_visualizer as tv


def build_mock_sparse_tensor(sessions,title, num_users, num_articles):
    '''
    A mocked version of a tensor that takes well aligned input and sends the tensor to a visualizer.
    and saves it in the resources folder.
    As well as generating recommendations 
    Args:
        sessions: a pd.dataframe with columns user_id, article_id, expected_read
        name: the filename to save the plot to. With no extension.
    '''
    # Sort the indices
    sessions = sessions.sort_values(['user_id', 'article_id'])
    indices = sessions[['user_id', 'article_id']].values
    values = sessions['expected_read'].values

    tensor = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[num_users, num_articles]
    )
    dense_tensor = tf.sparse.to_dense(tensor)
    # Print the tensor within TensorFlow session
    with tf.compat.v1.Session() as sess:
        tensor_value = sess.run(dense_tensor)
        tv.visualize_tensor(tensor_value, title)    
    return tensor
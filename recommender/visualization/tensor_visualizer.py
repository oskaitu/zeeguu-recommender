
import matplotlib.pyplot as plt
import os

from recommender.utils.recommender_utils import get_resource_path

def visualize_tensor(tensor, name):
    """
    Visualize a tensor as a heatmap.
    Args:
        tensor: a tensorflow session tf.compat.v1.Session()
        name: the filename to save the plot to.
    """

    name = os.path.join(get_resource_path(), name + '.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(tensor, cmap='inferno', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('Mock Tensor')
    plt.xlabel('Article ID')
    plt.ylabel('User ID')
    plt.grid(True)
    plt.savefig(name)
    
    print(f"Plot saved to {name}")

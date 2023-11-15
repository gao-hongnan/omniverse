```
import matplotlib.pyplot as plt
import numpy as np

def plot_vectors(vectors, colors, figsize=(8, 8), xlim=(-2, 15), ylim=(-2, 15)):
    """
    Plots vectors in 2D and their resultant vector.

    Parameters:
    vectors: List of tuples, where each tuple represents a vector as (start_x, start_y, direction_x, direction_y)
    colors: List of colors for each vector
    figsize: Size of the figure
    xlim, ylim: Limits for the plot's x and y axes
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot each vector
    for (start_x, start_y, direction_x, direction_y), color in zip(vectors, colors):
        ax.quiver(start_x, start_y, direction_x, direction_y, angles='xy', scale_units='xy', scale=1, color=color)
        # Annotation
        plt.text(start_x + direction_x / 2, start_y + direction_y / 2, f'({direction_x},{direction_y})', color=color, fontsize=12)

    # Resultant vector
    res_x = sum([direction_x for _, _, direction_x, _ in vectors])
    res_y = sum([direction_y for _, _, direction_y, _ in vectors])
    ax.quiver(0, 0, res_x, res_y, angles='xy', scale_units='xy', scale=1, color='green', width=0.005)
    plt.text(res_x / 2, res_y / 2, f'({res_x},{res_y})', color='green', fontsize=12)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.grid(True)
    plt.show()

# Example usage
vectors = [(0, 0, 4, 7), (0, 0, 8, 4)]  # (start_x, start_y, direction_x, direction_y)
colors = ['blue', 'red']
plot_vectors(vectors, colors)
```

import matplotlib.pyplot as plt
import numpy as np


class VectorPlotter:
    def __init__(self, figsize=(9, 9)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.vectors = []
        self.colors = []

    def add_label(self, vector, label):
        X, Y, _, _ = zip(*vector)
        self.ax.text(x=X[0], y=Y[0], s=label, fontsize=16)

    def set_title(self, title):
        self.ax.set_title(title, size=18)

    def annotate(self, x, y, text, arrow_props=None):
        self.ax.annotate(
            text, xy=(x, y), xytext=(x, y), arrowprops=arrow_props, fontsize=16
        )

    def add_vector(self, vector, color):
        self.vectors.append(vector)
        self.colors.append(color)

    def plot(self):
        for vec, color in zip(self.vectors, self.colors):
            X, Y, U, V = zip(*vec)
            self.ax.quiver(
                X,
                Y,
                U,
                V,
                angles="xy",
                scale_units="xy",
                color=color,
                scale=1,
                alpha=0.6,
            )
        self.ax.set_xlim([0, 15])
        self.ax.set_ylim([0, 15])
        self.ax.set_xlabel("x-axis", fontsize=16)
        self.ax.set_ylabel("y-axis", fontsize=16)
        self.ax.grid()
        plt.show()

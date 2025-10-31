from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np



class Display: 
    """Class to display the positions of the robot in 3D space.
    """
    def __init__(self):
        self.positions = []

    def setup_positions(self, positions: list):
        """Positions that will be displayed.

        Args:
            positions (list): list of positions
        """
        self.positions = positions

    def add_positions(self, positions: list):
        """Add positions to the list of positions.

        Args:
            positions (list): list of positions
        """
        self.positions += positions

    def display_positions(self):
        """Displace the positions in 3D space.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for position in self.positions:
            x, y, z = position[:3, 3]
            u, v, w = position[:3, 0], position[:3, 1], position[:3, 2]
            ax.quiver(x, y, z, u[0], u[1], u[2], color='r', length=0.1, normalize=True)
            ax.quiver(x, y, z, v[0], v[1], v[2], color='g', length=0.1, normalize=True)
            ax.quiver(x, y, z, w[0], w[1], w[2], color='b', length=0.1, normalize=True)
            ax.scatter(x, y, z, c='r', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

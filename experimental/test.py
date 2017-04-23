import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plyfile import PlyData, PlyElement


def get_points(plydata):
    return np.hstack(
        [plydata.elements[0].data['x'].reshape(-1, 1),
         plydata.elements[0].data['y'].reshape(-1, 1),
         plydata.elements[0].data['z'].reshape(-1, 1)],
    )


if __name__ == '__main__':

    point_arrays = []
    for name in os.listdir('data/bunny/data'):
        if name.endswith('.ply'):
            point_arrays.append(
                get_points(PlyData.read(os.path.join('data/bunny/data', name)))
            )

    total_data = np.vstack(point_arrays)


    point_matrix_sampled = total_data[
        np.random.randint(0, total_data.shape[0], 1000),
        :
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        point_matrix_sampled[:, 0],
        point_matrix_sampled[:, 2],
        point_matrix_sampled[:, 1],
    )

    plt.show()

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections  as mc
from mpl_toolkits.mplot3d import Axes3D

from plyfile import PlyData, PlyElement


def get_vertices(plydata):
    return np.hstack(
        [plydata.elements[0].data['x'].reshape(-1, 1),
         plydata.elements[0].data['y'].reshape(-1, 1),
         plydata.elements[0].data['z'].reshape(-1, 1)],
    )


def get_faces(plydata):
    return np.vstack([x[0] for x in plydata.elements[1].data])


def get_vertex_normals(plydata, vertices=None):
    if vertices is None:
        vertices = get_vertices(plydata)

    vertex_normals = np.zeros_like(vertices)
    faces = get_faces(plydata)

    A = vertices[faces[:, 0], :]
    B = vertices[faces[:, 1], :]
    C = vertices[faces[:, 2], :]

    normals = -np.cross(A - B, A - C)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    vertex_normals[faces[:, 0], :] += normals
    vertex_normals[faces[:, 1], :] += normals
    vertex_normals[faces[:, 2], :] += normals

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return vertex_normals / norms



if __name__ == '__main__':

    plydata = PlyData.read('data/bunny.ply')
    normals = get_vertex_normals(plydata)
    total_data = get_vertices(plydata)

    sample_indices = np.random.randint(0, total_data.shape[0], 3000)

    point_matrix_sampled = total_data[
        sample_indices,
        :
    ]

    increased = 0.01 * normals[
        sample_indices,
        :
    ] + point_matrix_sampled

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        point_matrix_sampled[:, 0],
        point_matrix_sampled[:, 2],
        point_matrix_sampled[:, 1],
        c='#FF5F29', marker='.'
    )

    # ax.scatter(
    #     increased[:, 0],
    #     increased[:, 2],
    #     increased[:, 1],
    #     c='#40FF00', marker='.', alpha=0.5
    # )

    # lines = []
    # for i in xrange(point_matrix_sampled.shape[0]):
    #     start = point_matrix_sampled[i]
    #     end = increased[i]
    #     ax.plot(
    #         [start[0], end[0]], [start[2], end[2]], [start[1], end[1]],
    #         c='#1CA7D6', alpha=0.5
    #     )

    ax.axis('off')
    ax.set_axis_bgcolor((0, 0, 0))

    plt.show()

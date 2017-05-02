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

    for i in xrange(faces.shape[0]):
        A = vertices[faces[i, 0], :]
        B = vertices[faces[i, 1], :]
        C = vertices[faces[i, 2], :]
<<<<<<< HEAD
        
        normals = -np.cross(A - B, A - C)
        #print normals
        #print faces
        normals = normals / np.linalg.norm(normals)#, axis=1, keepdims=True)
        
        vertex_normals[faces[i, 0], :] += normals
        vertex_normals[faces[i, 1], :] += normals
        vertex_normals[faces[i, 2], :] += normals

        #print vertex_normals
=======

        normals = np.cross(A - B, A - C)
        normals = normals / np.linalg.norm(normals)

        vertex_normals[faces[i, 0], :] += normals
        vertex_normals[faces[i, 1], :] += normals
        vertex_normals[faces[i, 2], :] += normals
>>>>>>> 78d872129a0b5885628a78a375659afcac35fae1

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return vertex_normals / norms



if __name__ == '__main__':

<<<<<<< HEAD
    plydata = PlyData.read('cube.ply')
=======
    plydata = PlyData.read('data/cube_good.ply')
>>>>>>> 78d872129a0b5885628a78a375659afcac35fae1
    normals = get_vertex_normals(plydata)
    total_data = get_vertices(plydata)

    print normals

<<<<<<< HEAD
    #sample_indices = np.random.randint(0, total_data.shape[0], 3000)
=======
    # sample_indices = np.random.randint(0, total_data.shape[0], 1000)
>>>>>>> 78d872129a0b5885628a78a375659afcac35fae1
    sample_indices = np.arange(0, total_data.shape[0])

    point_matrix_sampled = total_data[
        sample_indices,
        :
    ]

<<<<<<< HEAD
    increased = 0.1 * normals[
=======
    increased = 0.3 * normals[
>>>>>>> 78d872129a0b5885628a78a375659afcac35fae1
        sample_indices,
        :
    ] + point_matrix_sampled

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        point_matrix_sampled[:, 0],
        point_matrix_sampled[:, 2],
        point_matrix_sampled[:, 1],
        c='#FF5F29', marker='x'
    )

    ax.scatter(
        increased[:, 0],
        increased[:, 2],
        increased[:, 1],
        c='#40FF00', marker='.', alpha=0.5
    )

<<<<<<< HEAD
    ax.scatter(
         increased[:, 0],
         increased[:, 2],
         increased[:, 1],
         c='#40FF00', marker='.', alpha=0.5
    )

    lines = []
    for i in xrange(point_matrix_sampled.shape[0]):
         start = point_matrix_sampled[i]
         end = increased[i]
         ax.plot(
             [start[0], end[0]], [start[2], end[2]], [start[1], end[1]],
             c='#1CA7D6', alpha=0.5
         )
=======
    lines = []
    for i in xrange(point_matrix_sampled.shape[0]):
        start = point_matrix_sampled[i]
        end = increased[i]
        ax.plot(
            [start[0], end[0]], [start[2], end[2]], [start[1], end[1]],
            c='#1CA7D6', alpha=0.5
        )
>>>>>>> 78d872129a0b5885628a78a375659afcac35fae1

    ax.axis('off')
    ax.set_axis_bgcolor((0, 0, 0))

    plt.show()

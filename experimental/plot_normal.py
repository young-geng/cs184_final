import os
import argparse

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

        normals = np.cross(A - B, A - C)
        normals = normals / np.linalg.norm(normals)

        vertex_normals[faces[i, 0], :] += normals
        vertex_normals[faces[i, 1], :] += normals
        vertex_normals[faces[i, 2], :] += normals

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return vertex_normals / norms



if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--length', '-l', default=0.1, type=float, help='normal length'
    )

    parser.add_argument(
        '--sample', '-s', default=1, type=float, help='sample fraction of points'
    )

    parser.add_argument(
        '--no_line', '-n', action='store_true', default=False,
        help='Do not plot lines between vertices and normals'
    )


    parser.add_argument(
        'input', metavar='INPUT', type=str, nargs=1,
        help='input file'
    )


    args = parser.parse_args()

    plydata = PlyData.read(args.input[0])

    normals = get_vertex_normals(plydata)
    vertices = get_vertices(plydata)

    if args.sample < 1:
        sample_indices = np.random.choice(
            vertices.shape[0], int(vertices.shape[0] * args.sample),
            replace=False
        )

        vertices_sampled = vertices[
            sample_indices,
            :
        ]

        increased = args.length * normals[
            sample_indices,
            :
        ] + vertices_sampled
    else:
        vertices_sampled = vertices
        increased = args.length * normals + vertices


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        vertices_sampled[:, 0],
        vertices_sampled[:, 2],
        vertices_sampled[:, 1],
        c='#FF5F29', marker='.'
    )

    if args.length > 0:

        ax.scatter(
             increased[:, 0],
             increased[:, 2],
             increased[:, 1],
             c='#40FF00', marker='.', alpha=0.5
        )

        if not args.no_line:
            lines = []
            for i in xrange(vertices_sampled.shape[0]):
                 start = vertices_sampled[i]
                 end = increased[i]
                 ax.plot(
                     [start[0], end[0]], [start[2], end[2]], [start[1], end[1]],
                     c='#1CA7D6', alpha=0.5
                 )


    coord_min = min(np.min(increased), np.min(vertices_sampled))
    coord_max = max(np.max(increased), np.max(vertices_sampled))
    coord_extent = coord_max - coord_min

    coord_min -= 0.0 * coord_extent
    coord_max += 0.0 * coord_extent

    ax.set_xlim3d([coord_min, coord_max])
    ax.set_ylim3d([coord_min, coord_max])
    ax.set_zlim3d([coord_min, coord_max])

    ax.axis('off')
    ax.set_axis_bgcolor((0, 0, 0))

    plt.show()

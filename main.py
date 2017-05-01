import numpy as np

from plyfile import PlyData, PlyElement

from utils import *

from ball_pivot import *


if __name__ == '__main__':

    plydata = PlyData.read('data/bunny.ply')

    normals = get_vertex_normals(plydata)
    vertices = get_vertices(plydata)

    sample_indices = np.random.randint(0, vertices.shape[0], 1000)

    vs = VertexSet(vertices[sample_indices, :], normals[sample_indices, :])
    # vs = build_vertex_set_ply(plydata)

    mesh = pivot_ball(vs, 0.003)
    write_collada(vs.vertices, vs.normals, np.array(mesh.faces.keys()), 'temp/bunny2.dae')

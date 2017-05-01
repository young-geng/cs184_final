import numpy as np

from plyfile import PlyData, PlyElement

from utils import *

from ball_pivot import *


if __name__ == '__main__':

    plydata = PlyData.read('data/dart.ply')
    vs = build_vertex_set_ply(plydata)

    mesh = pivot_ball(vs, 8.6375)
    write_collada(vs.vertices, vs.normals, np.array(mesh.faces.keys()), 'temp/dart.dae')

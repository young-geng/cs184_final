import numpy as np

from plyfile import PlyData, PlyElement

from utils import *

from ball_pivot import *


if __name__ == '__main__':

    plydata = PlyData.read('data/bunny.ply')
    vs = build_vertex_set_ply(plydata)

    print seed_triangle(0.0003, vs)

import argparse

import numpy as np

from plyfile import PlyData, PlyElement

from utils import *

from ball_pivot import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--radius', '-r', required=True, type=float, nargs='+',
        help='radius'
    )

    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='input file'
    )

    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='output file'
    )
    args = parser.parse_args()

    plydata = PlyData.read(args.input)

    vs = build_vertex_set_ply(plydata)

    mesh = pivot_ball(vs, args.radius)
    write_collada(vs.vertices, vs.normals, np.array(mesh.faces.keys()), args.output)

import numpy as np

from collada import *


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


def write_collada(vertices, normals, triangle_indices, fname):
    mesh = Collada()
    effect = material.Effect("effect0", [], "phong", diffuse=(1,0,0), specular=(0,1,0))
    mat = material.Material("material0", "mymaterial", effect)
    mesh.effects.append(effect)
    mesh.materials.append(mat)

    vert_floats = np.array(vertices).flatten()
    normal_floats = np.array(normals).flatten()

    vert_src = source.FloatSource("cubeverts-array", np.array(vert_floats), ('X', 'Y', 'Z'))
    normal_src = source.FloatSource("cubenormals-array", np.array(normal_floats), ('X', 'Y', 'Z'))

    geom = geometry.Geometry(mesh, "geometry0", "mycube", [vert_src, normal_src])

    input_list = source.InputList()
    input_list.addInput(0, 'VERTEX', "#cubeverts-array")
    input_list.addInput(1, 'NORMAL', "#cubenormals-array")

    indices = np.array(triangle_indices).flatten()


    triset = geom.createTriangleSet(indices, input_list, "materialref")
    geom.primitives.append(triset)
    mesh.geometries.append(geom)

    matnode = scene.MaterialNode("materialref", mat, inputs=[])
    geomnode = scene.GeometryNode(geom, [matnode])
    node = scene.Node("node0", children=[geomnode])

    myscene = scene.Scene("myscene", [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene
    mesh.write(fname)



if __name__ == '__main__':

    vert_floats = np.array([[-50,  50,  50],
                            [ 50,  50,  50],
                            [-50, -50,  50],
                            [ 50, -50,  50],
                            [-50,  50, -50],
                            [ 50,  50, -50],
                            [-50, -50, -50],
                            [ 50, -50, -50]])

    normal_floats = np.array([[ 0,  0,  1],
                              [ 0,  0,  1],
                              [ 0,  0,  1],
                              [ 0,  0,  1],
                              [ 0,  1,  0],
                              [ 0,  1,  0],
                              [ 0,  1,  0],
                              [ 0,  1,  0],
                              [ 0, -1,  0],
                              [ 0, -1,  0],
                              [ 0, -1,  0],
                              [ 0, -1,  0],
                              [-1,  0,  0],
                              [-1,  0,  0],
                              [-1,  0,  0],
                              [-1,  0,  0],
                              [ 1,  0,  0],
                              [ 1,  0,  0],
                              [ 1,  0,  0],
                              [ 1,  0,  0],
                              [ 0,  0, -1],
                              [ 0,  0, -1],
                              [ 0,  0, -1],
                              [ 0,  0, -1]])

    indices = np.array([[ 0,  0,  2],
                        [ 1,  3,  2],
                        [ 0,  0,  3],
                        [ 2,  1,  3],
                        [ 0,  4,  1],
                        [ 5,  5,  6],
                        [ 0,  4,  5],
                        [ 6,  4,  7],
                        [ 6,  8,  7],
                        [ 9,  3, 10],
                        [ 6,  8,  3],
                        [10,  2, 11],
                        [ 0, 12,  4],
                        [13,  6, 14],
                        [ 0, 12,  6],
                        [14,  2, 15],
                        [ 3, 16,  7],
                        [17,  5, 18],
                        [ 3, 16,  5],
                        [18,  1, 19],
                        [ 5, 20,  7],
                        [21,  6, 22],
                        [ 5, 20,  6],
                        [22,  4, 23]])


    # import pdb; pdb.set_trace()
    write_collada(vert_floats, normal_floats, indices, 'temp/cube.dae')

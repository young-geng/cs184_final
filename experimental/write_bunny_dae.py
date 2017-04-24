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


def get_face_normals(plydata):
    vertices = get_vertices(plydata)
    faces = get_faces(plydata)

    A = vertices[faces[:, 0], :]
    B = vertices[faces[:, 1], :]
    C = vertices[faces[:, 2], :]

    normals = -np.cross(A - B, A - C)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


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
    vertices = np.array(vertices)
    normals = np.array(normals)
    triangle_indices = np.array(triangle_indices, dtype=np.int32)

    assert len(vertices.shape) == 2 and vertices.shape[1] == 3
    assert vertices.shape == normals.shape
    assert len(triangle_indices.shape) == 2 and triangle_indices.shape[1] == 3


    mesh = Collada()
    effect = material.Effect("effect0", [], "phong", diffuse=(1,0,0), specular=(0,1,0))
    mat = material.Material("material0", "mymaterial", effect)
    mesh.effects.append(effect)
    mesh.materials.append(mat)

    vert_floats = vertices.flatten()
    normal_floats = normals.flatten()

    vert_src = source.FloatSource("cubeverts-array", np.array(vert_floats), ('X', 'Y', 'Z'))
    normal_src = source.FloatSource("cubenormals-array", np.array(normal_floats), ('X', 'Y', 'Z'))

    geom = geometry.Geometry(mesh, "geometry0", "mycube", [vert_src, normal_src])

    input_list = source.InputList()
    input_list.addInput(0, 'VERTEX', "#cubeverts-array")
    input_list.addInput(1, 'NORMAL', "#cubenormals-array")

    indices = np.hstack([triangle_indices, triangle_indices]).flatten()


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

    plydata = PlyData.read('data/bunny.ply')
    normals = get_vertex_normals(plydata)
    vertices = get_vertices(plydata)
    faces = get_faces(plydata)


    write_collada(vertices, normals, faces, 'temp/bunny.dae')

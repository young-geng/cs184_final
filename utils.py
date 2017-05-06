import numpy as np

from plyfile import PlyData, PlyElement

import pyflann

import collada


class VertexSet(object):

    def __init__(self, vertices, normals):
        assert vertices.shape == normals.shape
        self._vertices = vertices.astype(np.float64)
        self._normals = normals.astype(np.float64)

        self.flann = pyflann.FLANN()
        self.flann_index = self.flann.build_index(
            vertices, algorithm='kdtree_simple', target_precision=1, checks=1024
        )

    def neighbor_search(self, p, k, max_results=32):
        return self.flann.nn_index(p, k, checks=max_results)

    def radius_search(self, p, r, max_results=1024):
        return self.flann.nn_radius(p, r*r, checks=max_results)

    def __len__(self):
        return self.vertices.shape[0]

    @property
    def vertices(self):
        return self._vertices

    @property
    def normals(self):
        return self._normals

    def __getitem__(self, index):
        return self.vertices[index]

    def __iter__(self):
        return iter(self.vertices)


def build_vertex_set_ply(plydata, invert_normal=False):
    normals = get_vertex_normals(plydata)
    if invert_normal:
        normals = -normals
    vertices = get_vertices(plydata)
    return VertexSet(vertices, normals)


def get_vertices(plydata):
    return np.hstack(
        [plydata.elements[0].data['x'].reshape(-1, 1),
         plydata.elements[0].data['y'].reshape(-1, 1),
         plydata.elements[0].data['z'].reshape(-1, 1)],
    ).astype(np.float64)


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


def write_collada(vertices, normals, triangle_indices, fname):
    vertices = np.array(vertices)
    normals = np.array(normals)
    triangle_indices = np.array(triangle_indices, dtype=np.int32)


    for i in xrange(triangle_indices.shape[0]):
        k1, k2, k3 = triangle_indices[i, :]
        x, y, z = vertices[k1], vertices[k2], vertices[k3]
        nx = normals[k1]
        if np.dot(np.cross(y - x, x - z), nx) < 0:
            triangle_indices[i, :] = np.array([k3, k2, k1], dtype=np.int32)


    assert len(vertices.shape) == 2 and vertices.shape[1] == 3
    assert vertices.shape == normals.shape
    assert len(triangle_indices.shape) == 2 and triangle_indices.shape[1] == 3


    mesh = collada.Collada()
    effect = collada.material.Effect("effect0", [], "phong", diffuse=(1,0,0), specular=(0,1,0))
    mat = collada.material.Material("material0", "mymaterial", effect)
    mesh.effects.append(effect)
    mesh.materials.append(mat)

    vert_floats = vertices.flatten()
    normal_floats = normals.flatten()

    vert_src = collada.source.FloatSource("cubeverts-array", np.array(vert_floats), ('X', 'Y', 'Z'))
    normal_src = collada.source.FloatSource("cubenormals-array", np.array(normal_floats), ('X', 'Y', 'Z'))

    geom = collada.geometry.Geometry(mesh, "geometry0", "mycube", [vert_src, normal_src])

    input_list = collada.source.InputList()
    input_list.addInput(0, 'VERTEX', "#cubeverts-array")
    input_list.addInput(1, 'NORMAL', "#cubenormals-array")

    indices = np.hstack([triangle_indices, triangle_indices]).flatten()


    triset = geom.createTriangleSet(indices, input_list, "materialref")
    geom.primitives.append(triset)
    mesh.geometries.append(geom)

    matnode = collada.scene.MaterialNode("materialref", mat, inputs=[])
    geomnode = collada.scene.GeometryNode(geom, [matnode])
    node = collada.scene.Node("node0", children=[geomnode])

    myscene = collada.scene.Scene("myscene", [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene
    mesh.write(fname)

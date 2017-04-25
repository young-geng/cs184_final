import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plyfile import PlyData, PlyElement

class Vertex:

    def __init__(self, pos, norm):
        self.pos = pos
        self.norm = norm
        self.orph = True
        self.edges = []

    def bind(self, edge):
        if len(self.edges) == 0:
            self.edges = [(edge, 0)]
            return
        vec = edge.vec_to(self)
        dir = np.cross(self.norm, vec)
        vy = self.edges[0][0].vec_to(self)
        diry = np.cross(self.norm, vy)
        dirx = np.cross(self.norm, diry)
        key = np.dot(dir, diry)/(np.linalg.norm(dir)*np.linalg.norm(diry))
        sign = np.dot(dir, dirx)/(np.linalg.norm(dir)*np.linalg.norm(dirx))
        if sign < 0:
            key += 3
        else:
            key = 1-key
        self.edges.append((edge, key))
        self.edges = sorted(self.edges, key=lambda (e, k): k)
        self.orph = False

    def next(self, edge):
        if self.orph:
            return
        for i, (e, k) in enumerate(self.edges):
            if e == edge:
                if i == len(self.edges)-1:
                    return self.edges[0][0]
                else:
                    return self.edges[i+1][0]

    def neighbors(self):
        if self.orph:
            return
        for (e, k) in self.edges:
            yield e.another(self)

    def connected(self, v):
        return (v in self.neighbors())

    def find_edge(self, v):
        for (e, k) in self.edges:
            if e.another(self) == v:
                return e
        return None

    def faces(self):
        visited = []
        for (e, k) in self.edges:
            for f in e.faces:
                if not f in visited:
                    yield f
                    visited.append(f)
        
class Edge:

    def __init__(self, v1, v2):
        self.v = [v1, v2]
        v1.bind(self)
        v2.bind(self)
        self.faces = []

    def another(self, vertex):
        if vertex == self.v[0]:
            return self.v[1]
        else:
            return self.v[0]

    def vec_to(self, vertex):
        if vertex == self.v[0]:
            return self.v[0].pos - self.v[1].pos
        else:
            return self.v[1].pos - self.v[0].pos

    def add_face(self, face):
        self.faces.append(face)

    def dual_face(self, face):
        if len(self.faces) <= 1:
            return None
        else:
            if self.faces[0] == face:
                return self.faces[1]
            else:
                return self.faces[0]

class Face:

    def __init__(self, v1, v2, v3):
        self.v = [v1, v2, v3]
        self.e = []
        if v1.connected(v2):
            self.add_edge(v1.find_edge(v2))
        else:
            self.add_edge(Edge(v1, v2))
        if v3.connected(v2):
            self.add_edge(v3.find_edge(v2))
        else:
            self.add_edge(Edge(v3, v2))
        if v1.connected(v3):
            self.add_edge(v1.find_edge(v3))
        else:
            self.add_edge(Edge(v1, v3))

    def add_edge(self, edge):
        self.e.append(edge)
        edge.add_face(self)

    def adjacent_faces(self):
        for edge in self.e:
            yield edge.dual_face(self)

class Mesh:

    def __init__(self, points):
        self.points = points
        self.faces = []

    def connect(self, idx):
        i1, i2, i3 = idx
        self.faces.append(Face(self.points[i1], self.points[i2], self.points[i3]))


if __name__ == '__main__':
    from test import get_vertices, get_faces, get_vertex_normals
    plydata = PlyData.read('data/bunny.ply')
    normals = get_vertex_normals(plydata)
    verts = get_vertices(plydata)
    faces = get_faces(plydata)
    pts = []
    for i in xrange(len(verts)):
        pts.append(Vertex(verts[i], normals[i]))
    mesh = Mesh(pts)
    mesh.connect([0, 1, 2])

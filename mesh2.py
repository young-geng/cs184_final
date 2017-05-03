import numpy as np


def sorted_tuple(*args):
    return tuple(sorted(args))

def edges_of_faces(i, j, k):
    for t in [(i, j), (j, k), (i, k)]:
        yield sorted_tuple(*t)

def is_unique(*args):
    return len(args) == len(set(args))


class Mesh(object):

    def __init__(self):

        self._vertices = {}
        self._edges = {}
        self._faces = {}
        self._boundary_edges = {}

        self._edges_of_vertex = {}
        self._faces_of_vertex = {}
        self._faces_of_edge = {}

    def add_vertex(self, *args):
        args = list(args)
        for i in args:
            if i in self.vertices:
                continue
            self.vertices[i] = None
            self.edges_of_vertex[i] = []
            self.faces_of_vertex[i] = []

    def add_edge(self, *args):
        args = list(args)
        while len(args) > 0:
            i = args.pop(0)
            j = args.pop(0)

            t = sorted_tuple(i, j)

            assert i in self.vertices and j in self.vertices
            assert is_unique(i, j)

            if t in self.edges:
                continue

            self.edges[t] = None
            self.edges_of_vertex[i].append(t)
            self.edges_of_vertex[j].append(t)
            self.faces_of_edge[t] = []

    def add_face(self, *args):
        args = list(args)
        while len(args) > 0:
            i = args.pop(0)
            j = args.pop(0)
            k = args.pop(0)
            t = sorted_tuple(i, j, k)

            assert i in self.vertices and j in self.vertices and k in self.vertices
            assert is_unique(i, j, k)
            for edge in edges_of_faces(i, j, k):
                assert edge in self.edges

            if t in self.faces:
                raise ValueError('Face already in mesh')
                continue

            self.faces[t] = None

            self.faces_of_vertex[i].append(t)
            self.faces_of_vertex[j].append(t)
            self.faces_of_vertex[k].append(t)

            for edge in edges_of_faces(i, j, k):
                self.faces_of_edge[edge].append(t)

    def is_edge(self, i, j):
        return sorted_tuple(i, j) in self.edges

    def is_face(self, i, j, k):
        return sorted_tuple(i, j, k) in self.faces

    def is_vertex(self, i):
        return i in self.vertices

    def is_inner_vertex(self, i):
        if len(self.faces_of_vertex[i]) == 0:
            return False
        for edge in self.edges_of_vertex[i]:
            if self.is_boundary(*edge):
                return False
        return True

    def is_boundary(self, i, j):
        return sorted_tuple(i, j) in self.boundary_edges

    def set_boundary(self, i, j):
        t = sorted_tuple(i, j)
        assert i in self.vertices and j in self.vertices
        assert is_unique(i, j)
        assert t in self.edges
        self.boundary_edges[t] = None

    def clear_boundary_edges(self):
        self._boundary_edges = {}

    @property
    def vertices(self):
        return self._vertices

    @property
    def edges(self):
        return self._edges

    @property
    def faces(self):
        return self._faces

    @property
    def edges_of_vertex(self):
        return self._edges_of_vertex

    @property
    def faces_of_vertex(self):
        return self._faces_of_vertex

    @property
    def faces_of_edge(self):
        return self._faces_of_edge

    @property
    def boundary_edges(self):
        return self._boundary_edges

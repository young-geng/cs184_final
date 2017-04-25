import numpy as np

from utils import *

FLOAT_EPS = np.finfo(np.float32).eps

def ball_compatible(p, q, s, r, vertex_set):

    A = vertex_set[p]
    B = vertex_set[q]
    C = vertex_set[s]

    na = vertex_set.normals[p]
    nb = vertex_set.normals[q]
    nc = vertex_set.normals[s]


    sq = lambda x: x ** 2

    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)

    rc2 = (sq(a) * sq(b) * sq(c)) / (a + b + c) / (b + c - a) / (c + a - b) / (a + b - c)

    H_bary = np.array([
        sq(a) * (sq(b) + sq(c) - sq(a)),
        sq(b) * (sq(c) + sq(a) - sq(b)),
        sq(c) * (sq(a) + sq(b) - sq(c))
    ])

    H_bary = H_bary / np.sum(H_bary)
    H = H_bary[0] * A + H_bary[1] * B + H_bary[2] * C
    n = np.cross(A - B, A - C)
    n = n / np.linalg.norm(n)
    if np.dot(n, na) + np.dot(n, nb) + np.dot(n, nc) < 0:
        n = - n

    if sq(r) - rc2 <= 0:
        return False

    O = H + np.sqrt(sq(r) - rc2) * n

    return len(vertex_set.radius_search(O, np.linalg.norm(O - A))[0]) <= 3


def seed_triangle(radius, vertex_set):

    for p, _ in enumerate(vertex_set):
        neighbor_indices, _ = vertex_set.radius_search(vertex_set[p], radius * 2, 20)

        for j in xrange(len(neighbor_indices)):
            if neighbor_indices[j] == p:
                continue
            for k in xrange(j + 1, len(neighbor_indices)):
                if neighbor_indices[k] == p:
                    continue

                q = neighbor_indices[j]
                s = neighbor_indices[k]

                if ball_compatible(p, q, s, radius, vertex_set):
                    return np.array([p, q, s], dtype=np.int32)
    return None

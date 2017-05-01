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

    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)

    rc2 = (
        (np.square(a) * np.square(b) * np.square(c))
        / (a + b + c) / (b + c - a) / (c + a - b) / (a + b - c)
    )

    H_bary = np.array([
        np.square(a) * (np.square(b) + np.square(c) - np.square(a)),
        np.square(b) * (np.square(c) + np.square(a) - np.square(b)),
        np.square(c) * (np.square(a) + np.square(b) - np.square(c))
    ])

    H_bary = H_bary / np.sum(H_bary)
    H = H_bary[0] * A + H_bary[1] * B + H_bary[2] * C
    n = np.cross(A - B, A - C)
    n = n / np.linalg.norm(n)
    if np.dot(n, na) + np.dot(n, nb) + np.dot(n, nc) < 0:
        n = - n

    if np.square(r) - rc2 <= 0:
        return False

    O = H + np.sqrt(np.square(r) - rc2) * n

    if len(vertex_set.radius_search(O, np.linalg.norm(O - A))[0]) <= 3:
        return O
    else:
        return None


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

def is_inner_vertex(i, mesh, edge_front):
    if not mesh.is_inner_vertex(i):
        return False
    for e in mesh.edges_of_vertex[i]:
        if e in edge_front:
            return False
    return True

def find_candidate(i, j, vertex_set, radius, mesh, edge_front):

    t = sorted_tuple(i, j)
    p, q, s = mesh.faces_of_edge[t][0]
    O = ball_compatible(p, q, s, radius, vertex_set)
    A = vertex_set[i]
    B = vertex_set[j]
    m = (A + B) / 2
    new_radius = np.linalg.norm(m - O) + radius
    theta_min = 2*np.pi
    idx, dis = vertex_set.radius_search(m, new_radius)
    candidate = None
    for v in idx:
        if is_inner_vertex(v, mesh, edge_front):
            continue
        if not ball_compatible(v, i, j, radius, vertex_set):
            continue
        o = ball_compatible(v, i, j, radius, vertex_set)
        theta = np.arccos(np.dot(m - O, m - o) / np.linalg.norm(m - O) / np.linalg.norm(m - o))
        if theta < theta_min:
            candidate = v
            theta_min = theta
    return candidate

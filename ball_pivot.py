import numpy as np

from utils import *

from mesh2 import *

FLOAT_EPS = np.finfo(np.float32).eps

def ball_compatible(p, q, s, r, vertex_set):

    p, q, s = sorted_tuple(p, q, s)

    A = vertex_set[p]
    B = vertex_set[q]
    C = vertex_set[s]

    if np.linalg.norm(A - B) < np.finfo(float).eps \
       or np.linalg.norm(A - C) < np.finfo(float).eps \
       or np.linalg.norm(B - C) < np.finfo(float).eps:

       return None

    na = vertex_set.normals[p]
    nb = vertex_set.normals[q]
    nc = vertex_set.normals[s]

    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)

    if a + b == c or a + c == b or b + c == a:
        return None

    rc2 = (
        (np.square(a) * np.square(b) * np.square(c))
        / (a + b + c) / (b + c - a) / (c + a - b) / (a + b - c)
    )

    H_bary = np.array([
        np.square(a) * (np.square(b) + np.square(c) - np.square(a)),
        np.square(b) * (np.square(c) + np.square(a) - np.square(b)),
        np.square(c) * (np.square(a) + np.square(b) - np.square(c))
    ])

    if np.sum(H_bary) < FLOAT_EPS:
        return None

    H_bary = H_bary / np.sum(H_bary)
    H = H_bary[0] * A + H_bary[1] * B + H_bary[2] * C
    n = np.cross(A - B, A - C)
    n = n / np.linalg.norm(n)
    if np.dot(n, na) + np.dot(n, nb) + np.dot(n, nc) < 0:
        n = - n

    if np.square(r) - rc2 <= 0:
        return None

    O = H + np.sqrt(np.square(r) - rc2) * n

    #print p, q, s, len(vertex_set.radius_search(O, r-FLOAT_EPS)[0]), "here!"

    s = set(vertex_set.radius_search(O, r)[0]) - set([p, q, s])
    if len(s) == 0:
        return O
    else:
        return None

    #if len(vertex_set.radius_search(O, r)[0]) == 3:
    #    return O
    #else:
    #    return None


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

                if ball_compatible(p, q, s, radius, vertex_set) is not None:
                    return int(p), int(q), int(s)
    return None

def is_inner_vertex(i, mesh, edge_front):
    if not mesh.is_vertex(i) or not mesh.is_inner_vertex(i):
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
    theta_min = 2 * np.pi
    idx, dis = vertex_set.radius_search(m, new_radius)
    candidate = None
    for v in idx:
        if v == i or v == j:
            continue
        if mesh.is_face(v, i, j):
            continue
        if is_inner_vertex(v, mesh, edge_front):
            continue
        o = ball_compatible(v, i, j, radius, vertex_set)
        if o is None:
            continue
        va = -(m - O) / np.linalg.norm(m - O)
        vb = -(m - o) / np.linalg.norm(m - o)
        theta = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        #theta = np.arccos(np.dot(m - O, m - o) / np.linalg.norm(m - O) / np.linalg.norm(m - o))
        if np.dot(A - B, np.cross(va, vb)) < 0:
            theta = 2 * np.pi - theta
        if theta < theta_min:# and abs(theta) > FLOAT_EPS:
            candidate = v
            theta_min = theta
    return candidate


def pivot_ball(vertex_set, radius):
    s0, s1, s2 = tuple(seed_triangle(radius, vertex_set))
    print s0, s1, s2
    mesh = Mesh()
    mesh.add_vertex(s0, s1, s2)
    mesh.add_edge(s0, s1, s1, s2, s0, s2)
    mesh.add_face(s0, s1, s2)

    edge_front = [
        sorted_tuple(s0, s1),
        sorted_tuple(s1, s2),
        sorted_tuple(s0, s2)
    ]

    total_faces = 0

    while len(edge_front) > 0:
        i, j = edge_front.pop(0)
        print 'Processing edge: {}, {}'.format(i, j)
        if mesh.is_boundary(i, j) or len(mesh.faces_of_edge[sorted_tuple(i, j)]) == 2:
            continue

        v = find_candidate(i, j, vertex_set, radius, mesh, edge_front)

        if v is None:
            mesh.set_boundary(i, j)
            continue

        # import pdb; pdb.set_trace()

        mesh.add_vertex(v)
        mesh.add_edge(i, v, j, v)
        mesh.add_face(i, j, v)
        total_faces += 1
        print 'Faces: {}, [{}, {}, {}]'.format(total_faces, i, j, v)

        es = sorted_tuple(i, v)
        et = sorted_tuple(j, v)

        if len(mesh.faces_of_edge[es]) != 2:
            edge_front.append(es)

        if len(mesh.faces_of_edge[et]) != 2:
            edge_front.append(et)

    return mesh

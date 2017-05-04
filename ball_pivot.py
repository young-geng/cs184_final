import numpy as np

from utils import *

from mesh2 import *

FLOAT_EPS = np.finfo(np.float32).eps
DOUBLE_EPS = np.finfo(np.float64).eps
BALL_EPS = 1e-4


def normalize_vector(v):
    return v / np.linalg.norm(v)

def project_vector(u, v):
    return np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v) * v


def ball_compatible(p, q, s, r, vertex_set, check_empty=True):

    p, q, s = sorted_tuple(p, q, s)

    A = vertex_set[p]
    B = vertex_set[q]
    C = vertex_set[s]

    if np.linalg.norm(A - B) < FLOAT_EPS \
       or np.linalg.norm(A - C) < FLOAT_EPS \
       or np.linalg.norm(B - C) < FLOAT_EPS:

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

    if np.abs(np.sum(H_bary)) < 100 * DOUBLE_EPS:
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

    if not check_empty:
        return O

    distances = vertex_set.radius_search(O, r)[1]
    if len(distances) == 0 or r - np.sqrt(np.min(distances)) < BALL_EPS:
        return O
    else:
        return None


def seed_triangle(radius, vertex_set):
    rand_indices = np.random.permutation(range(len(vertex_set)))
    for idx, p in enumerate(rand_indices):
        if idx % 50 == 0:
            print 'trying vertex {}, tried {} / {}'.format(p, idx, len(vertex_set))
        neighbor_indices, _ = vertex_set.radius_search(vertex_set[p], radius * 2, 1024)

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


def calculate_theta(A, B, C, nA, nB, nC, old_O, new_O):
    m = 0.5 * (A + B)
    cos_theta = np.dot(new_O - m, old_O - m) / np.linalg.norm(new_O - m) / np.linalg.norm(old_O - m)
    raw_theta = np.arccos(np.clip(cos_theta, -1, 1))

    normal_ABC_average = A + B + C
    normal_ABC = np.cross(A - B, C - B)
    normal_ABC = normal_ABC / np.linalg.norm(normal_ABC)
    if np.dot(normal_ABC, normal_ABC_average) < 0:
        normal_ABC = -normal_ABC

    normal_OAB = np.cross(B - old_O, A - old_O)
    normal_OAB = normal_OAB / np.linalg.norm(normal_OAB)
    if np.dot(normal_OAB, normal_ABC) < 0:
        normal_OAB = -normal_OAB

    if np.dot(normal_OAB, new_O - m) < 0:
        theta = 2 * np.pi - raw_theta
    else:
        theta = raw_theta

    return theta

def intersect(vertex_set, e1, e2):

    v1 = vertex_set[e1[0]]
    v2 = vertex_set[e1[1]]
    u1 = vertex_set[e2[0]]
    u2 = vertex_set[e2[1]]
    fn = np.cross(u1 - v1, u2 - v1)
    if abs(np.dot(fn, v2 - v1)) > FLOAT_EPS:
        return False
    a1 = np.arccos(np.clip(np.dot(u2 - v1, v2 - v1)/ np.linalg.norm(u2 - v1) / np.linalg.norm(v2 - v1), -1, 1))
    a2 = np.arccos(np.clip(np.dot(u1 - v1, v2 - v1)/ np.linalg.norm(u1 - v1) / np.linalg.norm(v2 - v1), -1, 1))
    a = np.arccos(np.clip(np.dot(u2 - v1, u1 - v1)/ np.linalg.norm(u2 - v1) / np.linalg.norm(u1 - v1), -1, 1))
    b1 = np.arccos(np.clip(np.dot(v2 - u1, u2 - u1)/ np.linalg.norm(v2 - u1) / np.linalg.norm(u2 - u1), -1, 1))
    b2 = np.arccos(np.clip(np.dot(v1 - u1, u2 - u1)/ np.linalg.norm(v1 - u1) / np.linalg.norm(u2 - u1), -1, 1))
    b = np.arccos(np.clip(np.dot(v2 - u1, v1 - u1)/ np.linalg.norm(v2 - u1) / np.linalg.norm(v1 - u1), -1, 1))
    if abs(a - (a1 + a2)) <= FLOAT_EPS and abs(b - (b1 + b2)) <= FLOAT_EPS:
        return True
    return False

def find_candidate(i, j, vertex_set, radius, mesh, edge_front):

    t = sorted_tuple(i, j)
    p, q, s = mesh.faces_of_edge[t][0]
    O = ball_compatible(p, q, s, radius, vertex_set, check_empty=False)
    A = vertex_set[i]
    B = vertex_set[j]
    k = (set([p, q, s]) - set([i ,j])).pop()
    C = vertex_set[k]
    na = vertex_set.normals[i]
    nb = vertex_set.normals[j]
    nc = vertex_set.normals[k]

    # if np.dot(np.cross(B - A, A - C), na) < 0:
    #     A, B = B, A

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

        skip = False
        for edge in mesh.edges_of_vertex[i]:
            if not v in edge and not j in edge:
                if intersect(vertex_set, sorted_tuple(v, j), edge):
                    skip = True
                    break
        for edge in mesh.edges_of_vertex[j]:
            if not v in edge and not i in edge:
                if intersect(vertex_set, sorted_tuple(v, i), edge):
                    skip = True
                    break
        if skip:
            continue

        if is_inner_vertex(v, mesh, edge_front):
            continue
        o = ball_compatible(v, i, j, radius, vertex_set)
        if o is None:
            continue
        # va = -(m - O) / np.linalg.norm(m - O)
        # vb = -(m - o) / np.linalg.norm(m - o)
        # theta = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        # #theta = np.arccos(np.dot(m - O, m - o) / np.linalg.norm(m - O) / np.linalg.norm(m - o))
        # if np.dot(A - B, np.cross(va, vb)) < 0:
        #     theta = 2 * np.pi - theta

        theta = calculate_theta(A, B, C, na, nb, nc, O, o)

        face_vector_1 = vertex_set[v] - m
        face_vector_2 = C - m

        v_perpendicular = normalize_vector(face_vector_1 - project_vector(face_vector_1, A - B))
        C_perpendicular = normalize_vector(face_vector_2 - project_vector(face_vector_2, A - B))


        faces_dot = np.dot(
            v_perpendicular,
            C_perpendicular
        )

        if faces_dot > 1 - 1e-1:
            continue

        if theta < theta_min:# and abs(theta) > FLOAT_EPS:
            candidate = v
            theta_min = theta
    return candidate

def generate_mesh(mesh, edge_front, radius, vertex_set, total_faces):

    while len(edge_front) > 0:
        i, j = edge_front.pop(0)
        #print 'Processing edge: {}, {}'.format(i, j)

        if mesh.is_boundary(i, j) or len(mesh.faces_of_edge[sorted_tuple(i, j)]) >= 2:
            continue

        v = find_candidate(i, j, vertex_set, radius, mesh, edge_front)

        if v is None:
            mesh.set_boundary(i, j)
            continue

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

    return mesh, total_faces

def pivot_ball(vertex_set, radii):
    s0, s1, s2 = tuple(seed_triangle(radii[0], vertex_set))
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

    # mesh, total_faces = generate_mesh(mesh, edge_front, radius, vertex_set, total_faces)

    for radius in radii:
        mesh.clear_boundary_edges()
        mesh, total_faces = generate_mesh(mesh, edge_front, radius, vertex_set, total_faces)
        edge_front = mesh.boundary_edges.keys()

    return mesh

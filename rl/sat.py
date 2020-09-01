def normalize(v):
    from math import sqrt
    norm = sqrt(v[0] ** 2 + v[1] ** 2)
    return (v[0] / norm, v[1] / norm)

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1];

def edge_direction(p0, p1):
    return (p1[0] - p0[0], p1[1] - p0[1]);

def orthogonal(v):
    return (v[1], -v[0])

def vertices_to_edges(vertices):
    return [edge_direction(vertices[i], vertices[(i + 1) % len(vertices)]) \
        for i in range(len(vertices))]

def project(vertices, axis):
    dots = [dot(vertex, axis) for vertex in vertices]
    return [min(dots), max(dots)]

def contains(n, range_):
    a = range_[0]
    b = range_[1]
    if b < a:
        a = range_[1]
        b = range_[0]
    return (n >= a) and (n <= b);

def overlap(a, b):
    if contains(a[0], b):
        return True;
    if contains(a[1], b):
        return True;
    if contains(b[0], a):
        return True;
    if contains(b[1], a):
        return True;
    return False;

def separating_axis_theorem(vertices_a, vertices_b):
    edges_a = vertices_to_edges(vertices_a);
    edges_b = vertices_to_edges(vertices_b);

    edges = edges_a + edges_b

    axes = [normalize(orthogonal(edge)) for edge in edges]

    for i in range(len(axes)):
        projection_a = project(vertices_a, axes[i])
        projection_b = project(vertices_b, axes[i])
        overlapping = overlap(projection_a, projection_b)
        if not overlapping:
            return False;
    return True

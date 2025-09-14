"""
Simple structured Delaunay triangulation in 2D with Bowyer-Watson algorithm.

Written by Jose M. Espadero ( http://github.com/jmespadero/pyDelaunay2D )
Based on code from Ayron Catteau. Published at http://github.com/ayron/delaunay

Just pretend to be simple and didactic. The only requisite is numpy.
Robust checks disabled by default. May not work in degenerate set of points.
"""

import numpy as np

class Delaunay2D:
    """
    Class to compute a Delaunay triangulation in 2D
    ref: http://en.wikipedia.org/wiki/Bowyer-Watson_algorithm
    ref: http://www.geom.uiuc.edu/~samuelp/del_project.html
    """

    def __init__(self, data, center=(0, 0), radius=9999):
        """ Init and create a new frame to contain the triangulation
        center -- Optional position for the center of the frame. Default (0,0)
        radius -- Optional distance from corners to the center.
        """
        center = np.asarray(center)
        # Create coordinates for the corners of the frame
        self.coords = [center+radius*np.array((-1, -1)),
                       center+radius*np.array((+1, -1)),
                       center+radius*np.array((+1, +1)),
                       center+radius*np.array((-1, +1))]

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)
        
        # Add points to triangulation
        if data is not None and len(data) > 0:
            for point in data:
                self.addPoint(point)

    def circumcenter(self, tri):
        """Compute circumcenter and circumradius of a triangle in 2D."""
        pts = np.asarray([self.coords[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                      [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)
        radius = np.sum(np.square(pts[0] - center))
        return (center, radius)

    def inCircleFast(self, tri, p):
        """Check if point p is inside of precomputed circumcircle of tri."""
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def addPoint(self, p):
        """Add a point to the current DT, and refine it."""
        p = np.asarray(p)
        idx = len(self.coords)
        self.coords.append(p)

        bad_triangles = []
        for T in self.triangles:
            if self.inCircleFast(T, p):
                bad_triangles.append(T)

        boundary = []
        T = bad_triangles[0]
        edge = 0
        while True:
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))
                edge = (edge + 1) % 3
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            T = (idx, e0, e1)
            self.circles[T] = self.circumcenter(T)
            self.triangles[T] = [tri_op, None, None]
            if tri_op:
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh and e1 in neigh and e0 in neigh:
                        self.triangles[tri_op][i] = T
            new_triangles.append(T)

        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i+1) % N]
            self.triangles[T][2] = new_triangles[(i-1) % N]

    def exportTriangles(self):
        """Export the current list of Delaunay triangles."""
        return [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def find_simplex(self, coords):
        """
        Needs a lot of optimization
        """
        coords = np.asarray(coords)
        if coords.ndim == 1:
            coords = coords[None, :]
            single_coord = True
        else:
            single_coord = False

        exported_triangles = self.exportTriangles()
        internal_to_exported = {
            (a + 4, b + 4, c + 4): i
            for i, (a, b, c) in enumerate(exported_triangles)
        }

        results = np.full(len(coords), -1, dtype=int)

        # start from any triangle
        current_tri = next(iter(self.triangles.keys()))
        max_steps = 500

        for pi, p in enumerate(coords):
            px, py = p
            tri = current_tri
            for _ in range(max_steps):
                if tri is None:
                    break
                a, b, c = tri # unpack triangle
                ax, ay = self.coords[a]
                bx, by = self.coords[b]
                cx, cy = self.coords[c]

                # cross products relative to edges
                c0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                c1 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
                c2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)

                has_neg = (c0 < 0) or (c1 < 0) or (c2 < 0)
                has_pos = (c0 > 0) or (c1 > 0) or (c2 > 0)

                if not (has_neg and has_pos):
                    # inside
                    results[pi] = internal_to_exported.get(tri, -1)
                    current_tri = tri
                    break

                # walk across edge where point lies outside
                if c0 < 0:
                    tri = self.triangles[tri][2]  # opp vertex c
                elif c1 < 0:
                    tri = self.triangles[tri][0]  # opp vertex a
                elif c2 < 0:
                    tri = self.triangles[tri][1]  # opp vertex b
                else:
                    tri = None

        return results[0] if single_coord else results

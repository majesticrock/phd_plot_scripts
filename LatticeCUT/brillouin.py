from scipy.spatial import HalfspaceIntersection, ConvexHull
import numpy as np

def reciprocal_basis(system, a=1.0):
    """Return reciprocal lattice basis vectors for the given system (direct lattice constant a)."""
    if system == 'sc':
        # reciprocal of sc is sc
        b1 = 2*np.pi/a * np.array([1,0,0])
        b2 = 2*np.pi/a * np.array([0,1,0])
        b3 = 2*np.pi/a * np.array([0,0,1])
    elif system == 'bcc':
        # reciprocal of bcc is fcc
        b = 2*np.pi/a
        b1 = b * np.array([ 1, 1,-1])
        b2 = b * np.array([ 1,-1, 1])
        b3 = b * np.array([-1, 1, 1])
    elif system == 'fcc':
        # reciprocal of fcc is bcc
        b = 2*np.pi/a
        b1 = b * np.array([1,-1, 1])
        b2 = b * np.array([1, 1,-1])
        b3 = b * np.array([-1,1, 1])
    else:
        raise ValueError("Unknown system")
    return np.stack([b1,b2,b3],axis=1)

def find_feasible_point(halfspaces):
    # crude method: just average the normals, push a bit inside
    avg = halfspaces[:,:2].mean(axis=0)
    if np.allclose(avg, 0):
        return np.array([0.0, 0.0])
    return -0.01 * avg / np.linalg.norm(avg)

def bz_slice(system, kz=0.0, a=1.0, nvec=3):
    B = reciprocal_basis(system, a=a)

    rng = range(-nvec, nvec+1)
    Gs = []
    for i in rng:
        for j in rng:
            for k in rng:
                if i==j==k==0:
                    continue
                Gs.append(i*B[:,0] + j*B[:,1] + k*B[:,2])
    Gs = np.array(Gs)

    halfspaces = []
    for G in Gs:
        Gx, Gy, Gz = G
        offset = 0.5*np.dot(G, G) - kz*Gz
        halfspaces.append([Gx, Gy, -offset])
    halfspaces = np.array(halfspaces)

    # interior point: origin projected into plane
    interior = find_feasible_point(halfspaces)
    
    hs = HalfspaceIntersection(halfspaces, interior)
    pts = hs.intersections

    hull = ConvexHull(pts)
    verts = pts[hull.vertices]

    return verts

import numpy as np
from scipy.linalg import sqrtm
def generate_conic(C):
    (Nf, No) = C.shape
    Nf = np.floor_divide(Nf, 3)
    No = np.floor_divide(No, 3)
    W = np.zeros((Nf*6, No))
    for f in range(Nf):
        for o in range(No):
            Cadj = C[3*f:3*f+3, 3*o:3*o+3]
            Cadjv = sym_to_vec3(Cadj)
            Cadjv = Cadjv/-Cadjv[-1]
            W[6*f:6*f+6, o:o+1] = Cadjv
    return W

def sfm(W):
    (Nf, No) = W.shape
    Nf = np.floor_divide(Nf, 6)

    mat = np.zeros((2*Nf, No))
    for f in range(Nf):
        mat[2*f, :] = -W[6*f+2, :]
        mat[2*f+1, :] = -W[6*f+4, :]
    
    mat = mat - np.reshape(np.mean(mat, axis=1), (-1, 1))

    U, D, Vh = np.linalg.svd(mat)
    D = np.diag(D)
    D = D[0:3, 0:3]
    M = U[:, 0:3]
    S = Vh[0:3, :]
    M = M.dot(np.sqrt(D))
    S = np.sqrt(D).dot(S)
    return M, S

def fact_metric_constraint(Mhat, Shat):
    f = np.floor_divide(Mhat.shape[0], 2)
    M = np.zeros(Mhat.shape)
    S = np.zeros(Shat.shape)
    Q = np.zeros((3, 3))
    A = np.zeros((3*f+1, 6))
    for n in range(f):
        A[n, :] = fact_leq(Mhat[2*n, :], Mhat[2*n, :]) - fact_leq(Mhat[2*n+1, :], Mhat[2*n+1, :])
        A[n+f, :] = fact_leq(Mhat[2*n, :], Mhat[2*n+1, :])
        A[n+2*f, :] = fact_leq(Mhat[2*n+1, :], Mhat[2*n, :])
    A[3*f, :] = fact_leq(Mhat[0, :], Mhat[0, :])
    b = np.zeros((3*f+1))
    b[-1] = 1
    v = np.linalg.lstsq(A, b)[0]
    C = np.zeros((3, 3))
    C[0, 0] = v[0]
    C[0, 1] = v[1]
    C[0, 2] = v[2]
    C[1, 1] = v[3]
    C[1, 2] = v[4]
    C[2, 2] = v[5]
    C[1, 0] = C[0, 1]
    C[2, 0] = C[0, 2]
    C[2, 1] = C[1, 2]
    D, V = np.linalg.eig(C)
    if np.sum(D<1) > 0:
        O1, lamb, O2h = np.linalg.svd(C)
        lamb = np.diag(lamb)
        G = O1.dot(np.sqrt(lamb))
        num = np.zeros((2*f, 3))
        den = np.zeros((2*f, 3))
        for m in range(f):
            num[2*m:2*m+2, :] = Mhat[2*m:2*m+2, :].dot(G)
            den[2*m:2*m+2, :] = np.linalg.pinv(Mhat[2*m:2*m+2, :].dot(G).T)
        tmp = np.linalg.lstsq(num, den)[0]
        Q = G.dot(sqrtm(tmp))
    else:
        Q = np.linalg.cholesky(C)
    M = Mhat.dot(Q)
    S = np.linalg.inv(Q).dot(Shat)
    return M, S


def fact_leq(r1, r2):
    v = np.zeros(6)
    v[0] = r1[0]*r2[0]
    v[1] = r1[0]*r2[1]+r1[1]*r2[0]
    v[2] = r1[0]*r2[2]+r1[2]*r2[0]
    v[3] = r1[1]*r2[1]
    v[4] = r1[1]*r2[2]+r1[2]*r2[1]
    v[5] = r1[2]*r2[2]
    return v

def rebuild_Gr(R):
    Nf = R.shape[0]
    Nf = np.floor_divide(Nf, 2)
    Gr = []
    for f in range(Nf):
        Pf = np.zeros((3, 4))
        Pf[0:2, 0:3] = R[2*f:2*f+2, :]
        Pf[2, 3] = 1
        Y = computeY()
        W = computeW()
        Gf = Y.dot(np.kron(Pf, Pf)).dot(W)
        Gr_f = Gf[[0, 1, 3], :]
        Gr_f = Gr_f[:, [0, 1, 2, 4, 5, 7]]
        Gr.append(Gr_f)
    Gr = np.vstack(Gr)
    return Gr
        
def computeY():
    Y = np.zeros((6, 9))
    Y[0, 0] = 1
    Y[1, 1] = 1
    Y[2, 2] = 1
    Y[3, 4] = 1
    Y[4, 5] = 1
    Y[5, 8] = 1
    return Y

def computeW():
    W = np.zeros((16, 10))
    W[0, 0] = 1
    W[1, 1] = 1
    W[4, 1] = 1
    W[2, 2] = 1
    W[8, 2] = 1
    W[3, 3] = 1
    W[12, 3] = 1
    W[5, 4] = 1
    W[6, 5] = 1
    W[9, 5] = 1
    W[7, 6] = 1
    W[13, 6] = 1
    W[10, 7] = 1
    W[11, 8] = 1
    W[14, 8] = 1
    W[15, 9] = 1
    return W

def center_ellipses(W):
    (Nf, No) = W.shape
    Nf = np.floor_divide(Nf, 6)
    Ccenter = np.zeros((3*Nf, No))
    for f in range(Nf):
        for o in range(No):
            Cvec = W[6*f: 6*f+6, o]
            T = np.array([[1, 0, Cvec[2]], [0, 1, Cvec[4]], [0, 0, 1]])
            C = vec_to_sym3(Cvec)
            C = T.dot(C).dot(T.T)
            Cvec = sym_to_vec3(C)
            Ccenter[3*f:3*f+3, o:o+1] = Cvec[np.array([0, 1, 3]), :]
    return Ccenter
            
def vec_to_sym3(v):
    M = np.zeros((3, 3))
    M[0, 0] = v[0]
    M[1, 0] = v[1]
    M[0, 1] = v[1]
    M[2, 0] = v[2]
    M[0, 2] = v[2]
    M[1, 1] = v[3]
    M[1, 2] = v[4]
    M[2, 1] = v[4]
    M[2, 2] = v[5]
    return M

def sym_to_vec3(M):
    v = np.zeros(6)
    v[0] = M[0, 0]
    v[1] = M[1, 0]
    v[2] = M[2, 0]
    v[3] = M[1, 1]
    v[4] = M[1, 2]
    v[5] = M[2, 2]
    v = np.reshape(v, (-1, 1))
    return v

def rebuild_qvec(Q):
    n = Q.shape[1]
    qv = []
    qv.append(Q[0:3, :])
    qv.append(np.zeros((1, n)))
    qv.append(Q[3:5, :])
    qv.append(np.zeros((1, n)))
    qv.append(Q[5:6, :])
    qv.append(np.zeros((1, n)))
    qv.append(np.ones((1, n)))
    return np.vstack(qv)

def quadric_to_ellipsoid(qvec):
    No = qvec.shape[1]
    Es = []
    for o in range(No):
        q = qvec[[0, 1, 2, 4, 5, 7], o:o+1]
        q = vec_to_sym3(q)
        Q = np.zeros((4, 4))
        Q[3, 3] = qvec[9, o]
        Q[0:3, 0:3] = q
        Q = Q/-Q[3, 3]
        
        C = Q[0:3, 3]
        T = np.array([[1, 0, 0, -C[0]], [0, 1, 0, -C[1]], [0, 0, 1, -C[2]], [0, 0, 0, 1]])
        Qcent = T.dot(Q).dot(T.T)
        D, V = np.linalg.eig(Qcent[0:3, 0:3])
        E = {}
        E['C'] = C
        E['u0'] = V[:, 0]
        E['u1'] = V[:, 1]
        E['u2'] = V[:, 2]
        E['e0'] = np.sqrt(np.abs(D[0]))
        E['e1'] = np.sqrt(np.abs(D[1]))
        E['e2'] = np.sqrt(np.abs(D[2]))
        Es.append(E)
    return Es

def ellipsoid_to_quadric(Es):
    No = len(Es)
    qvec = np.zeros((10, No))
    for o in range(No):
        E = Es[o]
        Q = np.diag([E['e0']**2, E['e1']**2, E['e2']**2, -1])
        Z = np.hstack((E['u0'], E['u1'], E['u2']))
        Z = np.reshape(Z, (3, 3))
        Z = Z.T
        Z = np.vstack((Z, E['C'].T))
        Z = np.hstack((Z, np.array([[0], [0], [0], [1]])))
        Q = Z.dot(Q).dot(Z.T)
        q = np.zeros(10)
        q[0] = Q[0, 0]
        q[1] = Q[0, 1]
        q[2] = Q[0, 2]
        q[4] = Q[1, 1]
        q[5] = Q[1, 2]
        q[7] = Q[2, 2]
        q[9] = Q[3, 3]
        qvec[:, o] = q
    return qvec

def recombine_ellipsoid(quad, S):
    rec = []
    No = S.shape[1]
    for o in range(No):
        T = np.eye(4)
        T[0, 3] = S[0, o]
        T[1, 3] = S[1, o]
        T[2, 3] = S[2, o]
        Qvec = quad[:, o]
        Q = np.zeros((4, 4))
        
        Q[3, 3] = -1
        Q[0, 0] = Qvec[0]
        Q[0, 1] = Qvec[1]
        Q[0, 2] = Qvec[2]
        Q[1, 0] = Qvec[1]
        Q[1, 1] = Qvec[3]
        Q[1, 2] = Qvec[4]
        Q[2, 0] = Qvec[2]
        Q[2, 1] = Qvec[4]
        Q[2, 2] = Qvec[5]
        
        Quad = T.dot(Q).dot(T.T)
        rec.append(Quad)
    return rec
        

























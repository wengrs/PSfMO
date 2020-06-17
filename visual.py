import numpy as np
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation

def repQ(R1, Rec):
    No = len(Rec)
    Nf = R1.shape[0]
    Nf = np.floor_divide(Nf, 2)
    C = np.zeros((3*Nf, 3*No))
    for f in range(Nf):
        projM = R1[2*f:2*f+2, :]
        P = np.zeros((3, 4))
        P[2, 3] = 1
        P[0:2, 0:3] = projM
        for o in range(No):
            Ctemp = P.dot(Rec[o]).dot(P.T)
            C[3*f:3*f+3, 3*o: 3*o+3] = Ctemp
    return C
    
def plot_box(box, ax):
    No = box.shape[0]
    No = np.floor_divide(No, 4)
    
    for o in range(No):
        b = box[4*o:4*o+4]
        xs = np.abs(b[2]-b[0])
        ys = np.abs(b[3]-b[1])
        rect = patches.Rectangle((b[0], b[1]), xs, ys, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
def plot_ellipse(conic, T, ax, color):
    No = conic.shape[1]
    No = np.floor_divide(No, 3)
    for o in range(No):
        t = np.eye(3)
        t[0, 2] = -T[0]
        t[1, 2] = -T[1]
        ctemp = conic[:, 3*o:3*o+3]
        ct = t.dot(ctemp).dot(t.T)
        center = -ct[0:2, 2]
        
        G = np.eye(3)
        G[0, 2] = -center[0]
        G[1, 2] = -center[1]
        Chcenter = G.dot(ct).dot(G.T)
        axes, R = np.linalg.eig(Chcenter[0:2, 0:2])
        axes = np.sqrt(axes)
        
        r = np.arccos(R[0, 0])
        
        ellip = patches.Ellipse(center, 2*axes[0], 2*axes[1], angle=r, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(ellip)
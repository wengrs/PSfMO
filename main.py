import numpy as np
import scipy.io as sio
import sfmo
import psfmo
import visual
import matplotlib.pyplot as plt

mat = sio.loadmat('C.mat')
C = np.array(mat['C'])
mat = sio.loadmat('T.mat')
T = mat['T']
W = sfmo.generate_conic(C)
Rtilde, Stilde = sfmo.sfm(W)
R1, S1 = sfmo.fact_metric_constraint(Rtilde, Stilde)
Gr = sfmo.rebuild_Gr(R1)
Ccenter = sfmo.center_ellipses(W)
Quadrics_centered = np.linalg.lstsq(Gr, Ccenter)[0]
qvec_sfmo = sfmo.rebuild_qvec(Quadrics_centered)
Es = sfmo.quadric_to_ellipsoid(qvec_sfmo)
qvec = sfmo.ellipsoid_to_quadric(Es)
Rec = sfmo.recombine_ellipsoid(Quadrics_centered, S1)
Rec_new = sfmo.recombine_ellipsoid(qvec[[0, 1, 2, 4, 5, 7], :], S1)

Gs = sfmo.quadric_to_ellipsoid(qvec)
classes = np.array([2, 2, 2, 2], dtype=int)
m_values = np.ones((78, 4), dtype=int)
default_noise = 0.01
prs = psfmo.Prs()
prs.gmms, prs.mu, prs.ps =  psfmo.init_PRS()
Gest,Eh,scales,axes_red,Sigma,out =psfmo.em(prs,classes,Ccenter,Gr,m_values,Gs,default_noise)
Rec_post = sfmo.recombine_ellipsoid(Eh, S1)

Nf = np.floor_divide(C.shape[0], 3)
fs = np.random.randint(0, Nf, 10)
for f in fs: 
    mat = sio.loadmat('bbx.mat')
    bbx = np.array(mat['bbx'])
    mat = sio.loadmat('Im.mat')
    Img = mat['Img']
    img = Img[0, f]
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    visual.plot_box(bbx[f, :], ax)
    c_sfmo = visual.repQ(R1, Rec)
    c_psfmo = visual.repQ(R1, Rec_post)
    visual.plot_ellipse(c_sfmo[3*f:3*f+3, :], T[:, f], ax, 'y')
    visual.plot_ellipse(c_psfmo[3*f:3*f+3, :], T[:, f], ax, 'b')
    
    
    plt.show()

import scipy.io as sio
from scipy.stats import multivariate_normal
import numpy as np
import numpy.matlib
import itertools
from scipy.spatial.transform import Rotation
from sklearn.mixture import GaussianMixture as GM
import emcee

class Prs:
    mu = []
    ps = []
    gmms = []

class Out:
    Ehi = []
    costg = []
    coste = []

def get_angles(Gs):
    n_o = len(Gs)
    angles = np.zeros((3,n_o),dtype=float)
    for i in range(n_o):
        R = np.zeros((3,3),dtype=float)
        R[:,0]=Gs[i]['u0']
        R[:,1]=Gs[i]['u1']
        R[:,2]=Gs[i]['u2']
        if np.linalg.det(R) < 0:
            R[:,2]=-np.squeeze(Gs[i]['u2'])

        r = Rotation.from_matrix(R)
        rvec = r.as_euler('zyx',degrees=True)[::-1]
        angles[:,i] = rvec.T
        
    return angles


def init_PRS():
    data = sio.loadmat('stats.mat')
    ax_distrs = data['ax_distrs']
    Axd = ax_distrs[0,0]
    mus = np.zeros((3,len(Axd)),dtype=float)
    ps = np.zeros((3, len(Axd)*2),dtype=float)
    n = 0
    GMs = []
    for value in Axd:
        a = value * value
        mu = np.mean(a, axis = 1)
        temp = np.matlib.repmat(mu, 1, np.shape(a)[1])
        temp = np.reshape(temp, (np.shape(a)[1],3))
        a = a - temp.T
        eigv,V = np.linalg.eig( np.dot(a,a.T) )
        idx = eigv.argsort()#[::-1]
        V = V[:,idx]
        p = V[:,1:3]
        # p = np.c_[p,V[:,0]]
        ared = np.dot(p.T, a)
        try:
            gmm = GM(n_components=10).fit(ared.T)
        except:
            print('cannot do it with 10 components, trying with 3')
            try:
                gmm = GM(n_components=3).fit(ared.T)
            except:
                print('cannot do it with 3 components, trying with 1')
                gmm = GM(n_components=1).fit(ared.T)

        GMs.append(gmm)
        mus[:,n] = mu
        ps[:,2*n:2*(n+1)] = p
        n+=1

        
    return GMs, mus, ps

def h_exp_first(prs,classes,measure,C,Sigma_noise,axes_post,scales,angles_post,Gs):

    ####################################
    # E step function: computing the sufficient statistics
    # inputs:
    #    prs: the prior distribution parameters
    #    classes: the object labels
    #    measure: the observed conics
    #    C: the initial sfmo camera
    #    Gs: the ellipsoid version of the previous
    #    inc: contains various variable usefull for debugging and inspecting the performances on a per iteration basis
    #    defaultNoise: initialisation of the noise covariance matrix. If set to 0, it will default to 1 and will be estimated if not, it will be kept as it is.
    #    Nmax: the maximum number of iterations. With the use of the MCMC, the em algorithm is not guaranted to converge. If not provided, it will default to 1, i.e. only one Estep is done. Note that if the defaultNoise is different than 0 

    # outputs:
    #    C: the camera matrix. In this version, it is left unchanged.
    #    Eh: the expectation of the latent variables: here these are the dual centred quadrics as a 6 dim vector.
    #    scales, axe_red, : the normalise axis length and their scale
    #    Sigma_noise: the noise covariance matrix
    #    out: Will contain the estimated quadric with the 6 different possible correspondences between the quadric and the prior, as mentioned in the end of Sec 2.3 of the PSFMO ICCV 2017 pqper
    ####################################

    _, n_obj = np.shape(measure)
    prec_data = np.linalg.inv(Sigma_noise)
    size_h = np.shape(C)[1]
    Eh = np.zeros((size_h,n_obj), dtype=float) # expected value of h under the distribution q(h)
    Ehh = np.zeros((size_h, size_h, n_obj),dtype=float) #E[hh'] another sufficient statistics which will be needed for the M step
    chs = np.zeros((n_obj,3), dtype=float)
    axes_red = np.zeros((2,n_obj),dtype=float)

    out = Out()
    out.Ehi = np.zeros((n_obj, 6, 6))
    out.costg = np.zeros((n_obj, 6))
    out.coste = np.zeros((n_obj, 6))
    for i in range(n_obj):
        sfmo_axes = np.zeros((1,3),dtype=float)
        sfmo_axes[0,0] = Gs[i]['e0']
        sfmo_axes[0,1] = Gs[i]['e1']
        sfmo_axes[0,2] = Gs[i]['e2']

        init_angle = angles_post[:,i]
        rvec = init_angle[::-1]
        r = Rotation.from_euler('zyx',rvec,degrees=True)
        R = r.as_matrix()

        Rm = np.zeros((3,6),dtype=float)
        Rm[:,0] = R[0,:] * R[0,:]
        Rm[:,1] = R[0,:] * R[1,:]
        Rm[:,2] = R[0,:] * R[2,:]
        Rm[:,3] = R[1,:] * R[1,:]
        Rm[:,4] = R[1,:] * R[2,:]
        Rm[:,5] = R[2,:] * R[2,:]

        MyArray = np.arange(3)
        permut = itertools.permutations(MyArray)
        permut_array = np.empty((0,3),dtype=int)
        for p in permut:
            permut_array = np.append(permut_array,np.atleast_2d(p),axis=0)
        perm = permut_array[::-1]
        gmm = prs.gmms[i]
        sc = np.linalg.norm(sfmo_axes)

        models = np.zeros((3*np.shape(perm)[0], 1000))
        logP = np.zeros((np.shape(perm)[0],1000))
        for cc in range(np.shape(perm)[0]):
            ch = perm[cc,:]
            h, sc, P, mu = ell_to_latent(ch,prs,Gs[i],classes[i])

            minit = np.zeros((8,3))
            minit[:,0] = np.array([0.1, 1, 10, 100, 0.1, 1, 10, 100 ])*sc
            rv = multivariate_normal(h, np.eye(2))
            for k in range(8):
                minit[k,1:3] = rv.rvs()

            ndim = 3
            nwalkers = 8
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_hammer_obs2, args=[measure[:,i],prec_data,gmm,Rm,P,mu,C])
            sampler.run_mcmc(minit, 125)
            modi = sampler.get_chain()
            lgP = sampler.get_log_prob()

            models[3*cc:3*cc+3] = np.reshape(modi,(1000,3)).T
            logP[cc,:] = np.squeeze(np.reshape(lgP,(1000,1)))
            iis = np.where(logP[cc,:] == np.max(logP[cc,:]))
            ii = iis[0][0]

            Ehi = models[3*cc:3*cc+3, ii]
            axes_norm = np.dot(P, Ehi[1:3]) + mu
            if axes_norm[1] < 0:
                axes_norm[1] = 0.1
            if axes_norm[2] < 0:
                axes_norm[2] = 0.1

            l = (Ehi[0]*np.sqrt(axes_norm))*(Ehi[0]*np.sqrt(axes_norm))
            out.Ehi[i,:,cc] = np.dot(Rm.T,l)
            out.costg[i,cc] = log_hammer_obs2(Ehi,measure[:,i],prec_data,gmm,Rm,P,mu,C)
            out.coste[i,cc] = np.linalg.norm( np.dot(np.dot(C,Rm.T),l) - measure[:,i])

        r = np.argmax(np.amax(logP,axis = 1))
        index = np.argmax(logP[r,:])
        h = models[3*r:3*r+3, index]
        ch = perm[r,:]
        _,Ehi = latent_to_ell(ch,prs,h,classes[i], Rm)
        P = prs.ps[ch, (classes[i]-1)*2:classes[i]*2]
        covi = np.cov((np.dot(Rm.T, (np.matlib.repmat(models[r*3,:], 3, 1)) * (np.dot(P,models[r*3+1:r*3+3,:])))))
        Ehhi = np.dot(np.reshape(Ehi,(6,1)),np.reshape(Ehi,(1,6))) + covi

        Eh[:,i] = Ehi
        Ehh[:,:,i] = Ehhi
        # print()

    return Eh,Ehh,scales,axes_post,axes_red,angles_post,chs,out

def log_hammer_obs2(m,mu_data,prec_data,gmm,R,P,mu,C):
    axes = m[0] * np.sqrt(np.abs(np.dot(P,m[1:3]) + mu))
    axes = axes * axes
    obs_gen = np.dot(np.dot(C,R.T),axes)
    first = np.dot(np.dot((obs_gen - mu_data).T, prec_data), (obs_gen - mu_data))/len(mu_data)
    sec = min(1e15, -gmm.score_samples(np.reshape(m[1:3],(1,2))))

    if sec > 1 and first < 50:
        sec = sec * 1000

    if sec < -1 or first > 50:
        if sec < 2:
            first = first * 1000
        else:
            sec = sec * 1000

    r = -1 * ( first + sec)

    return r

def latent_to_ell(ch,prs,h,obj_class,Rm):
    P = prs.ps[ch,(obj_class*2-2):(obj_class*2)]
    mu = prs.mu[ch, obj_class-1]
    axes_norm = np.dot(P, h[1:3]) + mu
    l = (h[0]*np.sqrt(axes_norm))*(h[0]*np.sqrt(axes_norm))
    e = np.dot(Rm.T, l)
    return l,e

def ell_to_latent(ch,prs,Gs,obj_class):
    l = np.zeros((3,1),dtype=float)
    l[0,0] = Gs['e0']
    l[1,0] = Gs['e1']
    l[2,0] = Gs['e2']

    sc = np.linalg.norm(l)
    mu = prs.mu[ch, obj_class-1]
    P = prs.ps[ch,(obj_class*2-2):(obj_class*2)]
    l = l/sc
    h = np.dot(P.T,(np.squeeze(np.multiply(l,l)) - mu))

    return h, sc, P, mu

def em(prs,classes,measure,C,m_values,Gs,default_noise):

    ####################################
    # Compute the expectations of the axis length, orientation.
    # inputs:
    #    prs: the prior distribution parameters
    #    classes: the object labels
    #    measure: the observed conics
    #    C: the initial sfmo camera
    #    m_values: missing data pattern, it is a boolean matrix
    #    Gs: the ellipsoid version of the previous
    #    inc: contains various variable usefull for debugging and inspecting the performances on a per iteration basis
    #    defaultNoise: initialisation of the noise covariance matrix. If set to 0, it will default to 1 and will be estimated if not, it will be kept as it is.
    #    Nmax: the maximum number of iterations. With the use of the MCMC, the em algorithm is not guaranted to converge. If not provided, it will default to 1, i.e. only one Estep is done. Note that if the defaultNoise is different than 0 

    # outputs:
    #    C: the camera matrix. In this version, it is left unchanged.
    #    Eh: the expectation of the latent variables: here these are the dual centred quadrics as a 6 dim vector.
    #    scales, axe_red, : the normalise axis length and their scale
    #    Sigma_noise: the noise covariance matrix
    #    out: Will contain the estimated quadric with the 6 different possible correspondences between the quadric and the prior, as mentioned in the end of Sec 2.3 of the PSFMO ICCV 2017 pqper
    ####################################

    Nmax = 1
    # normalising
    mm = np.mean(measure[:])
    measure = measure/mm
    C=C/mm

    n_images3, n_obj = np.shape(measure)
    Sigma_noise = np.eye(n_images3)
    if default_noise != 0:
        Sigma_noise = Sigma_noise * default_noise

    Lold = -np.inf
    Cold = C
    Sigma_noiseold = Sigma_noise
    Ehold = np.zeros((6,n_obj), dtype=float)
    Ehh0ld = np.zeros((6,6,n_obj), dtype=float)
    scales = np.zeros((n_obj,1),dtype=float)
    axes = np.zeros((3,n_obj),dtype=float)
    angles_pre=get_angles(Gs)
    for n in range(Nmax):
        # estep
        Eh,Ehh,scales_post,axes_post,axes_red,angles_post,_,out = h_exp_first(prs,classes,measure,C,Sigma_noise,axes,scales,angles_pre,Gs)
        
        Eh[np.isnan(Eh[:])]=Ehold[np.isnan(Eh[:])]
        Eh[np.isinf(Eh[:])]=Ehold[np.isinf(Eh[:])]
        Ehh[np.isnan(Ehh[:])]=Ehh0ld[np.isnan(Ehh[:])]
        Ehh[np.isinf(Ehh[:])]=Ehh0ld[np.isinf(Ehh[:])]
        # mstep
        # computing the log-likelihood
        L = 0
        for i in range(n_obj):
            ux = np.dot(C,Eh[:,i])
            Covx = Sigma_noise + np.dot(np.dot(C,np.eye(6)),C.T)
            L = L - 0.5*(n_images3*np.log(2*np.pi) + np.shape(measure)[0]*np.log(Covx[0,0]) + np.dot(np.dot((measure[m_values[:,i]==1,i] - ux[m_values[:,i]==1]).T, np.linalg.inv(Covx[:, m_values[:,i]==1][m_values[:,i]==1,:])),(measure[m_values[:,i]==1,i] - ux[m_values[:,i]==1])) )

        if L > Lold:
            Lold = L
            Cold = C
            Ehold = Eh
            Ehh0ld = Ehh
            Sigma_noiseold = Sigma_noise
            scales = scales_post
            axes = axes_post
            angles_pre = angles_post
        else:
            break
    
    C = Cold
    Sigma_noise = Sigma_noiseold
    Eh = Ehold

    return C,Eh,scales,axes_red,Sigma_noise,out



if __name__ == "__main__":
    SfMO_result = sio.loadmat('result_SfMO.mat')
    Ccenter = SfMO_result['Ccenter']
    classes = SfMO_result['classes']
    Gs = SfMO_result['Gs']
    inc = SfMO_result['inc']
    m_values = SfMO_result['m_values']
    Gred = SfMO_result['Gred']

    prs = Prs()
    prs.gmms, prs.mu, prs.ps =  init_PRS()

    default_noise = 0.01
    Gest,Eh,scales,axes_red,Sigma,out =em(prs,classes,Ccenter,Gred,m_values,Gs,inc,default_noise)

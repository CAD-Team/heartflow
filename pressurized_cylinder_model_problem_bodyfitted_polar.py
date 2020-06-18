from nutils import*
from nutils.pointsseq import PointsSequence
import numpy as np
from matplotlib import pyplot as plt
import os


def locatesample(fromsample, fromgeom, totopo, togeom, tol, **kwargs):
    '''Clone ``fromsample`` onto unrelated topology ``totopo``

    Create a sample ``tosample`` of ``totopo`` such that ``fromgeom`` and
    ``togeom`` are equal on the respective samples and such that integrals are
    equal.

    Parameters
    ----------
    fromsample: :class:`nutils.sample.Sample`
      The sample to be located in ``totopo``.
    fromgeom: :class:`nutils.function.Array`
      The geometry evaluable on ``fromsample``.
    totopo: :class:`nutils.topology.Topology`
      The topology to create ``tosample`` on.
    togeom: :class:`nutils.function.Array`
      The geometry evaluable on ``totopo``.
    **kwargs:
      All keyword arguments are passed to
      :meth:`nutils.topology.Topology.locate`.

    Returns
    -------
    tosample: :class:`nutils.sample.Sample`
      The sample of ``totopo``.

    '''

    tosample = totopo.locate(togeom, fromsample.eval(fromgeom), tol=tol, **kwargs)

    # Copy the weights from `fromsample` and account for the change in local
    # coordinates via the common geometry.
    weights = fromsample.eval(function.J(fromgeom)) / tosample.eval(function.J(togeom))
    for p, i in zip(fromsample.points, fromsample.index):
        weights[i] = p.weights
    weightedpoints = tuple(points.CoordsWeightsPoints(p.coords, weights[i]) for p, i in zip(tosample.points, tosample.index))
    weightedpoints = PointsSequence.from_iter(weightedpoints, 3)
    return sample.Sample.new(tosample.transforms, weightedpoints, tosample.index)


def nID(i, j, k, nez, ney, n):
    ii = i % n
    jj = j % n
    kk = k % n
    I = i // n
    J = j // n
    K = k // n
    e = K + J * nez + I * nez * ney
    nid = e * n * n * n + kk + jj * n + ii * n * n
    return nid 


def TetCube(i, j, k, nez, ney, n):
    zero =  nID(i, j, k, nez, ney, n)
    one =   nID(i, j, k+1, nez, ney, n)
    two =   nID(i, j+1, k, nez, ney, n)
    three = nID(i, j+1, k+1, nez, ney, n)
    four =  nID(i+1, j, k, nez, ney, n)
    five =  nID(i+1, j, k+1, nez, ney, n)
    six =   nID(i+1, j+1, k, nez, ney, n)
    seven = nID(i+1, j+1, k+1, nez, ney, n)
    t1 = [ zero, one, two, four]
    t2 = [ one, two, four, five]
    t3 = [ two, four, five, six]
    t4 = [ one, two, three, five]
    t5 = [ two, three, five, six]
    t6 = [ three, five, six, seven]
    return [t1,t2,t3,t4,t5,t6]

def GaussTri(topo, degree):
    n = int(np.ceil((degree+1)/2))
    nx = topo.shape[0] * n
    ny = topo.shape[1] * n
    nz = topo.shape[2] * n
    ntris = (nx-1) * (ny-1) * (nz-1) * 6
    tri = np.ndarray([ntris,4],dtype=int)
    ind = 0
    for i in range(nx-1):
        for j in range(ny-1):
            for k in range(nz-1):
                elemtris = TetCube(i, j, k, topo.shape[2], topo.shape[1], n)
                for m in range(6):
                    tri[ind,:] = elemtris[m]
                    ind = ind + 1
    return tri

def GaussSort(x, topo, n):
    nx = topo.shape[0] * n
    ny = topo.shape[1] * n
    nz = topo.shape[2] * n
    nez = topo.shape[2]
    ney = topo.shape[1]
    y = x.copy()
    ind = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                nid = nID(i, j, k, nez, ney, n)
                y[ind] = x[nid]
                ind = ind + 1
    print(y)
    return y

def Solve(res, cons, callback):
    return solver.solve_linear('lhs', residual=res, constrain=cons, linsolver='cg', linatol=1e-14, linprecon='diag', lincallback=callback)


def FindSpan(p, m, U, u):
    low = p - 1
    high = m - p
    if u == U[high]:
        return high - 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid+1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

def Refine(p, m, U, P, r, X, a, b):
    if r < 1:
        return U,P

    n = m - p 
    V = [0.0]*(m + r)
    Q = np.zeros([n+r+1,3])
    r = r - 1

    for j in range(0, a-p+2):
        Q[j,:] = P[j,:]
    for j in range(b, n+1):
        Q[j+r+1,:] = P[j,:]
    for j in range(0, a+1):
        V[j] = U[j]
    for j in range(b+p, m):
        V[j+r+1] = U[j]
    i = b + p - 1
    k = b + p + r

    for j in range(r, -1, -1):
        while X[j] <= U[i] and i > a:
            Q[k-p,:] = P[i-p, :]
            V[k] = U[i]
            k = k - 1
            i = i - 1
        Q[k-p, :] = Q[k - p + 1, :]
        for l in range(1,p+1):
            ind = k - p + l
            alfa = V[k+l] - X[j]
            if np.abs(alfa) == 0.0:
                Q[ind, :] = Q[ind+1,:]
            else:
                alfa = alfa / (V[k+l]-U[i-p+l])
                Q[ind, :] = alfa * Q[ind, :] + (1.0 - alfa) * Q[ind + 1, :]
        V[k] = X[j]
        k = k - 1
    return V, Q



def NURBBSCylinder(ri, ro, t, nr, nt, nz):
    p = 2
    U =[0,0,1,1]
    Pw = np.ndarray([3,3])
    Pw[0,:] = [1,0,1]
    Pw[1,:] = [1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2)]
    Pw[2,:] = [0,1,1]
    u = np.linspace(0,1,nt+1)
    X = u[1:len(u)-1]
    a = FindSpan(p, len(U), U, X[0])
    b = FindSpan(p, len(U), U, X[-1]) + 1
    V, Qw = Refine(p, len(U), U, Pw, len(X), X, a, b)
    Q = Qw.copy()
    for i in range(len(Qw)):
        Q[i,0:2] /= Qw[i,2]
    npr = 2 + nr
    npt = 2 + nt
    npz = 2 + nz
    R = np.linspace(ri,ro,npr)
    Z = np.linspace(-t,t,npz)
    P = np.zeros([npz * npr * npt, 3])
    weights = np.zeros([npz * npr * npt])
    ind = 0
    for i in range(npt):
        for j in range(npr):
            for k in range(npz):
                P[ind ,:] = [R[j] * Q[i,0], R[j] * Q[i,1], Z[k]]
                weights[ind] = Q[i,2]
                ind = ind + 1
    return P, weights

def BoundaryFittedSolution(Nr, Nt, nu_wall, E_wall, ri, ro, pi, basis_degree, gauss_degree, PLOT3D, label, nSamples):

    Navg = (Nr + Nt)/2
    t = (ro - ri) / (2 * Navg)

    Nz = 1

    P, weights = NURBBSCylinder(ri, ro, t, Nr, Nt, Nz)

    th = np.linspace(0, np.pi/2, Nt + 1)
    r = np.linspace(ri,ro,Nr+1)
    z = np.linspace(-t,t,Nz+1)

    omega = function.Namespace()
    omega_topo, omega.uvw = mesh.rectilinear([th,r,z])
    omega.x_i = '<uvw_1 cos(uvw_0) , uvw_1 sin(uvw_0) , uvw_2 >_i'
    
    omega.pi = pi
    omega.traction_i = '<pi cos(uvw_0), pi sin(uvw_0) , 0 >_i'

    # Add Mat Props functions to namespace
    omega.nu = nu_wall
    omega.E = E_wall
    omega.mu = 'E / (2 (1 + nu))'
    omega.lmbda = 'E nu / ( (1 + nu) (1 - 2 nu) )'

    # Define Analysis
    omega.basis = omega_topo.basis('spline',degree = basis_degree)
    omega.ubasis = omega.basis.vector(3)
    omega.u_i = 'ubasis_ni ?lhs_n'
    omega.X_i = 'x_i + u_i'
    omega.strain_ij = '(u_i,j + u_j,i) / 2'
    omega.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'
    omega.meanstress = 'stress_kk / 3'
    omega.S_ij = 'stress_ij - (stress_kk) δ_ij / 3'
    omega.vonmises = 'sqrt(3 S_ij S_ij / 2)'
    omega.disp = 'sqrt(u_i u_i)'
    omega.r = 'sqrt( x_i x_i )'
    omega.cos = 'x_0 / r'
    omega.sin = 'x_1 / r'
    omega.Qinv_ij = '< < cos , sin , 0 >_j , < -sin , cos , 0 >_j , < 0 , 0 , 1 >_j >_i'
    omega.sigma_kl = 'stress_ij Qinv_kj Qinv_li '
    omega.eps_kl = 'strain_ij Qinv_kj Qinv_li '
    omega.du_i = 'Qinv_ij u_j'
    omega.sum = 'nu ( sigma_00 + sigma_11 )'
    
    # Stiffness Matrix
    K = omega_topo.integral('ubasis_ni,j stress_ij d:x' @ omega, degree=gauss_degree)

    # Force Vector
    F = omega_topo.boundary['bottom'].integral('traction_i ubasis_ni d:x' @ omega, degree=gauss_degree)

    # Constrain Omega
    sqr  = omega_topo.boundary['right'].integral('u_0 u_0 d:x' @ omega, degree=2*basis_degree)
    sqr += omega_topo.boundary['left'].integral('u_1 u_1 d:x' @ omega, degree=2*basis_degree)
    sqr += omega_topo.integral('u_2 u_2 d:x' @ omega, degree=2*basis_degree)
    cons = solver.optimize('lhs', sqr, droptol=1e-15, linsolver='cg', linatol=1e-15, linprecon='diag')


    # Initialize Residual Vector
    residuals = []
    def AddResiualNorm(res):
        residuals.append(res)

    # Solve
    lhs = Solve(K-F, cons, AddResiualNorm)

    samplepts = omega_topo.sample('gauss', gauss_degree)
    x = samplepts.eval(omega.x)
    E, nu = samplepts.eval([omega.E, omega.nu])
    meanstress, vonmises, disp = samplepts.eval(['meanstress', 'vonmises', 'du_i'] @ omega, lhs=lhs)
    sigmarr, sigmatt, sigmazz, sigmart, sigmatz, sigmarz = samplepts.eval(['sigma_00', 'sigma_11','sigma_22', 'sigma_01','sigma_12', 'sigma_02'] @ omega, lhs=lhs)
    epsrr, epstt, epszz, epsrt, epstz, epsrz = samplepts.eval(['eps_00', 'eps_11','eps_22', 'eps_01','eps_12', 'eps_02'] @ omega, lhs=lhs)
    ssum = samplepts.eval(omega.sum, lhs=lhs)

    # Gauss Mesh
    gauss = function.Namespace()
    n = int(np.ceil((gauss_degree+1)/2))
    nx = omega_topo.shape[0] * n
    ny = omega_topo.shape[1] * n
    nz = omega_topo.shape[2] * n
    gx = np.linspace(0,1,nx)
    gy = np.linspace(0,1,ny)
    gz = np.linspace(0,1,nz)
    gauss_topo, gauss.uvw = mesh.rectilinear([gx,gy,gz])

    # copy and sort samples
    x_sorted = x.copy()
    meanstress_sorted = meanstress.copy()
    vonmises_sorted = vonmises.copy()
    E_sorted = E.copy()
    nu_sorted = nu.copy()
    disp_sorted = disp.copy()
    sigmarr_sorted = sigmarr.copy()
    sigmatt_sorted = sigmatt.copy()
    sigmazz_sorted = sigmazz.copy()
    sigmart_sorted = sigmart.copy()
    sigmarz_sorted = sigmarz.copy()
    sigmatz_sorted = sigmatz.copy()
    epsrr_sorted = epsrr.copy()
    epstt_sorted = epstt.copy()
    epszz_sorted = epszz.copy()
    epsrt_sorted = epsrt.copy()
    epsrz_sorted = epsrz.copy()
    epstz_sorted = epstz.copy()
    ssum_sorted = ssum.copy()


    ind = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                nid = nID(i, j, k, omega_topo.shape[2], omega_topo.shape[1], n)
                x_sorted[ind] = x[nid]
                meanstress_sorted[ind] = meanstress[nid]
                vonmises_sorted[ind] = vonmises[nid]
                E_sorted[ind] = E[nid]
                nu_sorted[ind] = nu[nid]
                disp_sorted[ind] = disp[nid]
                sigmarr_sorted[ind] = sigmarr[nid]
                sigmatt_sorted[ind] = sigmatt[nid]
                sigmazz_sorted[ind] = sigmazz[nid]
                sigmart_sorted[ind] = sigmart[nid]
                sigmarz_sorted[ind] = sigmarz[nid]
                sigmatz_sorted[ind] = sigmatz[nid]
                epsrr_sorted[ind] = epsrr[nid]
                epstt_sorted[ind] = epstt[nid]
                epszz_sorted[ind] = epszz[nid]
                epsrt_sorted[ind] = epsrt[nid]
                epsrz_sorted[ind] = epsrz[nid]
                epstz_sorted[ind] = epstz[nid]
                ssum_sorted[ind] = ssum[nid]
                ind = ind + 1


    # create gauss sample interpolants
    gauss.linbasis = gauss_topo.basis('spline',degree=1)
    gauss.xx = gauss.linbasis.dot(x_sorted[:,0])
    gauss.yy = gauss.linbasis.dot(x_sorted[:,1])
    gauss.zz = gauss.linbasis.dot(x_sorted[:,2])
    gauss.ur = gauss.linbasis.dot(disp_sorted[:,0])
    gauss.ut = gauss.linbasis.dot(disp_sorted[:,1])
    gauss.uz = gauss.linbasis.dot(disp_sorted[:,2])
    gauss.x_i = '<xx, yy, zz>_i'
    gauss.u_i = '<ur, ut, uz>_i'
    gauss.meanstress = gauss.linbasis.dot(meanstress_sorted)
    gauss.vonmises = gauss.linbasis.dot(vonmises_sorted)
    gauss.E = gauss.linbasis.dot(E_sorted)
    gauss.nu = gauss.linbasis.dot(nu_sorted)
    gauss.sigmarr = gauss.linbasis.dot(sigmarr_sorted)
    gauss.sigmatt = gauss.linbasis.dot(sigmatt_sorted)
    gauss.sigmazz = gauss.linbasis.dot(sigmazz_sorted)
    gauss.sigmart = gauss.linbasis.dot(sigmart_sorted)
    gauss.sigmarz = gauss.linbasis.dot(sigmarz_sorted)
    gauss.sigmatz = gauss.linbasis.dot(sigmatz_sorted)
    gauss.epsrr = gauss.linbasis.dot(epsrr_sorted)
    gauss.epstt = gauss.linbasis.dot(epstt_sorted)
    gauss.epszz = gauss.linbasis.dot(epszz_sorted)
    gauss.epsrt = gauss.linbasis.dot(epsrt_sorted)
    gauss.epsrz = gauss.linbasis.dot(epsrz_sorted)
    gauss.epstz = gauss.linbasis.dot(epstz_sorted)
    gauss.sum = gauss.linbasis.dot(ssum_sorted)


    # Plot Stress Results
    if PLOT3D == True:
        name = "pressurized_cylinder_model_problem"
        tri = GaussTri(omega_topo, gauss_degree)
        export.vtk( name ,tri, x, E=E, nu=nu, u=disp, sigmarr=sigmarr, sigmatt=sigmatt, sigmazz=sigmazz, meanstress=meanstress, vonmises=vonmises )


    # Define slice
    ns = function.Namespace()
    eps = (ro - ri) / (2 * Nr)
    topo, ns.t = mesh.rectilinear([np.linspace(ri+eps,ro-eps,nSamples+1)])
    ns.rgeom_i = '< t_0 / sqrt(2), t_0 / sqrt(2), 0 >_i'
    ns.r = 't_0'

    # sample
    samplepts = topo.sample('gauss',1)
    pltpts = locatesample(samplepts, ns.rgeom, gauss_topo, gauss.x, 1e-7)
    r = samplepts.eval(ns.r)
    vonmises = pltpts.eval(gauss.vonmises)
    meanstress = pltpts.eval(gauss.meanstress)
    ur = pltpts.eval(gauss.u[0])
    ut = pltpts.eval(gauss.u[1])
    uz = pltpts.eval(gauss.u[2])
    E = pltpts.eval(gauss.E)
    nu = pltpts.eval(gauss.nu)
    sigmarr = pltpts.eval(gauss.sigmarr)
    sigmatt = pltpts.eval(gauss.sigmatt)
    sigmazz = pltpts.eval(gauss.sigmazz)
    sigmart = pltpts.eval(gauss.sigmart)
    sigmarz = pltpts.eval(gauss.sigmarz)
    sigmatz = pltpts.eval(gauss.sigmatz)
    epsrr = pltpts.eval(gauss.epsrr)
    epstt = pltpts.eval(gauss.epstt)
    epszz = pltpts.eval(gauss.epszz)
    epsrt = pltpts.eval(gauss.epsrt)
    epsrz = pltpts.eval(gauss.epsrz)
    epstz = pltpts.eval(gauss.epstz)
    ssum = pltpts.eval(gauss.sum)


    vals = {}
    vals["vonmises"] = vonmises
    vals["meanstress"] = meanstress
    vals["ur"] = ur
    vals["ut"] = ut
    vals["uz"] = uz
    vals["sigmarr"] = sigmarr
    vals["sigmatt"] = sigmatt
    vals["sigmazz"] = sigmazz
    vals["sigmart"] = sigmart
    vals["sigmarz"] = sigmarz
    vals["sigmatz"] = sigmatz
    vals["epsrr"] = epsrr
    vals["epstt"] = epstt
    vals["epszz"] = epszz
    vals["epsrt"] = epsrt
    vals["epsrz"] = epsrz
    vals["epstz"] = epstz
    vals["sum"] = ssum

    print("finished case: " + label)


    return r, vals, residuals




def ExactSolution(ri, ro, pi, nu_wall, E_wall, nSamples):

    # model problem name
    model_problem_name = "cylinder"

    # Exact Solutions
    s = function.Namespace()
    nSamples = 100
    topo, s.x = mesh.rectilinear([np.linspace(ri,ro,nSamples+1)])
    samplepts = topo.sample('gauss',1)
    s.r = 'x_0'
    s.pi = pi
    s.po = 0
    s.ri = ri
    s.ro = ro
    s.nu = nu_wall
    s.E = E_wall
    s.sigmatt = '((pi ri^2 - po ro^2) / (ro^2 - ri^2)) + ((pi - po) ri^2 ro^2) / (r^2 (ro^2 - ri^2))'
    s.sigmarr = '((pi ri^2 - po ro^2) / (ro^2 - ri^2)) - ((pi - po) ri^2 ro^2) / (r^2 (ro^2 - ri^2))'
    s.sigmazz = '2 nu (pi ri^2 - po ro^2) / (ro^2 - ri^2)'
    s.meanstress = '( sigmatt + sigmarr + sigmazz ) / 3'
    s.vonmises = 'sqrt(  ( (sigmarr - sigmatt)^2 + (sigmatt - sigmazz)^2 + (sigmazz - sigmarr)^2  ) / 2  )'
    s.d1 = '(1 + nu) ri^2 ro^2 / (E (ro^2 - ri^2))'
    s.d2 = '( (1 - 2 nu) pi ) / ro^2'
    s.ur = 'd1 ( (pi / r ) + d2 r )'
    s.epsrr = '(- d1 pi / r^2) + d1 d2'
    s.sum = "nu ( sigmatt + sigmarr )"



    # Sample Exact Solutions
    vals = {}
    r = samplepts.eval(s.r)
    vals["vonmises"] = samplepts.eval(s.vonmises)
    vals["meanstress"] = samplepts.eval(s.meanstress)
    vals["ur"] = samplepts.eval(s.ur)
    vals["ut"] = np.zeros([nSamples])
    vals["uz"] = np.zeros([nSamples])
    vals["sigmarr"] = samplepts.eval(s.sigmarr)
    vals["sigmatt"] = samplepts.eval(s.sigmatt)
    vals["sigmazz"] = samplepts.eval(s.sigmazz)
    vals["sigmart"] = np.zeros([nSamples])
    vals["sigmarz"] = np.zeros([nSamples])
    vals["sigmatz"] = np.zeros([nSamples])
    vals["epsrr"] = samplepts.eval(s.epsrr)
    vals["epstt"] = np.zeros([nSamples])
    vals["epszz"] = np.zeros([nSamples])
    vals["epsrt"] = np.zeros([nSamples])
    vals["epsrz"] = np.zeros([nSamples])
    vals["epstz"] = np.zeros([nSamples])
    vals["sum"] = samplepts.eval(s.sum)


    return r, vals

def InitializePlots(keys):
    figs = {}
    axs = {}
    for key in keys:
        figs[key] = plt.figure()
        axs[key] = figs[key].add_subplot(111)
    return figs, axs

def PlotResidual(res, model_problem_name, study_name, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    niter = np.arange(0,len(res),1)
    ax.plot(niter, res, label=label)
    ax.set_title('Iterative Solver Residual')
    ax.set_xlabel('k')
    ax.set_ylabel('|| K $x_{k}$ - F ||')
    ax.set_yscale('log')
    ax.legend()
    fdir = "Results/" + model_problem_name + "/" + study_name + "/Residuals" 
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = label
    fext = ".png"
    fpath = fdir + "/" + fname + fext
    fig.savefig(fpath)
    print("saved /heartflow/" + fpath)
    plt.close(fig)

def Plot(axs, r, vals, label):
    for key in axs.keys():
        if key in vals:
            axs[key].plot(r, vals[key], label=label)


def Export(model_problem_name, study_name, figs, axs, titles, ylabels):

    for key in figs.keys():
        axs[key].set_title(titles[key])
        axs[key].set_xlabel('r')
        axs[key].set_ylabel(ylabels[key])
        axs[key].legend()
        fdir = "Results/" + model_problem_name + "/" + study_name
        fname = key
        fext = ".png"
        fpath = fdir + "/" + fname + fext
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        figs[key].savefig(fpath)
        print("saved /heartflow/" + fpath)

def CloseFigs(figs):
    for key in figs:
        plt.close( figs[key] )

def Normalize(normalization_factors, vals):
    for key in vals:
        if key in normalization_factors:
            vals[key] /= normalization_factors[key]
    return vals


def BoundaryFittedMeshResolutionStudy(Nr, Nt, nu_wall, E_wall, ri, ro, pi, basis_degree, gauss_degree, nSamples, model_problem_name):

    # study name
    study_name = "boundary_fitted" 

    # Study Arrays
    nCases = len(Nr)

    # initialize plots
    titles = {}
    titles["vonmises"] = "Von Mises Stress"
    titles["meanstress"] = "Mean Stress"
    titles["ur"] = "Radial Displacement"
    titles["ut"] = "Circumfrencial Displacement"
    titles["uz"] = "Axial Displacement"
    titles["sigmarr"] = "Radial Stress"
    titles["sigmatt"] = "Hoop Stress"
    titles["sigmazz"] = "Axial Stress"
    titles["sigmart"] = "$ r - \\theta $ Shear Stress"
    titles["sigmarz"] = "r - z Shear Stress"
    titles["sigmatz"] = "$ \\theta $ - z Shear Stress"
    titles["epsrr"] = "Radial Strain"
    titles["sum"] = "Axial Stress from Sum of Computed Radial and Hoop Stress values"
    ylabels = {}
    ylabels["vonmises"] = "$\sigma_{vm} / \sigma_{0}$"
    ylabels["meanstress"] = "$\sigma_{mean} / \sigma_{0}$"
    ylabels["ur"] = "$u_{r} / u_{0}$"
    ylabels["ut"] = "$u_{\\theta}$"
    ylabels["uz"] = "$u_{z}$"
    ylabels["sigmarr"] = "$\sigma_{rr} / \sigma_{0}$"
    ylabels["sigmatt"] = "$\sigma_{\\theta \\theta} / \sigma_{0}$"
    ylabels["sigmazz"] = "$\sigma_{zz} / \sigma_{0}$"
    ylabels["sigmart"] = "$\sigma_{r\\theta}$"
    ylabels["sigmarz"] = "$\sigma_{rz}$"
    ylabels["sigmatz"] = "$\sigma_{\\theta z}$"
    ylabels["epsrr"] = "$\\epsilon_{rr}$"
    ylabels["sum"] = "$\\nu ( \sigma_{rr} + \sigma_{\\theta \\theta} )$"
    figs, axs = InitializePlots(titles.keys())


    # exact solution
    r_exact, vals_exact = ExactSolution(ri, ro, pi, nu_wall, E_wall, nSamples)

    # normalization factors
    max_vals = {}
    for key in vals_exact:
        max_val = np.max(np.abs(vals_exact[key]))
        max_vals[key] = 1 if max_val == 0 else max_val

    # Normalize
    #vals_exact = Normalize(max_vals, vals_exact)
    # Plot Exact Solution
    Plot(axs, r_exact, vals_exact, 'exact')

    # loop cases
    for i in range(nCases):
        # label
        label = str(Nr[i]) + " X " + str(Nt[i]) + " X " + str(1) + " elements"
        # 3D Plot
        PLOT3D = i == nCases - 1
        # Run
        r, vals, res = BoundaryFittedSolution(Nr[i], Nt[i], nu_wall, E_wall, ri, ro, pi, basis_degree, gauss_degree, PLOT3D, label, nSamples)
        # normalize
        #vals = Normalize(max_vals, vals)
        # Plot numerical solution
        Plot(axs, r, vals, label)
        # Plot solution residual
        PlotResidual(res, model_problem_name, study_name, label)


    # export plots
    Export(model_problem_name, study_name, figs, axs, titles, ylabels)

    # Close figs
    CloseFigs(figs)

    print("finished study: " + study_name)




def main():

    # DEFINE DEFAULT VALUES

    # model problem name
    model_problem_name = "cylinder_bodyfitted_polar_not_normalized"

    # outer radius
    ro = 3.2 / 2

    # inner radius
    ri = 1.75 / 2

    # inner pressure [mPa]
    pi = .012 

    # cylinder wall properties [mPa]
    nu_wall =  0.3
    E_wall  =  0.1

    # basis
    basis_degree = 2

    # quadrature order
    gauss_degree = 5

    # number of plot sample points
    nSamples = 100


    ########################################

    
    # Run studies
    Nr = [40, 80]
    Nt = [80, 160]
    BoundaryFittedMeshResolutionStudy(Nr, Nt, nu_wall, E_wall, ri, ro, pi, basis_degree, gauss_degree, nSamples, model_problem_name)


if __name__ == '__main__':
    cli.run(main)
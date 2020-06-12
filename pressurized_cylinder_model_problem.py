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


def Constrain(omega_topo, omega, basis_degree, BC_TYPE):
    # Constrain Omega
    sqr  = omega_topo.boundary['left'].integral('u_0 u_0 d:x' @ omega, degree = 2*basis_degree)
    sqr += omega_topo.boundary['bottom'].integral('u_1 u_1 d:x' @ omega, degree = 2*basis_degree)

    if BC_TYPE == "D":
        sqr += omega_topo.boundary['top'].integral('u_0 u_0 + u_1 u_1 d:x' @ omega, degree = 2*basis_degree)
        sqr += omega_topo.boundary['right'].integral('u_0 u_0 + u_1 u_1 d:x' @ omega, degree = 2*basis_degree)

    sqr += omega_topo.integral('u_2 u_2 d:x' @ omega, degree = 2*basis_degree)
    cons = solver.optimize('lhs', sqr, droptol=1e-15, linsolver='cg', linatol=1e-10, linprecon='diag')

    return cons


def Run(L, Nx, Ny, Nu, Nv, nu_wall, E_wall, nu_air, E_air, ri, ro, pi, basis_degree, gauss_degree, BC_TYPE, PLOT3D, label, nSamples):

    # mat prop functions
    class PoissonRatio(function.Pointwise):
        @staticmethod
        def evalf(x,y,z):
            return nu_air + (nu_wall - nu_air) * np.heaviside(x**2 + y**2 - ri**2 ,1) - (nu_wall - nu_air)*np.heaviside(x**2 + y**2 - ro**2 ,0)
        def _derivative(self, var, seen):
            return np.zeros(self.shape+var.shape)

    class YoungsModulus(function.Pointwise):
        @staticmethod
        def evalf(x,y,z):
            return E_air + (E_wall - E_air) * np.heaviside(x**2 + y**2 - ri**2 ,1) - (E_wall - E_air)*np.heaviside(x**2 + y**2 - ro**2 ,0)
        def _derivative(self, var, seen):
            return np.zeros(self.shape+var.shape)


    # thickness
    Navg = (Nx + Ny)/2
    t = L / (2 * Navg)
    Nz = 1

    # background mesh
    omega = function.Namespace()

    x = np.linspace(0, L, Nx+1)
    y = np.linspace(0, L, Ny+1)
    z = np.linspace(-t/2,t/2,Nz+1)

    omega_topo, omega.x = mesh.rectilinear([x,y,z])
    
    gamma = function.Namespace()

    u = np.linspace(0, np.pi / 2, Nu+1)
    v = np.linspace(-t/2, t/2, Nv+1)

    gamma_topo, gamma.uv = mesh.rectilinear([u,v])

    n_verts = len(u) * len(v)
    x_verts = [0]*(n_verts)
    y_verts = [0]*(n_verts)
    z_verts = [0]*(n_verts)

    t_x = [0]*(n_verts)
    t_y = [0]*(n_verts)
    t_z = [0]*(n_verts)

    k = 0
    for i in range(len(u)):
        for j in range(len(v)):
            x_verts[k] = ri * np.cos(u[i])
            y_verts[k] = ri * np.sin(u[i])
            z_verts[k] = v[j]
            t_x[k] = pi * np.cos(u[i])
            t_y[k] = pi * np.sin(u[i])
            t_z[k] = 0
            k = k + 1

    gamma.linbasis = gamma_topo.basis('spline',degree=1)
    gamma.xx = gamma.linbasis.dot(x_verts)
    gamma.xy = gamma.linbasis.dot(y_verts)
    gamma.xz = gamma.linbasis.dot(z_verts)
    gamma.tx = gamma.linbasis.dot(t_x)
    gamma.ty = gamma.linbasis.dot(t_y)
    gamma.tz = gamma.linbasis.dot(t_z)
    gamma.x_i = '<xx, xy, xz>_i'
    gamma.traction_i = '<tx, ty, tz>_i'

    # Add Mat Props functions to namespace
    omega.nu = PoissonRatio(omega.x[0], omega.x[1], omega.x[2])
    omega.E = YoungsModulus(omega.x[0], omega.x[1], omega.x[2])
    omega.mu = 'E / (2 (1 + nu))'
    omega.lmbda = 'E nu / ( (1 + nu) (1 - 2 nu) )'

    # Build Quadrature rule
    degree_gamma = 1
    sample_trimmed_gamma = gamma_topo.sample('gauss', degree_gamma)
    sample_trimmed_omega = locatesample(sample_trimmed_gamma, gamma.x, omega_topo, omega.x,10000000000)

    # Rebuild traction function on Omega
    omega.traction = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.traction))
    omega.Jgamma = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(function.J(gamma.x)))

    # Define Analysis
    omega.basis = omega_topo.basis('spline',degree = basis_degree)
    omega.ubasis = omega.basis.vector(3)
    omega.u_i = 'ubasis_ni ?lhs_n'
    omega.X_i = 'x_i + u_i'
    omega.strain_ij = '(u_i,j + u_j,i) / 2'
    omega.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'
    omega.meanstrain = 'strain_kk / 3'
    omega.meanstress = 'stress_kk / 3'
    omega.S_ij = 'stress_ij - (stress_kk) δ_ij / 3'
    omega.vonmises = 'sqrt(3 S_ij S_ij / 2)'
    omega.disp = 'sqrt(u_i u_i)'
    omega.r = 'sqrt( x_i x_i )'
    omega.cos = 'x_0 / r'
    omega.sin = 'x_1 / r'
    omega.Qinv_ij = '< < cos , sin , 0 >_j , < -sin , cos , 0 >_j , < 0 , 0 , 1 >_j >_i'
    omega.sigma_kl = 'stress_ij Qinv_kj Qinv_li '
    omega.du_i = 'Qinv_ij u_j'
    
    # Stiffness Matrix
    K = omega_topo.integral('ubasis_ni,j stress_ij d:x' @ omega, degree = gauss_degree)

    # Force Vector
    F = sample_trimmed_omega.integral('traction_i Jgamma ubasis_ni' @ omega)

    # Constrain Omega
    cons = Constrain(omega_topo, omega, basis_degree, BC_TYPE)

    # Solve
    lhs = solver.solve_linear('lhs', residual=K-F, constrain=cons, linsolver='cg', linatol=1e-7, linprecon='diag')

    samplepts = omega_topo.sample('gauss', gauss_degree)
    x = samplepts.eval(omega.x)
    E, nu = samplepts.eval([omega.E, omega.nu])
    meanstress, vonmises, disp = samplepts.eval(['meanstress', 'vonmises', 'du_i'] @ omega, lhs=lhs)
    sigmarr, sigmatt, sigmazz, sigmart, sigmatz, sigmarz = samplepts.eval(['sigma_00', 'sigma_11','sigma_22', 'sigma_01','sigma_12', 'sigma_02'] @ omega, lhs=lhs)

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


    # Plot Stress Results
    if PLOT3D == True:
        name = "pressurized_cylinder_model_problem"
        tri = GaussTri(omega_topo, gauss_degree)
        export.vtk( name ,tri, x, E=E, nu=nu, u=disp, sigmarr=sigmarr, sigmatt=sigmatt, sigmazz=sigmazz, meanstress=meanstress, vonmises=vonmises )


    # Define slice
    ns = function.Namespace()
    topo, ns.t = mesh.rectilinear([np.linspace(ri,ro,nSamples+1)])
    ns.rgeom_i = '< t_0 / sqrt(2), t_0 / sqrt(2), 0 >_i'
    ns.r = 't_0'

    # sample
    samplepts = topo.sample('gauss',1)
    pltpts = locatesample(samplepts, ns.rgeom, gauss_topo, gauss.x, 10000000000)
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

    print("finished case: " + label)

    return r, vals



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
    s.sigmazz = '(pi ri^2 - po ro^2) / (ro^2 - ri^2)'
    s.meanstress = '( sigmatt + sigmarr + sigmazz ) / 3'
    s.vonmises = 'sqrt(  ( (sigmarr - sigmatt)^2 + (sigmatt - sigmazz)^2 + (sigmazz - sigmarr)^2  ) / 2  )'
    s.C1 = '(1 - nu) ((ri^2 pi) / (ro^2 - ri^2)) / E'
    s.C2 = '(1 + nu) ((ri^2 ro^2 pi) / (ro^2 - ri^2)) / E'
    s.ur = 'C1 r + ( C2 / r )'

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

    return r, vals

def InitializePlots(keys):
    figs = {}
    axs = {}
    for key in keys:
        figs[key] = plt.figure()
        axs[key] = figs[key].add_subplot(111)
    return figs, axs

def Plot(axs, r, vals, label):
    for key in axs.keys():
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
        plt.close(figs[key])

def Normalize(normalization_factors, vals):
    for key in normalization_factors:
        vals[key] /= normalization_factors[key]
    return vals

def MeshResolutionStudy(ro, ri, pi, nu_wall, E_wall, nu_air, E_air, L, Nu, Nv, basis_degree, gauss_degree, nSamples, model_problem_name, BC_TYPE, N):

    # study name
    study_name = "mesh_resolution_" + BC_TYPE 

    # Study Arrays
    nCases = len(N)

    # initialize plots
    keys = ["vonmises", "meanstress", "ur", "ut", "uz", "sigmarr", "sigmatt", "sigmazz", "sigmart", "sigmarz", "sigmatz"]
    titles = {}
    titles["vonmises"] = "Von Mises Stress"
    titles["meanstress"] = "Mean Stress"
    titles["ur"] = "Radial Displacement"
    titles["ut"] = "Circumfrencial Displacement"
    titles["uz"] = "Axial Displacement"
    titles["sigmarr"] = "Radial Stress"
    titles["sigmatt"] = "Hoop Stress"
    titles["sigmazz"] = "Axial Stress"
    titles["sigmart"] = "$ r - \theta Shear Stress $"
    titles["sigmarz"] = "r - z Shear Stress"
    titles["sigmatz"] = "$ \theta - z Shear Stress $"
    ylabels = {}
    ylabels["vonmises"] = "$\sigma_{vm} / sigma_{0}$"
    ylabels["meanstress"] = "$\sigma_{mean} / sigma_{0}$"
    ylabels["ur"] = "$u_{r} / u_{0}$"
    ylabels["ut"] = "$u_{\theta}$"
    ylabels["uz"] = "$u_{z}$"
    ylabels["sigmarr"] = "$\sigma_{rr} / sigma_{0}$"
    ylabels["sigmatt"] = "$\sigma_{\theta \theta} / sigma_{0}$"
    ylabels["sigmazz"] = "$\sigma_{zz} / sigma_{0}$"
    ylabels["sigmart"] = "$\sigma_{rt}$"
    ylabels["sigmarz"] = "$\sigma_{rz}$"
    ylabels["sigmatz"] = "$\sigma_{tz}$"
    figs, axs = InitializePlots(keys)

    # exact solution
    r_exact, vals_exact = ExactSolution(ri, ro, pi, nu_wall, E_wall, nSamples)

    # normalization factors
    max_vals = {}
    for key in keys:
        max_val = np.max(np.abs(vals_exact[key]))
        max_vals[key] = 1 if max_val == 0 else max_val

    # Normalize
    vals_exact = Normalize(max_vals, vals_exact)
    # Plot Exact Solution
    Plot(axs, r_exact, vals_exact, 'exact')

    # loop cases
    for i in range(nCases):
        # label
        label = "N = " + str(N[i])
        # 3D Plot
        PLOT3D = i == nCases - 1
        # Run
        r, vals = Run(L, N[i], N[i], Nu, Nv, nu_wall, E_wall, nu_air, E_air, ri, ro, pi, basis_degree, gauss_degree, BC_TYPE, PLOT3D, label, nSamples)
        # Normalize
        vals = Normalize(max_vals, vals)
        # Plot numerical solution
        Plot(axs, r, vals, label)

    # export plots
    Export(model_problem_name, study_name, figs, axs, titles, ylabels)

    # close figs
    CloseFigs(figs)

    print("finished study: " + study_name)

def CompressibilityStudy(ro, ri, pi, E_wall, nu_air, E_air, L, Nx, Ny, Nu, Nv, basis_degree, gauss_degree, nSamples, model_problem_name, BC_TYPE, nu_wall):
    # study name
    study_name = "compressibility_" + BC_TYPE 

    # Study Arrays
    nCases = len(nu_wall)

    # initialize plots
    keys = ["vonmises", "meanstress", "sigmarr", "sigmatt", "sigmazz", "sigmart", "sigmarz", "sigmatz"]
    titles = {}
    titles["vonmises"] = "Von Mises Stress"
    titles["meanstress"] = "Mean Stress"
    titles["sigmarr"] = "Radial Stress"
    titles["sigmatt"] = "Hoop Stress"
    titles["sigmazz"] = "Axial Stress"
    titles["sigmart"] = "$ r - \theta Shear Stress $"
    titles["sigmarz"] = "r - z Shear Stress"
    titles["sigmatz"] = "$ \theta - z Shear Stress $"
    ylabels = {}
    ylabels["vonmises"] = "$\sigma_{vm} / sigma_{0}$"
    ylabels["meanstress"] = "$\sigma_{mean} / sigma_{0}$"
    ylabels["sigmarr"] = "$\sigma_{rr} / sigma_{0}$"
    ylabels["sigmatt"] = "$\sigma_{\theta \theta} / sigma_{0}$"
    ylabels["sigmazz"] = "$\sigma_{zz} / sigma_{0}$"
    ylabels["sigmart"] = "$\sigma_{rt}$"
    ylabels["sigmarz"] = "$\sigma_{rz}$"
    ylabels["sigmatz"] = "$\sigma_{tz}$"
    figs, axs = InitializePlots(keys)

    # exact solution
    r_exact, vals_exact = ExactSolution(ri, ro, pi, nu_wall[0], E_wall, nSamples)

    # normalization factors
    max_vals = {}
    for key in keys:
        max_val = np.max(np.abs(vals_exact[key]))
        max_vals[key] = 1 if max_val == 0 else max_val

    # Normalize
    vals_exact = Normalize(max_vals, vals_exact)
    # Plot Exact Solution
    Plot(axs, r_exact, vals_exact, 'exact')

    # loop cases
    for i in range(nCases):
        # label
        label = "$ \nu_{wall} $ = " + str(nu_wall[i])
        # 3D Plot
        PLOT3D = i == nCases - 1
        # Run
        r, vals = Run(L, Nx, Ny, Nu, Nv, nu_wall[i], E_wall, nu_air, E_air, ri, ro, pi, basis_degree, gauss_degree, BC_TYPE, PLOT3D, label, nSamples)
        # Normalize
        vals = Normalize(max_vals, vals)
        # Plot numerical solution
        Plot(axs, r, vals, label)

    # export plots
    Export(model_problem_name, study_name, figs, axs, titles, ylabels)

    # close figs
    CloseFigs(figs)

    print("finished study: " + study_name)


def main():

    # DEFINE DEFAULT VALUES

    # model problem name
    model_problem_name = "cylinder"

    # outer radius
    ro = 3.2 / 2

    # inner radius
    ri = 1.75 / 2

    # inner pressure [mPa]
    pi = .012 

    # cylinder wall properties [mPa]
    nu_wall =  0.3
    E_wall  =  .1

    # Air properties [mPa]
    nu_air  =  0.0
    E_air   =  0.00001

    # number immersed boundary elements
    Nu = 1000
    Nv = 10

    # basis
    basis_degree = 2

    # quadrature order
    gauss_degree = 3

    # Domain size
    L = 2 * ro

    # number voxels
    Nx = 100
    Ny = 100

    # number of plot sample points
    nSamples = 100

    # Boundary condition type: "D" for Dirichlet or "N" for Neumann
    BC_TYPE = "N"

    ########################################

    # Run studies

    # Mesh Resolution Study
    N = [50, 100, 150]
    MeshResolutionStudy(ro, ri, pi, nu_wall, E_wall, nu_air, E_air, L, Nu, Nv, basis_degree, gauss_degree, nSamples, model_problem_name, "D", N)
    MeshResolutionStudy(ro, ri, pi, nu_wall, E_wall, nu_air, E_air, L, Nu, Nv, basis_degree, gauss_degree, nSamples, model_problem_name, "N", N)

    # Compressibility Study
    nu_wall = [0.3, 0.4, 0.45, 0.49]
    CompressibilityStudy(ro, ri, pi, E_wall, nu_air, E_air, L, Nx, Ny, Nu, Nv, basis_degree, gauss_degree, nSamples, model_problem_name, "D", nu_wall)
    CompressibilityStudy(ro, ri, pi, E_wall, nu_air, E_air, L, Nx, Ny, Nu, Nv, basis_degree, gauss_degree, nSamples, model_problem_name, "N", nu_wall)




if __name__ == '__main__':
	cli.run(main)
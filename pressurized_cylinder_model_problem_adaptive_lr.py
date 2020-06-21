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


def Solve(res, cons, callback):
    return solver.solve_linear('lhs', residual=res, constrain=cons, linsolver='cg', linatol=1e-14, linprecon='diag', lincallback=callback)


def RefineBySDF(topo, sdf, nrefine):
    refined_topo = topo
    for n in range(nrefine):
        elems_to_refine = []
        k = 0
        bez = refined_topo.sample('bezier',2)
        sd = bez.eval(sdf)
        sd = sd.reshape( [int(len(sd)/8), 8] )
        for i in range(len(sd)):
            if any(np.sign(sdval) != np.sign(sd[i][0]) for sdval in sd[i,:])
            k = k + 1
        refined_topo = refined_topo.refined_by(refined_topo.transforms[np.array(elems_to_refine)])
    return refined_topo


def Run(L, Nx, Ny, Nu, Nv, nu_wall, E_wall, nu_air, E_air, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, label, nSamples, PLOT3D):

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
    z = np.linspace(-t,t,Nz+1)

    omega_topo, omega.x = mesh.rectilinear([x,y,z])
    
    # immersed boundary mesh
    gamma = function.Namespace()

    u = np.linspace(0, np.pi / 2, Nu+1)
    v = np.linspace(-t, t, Nv+1)

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

    # signed distance fields
    omega.sdfri = 'x_i x_i - ri^2'
    omega.sdfwall = -1 + 2 * ( np.heaviside(x**2 + y**2 - ri**2 ,1) - np.heaviside(x**2 + y**2 - ro**2 ,0) )

    # refine background topology for basis
    refined_omega_topo = RefineBySDF(omega_topo, omega.sdfri, nref)
    omega.basis = refined_omega_topo.basis('th-spline', degree = basis_degree)

    # refine background topology for quadrature rule
    refined_quadrature_topo = RefineBySDF(omega_topo, omega.sdfwall, nqref)
    gauss_sample = refined_quadrature_topo.sample('gauss', gauss_degree)

    # Build Immersed Boundary Quadrature Rule
    degree_gamma = 1
    sample_trimmed_gamma = gamma_topo.sample('gauss', degree_gamma)
    sample_trimmed_omega = locatesample(sample_trimmed_gamma, gamma.x, omega_topo, omega.x,10000000000)

    # Rebuild traction function on Omega
    omega.traction = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.traction))
    omega.Jgamma = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(function.J(gamma.x)))

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
    omega.eps_kl =  'strain_ij Qinv_kj Qinv_li '
    
    # Stiffness Matrix
    K = gauss_sample.integral('ubasis_ni,j stress_ij d:x' @ omega)

    # Force Vector
    F = sample_trimmed_omega.integral('traction_i Jgamma ubasis_ni' @ omega)

    # Constrain Omega
    sqr  = omega_topo.boundary['left'].integral('u_0 u_0 d:x' @ omega, degree = 2*basis_degree)
    sqr += omega_topo.boundary['bottom'].integral('u_1 u_1 d:x' @ omega, degree = 2*basis_degree)

    if BC_TYPE == "D":
        sqr += omega_topo.boundary['top'].integral('( u_0 u_0 + u_1 u_1 ) d:x' @ omega, degree = 2*basis_degree)
        sqr += omega_topo.boundary['right'].integral('( u_0 u_0 + u_1 u_1 ) d:x' @ omega, degree = 2*basis_degree)

    sqr += omega_topo.integral('u_2 u_2 d:x' @ omega, degree = 2*basis_degree)
    cons = solver.optimize('lhs', sqr, droptol=1e-15, linsolver='cg', linatol=1e-10, linprecon='diag')

    # Initialize Residual Vector
    residuals = []
    def AddResiualNorm(res):
        residuals.append(res)

    # Solve
    lhs = Solve(K-F, cons, AddResiualNorm)    

    # Plot Stress Results
    if PLOT3D == True:
        samplepts = refined_omega_topo.sample('bezier', 2)
        x = samplepts.eval(omega.x)
        E, nu = samplepts.eval([omega.E, omega.nu])
        meanstress, vonmises, disp = samplepts.eval(['meanstress', 'vonmises', 'du_i'] @ omega, lhs=lhs)
        sigmarr, sigmatt, sigmazz, sigmart, sigmatz, sigmarz = samplepts.eval(['sigma_00', 'sigma_11','sigma_22', 'sigma_01','sigma_12', 'sigma_02'] @ omega, lhs=lhs)
        name = "pressurized_cylinder_model_problem"
        export.vtk( name, samplepts.tri, x, E=E, nu=nu, u=disp, sigmarr=sigmarr, sigmatt=sigmatt, sigmazz=sigmazz, meanstress=meanstress, vonmises=vonmises)


    # Define slice
    ns = function.Namespace()
    topo, ns.t = mesh.rectilinear([np.linspace(ri,ro,nSamples+1)]) 
    ns.angle = np.pi / 4
    ns.rgeom_i = '< t_0 cos(angle), t_0 sin(angle), 0 >_i'
    ns.r = 't_0'

    # sample
    samplepts = topo.sample('gauss',1)
    pltpts = locatesample(samplepts, ns.rgeom, omega_topo, omega.x, 10000000000)
    r = samplepts.eval(ns.r)
    vonmises = pltpts.eval(omega.vonmises, lhs=lhs)
    meanstress = pltpts.eval(omega.meanstress, lhs=lhs)
    ur = pltpts.eval(omega.du[0], lhs=lhs)
    ut = pltpts.eval(omega.du[1], lhs=lhs)
    uz = pltpts.eval(omega.du[2], lhs=lhs)
    E = pltpts.eval(omega.E)
    nu = pltpts.eval(omega.nu)
    sigmarr = pltpts.eval('sigma_00' @ omega, lhs=lhs)
    sigmatt = pltpts.eval('sigma_11' @ omega, lhs=lhs)
    sigmazz = pltpts.eval('sigma_22' @ omega, lhs=lhs)
    sigmart = pltpts.eval('sigma_01' @ omega, lhs=lhs)
    sigmarz = pltpts.eval('sigma_02' @ omega, lhs=lhs)
    sigmatz = pltpts.eval('sigma_12' @ omega, lhs=lhs)
    epsrr = pltpts.eval('eps_00' @ omega, lhs=lhs)
    epstt = pltpts.eval('eps_11' @ omega, lhs=lhs)
    epszz = pltpts.eval('eps_22' @ omega, lhs=lhs)
    epsrt = pltpts.eval('eps_01' @ omega, lhs=lhs)
    epsrz = pltpts.eval('eps_02' @ omega, lhs=lhs)
    epstz = pltpts.eval('eps_12' @ omega, lhs=lhs)

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

    print("finished case: " + label)

    return r, vals, residuals



def ExactSolution(ri, ro, pi, nu_wall, E_wall, nSamples):

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


def Export(axs, figs, dir, titles, xlabels, ylabels):

    for key in axs:
        axs[key].set_title(titles[key])
        axs[key].set_xlabel(xlabels[key])
        axs[key].set_ylabel(ylabels[key])
        axs[key].legend()
        fdir = "Results/" + dir
        fname = titles[key]
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


def PlotCase(axs, plots, r, vals, case_name):
    for key in axs:
        line = axs[key].plot(r,vals[plots[key][0]],label=case_name)
        col = line[0].get_color()
        for plot in range(1, len(plots[key])):
            axs[key].plot(r,vals[plot],color=col)



def CompressibilityStudy(L, Nx, Ny, Nu, Nv, nu_wall, E_wall, nu_air, E_air, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name):
    # study name
    study_name = "compressibility"

    # define figures
    plot_keys = ["stress", "disp"]
    figs, axs = InitializePlots(keys)

    # Define plots
    plots = {}
    plots["stress"] = ["sigmatt", "sigmazz", "sigmarr"]
    plots["disp"] = ["ur", "ut"]

    # Y Labels
    ylabels = {}
    ylabels["stress"] = "[MPa]"
    ylabels["disp"] = "[mm]"

    # X Labels
    xlabels = {}
    xlabels["stress"] = "r [mm]"
    xlabels["disp"] = "r [mm]"

    # Titles
    titles = {}
    titles["stress"] = "Stress Components"
    titles["disp"] = "Raidial Displacement"


    # inputs
    PLOT3D = False
    nSamples = 100

    # exact solution
    r_exact, vals_exact = ExactSolution(ri, ro, pi, nu_wall, E_wall, nSamples)
    PlotCase(axs, plots, r_exact, vals_exact, "exact")

    # cases
    ncases = len(nu_wall)
    for i in range(ncases):
        case_name = "$\\nu = $" + str(nu_wall[i])
        r, vals, res = Run(L, Nx, Ny, Nu, Nv, nu_wall[i], E_wall, nu_air, E_air, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, label, nSamples, PLOT3D)
        PlotCase(axs, plots, r, vals, case_name)

    # export figures
    dir = model_problem_name + "/" + study_name
    Export(axs, figs, dir, titles, xlabels, ylabels)

    # close figs
    CloseFigs(figs)



def main():

    # DEFINE DEFAULT VALUES

    # model problem name
    model_problem_name = "adaptive_lr_cylinder"

    # outer radius
    ro = 3.2 / 2

    # inner radius
    ri = 1.75 / 2

    # inner pressure [mPa]
    pi = .012 

    # cylinder wall properties [mPa]
    nu_wall =  0.3
    E_wall  =  0.1

    # Air properties [mPa]
    nu_air  =  0.0
    E_air   =  0.000001 * E_wall

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
    Nx = 5
    Ny = 5

    # number of plot sample points
    nSamples = 100

    # Boundary condition type: "D" for Dirichlet or "N" for Neumann
    BC_TYPE = "D"

    # nrefine for basis
    nref = 2

    # nrefine for quadrature rule
    nqref = 3

    ########################################

    
    # Run studies
    poisson_ratios = [0.3, 0.4, 0.45, 0.49]
    CompressibilityStudy(L, Nx, Ny, Nu, Nv, poisson_ratios, E_wall, nu_air, E_air, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name)


if __name__ == '__main__':
	cli.run(main)
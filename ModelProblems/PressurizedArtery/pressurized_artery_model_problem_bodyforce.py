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

def CalcYoungsModulus(G, nu):
    return 2 * G * (1 + nu)

def Solve(res, cons, callback):
    return solver.solve_linear('lhs', residual=res, constrain=cons, linsolver='cg', linatol=1e-14, linprecon='diag', lincallback=callback)


def RefineBySDF(topo, sdf, nrefine):
    refined_topo = topo
    for n in range(nrefine):
        elems_to_refine = []
        k = 0
        bez = refined_topo.sample('bezier',2)
        sd = bez.eval(sdf)
        sd = sd.reshape( [len(sd)//8, 8] )
        for i in range(len(sd)):
            if any(np.sign(sdval) != np.sign(sd[i][0]) for sdval in sd[i,:]):
                elems_to_refine.append(k)
            k = k + 1
        refined_topo = refined_topo.refined_by(refined_topo.transforms[np.array(elems_to_refine)])
    return refined_topo

def BloodElements(topo, Nt, Nrb, Nra, Nrh):
    ielems = np.ndarray([Nt * Nrb], dtype=int)
    k = 0
    for i in range(Nt):
        for j in range(Nrb):
            ielems[k] = i * (Nrb + Nra + Nrh) + j
            k+=1
    return topology.SubsetTopology(topo, [topo.references[i] if i in ielems else topo.references[i].empty for i in range(len(topo.references))])

def BloodTopBoundary(boundary_topo, geom, Rb, TOL):
    verts = boundary_topo.sample('bezier', 2).eval(geom)
    ielems = []
    k = 0
    for i in range(len(boundary_topo)):
        isOnBoundary = True
        for j in range(4):
            if np.abs(verts[k][0]**2 + verts[k][1]**2 - Rb**2) > TOL:
                isOnBoundary = False
                k += 1
                break
            k += 1
        if isOnBoundary:
            ielems.append(i)
    ielems = np.array(ielems,dtype=int)
    print(ielems)
    return topology.SubsetTopology(boundary_topo, [boundary_topo.references[i] if i in ielems else boundary_topo.references[i].empty for i in range(len(boundary_topo.references))])

def BoundaryFittedSolution(L, eps, Nt, Nrb, Nra, Nrh, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nSamples, PLOT3D, expr):
    # mat prop functions
    class PoissonRatio(function.Pointwise):
        @staticmethod
        def evalf(x,y,z):
            val0 = nu_b * (1.0 - np.heaviside(x**2 + y**2 - ri**2 ,1))
            val1 = nu_a * (np.heaviside(x**2 + y**2 - ri**2 ,1) - np.heaviside(x**2 + y**2 - ro**2 ,0))
            val2 = nu_h * (np.heaviside(x**2 + y**2 - ro**2 ,0))
            return val0 + val1 + val2
        def _derivative(self, var, seen):
            return np.zeros(self.shape+var.shape)
    class YoungsModulus(function.Pointwise):
        @staticmethod
        def evalf(x,y,z):
            val0 = E_b * (1.0 - np.heaviside(x**2 + y**2 - ri**2 ,1))
            val1 = E_a * (np.heaviside(x**2 + y**2 - ri**2 ,1) - np.heaviside(x**2 + y**2 - ro**2 ,0))
            val2 = E_h * (np.heaviside(x**2 + y**2 - ro**2 ,0))
            return val0 + val1 + val2
        def _derivative(self, var, seen):
            return np.zeros(self.shape+var.shape)


    # thickness
    Navg = ((Nrb + Nra + Nrh)/3 + Nt)/2
    t = (L - eps) / (2 * Navg)
    Nz = 1

    # background mesh
    omega = function.Namespace()

    th = np.linspace(0, np.pi/2, Nt + 1)
    rb = np.linspace(eps,ri,Nrb+1)
    ra = np.linspace(ri,ro,Nra+1)
    rh = np.linspace(ro,L,Nrh+1)
    r = np.concatenate([rb, ra[1:len(ra)-1], rh])
    z = np.linspace(-t,t,Nz+1)

    # build knot vectors
    knotvals = [th,r,z]
    knotmults = [np.ones(len(th),dtype=int), np.ones(len(r),dtype=int), np.ones(len(z),dtype=int)]
    for i in range(len(knotmults)):
    	knotmults[i][0] = basis_degree
    	knotmults[i][-1] = basis_degree
    knotmults[1][len(rb)-1] = basis_degree
    knotmults[1][len(rb) - 1 + len(ra) - 1] = basis_degree

    omega = function.Namespace()
    omega_topo, omega.uvw = mesh.rectilinear([th,r,z])
    omega.x_i = '<uvw_1 cos(uvw_0) , uvw_1 sin(uvw_0) , uvw_2 >_i'

    # build basis
    omega.basis = omega_topo.basis('spline', degree = basis_degree, knotvalues = knotvals, knotmultiplicities=knotmults)
    
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

    # project
    omega.lmbda = omega_topo.projection(omega.lmbda, onto=omega_topo.basis('spline',degree=3), geometry=omega.x, ptype='convolute', ischeme='gauss{}'.format(3),solver='cg', atol=1e-5, precon='diag')
    omega.mu = omega_topo.projection(omega.mu, onto=omega_topo.basis('spline',degree=3), geometry=omega.x, ptype='convolute', ischeme='gauss{}'.format(3),solver='cg', atol=1e-5, precon='diag')
    
    # Build Immersed Boundary Quadrature Rule
    degree_gamma = 1
    sample_gamma = gamma_topo.sample('gauss', degree_gamma)
    #sample_omega = locatesample(sample_gamma, gamma.x, omega_topo, omega.x,1e-7)

    # Rebuild traction function on Omega
    #omega.traction = sample_omega.asfunction(sample_gamma.eval(gamma.traction))
    #omega.Jgamma = sample_omega.asfunction(sample_gamma.eval(function.J(gamma.x)))

    # define analysis
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
    omega.sigmatt = 'sigma_11'
    omega.sigmarr = 'sigma_00'
    omega.sigmazz = 'sigma_22'
    omega.ur = 'du_0'
    
    # the blood problem
    blood = function.Namespace()
    blood_topo = BloodElements(omega_topo, Nt, Nrb, Nra, Nrh)
    blood.V = blood_topo.integral('d:x' @ omega, degree=4).eval()
    blood.x = omega.x
    blood.basis = blood_topo.basis('spline', degree=basis_degree)
    blood.ubasis = blood.basis.vector(3)
    blood.E = E_b
    blood.nu = nu_b
    blood.mu = 'E / (2 (1 + nu))'
    blood.u_i = 'ubasis_ni ?alpha_n'
    blood.lmbda = 'E nu / ( (1 + nu) (1 - 2 nu) )'
    blood.strain_ij = '(u_i,j + u_j,i) / 2'
    blood.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'
    blood.pi = pi
    Kblood = blood_topo.integral('ubasis_ni,j stress_ij d:x' @ blood, degree=6)
    blood_topo_top_boundary = BloodTopBoundary(blood_topo.boundary, blood.x, ri, .0001)
    print("blood top boundary size: " + str(len(blood_topo_top_boundary)))
    Fblood = blood_topo_top_boundary.integral('pi n_i ubasis_ni d:x' @ blood, degree=6)
    # Constrain Omega
    sqr_blood  = blood_topo.boundary['right'].integral('u_0 u_0 d:x' @ blood, degree=6)
    sqr_blood += blood_topo.boundary['left'].integral('u_1 u_1 d:x' @ blood, degree=6)
    sqr_blood += blood_topo.integral('u_2 u_2 d:x' @ blood, degree=6)
    cons_blood = solver.optimize('alpha', sqr_blood, droptol=1e-15, linsolver='cg', linatol=1e-15, linprecon='diag')

    alpha = solver.solve_linear('alpha', residual=Kblood-Fblood, constrain=cons_blood, linsolver='cg', linatol=1e-14, linprecon='diag')

    # Stiffness Matrix
    K = omega_topo.integral('ubasis_ni,j stress_ij d:x' @ omega, degree=7)

    # Force Vector
    #F = sample_omega.integral('traction_i Jgamma ubasis_ni' @ omega)
    blood.bloodindicator = omega_topo.indicator(blood_topo)
    blood.fullbasis = omega.ubasis

    F = blood_topo.integral('stress_ij,j fullbasis_ni d:x' @ blood(alpha=alpha), degree=7)


    # Constrain Omega
    sqr  = omega_topo.boundary['right'].integral('u_0 u_0 d:x' @ omega, degree=6)
    sqr += omega_topo.boundary['left'].integral('u_1 u_1 d:x' @ omega, degree=6)
    sqr += omega_topo.integral('u_2 u_2 d:x' @ omega, degree=6)
    cons = solver.optimize('lhs', sqr, droptol=1e-15, linsolver='cg', linatol=1e-15, linprecon='diag')

    # Initialize Residual Vector
    residuals = []
    def AddResiualNorm(res):
        residuals.append(res)

    # Solve
    lhs = Solve(K-F, cons, AddResiualNorm)  

    # Plot Stress Results
    if PLOT3D == True:
        samplepts = omega_topo.sample('bezier', 2)
        x = samplepts.eval(omega.x)
        E, nu = samplepts.eval([omega.E, omega.nu])
        meanstress, vonmises, disp = samplepts.eval(['meanstress', 'vonmises', 'du_i'] @ omega, lhs=lhs)
        sigmarr, sigmatt, sigmazz, sigmart, sigmatz, sigmarz = samplepts.eval(['sigma_00', 'sigma_11','sigma_22', 'sigma_01','sigma_12', 'sigma_02'] @ omega, lhs=lhs)
        name = "pressurized_cylinder_model_problem"
        export.vtk( name, samplepts.tri, x, E=E, nu=nu, u=disp, sigmarr=sigmarr, sigmatt=sigmatt, sigmazz=sigmazz, meanstress=meanstress, vonmises=vonmises)

    # Define slice
    ns = function.Namespace()
    eps = (ro - ri) / (2 * Nra)
    topo, ns.t = mesh.rectilinear([np.linspace(ri+eps,ro-eps,nSamples+1)])
    ns.angle = np.pi / 5
    ns.rgeom_i = '< t_0 cos(angle), t_0 sin(angle), 0 >_i'
    ns.r = 't_0'

    # sample
    #samplepts = topo.sample('gauss',1)
    #pltpts = locatesample(samplepts, ns.rgeom, omega_topo, omega.x, 1e-7)

    # build sample
    nsamplesperelem = 20
    # build ielem array
    n = Nra + Nrb + Nrh
    xis = np.ndarray([n*nsamplesperelem,3], dtype=float)
    ielems = np.ndarray([n*nsamplesperelem], dtype=int)
    print(xis.shape)
    m = int(Nt / 2)
    if Nt%2 == 0:
        for i in range(n):
            ielem = n * m + i
            xi = 0
            eta = np.linspace(0,1,nsamplesperelem)
            for j in range(nsamplesperelem):
                xis[i*nsamplesperelem+j]=[xi,eta[j],0.5]
                ielems[i*nsamplesperelem+j] = ielem
    else:
        for i in range(n):
            ielem = n * m + i
            xi = 0.5
            eta = np.linspace(0,1,nsamplesperelem)
            for j in range(nsamplesperelem):
                xis[i*nsamplesperelem+j]=[xi,eta[j],0.5]
                ielems[i*nsamplesperelem+j] = ielem

    pltpts = omega_topo._sample(ielems, xis)
    print(pltpts.eval(omega.x))
    # evaluate expressions
    vals_artery = {}
    r_artery = []
    for rval in r:
        if ri <= rval and rval <= ro:
            r_artery.append(rval)
    for key in expr:
        evals = pltpts.eval(expr[key] @ omega, lhs=lhs)
        # vals[key] = pltpts.eval(expr[key] @ omega, lhs=lhs)
        vals = []
        for val, rval in zip(evals, r):
            if ri <= rval and rval <= ro:
                vals.append(val)
        vals_artery[key] = vals

    r = pltpts.eval(omega.r)


    #return r, vals, residuals
    return r_artery, vals_artery, residuals



def Run(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, nSamples, PLOT3D, expr):

    # mat prop functions
    class PoissonRatio(function.Pointwise):
        @staticmethod
        def evalf(x,y,z):
            val0 = nu_b * (1.0 - np.heaviside(x**2 + y**2 - ri**2 ,1))
            val1 = nu_a * (np.heaviside(x**2 + y**2 - ri**2 ,1) - np.heaviside(x**2 + y**2 - ro**2 ,0))
            val2 = nu_h * (np.heaviside(x**2 + y**2 - ro**2 ,0))
            return val0 + val1 + val2
        def _derivative(self, var, seen):
            return np.zeros(self.shape+var.shape)
    class YoungsModulus(function.Pointwise):
        @staticmethod
        def evalf(x,y,z):
            val0 = E_b * (1.0 - np.heaviside(x**2 + y**2 - ri**2 ,1))
            val1 = E_a * (np.heaviside(x**2 + y**2 - ri**2 ,1) - np.heaviside(x**2 + y**2 - ro**2 ,0))
            val2 = E_h * (np.heaviside(x**2 + y**2 - ro**2 ,0))
            return val0 + val1 + val2
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

    # signed distance fields
    omega.ri = ri
    omega.ro = ro
    omega.sdfri = 'x_i x_i - ri^2'
    omega.sdfro = 'x_i x_i - ro^2'

    # refine background topology for basis
    refined_omega_topo = RefineBySDF(omega_topo, omega.sdfri, nref)
    #refined_omega_topo = RefineBySDF(refined_omega_topo, omega.sdfro, nref)
    omega.basis = refined_omega_topo.basis('th-spline', degree = basis_degree)

    # refine background topology for quadrature rule
    refined_quadrature_topo = RefineBySDF(refined_omega_topo, omega.sdfri, nqref)
    refined_quadrature_topo = RefineBySDF(refined_quadrature_topo, omega.sdfro, nqref + nref)
    gauss_sample = refined_quadrature_topo.sample('gauss', gauss_degree)

    # Build Immersed Boundary Quadrature Rule
    degree_gamma = 1
    sample_gamma = gamma_topo.sample('gauss', degree_gamma)
    sample_omega = locatesample(sample_gamma, gamma.x, refined_omega_topo, omega.x,1e-7)

    # Rebuild traction function on Omega
    omega.traction = sample_omega.asfunction(sample_gamma.eval(gamma.traction))
    omega.Jgamma = sample_omega.asfunction(sample_gamma.eval(function.J(gamma.x)))

    # define analysis
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
    omega.sigmatt = 'sigma_11'
    omega.sigmarr = 'sigma_00'
    omega.sigmazz = 'sigma_22'
    omega.ur = 'du_0'

    # Stiffness Matrix
    K = gauss_sample.integral('ubasis_ni,j stress_ij d:x' @ omega)

    # Force Vector
    F = sample_omega.integral('traction_i Jgamma ubasis_ni' @ omega)

    # Constrain Omega
    sqr  = refined_omega_topo.boundary['left'].integral('u_0 u_0 d:x' @ omega, degree = 2*basis_degree)
    sqr += refined_omega_topo.boundary['bottom'].integral('u_1 u_1 d:x' @ omega, degree = 2*basis_degree)

    if BC_TYPE == "D":
        sqr += refined_omega_topo.boundary['top'].integral('( u_0 u_0 + u_1 u_1 ) d:x' @ omega, degree = 2*basis_degree)
        sqr += refined_omega_topo.boundary['right'].integral('( u_0 u_0 + u_1 u_1 ) d:x' @ omega, degree = 2*basis_degree)

    sqr += refined_omega_topo.integral('u_2 u_2 d:x' @ omega, degree = 2*basis_degree)
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
    pltpts = locatesample(samplepts, ns.rgeom, refined_omega_topo, omega.x, 1e-7)

    # evaluate expressions
    vals = {}
    for key in expr:
        vals[key] = pltpts.eval(expr[key] @ omega, lhs=lhs)

    r = pltpts.eval(omega.r)


    return r, vals, residuals


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


def Export(axs, figs, dir, titles, xlabels, ylabels, ylims):
    i = 0
    for key in axs:
        title = "plot_" + str(i)
        if key in titles:
            title = titles[key]
            axs[key].set_title(title)
        if key in xlabels:
            axs[key].set_xlabel(xlabels[key])
        else:
            axs[key].set_xlabel("r [mm]")
        if key in ylabels:
            axs[key].set_ylabel(ylabels[key])
        else:
            axs[key].set_ylabel("[MPa]")
        axs[key].legend()
        if key in ylims:
            axs[key].set_ylim(ylims[key])
        fdir = "Results/" + dir
        fname = title
        fext = ".png"
        fpath = fdir + "/" + fname + fext
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        figs[key].savefig(fpath)
        print("saved /heartflow/" + fpath)
        i = i + 1

def WriteAnalysisProperties(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name, dir):
    OUT_DIR = "Results/" + dir
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    path = OUT_DIR + "/" + "analysis_properties.txt"
    f = open(path, "w+")
    f.write(model_problem_name + "\n")
    f.write("L = " + str(L )+ "\n")
    f.write("(Nx, Ny, Nz) = " + str((Nx,Ny,1))+ "\n")
    f.write("(Nu, Nv) = " + str((Nu,Nv))+ "\n")
    f.write("nu_blood = " + str(nu_b)+ "\n")
    f.write("E_blood = " + str(E_b)+ "\n")
    f.write("nu_artery = " + str(nu_a)+ "\n")
    f.write("E_artery = " + str(E_a)+ "\n")
    f.write("nu_heart = " + str(nu_h)+ "\n")
    f.write("E_heart = " + str(E_h)+ "\n")
    f.write("(ri, ro) = " + str((ri, ro))+ "\n")
    f.write("pi = " + str(pi)+ "\n")
    f.write("basis degree = " + str(basis_degree)+ "\n")
    f.write("gauss_degree = " + str(gauss_degree)+ "\n")
    f.write("Levels of mesh refinement = " + str(nref)+ "\n")
    f.write("Levels of quadrature refinement = " + str(nqref)+ "\n")
    f.write("BC Type = " + str(BC_TYPE)+ "\n")
    f.close()
    print("saved /heartflow/" + path)

def CloseFigs(figs):
    for key in figs:
        plt.close( figs[key] )

def Normalize(normalization_factors, vals):
    for key in vals:
        if key in normalization_factors:
            vals[key] /= normalization_factors[key]
    return vals

def PlotExactCase(axs, plots, r, vals):
    for key in axs:
        line = axs[key].plot(r,vals[plots[key][0]],label="exact",color='black',linestyle='dashed')
        for i in range(1, len(plots[key])):
            axs[key].plot(r,vals[plots[key][i]],color='black',linestyle='dashed')
def PlotCase(axs, plots, r, vals, case_name):
    for key in axs:
        line = axs[key].plot(r,vals[plots[key][0]],label=case_name)
        col = line[0].get_color()
        for i in range(1, len(plots[key])):
            axs[key].plot(r,vals[plots[key][i]],color=col)

def PlotInterfaces(axs, plots, ri, ro):
    for key in axs:
        ylims = axs[key].get_ylim()
        axs[key].plot([ri, ri],[ylims[0], ylims[1]],'k--')
        axs[key].plot([ro, ro],[ylims[0], ylims[1]],'k--')

def BoundaryFittedStudy(L, Nt, Nr, Nu, Nv, eps, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, model_problem_name, study_name = "boundary_fitted_bf"):

    # inputs
    PLOT3D = False
    nSamples = 100

    # define figures
    plot_keys = ["sigmazz","sigmarr","sigmatt", "disp"]
    figs, axs = InitializePlots(plot_keys)

    # Define Function Expressions to sample
    expr = {}
    expr["r"] = "r"
    expr["sigmatt"] = "sigmatt"
    expr["sigmarr"] = "sigmarr"
    expr["sigmazz"] = "sigmazz"
    expr["ur"] = "ur"

    # Define plots
    plots = {}
    plots["sigmatt"] = ["sigmatt"]
    plots["sigmarr"] = ["sigmarr"]
    plots["sigmazz"] = ["sigmazz"]
    plots["disp"] = ["ur"]

    # Y Labels
    ylabels = {}
    ylabels["sigmatt"] = "[MPa]"
    ylabels["sigmarr"] = "[MPa]"
    ylabels["sigmazz"] = "[MPa]"
    ylabels["disp"] = "[mm]"

    # X Labels
    xlabels = {}
    xlabels["sigmatt"] = "r [mm]"
    xlabels["sigmazz"] = "r [mm]"
    xlabels["sigmarr"] = "r [mm]"
    xlabels["disp"] = "r [mm]"

    # Titles
    titles = {}
    titles["sigmatt"] = "sigmatt"
    titles["sigmarr"] = "sigmarr"
    titles["sigmazz"] = "sigmazz"
    titles["disp"] = "Radial Displacement"

    # exact solution
    # r_exact, vals_exact = ExactSolution(ri, ro, pi, nu_wall, E_wall, nSamples, expr)
    # PlotExactCase(axs, plots, r_exact, vals_exact)

    # y axis limits
    ylims = {}

    # cases
    ncases = len(Nt)
    for i in range(ncases):
        #Nrb = np.max( [1, int( np.floor( Nr[i] * ri / L ) )] )
        #Nrh = np.max( [1, int( np.floor( Nr[i] * (L - ro) / L ) ) ])
        #Nra = np.max([1, Nr[i] - Nrb - Nrh])
        Nrb = np.max([1, int(Nr[i]/3)])
        Nrh = np.max([1, int(Nr[i]/3)])
        Nra = Nr[i]
        print("elems: " + str((Nrb, Nra, Nrh)))
        case_name = "$N_{\\theta} \\times N_{r}$ = " + str(Nt[i]) + "$ \\times$ " + str(Nrb + Nrh + Nra)
        PLOT3D = i==0
        r, vals, res = BoundaryFittedSolution(L, eps, Nt[i], Nrb, Nra, Nrh, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nSamples, PLOT3D, expr)
        PlotCase(axs, plots, r, vals, case_name)
        print("finished case: " + case_name)

    PlotInterfaces(axs, plots, ri, ro)

    # export figures
    dir = model_problem_name + "/" + study_name
    Export(axs, figs, dir, titles, xlabels, ylabels, ylims)
    # close figs
    CloseFigs(figs)


def MeshRefinementStudy(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name, study_name = "mesh_refinement"):

    # inputs
    PLOT3D = False
    nSamples = 100

    # define figures
    plot_keys = ["stress", "disp"]
    figs, axs = InitializePlots(plot_keys)

    # Define Function Expressions to sample
    expr = {}
    expr["r"] = "r"
    expr["sigmatt"] = "sigmatt"
    expr["sigmarr"] = "sigmarr"
    expr["sigmazz"] = "sigmazz"
    expr["ur"] = "ur"

    # Define plots
    plots = {}
    plots["stress"] = ["sigmatt", "sigmazz", "sigmarr"]
    plots["disp"] = ["ur"]

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
    titles["disp"] = "Radial Displacement"

    # exact solution
    # r_exact, vals_exact = ExactSolution(ri, ro, pi, nu_wall, E_wall, nSamples, expr)
    # PlotExactCase(axs, plots, r_exact, vals_exact)

    # y axis limits
    ylims = {}

    # cases
    ncases = len(Nx)
    for i in range(ncases):
        case_name = "$N_{x} \\times N_{y}$ = " + str(Nx[i]) + "$ \\times$ " + str(Ny[i])
        r, vals, res = Run(L, Nx[i], Ny[i], Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, nSamples, PLOT3D, expr)
        PlotCase(axs, plots, r, vals, case_name)
        print("finished case: " + case_name)

    # export figures
    dir = model_problem_name + "/" + study_name
    Export(axs, figs, dir, titles, xlabels, ylabels, ylims)
    WriteAnalysisProperties(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name, dir)

    # close figs
    CloseFigs(figs)

def LocalRefinementStudy(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name, study_name = "local_refinement"):

    # inputs
    PLOT3D = False
    nSamples = 100

    # define figures
    plot_keys = ["stress", "disp"]
    figs, axs = InitializePlots(plot_keys)

    # Define Function Expressions to sample
    expr = {}
    expr["r"] = "r"
    expr["sigmatt"] = "sigmatt"
    expr["sigmarr"] = "sigmarr"
    expr["sigmazz"] = "sigmazz"
    expr["ur"] = "ur"

    # Define plots
    plots = {}
    plots["stress"] = ["sigmatt", "sigmazz", "sigmarr"]
    plots["disp"] = ["ur"]

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
    titles["disp"] = "Radial Displacement"

    # exact solution
    # r_exact, vals_exact = ExactSolution(ri, ro, pi, nu_wall, E_wall, nSamples, expr)
    # PlotExactCase(axs, plots, r_exact, vals_exact)

    # y axis limits
    ylims = {}

    # cases
    ncases = len(nref)
    for i in range(ncases):
        case_name = "$N_{h}$ = " + str(nref[i]) + ", $N_{q}$ = " + str(nqref[i])
        r, vals, res = Run(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref[i], nqref[i], BC_TYPE, nSamples, PLOT3D, expr)
        PlotCase(axs, plots, r, vals, case_name)
        print("finished case: " + case_name)

    # export figures
    dir = model_problem_name + "/" + study_name
    Export(axs, figs, dir, titles, xlabels, ylabels, ylims)
    WriteAnalysisProperties(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name, dir)

    # close figs
    CloseFigs(figs)

def CompressibilityStudy(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name, study_name = "compressibility"):
    # inputs
    PLOT3D = False
    nSamples = 100

    # define figures
    plot_keys = ["stress", "sigmazz", "sigmatt", "sigmarr"]
    figs, axs = InitializePlots(plot_keys)

    # Define Function Expressions to sample
    expr = {}
    expr["r"] = "r"
    expr["sigmatt"] = "sigmatt"
    expr["sigmarr"] = "sigmarr"
    expr["sigmazz_norm"] = "sigmazz / nu"

    # Define plots
    plots = {}
    plots["stress"] = ["sigmatt", "sigmarr"]
    plots["sigmazz"] = ["sigmazz_norm"]
    plots["sigmatt"] = ["sigmatt"]
    plots["sigmarr"] = ["sigmarr"]

    # Y Labels - default [MPa]
    ylabels = {}

    # X Labels - default r [mm]
    xlabels = {}

    # Titles
    titles = {}
    titles["stress"] = "Stress Components"
    titles["sigmazz"] = "Normalized Axial Stress"
    titles["sigmatt"] = "Hoop Stress"
    titles["sigmarr"] = "Radial Stress"


    # exact solution
    # r_exact, vals_exact = ExactSolution(ri, ro, pi, nu_wall[0], E_wall, nSamples, expr)
    # PlotExactCase(axs, plots, r_exact, vals_exact)

    # y axis limits
    ylims = {}
    ylims["sigmazz"] = axs["stress"].get_ylim()


    # cases
    ncases = len(nu_h)
    for i in range(ncases):
        case_name = "$\\nu = $" + str(nu_h[i])
        r, vals, res = Run(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h[i], E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, nSamples, PLOT3D, expr)
        PlotCase(axs, plots, r, vals, case_name)
        print("finished case: " + case_name)

    # export figures
    dir = model_problem_name + "/" + study_name
    Export(axs, figs, dir, titles, xlabels, ylabels, ylims)
    WriteAnalysisProperties(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name, dir)

    # close figs
    CloseFigs(figs)

def main():

    # DEFINE DEFAULT VALUES

    # model problem name
    model_problem_name = "artery"

    # outer radius
    ro = 3.2 / 2

    # inner radius
    ri = 1.75 / 2

    # inner pressure [mPa]
    pi = .012 

    # blood wall properties [mPa]
    nu_b =  0.0
    G_b  =  .000001
    E_b = CalcYoungsModulus(G_b, nu_b)

    # artery wall properties [mPa]
    nu_a =  0.3
    E_a  =  .0042

    # heart tissue properties [mPa]
    nu_h  =  0.4
    E_h   =  .06

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
    Nx = 25
    Ny = 25

    # number of plot sample points
    nSamples = 100

    # Boundary condition type: "D" for Dirichlet or "N" for Neumann
    BC_TYPE = "D"

    # nrefine for basis
    nref = 0

    # nrefine for quadrature rule
    nqref = 5

    ########################################

    
    # Run studies
    eps = ri / 10
    #Nr = [30,60,120]
    #Nt = [30,60,120]
    N = [60]
    BoundaryFittedStudy(L, N, N, Nu, Nv, eps, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, model_problem_name)

    #N = [25,50,100]
    #MeshRefinementStudy(L, N, N, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name)

    #nrefine = [0,1,2]
    #nqrefine = [4,3,2]
    #LocalRefinementStudy(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, nu_h, E_h, ri, ro, pi, basis_degree, gauss_degree, nrefine, nqrefine, BC_TYPE, model_problem_name)

    #poisson_ratios = [0.27, 0.4, 0.45, 0.48]
    #CompressibilityStudy(L, Nx, Ny, Nu, Nv, nu_b, E_b, nu_a, E_a, poisson_ratios, E_h, ri, ro, pi, basis_degree, gauss_degree, nref, nqref, BC_TYPE, model_problem_name)



if __name__ == '__main__':
    cli.run(main)


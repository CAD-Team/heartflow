from nutils import*
from nutils.pointsseq import PointsSequence
import numpy as np
from matplotlib import pyplot as plt



def CalcYoungsModulus(K, mu):
    return 9 * mu * K / (3 * K + mu)
def CalcShearModulus(E, nu):
    return E / (2 *  (1 + nu))
def CalcBulkModulus(E, nu):
    return E / (3 * (1 - 2 * nu))

class PoissonRatio(function.Pointwise):
    @staticmethod
    def evalf(x,y,z):
        nu_in = 0.0
        val0 = nu_in * (1.0 - np.heaviside(x**2 + y**2 - ri**2 ,1))
        val1 = nu_wall * (np.heaviside(x**2 + y**2 - ri**2 ,1) - np.heaviside(x**2 + y**2 - ro**2 ,0))
        val2 = nu_out * (np.heaviside(x**2 + y**2 - ro**2 ,0))
        return val0 + val1 + val2
    def _derivative(self, var, seen):
        return np.zeros(self.shape+var.shape)
class ShearModulus(function.Pointwise):
    @staticmethod
    def evalf(x,y,z):
        mu_wall = CalcShearModulus(E_wall, nu_wall)
        mu_out = CalcShearModulus(E_out, nu_out)
        val0 = mu_in * (1.0 - np.heaviside(x**2 + y**2 - ri**2 ,1))
        val1 = mu_wall * (np.heaviside(x**2 + y**2 - ri**2 ,1) - np.heaviside(x**2 + y**2 - ro**2 ,0))
        val2 = mu_out * (np.heaviside(x**2 + y**2 - ro**2 ,0))
        return val0 + val1 + val2
    def _derivative(self, var, seen):
        return np.zeros(self.shape+var.shape)
class YoungsModulus(function.Pointwise):
    @staticmethod
    def evalf(x,y,z):
        E_in = CalcYoungsModulus(K_in, mu_in)
        val0 = E_in * (1.0 - np.heaviside(x**2 + y**2 - ri**2 ,1))
        val1 = E_wall * (np.heaviside(x**2 + y**2 - ri**2 ,1) - np.heaviside(x**2 + y**2 - ro**2 ,0))
        val2 = E_out * (np.heaviside(x**2 + y**2 - ro**2 ,0))
        return val0 + val1 + val2
    def _derivative(self, var, seen):
        return np.zeros(self.shape+var.shape)
class BulkModulus(function.Pointwise):
    @staticmethod
    def evalf(x,y,z):
        K_wall = CalcBulkModulus(E_wall, nu_wall)
        K_out = CalcBulkModulus(E_out, nu_out)
        val0 = K_in * (1.0 - np.heaviside(x**2 + y**2 - ri**2 ,1))
        val1 = K_wall * (np.heaviside(x**2 + y**2 - ri**2 ,1) - np.heaviside(x**2 + y**2 - ro**2 ,0))
        val2 = K_out * (np.heaviside(x**2 + y**2 - ro**2 ,0))
        return val0 + val1 + val2
    def _derivative(self, var, seen):
        return np.zeros(self.shape+var.shape)
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


# global variables

# outer radius
ro = 3.2 / 2
# inner radius
ri = 1.75 / 2

# inner pressure [mPa]
pi = .012 

# Artery properties [mPa]
nu_wall =  0.40
E_wall  =  0.10

# Heart Tissue properties [mPa]
nu_out  =  0.40
E_out   =  0.06

# Blood properties [mPa]
K_in = 2.2e3
mu_in = .001  


def Run(L,t,Nx,Ny,Nz,Nu,Nv,basis_degree, gauss_degree, PLOT3D):
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
    omega.mu = ShearModulus(omega.x[0], omega.x[1], omega.x[2])
    omega.E = YoungsModulus(omega.x[0], omega.x[1], omega.x[2])
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
    omega.Q_ij = '< < cos , -sin , 0 >_j , < sin , cos , 0 >_j , < 0 , 0 , 1 >_j >_i'
    omega.sigma_kl = 'stress_ij Q_kj Q_li '
    omega.du_i = 'Q_ij u_j'
    
    # Stiffness Matrix
    K = omega_topo.integral('ubasis_ni,j stress_ij d:x' @ omega, degree = gauss_degree)

    # Force Vector
    F = sample_trimmed_omega.integral('traction_i Jgamma ubasis_ni' @ omega)

    # Constrain Omega
    sqr  = omega_topo.boundary['left'].integral('u_0 u_0 d:x' @ omega, degree = 2*basis_degree)
    sqr += omega_topo.boundary['bottom'].integral('u_1 u_1 d:x' @ omega, degree = 2*basis_degree)
    sqr += omega_topo.boundary['top'].integral('u_0 u_0 + u_1 u_1 d:x' @ omega, degree = 2*basis_degree)
    sqr += omega_topo.boundary['right'].integral('u_0 u_0 + u_1 u_1 d:x' @ omega, degree = 2*basis_degree)
    sqr += omega_topo.integral('u_2 u_2 d:x' @ omega, degree = 2*basis_degree)
    cons = solver.optimize('lhs', sqr, droptol=1e-15, linsolver='cg', linatol=1e-10, linprecon='diag')

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


    # 1D plot
    nSamples= 100
    ns = function.Namespace()
    topo, ns.t = mesh.rectilinear([np.linspace(ri,ro,nSamples+1)])
    ns.rgeom_i = '< t_0 / sqrt(2), t_0 / sqrt(2), 0 >_i'
    ns.r = 't_0'
    samplepts = topo.sample('gauss',1)
    pltpts = locatesample(samplepts, ns.rgeom, gauss_topo, gauss.x, 10000000000)
    vonmises = pltpts.eval(gauss.vonmises)
    r = samplepts.eval(ns.r)
    plt.plot(r, vonmises, label= "Dirichlet - " + str(L / ro))




def main():
    # background mesh size
    nelems = 50

    #Lf = [1.05, 1.2, 2, 3]
    Lf = [1.05]

    # number immersed boundary elements
    Nu = 1000
    Nv = 10

    # basis
    degree_basis = 2

    # quadrature order
    degree_gauss = 5

    for i in range(len(Lf)):
        # length
        L = Lf[i] * ro
        # number voxels
        Nx = int( np.ceil(nelems * Lf[i]) )
        Ny = int( np.ceil(nelems * Lf[i]) )
        Nz = 1
        # artery thickness
        t = L / (2 * Nx)
        Run(L, t, Nx, Ny, Nz, Nu, Nv, degree_basis, degree_gauss, True)

    plt.title("Von Mises Stress")
    plt.xlabel('d')
    plt.ylabel('$\sigma_{vonmises} / \sigma_{max}$')
    plt.legend()
    plt.savefig("bc_study.png")
    plt.show()



if __name__ == '__main__':
	cli.run(main)

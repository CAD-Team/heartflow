from nutils import *
from nutils.pointsseq import PointsSequence
import numpy as np
import meshio
import treelog
import logging


def clamp(min, max, val):
    return np.max(np.min(val, max), min)


def lerp(x0, x1, y0, y1, val):
    return y0 + (y1 - y0) / (x1 - x0) * (val - x0)


def CalcShearModulus(E, nu):
    return E / (2 * (1 + nu))


def CalcE(mu, K):
    return 9 * mu * K / (3 * K + mu)


def CalcPoisson(mu, K):
    return (3 * K - 2 * mu) / (2 * (3 * K + mu))


def CalcBulkModulus(E, nu):
    return E / (3 * (1 - 2 * nu))


def CalcLambda(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu))


def ElasticityModulus(hu, label):
    if label == 3:
        # Non Calcified Plaque
        return 1000 / 1000
    elif label == 0:
        # heart tissue
        return 60 / 1000
    elif label == 1:
        # artery wall
        return 1 / 1000
    elif label == 2:
        # artery blood
        return 1/1000
    elif label == 4:
        # calcified plaque
        return 10000 / 1000


def ShearModulus(hu, label):
    nu = PoissonRatio(hu, label)
    E = ElasticityModulus(hu, label)
    return CalcShearModulus(E, nu)


def PoissonRatio(hu, label):
    if label == 3:
        # Non Calcified Plaque
        return 0.27
    elif label == 0:
        # heart tissue
        return 0.4
    elif label == 1:
        # artery wall
        return 0.27
    elif label == 2:
        # artery blood
        return 0.0
    elif label == 4:
        # calcified plaque
        return 0.31


def BulkModulus(hu, label):
    nu = PoissonRatio(hu, label)
    E = ElasticityModulus(hu, label)
    return CalcBulkModulus(E, nu)

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


def locatesample(fromsample, fromgeom, totopo, togeom, tol, **kwargs):
    """Clone ``fromsample`` onto unrelated topology ``totopo``

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

    """

    tosample = totopo.locate(togeom, fromsample.eval(fromgeom), tol=tol, **kwargs)

    # Copy the weights from `fromsample` and account for the change in local
    # coordinates via the common geometry.
    weights = fromsample.eval(function.J(fromgeom)) / tosample.eval(function.J(togeom))
    for p, i in zip(fromsample.points, fromsample.index):
        weights[i] = p.weights
    weightedpoints = tuple(
        points.CoordsWeightsPoints(p.coords, weights[i])
        for p, i in zip(tosample.points, tosample.index)
    )
    weightedpoints = PointsSequence.from_iter(weightedpoints, 3)

    return sample.Sample.new(tosample.transforms, weightedpoints, index=tosample.index)


def isInside(A, trim_xmin, trim_xmax, trim_ymin, trim_ymax, trim_zmin, trim_zmax, dx, dy, dz):
    if A[0] <= trim_xmin + dx or trim_xmax - dx <= A[0]:
        return False
    if A[1] <= trim_ymin + dy or trim_ymax - dy <= A[1]:
        return False
    if A[2] <= trim_zmin + dz or trim_zmax - dz <= A[2]:
        return False
    return True


def Log(msg):
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    logging.warning(msg)


def main():

    Log("Import Image Data")

    # Cropped Image Data Properties
    fname = "data\\hu.txt"
    shape = [50, 44, 41]
    cropped_image_origin = [-24.843, 193.842, 625.724]

    # Model_00 Image Data Properties
    spacing = [0.4609375, 0.4609375, 0.3999939]

    # Read HU Data File
    data = np.loadtxt(fname, skiprows=3).reshape(shape, order="F")

    fname = "data\\hu_labels.txt"

    # Read HU Label Map Data File
    label_data = np.loadtxt(fname, skiprows=3).reshape(shape, order="F")

    # Calculate Image Size
    Lx = (data.shape[0] - 1) * spacing[0]
    Ly = (data.shape[1] - 1) * spacing[1]
    Lz = (data.shape[2] - 1) * spacing[2]

    # Log Info
    print("----- IMAGE DATA PROPERTIES -------")
    print(" IMAGE SHAPE: " + str(data.shape) )
    print("  IMAGE SIZE: " + str((Lx, Ly, Lz)) )
    print("  VOXEL SIZE: " + str((spacing[0], spacing[1], spacing[2])) )

    # Define Bounding Box
    # IMPORTANT NOTE: REFLECTIONS ON THE X AND Y COORDINATES ARE APPLIED HERE IN ORDER TO ALIGN THE IMAGE WITH THE STL FILE
    xmin = -cropped_image_origin[0]
    xmax = xmin + Lx
    ymin = -cropped_image_origin[1]
    ymax = ymin + Ly
    zmin = cropped_image_origin[2]
    zmax = zmin + Lz

    # Log Info
    print("----- IMAGE BOUNDING BOX -------")
    print("x: ", (xmin, xmax))
    print("y: ", (ymin, ymax))
    print("z: ", (zmin, zmax))

    Log("Initialize Background Mesh")

    # Define the Omega namespace
    omega = function.Namespace()

    # Build the background mesh
    x = np.linspace(xmin, xmax, shape[0])
    y = np.linspace(ymin, ymax, shape[1])
    z = np.linspace(zmin, zmax, shape[2])
    omega_topo, omega.x = mesh.rectilinear([x, y, z])

    # construct HU function
    omega.linbasis = omega_topo.basis("spline", degree=1)
    HU_vals = data.reshape( [data.shape[0] * data.shape[1] * data.shape[2], 1, 1], order="C" ).flatten()

    # Import HU Labels
    HU_labels = label_data.reshape( [label_data.shape[0] * label_data.shape[1] * label_data.shape[2], 1, 1], order="C" ).flatten()

    # projection type
    ptype = "convolute"

    # Construct HU function
    omega.HU = omega.linbasis.dot(HU_vals)

    # Construct Label Function
    omega.label = omega.linbasis.dot(HU_labels)

    # Find Min and Max HU Vals
    min_hu = min(HU_vals)
    max_hu = max(HU_vals)

    # Log Info
    print("----- HOUNSFIELD UNITS -------")
    print("MIN: ", min_hu)
    print("MAX: ", max_hu)

    # Construct Shear Modulus Function
    mu_vals = np.zeros(len(HU_vals))
    for i in range(len(HU_vals)):
        mu_vals[i] = ShearModulus(HU_vals[i], HU_labels[i])
    omega.mu = omega.linbasis.dot(mu_vals)

    # Construct Elasticity Modulus Function
    E_vals = np.zeros(len(HU_vals))
    for i in range(len(HU_vals)):
        E_vals[i] = ElasticityModulus(HU_vals[i], HU_labels[i])
    omega.E = omega.linbasis.dot(E_vals)

    # Construct Poisson Ratio Function
    poisson_vals = np.zeros(len(HU_vals))
    for i in range(len(HU_vals)):
        poisson_vals[i] = PoissonRatio(HU_vals[i], HU_labels[i])
    omega.poisson = omega.linbasis.dot(poisson_vals)

    # Construct Bulk Modulus Function
    K_vals = np.zeros(len(HU_vals))
    for i in range(len(HU_vals)):
        K_vals[i] = CalcBulkModulus(E_vals[i], poisson_vals[i])
    omega.K = omega.linbasis.dot(K_vals)

    # Construct Lame parameter Function
    lmbda_vals = np.zeros(len(HU_vals))
    for i in range(len(HU_vals)):
        lmbda_vals[i] = CalcLambda(E_vals[i], poisson_vals[i])
    omega.lmbda = omega.linbasis.dot(lmbda_vals)

    # Plot Omega
    # bezier = omega_topo.sample('bezier', 3)
    # x, hu, E, nu, mu = bezier.eval(['x_i', 'HU', 'E', 'poisson', 'mu'] @ omega)
    # export.vtk('model_00_cropped_omega',bezier.tri,x, hu=hu, E=E, nu=nu, mu=mu)

    Log("Initialize Lumen Mesh")

    # Read lumen Mesh Data
    fname_data = "data\\flow_sim.vtu"
    lumen_mesh = meshio.read(fname_data, file_format="vtu")
    lumen_mesh_cells = np.sort(lumen_mesh.cells[0].data)
    lumen_mesh_verts = np.array(lumen_mesh.points)
    lumen_mesh_normals = np.array(lumen_mesh.point_data["normal"])
    """
    verts given in mm
    tractions given in mm-gm-s units, but not scaled correctly by factor of 10
        - found that scaling by 1e-7 gave unrealistic stresses
    """
    lumen_mesh_tractions = 1e-6 * np.array(lumen_mesh.point_data["Traction"])

    # Define the Gamma namespace
    gamma = function.Namespace()

    # Construct Lumen Mesh
    gamma_topo, gamma.x = mesh.simplex( lumen_mesh_cells, lumen_mesh_cells, lumen_mesh_verts, {}, {}, {} )

    # Log Info
    print("----- LUMEN MESH PROPERTIES -------")
    print("NUMBER OF ELEMENTS: ", len(gamma_topo))
    print("   NUMBER OF VERTS: ", len(lumen_mesh_verts))

    # Construct Linear Basis on Lumen Mesh
    """
    .basis("spline", deg=1) didn't work. why?
    only basis_std worked...
    """
    gamma.linbasis = gamma_topo.basis_std(1)

    # Construct Traction Function on Lumen Mesh
    gamma.tx = gamma.linbasis.dot(lumen_mesh_tractions[:, 0])
    gamma.ty = gamma.linbasis.dot(lumen_mesh_tractions[:, 1])
    gamma.tz = gamma.linbasis.dot(lumen_mesh_tractions[:, 2])
    """
    numpy concat
    """
    gamma.traction_i = "< tx, ty, tz >_i"

    # Trim Lumen Mesh
    cellIsInside = [False] * len(lumen_mesh_cells)
    for i in range(len(lumen_mesh_cells)):
        A = lumen_mesh_verts[lumen_mesh_cells[i][0]]
        B = lumen_mesh_verts[lumen_mesh_cells[i][1]]
        C = lumen_mesh_verts[lumen_mesh_cells[i][2]]
        cellIsInside[i] = (
            isInside(A, xmin, xmax, ymin, ymax, zmin, zmax, spacing[0], spacing[1], spacing[2])
            and isInside(B, xmin, xmax, ymin, ymax, zmin, zmax, spacing[0], spacing[1], spacing[2])
            and isInside(C, xmin, xmax, ymin, ymax, zmin, zmax, spacing[0], spacing[1], spacing[2]) )
    gamma_ttopo = topology.SubsetTopology(
        gamma_topo,
        [
            elemreference if isInside else elemreference.empty
            for isInside, elemreference in zip(cellIsInside, gamma_topo.references)
        ]
    )

    # Log Info
    print("----- LUMEN MESH PROPERTIES -------")
    print("NUMBER OF ELEMENTS: ", len(gamma_ttopo))

    # Plot trimmed Gamma
    bezier = gamma_ttopo.sample("bezier", 3)
    points, vals = bezier.eval([gamma.x, gamma.traction])
    export.vtk( "model_00_cropped_test_trimmed_lumen_mesh", bezier.tri, points, traction=vals)

    Log("Construct Quadrature Rule")

    # Construct Trimmed Lumen Mesh Quadrature Rule on Omega
    nqpts = 1
    sample_trimmed_gamma = gamma_ttopo.sample("gauss", nqpts)
    sample_trimmed_omega = locatesample( sample_trimmed_gamma, gamma.x, omega_topo, omega.x, 10000000000)

    Log("Map Functions From Lumen Mesh to Background Mesh")

    # Construct Lumen Mesh Traction Function on Omega
    omega.tx = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.tx))
    omega.ty = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.ty))
    omega.tz = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.tz))
    omega.traction_i = "< tx, ty, tz >_i"

    # Construct Lumen Mesh Jacobian Function on Omega
    omega.Jgamma = sample_trimmed_omega.asfunction( sample_trimmed_gamma.eval(function.J(gamma.x)) )

    Log("Setup Analysis")

    # Define Analysis
    omega.quadbasis = omega_topo.basis("spline", degree=2)
    omega.ubasis = omega.quadbasis.vector(3)
    omega.u_i = "ubasis_ni ?lhs_n"
    omega.X_i = "x_i + u_i"
    omega.strain_ij = "(u_i,j + u_j,i) / 2"
    omega.stress_ij = "lmbda strain_kk δ_ij + 2 mu strain_ij"
    omega.S_ij = "stress_ij - (stress_kk) δ_ij / 3"
    omega.vonmises = "sqrt(3 S_ij S_ij / 2)"
    omega.meanstrain = "strain_kk / 3"
    omega.meanstress = "stress_kk / 3"
    omega.disp = "sqrt(u_i u_i)"
    omega.normalizedvonmises = "vonmises / E"
    omega.normalizedmeanstress = "meanstress / E"

    # Stiffness Matrix
    deg = 3
    K = omega_topo.integral("ubasis_ni,j stress_ij d:x" @ omega, degree=deg)

    # Force Vector
    F = sample_trimmed_omega.integral("traction_i Jgamma ubasis_ni" @ omega)

    Log("Compute Constraints")

    # Constrain Omega
    sqr = omega_topo.boundary.integral("u_i u_i d:x" @ omega, degree=4)
    cons = solver.optimize( "lhs", sqr, droptol=1e-15, linsolver="cg", linatol=1e-10, linprecon="diag" )

    Log("Solve")

    # Solve
    lhs = solver.solve_linear(
        "lhs",
        residual=K - F,
        constrain=cons,
        linsolver="cg",
        linatol=1e-7,
        linprecon="diag",
    )

    Log("Export Results")

    # Read Outer Wall Implicit Representation
    trim_data = np.loadtxt("data\\trim.txt", skiprows=3).reshape(shape, order="F")

    # Build Trimming Function
    trim = trim_data.reshape( [trim_data.shape[0] * trim_data.shape[1] * trim_data.shape[2], 1, 1], order="C" ).flatten()
    omega.trim = omega.linbasis.dot(trim)

    # Read Inner Wall Implicit Representation
    trim_lumen_data = np.loadtxt("data\\trim_lumen_1.txt", skiprows=3).reshape( shape, order="F" )

    # Build Trimming Function
    trim_lumen = trim_lumen_data.reshape( [trim_data.shape[0] * trim_data.shape[1] * trim_data.shape[2], 1, 1], order="C" ).flatten()
    omega.trimlumen = omega.linbasis.dot(trim_lumen)

    # sample
    bezier = omega_topo.sample("gauss", deg)
    
    # eval
    x, hu, E, nu, mu, K = bezier.eval(['x_i', 'HU', 'E', 'poisson', 'mu', 'K'] @ omega)
    meanstress, meanstrain, vonmises, disp = bezier.eval(['meanstress', 'meanstrain', 'vonmises', 'disp'] @ omega, lhs=lhs)
    label, trim, trim_lumen = bezier.eval(['label', 'trim', 'trimlumen'] @ omega, lhs=lhs)
    normalizedvonmises, normalizedmeanstress = bezier.eval( ["normalizedvonmises", "normalizedmeanstress"] @ omega, lhs=lhs )
    stress = bezier.eval(omega.stress, lhs=lhs)
    strain = bezier.eval(omega.strain, lhs=lhs)

    # calculate principle stresses
    sigma_1 = [0] * len(stress)
    sigma_2 = [0] * len(stress)
    sigma_3 = [0] * len(stress)
    eps_1 = [0] * len(strain)
    eps_2 = [0] * len(strain)
    eps_3 = [0] * len(strain)

    for i in range(len(stress)):
        val, vec = np.linalg.eig(stress[i])
        val = np.sort(val)
        sigma_1[i] = val[0]
        sigma_2[i] = val[1]
        sigma_3[i] = val[2]
        val, vec = np.linalg.eig(strain[i])
        val = np.sort(val)
        eps_1[i] = val[0]
        eps_2[i] = val[1]
        eps_3[i] = val[2]

    # Plot Stress Results
    export.vtk(
        "model_00_cropped_test_stress",
        GaussTri(omega_topo, deg),
        x,
        hu=hu,
        E=E,
        nu=nu,
        mu=mu,
        K=K,
        vonmises=vonmises,
        meanstress=meanstress,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        sigma_3=sigma_3,
        disp=disp,
        meanstrain=meanstrain,
        eps_1=eps_1,
        eps_2=eps_2,
        eps_3=eps_3,
        label=label,
        trim=trim,
        normalizedmeanstress=normalizedmeanstress,
        normalizedvonmises=normalizedvonmises,
        trim_lumen=trim_lumen,
    )


if __name__ == "__main__":
    cli.run(main)

from nutils import *
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


def ElasticityModulus(hu, label):
    if label == 3:
        # Non Calcified Plaque
        return 4.2 / 1000
    elif label == 0:
        # heart tissue
        return 60 / 1000
    elif label == 1:
        # artery wall
        return 100 / 1000
    elif label == 2:
        # artery blood
        return CalcE(ShearModulus(hu, label), 2.2 * 1000)
    elif label == 4:
        # calcified plaque
        return 2.1e7 / 1000


def ShearModulus(hu, label):
    if label == 2:
        # artery blood
        return 1 / 1000
    else:
        nu = PoissonRatio(hu, label)
        E = ElasticityModulus(hu, label)
        return CalcShearModulus(E, nu)


def PoissonRatio(hu, label):
    if label == 3:
        # Non Calcified Plaque
        return 0.45
    elif label == 0:
        # heart tissue
        return 0.45
    elif label == 1:
        # artery wall
        return 0.45
    elif label == 2:
        # artery blood
        return CalcPoisson(ShearModulus(hu, label), 2.2 * 1000)
    elif label == 4:
        # calcified plaque
        return 0.27


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
        weights[i] = 1
    weightedpoints = tuple(
        points.CoordsWeightsPoints(p.coords, weights[i])
        for p, i in zip(tosample.points, tosample.index)
    )

    return sample.Sample(tosample.transforms, weightedpoints, tosample.index)


def isInside(
    A, trim_xmin, trim_xmax, trim_ymin, trim_ymax, trim_zmin, trim_zmax, dx, dy, dz
):
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
    try:
        Log("Import Image Data")

        # Cropped Image Data Properties
        fname = "data\\hu.txt"
        shape = [50, 44, 41]
        cropped_image_origin = [-24.843, 193.842, 625.724]

        # Model_00 Image Data Properties
        image_origin = [84.76953125, 288.26953125, 505.27499390]
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

        # Calculate Voxel Size
        dx = Lx / (data.shape[0] - 1)
        dy = Ly / (data.shape[1] - 1)
        dz = Lz / (data.shape[2] - 1)

        # Log Info
        print("----- IMAGE DATA PROPERTIES -------")
        print(" IMAGE SHAPE: " + str(data.shape))
        print("  IMAGE SIZE: " + str((Lx, Ly, Lz)))
        print("  VOXEL SIZE: " + str((dx, dy, dz)))

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
        HU_vals = data.reshape(
            [data.shape[0] * data.shape[1] * data.shape[2], 1, 1], order="C"
        ).flatten()

        # Import HU Labels
        HU_labels = label_data.reshape(
            [label_data.shape[0] * label_data.shape[1] * label_data.shape[2], 1, 1],
            order="C",
        ).flatten()

        # projection type
        ptype = "lsqr"

        # Construct HU function
        omega.HU = omega.linbasis.dot(HU_vals)
        omega.HU = omega_topo.projection(
            omega.HU,
            onto=omega_topo.basis("spline", degree=2),
            geometry=omega.x,
            ptype=ptype,
            ischeme="gauss{}".format(2),
            solver="cg",
            atol=1e-5,
            precon="diag",
        )

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
        omega.mu = omega_topo.projection(
            omega.mu,
            onto=omega_topo.basis("spline", degree=2),
            geometry=omega.x,
            ptype=ptype,
            ischeme="gauss{}".format(2),
            solver="cg",
            atol=1e-10,
            precon="diag",
        )

        # Construct Elasticity Modulus Function
        E_vals = np.zeros(len(HU_vals))
        for i in range(len(HU_vals)):
            E_vals[i] = ElasticityModulus(HU_vals[i], HU_labels[i])
        omega.E = omega.linbasis.dot(E_vals)
        # omega.E = omega_topo.projection(omega.E, onto=omega_topo.basis('spline',degree=2), geometry=omega.x, ptype=ptype , ischeme='gauss{}'.format(2),solver='cg', atol=1e-10, precon='diag')

        # Construct Poisson Ratio Function
        poisson_vals = np.zeros(len(HU_vals))
        for i in range(len(HU_vals)):
            poisson_vals[i] = PoissonRatio(HU_vals[i], HU_labels[i])
        omega.poisson = omega.linbasis.dot(poisson_vals)
        # omega.poisson = omega_topo.projection(omega.poisson, onto=omega_topo.basis('spline',degree=2), geometry=omega.x, ptype=ptype, ischeme='gauss{}'.format(2),solver='cg', atol=1e-10, precon='diag')

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
        lumen_mesh_tractions = 1e-6 * np.array(lumen_mesh.point_data["Traction"])

        # Define the Gamma namespace
        gamma = function.Namespace()

        # Construct Lumen Mesh
        gamma_topo, gamma.x = mesh.simplex(
            lumen_mesh_cells, lumen_mesh_cells, lumen_mesh_verts, {}, {}, {}
        )

        # Log Info
        print("----- LUMEN MESH PROPERTIES -------")
        print("NUMBER OF ELEMENTS: ", len(gamma_topo))
        print("   NUMBER OF VERTS: ", len(lumen_mesh_verts))

        # Construct Linear Basis on Lumen Mesh
        gamma.linbasis = gamma_topo.basis_std(1)

        # Construct Traction Function on Lumen Mesh
        gamma.tx = gamma.linbasis.dot(lumen_mesh_tractions[:, 0])
        gamma.ty = gamma.linbasis.dot(lumen_mesh_tractions[:, 1])
        gamma.tz = gamma.linbasis.dot(lumen_mesh_tractions[:, 2])
        gamma.traction_i = "< tx, ty, tz >_i"

        # Trim Lumen Mesh
        cellIsInside = [False] * len(lumen_mesh_cells)
        for i in range(len(lumen_mesh_cells)):
            A = lumen_mesh_verts[lumen_mesh_cells[i][0]]
            B = lumen_mesh_verts[lumen_mesh_cells[i][1]]
            C = lumen_mesh_verts[lumen_mesh_cells[i][2]]
            cellIsInside[i] = (
                isInside(A, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz)
                and isInside(B, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz)
                and isInside(C, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz)
            )
        gamma_ttopo = topology.SubsetTopology(
            gamma_topo,
            [
                elemreference if isInside else elemreference.empty
                for isInside, elemreference in zip(cellIsInside, gamma_topo.references)
            ],
        )

        # Log Info
        print("----- LUMEN MESH PROPERTIES -------")
        print("NUMBER OF ELEMENTS: ", len(gamma_ttopo))

        # Plot trimmed Gamma
        bezier = gamma_ttopo.sample("bezier", 3)
        points, vals = bezier.eval([gamma.x, gamma.traction])
        export.vtk(
            "model_00_cropped_test_trimmed_lumen_mesh",
            bezier.tri,
            points,
            traction=vals,
        )

        Log("Construct Quadrature Rule")

        # Construct Trimmed Lumen Mesh Quadrature Rule on Omega
        nqpts = 1
        sample_trimmed_gamma = gamma_ttopo.sample("gauss", nqpts)
        sample_trimmed_omega = locatesample(
            sample_trimmed_gamma, gamma.x, omega_topo, omega.x, 10000000000
        )

        Log("Map Functions From Lumen Mesh to Background Mesh")

        # Construct Lumen Mesh Traction Function on Omega
        omega.tx = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.tx))
        omega.ty = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.ty))
        omega.tz = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.tz))
        omega.traction_i = "< tx, ty, tz >_i"

        # Construct Lumen Mesh Jacobian Function on Omega
        omega.Jgamma = sample_trimmed_omega.asfunction(
            sample_trimmed_gamma.eval(function.J(gamma.x))
        )

        Log("Setup Analysis")

        # Define Analysis
        omega.quadbasis = omega_topo.basis("spline", degree=2)
        omega.lmbda = "E poisson / ( (1 + poisson) (1 - 2 poisson) )"
        omega.lmbda = omega_topo.projection(
            omega.lmbda,
            onto=omega_topo.basis("spline", degree=2),
            geometry=omega.x,
            ptype=ptype,
            ischeme="gauss{}".format(2),
            solver="cg",
            atol=1e-10,
            precon="diag",
        )
        omega.ubasis = omega.quadbasis.vector(3)
        omega.u_i = "ubasis_ni ?lhs_n"
        omega.X_i = "x_i + u_i"
        omega.strain_ij = "(u_i,j + u_j,i) / 2"
        omega.stress_ij = "lmbda strain_kk δ_ij + 2 mu strain_ij"
        omega.S_ij = "stress_ij - (stress_kk) δ_ij / 3"
        omega.vonmises = "sqrt(3 S_ij S_ij / 2)"
        omega.disp = "sqrt(u_i u_i)"

        # Stiffness Matrix
        K = omega_topo.integral("ubasis_ni,j stress_ij d:x" @ omega, degree=4)

        # Force Vector
        F = sample_trimmed_omega.integral("traction_i Jgamma ubasis_ni" @ omega)

        Log("Compute Constraints")

        # Constrain Omega
        sqr = omega_topo.boundary.integral("u_i u_i d:x" @ omega, degree=4)
        cons = solver.optimize(
            "lhs", sqr, droptol=1e-15, linsolver="cg", linatol=1e-10, linprecon="diag"
        )

        Log("Solve")

        # Solve
        lhs = solver.solve_linear("lhs", K - F, constrain=cons)

        Log("Export Results")

        # Read Outer Wall Implicit Representation
        trim_data = np.loadtxt("data\\trim.txt", skiprows=3).reshape(shape, order="F")

        # Build Trimming Function
        trim = trim_data.reshape(
            [trim_data.shape[0] * trim_data.shape[1] * trim_data.shape[2], 1, 1],
            order="C",
        ).flatten()
        omega.trim = omega.linbasis.dot(trim)

        # sample
        bezier = omega_topo.sample("bezier", 3)
        (
            x,
            vonmises,
            meanstress,
            stress,
            strain,
            meanstrain,
            disp,
            hu,
            E,
            nu,
            mu,
            label,
            trim,
        ) = bezier.eval(
            [
                "x_i",
                "vonmises",
                "meanstress",
                "stress_ij",
                "strain_ij",
                "meanstrain",
                "u_i",
                "HU",
                "E",
                "poisson",
                "mu",
                "label",
                "trim",
            ]
            @ omega,
            lhs=lhs,
        )

        # calculate principle stresses
        max_principle_stress = [0] * len(stress)
        sigma_1 = [0] * len(stress)
        sigma_2 = [0] * len(stress)
        sigma_3 = [0] * len(stress)
        max_principle_strain = [0] * len(strain)
        eps_1 = [0] * len(strain)
        eps_2 = [0] * len(strain)
        eps_3 = [0] * len(strain)

        for i in range(len(stress)):
            val, vec = np.linalg.eig(stress[i])
            max_principle_stress[i] = np.max(val)
            sigma_1[i] = val[0]
            sigma_2[i] = val[1]
            sigma_3[i] = val[2]
            val, vec = np.linalg.eig(strain[i])
            max_principle_strain[i] = np.max(val)
            eps_1[i] = val[0]
            eps_2[i] = val[1]
            eps_3[i] = val[2]

        # Plot Stress Results
        export.vtk(
            "model_00_cropped_test_stress",
            bezier.tri,
            x,
            vonmises=vonmises,
            meanstress=meanstress,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            sigma_3=sigma_3,
            max_principle_stress=max_principle_stress,
            meanstrain=meanstrain,
            eps_1=eps_1,
            eps_2=eps_2,
            eps_3=eps_3,
            max_principle_strain=max_principle_strain,
            disp=disp,
            hu=hu,
            E=E,
            nu=nu,
            mu=mu,
            label=label,
            trim=trim,
        )
    except:
        logging.exception("HELLO")
        raise


if __name__ == "__main__":
    cli.run(main)

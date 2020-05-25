from nutils import*
from nutils.pointsseq import PointsSequence
import numpy as np
import meshio
import treelog
import logging
from utils.data import VoxelData, GenericData
from utils.materials import project_onto_basis, init_material_properties
from utils.materials import (
    ElasticityModulus,
    ShearModulus,
    CalcBulkModulus,
    CalcPoissonRatio,
)

#
# Model_00 Image Data Properties:
#   image_origin = [84.76953125, 288.26953125, 505.27499390]
def main():

	Log("Import Image Data")

    #
    # Hounsfield Unit Data from Cropped Image
    hu_data = VoxelData(
        file_name="hu.txt",
        shape=[50, 44, 41],
        origin=[-24.843, 193.842, 625.724],
        skiprows=3,
        voxel_dims=[0.4609375 , 0.4609375, 0.3999939]
    )

    #
	# Read HU Label Map Data File
	label_data = GenericData(
        file_name="hu_labels.txt",
        shape=[50, 44, 41],
        skiprows=3
    )

	print(hu_data.image_properties)


	Log("Initialize Background Mesh")

	# Define the Omega namespace
	omega = function.Namespace()

	# Build the background mesh
	omega_topo, omega.x = construct_background_mesh(voxel_data=hu_data)

    # Get flattened data
    HU_vals = hu_data.flattened
	HU_labels = label_data.flattened

	# construct HU function
	omega.linbasis = omega_topo.basis('spline',degree=1)
	
	# Import HU Labels
	# projection type
	ptype = "convolute"
    ischeme_order = 2

	# Construct HU function
	omega.HU = omega.linbasis.dot(HU_vals)
	omega.HU = omega_topo.projection(omega.HU, onto=omega_topo.basis('spline',degree=2), geometry=omega.x, ptype=ptype, ischeme=f'gauss{ischeme_order}',solver='cg', atol=1e-5, precon='diag')

	# Construct Label Function
	omega.label = omega.linbasis.dot(HU_labels)

	# Log Info
	print(
        "----- HOUNSFIELD UNITS -------\n",
        f"MIN: {hu_data.min_val}\n",
        f"MAX: {hu_data.max_val}"
    )

    #
    # Guarantees that there's a placeholder
    # for the material properties
    init_material_properties(omega)

    #
    # List properties and corresponding function call
    props = [
        (omega.mu, ShearModulus),
        (omega.E, ElasticityModulus),
        (omega.poisson, PoissonRatio),
        (omega.K, CalcBulkModulus),
        (omega.lmbda, CalcLambda)
    ]

    #
    # Project material property functions
    # onto basis
    for prop in props:
        matprop, matprop_fn = prop
        matprop = project_function_onto_basis(   
        function=matprop_fn,
        basis=omega.linbasis,
        geometry=omega.x,
        topology=omega_topo,
        flattened_data=HU_vals,
        flattened_labels=HU_labels
    )

	Log("Initialize Lumen Mesh")

	# Read lumen Mesh Data
	fname_data = "data\\flow_sim.vtu"
	lumen_mesh = meshio.read(
	    fname_data,
	    file_format="vtu"
	)
	lumen_mesh_cells = np.sort(lumen_mesh.cells[0].data)
	lumen_mesh_verts = np.array(lumen_mesh.points)
	lumen_mesh_normals = np.array(lumen_mesh.point_data['normal'])
	lumen_mesh_tractions = 1e-6 * np.array(lumen_mesh.point_data['Traction'])

	# Define the Gamma namespace
	gamma = function.Namespace()

	# Construct Lumen Mesh
	gamma_topo, gamma.x = mesh.simplex(lumen_mesh_cells,lumen_mesh_cells,lumen_mesh_verts,{},{},{})

	# Log Info
	print("----- LUMEN MESH PROPERTIES -------")
	print("NUMBER OF ELEMENTS: ", len(gamma_topo))
	print("   NUMBER OF VERTS: ", len(lumen_mesh_verts))

	# Construct Linear Basis on Lumen Mesh
	gamma.linbasis = gamma_topo.basis_std(1)

	# Construct Traction Function on Lumen Mesh
	gamma.tx = gamma.linbasis.dot(lumen_mesh_tractions[:,0])
	gamma.ty = gamma.linbasis.dot(lumen_mesh_tractions[:,1])
	gamma.tz = gamma.linbasis.dot(lumen_mesh_tractions[:,2])
	gamma.traction_i = '< tx, ty, tz >_i'

	# Trim Lumen Mesh
	cellIsInside = [False]*len(lumen_mesh_cells)
	for i in range(len(lumen_mesh_cells)):
	    A = lumen_mesh_verts[lumen_mesh_cells[i][0]]
	    B = lumen_mesh_verts[lumen_mesh_cells[i][1]]
	    C = lumen_mesh_verts[lumen_mesh_cells[i][2]]
	    cellIsInside[i] = isInside(A, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz) and isInside(B, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz) and isInside(C, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz)  
	gamma_ttopo = topology.SubsetTopology(gamma_topo, [elemreference  if  isInside else  elemreference.empty for isInside,elemreference in zip(cellIsInside,gamma_topo.references)])

	# Log Info
	print("----- LUMEN MESH PROPERTIES -------")
	print("NUMBER OF ELEMENTS: ", len(gamma_ttopo))

	# Plot trimmed Gamma
	bezier = gamma_ttopo.sample('bezier', 3)
	points, vals = bezier.eval([gamma.x, gamma.traction])
	export.vtk("model_00_cropped_test_trimmed_lumen_mesh", bezier.tri, points, traction=vals)



	Log("Construct Quadrature Rule")

	# Construct Trimmed Lumen Mesh Quadrature Rule on Omega
    sample_trimmed_gamma = gamma_ttopo.sample('gauss', nqpts)
    sample_trimmed_omega = locatesample(sample_trimmed_gamma, gamma.x, omega_topo, omega.x,10000000000)

	Log("Map Functions From Lumen Mesh to Background Mesh")

	# Construct Lumen Mesh Traction Function on Omega
	omega.tx = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.tx))
	omega.ty = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.ty))
	omega.tz = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(gamma.tz))
	omega.traction_i = '< tx, ty, tz >_i'

	# Construct Lumen Mesh Jacobian Function on Omega
	omega.Jgamma = sample_trimmed_omega.asfunction(sample_trimmed_gamma.eval(function.J(gamma.x)))

	Log("Setup Analysis")

	# Setup Analysis
	init_analysis_params(namespace=omega, topology=omega_topo)
	
    # Stiffness Matrix
	K = omega_topo.integral('ubasis_ni,j stress_ij d:x' @ omega, degree=3)

	# Force Vector
	F = sample_trimmed_omega.integral('traction_i Jgamma ubasis_ni' @ omega)




	Log("Compute Constraints")

	# Constrain Omega
	sqr = omega_topo.boundary.integral('u_i u_i d:x' @ omega, degree=4)
	cons = solver.optimize('lhs', sqr, droptol=1e-15, linsolver='cg', linatol=1e-10, linprecon='diag')




	Log("Solve")

	# Solve
	lhs = solver.solve_linear('lhs', residual=K-F, constrain=cons, linsolver='cg', linatol=1e-7, linprecon='diag')





	Log("Export Results")

	# Read Outer Wall Implicit Representation
	trim_data = np.loadtxt("data\\trim.txt",skiprows=3).reshape(shape,order='F')

	# Build Trimming Function
	trim = trim_data.reshape([trim_data.shape[0]*trim_data.shape[1]*trim_data.shape[2],1,1],order='C').flatten()
	omega.trim = omega.linbasis.dot(trim)


	# Read Inner Wall Implicit Representation
	trim_lumen_data = np.loadtxt("data\\trim_lumen.txt",skiprows=3).reshape(shape,order='F')

	# Build Trimming Function
	trim_lumen = trim_lumen_data.reshape([trim_data.shape[0]*trim_data.shape[1]*trim_data.shape[2],1,1],order='C').flatten()
	omega.trimlumen = -omega.linbasis.dot(trim_lumen)


	# sample
	bezier = omega_topo.sample('bezier', 3)
	x, hu, E, nu, mu, K, meanstress, meanstrain, vonmises, disp, label, trim, trim_lumen = bezier.eval(['x_i', 'HU', 'E', 'poisson', 'mu', 'K', 'meanstress', 'meanstrain', 'vonmises', 'u_i', 'label', 'trim', 'trimlumen'] @ omega, lhs=lhs)
	normalizedvonmises, normalizedmeanstress = bezier.eval(['normalizedvonmises', 'normalizedmeanstress'] @ omega, lhs=lhs)
	stress = bezier.eval(omega.stress, lhs=lhs)
	strain = bezier.eval(omega.strain, lhs=lhs)
	
	# calculate principle stresses
	max_principle_stress = [0]*len(stress)
	sigma_1 = [0]*len(stress)
	sigma_2 = [0]*len(stress)
	sigma_3 = [0]*len(stress)
	max_principle_strain = [0]*len(strain)
	eps_1 = [0]*len(strain)
	eps_2 = [0]*len(strain)
	eps_3 = [0]*len(strain)

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
	export.vtk('model_00_cropped_test_stress',bezier.tri,x, hu=hu, E=E, nu=nu, mu=mu, K=K, vonmises=vonmises, meanstress=meanstress, sigma_1=sigma_1, sigma_2=sigma_2, sigma_3=sigma_3, max_principle_stress=max_principle_stress, disp=disp, meanstrain=meanstrain, eps_1=eps_1, eps_2=eps_2, eps_3=eps_3, max_principle_strain=max_principle_strain, label=label, trim=trim, normalizedmeanstress=normalizedmeanstress, normalizedvonmises=normalizedvonmises, trim_lumen=trim_lumen)



if __name__ == '__main__':
	cli.run(main)
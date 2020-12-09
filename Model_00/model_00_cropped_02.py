from nutils import*
import numpy as np
import vtk
import image_based_analysis as im

class MaterialModel:
    def __init__(self, tissue_map, E_map, nu_map):
        self.tissue_map = tissue_map
        new_E_map = {}
        new_nu_map = {}
        for key in E_map:
            new_E_map[tissue_map[key]] = E_map[key]
        for key in nu_map:
            new_nu_map[tissue_map[key]] = nu_map[key]
        self.E_map = im.IM_LabelMap(new_E_map)
        self.nu_map = im.IM_LabelMap(new_nu_map)
    def E(self, key):
        return self.E_map[self.tissue_map[key]]
    def nu(self, key):
        return self.nu_map[self.tissue_map[key]]

def ScaleTissueMap(tissue_map):
    scaled_tissue_map = {}
    i = 0
    for key in tissue_map:
        scaled_tissue_map[tissue_map[key]] = i
        i+=1
    return im.IM_LabelMap(scaled_tissue_map)

class IM_AABB:
    def __init__(self, position, size):
        self.position = position
        self.size = size
    def ContainsPoint(self, pt):
        return np.all(np.logical_and(0 <= pt - self.position,  pt - self.position <= self.size))
    def ToMesh(self, shape):
        return Rectilinear(self.position, self.position + self.size, shape)
    def Volume(self):
        return np.product(self.size)
    def Center(self):
        return self.position + self.size / 2
    def Verts(self):
        a = self.position
        b = self.position + size[2]
        c = self.position + size[1]
        d = self.position + size[1] + size[2]
        e = self.position + size[0]
        f = self.position + size[2] + size[0]
        g = self.position + size[1] + size[0]
        h = self.position + size[1] + size[2] + size[0]

def RotationMatrix(axis, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R1 = np.array([c + axis[0]**2 * (1 - c) , axis[0] * axis[1] * (1 - c) - axis[2] * s, axis[0] * axis[2] * (1 - c) + axis[1] * s])
    R2 = np.array([axis[0] * axis[1] * (1 - c) + axis[2] * s , axis[1]**2 * (1 - c) + c, axis[1] * axis[2] * (1 - c) - axis[0] * s])
    R3 = np.array([axis[0] * axis[2] * (1 - c) - axis[1] * s , axis[1] * axis[2] * (1 - c) + axis[0] * s, axis[2]**2 * (1 - c) + c])
    return np.array([R1, R2, R3])

def PlaneRotationMatrix():
    xrot = np.array([[1,0,0],[0,np.cos(angles[0]), -np.sin(angles[0])],[0,np.sin(angles[0]), np.cos(angles[0])]])
    yrot = np.array([[np.cos(angles[1]), 0, -np.sin(angles[1])],[0,1,0], [np.sin(angles[1]), 0, np.cos(angles[1])]])
    zrot = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],[np.sin(angles[2]), np.cos(angles[2]), 0], [0,0,1]])
    return zrot.dot(yrot.dot(xrot))

def GetParaviewBoxTransformationMatrix(angles):
    xrot = np.array([[1,0,0],[0,np.cos(angles[0]), -np.sin(angles[0])],[0,np.sin(angles[0]), np.cos(angles[0])]])
    ny = xrot.dot(np.array([0,1,0]))
    yrot = RotationMatrix(ny, angles[1])
    zrot = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],[np.sin(angles[2]), np.cos(angles[2]), 0], [0,0,1]])
    return zrot.dot(yrot.dot(xrot))

class IM_TightBB:
    def __init__(self, position, size, M):
        self.position = position
        self.size = size
        self.M = M
    def Verts(self):
        axes = self.M.dot(np.diag(self.size)).transpose()
        a = self.position
        b = self.position + axes[2]
        c = self.position + axes[1]
        d = self.position + axes[1] + axes[2]
        e = self.position + axes[0]
        f = self.position + axes[2] + axes[0]
        g = self.position + axes[1] + axes[0]
        h = self.position + axes[1] + axes[2] + axes[0]
        return np.array([a,b,c,d,e,f,g,h])        
    def GetAABB(self):
        verts = self.Verts()
        min_bounds = np.min(verts, axis=0)
        max_bounds = np.max(verts, axis=0)
        return IM_AABB(min_bounds, max_bounds - min_bounds)
    def ContainsPoint(self, pt):
        p = self.M.transpose().dot(pt - self.position)
        return np.all(np.logical_and(0 <= p, p <= self.size))
    def ToMesh(self, shape):
        axes = self.M * np.diag(self.size)
        ns = function.Namespace()
        topo, ns.xi = im.Rectilinear(np.zeros([3]), self.size, shape)
        ns.M = function.asarray(self.M)
        ns.origin = function.asarray(self.position)
        ns.x_i = 'M_ij xi_j + origin_i'
        return topo, ns.xi, ns.x
    def Volume(self):
        return np.product(self.size)
    def Center(self):
        return np.mean(self.Verts(), axis=0)

class MaterialPropertiesLabel(function.Array):
    @types.apply_annotations
    def __init__(self, material_model, *args:function.asarrays):
        self.material_model = material_model
        retval = self.evalf(*[np.ones((), dtype=arg.dtype) for arg in args])
        shapes = set(arg.shape for arg in args)
        assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
        shape, = shapes
        self.args = args
        super().__init__(args=args, shape=shape, dtype=retval.dtype)
        
    def evalf(self, wall_sdf, autoplaque):
        vals = autoplaque.copy()
        vals[np.logical_and(autoplaque==self.material_model.tissue_map["background"] , wall_sdf < 0)] = self.material_model.tissue_map["artery"]
        return vals    
        
    def _derivative(self, var, seen):
        return np.zeros(self.shape + var.shape)

def RefineVoxelMeshByTolerance(topo, f, nrefine, tol):
    dim = topo.ndims
    refined_topo = topo
    for n in range(nrefine):
        elems_to_refine = []
        k = 0
        bez = refined_topo.sample('bezier',2)
        vals = bez.eval(f)
        vals = vals.reshape( [len(vals)//2**dim, 2**dim] )
        for i in range(len(vals)):
            if np.any(abs(vals[i,:]) < tol):
                elems_to_refine.append(k)
            k = k + 1
        if len(elems_to_refine) != 0:
            refined_topo = refined_topo.refined_by(refined_topo.transforms[np.array(elems_to_refine)])
    return refined_topo

def GetAffineTransformation(structured_topo, geom):
    nelems = len(structured_topo)
    shape = structured_topo.shape
    dim = structured_topo.ndims
    ielems = np.array([0,nelems-1],dtype=int)
    xis = np.array([np.zeros([dim]), np.ones([dim])],dtype=float)
    corner_sample = structured_topo._sample(ielems, xis)
    corners = corner_sample.eval(geom)
    geom0 = corners[0]
    scale = (corners[1] - corners[0]) / shape
    return geom0, scale

def LocateSampleOnLinearMesh(fromsample, fromgeom, totopo, togeom, eps=0, **kwargs):
    geom0, scale = im.GetAffineTransformation(totopo, togeom)
    print(geom0)
    print(scale)
    coords = fromsample.eval(fromgeom)
    weights = im.GetWeights(fromsample)
    return totopo._locate(geom0, scale, coords, eps=eps, weights=weights)


def main():
	# INPUTS

	# Set data paths
	lumen_flow_sim_path = "data/flow_sim.vtu"
	autoplaque_img_path = "data/autoplaque.mha"
	wallsdf_img_path = "data/outer_wall.mha"

	# Define Material Model
	tissue_map = {"blood" : -1 , "background" : 0, "artery" : 33, "fibrous" : 75, "calcified" : 225}
	nu_map = {"blood" : 0, "background" : 0.4, "artery" : 0.27, "fibrous" : 0.27, "calcified" : .31}
	E_map = {"blood" : 2 / 1000, "background" : 60 / 1000, "artery" : 100 / 1000, "fibrous" : 1000 / 1000, "calcified" : 10000 / 1000}

	# Define Material properties projection degree
	p_conv = 1

	# Define Region of Interest
	position = (39.35025102748692, -182.88678578819045, 635.1077093084257)
	U = np.array([1,0,0])
	V = np.array([0,1,0])
	W = np.array([0,0,1])

	# Define Mesh size
	Nu = 10
	Nv = 10
	Nw = 10

	# Define refinement scheme
	nref = 1
	delta = 1/5
	nqref = 1

	# Define basis degree
	p = 2

	# Define stiffness Quadrature Rule
	q = 3

	# Define Immersed Boundary Quadrature Rule
	qu = 1

	# Define post processing projection degree
	p_post = 1

	# Define Export parameters
	plots = ["disp_smooth", "vonmises_smooth"]
	res = 3

	# Process Inputs

	# Build Mesh
	lumen_mesh = im.ReadVTKMesh(lumen_flow_sim_path)

	# Build Images
	autoplaque_img = im.IM_Image_CreateFromVTKMetaImageFile(autoplaque_img_path)
	wallsdf_img = im.IM_Image_CreateFromVTKMetaImageFile(wallsdf_img_path)

	# Build Material Model
	material_model = MaterialModel(tissue_map, E_map, nu_map)

	# Build Immersed Lumen mesh, Gamma

	# construct lumen ROI bounding box
	position = np.array([46.80474441461811,-189.58819670192366,636.9385220611223])
	rotation = np.pi / 180 * np.array([-6.63176249753866, -140.30783608141263, 26.96745476126069])
	size = np.array([4.380050382199371, 12.324963385845773, 3.76428591221843])
	M = GetParaviewBoxTransformationMatrix(rotation)
	lumen_bb = IM_TightBB(position, size, M)

	# convert vtk mesh to trimesh
	lumen_tri_mesh = im.IM_TriMesh_CreateFromVTKMesh(lumen_mesh, data_keys={"Traction", "normal"})

	# Trim Lumen mesh by bounding volume
	lumen_tri_mesh.TrimByBoundingBox(lumen_bb)

	# convert trimesh to nutils mesh
	gamma = function.Namespace()
	gamma_topo, gamma.x = lumen_tri_mesh.ToMesh()

	# build traction and normal functions
	gamma.linbasis = gamma_topo.basis('spline',1)
	normals = lumen_tri_mesh.GetData("normal")
	gamma.Nx = gamma.linbasis.dot(normals[:,0])
	gamma.Ny = gamma.linbasis.dot(normals[:,1])
	gamma.Nz = gamma.linbasis.dot(normals[:,2])
	gamma.N_i = '<Nx, Ny, Nz>_i'
	tractions = 1e-6 * lumen_tri_mesh.GetData("Traction")
	gamma.Tx = gamma.linbasis.dot(tractions[:,0])
	gamma.Ty = gamma.linbasis.dot(tractions[:,1])
	gamma.Tz = gamma.linbasis.dot(tractions[:,2])
	gamma.T_i = '<Tx, Ty, Tz>_i'

	# export gamma
	bez = gamma_topo.sample('bezier', 2)
	xvals, Tvals, Nvals = bez.eval([gamma.x, gamma.T, gamma.N])
	export.vtk("gamma_test", bez.tri, xvals, Tvals=Tvals, Nvals=Nvals)

	# Build Analysis Mesh, Omega

	# scale lumen ROI box
	scale = [2,1.1,2]
	scaled_size = lumen_bb.size * scale
	scaled_position = lumen_bb.position - M.dot((scaled_size - lumen_bb.size) / 2)
	omega_bb = IM_TightBB(scaled_position, scaled_size, lumen_bb.M)

	# build mesh
	omega = function.Namespace()
	omega_topo, omega.xi, omega.x = omega_bb.ToMesh([Nu, Nv, Nw])

	# build rotated coordinates
	gamma.M = function.asarray(M)
	gamma.pos = function.asarray(omega_bb.position)
	gamma.xi_i = 'M_ji (x_j - pos_j)'

	# Build Autoplaque Image Mesh, Alpha

	# Calculate AABB for analysis mesh
	aabb = omega_bb.GetAABB()

	# Extract subregion of autoplaque image
	alpha = function.Namespace()
	min_voxel_coords = autoplaque_img.VoxelCoordinates(aabb.position)
	max_voxel_coords = autoplaque_img.VoxelCoordinates(aabb.position + aabb.size)
	autoplaque_subimg = autoplaque_img.SubImage(min_voxel_coords, max_voxel_coords)

	# convert to mesh
	alpha_topo, alpha.x = autoplaque_subimg.ToMesh()

	# build rotated coordinates
	alpha.M = function.asarray(M)
	alpha.xi_i = 'M_ji x_j'

	# Construct material property functions on Alpha

	# Convert Images to Functions
	alpha.A = im.VoxelImageFunction(autoplaque_subimg, alpha.x[0], alpha.x[1], alpha.x[2])
	alpha.W = im.BilinearVoxelImageFunction(wallsdf_img, alpha.x[0], alpha.x[1], alpha.x[2])

	# build material properties label function in 3d
	alpha.L0 = MaterialPropertiesLabel(material_model, alpha.W, alpha.A)

	# build scaled representation
	scaled_tissue_map = ScaleTissueMap(material_model.tissue_map)
	alpha.Ls0 = im.LabelMapFunc(scaled_tissue_map, alpha.L0)

	# Create Label Map Functions for Material Properties
	alpha.nu0 = im.LabelMapFunc(material_model.nu_map, alpha.L0)
	alpha.E0 = im.LabelMapFunc(material_model.E_map, alpha.L0)

	# Define Lame Parameters
	alpha.mu0 = 'E0 / (2 (1 + nu0))'
	alpha.lmbda0 = 'E0 nu0 / ( (1 + nu0) (1 - 2 nu0) )'

	# project material properties
	alpha.convbasis = alpha_topo.basis('spline', p_conv)
	alpha.mu = alpha_topo.projection(alpha.mu0, onto=alpha.convbasis, geometry=alpha.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv), solver="cg", atol=1e-10, precon="diag")
	alpha.lmbda = alpha_topo.projection(alpha.lmbda0, onto=alpha.convbasis, geometry=alpha.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv), solver="cg", atol=1e-10, precon="diag")
	alpha.E = alpha_topo.projection(alpha.E0, onto=alpha.convbasis, geometry=alpha.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv), solver="cg", atol=1e-10, precon="diag")
	alpha.nu = alpha_topo.projection(alpha.nu0, onto=alpha.convbasis, geometry=alpha.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv), solver="cg", atol=1e-10, precon="diag")
	alpha.Ls = alpha_topo.projection(alpha.Ls0, onto=alpha.convbasis, geometry=alpha.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv), solver="cg", atol=1e-10, precon="diag")


	# verify alpha
	bez = alpha_topo.sample('bezier', 5)
	xvals = bez.eval(alpha.x)
	Ls0vals = bez.eval(alpha.Ls0)
	Lsvals = bez.eval(alpha.Ls)
	Evals = bez.eval(alpha.E)
	nuvals = bez.eval(alpha.nu)
	E0vals = bez.eval(alpha.E0)
	nu0vals = bez.eval(alpha.nu0)
	Avals = bez.eval(alpha.A)
	Wvals = bez.eval(alpha.W)
	export.vtk("alpha_test", bez.tri, xvals, Ls0=Ls0vals, Ls=Lsvals, E0=E0vals, E=Evals, nu0=nu0vals, nu=nuvals, A=Avals, W=Wvals)

	# Refine Mesh and Quadrature

	# create distance field for lumen mesh
	omega.D = im.DistanceFromTriMesh(lumen_tri_mesh, omega.x[0], omega.x[1], omega.x[2])

	# refine background topology for basis
	refined_omega_topo = RefineVoxelMeshByTolerance(omega_topo, omega.D, nref, delta)
	omega.basis = refined_omega_topo.basis('th-spline', degree = p)

	# refine background topology for quadrature rule
	refined_quadrature_topo = RefineVoxelMeshByTolerance(refined_omega_topo, omega.D, nref, delta)


	# Construct material property functions on Omega

	# Build samples
	gauss_omega = refined_quadrature_topo.sample('gauss', q)
	gauss_gamma = gamma_topo.sample('gauss', qu)

	# Project samples
	gauss_omega_proj = im.ProjectSampleTriMesh(omega.x, gamma_topo, gamma.x, gauss_omega)

	# map samples
	gauss_alpha = LocateSampleOnLinearMesh(gauss_omega, omega.x, alpha_topo, alpha.x)
	# boundary_gauss_omega = LocateSampleOnLinearMesh(gauss_gamma, gamma.xi, omega_topo, omega.xi)
	boundary_gauss_omega = im.locatesample(gauss_gamma, gamma.xi, refined_omega_topo, omega.xi, 1e-7)

	# build blood indicator function
	omega.N = im.AsFunction(gauss_omega_proj, gamma.N, gauss_omega)
	omega.Px = im.AsFunction(gauss_omega_proj, gamma.x, gauss_omega)
	omega.dx_i = 'x_i - Px_i'
	omega.dist = 'sqrt(dx_i dx_i)'
	omega.d = 'dx_i N_i'
	omega.I = im.Heaviside(omega.d)

	# convert evaluated samples on alpha to functions on omega at gauss points
	omega.Enoblood      =  im.AsFunction(gauss_alpha, alpha.E,     gauss_omega)
	omega.lmbdanoblood  =  im.AsFunction(gauss_alpha, alpha.lmbda, gauss_omega)
	omega.munoblood     =  im.AsFunction(gauss_alpha, alpha.mu,    gauss_omega)
	omega.nunoblood     =  im.AsFunction(gauss_alpha, alpha.nu,    gauss_omega)
	omega.Lsnoblood     =  im.AsFunction(gauss_alpha, alpha.Ls,    gauss_omega)
	 
	# overwrite blood region at gauss points
	omega.Eblood = material_model.E("blood")
	omega.nublood = material_model.nu("blood")
	omega.mublood = 'Eblood / (2 (1 + nublood))'
	omega.lmbdablood = 'Eblood nublood / ( (1 + nublood) (1 - 2 nublood) )'
	omega.E = 'I Enoblood + (1 - I) Eblood'
	omega.Ls = 'I Lsnoblood'
	omega.nu = 'I nunoblood + (1 - I) nublood'
	omega.mu = 'I munoblood + (1 - I) mublood'
	omega.lmbda = 'I lmbdanoblood + (1 - I) lmbdablood'

	# define analysis
	omega.T = im.AsFunction(gauss_gamma, gamma.T, boundary_gauss_omega)
	omega.J = im.AsFunction(gauss_gamma, function.J(gamma.x), boundary_gauss_omega)
	omega.ubasis = omega.basis.vector(3)
	omega.u_i = 'ubasis_ni ?lhs_n'
	omega.X_i = 'x_i + u_i'
	omega.strain_ij = '( u_i,j + u_j,i ) / 2'
	omega.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'
	omega.meanstress = 'stress_kk / 3'
	omega.S_ij = 'stress_ij - (stress_kk) δ_ij / 3'
	omega.vonmises = 'sqrt(3 S_ij S_ij / 2)'
	omega.disp = 'sqrt(u_i u_i)'

	# Stiffness Matrix
	K = gauss_omega.integral('ubasis_ni,j stress_ij d:x' @ omega)

	# Force Vector
	F = boundary_gauss_omega.integral('T_i J ubasis_ni' @ omega)

	# Constrain Omega
	sqr  = refined_omega_topo.boundary['left'].integral('u_i u_i d:x' @ omega, degree = 2*p)
	sqr += refined_omega_topo.boundary['bottom'].integral('u_i u_i d:x' @ omega, degree = 2*p)
	sqr += refined_omega_topo.boundary['top'].integral('u_i u_i d:x' @ omega, degree = 2*p)
	sqr += refined_omega_topo.boundary['right'].integral('u_i u_i d:x' @ omega, degree = 2*p)
	cons = solver.optimize('lhs', sqr, droptol=1e-15, linsolver="cg", linatol=1e-10, linprecon="diag" )

	# Solve
	# lhs = solver.solve_linear('lhs', residual=K-F, constrain=cons, linsolver="cg", linatol=1e-7) #  linprecon="diag"
	lhs = solver.solve_linear('lhs', residual=K-F, constrain=cons)

	# plot sample
	bez = refined_omega_topo.sample('bezier', res)
	bez_alpha = LocateSampleOnLinearMesh(bez, omega.x, alpha_topo, alpha.x)

	# Project samples
	bez_proj = im.ProjectSampleTriMesh(omega.xi, gamma_topo, gamma.xi, bez)

	# define plotting functions
	ns = function.Namespace()
	ns.x = omega.x

	ns.A = im.AsFunction(bez_alpha, alpha.A, bez)
	ns.W = im.AsFunction(bez_alpha, alpha.W, bez)

	ns.E0noblood = im.AsFunction(bez_alpha, alpha.E0, bez)
	ns.nu0noblood = im.AsFunction(bez_alpha, alpha.nu0, bez)
	ns.lmbda0noblood = im.AsFunction(bez_alpha, alpha.lmbda0, bez)
	ns.mu0noblood = im.AsFunction(bez_alpha, alpha.mu0, bez)
	ns.Ls0noblood = im.AsFunction(bez_alpha, alpha.Ls0, bez)

	ns.Enoblood = im.AsFunction(bez_alpha, alpha.E, bez)
	ns.nunoblood = im.AsFunction(bez_alpha, alpha.nu, bez)
	ns.lmbdanoblood = im.AsFunction(bez_alpha, alpha.lmbda, bez)
	ns.munoblood = im.AsFunction(bez_alpha, alpha.mu, bez)
	ns.Lsnoblood = im.AsFunction(bez_alpha, alpha.Ls, bez)

	ns.N = im.AsFunction(bez_proj, gamma.N, bez)
	ns.Px = im.AsFunction(bez_proj, gamma.x, bez)
	ns.dx_i = 'x_i - Px_i'
	ns.dist = 'sqrt(dx_i dx_i)'
	ns.d = 'dx_i N_i'
	ns.I = im.Heaviside(ns.d)

	ns.Eblood = material_model.E("blood")
	ns.nublood = material_model.nu("blood")
	ns.mublood = 'Eblood / (2 (1 + nublood))'
	ns.lmbdablood = 'Eblood nublood / ( (1 + nublood) (1 - 2 nublood) )'

	ns.E0 = 'I E0noblood + (1 - I) Eblood'
	ns.nu0 = 'I nu0noblood + (1 - I) nublood'
	ns.mu0 = 'I mu0noblood + (1 - I) mublood'
	ns.lmbda0 = 'I lmbda0noblood + (1 - I) lmbdablood'
	ns.Ls0 = 'I Ls0noblood'

	ns.E = 'I Enoblood + (1 - I) Eblood'
	ns.nu = 'I nunoblood + (1 - I) nublood'
	ns.mu = 'I munoblood + (1 - I) mublood'
	ns.lmbda = 'I lmbdanoblood + (1 - I) lmbdablood'
	ns.Ls = 'I Lsnoblood'

	ns.basis = omega.basis
	ns.ubasis = ns.basis.vector(3)
	ns.u_i = 'ubasis_ni ?lhs_n'
	ns.X_i = 'x_i + u_i'
	ns.strain_ij = '( u_i,j + u_j,i ) / 2'
	ns.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'
	ns.meanstress = 'stress_kk / 3'
	ns.S_ij = 'stress_ij - (stress_kk) δ_ij / 3'
	ns.vonmises = 'sqrt(3 S_ij S_ij / 2)'
	ns.disp = 'sqrt(u_i u_i)'

	# Post processed results

	# projection gauss points sample
	proj_gauss_omega = refined_omega_topo.sample('gauss', 2 * p_conv)
	proj_gauss_alpha = LocateSampleOnLinearMesh(proj_gauss_omega, omega.x, alpha_topo, alpha.x)

	# define plotting functions
	pp = function.Namespace()
	pp.xi = omega.xi
	pp.x = omega.x

	proj_gauss_gamma = im.ProjectSampleTriMesh(pp.xi, gamma_topo, gamma.xi, proj_gauss_omega)

	pp.N = im.AsFunction(proj_gauss_gamma, gamma.N, proj_gauss_omega)
	pp.Px = im.AsFunction(proj_gauss_gamma, gamma.x, proj_gauss_omega)
	pp.dx_i = 'x_i - Px_i'
	pp.dist = 'sqrt(dx_i dx_i)'
	pp.d = 'dx_i N_i'
	pp.I = im.Heaviside(pp.d)

	pp.A = im.AsFunction(proj_gauss_alpha, alpha.A, proj_gauss_omega)
	pp.W = im.AsFunction(proj_gauss_alpha, alpha.W, proj_gauss_omega)

	pp.E0noblood = im.AsFunction(proj_gauss_alpha, alpha.E0, proj_gauss_omega)
	pp.nu0noblood = im.AsFunction(proj_gauss_alpha, alpha.nu0, proj_gauss_omega)
	pp.lmbda0noblood = im.AsFunction(proj_gauss_alpha, alpha.lmbda0, proj_gauss_omega)
	pp.mu0noblood = im.AsFunction(proj_gauss_alpha, alpha.mu0, proj_gauss_omega)
	pp.Ls0noblood = im.AsFunction(proj_gauss_alpha, alpha.Ls0, proj_gauss_omega)

	pp.Enoblood = im.AsFunction(proj_gauss_alpha, alpha.E, proj_gauss_omega)
	pp.nunoblood = im.AsFunction(proj_gauss_alpha, alpha.nu, proj_gauss_omega)
	pp.lmbdanoblood = im.AsFunction(proj_gauss_alpha, alpha.lmbda, proj_gauss_omega)
	pp.munoblood = im.AsFunction(proj_gauss_alpha, alpha.mu, proj_gauss_omega)
	pp.Lsnoblood = im.AsFunction(proj_gauss_alpha, alpha.Ls, proj_gauss_omega)

	pp.Eblood = material_model.E("blood")
	pp.nublood = material_model.nu("blood")
	pp.mublood = 'Eblood / (2 (1 + nublood))'
	pp.lmbdablood = 'Eblood nublood / ( (1 + nublood) (1 - 2 nublood) )'

	pp.E0 = 'I E0noblood + (1 - I) Eblood'
	pp.nu0 = 'I nu0noblood + (1 - I) nublood'
	pp.mu0 = 'I mu0noblood + (1 - I) mublood'
	pp.lmbda0 = 'I lmbda0noblood + (1 - I) lmbdablood'
	pp.Ls0 = 'I Ls0noblood'

	pp.E = 'I Enoblood + (1 - I) Eblood'
	pp.nu = 'I nunoblood + (1 - I) nublood'
	pp.mu = 'I munoblood + (1 - I) mublood'
	pp.lmbda = 'I lmbdanoblood + (1 - I) lmbdablood'
	pp.Ls = 'I Lsnoblood'

	pp.basis = omega.basis
	pp.ubasis = ns.basis.vector(3)
	pp.u_i = 'ubasis_ni ?lhs_n'
	pp.X_i = 'x_i + u_i'
	pp.strain_ij = '( u_i,j + u_j,i ) / 2'
	pp.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'
	pp.meanstress = 'stress_kk / 3'
	pp.S_ij = 'stress_ij - (stress_kk) δ_ij / 3'
	pp.vonmises = 'sqrt(3 S_ij S_ij / 2)'
	pp.disp = 'sqrt(u_i u_i)'

	# convolute projection post processing
	proj_basis = refined_omega_topo.basis('th-spline', degree=p_conv)
	pp.vonmisesproj = refined_omega_topo.projection('vonmises' @ pp(lhs=lhs), onto=proj_basis, geometry=pp.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv), solver="cg", atol=1e-10, precon="diag")
	pp.dispproj = refined_omega_topo.projection('disp' @ pp(lhs=lhs), onto=proj_basis, geometry=pp.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv), solver="cg", atol=1e-10, precon="diag")
	pp.meanstressproj = refined_omega_topo.projection('meanstress' @ pp(lhs=lhs), onto=proj_basis, geometry=pp.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv), solver="cg", atol=1e-10, precon="diag")

	# plot

	# evaluate geometry
	xvals = bez.eval(pp.x)

	# evaluate functions
	print("evaluate vonmises proj")
	vonmises_vals_pp = bez.eval(pp.vonmisesproj)
	print("evaluate vonmises")
	vonmises_vals = bez.eval(ns.vonmises, lhs=lhs)
	print("evaluate meanstress")
	meanstress_vals = bez.eval(ns.meanstress, lhs=lhs)
	meanstress_vals_pp = bez.eval(pp.meanstressproj, lhs=lhs)
	print("evaluate disp")
	disp_vals_pp = bez.eval(pp.dispproj)
	disp_vals = bez.eval(ns.disp, lhs=lhs)
	print("evaluate Ls")
	Ls0_vals = bez.eval(ns.Ls0)
	Ls_vals = bez.eval(ns.Ls)
	print("evaluate E")
	E0vals = bez.eval(ns.E0)
	Evals = bez.eval(ns.E)
	print("evaluate nu")
	nu0vals = bez.eval(ns.nu0)
	nuvals = bez.eval(ns.nu)
	print("evaluate I")
	Ivals = bez.eval(ns.I)
	print("evaluate d")
	dvals = bez.eval(ns.d)
	print("evaluate sdf")
	sdfvals = Ivals * dvals
	print("evaluate A")
	Avals = bez.eval(ns.A)
	print("evaluate W")
	Wvals = bez.eval(ns.W)

	# write vtk
	export.vtk("model_00_cropped_02", bez.tri, xvals, vonmises_pp = vonmises_vals_pp, vonmises = vonmises_vals, disp_pp = disp_vals_pp, disp = disp_vals, labels_pp = Ls_vals, labels = Ls0_vals, E0=E0vals, E=Evals, nu0=nu0vals, nu=nuvals, I=Ivals, d=dvals, sdf=sdfvals, A=Avals, W=Wvals)


if __name__ == "__main__":
    cli.run(main)

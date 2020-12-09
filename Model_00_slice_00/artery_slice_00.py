from nutils import*
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors, collections, cm
import vtk
import image_based_analysis as im

def SortPointsByAngle(pts_array):
    pts = pts_array.copy()
    angles = np.zeros([len(pts)])
    u = pts[0]
    u /= np.linalg.norm(u)
    u = np.array([u[0], u[1], 0])
    for i in range(len(pts)):
        v = pts[i]
        v /= np.linalg.norm(v)
        v = np.array([v[0], v[1], 0])
        s = np.sign( np.cross(u,v)[2] )
        angles[i] = s * np.arccos(np.clip(v.dot(u),-1,1))
        if s < 0:
            angles[i] += 2 * np.pi
    inds = angles.argsort()
    return inds


def BuildConnectivityArray(npts):
    connectivity = np.zeros([npts,2], dtype=int)
    for i in range(npts):
        connectivity[i] = [i, (i+1) % npts]
    return connectivity


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
        
    def evalf(self, blood_indicator, wall_sdf, autoplaque):
        vals = autoplaque.copy()
        vals[np.logical_and(autoplaque==self.material_model.tissue_map["background"] , wall_sdf < 0)] = self.material_model.tissue_map["artery"]
        vals[blood_indicator < 0] = self.material_model.tissue_map["blood"]
        return vals    
        
    def _derivative(self, var, seen):
        return np.zeros(self.shape + var.shape)


class MaterialModel:
    def __init__(self, tissue_map, E_map, nu_map):
        self.tissue_map = tissue_map
        self.E_map = {}
        self.nu_map = {}
        for key in E_map:
            self.E_map[tissue_map[key]] = E_map[key]
        for key in nu_map:
            self.nu_map[tissue_map[key]] = nu_map[key]

class Circle:
    def __init__(self, plane, radius):
        self.plane = plane
        self.radius = radius


def PlotMesh(topo, geom, f, n=5):
    sample2d = topo.sample('bezier', n)
    xvals = sample2d.eval(geom)
    fvals = sample2d.eval(f)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Mesh")
    ax.tripcolor(xvals[:,0], xvals[:,1], sample2d.tri, fvals, shading='gouraud', rasterized=True)
    sample_verts = topo.sample('bezier',2)
    verts = sample_verts.eval(geom)
    ax.add_collection(collections.LineCollection(verts[sample_verts.hull], colors='w', linewidth=0.5, alpha=1))
    return fig, ax

def Run(lumen_mesh, autoplaque_img, wallsdf_img, material_model, p_conv, slice_region, Nx, Ny, Lf, p, nref, delta, nqref, q, qu, p_proj, vals, res):

    # Slice Mesh
    lumen_slice = im.SliceVTKMesh(lumen_mesh, slice_region.plane)

    # Get Slice Data
    pts = im.GetVTKMeshPoints(lumen_slice)
    tractions = im.GetVTKMeshData(lumen_slice, "Traction")

    # Transform data to local coordinates
    pts_transformed = im.VTKPlaneTransformPoints(slice_region.plane, pts)
    tractions_transformed = im.VTKPlaneTransformVectors(slice_region.plane, tractions)

    # filter slice to isolate a single artery
    pts_filtered = np.array([pt for pt in pts_transformed if np.linalg.norm(pt) < slice_region.radius])
    tractions_filtered = np.array([t for t, pt in zip(tractions_transformed,pts_transformed) if np.linalg.norm(pt) < slice_region.radius])

    # sort data by angle
    inds = SortPointsByAngle(pts_filtered)
    sorted_pts = pts_filtered[inds]
    sorted_tractions = tractions_filtered[inds]

    # create lumen slice mesh
    gamma = function.Namespace()
    lumen_polygon = im.IM_Polygon(sorted_pts)
    gamma_topo, gamma.x = lumen_polygon.ToNutilsMesh()

    # Create Traction Function on Lumen Mesh
    gamma.linbasis = gamma_topo.basis('spline', degree=1)
    gamma.tx = gamma.linbasis.dot(sorted_tractions[:,0])
    gamma.ty = gamma.linbasis.dot(sorted_tractions[:,1])
    gamma.traction_i = '<tx, ty>_i'

    # create background mesh
    omega = function.Namespace()
    centroid = lumen_polygon.Centroid()
    diameter = lumen_polygon.EffectiveDiameter()
    xmin = centroid[0] - Lf * diameter
    xmax = centroid[0] + Lf * diameter
    ymin = centroid[1] - Lf * diameter
    ymax = centroid[1] + Lf * diameter
    x = np.linspace(xmin, xmax, Nx+1)
    y = np.linspace(ymin, ymax, Ny+1)
    omega_topo, omega.x = mesh.rectilinear([x,y])

    # Define Inverse Plane Transformation Function
    M = im.GetVTKPlaneTransformationMatrix(slice_region.plane)
    omega.M = function.asarray(M)
    omega.origin = function.asarray(slice_region.plane.GetOrigin())
    omega.Mx_j = 'M_ij x_i + origin_j'

    # Define Indicator function inside lumen slice
    omega.I = im.PolygonInclusion(lumen_polygon, -1, 1, omega.x[0], omega.x[1])

    # Define Indicator function inside lumen slice
    omega.A = im.VoxelImageFunction(autoplaque_img, omega.Mx[0], omega.Mx[1], omega.Mx[2])

    # Define Wall SDF function
    omega.W = im.BilinearVoxelImageFunction(wallsdf_img, omega.Mx[0], omega.Mx[1], omega.Mx[2])

    # Define Material Label function
    omega.L = MaterialPropertiesLabel(material_model, omega.I, omega.W, omega.A)

    # Define Labelmaps
    nu_map = im.IM_LabelMap(material_model.nu_map)
    E_map = im.IM_LabelMap(material_model.E_map)

    # Define Material Properties Functions
    omega.nu = im.LabelMapFunc(nu_map, omega.L)
    omega.E = im.LabelMapFunc(E_map, omega.L)
    omega.mu0 = 'E / (2 (1 + nu))'
    omega.lmbda0 = 'E nu / ( (1 + nu) (1 - 2 nu) )'

    # project Material properties
    omega.convbasis = omega_topo.basis('spline', p_conv)
    omega.mu = omega_topo.projection(omega.mu0, onto=omega.convbasis, geometry=omega.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv))
    omega.lmbda = omega_topo.projection(omega.lmbda0, onto=omega.convbasis, geometry=omega.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv))
    omega.Eproj = omega_topo.projection(omega.E, onto=omega.convbasis, geometry=omega.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv))

    # reset the blood region
    omega.isoutside = im.PolygonInclusion(lumen_polygon, 0, 1, omega.x[0], omega.x[1])
    omega.isinside = im.PolygonInclusion(lumen_polygon, 1, 0, omega.x[0], omega.x[1])
    omega.mu = 'isoutside mu + isinside mu0'
    omega.lmbda = 'isoutside lmbda + isinside lmbda0'
    omega.Eproj = 'isoutside Eproj + isinside E'

    # create signed distance field for lumen mesh
    omega.D = im.PolygonClosestPoint(lumen_polygon, omega.x[0], omega.x[1])
    omega.sdf = 'I D'

    # refine background topology for basis
    refined_omega_topo = im.RefineVoxelMeshBySDF(omega_topo, omega.sdf, nref, 2, delta=delta)
    omega.basis = refined_omega_topo.basis('th-spline', degree = p)

    # refine background topology for quadrature rule
    refined_quadrature_topo = im.RefineVoxelMeshBySDF(refined_omega_topo, omega.sdf, nqref, 2, delta=delta * diameter)
    gauss_sample = refined_quadrature_topo.sample('gauss', q)

    # plot mesh
    tissue_map_scaled = im.IM_LabelMap( {"blood" : 0 , "background" : 1, "artery" : 2, "fibrous" : 3, "calcified" : 4} )
    tissue_map_scaled.Remap(E_map.map.keys())
    omega.scaledL = im.LabelMapFunc(tissue_map_scaled, omega.L)
    omega.scaledLproj = omega_topo.projection(omega.scaledL, onto=omega.convbasis, geometry=omega.x, ptype='convolute', ischeme='gauss{}'.format(2 * p_conv))

    fig, ax = PlotMesh(refined_omega_topo, omega.x, omega.scaledL)
    fpath = "Results/mesh_Lscaled_" + str(Nx) + "_X_" + str(Ny) + "_nref_" + str(nref) + "_nqref_" + str(nqref) + "_Lf_" + str(Lf) + "_delta_" + str(delta) + ".png"
    print("saved " + fpath)
    fig.savefig(fpath)
    plt.close(fig)

    fig, ax = PlotMesh(refined_omega_topo, omega.x, omega.scaledLproj)
    fpath = "Results/mesh_Lscaledproj_" + str(Nx) + "_X_" + str(Ny) + "_nref_" + str(nref) + "_nqref_" + str(nqref) + "_Lf_" + str(Lf) + "_delta_" + str(delta) + ".png"
    print("saved " + fpath)
    fig.savefig(fpath)
    plt.close(fig)

    fig, ax = PlotMesh(refined_omega_topo, omega.x, omega.L)
    fpath = "Results/mesh_L_" + str(Nx) + "_X_" + str(Ny) + "_nref_" + str(nref) + "_nqref_" + str(nqref) + "_Lf_" + str(Lf) + "_delta_" + str(delta) + ".png"
    print("saved " + fpath)
    fig.savefig(fpath)
    plt.close(fig)

    # plot mesh
    fig, ax = PlotMesh(refined_omega_topo, omega.x, omega.E)
    fpath = "Results/mesh_E_" + str(Nx) + "_X_" + str(Ny) + "_nref_" + str(nref) + "_nqref_" + str(nqref) + "_Lf_" + str(Lf) + "_delta_" + str(delta) + ".png"
    print("saved " + fpath)
    fig.savefig(fpath)
    plt.close(fig)

    # plot mesh
    fig, ax = PlotMesh(refined_omega_topo, omega.x, omega.Eproj)
    fpath = "Results/mesh_Eproj_" + str(Nx) + "_X_" + str(Ny) + "_nref_" + str(nref) + "_nqref_" + str(nqref) + "_Lf_" + str(Lf) + "_delta_" + str(delta) + ".png"
    print("saved " + fpath)
    fig.savefig(fpath)
    plt.close(fig)

    # Build Immersed Boundary Quadrature Rule
    tol = 1e-7
    sample_gamma = gamma_topo.sample('gauss', qu)
    print(sample_gamma.points.npoints)
    sample_omega = im.locatesample(sample_gamma, gamma.x, refined_omega_topo, omega.x, tol, 2)

    # Rebuild traction function on Omega
    omega.traction = sample_omega.asfunction(sample_gamma.eval(gamma.traction))
    omega.Jgamma = sample_omega.asfunction(sample_gamma.eval(function.J(gamma.x)))

    # Define Analysis
    omega.ubasis = omega.basis.vector(2)
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
    omega.Qinv_ij = '< < cos , sin >_j , < -sin , cos >_j >_i'
    omega.sigma_kl = 'stress_ij Qinv_kj Qinv_li '
    omega.ubar_i = 'Qinv_ij u_j'
    omega.eps_kl =  'strain_ij Qinv_kj Qinv_li '
    omega.sigmatt = 'sigma_11'
    omega.sigmarr = 'sigma_00'
    omega.ur = 'ubar_0'

    # Stiffness Matrix
    K = gauss_sample.integral('ubasis_ni,j stress_ij d:x' @ omega)

    # Force Vector
    F = sample_omega.integral('traction_i Jgamma ubasis_ni' @ omega)

    # Constrain Omega
    sqr  = refined_omega_topo.boundary['left'].integral('u_i u_i d:x' @ omega, degree = 2*p)
    sqr += refined_omega_topo.boundary['bottom'].integral('u_i u_i d:x' @ omega, degree = 2*p)
    sqr += refined_omega_topo.boundary['top'].integral('u_i u_i d:x' @ omega, degree = 2*p)
    sqr += refined_omega_topo.boundary['right'].integral('u_i u_i d:x' @ omega, degree = 2*p)
    cons = solver.optimize( 'lhs', sqr, droptol=1e-15, linsolver="cg", linatol=1e-7, linprecon="diag" )

    # Solve
    lhs = solver.solve_linear( "lhs", residual= K-F, constrain=cons, linsolver="cg", linatol=1e-7, linprecon="diag" )

    # Post Processing
    smooth_functions = {}

    #Plot 2d
    sample2d = refined_omega_topo.sample('bezier', res)
    x_vals = sample2d.eval(omega.x)
    proj_basis = refined_omega_topo.basis('th-spline', degree=p_proj)
    for key in vals:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        if "_smooth" in key:
            stripped_key = key[0:len(key)-7]
            smooth_functions[key] = refined_omega_topo.projection(stripped_key @ omega(lhs=lhs), onto=proj_basis, geometry=omega.x, ptype='convolute', ischeme='gauss{}'.format(q))
            z = sample2d.eval(smooth_functions[key])
            ax.set_title(stripped_key)
        else:
            z = sample2d.eval(key @ omega, lhs=lhs)
            z[np.logical_or( np.isnan(z) , np.isinf(z) )] = 0
            ax.set_title(key)
        ax.tripcolor(x_vals[:,0], x_vals[:,1], sample2d.tri, z, shading='gouraud', rasterized=True)
        fdir = "Results"
        fname = key + "_" + str(Nx) + "_X_" + str(Ny) + "_nref_" + str(nref) + "_nqref_" + str(nqref) + "_Lf_" + str(Lf) + "_delta_" + str(delta)
        fext = ".png"
        fpath = fdir + "/" + fname + fext
        fig.savefig(fpath)
        print("saved " + fpath)
        plt.close(fig)


def main():
    # Read mesh
    fname = "data/flow_sim.vtu"
    lumen_mesh = im.ReadVTKMesh(fname)

    # Read in Autoplaque Image Data
    fname = "data/autoplaque.mha"
    autoplaque_img = im.IM_Image_CreateFromVTKMetaImageFile(fname)

    # Read in Outer wall sdf Image Data
    fname = "data/outer_wall.mha"
    wallsdf_img = im.IM_Image_CreateFromVTKMetaImageFile(fname)

    # Define Material Model
    tissue_map = {"blood" : -1 , "background" : 0, "artery" : 33, "fibrous" : 75, "calcified" : 225}
    nu_map = {"blood" : 0, "background" : 0.4, "artery" : 0.27, "fibrous" : 0.27, "calcified" : .31}
    E_map = {"blood" : 2 / 1000, "background" : 60 / 1000, "artery" : 100 / 1000, "fibrous" : 1000 / 1000, "calcified" : 10000 / 1000}
    material_model = MaterialModel(tissue_map, E_map, nu_map)

    # Define Material properties projection degree
    p_conv = 2

    # Build Slicing Plane
    origin = (39.35025102748692, -182.88678578819045, 635.1077093084257)
    normal = (0.16669412398742134, -0.9856447804529322, 0.026784991953651572)
    radius = 4
    plane = im.BuildVTKPlane(origin, normal)
    slice_region = Circle(plane, radius)

    # Define mesh size
    Nx = 50
    Ny = 50
    Lf = 2

    # Define basis degree
    p = 2

    # Define refinement scheme
    nref = 2
    delta = 1 / 20
    nqref = 3

    # Define stiffness Quadrature Rule
    q = 3

    # Define Immersed Boundary Quadrature Rule
    qu = 20

    # Define post processing projection degree
    p_proj = 1

    # Define Export parameters
    vals = ["E", "Eproj", "L", "scaledL", "scaledLproj", "stress_00_smooth", "stress_11_smooth", "disp_smooth", "vonmises_smooth"]
    res = 10

    Run(lumen_mesh, autoplaque_img, wallsdf_img, material_model, p_conv, slice_region, Nx, Ny, Lf, p, nref, delta, nqref, q, qu, p_proj, vals, res)



if __name__ == "__main__":
    cli.run(main)

from numpy import linspace
from nutils.mesh import rectilinear

def construct_background_mesh(*, voxel_data):
    img_prop = voxel_data.image_properties
    
	x = linspace(img_prop.xmin, img_prop.xmax, img_prop.shape[0])
	y = linspace(img_prop.ymin, img_prop.ymax, img_prop.shape[1])
	z = linspace(img_prop.zmin, img_prop.zmax, img_prop.shape[2])
	
    return rectilinear([x,y,z])

def init_analysis_params(*, namespace, topology):
    namespace.quadbasis = topology.basis('spline',degree = 2)
	namespace.ubasis = omega.quadbasis.vector(3)
	namespace.u_i = 'ubasis_ni ?lhs_n'
	namespace.X_i = 'x_i + u_i'
	namespace.strain_ij = '(u_i,j + u_j,i) / 2'
	namespace.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'
	namespace.S_ij = 'stress_ij - (stress_kk) δ_ij / 3'
	namespace.vonmises = 'sqrt(3 S_ij S_ij / 2)'
	namespace.meanstrain = 'strain_kk / 3'
	namespace.meanstress = 'stress_kk / 3'
	namespace.disp = 'sqrt(u_i u_i)'
	namespace.normalizedvonmises = 'vonmises / E'
	namespace.normalizedmeanstress = 'meanstress / E'
    return
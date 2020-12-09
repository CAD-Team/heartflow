from nutils import*
from nutils.pointsseq import PointsSequence
import numpy as np
import vtk
from vtk.util import numpy_support
from matplotlib import pyplot as plt
from matplotlib import collections



def IM_Image_CreateFromVTKMetaImage(metaimage):
	origin = GetMetaImageOrigin(metaimage)
	spacing = GetMetaImageSpacing(metaimage)
	shape = GetMetaImageShape(metaimage)
	vals = GetMetaImageValues(metaimage)
	return IM_Image(origin, spacing, shape, vals)

class IM_Image :
	def __init__(self, origin, spacing, shape, vals):
		self.origin = origin
		self.spacing = spacing
		self.shape = shape
		self.vals = vals

	def GetSize(self):
		return self.shape * self.spacing

	def GetAABB(self):
		ptA = self.origin - self.spacing / 2
		ptB = ptA + self.GetSize()
		return IM_AABB(ptA, ptB)

	def ToNutilsMesh(self, shape=None):
		if shape == None:
			shape = self.shape
		ptA = self.origin - self.spacing / 2
		ptB = ptA + self.GetSize()
		return Rectilinear(ptA, ptB, shape)

	def SubImage(self, min_voxel_coords, max_voxel_coords):
		shape = max_voxel_coords - min_voxel_coords + 1
		origin = self.origin + min_voxel_coords * self.spacing
		subset_vals = np.ndarray([np.product(shape)],dtype=float)
		for i in range(shape[0]):
			for j in range(shape[1]):
				for k in range(shape[2]):
					vid = self.VoxelID(min_voxel_coords[0] + i, min_voxel_coords[1] + j, min_voxel_coords[2] + k)
					subset_vals[k + j * shape[2] + i * shape[2] * shape[1]] = self.vals[vid]
		return IM_Image(origin, self.spacing.copy(), shape, subset_vals)

	def VoxelCoordinates(self, pt):
		x = np.floor( (pt + self.spacing / 2 - self.origin) / self.spacing )
		x = np.clip(x, 0, self.shape - 1)
		return x.astype(int)

	def VoxelID(self, i, j, k):
		return k + j * self.shape[2] + i * self.shape[2] * self.shape[1]

	def GetValue(self, pt):
		coords = self.VoxelCoordinates(pt)
		vid = VoxelID(coords[0], coords[1], coords[2])
		return self.values[vid]

	def GetSize(self):
		return self.shape * self.spacing

def IM_Image_CreateFromVTKMetaImageFile(fname):
	metaimage = ReadMetaImage(fname)
	return IM_Image_CreateFromVTKMetaImage(metaimage)

def ReadMetaImage(fname):
	rd = vtk.vtkMetaImageReader()
	rd.SetFileName(fname)
	rd.Update()
	return rd.GetOutput()

def GetMetaImageOrigin(image):
	return np.array(image.GetOrigin())

def GetMetaImageSpacing(image):
	return np.array(image.GetSpacing())

def GetMetaImageShape(image):
	return np.array(image.GetDimensions())

def GetMetaImageValues(image):
	shape = GetMetaImageShape(image)
	data = image.GetPointData().GetArray("MetaImage")
	data_array = numpy_support.vtk_to_numpy(data)
	data_array = data_array.reshape(shape, order="F").T.reshape(np.product(shape), order="F")
	return np.array(data_array, dtype=float)

def RectilinearSubtopology(topo, lower_bounds, upper_bounds):
	return topology.SubsetTopology(topo, [topo.references[k + j * topo.shape[2] + i * topo.shape[2] * topo.shape[1]] if lower_bounds[0] < i < upper_bounds[0] and lower_bounds[1] < j < upper_bounds[1] and lower_bounds[2] < k < upper_bounds[2] else topo.references[k + j * topo.shape[2] + i * topo.shape[2] * topo.shape[1]].empty for i in range(topo.shape[0]) for j in range(topo.shape[1]) for k in range(topo.shape[2])])

def ImageSubtopology(img, voxel_coords_A, voxel_coords_B):
    ptA = img.origin + voxel_coords_A * img.spacing - img.spacing / 2
    ptB = img.origin + voxel_coords_B * img.spacing + img.spacing / 2
    shape = voxelB - voxelA + 1
    return Rectilinear(ptA, ptB, shape)

def RectilinearSample(topo, voxelA, voxelB): # figure out how to build gauss /bez samples on only part of a mesh
	return


def IM_TriMesh_CreateFromVTKMesh(mesh, data_keys={}):
	verts = GetVTKMeshPoints(mesh)
	tris = GetVTKMeshCells(mesh)
	data = {}
	for name in data_keys:
		data[name] = GetVTKMeshData(mesh, name)
	return IM_TriMesh(tris, verts, data)

def IM_TriMesh_CreateFromVTKPlyData(ply, data_keys={}): # figure out how to get plydata cells and update this as well as add getcells function for plydata
	verts = GetVTKMeshPoints(mesh)
	data = {}
	for name in data_keys:
		data[name] = GetVTKMeshData(mesh, name)
	return IM_TriMesh(None, verts, data)

class IM_TriMesh:
	def __init__(self, tris, verts, data={}):
		self.tris = tris
		self.verts = verts
		self.data = data

	def GetData(self, name):
		return self.data[name]

	def ToNutilsMesh(self):
		return BuildSimplexMesh(self.verts, self.tris)

	def TrimByAABB(self, aabb):
		self.verts, self.tris, self.data = TrimMeshDataByAABB(self.verts, self.tris, self.data, aabb)

	def Copy(self):
		tris = self.tris.copy()
		verts = self.verts.copy()
		data = {}
		for key in self.data:
			data[key] = self.data[key].copy()
		return IM_TriMesh(tris,verts,data)

	def GetFunction(self, name, topo=None): # FIX!
		if topo==None:
			topo, geom = self.ToNutilsMesh()
		basis = topo.basis('spline',1)
		data = self.GetData(name)
		return function.asarray(sum([b * d for b,d in zip(basis, data)]))


def TrimMeshDataByAABB(verts, cells, data, aabb):
	new_cells = []
	new_verts = []
	new_data = {}

	j = 0
	vert_map = [-1] * len(verts)
	for i in range(len(verts)):
		if aabb.ContainsPoint(verts[i]):
			vert_map[i] = j
			j += 1
			new_verts.append(verts[i])
	for cell in cells:
		new_cell = [vert_map[i] for i in cell]
		if np.all( [i != -1 for i in new_cell] ):
			new_cells.append(new_cell)

	for key in data:
		shape = np.array(data[key].shape)
		shape[0] = j
		arr = np.ndarray(shape)
		for i in range(len(vert_map)):
			if vert_map[i] == -1:
				continue
			d = data[key][i]
			if np.isscalar(d):
				arr[vert_map[i]] = d
			else:
				arr[vert_map[i]] = d.copy()
		new_data[key] = arr


	return new_verts, new_cells, new_data


def IM_TriMesh_CreateFromVTKMeshFile(fname, data_names):
	mesh = ReadVTKMesh(fname)
	return IM_TriMesh_CreateFromVTKMesh(mesh, data_names)

def BuildSimplexMesh(pts, tris):
	tris_sorted = np.sort(tris)
	return  mesh.simplex( tris_sorted, tris_sorted, pts, {}, {}, {} )

def ReadVTKMesh(fname):
	reader = vtk.vtkXMLUnstructuredGridReader()
	reader.SetFileName(fname)
	reader.Update()
	return reader.GetOutput()

def GetVTKMeshPoints(mesh):
	pts = mesh.GetPoints().GetData()
	pts_array = numpy_support.vtk_to_numpy(pts)
	return pts_array

def GetVTKMeshCells(mesh):
	cells = mesh.GetCells().GetConnectivityArray()
	cells_array = numpy_support.vtk_to_numpy(cells)
	return cells_array.reshape([cells_array.shape[0] // 3, 3])

def GetVTKMeshData(mesh, data_name):
	data = mesh.GetPointData().GetArray(data_name)
	data_array = numpy_support.vtk_to_numpy(data)
	return data_array

def BuildVTKPlane(origin, normal):
	plane = vtk.vtkPlane()
	plane.SetOrigin(origin)
	plane.SetNormal(normal)
	return plane

def GetVTKPlaneTransformationMatrix(plane):
	normal = plane.GetNormal()
	if(normal[0] != 0):
		xaxis = np.array([-(normal[1]+normal[2])/normal[0], 1, 1])
		yaxis = np.cross(normal, xaxis)
	elif(normal[1] != 0):
		xaxis = np.array([1, -(normal[0]+normal[2])/normal[1], 1])
		yaxis = np.crossP(normal, xaxis)
	elif(normal[2] != 0):
		xaxis = np.array([1, 1, -(normal[0]+normal[1])/normal[2]])
		yaxis = np.cross(normal, xaxis)
	xaxis /= np.linalg.norm(xaxis)
	yaxis /= np.linalg.norm(yaxis)
	return [xaxis, yaxis]

def VTKPlaneTransformVectors(plane, vectors, M=None):
	if M==None:
		M = GetVTKPlaneTransformationMatrix(plane)
	transformed_vectors = np.ndarray([vectors.shape[0],vectors.shape[1]-1])
	for i in range(len(vectors)):
		transformed_vectors[i] = [M[0].dot(vectors[i]), M[1].dot(vectors[i])]
	return transformed_vectors

def VTKPlaneTranslatePoints(plane, pts):
	pts_translated = np.zeros(pts.shape)
	for i in range(len(pts)):
		pts_translated[i] = pts[i] - plane.GetOrigin()
	return pts_translated

def VTKPlaneTransformPoints(plane, pts, M=None):
	pts_translated = VTKPlaneTranslatePoints(plane, pts)
	return VTKPlaneTransformVectors(plane, pts_translated, M=M)


def SliceVTKMesh(mesh, plane):
	cutter = vtk.vtkCutter()
	cutter.SetCutFunction(plane)
	cutter.SetInputData(mesh)
	cutter.Update()
	return cutter.GetOutput(0)

def GetPolyDataConnectivity(polydata):
	cells = polydata.GetPolys()
	conn = numpy_support.vtk_to_numpy( cells.GetConnectivityArray() )
	return conn

def WriteVTK(topo, geom, functions, n_samples_per_elem, fname, **args):
	bez = topo.sample('bezier', n_samples_per_elem)
	x = bez.eval(geom)
	kwargs = {}
	for func in functions:
		kwargs[func] = bez.eval(functions[func], **args)
	export.vtk(fname, bez.tri, x, **kwargs)

class BilinearVoxelImageFunction(function.Array):
	@types.apply_annotations
	def __init__(self, image, *args:function.asarrays):
		self.image = image
		shapes = set(arg.shape for arg in args)
		assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
		shape, = shapes
		self.args = args
		super().__init__(args=args, shape=shape, dtype=np.float64)
	    
	def PointToVoxelCoordinates(self, pt):
		x = np.floor( (pt + self.image.spacing / 2 - self.image.origin) / self.image.spacing )
		return x.astype(int)

	def VoxelCoordinatesToID(self, coords, order='C'):
		if order == 'F':
			return coords[0] + coords[1] * self.image.shape[0] + coords[2] * self.image.shape[0] * self.image.shape[1]
		else:
			return coords[2] + coords[1] * self.image.shape[2] + coords[0] * self.image.shape[2] * self.image.shape[1]

	def EvaluateMetaImage(self, pt):
		coords = self.PointToVoxelCoordinates(pt)
		voxelID = self.VoxelCoordinatesToID(coords)
		return self.image.vals[voxelID]

	def ShapeFunction0(self, xi):
		return 1.0 - xi

	def ShapeFunction1(self, xi):
		return xi

	def InterpolateCube(self, xi, coefs):
		val = 0.0
		f = [self.ShapeFunction0, self.ShapeFunction1]
		for i in range(2):
		    for j in range(2):
		        for k in range(2):
		            val += coefs[i,j,k] * f[i](xi[0]) * f[j](xi[1]) * f[k](xi[2])
		return val            

	def GetVoxelPosition(self, coords):
		return self.image.origin + self.image.spacing * coords

	def MapPointToLocal(self, pt, coords):
		pos = self.GetVoxelPosition(coords)
		xi = (pt - pos) / (self.image.spacing / 2)
		return xi

	def GetInterpolationCoords(self, pt):
		coords = self.PointToVoxelCoordinates(pt)
		xi = self.MapPointToLocal(pt, coords)
		for i in range(len(xi)):
		    if xi[i] < 0:
		        coords[i] -= 1
		return coords

	def ClipCoords(self, coords):
		for i in range(3):
			coords[i] = np.clip(coords[i], 0,self.image.shape[i] - 1)
		return coords

	def InterpolateMetaImage(self, pt):
		coords = self.GetInterpolationCoords(pt)
		xi = self.MapPointToLocal(pt, coords) / 2
		coefs = np.zeros([2,2,2])
		for i in range(2):
			for j in range(2):
				for k in range(2):
					vcoords = [coords[0] + i, coords[1] + j, coords[2] + k]
					clipped_coords = self.ClipCoords(vcoords)
					vID =  self.VoxelCoordinatesToID(clipped_coords)
					coefs[i,j,k] = self.image.vals[vID]
		return self.InterpolateCube(xi, coefs)


	def evalf(self, x, y, z):
		return np.array([self.InterpolateMetaImage([xx, yy, zz]) for xx,yy,zz in zip(x,y,z)])

	def _derivative(self, var, seen):
		return np.zeros(self.shape + var.shape)


class VoxelImageFunction(function.Array):
	@types.apply_annotations
	def __init__(self, image, *args:function.asarrays):
		self.image = image
		shapes = set(arg.shape for arg in args)
		assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
		shape, = shapes
		self.args = args
		super().__init__(args=args, shape=shape, dtype=np.float64)

	def PointToVoxelCoordinates(self, pt):
		x = np.floor( (pt + self.image.spacing / 2 - self.image.origin) / self.image.spacing )
		x = np.clip(x,0,self.image.shape - 1)
		return x.astype(int)

	def VoxelCoordinatesToID(self, coords, order='C'):
		if order == 'F':
			return coords[0] + coords[1] * self.image.shape[0] + coords[2] * self.image.shape[0] * self.image.shape[1]
		else:
			return coords[2] + coords[1] * self.image.shape[2] + coords[0] * self.image.shape[2] * self.image.shape[1]

	def EvalImage(self, pt):
		voxel_coords = self.PointToVoxelCoordinates(pt)
		voxel_ID = self.VoxelCoordinatesToID(voxel_coords)
		return self.image.vals[voxel_ID]

	def evalf(self, x, y, z):
		return np.array([self.EvalImage(np.array([xx, yy, zz])) for xx,yy,zz in zip(x,y,z)])

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
	dim = totopo.ndims
	tosample = totopo.locate(togeom, fromsample.eval(fromgeom), tol=tol, **kwargs)

	# Copy the weights from `fromsample` and account for the change in local
	# coordinates via the common geometry.
	weights = fromsample.eval(function.J(fromgeom)) / tosample.eval(function.J(togeom))
	for p, i in zip(fromsample.points, fromsample.index):
		weights[i] = p.weights
	weightedpoints = tuple(points.CoordsWeightsPoints(p.coords, weights[i]) for p, i in zip(tosample.points, tosample.index))
	weightedpoints = PointsSequence.from_iter(weightedpoints, dim)
	return sample.Sample.new(tosample.transforms, weightedpoints, tosample.index)


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

def GetWeights(sample):
    weights = np.ndarray([sample.points.npoints])
    for p, i in zip(sample.points, sample.index):
        weights[i] = p.weights
    return weights

def LocateRectilinearSample(fromsample, fromgeom, geom0, scale, totopo, eps=0):
	dim = totopo.ndims
	weights = GetWeights(fromsample)
	return LocateRectilinear(totopo, geom0, scale, fromsample.eval(fromgeom), eps=eps, weights=weights)

def LocateRectilinear(topo, geom0, scale, coords, eps=0, weights=None):
    coords = np.asarray(coords, dtype=float)
    return topo._locate(geom0, scale, coords, eps=eps, weights=weights)

def Rectilinear(cornerA, cornerB, shape):
    return mesh.rectilinear( [np.linspace(xmin,xmax,nelems+1) for xmin,xmax,nelems in zip(cornerA,cornerB,shape)] )


def RefineVoxelMeshBySDF(topo, sdf, nrefine, delta=0.0):
	dim = topo.ndims
	refined_topo = topo
	for n in range(nrefine):
		elems_to_refine = []
		k = 0
		bez = refined_topo.sample('bezier',2)
		sd = bez.eval(sdf)
		sd[np.abs(sd)<delta] = 0
		sd = sd.reshape( [len(sd)//2**dim, 2**dim] )
		for i in range(len(sd)):
			if np.any(np.not_equal(np.sign(sd[i,:]),np.sign(sd[i,0]))) or np.any(np.equal(np.sign(sd[i,:]),0)):
				elems_to_refine.append(k)
			k = k + 1
		if len(elems_to_refine) != 0:
			refined_topo = refined_topo.refined_by(refined_topo.transforms[np.array(elems_to_refine)])
	return refined_topo


class IM_Polygon:
    def __init__(self, verts):
        self.verts = verts.copy()
        
    def Centroid(self):
        avg = self.verts[0].copy()
        for i in range(1, len(self.verts)):
            avg += self.verts[i]
        return avg / len(self.verts)

    def EffectiveDiameter(self):
        d = 0.0
        A = self.verts[0].copy()
        for i in range(1,len(self.verts)+1):
            B = self.verts[i % len(self.verts)]
            d += np.linalg.norm(B-A)
            A = B
        return d / np.pi
    
    def NumberOfSides(self):
        return len(self.verts)

    def ToNutilsMesh(self):
        conn = BuildLineConnectivityArray(len(self.verts))
        return BuildSimplexMesh(self.verts, conn)


def BuildLineConnectivityArray(npts):
    connectivity = np.zeros([npts,2], dtype=int)
    for i in range(npts):
        connectivity[i] = [i, (i+1) % npts]
    return connectivity

class PolygonInclusion(function.Array):
    @types.apply_annotations
    def __init__(self, polygon, val_in, val_out, *args:function.asarrays):
        self.polygon = polygon
        self.val_in = val_in
        self.val_out = val_out
        shapes = set(arg.shape for arg in args)
        assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
        shape, = shapes
        self.args = args
        super().__init__(args=args, shape=shape, dtype=np.float64)
    
    def IsInside(self, pt):
        nIntersections = 0
        p1 = self.polygon.verts[0].copy()
        for i in range(1,self.polygon.NumberOfSides()+1):
            p2 = self.polygon.verts[i % self.polygon.NumberOfSides()].copy()
            if pt[1] > min([p1[1],p2[1]]):
                if pt[1] <= max([p1[1],p2[1]]):
                    if pt[0] <= max([p1[0],p2[0]]):
                        if p1[1] != p2[1]:
                            xint = (pt[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                            if p1[0] == p2[0] or pt[0] <= xint:
                                nIntersections+=1
            p1 = p2
        if nIntersections % 2 == 0:
            return self.val_out
        else:
            return self.val_in
    
    def evalf(self, x, y):
        return np.array([self.IsInside(np.array([xx, yy])) for xx,yy in zip(x,y)])

    def _derivative(self, var, seen):
        return np.zeros(self.shape + var.shape)


# Polygon Closest Point
class PolygonClosestPoint(function.Array):
    @types.apply_annotations
    def __init__(self, polygon, *args:function.asarrays):
        self.polygon = polygon
        shapes = set(arg.shape for arg in args)
        assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
        shape, = shapes
        self.args = args
        super().__init__(args=args, shape=shape, dtype=np.float64)
        
        
    def square(self, x):
        return x * x

    def distance_squared(self, vx, vy, wx, wy):
        return self.square(vx - wx) + self.square(vy - wy)

    def DistancePointToLine(self, vx, vy, wx, wy, px, py):
        d2 =  self.distance_squared(vx, vy, wx, wy) 
        if d2 == 0: 
            return np.sqrt(self.distance_squared(px, py, vx, vy))

        t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / d2;

        if t < 0:
            return np.sqrt(self.distance_squared(px, py, vx, vy))
        elif t > 1.0:
            return np.sqrt(self.distance_squared(px, py, wx, wy))
        else:
            projx = vx + t * (wx - vx) 
            projy = vy + t * (wy - vy)
            return np.sqrt(self.distance_squared(px, py, projx, projy))

    def DistancePointToPolygon(self, pt):
        A = self.polygon.verts[0]
        dmin = np.linalg.norm(A - pt)
        for i in range(1,self.polygon.NumberOfSides()+1):
            B = self.polygon.verts[i % self.polygon.NumberOfSides()]
            d = self.DistancePointToLine(A[0],A[1],B[0],B[1],pt[0],pt[1])
            if d < dmin:
                dmin = d
            A = B
        return dmin
    
    def evalf(self, x, y):
        return np.array([self.DistancePointToPolygon(np.array([xx, yy])) for xx,yy in zip(x,y)])

    def _derivative(self, var, seen):
        return np.zeros(self.shape + var.shape)

class LabelMapFunc(function.Array):
    @types.apply_annotations
    def __init__(self, label_map, *args:function.asarrays):
        self.label_map = label_map
        retval = self.evalf(*[np.ones((), dtype=arg.dtype) for arg in args])
        shapes = set(arg.shape for arg in args)
        assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
        shape, = shapes
        self.args = args
        super().__init__(args=args, shape=shape, dtype=retval.dtype)
        
    def evalf(self, label):
        if label.shape == ():
            return label
        return np.array([self.label_map.map[key] for key in label])      
        
    def _derivative(self, var, seen):
        return np.zeros(self.shape + var.shape)


class IM_LabelMap:
    def __init__(self, labelmap):
        self.map = labelmap

    def __getitem__(self, key):
        return self.map[key]

def CreatePlot(title="",size=(10,10)):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title)
    return fig, ax

# Plot function f over 2D mesh, (topo, geom) on axis, ax with resolution n X n samples per element
def PlotMesh(ax, topo, geom, f=function.asarray(0), n=2, **kwargs):
    bez = topo.sample('bezier', n)
    xvals = bez.eval(geom)
    fvals = bez.eval(f, **kwargs)
    ax.tripcolor(xvals[:,0], xvals[:,1], bez.tri, fvals, shading='gouraud', rasterized=True)
    return ax

# plots mesh wireframe for mesh (topo, geom) on axes, ax with specified color
def PlotMeshWireframe(ax, topo, geom, color='w'):
    sample_verts = topo.sample('bezier',2)
    verts = sample_verts.eval(geom)
    lines = np.array([ verts[sample_verts.hull[i]] for i in range(len(sample_verts.hull))])
    ax.add_collection(collections.LineCollection(lines, colors=color, linewidth=0.5, alpha=1))
    return ax

# Creates a random sample on topology topo with n samples per element, over element subdomain [min xi max xi] X [min eta max eta]
def CreateRandomSample(topo, n, dim=3, min_xi=None, max_xi=None):
	# set min and max xi vals
    if min_xi == None:
        min_xi = np.zeros([dim])
    if max_xi == None:
        max_xi = np.ones([dim])

    # number of elems
    nelems = np.product(topo.shape)
    
    # init arrays
    elem_indices = np.ndarray([nelems * n], dtype=int)
    xis = np.ndarray([nelems * n, dim], dtype=float)
    
    # add sample points
    for i in range(nelems):
        for j in range(n):
            elem_indices[i*n + j] = i;
            xi = (max_xi - min_xi) * np.random.rand(dim) + min_xi
            xis[i*n + j] = xi       
        
    return topo._sample(elem_indices, xis)



class TriMeshWindingNumber(function.Array):
    @types.apply_annotations
    def __init__(self, cell_array, *args:function.asarrays):
        self.verts = cell_array.verts
        self.tris = cell_array.cells
        shapes = set(arg.shape for arg in args)
        assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
        shape, = shapes
        self.args = args
        super().__init__(args=args, shape=shape, dtype=np.float64)
    
    def WindingNumberTriangle(self, v0, v1, v2, pt):
        a = v0 - pt
        b = v1 - pt
        c = v2 - pt
        det = np.linalg.det([a,b,c])
        mag_a = np.linalg.norm(a)
        mag_b = np.linalg.norm(b)
        mag_c = np.linalg.norm(c)
        return 2 * np.arctan( det / (mag_a * mag_b * mag_c + a.dot(b) * mag_c + b.dot(c) * mag_a + c.dot(a) * mag_b) )

    def WindingNumberTriMesh(self, pt):
        winding_number = 0
        for tri in self.tris:
            v0 = self.verts[tri[0]]
            v1 = self.verts[tri[1]]
            v2 = self.verts[tri[2]]
            winding_number += self.WindingNumberTriangle(v0,v1,v2, pt)
        return winding_number / (4 * np.pi)

    def evalf(self, x, y, z):
        return np.array([self.WindingNumberTriMesh(np.array([xx,yy,zz])) for xx,yy,zz in zip(x,y,z)])
        
    def _derivative(self, var, seen):
        return np.zeros(self.shape + var.shape)


class Heaviside(function.Array):
    @types.apply_annotations
    def __init__(self, x0, y0, y1, y2, *args:function.asarrays):
        self.x0 = x0
        self.y0 = y0
        self.y1 = y1
        self.y2 = y2
        shapes = set(arg.shape for arg in args)
        assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
        shape, = shapes
        self.args = args
        super().__init__(args=args, shape=shape, dtype=np.float64)

    def evalf(self, x):
        return self.y1 + (self.y2 - self.y1) * np.heaviside(x-self.x0, (self.y0-self.y1) / (self.y2 - self.y1))
    
    def _derivative(self, var, seen):
        return np.zeros(self.shape + var.shape)


class IM_AABB:
    def __init__(self, ptA, ptB):
        self.ptA = ptA
        self.ptB = ptB
    def ContainsPoint(self, pt):
        return np.all([pt[i] >= self.ptA[i] and pt[i] <= self.ptB[i] for i in range(len(pt))], axis=0)
    def Size(self):
        return np.abs(self.ptB - self.ptA)
    def ToNutilsMesh(self, shape):
        return Rectilinear(self.ptA, self.ptB, shape)


class IsInsideAABB(function.Array):
    @types.apply_annotations
    def __init__(self, AABB, *args:function.asarrays):
        self.AABB = AABB
        shapes = set(arg.shape for arg in args)
        assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
        shape, = shapes
        self.args = args
        super().__init__(args=args, shape=shape, dtype=np.float64)
        
    def evalf(self, x, y, z):
        return np.array([self.AABB.ContainsPoint(np.array([xx,yy,zz])) for xx,yy,zz in zip(x,y,z)])
        
    def _derivative(self, var, seen):
        return np.zeros(self.shape + var.shape)


def GetVertsAndCells(topo, geom):
    vert_sample = topo.sample(*element.parse_legacy_ischeme('vertex'))
    cells = np.array(vert_sample.index)
    verts = vert_sample.eval(geom)
    return verts, cells
    
def SelectSubsetTopology(topo, select):
    return topology.SubsetTopology(topo, [ref if b else ref.empty for ref, b in zip(topo.references, select)])

def TrimMeshByAABB(topo, geom, aabb):
    verts, cells = GetVertsAndCells(topo, geom)
    select = [False] * len(cells)
    for i in range(len(cells)):
        select[i] = np.all([aabb.ContainsPoint(pt) for pt in cells[i]])
    return SelectSubsetTopology(topo, select)


class DistanceFromTriMesh(function.Array):
    @types.apply_annotations
    def __init__(self, tri_mesh, *args:function.asarrays):
        self.verts = tri_mesh.verts
        self.tris = tri_mesh.tris
        shapes = set(arg.shape for arg in args)
        assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
        shape, = shapes
        self.args = args
        super().__init__(args=args, shape=shape, dtype=np.float64)
    
    def evalf(self, x, y, z):
        return np.array([DistancePointTriMesh(self.verts, self.tris, np.array([xx,yy,zz])) for xx,yy,zz in zip(x,y,z)])
    
    def _derivative(self, var, seen):
        return np.zeros(self.shape + var.shape)



def ProjectPointOntoLine(p, a, b):
    u = b - a
    u2 = u.dot(u)
    
    if u2 == 0: 
        return a.copy()

    x = p - a
    t = x.dot(u) / u2

    if t < 0:
        return 0, a.copy()
    elif t > 1.0:
        return 1, b.copy()
    else:
        return t, a + t * u

def ProjectPointOntoTriangle(p, p1, p2, p3):
    u = p2 - p1
    v = p3 - p1
    w = p3 - p2
    n = np.cross(u,v)
    x = p - p1
    y = p - p2
    n2 = n.dot(n) 
    gamma = np.cross(u,x).dot(n) / n2
    beta = np.cross(x,v).dot(n) / n2
    alpha = np.cross(w,y).dot(n) / n2

    if 0 <= gamma and gamma <= 1 and 0 <= beta and beta <= 1 and 0 <= beta and beta <= 1:
        xi = np.array([alpha, beta, gamma])
        return xi, alpha * p1 + beta * p2 + gamma * p3
    
    if gamma < 0:
        t, proj = ProjectPointOntoLine(p, p1, p2)
        xi = np.array([t, 1-t, 0])
    if beta < 0:
        t, proj = ProjectPointOntoLine(p, p1, p3)
        xi = np.array([t, 0, 1-t])
    if alpha < 0:
        t, proj = ProjectPointOntoLine(p, p2, p3)
        xi = np.array([0, t, 1-t])

    return xi, proj

def norm2(x):
	return x.dot(x)

def DistancePointTriangle(pt, a, b, c):
    ab = b - a
    bc = c - b
    ca = c - a
    ap = pt - a
    bp = pt - b
    cp = pt - c
    n = np.cross(ab, ca)
    sa = np.sign( ap.dot(np.cross(ab, n)) )
    sb = np.sign( bp.dot(np.cross(bc, n)) )
    sc = np.sign( cp.dot(np.cross(ca, n)) )
    if sa + sb + sc < 2:
        da = norm2(ab * np.clip( ap.dot(ab) / norm2(ab), 0, 1 ) - ap)
        db = norm2(bc * np.clip( bp.dot(bc) / norm2(bc), 0, 1 ) - bp)
        dc = norm2(ca * np.clip( cp.dot(ca) / norm2(ca), 0, 1 ) - cp)
        return np.sqrt( min( min(da, db) , dc) )
    else:
        return np.sqrt( n.dot(ap) * n.dot(ap) / n.dot(n) )

def ProjectPointOntoTriMesh(verts, tris, pt):
    closest_tri = 0
    min_dist = np.linalg.norm(pt-verts[tris[0][0]])
    proj = (np.array([1,0,0]), verts[tris[0][0]])
    for i in range(len(tris)):
        tri = tris[i]
        a = verts[tri[0]]
        b = verts[tri[1]]
        c = verts[tri[2]]
        triproj = ProjectPointOntoTriangle(pt,a,b,c)
        dist = np.linalg.norm(triproj[1] - pt)
        if dist < min_dist:
            min_dist = dist
            closest_tri = i
            proj = triproj
    return closest_tri, proj

def ProjectSampleTriMesh(fromgeom, totopo, togeom, fromsample):
    verts, tris = GetVertsAndCells(totopo, togeom)
    pts = fromsample.eval(fromgeom)
    xis = np.ndarray([pts.shape[0],2], dtype=float)
    ielems = np.ndarray(pts.shape[0], dtype=int)

    for i in range(len(pts)):
        print("projecting point " + str(i+1) + " / " + str(len(pts)) )
        ielems[i], proj = ProjectPointOntoTriMesh(verts, tris, pts[i])
        xis[i] = proj[0][0:2]
    return totopo._sample(ielems, xis)

def DistancePointTriMesh(verts, tris, pt):
	return np.linalg.norm(pt - ProjectPointOntoTriMesh(verts, tris, pt)[0][1])

class ProjectOntoTriMesh(function.Array):
    @types.apply_annotations
    def __init__(self, tri_mesh, *args:function.asarrays):
        self.verts = tri_mesh.verts
        self.tris = tri_mesh.tris
        shapes = set(arg.shape for arg in args)
        assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
        shape, = shapes
        self.args = args
        super().__init__(args=args, shape=shape, dtype=np.float64)
    
    def evalf(self, x):
        return np.array([ProjectPointOntoTriMesh(self.verts, self.tris, pt)[1][1] for pt in x])
    
    def _derivative(self, var, seen):
        return np.zeros(self.shape + var.shape)


def AsFunction(fromsample, f, tosample, **kwargs):
	return tosample.asfunction(fromsample.eval(f, **kwargs))
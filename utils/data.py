from numpy import loadtxt, prod as product,
from numpy import min as numpy_min, max as numpy_max
from utils.misc import get_file_name

class GenericData:

    def __init__(
            self,
            *,
            file_name,
            shape,
            skiprows
        ):
        self.data = self._load(file_name, shape, skiprows=skiprows)
        self.shape = shape
        self.file_name = file_name
        self.size = product(self.shape)
        self.min_val = numpy_min(self.data)
        self.max_val = numpy_max(self.data)

    @property
    def flattened(self):
        """Flatten the data with C ordering"""
        #
        # NOTE: I think you can just call flatten(order='C')
        return self.data.reshape([self.size, 1, 1],order='C').flatten()

    @staticmethod
    def _load(file_name, shape, skiprows):
        data = loadtxt(
            get_file_name(file_name),
            skiprows=skiprows
        )
        return data.reshape(shape, order="F")

class VoxelData(GenericData):

    def __init__(
            self,
            *,
            file_name,
            shape,
            origin,
            skiprows,
            voxel_dims
        ):
        #
        # Load data
        super().__init__(
            file_name=file_name,
            shape=shape,
            skiprows=skiprows
        )

        #
        # Set image properties
        self.image_properties = ImageProperties(
            shape=self.shape,
            origin=origin,
            voxel_dims=voxel_dims
        )

class ImageProperties:

    def __init__(
        self,
        *,
        shape,
        origin,
        voxel_dims
    ):
        self.shape = shape
        self.origin = origin
        self.dx, self.dy, self.dz = voxel_dims
        self.Lx, self.Ly, self.Lz = [
            (self.shape[i] - 1) * voxel_dims[i] for i in range(3)
        ] # number of voxels in i-th direction * i-th voxel dimension

        #
        # IMPORTANT NOTE: REFLECTIONS ON THE X AND Y COORDINATES ARE 
        # APPLIED HERE IN ORDER TO ALIGN THE IMAGE WITH THE STL FILE
        self.xmin = -self.origin[0]
        self.xmax = self.xmin + self.Lx
        self.ymin = -self.origin[1]
        self.ymax = self.ymin + self.Ly
        self.zmin = self.origin[2]
        self.zmax = self.zmin + self.Lz

    #
    # Format the print statement
    def __str__(self):
        return f"""----- IMAGE PROPERTIES -------
 IMAGE SHAPE: {self.shape}
 IMAGE SIZE: {self.Lx} x {self.Ly} x {self.Lz}
 VOXEL SIZE: {self.dx} x {self.dy} x {self.dz}

----- IMAGE BOUNDING BOX -------
 x: ({self.xmin}, {self.xmax})
 y: ({self.ymin}, {self.ymax})
 z: ({self.zmin}, {self.zmax})
"""
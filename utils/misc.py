from numpy import max as np_max, min as np_min
from nutils.sample import Sample
from nutils.pointsseq import PointsSequence
from nutils.points import CoordsWeightsPoints
from nutils.function import J
from logging import basicConfig, warning, exception
from utils.constants import INFO, ERROR, DATA_DIR
from os.path import join


def clamp(min_val, max_val, val):
    return np_max(np_min(val, max_val), min_val)


def lerp(x0, x1, y0, y1, val):
    return y0 + (y1 - y0) / (x1 - x0) * (val - x0)


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
    weights = fromsample.eval(J(fromgeom)) / tosample.eval(J(togeom))
    for p, i in zip(fromsample.points, fromsample.index):
        weights[i] = 1
    weightedpoints = tuple(
        CoordsWeightsPoints(p.coords, weights[i])
        for p, i in zip(tosample.points, tosample.index)
    )
    weightedpoints = PointsSequence.from_iter(weightedpoints, 3)

    return Sample.new(tosample.transforms, weightedpoints, index=tosample.index)


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


def Log(msg, level=INFO):
    basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    if level == INFO:
        warning(msg)
    elif level == ERROR:
        exception(msg)
    return


def get_file_name(local_file_name, directory=DATA_DIR):
    return join(directory, local_file_name)

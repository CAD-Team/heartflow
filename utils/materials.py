
def init_material_properties(*, namespace):
    namespace.mu = []
    namespace.K = []
    namespace.E = []
    namespace.poisson = []
    namespace.lmbda = []

def project_onto_basis(
    *,
    function,
    basis,
    geometry,
    topology,
    flattened_data,
    flattened_labels
):
    # Construct Shear Modulus Function
	vals = np.zeros(len(flattened_data))
	for i in range(len(HU_vals)):
        #
        # NOTE: Function must take hounsfield units
        # and material label as arguments
	    vals[i] = function(
            flattened_data[i], flattened_labels[i]
        )
	_temp = basis.dot(vals)
	return topology.projection(
        _temp, 
        onto=topology.basis('spline',degree=2), 
        geometry=geometry, 
        ptype=ptype, 
        ischeme='gauss{}'.format(2),
        solver='cg', 
        atol=1e-10, 
        precon='diag',
    )

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


def BulkModulus(hu, label):
    if label == 2:
        # artery blood
        return 2.2 * 1000
    else:
        nu = PoissonRatio(hu, label)
        E = ElasticityModulus(hu, label)
        return CalcBulkModulus(E, nu)

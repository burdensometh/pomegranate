# test_bayesian_network.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import itertools

import numpy
import torch
import pytest

from pomegranate.bayesian_network import BayesianNetwork
from pomegranate.distributions import Exponential
from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical


from .tools import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


def assert_tuple_equal(x, y):
	assert len(x) == len(y)

	for i in range(len(x)):
		assert len(x[i]) == len(y[i])

		if len(x[i]) > 0:
			for j in range(len(x[i])):
				assert x[i][j] == y[i][j]


@pytest.fixture
def X():
	return [[1, 2, 0, 0],
		 [0, 0, 1, 0],
		 [1, 1, 1, 0],
		 [1, 2, 1, 1],
		 [1, 1, 0, 1],
		 [0, 1, 0, 1],
		 [0, 1, 0, 0],
		 [1, 0, 1, 1],
		 [1, 1, 0, 1],
		 [0, 2, 1, 0],
		 [0, 0, 0, 1]]


@pytest.fixture
def w():
	return [0, 1.3, 0, 1.2, 2.5, 6.1, 0, 0, 2.1, 0.3, 3.3]


@pytest.fixture
def X_masked(X):
	mask = torch.tensor(numpy.array([
		[False, True,  True,  True ],
		[True,  True,  False, False],
		[False, False, False, True ],
		[True,  True,  True,  True ],
		[False, True,  False, False],
		[True,  True,  True,  False],
		[False, False, False, True ],
		[True,  False, True,  True ],
		[True,  True,  True,  False],
		[True,  True,  True,  True ],
		[True,  False, True,  False]]))

	X = torch.tensor(numpy.array(X))
	return torch.masked.MaskedTensor(X, mask=mask)


@pytest.fixture
def distributions():
	d1 = Categorical([[0.8, 0.1]])
	d2 = Categorical([[0.3, 0.6, 0.1]])
	d3 = Categorical([[0.9, 0.1]])
	d4 = Categorical([[0.3, 0.7]])

	d12 = ConditionalCategorical([[[0.4, 0.6], [0.7, 0.3]]])
	d22 = ConditionalCategorical([[[0.7, 0.1, 0.2], [0.5, 0.4, 0.1]]])
	d32 = ConditionalCategorical([[[0.5, 0.5], [0.1, 0.9]]])

	d13 = ConditionalCategorical([[
		[[0.6, 0.4], [0.3, 0.7]],
		[[0.5, 0.5], [0.2, 0.8]],
		[[0.3, 0.7], [0.9, 0.1]]]]) 

	return d1, d2, d3, d4, d12, d22, d32, d13

###


def test_initialization(distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork()
	assert len(model.distributions) == 0
	assert len(model.edges) == 0
	assert len(model._marginal_mapping) == 0
	assert len(model._factor_mapping) == 0 

	assert len(model._factor_graph.factors) == 0
	assert len(model._factor_graph.marginals) == 0
	assert len(model._factor_graph._factor_idxs) == 0
	assert len(model._factor_graph._marginal_idxs) == 0 
	assert sum(map(len, model._factor_graph._factor_edges)) == 0
	assert sum(map(len, model._factor_graph._marginal_edges)) == 0


	model = BayesianNetwork([d1])
	assert len(model.distributions) == 1
	assert len(model.edges) == 0
	assert len(model._marginal_mapping) == 1
	assert len(model._factor_mapping) == 1

	assert len(model._factor_graph.factors) == 1
	assert len(model._factor_graph.marginals) == 1
	assert len(model._factor_graph._factor_idxs) == 1
	assert len(model._factor_graph._marginal_idxs) == 1
	assert sum(map(len, model._factor_graph._factor_edges)) == 1
	assert sum(map(len, model._factor_graph._marginal_edges)) == 1


	model = BayesianNetwork([d1, d2])
	assert len(model.distributions) == 2
	assert len(model.edges) == 0
	assert len(model._marginal_mapping) == 2
	assert len(model._factor_mapping) == 2

	assert len(model._factor_graph.factors) == 2
	assert len(model._factor_graph.marginals) == 2
	assert len(model._factor_graph._factor_idxs) == 2
	assert len(model._factor_graph._marginal_idxs) == 2
	assert sum(map(len, model._factor_graph._factor_edges)) == 2
	assert sum(map(len, model._factor_graph._marginal_edges)) == 2


	model = BayesianNetwork([d1, d12], [(d1, d12)])
	assert len(model.distributions) == 2
	assert len(model.edges) == 1
	assert len(model._marginal_mapping) == 2
	assert len(model._factor_mapping) == 2

	assert len(model._factor_graph.factors) == 2
	assert len(model._factor_graph.marginals) == 2
	assert len(model._factor_graph._factor_idxs) == 2
	assert len(model._factor_graph._marginal_idxs) == 2
	assert sum(map(len, model._factor_graph._factor_edges)) == 3
	assert sum(map(len, model._factor_graph._marginal_edges)) == 3


def test_initialization_raises(distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	assert_raises(ValueError, BayesianNetwork, [d1, d12], [(d12, d1)])
	assert_raises(ValueError, BayesianNetwork, None, [(d1, d12)])
	assert_raises(ValueError, BayesianNetwork, [d1], [(d1, d12)])
	assert_raises(ValueError, BayesianNetwork, [d12], [(d1, d12)])
	assert_raises(ValueError, BayesianNetwork, [d1, d12], [(d1, d1)])
	assert_raises(ValueError, BayesianNetwork, [d1, d12], [(d12, d12)])


def test_add_distribution(distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork()
	assert len(model.distributions) == 0
	assert len(model.edges) == 0
	assert model._marginal_mapping == {}
	assert model._factor_mapping == {}

	assert len(model._factor_graph.factors) == 0
	assert len(model._factor_graph.marginals) == 0
	assert model._factor_graph._factor_idxs == {}
	assert model._factor_graph._factor_edges == []
	assert model._factor_graph._marginal_idxs == {}
	assert model._factor_graph._marginal_edges == []

	model.add_distribution(d1)
	assert len(model.distributions) == 1
	assert len(model.edges) == 0
	assert len(model._marginal_mapping) == 1
	assert len(model._factor_mapping) == 1

	assert len(model._factor_graph.factors) == 1
	assert len(model._factor_graph.marginals) == 1
	assert len(model._factor_graph._factor_idxs) == 1
	assert model._factor_graph._factor_edges == [[0]]
	assert len(model._factor_graph._marginal_idxs) == 1
	assert model._factor_graph._marginal_edges == [[0]]


	model.add_distribution(d12)
	assert len(model.distributions) == 2
	assert len(model.edges) == 0
	assert len(model._marginal_mapping) == 2
	assert len(model._factor_mapping) == 2

	assert len(model._factor_graph.factors) == 2
	assert len(model._factor_graph.marginals) == 2
	assert len(model._factor_graph._factor_idxs) == 2
	assert model._factor_graph._factor_edges == [[0], [1]]
	assert len(model._factor_graph._marginal_idxs) == 2
	assert model._factor_graph._marginal_edges == [[0], [1]]


def test_add_distribution_raises(distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork()
	assert_raises(ValueError, model.add_distribution, [d1])
	assert_raises(ValueError, model.add_distribution, None)
	assert_raises(ValueError, model.add_distribution, Exponential())


def test_add_edge(distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork([d1, d2, d12, d22, d13])
	assert len(model.distributions) == 5
	assert len(model.edges) == 0
	assert model._factor_graph._factor_edges == [[0], [1], [2], [3], [4]]
	assert model._factor_graph._marginal_edges == [[0], [1], [2], [3], [4]]

	model.add_edge(d1, d12)
	assert len(model.distributions) == 5
	assert len(model.edges) == 1
	assert model._factor_graph._factor_edges == [[0], [1], [0, 2], [3], [4]]
	assert model._factor_graph._marginal_edges == [[2, 0], [1], [2], [3], [4]]

	model.add_edge(d1, d22)
	assert len(model.distributions) == 5
	assert len(model.edges) == 2
	assert model._factor_graph._factor_edges == [[0], [1], [0, 2], [0, 3], [4]]
	assert model._factor_graph._marginal_edges == [[2, 3, 0], [1], [2], [3], 
		[4]]

	model.add_edge(d2, d13)
	model.add_edge(d1, d13)
	assert len(model.distributions) == 5
	assert len(model.edges) == 4
	assert model._factor_graph._factor_edges == [[0], [1], [0, 2], [0, 3], 
		[1, 0, 4]]
	assert model._factor_graph._marginal_edges == [[2, 3, 4, 0], [4, 1], [2], 
		[3], [4]]


def test_add_edge_raises(distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork([d1, d2, d12, d22, d13])

	assert_raises(ValueError, model.add_edge, d1, d1)
	assert_raises(ValueError, model.add_edge, d2, d1)
	assert_raises(ValueError, model.add_edge, d12, d1)
	assert_raises(ValueError, model.add_edge, None, d1)
	assert_raises(ValueError, model.add_edge, d1, None)
	assert_raises(ValueError, model.add_edge, d3, d1)
	assert_raises(ValueError, model.add_edge, d1, d3)


###


@pytest.mark.sample
def test_sample(distributions):
	torch.manual_seed(0)

	d1, d2, d3, d4, d12, d22, d32, d13 = distributions
	model = BayesianNetwork([d1, d22, d13, d4], [(d1, d22), (d22, d13), 
		(d4, d13)])

	X = model.sample(1)
	assert_array_equal(X, [[1, 1, 1, 1]])

	X = model.sample(5)
	assert_array_equal(X, 
		[[0, 0, 1, 1],
		 [0, 0, 0, 1],
		 [0, 2, 1, 0],
		 [0, 1, 1, 1],
		 [0, 0, 1, 1]])


###


def test_log_probability(X, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork([d1, d22, d13, d4], [(d1, d22), (d22, d13), 
		(d4, d13)])
	logps = model.log_probability(X)

	assert_array_almost_equal(logps, [-7.013116, -2.700082, -5.115996, 
		-7.264431, -5.184989, -4.491842, -4.422849, -3.709082, -5.184989, 
		-3.393229, -2.140466])


def test_probability(X, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork([d1, d22, d13, d4], [(d1, d22), (d22, d13), 
		(d4, d13)])
	ps = model.probability(X)

	assert_array_almost_equal(ps, numpy.exp([-7.013116, -2.700082, -5.115996, 
		-7.264431, -5.184989, -4.491842, -4.422849, -3.709082, -5.184989, 
		-3.393229, -2.140466]), 4)


def test_predict_proba_one_node():
	d = Categorical([[0.23, 0.17, 0.60]])

	X = torch.tensor([[0], [0], [1], [2]])
	mask = torch.tensor([[False], [True], [True], [True]])
	X_masked = torch.masked.MaskedTensor(X, mask=mask)

	model = BayesianNetwork([d])
	y_hat = model.predict_proba(X_masked)

	assert_array_almost_equal(y_hat[0], 
		[[0.23, 0.17, 0.60],
		 [1.00, 0.00, 0.00],
		 [0.00, 1.00, 0.00],
		 [0.00, 0.00, 1.00]])


def test_predict_proba_one_node_raises():
	d = Categorical([[0.23, 0.17, 0.60]])
	model = BayesianNetwork([d])

	X = torch.tensor([[0, 0]])
	mask = torch.tensor([[False, True]])
	X_masked = torch.masked.MaskedTensor(X, mask=mask)
	assert_raises(ValueError, model.predict_proba, X_masked)

	X = torch.tensor([[3]])
	mask = torch.tensor([[True]])
	X_masked = torch.masked.MaskedTensor(X, mask=mask)
	assert_raises(IndexError, model.predict_proba, X_masked)

	mask = torch.tensor([[False]])
	X_masked = torch.masked.MaskedTensor(X, mask=mask)
	model.predict_proba(X_masked)


def test_predict_proba_one_edge(distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork([d1, d22], [(d1, d22)])

	X = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 2]])
	mask = torch.tensor([
		[False, False], 
		[True,  False], 
		[True,  True ], 
		[False, True ]
	])
	X_masked = torch.masked.MaskedTensor(X, mask=mask)

	y_hat = model.predict_proba(X_masked)

	assert_array_almost_equal(y_hat[0], 
		[[0.8889, 0.1111],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.9412, 0.0588]], 4)

	assert_array_almost_equal(y_hat[1],
		[[0.6778, 0.1333, 0.1889],
		 [0.7000, 0.1000, 0.2000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 0.0000, 1.0000]], 4)


def test_predict_proba_monty_hall():
	p = numpy.array([[
		 [[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], 
		 [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]],
		 [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
	]])

	d1 = Categorical([[1./3, 1./3, 1./3]])
	d2 = Categorical([[1./3, 1./3, 1./3]])
	d3 = ConditionalCategorical(p) 

	X = torch.tensor([
		[0, 0, 0],
		[0, 1, 0],
		[0, 2, 0],
		[1, 0, 0],
		[2, 0, 0]
	])

	mask = torch.tensor([
		[False, False, False],
		[False, True,  False],
		[False, True,  False],
		[True,  False, False],
		[True,  False, False]
	])

	X_masked = torch.masked.MaskedTensor(X, mask=mask)

	model = BayesianNetwork([d1, d2, d3], [(d1, d3), (d2, d3)])
	y_hat = model.predict_proba(X_masked)

	assert_array_almost_equal(y_hat[0], 
		[[1./3, 1./3, 1./3],
		 [1./3, 1./3, 1./3],
		 [1./3, 1./3, 1./3],
		 [0.0, 1.0, 0.0],
		 [0.0, 0.0, 1.0]])

	assert_array_almost_equal(y_hat[1],
		[[1./3, 1./3, 1./3],
		 [0.0, 1.0, 0.0],
		 [0.0, 0.0, 1.0],
		 [1./3, 1./3, 1./3],
		 [1./3, 1./3, 1./3]])

	assert_array_almost_equal(y_hat[2],
		[[1./3, 1./3, 1./3],
		 [0.5000, 0.0000, 0.5000],
		 [0.5000, 0.5000, 0.0000],
		 [0.5000, 0.0000, 0.5000],
		 [0.5000, 0.5000, 0.0000]])


	X = torch.tensor([
		[0, 0, 0],
		[0, 1, 0],
		[0, 2, 0],
		[1, 0, 0],
		[2, 0, 0],
		[0, 0, 1],
		[0, 1, 0],
		[2, 1, 0]
	])

	mask = torch.tensor([
		[True,  True,  False],
		[True,  True,  False],
		[True,  True,  False],
		[True,  True, False],
		[True,  True, False],
		[False, True,  True],
		[False, False, True],
		[True , False, True]
	])

	X_masked = torch.masked.MaskedTensor(X, mask=mask)
	y_hat = model.predict_proba(X_masked)

	assert_array_almost_equal(y_hat[0], 
		[[1.0000, 0.0000, 0.0000],
		 [1.0000, 0.0000, 0.0000],
		 [1.0000, 0.0000, 0.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 0.0000, 1.0000],
		 [1./3  , 0.0000, 2./3  ],
		 [0.0000, 0.5000, 0.5000],
		 [0.0000, 0.0000, 1.0000]])

	assert_array_almost_equal(y_hat[1],
		[[1.0000, 0.0000, 0.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 0.0000, 1.0000],
		 [1.0000, 0.0000, 0.0000],
		 [1.0000, 0.0000, 0.0000],
		 [1.0000, 0.0000, 0.0000],
		 [0.0000, 0.5000, 0.5000],
		 [0.0000, 2./3  , 1./3  ]])

	assert_array_almost_equal(y_hat[2],
		[[0.0000, 0.5000, 0.5000],
		 [0.0000, 0.0000, 1.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 0.0000, 1.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 1.0000, 0.0000],
		 [1.0000, 0.0000, 0.0000],
		 [1.0000, 0.0000, 0.0000]])


def test_predict_proba_simple(X_masked, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork([d1, d22, d13, d4], [(d1, d22), (d22, d13), 
		(d4, d13)])
	y_hat = model.predict_proba(X_masked)

	assert_array_almost_equal(y_hat[0], 
		[[0.9412, 0.0588],
		 [1.0000, 0.0000],
		 [0.8889, 0.1111],
		 [0.0000, 1.0000],
		 [0.6667, 0.3333],
		 [1.0000, 0.0000],
		 [0.8889, 0.1111],
		 [0.0000, 1.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [1.0000, 0.0000]], 4)

	assert_array_almost_equal(y_hat[1],
		[[0.0000, 0.0000, 1.0000],
		 [1.0000, 0.0000, 0.0000],
		 [0.6778, 0.1333, 0.1889],
		 [0.0000, 0.0000, 1.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.6778, 0.1333, 0.1889],
		 [0.5147, 0.4706, 0.0147],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 0.0000, 1.0000],
		 [0.6121, 0.0650, 0.3229]], 4)

	assert_array_almost_equal(y_hat[2],
		[[1.0000, 0.0000],
		 [0.3900, 0.6100],
		 [0.5300, 0.4700],
		 [0.0000, 1.0000],
		 [0.2900, 0.7100],
		 [1.0000, 0.0000],
		 [0.5300, 0.4700],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000]], 4)

	assert_array_almost_equal(y_hat[3],
		[[1.0000, 0.0000],
		 [0.3000, 0.7000],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.3000, 0.7000],
		 [0.5172, 0.4828],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.5172, 0.4828],
		 [1.0000, 0.0000],
		 [0.3565, 0.6435]], 4)


def test_predict_proba_diamond(X_masked, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), 
		(d22, d13), (d32, d13)])
	y_hat = model.predict_proba(X_masked)

	assert_array_almost_equal(y_hat[0], 
		[[0.3000, 0.7000],
		 [1.0000, 0.0000],
		 [0.4700, 0.5300],
		 [0.0000, 1.0000],
		 [0.2416, 0.7584],
		 [1.0000, 0.0000],
		 [0.4700, 0.5300],
		 [0.0000, 1.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [1.0000, 0.0000]], 4)

	assert_array_almost_equal(y_hat[1],
		[[0.0000, 0.0000, 1.0000],
		 [1.0000, 0.0000, 0.0000],
		 [0.7000, 0.1000, 0.2000],
		 [0.0000, 0.0000, 1.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.7000, 0.1000, 0.2000],
		 [0.5147, 0.4706, 0.0147],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 0.0000, 1.0000],
		 [0.7204, 0.1844, 0.0952]], 4)

	assert_array_almost_equal(y_hat[2],
		[[1.0000, 0.0000],
		 [0.4000, 0.6000],
		 [0.5000, 0.5000],
		 [0.0000, 1.0000],
		 [0.1387, 0.8613],
		 [1.0000, 0.0000],
		 [0.5000, 0.5000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000]], 4)

	assert_array_almost_equal(y_hat[3],
		[[1.0000, 0.0000],
		 [0.4500, 0.5500],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.0968, 0.9032],
		 [0.3488, 0.6512],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.3488, 0.6512],
		 [1.0000, 0.0000],
		 [0.6818, 0.3182]], 4)


def test_predict_proba_cycle(X, X_masked, distributions):
	model = BayesianNetwork(structure=((1,), (2,), (3,), (0,))).fit(X)
	y_hat = model.predict_proba(X_masked)

	assert_array_almost_equal(y_hat[0], 
		[[0.4737, 0.5263],
		 [1.0000, 0.0000],
		 [0.6065, 0.3935],
		 [0.0000, 1.0000],
		 [0.3825, 0.6175],
		 [1.0000, 0.0000],
		 [0.6065, 0.3935],
		 [0.0000, 1.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [1.0000, 0.0000]], 4)

	assert_array_almost_equal(y_hat[1],
		[[0.0000, 0.0000, 1.0000],
		 [1.0000, 0.0000, 0.0000],
		 [0.3435, 0.3728, 0.2837],
		 [0.0000, 0.0000, 1.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.3435, 0.3728, 0.2837],
		 [0.2564, 0.2308, 0.5128],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 0.0000, 1.0000],
		 [0.2564, 0.6154, 0.1282]], 4)

	assert_array_almost_equal(y_hat[2],
		[[1.0000, 0.0000],
		 [0.2997, 0.7003],
		 [0.3935, 0.6065],
		 [0.0000, 1.0000],
		 [0.8025, 0.1975],
		 [1.0000, 0.0000],
		 [0.3935, 0.6065],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000]], 4)

	assert_array_almost_equal(y_hat[3],
		[[1.0000, 0.0000],
		 [0.6530, 0.3470],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.3728, 0.6272],
		 [0.4737, 0.5263],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.2308, 0.7692],
		 [1.0000, 0.0000],
		 [0.4737, 0.5263]], 4)


def test_predict_log_proba_diamond(X_masked, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), 
		(d22, d13), (d32, d13)])
	y_hat = model.predict_log_proba(X_masked)

	assert_array_almost_equal(y_hat[0], 
		numpy.log([[0.3000, 0.7000],
		 [1.0000, 0.0000],
		 [0.4700, 0.5300],
		 [0.0000, 1.0000],
		 [0.2416, 0.7584],
		 [1.0000, 0.0000],
		 [0.4700, 0.5300],
		 [0.0000, 1.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [1.0000, 0.0000]]), 3)

	assert_array_almost_equal(y_hat[1],
		numpy.log([[0.0000, 0.0000, 1.0000],
		 [1.0000, 0.0000, 0.0000],
		 [0.7000, 0.1000, 0.2000],
		 [0.0000, 0.0000, 1.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.7000, 0.1000, 0.2000],
		 [0.5147, 0.4706, 0.0147],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 0.0000, 1.0000],
		 [0.7204, 0.1844, 0.0952]]), 3)

	assert_array_almost_equal(y_hat[2],
		numpy.log([[1.0000, 0.0000],
		 [0.4000, 0.6000],
		 [0.5000, 0.5000],
		 [0.0000, 1.0000],
		 [0.1387, 0.8613],
		 [1.0000, 0.0000],
		 [0.5000, 0.5000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000]]), 3)

	assert_array_almost_equal(y_hat[3],
		numpy.log([[1.0000, 0.0000],
		 [0.4500, 0.5500],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.0968, 0.9032],
		 [0.3488, 0.6512],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.3488, 0.6512],
		 [1.0000, 0.0000],
		 [0.6818, 0.3182]]), 3)


def test_predict_cycle(X, X_masked, distributions):
	model = BayesianNetwork(structure=((1,), (2,), (3,), (0,))).fit(X)
	y_hat = model.predict(X_masked)
	
	assert_array_equal(y_hat,
		[[1, 2, 0, 0],
		 [0, 0, 1, 0],
		 [0, 1, 1, 0],
		 [1, 2, 1, 1],
		 [1, 1, 0, 1],
		 [0, 1, 0, 1],
		 [0, 1, 1, 0],
		 [1, 2, 1, 1],
		 [1, 1, 0, 1],
		 [0, 2, 1, 0],
		 [0, 1, 0, 1]])


def test_predict_diamond(X_masked, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), 
		(d22, d13), (d32, d13)])
	y_hat = model.predict(X_masked)

	assert_array_equal(y_hat,
		[[1, 2, 0, 0],
		 [0, 0, 1, 1],
		 [1, 0, 0, 0],
		 [1, 2, 1, 1],
		 [1, 1, 1, 1],
		 [0, 1, 0, 1],
		 [1, 0, 0, 0],
		 [1, 0, 1, 1],
		 [1, 1, 0, 1],
		 [0, 2, 1, 0],
		 [0, 0, 0, 0]])


def test_predict_chain(): 
	# Inspiration from https://github.com/jmschrei/pomegranate/issues/1108

	length = 23
	for i in [1, 5, 17, 25]:
		model = BayesianNetwork(max_iter=i)
		d0 = Categorical([[0.5, 0.5]])
		model.add_distributions([d0])

		ds = []
		for j in range(length-1):
			d = ConditionalCategorical([[[1.0, 0.0], [0.5, 0.5]]])
			ds.append(d)
			model.add_distributions([d])
			
			if j == 0:
				model.add_edge(d0, d)
			else:
				model.add_edge(ds[-2], d)

		e = [-1] * (length-1) + [1]
		X = torch.tensor([e])
		X_masked = torch.masked.MaskedTensor(X, mask=X >= 0)

		y_hat = model.predict(X_masked)
		assert y_hat.sum() == min((i + 1), length)


def test_fit(X, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions
	model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), 
		(d22, d13), (d32, d13)])

	model.fit(X)
	assert_array_almost_equal(model.distributions[0].probs[0],
		[[[1.0000, 0.0000],
		 [0.5000, 0.5000]],

		[[0.5000, 0.5000],
		 [0.0000, 1.0000]],

		[[0.0000, 1.0000],
		 [0.5000, 0.5000]]])
	assert_array_almost_equal(model.distributions[1].probs[0], 
		[[0.2000, 0.4000, 0.4000],
		 [0.3333, 0.5000, 0.1667]], 4)
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[0.4000, 0.6000],
		 [0.6667, 0.3333]], 4)
	assert_array_almost_equal(model.distributions[3].probs, 
		[[0.4545, 0.5455]], 4)


def test_fit_weighted(X, w, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions
	model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), 
		(d22, d13), (d32, d13)])

	model.fit(X, sample_weight=w)
	assert_array_almost_equal(model.distributions[0].probs[0],
		[[[1.0000, 0.0000],
		  [1.0000, 0.0000]],

		 [[0.5701, 0.4299],
		  [0.5000, 0.5000]],

		 [[0.5000, 0.5000],
		  [0.2000, 0.8000]]], 4)
	assert_array_almost_equal(model.distributions[1].probs[0], 
		[[0.8125, 0.0000, 0.1875],
		 [0.2171, 0.7039, 0.0789]], 4)
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[0.0000, 1.0000],
		 [0.9211, 0.0789]], 4)
	assert_array_almost_equal(model.distributions[3].probs, 
		[[0.0952, 0.9048]], 4)


def test_fit_structure(X):
	model = BayesianNetwork(structure=((1, 2), (3,), (3,), ()))
	model.fit(X)

	assert_array_almost_equal(model.distributions[0].probs[0],
		[[[1.0000, 0.0000],
		 [0.5000, 0.5000]],

		[[0.5000, 0.5000],
		 [0.0000, 1.0000]],

		[[0.0000, 1.0000],
		 [0.5000, 0.5000]]])
	assert_array_almost_equal(model.distributions[1].probs[0], 
		[[0.2000, 0.4000, 0.4000],
		 [0.3333, 0.5000, 0.1667]], 4)
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[0.4000, 0.6000],
		 [0.6667, 0.3333]], 4)
	assert_array_almost_equal(model.distributions[3].probs, 
		[[0.4545, 0.5455]], 4)


def test_fit_fg(X, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions
	model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), 
		(d22, d13), (d32, d13)])

	model.fit(X)
	assert_array_almost_equal(model._factor_graph.factors[0].probs,
		[[[0.1667, 0.0000],
		  [0.0833, 0.0833]],

		 [[0.0833, 0.0833],
		  [0.0000, 0.1667]],

		 [[0.0000, 0.1667],
		  [0.0833, 0.0833]]], 4)
	assert_array_almost_equal(model._factor_graph.factors[1].probs, 
		[[0.1000, 0.2000, 0.2000],
		 [0.1667, 0.2500, 0.0833]], 4)
	assert_array_almost_equal(model._factor_graph.factors[2].probs,
		[[0.2000, 0.3000],
		 [0.3333, 0.1667]], 4)
	assert_array_almost_equal(model._factor_graph.factors[3].probs, 
		[[0.4545, 0.5455]], 4)


def test_fit_weighted_fg(X, w, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions
	model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), 
		(d22, d13), (d32, d13)])

	model.fit(X, sample_weight=w)
	assert_array_almost_equal(model._factor_graph.factors[0].probs,
		[[[0.1667, 0.0000],
		  [0.1667, 0.0000]],

		 [[0.0950, 0.0717],
		  [0.0833, 0.0833]],

		 [[0.0833, 0.0833],
		  [0.0333, 0.1333]]], 4)
	assert_array_almost_equal(model._factor_graph.factors[1].probs, 
		[[0.4062, 0.0000, 0.0938],
		 [0.1086, 0.3520, 0.0395]], 4)
	assert_array_almost_equal(model._factor_graph.factors[2].probs,
		[[0.0000, 0.5000],
		 [0.4605, 0.0395]], 4)
	assert_array_almost_equal(model._factor_graph.factors[3].probs, 
		[[0.0952, 0.9048]], 4)


def test_fit_predict_proba(X, X_masked, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions

	model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), 
		(d22, d13), (d32, d13)])
	y_hat = model.predict_proba(X_masked)

	assert_array_almost_equal(y_hat[0], 
		[[0.3000, 0.7000],
		 [1.0000, 0.0000],
		 [0.4700, 0.5300],
		 [0.0000, 1.0000],
		 [0.2416, 0.7584],
		 [1.0000, 0.0000],
		 [0.4700, 0.5300],
		 [0.0000, 1.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [1.0000, 0.0000]], 4)

	assert_array_almost_equal(y_hat[1],
		[[0.0000, 0.0000, 1.0000],
		 [1.0000, 0.0000, 0.0000],
		 [0.7000, 0.1000, 0.2000],
		 [0.0000, 0.0000, 1.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.7000, 0.1000, 0.2000],
		 [0.5147, 0.4706, 0.0147],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 0.0000, 1.0000],
		 [0.7204, 0.1844, 0.0952]], 4)

	assert_array_almost_equal(y_hat[2],
		[[1.0000, 0.0000],
		 [0.4000, 0.6000],
		 [0.5000, 0.5000],
		 [0.0000, 1.0000],
		 [0.1387, 0.8613],
		 [1.0000, 0.0000],
		 [0.5000, 0.5000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000]], 4)

	assert_array_almost_equal(y_hat[3],
		[[1.0000, 0.0000],
		 [0.4500, 0.5500],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.0968, 0.9032],
		 [0.3488, 0.6512],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.3488, 0.6512],
		 [1.0000, 0.0000],
		 [0.6818, 0.3182]], 4)

	model.fit(X)
	y_hat = model.predict_proba(X_masked)

	assert_array_almost_equal(y_hat[0], 
		[[0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [0.3400, 0.6600],
		 [0.0000, 1.0000],
		 [0.2800, 0.7200],
		 [1.0000, 0.0000],
		 [0.3400, 0.6600],
		 [0.0000, 1.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [1.0000, 0.0000]], 4)

	assert_array_almost_equal(y_hat[1],
		[[0.0000, 0.0000, 1.0000],
		 [1.0000, 0.0000, 0.0000],
		 [0.2000, 0.4000, 0.4000],
		 [0.0000, 0.0000, 1.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 1.0000, 0.0000],
		 [0.2000, 0.4000, 0.4000],
		 [0.2222, 0.6667, 0.1111],
		 [0.0000, 1.0000, 0.0000],
		 [0.0000, 0.0000, 1.0000],
		 [0.5532, 0.4468, 0.0000]], 4)

	assert_array_almost_equal(y_hat[2],
		[[1.0000, 0.0000],
		 [0.7324, 0.2676],
		 [0.4000, 0.6000],
		 [0.0000, 1.0000],
		 [0.5600, 0.4400],
		 [1.0000, 0.0000],
		 [0.4000, 0.6000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [1.0000, 0.0000]], 4)

	assert_array_almost_equal(y_hat[3],
		[[1.0000, 0.0000],
		 [0.2958, 0.7042],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.4000, 0.6000],
		 [0.2857, 0.7143],
		 [1.0000, 0.0000],
		 [0.0000, 1.0000],
		 [0.2857, 0.7143],
		 [1.0000, 0.0000],
		 [0.2553, 0.7447]], 4)


def test_learn_structure_chow_liu(X):
	X = numpy.array(X)
	X[:,1] = X[:,3]

	model = BayesianNetwork(algorithm='chow-liu')
	model.fit(X)

	assert_tuple_equal(model._parents, ((), (0,), (1,), (1,)))

	assert_array_almost_equal(model.distributions[0].probs,
		[[0.4545, 0.5455]], 4)
	assert_array_almost_equal(model.distributions[1].probs[0], 
		[[0.6000, 0.4000], [0.3333, 0.6667]], 4)
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[0.4000, 0.6000], [0.6667, 0.3333]], 4)
	assert_array_almost_equal(model.distributions[3].probs[0], 
		[[1., 0.], [0., 1.]], 4)

	assert_array_almost_equal(model._factor_graph.factors[0].probs, 
		[[0.4545, 0.5455]], 4)
	assert_array_almost_equal(model._factor_graph.factors[1].probs, 
		[[0.3000, 0.2000], [0.1667, 0.3333]], 4)
	assert_array_almost_equal(model._factor_graph.factors[2].probs, 
		[[0.2000, 0.3000], [0.3333, 0.1667]], 4)
	assert_array_almost_equal(model._factor_graph.factors[3].probs, 
		[[0.5000, 0.0000], [0.0000, 0.5000]], 4)


def test_learn_structure_exact(X):
	X = numpy.array(X)
	X[:,1] = X[:,3]

	model = BayesianNetwork(algorithm='exact')
	model.fit(X)

	assert_tuple_equal(model._parents, ((), (), (0, 1), ()))

	assert_array_almost_equal(model.distributions[0].probs,
		[[0.4545, 0.5455]], 4)
	assert_array_almost_equal(model.distributions[1].probs, 
		[[0.4545, 0.5455]], 4)
	assert_array_almost_equal(model.distributions[3].probs,
		[[0.4545, 0.5455]], 4)
	assert_array_almost_equal(model.distributions[2].probs[0], 
		[[[0.3333, 0.6667],
		  [1.0000, 0.0000]],

		 [[0.5, 0.5],
		  [0.5, 0.5]]], 4)

	assert_array_almost_equal(model._factor_graph.factors[0].probs, 
		[[0.4545, 0.5455]], 4)
	assert_array_almost_equal(model._factor_graph.factors[1].probs, 
		[[0.4545, 0.5455]], 4)
	assert_array_almost_equal(model._factor_graph.factors[3].probs, 
		[[0.4545, 0.5455]], 4)
	assert_array_almost_equal(model._factor_graph.factors[2].probs, 
		[[[0.0833, 0.1667],
		  [0.2500, 0.0000]],

		 [[0.1250, 0.1250],
		  [0.1250, 0.1250]]], 4)


def test_summarize(X, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions
	model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), 
		(d22, d13), (d32, d13)])

	model.summarize(X[:4])
	model.summarize(X[4:])
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0].probs[0],
		[[[1.0000, 0.0000],
		 [0.5000, 0.5000]],

		[[0.5000, 0.5000],
		 [0.0000, 1.0000]],

		[[0.0000, 1.0000],
		 [0.5000, 0.5000]]])
	assert_array_almost_equal(model.distributions[1].probs[0], 
		[[0.2000, 0.4000, 0.4000],
		 [0.3333, 0.5000, 0.1667]], 4)
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[0.4000, 0.6000],
		 [0.6667, 0.3333]], 4)
	assert_array_almost_equal(model.distributions[3].probs, 
		[[0.4545, 0.5455]], 4)


def test_summarize_weighted(X, w, distributions):
	d1, d2, d3, d4, d12, d22, d32, d13 = distributions
	model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), 
		(d22, d13), (d32, d13)])

	model.summarize(X[:4], sample_weight=w[:4])
	model.summarize(X[4:], sample_weight=w[4:])
	model.from_summaries()

	assert_array_almost_equal(model.distributions[0].probs[0],
		[[[1.0000, 0.0000],
		  [1.0000, 0.0000]],

		 [[0.5701, 0.4299],
		  [0.5000, 0.5000]],

		 [[0.5000, 0.5000],
		  [0.2000, 0.8000]]], 4)
	assert_array_almost_equal(model.distributions[1].probs[0], 
		[[0.8125, 0.0000, 0.1875],
		 [0.2171, 0.7039, 0.0789]], 4)
	assert_array_almost_equal(model.distributions[2].probs[0],
		[[0.0000, 1.0000],
		 [0.9211, 0.0789]], 4)
	assert_array_almost_equal(model.distributions[3].probs, 
		[[0.0952, 0.9048]], 4)


###
# Soft evidence (per-variable nonnegative likelihood vectors)
###


def _monty_hall_bn():
	p = numpy.array([[
		 [[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
		 [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]],
		 [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
	]])

	d1 = Categorical([[1./3, 1./3, 1./3]])
	d2 = Categorical([[1./3, 1./3, 1./3]])
	d3 = ConditionalCategorical(p)

	return BayesianNetwork([d1, d2, d3], [(d1, d3), (d2, d3)]), p


def _unobserved(n, d=3):
	return torch.masked.MaskedTensor(torch.zeros(n, d, dtype=torch.int64),
		mask=torch.zeros(n, d, dtype=torch.bool))


def _brute_force_bn(p, likelihoods):
	n = next(e.shape[0] for e in likelihoods if e is not None)
	joint = numpy.zeros((n, 3, 3, 3))
	for c, g, mo in itertools.product(range(3), range(3), range(3)):
		v = (1. / 3) * (1. / 3) * p[0, c, g, mo]
		for j in range(n):
			w = v
			for i, val in enumerate((c, g, mo)):
				if likelihoods[i] is not None:
					w *= likelihoods[i][j, val]
			joint[j, c, g, mo] = w

	joint /= joint.sum(axis=(1, 2, 3), keepdims=True)
	return [joint.sum(axis=(2, 3)), joint.sum(axis=(1, 3)),
		joint.sum(axis=(1, 2))]


def test_bn_soft_evidence_matches_brute_force():
	model, p = _monty_hall_bn()
	l0 = numpy.array([[0.7, 0.2, 0.1], [2.0, 1.0, 0.5]])

	y_hat = model.predict_proba(_unobserved(2),
		evidence=[torch.tensor(l0), None, None])
	ref = _brute_force_bn(p, [l0, None, None])

	for i in range(3):
		assert_array_almost_equal(y_hat[i], ref[i], 5)


def test_bn_soft_evidence_scale_invariance():
	model, _ = _monty_hall_bn()
	l0 = torch.tensor([[0.7, 0.2, 0.1], [2.0, 1.0, 0.5]])

	y_hat = model.predict_proba(_unobserved(2), evidence=[l0, None, None])
	y_scaled = model.predict_proba(_unobserved(2),
		evidence=[l0 * 12.8, None, None])

	for a, b in zip(y_hat, y_scaled):
		assert_array_almost_equal(a, b, 6)


def test_bn_soft_evidence_shared_contract():
	# predict_proba, predict_log_proba and predict all take the same
	# evidence argument and never reduce a soft vector to an argmax label.
	model, _ = _monty_hall_bn()
	l0 = torch.tensor([[0.99, 0.01, 0.0]])

	y_hat = model.predict_proba(_unobserved(1), evidence=[l0, None, None])
	log_hat = model.predict_log_proba(_unobserved(1),
		evidence=[l0, None, None])
	pred = model.predict(_unobserved(1), evidence=[l0, None, None])

	assert_array_almost_equal(y_hat[0], [[0.99, 0.01, 0.0]])
	assert_array_almost_equal(log_hat[0], numpy.log([[0.99, 0.01, 0.0]]), 5)
	assert_array_equal(pred,
		torch.vstack([t.argmax(dim=1) for t in y_hat]).T)

	# The completion must differ from hard-evidence on category 0.
	hard = torch.masked.MaskedTensor(torch.tensor([[0, 0, 0]]),
		mask=torch.tensor([[True, False, False]]))
	y_hard = model.predict_proba(hard)
	assert not numpy.allclose(y_hard[0].numpy(), y_hat[0].numpy())


def test_bn_soft_evidence_mixed_rows_and_completion():
	# Different rows observe different variables: hard evidence in one row,
	# soft evidence on another variable in another row.
	model, p = _monty_hall_bn()

	X = torch.tensor([[1, 0, 0], [0, 0, 0]])
	mask = torch.tensor([[False, True, False],
		[False, False, False]])
	soft = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])

	y_hat = model.predict_proba(torch.masked.MaskedTensor(X, mask=mask),
		evidence=[soft, None, None])

	# Row 0 hard-observes variable 1 at category 0, so it is clamped there.
	assert_array_almost_equal(y_hat[1][0], [1.0, 0.0, 0.0])
	ref = _brute_force_bn(p, [soft.numpy(), None, None])
	assert_array_almost_equal(y_hat[0][1], ref[0][1], 5)

	pred = model.predict(torch.masked.MaskedTensor(X, mask=mask),
		evidence=[soft, None, None])
	# Hard-observed cell is preserved; the others come from posterior argmax.
	assert pred[0, 1] == 0
	assert_array_equal(pred[:, 0].numpy(),
		[t.argmax() for t in y_hat[0]])


def test_bn_soft_evidence_raises():
	model, _ = _monty_hall_bn()

	# All-zero likelihood vector.
	assert_raises(ValueError, model.predict_proba, _unobserved(1),
		evidence=[None, torch.zeros(1, 3), None])

	# Hard/soft conflict.
	X = torch.masked.MaskedTensor(torch.tensor([[1, 0, 0]]),
		mask=torch.tensor([[True, False, False]]))
	assert_raises(ValueError, model.predict_proba, X,
		evidence=[torch.tensor([[0.6, 0.0, 0.4]]), None, None])

	# Bad evidence length.
	assert_raises(ValueError, model.predict_proba, _unobserved(1),
		evidence=[None, None])


def test_bn_soft_evidence_does_not_mutate_model():
	model, _ = _monty_hall_bn()
	fg_before = [torch.clone(f.probs) for f in model._factor_graph.factors]
	m_before = [torch.clone(m.probs) for m in model._factor_graph.marginals]

	model.predict_proba(_unobserved(2), evidence=[
		torch.tensor([[0.7, 0.2, 0.1], [2.0, 1.0, 0.5]]), None, None])

	for f, b in zip(model._factor_graph.factors, fg_before):
		assert torch.allclose(f.probs, b)
	for m, b in zip(model._factor_graph.marginals, m_before):
		assert torch.allclose(m.probs, b)

# factor_graph.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import torch
import itertools

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution
from .distributions._distribution import ConditionalDistribution

from .distributions import Categorical
from .distributions import JointCategorical


class FactorGraph(Distribution):
	"""A factor graph object.

	A factor graph represents a probability distribution as a bipartite graph
	where marginal distributions of each dimension in the distribution are on
	one side of the graph and factors are on the other side. The distributions
	on the factor side encode probability estimates from the model, whereas the
	distributions on the marginal side encode probability estimates from the
	data. 

	Inference is done on the factor graph using the loopy belief propagation
	algorithm. This is an iterative algorithm where "messages" are passed
	along each edge between the marginals and the factors until the estimates
	for the marginals converges. In brief: each message represents what the
	generating node thinks its marginal distribution is with respect to the
	child node. Calculating each message involves marginalizing the parent
	node with respect to *every other* node. When the parent node is already
	a univariate distribution -- either because it is a marginal node or
	a univariate factor node -- no marginalization is needed and it sends
	itself as the message. Basically, a joint probability table will receive
	messages from all the marginal nodes that comprise its dimensions and,
	to each of those marginal nodes, it will send a message back saying what
	it (the joint probability table) thinks its marginal distribution is with
	respect to the messages from the OTHER marginals. More concretely, if a
	joint probability table has two dimensions with marginal node parents
	A and B, it will send a message to A that is itself after marginalizing out
	B, and will send a message to B that is itself after marginalizing out A. 

	..note:: It is worth noting that this algorithm is exact when the structure
	is a tree. If there exist any loops in the factors, i.e., you can draw a
	circle beginning with a factor and then hopping between marginals and
	factors and make it back to the factor without crossing any edges twice,
	the probabilities returned are approximate.


	Parameters
	----------
	factors: tuple or list or None
		A set of distribution objects. These do not need to be initialized,
		i.e. can be "Categorical()". Currently, they must be either Categorical
		or JointCategorical distributions. Default is None.

	marginals: tuple or list or None
		A set of distribution objects. These must be initialized and be
		Categorical distributions.

	edges: list or tuple or None
		A set of edges. Critically, the items in this list must be the
		distribution objects themselves, and the order that edges must match
		the order distributions in a multivariate distribution. Specifically,
		if you have a multivariate distribution, the first edge that includes
		it must correspond to the first dimension, the second edge must
		correspond to the second dimension, etc, and the total number of
		edges cannot exceed the number of dimensions. Default is None.

	max_iter: int, optional
		The number of iterations to do in the inference step as distributions
		are converging. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 1e-6.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

	def __init__(self, factors=None, marginals=None, edges=None, max_iter=20, 
		tol=1e-6, inertia=0.0, frozen=False, check_data=True, verbose=False):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "FactorGraph"

		self.factors = torch.nn.ModuleList([])
		self.marginals = torch.nn.ModuleList([])

		self._factor_idxs = {}
		self._marginal_idxs = {}

		self._factor_edges = []
		self._marginal_edges = []

		self.max_iter = _check_parameter(max_iter, "max_iter", min_value=1, 
			dtypes=[int, torch.int16, torch.int32, torch.int64])
		self.tol = _check_parameter(tol, "tol", min_value=0)
		self.verbose = verbose

		self.d = 0
		self._initialized = factors is not None and factors[0]._initialized

		if factors is not None:
			_check_parameter(factors, "factors", dtypes=(list, tuple))

			for factor in factors:
				self.add_factor(factor)

		if marginals is not None:
			_check_parameter(marginals, "marginals", dtypes=(list, tuple))

			for marginal in marginals:
				self.add_marginal(marginal)

		if edges is not None:
			_check_parameter(edges, "edges", dtypes=(list, tuple))

			for marginal, factor in edges:
				self.add_edge(marginal, factor)

		self._initialized = not factors

	def _initialize(self, d):
		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		return

	def add_factor(self, distribution):
		"""Adds a distribution to the set of factors.

		
		Parameters
		----------
		distribution: pomegranate.distributions.Distribution
			A distribution object to include as a node.
		"""

		if not isinstance(distribution, (Categorical, JointCategorical)):
			raise ValueError("Must be a Categorical or a JointCategorical"
				" distribution.")

		self.factors.append(distribution)
		self._factor_edges.append([])
		self._factor_idxs[distribution] = len(self.factors) - 1

		self._initialized = distribution._initialized

	def add_marginal(self, distribution):
		"""Adds a distribution to the set of marginals.

		This adds a distribution to the marginal side of the bipartate graph.
		This distribution must be univariate. 

		Parameters
		----------
		distribution: pomegranate.distributions.Distribution
			A distribution object to include as a node.
		"""

		if not isinstance(distribution, Categorical):
			raise ValueError("Must be a Categorical distribution.")
		
		self.marginals.append(distribution)
		self._marginal_edges.append([])
		self._marginal_idxs[distribution] = len(self.marginals) - 1

		self.d += 1

	def _check_evidence(self, X, evidence=None):
		"""Validate hard and soft evidence and return per-variable likelihoods.

		Hard evidence is given through a masked tensor, as before. Soft
		evidence is given, per variable, as a matrix of nonnegative likelihood
		vectors with one entry per category. The two forms share a single
		contract:

			- `evidence` is a list/tuple with one entry per marginal. An entry
			  of None means that no soft evidence is supplied for that variable.
			- Each non-None entry has shape (n, n_keys) for that variable, or
			  shape (n_keys,) when the same likelihood vector applies to every
			  row, and every value must be nonnegative.
			- An all-zero likelihood vector is invalid because it rules out
			  every category, so it raises a ValueError naming the bad rows.
			- When a variable carries both hard evidence and a soft likelihood
			  vector in the same row, the hard-observed category must have a
			  positive likelihood. Otherwise the two forms of evidence
			  conflict and a ValueError is raised.

		Soft evidence only parameterizes this inference call; nothing is
		written back to the marginal nodes or to the factor probability
		tables. Likelihood vectors are only defined up to a positive common
		multiple, so the vectors are normalized before message passing and
		scaling a vector by any positive constant leaves the posterior
		unchanged.


		Parameters
		----------
		X: torch.masked.MaskedTensor, shape=(-1, d)
			The hard evidence. Masked-in values are observed categories and
			masked-out values are missing.

		evidence: list, tuple, or None, optional
			One entry per marginal. Each entry is None or a nonnegative
			likelihood tensor of shape (n, n_keys) (or (n_keys,)). Default
			is None.


		Returns
		-------
		likelihoods: list of torch.Tensor or None
			A per-variable list where each entry is either None or a
			normalized likelihood tensor of shape (n, n_keys).
		"""

		if not isinstance(X, torch.masked.MaskedTensor):
			raise TypeError("X must be a torch.masked.MaskedTensor where the "
				"mask indicates which variables are observed.")

		n, nm = X.shape
		if nm != len(self.marginals):
			raise ValueError("X.shape[1] must match the number of marginals.")

		if evidence is None:
			evidence = [None] * nm

		if not isinstance(evidence, (list, tuple)) or len(evidence) != nm:
			raise ValueError("evidence must be a list or tuple with one entry "
				"per marginal, expected length {} but got length {}.".format(
					nm, 0 if evidence is None else len(evidence)))

		likelihoods = []
		for i, e in enumerate(evidence):
			n_keys = self.marginals[i].probs.shape[-1]

			if e is None:
				likelihoods.append(None)
				continue

			e = _cast_as_tensor(e, dtype=torch.float32)
			if e.ndim == 1:
				e = e.unsqueeze(0).expand(n, -1)
			elif e.ndim != 2:
				raise ValueError("evidence[{}] must have shape ({}, {}) or "
					"({},).".format(i, n, n_keys, n_keys))

			_check_parameter(e, "evidence[{}]".format(i), min_value=0.0,
				ndim=2, shape=(n, n_keys), check_parameter=self.check_data)

			zero_rows = (e > 0).sum(dim=1) == 0
			if zero_rows.any():
				rows = torch.nonzero(zero_rows, as_tuple=False).flatten()
				raise ValueError("Soft evidence for variable {} contains an "
					"all-zero likelihood vector in row(s) {}. Every "
					"likelihood vector must have at least one positive "
					"entry.".format(i, rows.tolist()))

			likelihoods.append(e / e.sum(dim=1, keepdims=True))

		# Check that hard evidence does not conflict with soft evidence:
		# a hard-observed category must retain positive likelihood.
		hard_mask = X._masked_mask
		hard_data = X._masked_data
		for i, e in enumerate(likelihoods):
			if e is None or not hard_mask[:, i].any():
				continue

			observed = hard_mask[:, i]
			values = hard_data[observed, i].long()
			compat = e[observed, :].gather(1, values.unsqueeze(1)).squeeze(1)
			conflicting = compat == 0
			if conflicting.any():
				rows = torch.nonzero(observed, as_tuple=False).flatten()
				raise ValueError("Hard and soft evidence for variable {} "
					"conflict in row(s) {}: the hard-observed category has "
					"zero likelihood.".format(i,
						rows[conflicting].tolist()))

		return likelihoods

	def add_edge(self, marginal, factor):
		"""Adds an undirected edge to the set of edges.

		Because a factor graph is a bipartite graph, one of the edges must be
		a marginal distribution and the other edge must be a factor 
		distribution.

		Parameters
		----------
		marginal: pomegranate.distributions.Distribution
			The marginal distribution to include in the edge.

		factor: pomegranate.distributions.Distribution
			The factor distribution to include in the edge.
		"""

		if marginal not in self._marginal_idxs:
			raise ValueError("Marginal distribution does not exist in graph.")
		
		if factor not in self._factor_idxs:
			raise ValueError("Factor distribution does not exist in graph.")

		m_idx = self._marginal_idxs[marginal]
		f_idx = self._factor_idxs[factor]

		self._factor_edges[f_idx].append(m_idx)
		self._marginal_edges[m_idx].append(f_idx)

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			check_parameter=self.check_data)
		
		logps = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
		for idxs, factor in zip(self._factor_edges, self.factors):
			logps += factor.log_probability(X[:, idxs])

		for i, marginal in enumerate(self.marginals):
			logps += marginal.log_probability(X[:, i:i+1])

		return logps

	def predict(self, X, evidence=None):
		"""Infers the maximum likelihood value for each missing value.

		This method infers a probability distribution for each of the missing
		values in the data. First, the sum-product algorithm is run to infer
		a probability distribution for each variable. Then, the maximum
		likelihood value is returned from that distribution.

		The input to this method must be a torch.masked.MaskedTensor where the
		mask specifies which variables are observed (mask = True) and which ones
		are not observed (mask = False) for each of the values. When setting
		mask = False, it does not matter what the corresponding value in the
		tensor is. Different sets of variables can be observed or missing in
		different examples.

		Soft evidence may be supplied through `evidence` using the same
		contract as `predict_proba`. The soft vectors are used to compute the
		posteriors and are never turned into hard labels beforehand; the
		argmax below is taken only on the resulting posteriors for the
		missing-value completion.

		Unlike the `predict_proba` and `predict_log_proba` methods, this
		method preserves the dimensions of the original data because it does
		not matter how many categories a variable can take when you're only
		returning the maximally likely one.


		Parameters
		----------
		X: torch.masked.MaskedTensor
			A masked tensor where the observed values are available and the
			unobserved values are missing, i.e., the mask is True for
			observed values and is False for missing values. It does not
			matter what the underlying value in the tensor is for the
			missing values.

		evidence: list, tuple, or None, optional
			Per-variable nonnegative likelihood vectors; see
			`predict_proba` for the full contract. Default is None.
		"""

		y = [t.argmax(dim=1) for t in self.predict_proba(X, evidence)]
		return torch.vstack(y).T.contiguous()

	def predict_proba(self, X, evidence=None):
		"""Predict the probability of each variable given some evidence.

		Given some evidence about the value that each variable takes, infer
		the value that each other variable takes. If no evidence is given,
		this returns the marginal value of each variable given the dependence
		structure in the network.

		Two forms of evidence are supported and can be combined:

			- Hard evidence takes the form of a masked tensor where masked-in
			  values are observed discrete categories and masked-out values
			  are missing.
			- Soft evidence takes the form, per variable, of a nonnegative
			  likelihood vector that assigns a relative credibility to each
			  category. These vectors are passed in through `evidence` and
			  participate in message passing alongside the hard evidence.
			  They are not converted into hard labels: the full vector is
			  used, even when one entry is much larger than the others, and
			  nothing is written back to the model's probability tables.

		Different rows can observe different variables, and a variable may
		carry hard evidence in some rows and soft evidence in others. Hard
		and soft evidence for the same variable in the same row must not
		conflict, and an all-zero likelihood vector is rejected; see
		`_check_evidence` for the full contract. Any positive common multiple
		of a likelihood vector yields the same posterior.


		Parameters
		----------
		X: torch.masked.MaskedTensor
			A masked tensor where the observed values are available and the
			unobserved values are missing, i.e., the mask is True for
			observed values and is False for missing values. It does not
			matter what the underlying value in the tensor is for the
			missing values.

		evidence: list, tuple, or None, optional
			One entry per marginal variable. An entry of None means no soft
			evidence for that variable. Otherwise the entry is a nonnegative
			tensor of likelihoods with shape (n, n_keys), or shape
			(n_keys,) when one vector is reused for every row. Default is
			None.
		"""

		nm = len(self.marginals)
		nf = len(self.factors)

		likelihoods = self._check_evidence(X, evidence)
		n = X.shape[0]

		# Pull out the current distributions from factors and marginals
		factors = []
		marginals = []
		prior_marginals = []
		current_marginals = []

		for i, m in enumerate(self.marginals):
			p = torch.clone(m.probs[0])
			p = p.repeat((n,) + tuple(1 for _ in p.shape))

			# Use the hard evidence in the data to set marginal distributions
			for j in range(n):
				if X._masked_mask[j, i] == True:
					value = X._masked_data[j, i]

					p[j] = 0
					p[j, value] = 1.0

			# Incorporate the soft evidence as a likelihood over categories.
			# The likelihood is multiplied in rather than replacing the
			# marginal, so hard evidence is never derived from an argmax and
			# the relative credibility of every category is retained.
			if likelihoods[i] is not None:
				p = p * likelihoods[i].to(p.device)
				p = p / p.sum(dim=1, keepdims=True)

			marginals.append(p)
			prior_marginals.append(torch.clone(p))
			current_marginals.append(torch.clone(p))

		for i, f in enumerate(self.factors):
			if not isinstance(f, Categorical):
				p = torch.clone(f.probs)
			else:
				p = torch.clone(f.probs[0])

			p = p.repeat((n,) + tuple(1 for _ in p.shape))
			factors.append(p)

		# Set the original in and out messages along the edges
		in_messages, out_messages = [], []

		for i, m in enumerate(marginals):
			k = len(self._marginal_edges[i])
			in_messages.append([])

			for j in range(k):
				in_messages[-1].append(m)

		for i in range(len(factors)):
			k = len(self._factor_edges[i])
			out_messages.append([])
			
			for j in range(k):
				marginal_idx = self._factor_edges[i][j]
				d_j = marginals[marginal_idx]
				out_messages[-1].append(d_j)


		# Begin iterations
		iteration = 0
		while iteration < self.max_iter:
			# Update the messages going into the nodes.
			for i, f in enumerate(factors):
				ni_edges = len(self._factor_edges[i])

				for k in range(ni_edges):
					message = torch.clone(f)

					shape = torch.ones(len(message.shape), dtype=torch.int32)
					shape[0] = n

					for l in range(ni_edges):
						if k == l:
							continue

						shape[l+1] = message.shape[l+1]						
						message *= out_messages[i][l].reshape(*shape)
						message = torch.sum(message, dim=l+1, keepdims=True)
						shape[l+1] = 1

					else:
						message = message.squeeze()
						if len(message.shape) == 1:
							message = message.unsqueeze(0)

					j = self._factor_edges[i][k]
					for ik, parent in enumerate(self._marginal_edges[j]):
						if parent == i:
							dims = tuple(range(1, len(message.shape)))
							in_messages[j][ik] = message / message.sum(
								dim=dims, keepdims=True)

							break


			# Calculate the current estimates of the marginals
			loss = 0
			for i, m in enumerate(marginals):
				current_marginals[i] = torch.clone(m)

				for k in range(len(self._marginal_edges[i])):
					current_marginals[i] *= in_messages[i][k]


				dims = tuple(range(1, len(current_marginals[i].shape)))
				current_marginals[i] /= current_marginals[i].sum(dim=dims, 
					keepdims=True)

				loss += torch.nn.KLDivLoss(reduction="batchmean")(torch.log(
					current_marginals[i] + 1e-8), prior_marginals[i])

			if self.verbose:
				print(iteration, loss.item())

			if loss < self.tol:
				break


			# Update the messages leaving based on the new marginals
			for i, m in enumerate(marginals):
				ni_edges = len(self._marginal_edges[i])

				for k in range(ni_edges):
					message = torch.clone(m)

					for l in range(ni_edges):
						if k == l:
							continue

						message *= in_messages[i][l]

					j = self._marginal_edges[i][k]
					for ik, parent in enumerate(self._factor_edges[j]):
						if parent == i:
							dims = tuple(range(1, len(message.shape)))
							out_messages[j][ik] = message / message.sum(
								dim=dims, keepdims=True)
							break

			prior_marginals = [torch.clone(d) for d in current_marginals]
			iteration += 1


		return current_marginals

	def predict_log_proba(self, X, evidence=None):
		"""Infers the log probability of each category given evidence.

		This method is a wrapper around the `predict_proba` method and simply
		takes the log of each returned tensor. The hard and soft evidence
		follow the same shared contract as `predict_proba`: soft likelihood
		vectors participate in message passing and are not argmaxed into hard
		labels.


		Parameters
		----------
		X: torch.masked.MaskedTensor, shape=(-1, d)
			The data to predict values for. The mask should correspond to
			whether the variable is observed in the example.

		evidence: list, tuple, or None, optional
			Per-variable nonnegative likelihood vectors; see
			`predict_proba` for the full contract. Default is None.


		Returns
		-------
		y: list of tensors, shape=(d,)
			A list of tensors where each tensor contains the log distribution
			of values for that dimension.
		"""

		return [torch.log(t) for t in self.predict_proba(X, evidence)]

	def fit(self, X, sample_weight=None):
		"""Fit the factors of the model to optionally weighted examples.

		This method will fit the provided factor distributions to the given
		data and their optional weights. It will not update the marginal
		distributions, as that information is already encoded in the joint
		probabilities.

		..note:: A structure must already be provided. Currently, structure
		learning of factor graphs is not supported.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		self
		"""

		self.summarize(X, sample_weight=sample_weight)
		self.from_summaries()
		return self

	def summarize(self, X, sample_weight=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache for each distribution
		in the network. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		if self.frozen:
			return

		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		X = _check_parameter(X, "X", ndim=2, check_parameter=self.check_data)

		for i, factor in enumerate(self.factors):
			factor.summarize(X[:, self._factor_edges[i]], 
				sample_weight=sample_weight[:,i])

	def from_summaries(self):
		if self.frozen:
			return

		for distribution in self.factors:
			distribution.from_summaries()

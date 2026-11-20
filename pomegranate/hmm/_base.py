# _base.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import torch

from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _check_parameter
from .._utils import partition_sequences

from ..distributions._distribution import Distribution

from ..kmeans import KMeans


NEGINF = float("-inf")


def _check_inputs(model, X, emissions, priors):
	if X is None and emissions is None:
		raise ValueError("Must pass in one of `X` or `emissions`.")

	emissions = _check_parameter(_cast_as_tensor(emissions), "emissions",
		ndim=3)
	if emissions is None:
		emissions = model._emission_matrix(X, priors=priors)

	return emissions


class FilteringState:
	"""A resumable filtering state for streaming observations.

	This object stores everything needed to continue the forward filter for a
	batch of sequences when the observations arrive in blocks, without having
	to re-run the entire prefix each time a new block arrives. It is created by
	`DenseHMM.filtering_state` (or `SparseHMM.filtering_state`), extended by
	calls to `filter`, and finalized by `finish_filtering`. Both HMM
	implementations use the exact same contract.

	The stored quantities are:

		`alpha`: the *normalized* forward probability of the current
			observation under each state. Each row sums to one, i.e. this is
			the filtering distribution over states at the last observed time
			step. Because it is normalized at every step, it does not
			underflow on long sequences the way raw forward probabilities do.

		`log_likelihood`: the cumulative log likelihood of every observation
			block consumed so far, i.e. the sum of the log scaling terms
			(log of the pre-normalization partition function at every step).
			It equals `logsumexp` of the unnormalized forward vector at the
			last time step. The unnormalized ("normalization-preserving")
			forward log probabilities used by the regular `forward` method
			can be reconstructed as `log(alpha) + log_likelihood` and are
			available through the `log_forward` property.

	The end probabilities are deliberately *not* applied while filtering:
	they are multiplied in only once, when `finish_filtering` is called.
	Consequently, filtering a sequence block by block and then finishing it
	gives the same log probability as a single `log_probability` call on the
	whole sequence.

	A state is bound to the model that created it and cannot be shared between
	models or across different sequence identities. It records the model's
	parameter version, dtype/device, and state ordering. When the model
	parameters are updated (e.g. via `from_summaries` or `add_edge`) the
	version changes and the state is refused; the state must either be
	discarded and replaced with a fresh one or explicitly re-bound with
	`reset_filtering_state`. A state that has been finalized through
	`finish_filtering` is closed and cannot be extended or reset.


	Parameters
	----------
	model: _BaseHMM
		The model that owns this state.

	log_alpha: torch.Tensor, shape=(batch_size, n_distributions)
		The normalized forward log probabilities at the current time step,
		given as log values. Each row is a log probability distribution.

	log_likelihood: torch.Tensor, shape=(batch_size,)
		The cumulative log likelihood of all observations consumed so far.

	sequence_ids: tuple
		The identities of the sequences in the batch. Continuation calls must
		present the same identities in the same order, which prevents blocks
		from different sequences from being mixed into one state.
	"""

	def __init__(self, model, log_alpha, log_likelihood, sequence_ids):
		self.model_name = model.name
		self.k = model.n_distributions
		self.dtype = log_alpha.dtype
		self.device = log_alpha.device
		self.parameter_version = model._parameter_version
		# Store the actual distribution objects so that any reordering of the
		# model's states is detected rather than silently misinterpreted.
		self.state_order = tuple(model.distributions)

		self._log_alpha = log_alpha.detach()
		self._log_likelihood = log_likelihood.detach()
		self.sequence_ids = tuple(sequence_ids)
		self.closed = False

	def __len__(self):
		return len(self.sequence_ids)

	@property
	def batch_size(self):
		"""The number of sequences represented by this state."""

		return self._log_alpha.shape[0]

	@property
	def alpha(self):
		"""The normalized forward probabilities at the current time step.

		Returns a tensor of shape (batch_size, n_distributions) where every
		row sums to one.
		"""

		return torch.exp(self._log_alpha)

	@property
	def log_alpha(self):
		"""The log of the normalized forward probabilities."""

		return self._log_alpha

	@property
	def log_likelihood(self):
		"""The cumulative log likelihood of all consumed observations."""

		return self._log_likelihood

	@property
	def log_forward(self):
		"""The unnormalized forward log probabilities at the current step.

		These are the same values the regular `forward` method would return
		for the final time step of the consumed prefix: a log probability
		mass over states that, in general, does not sum to one.
		"""

		return self._log_alpha + self._log_likelihood.unsqueeze(1)

	def __repr__(self):
		return ("{}({} sequences, {} states, log_likelihood={}, closed={})"
			.format(self.model_name, self.batch_size, self.k,
				self._log_likelihood.tolist(), self.closed))


class Silent(Distribution):
	def __init__(self):
		super().__init__(inertia=0.0, frozen=False, check_data=True)


class _BaseHMM(Distribution):
	"""A base hidden Markov model.

	A hidden Markov model is an extension of a mixture model to sequences by
	including a transition matrix between the elements of the mixture. Each of
	the algorithms for a hidden Markov model are essentially just a revision
	of those algorithms to incorporate this transition matrix.

	There are two main ways one can implement a hidden Markov model: either the
	transition matrix can be implemented in a dense, or a sparse, manner. If the
	transition matrix is dense, implementing it in a dense manner allows for
	the primary computation to use matrix multiplications which can be very
	fast. However, if the matrix is sparse, these matrix multiplications will
	be fairly slow and end up significantly slower than the sparse version of
	a matrix multiplication.

	This object is a wrapper for both implementations, which can be specified
	using the `kind` parameter. Choosing the right implementation will not
	effect the accuracy of the results but will change the speed at which they
	are calculated. 	

	Separately, there are two ways to instantiate the hidden Markov model. The
	first is by passing in a set of distributions, a dense transition matrix, 
	and optionally start/end probabilities. The second is to initialize the
	object without these and then to add edges using the `add_edge` method
	and to add nodes using the `add_nodes` method. Importantly, the way that
	you choose to initialize the hidden Markov model is independent of the
	implementation that you end up choosing. If you pass in a dense transition
	matrix, this will be converted to a sparse matrix with all the zeros
	dropped if you choose `kind='sparse'`.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	edges: numpy.ndarray, torch.Tensor, or None. shape=(k,k), optional
		A dense transition matrix of probabilities for how each node or
		distribution passed in connects to each other one. This can contain
		many zeroes, and when paired with `kind='sparse'`, will drop those
		elements from the matrix. Default is None.

	starts: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of starting at each node. If not provided, assumes
		these probabilities are uniform. Default is None.

	ends: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of ending at each node. If not provided, assumes
		these probabilities are uniform. Default is None.

	kind: str, 'sparse' or 'dense', optional
		The underlying implementation of the transition matrix to use.
		Default is 'sparse'. 

	init: str, optional
		The initialization to use for the k-means initialization approach.
		Default is 'first-k'. Must be one of:

			'first-k': Use the first k examples from the data set
			'random': Use a random set of k examples from the data set
			'submodular-facility-location': Use a facility location submodular
				objective to initialize the k-means algorithm
			'submodular-feature-based': Use a feature-based submodular objective
				to initialize the k-means algorithm.

	max_iter: int, optional
		The number of iterations to do in the EM step, which for HMMs is
		sometimes called Baum-Welch. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 0.1.

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

	random_state: int or None, optional
		The random state to make randomness deterministic. If None, not
		deterministic. Default is None.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling.

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

	def __init__(self, distributions=None, starts=None, ends=None, 
		init='random', max_iter=1000, tol=0.1, sample_length=None, 
		return_sample_paths=False, inertia=0.0, frozen=False, check_data=True,
		random_state=None, verbose=False):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)

		self.distributions = distributions
		n = len(distributions) if distributions is not None else None

		self.start = Silent()
		self.end = Silent()

		self.edges = None
		self.starts = None
		self.ends = None

		if starts is not None:
			starts = _check_parameter(_cast_as_tensor(starts), "starts", ndim=1, 
				shape=(n,), min_value=0., max_value=1., value_sum=1.0)
			self.starts = _cast_as_parameter(torch.log(starts))

		if ends is not None:
			ends = _check_parameter(_cast_as_tensor(ends), "ends", ndim=1, 
				shape=(n,), min_value=0., max_value=1.)
			self.ends = _cast_as_parameter(torch.log(ends))

		if not isinstance(random_state, numpy.random.RandomState):
			self.random_state = numpy.random.RandomState(random_state)
		else:
			self.random_state = random_state

		self.init = init
		self.max_iter = _check_parameter(max_iter, "max_iter", min_value=1, 
			ndim=0, dtypes=(int, torch.int32, torch.int64))
		self.tol = _check_parameter(tol, "tol", min_value=0., ndim=0)

		self.sample_length = sample_length
		self.return_sample_paths = return_sample_paths

		self.verbose = verbose
		self.d = self.distributions[0].d if distributions is not None else None

		# Monotonic counter bumped whenever the model parameters change.
		# Filtering states bind to the version they were created with and are
		# refused once the model has been updated underneath them.
		self._parameter_version = 0

	def _bump_parameter_version(self):
		"""Invalidate all filtering states created before this point."""

		self._parameter_version += 1

	def _initialize(self, X=None, sample_weight=None):
		"""Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d), optional
			The data to use to initialize the model. Default is None.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, len) or a vector of shape (-1,). If None, defaults to ones.
			Default is None.
		"""

		n = self.n_distributions
		if self.starts is None:
			self.starts = _cast_as_parameter(torch.log(torch.ones(n, 
				dtype=self.dtype, device=self.device) / n))

		if self.ends is None:
			self.ends = _cast_as_parameter(torch.log(torch.ones(n,
				dtype=self.dtype, device=self.device) / n))

		_init = all(d._initialized for d in self.distributions)
		if X is not None and not _init:
			if isinstance(X, list):
				d = _cast_as_tensor(X[0]).shape[-1]
				X = torch.cat([_cast_as_tensor(x).reshape(-1, d) for x in X],
					dim=0).unsqueeze(0)

			X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
				check_parameter=self.check_data)
			X = X.reshape(-1, X.shape[-1])

			if sample_weight is None:
				sample_weight = torch.ones(1, dtype=self.dtype, 
					device=self.device).to(X.device).expand(X.shape[0], 1)
			else:
				if isinstance(sample_weight, list):
					sample_weight = torch.cat([_cast_as_tensor(w).reshape(-1, 1) 
						for w in sample_weight], dim=0).to(X.device)

				sample_weight = _check_parameter(_cast_as_tensor(
					sample_weight).reshape(-1, 1), "sample_weight", 
					min_value=0., ndim=1, shape=(len(X),), 
					check_parameter=self.check_data).reshape(-1, 1)

			model = KMeans(self.n_distributions, init=self.init, max_iter=1, 
				random_state=self.random_state).to(self.device)

			y_hat = model.fit_predict(X, sample_weight=sample_weight)

			for i in range(self.n_distributions):
				self.distributions[i].fit(X[y_hat == i].cpu(), 
					sample_weight=sample_weight[y_hat == i].cpu())
				self.distributions[i].to(X.device)

			self.d = X.shape[-1]
			super()._initialize(X.shape[-1])

		self._initialized = True
		self._reset_cache()

	def _emission_matrix(self, X, priors=None):
		"""Return the emission/responsibility matrix.

		This method returns the log probability of each example under each
		distribution contained in the model with the log prior probability
		of each component added.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to evaluate. 

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observatons
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Default is None.

	
		Returns
		-------
		e: torch.Tensor, shape=(-1, len, self.k)
			A set of log probabilities for each example under each distribution.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
			shape=(-1, -1, self.d), check_parameter=self.check_data)

		n, k, _ = X.shape
		X = X.reshape(n*k, self.d)

		priors = _check_parameter(_cast_as_tensor(priors), "priors",
			ndim=3, shape=(n, k, self.k), min_value=0.0, max_value=1.0, 
			value_sum=1.0, value_sum_dim=-1, check_parameter=self.check_data)

		if not self._initialized:
			self._initialize()

		e = torch.empty((k, self.k, n), dtype=self.dtype, requires_grad=False, 
			device=self.device)
		
		for i, node in enumerate(self.distributions):
			logp = node.log_probability(X)
			if isinstance(logp, torch.masked.MaskedTensor):
				logp = logp._masked_data

			e[:, i] = logp.reshape(n, k).T

		e = e.permute(2, 0, 1)
		if priors is not None:
			e += torch.log(priors)

		return e

	@property
	def k(self):
		return len(self.distributions) if self.distributions is not None else 0

	@property
	def n_distributions(self):
		return len(self.distributions) if self.distributions is not None else 0

	def freeze(self):
		"""Freeze this model and all child distributions."""

		self.register_buffer("frozen", _cast_as_tensor(True))
		for d in self.distributions:
			d.freeze()

		return self

	def unfreeze(self):
		"""Unfreeze this model and all child distributions."""

		self.register_buffer("frozen", _cast_as_tensor(False))
		for d in self.distributions:
			d.unfreeze()

		return self

	def add_distribution(self, distribution):
		"""Add a distribution to the model.


		Parameters
		----------
		distribution: torchegranate.distributions.Distribution
			A distribution object.
		"""

		if self.distributions is None:
			self.distributions = []

		if not isinstance(distribution, Distribution):
			raise ValueError("distribution must be a distribution object.")

		self.distributions.append(distribution)
		self.d = distribution.d

	def add_distributions(self, distributions):
		"""Add a set of distributions to the model.

		This method will iterative call the `add_distribution`.


		Parameters
		----------
		distributions: list, tuple, iterable
			A set of distributions to add to the model.
		"""

		for distribution in distributions:
			self.add_distribution(distribution)

	def add_edge(self, start, end, probability):
		"""Add an edge to the model.

		This method takes in two distribution objects and the probability
		connecting the two and adds an edge to the model.


		Parameters
		----------
		start: torchegranate.distributions.Distribution
			The parent node for the edge

		end: torchegranate.distributions.Distribution
			The child node for the edge

		probability: float, (0, 1]
			The probability of connecting the two.
		"""

		if not isinstance(start, Distribution):
			raise ValueError("start must be a distribution.")

		if not isinstance(end, Distribution):
			raise ValueError("end must be a distribution.")

		if not isinstance(probability, float):
			raise ValueError("probability must be a float.")

		if self.edges is None:
			self.edges = []

		self.edges.append((start, end, probability))

	def probability(self, X, priors=None):
		"""Calculate the probability of each example.

		This method calculates the probability of each example given the
		parameters of the distribution. The examples must be given in a 3D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		prob: torch.Tensor, shape=(-1,)
			The probability of each example.
		"""

		return torch.exp(self.log_probability(X, priors=priors))

	def log_probability(self, X, priors=None):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 3D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		f = self.forward(X, priors=priors)
		return torch.logsumexp(f[:, -1] + self.ends, dim=1)

	def predict_log_proba(self, X, priors=None):
		"""Calculate the posterior probabilities for each example.

		This method calculates the log posterior probabilities for each example
		and then normalizes across each component of the model. These
		probabilities are calculated using the forward-backward algorithm.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n_distributions)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		r: torch.Tensor, shape=(-1, len, self.n_distributions)
			The log posterior probabilities for each example under each 
			component as calculated by the forward-backward algorithm.
		"""

		_, r, _, _, _ = self.forward_backward(X, priors=priors)
		return r

	def predict_proba(self, X, priors=None):
		"""Calculate the posterior probabilities for each example.

		This method calculates the posterior probabilities for each example
		and then normalizes across each component of the model. These
		probabilities are calculated using the forward-backward algorithm.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n_distributions)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		y: torch.Tensor, shape=(-1, len, self.n_distributions)
			The posterior probabilities for each example under each component
			as calculated by the forward-backward algorithm.
		"""

		return torch.exp(self.predict_log_proba(X, priors=priors))

	def predict(self, X, priors=None):
		"""Predicts the component for each observation.

		This method calculates the predicted component for each observation
		given the posterior probabilities as calculated by the forward-backward
		algorithm. Essentially, it is just the argmax over components.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		y: torch.Tensor, shape=(-1, len, self.k)
			The posterior probabilities for each example under each component
			as calculated by the forward-backward algorithm.
		"""

		return torch.argmax(self.predict_log_proba(X, priors=priors), dim=-1)

	def fit(self, X, sample_weight=None, priors=None):
		"""Fit the model to sequences with optional weights and priors.

		This method implements the core of the learning process. For hidden
		Markov models, this is a form of EM called "Baum-Welch" or "structured
		EM". This iterative algorithm will proceed until converging, either
		according to the threshold set by `tol` or until the maximum number
		of iterations set by `max_iter` has been hit.

		This method is largely a wrapper around the `summarize` and
		`from_summaries` methods. It's primary contribution is serving as a
		loop around these functions and to monitor convergence.

		Unlike other HMM methods, this method can handle variable length
		sequences by accepting a list of tensors where each tensor has a
		different sequence length. Then, summarization is done on each tensor
		sequentially. This will provide an exact update as if the entire data
		set was seen at the same time but will allow batched operations to be
		performed on each variable length tensor.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to evaluate. Because sequences can be variable
			length, there are three ways to format the sequences.
			
				1. Pass in a tensor of shape (n, length, dim), which can only 
				be done when each sequence is the same length. 

				2. Pass in a list of 3D tensors where each tensor has the shape 
				(n, length, dim). In this case, each tensor is a collection of 
				sequences of the same length and so sequences of different 
				lengths can be trained on. 

				3. Pass in a list of 2D tensors where each tensor has the shape
				(length, dim). In this case, sequences of the same length will
				be grouped together into the same tensor and fitting will
				proceed as if you had passed in data like way 2.

		sample_weight: list, numpy.ndarray, torch.Tensor or None, optional
			A set of weights for the examples. These must follow the same format
			as X.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities). This can be used to assign labels to observations
			by setting one of the probabilities for an observation to 1.0.
			Note that this can be used to assign hard labels, but does not
			have the same semantics for soft labels, in that it only
			influences the initial estimate of an observation being generated
			by a component, not gives a target. Must be formatted in the same
			shape as X. Default is None.


		Returns
		-------
		self
		"""

		X, sample_weight, priors = partition_sequences(X, 
			sample_weight=sample_weight, priors=priors, n_dists=self.k)

		# Initialize by concatenating across sequences
		if not self._initialized:
			if sample_weight is None:
				self._initialize(X)
			else:
				self._initialize(X, sample_weight=sample_weight)

		logp, last_logp = None, None
		for i in range(self.max_iter):
			start_time = time.time()

			# Train loop across all tensors
			logp = 0
			for j, X_ in enumerate(X):
				w_ = None if sample_weight is None else sample_weight[j]
				p_ = None if priors is None else priors[j]

				logp += self.summarize(X_, sample_weight=w_, priors=p_).sum()

			# Calculate and check improvement and optionally print it
			if i > 0:
				improvement = logp - last_logp
				duration = time.time() - start_time

				if self.verbose:
					print("[{}] Improvement: {}, Time: {:4.4}s".format(i, 
						improvement, duration))

				if improvement < self.tol:
					self._reset_cache()
					return self

			last_logp = logp
			self.from_summaries()

		# Calculate for the last iteration
		if self.verbose:
			logp = 0
			for j, X_ in enumerate(X):
				w_ = None if sample_weight is None else sample_weight[j]
				p_ = None if priors is None else priors[j]
				
				logp += self.summarize(X_, sample_weight=w_, priors=p_).sum()

			improvement = logp - last_logp
			duration = time.time() - start_time

			print("[{}] Improvement: {}, Time: {:4.4}s".format(i+1, 
				improvement, duration))

		self._reset_cache()
		return self

	def summarize(self, X, sample_weight=None, emissions=None, 
		priors=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, length, self.d) or a vector of shape (-1,). Default is ones.

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n_distributions)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
			shape=(-1, -1, self.d), check_parameter=self.check_data)
		emissions = _check_inputs(self, X, emissions, priors)

		if sample_weight is None:
			sample_weight = torch.ones(1, device=self.device).expand(
				emissions.shape[0], 1)
		else:
			sample_weight = _check_parameter(_cast_as_tensor(sample_weight),
				"sample_weight", min_value=0., ndim=1, 
				shape=(emissions.shape[0],), 
				check_parameter=self.check_data).reshape(-1, 1)

		if not self._initialized:
			self._initialize(X, sample_weight=sample_weight)

		return X, emissions, sample_weight

	def _check_filtering_state(self, state):
		"""Validate that a filtering state can be used with this model."""

		if not isinstance(state, FilteringState):
			raise ValueError("`state` must be a FilteringState returned by "
				"`{}.filtering_state`.".format(self.name))

		if state.model_name != self.name:
			raise ValueError("This {} cannot consume a state created by a "
				"{}.".format(self.name, state.model_name))

		if state.k != self.n_distributions:
			raise ValueError("The state was created for a model with {} "
				"distributions but this model has {}.".format(state.k,
					self.n_distributions))

		if state.parameter_version != self._parameter_version:
			raise ValueError("The model parameters have changed since this "
				"state was created. Create a new filtering state or call "
				"`reset_filtering_state` to explicitly rebind it.")

		if state.dtype != self.dtype or state.device != self.device:
			raise ValueError("The state was created with dtype {} on device "
				"{} but the model now uses dtype {} on device {}. Create a "
				"new filtering state or call `reset_filtering_state` to "
				"explicitly rebind it.".format(state.dtype, state.device,
					self.dtype, self.device))

		if state.state_order != tuple(self.distributions):
			raise ValueError("The ordering of the model's distributions has "
				"changed since this state was created. Create a new "
				"filtering state or call `reset_filtering_state` to "
				"explicitly rebind it.")

		if state.closed:
			raise ValueError("This filtering state has already been closed "
				"by `finish_filtering`; start a new sequence with "
				"`filtering_state`.")

	def filtering_state(self, X=None, emissions=None, priors=None,
		sequence_ids=None):
		"""Create a resumable filtering state for the first observation block.

		This begins one or more new sequences. The start probabilities are
		applied exactly once here; subsequent blocks must be passed to
		`filter` together with the returned state, which continues the filter
		without re-applying the starts and without re-processing any previous
		observations. End probabilities are not applied until
		`finish_filtering` is called.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			The first block of observations. Does not need to be passed in if
			`emissions` are.

		emissions: torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities for the first block.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities for assigning each symbol to each node.
			Default is None.

		sequence_ids: hashable, list, tuple or None, optional
			The identities of the sequences in the batch. These are stored in
			the state and must match, in order, on every continuation call so
			that blocks belonging to different sequences can never be mixed
			into the same state. Default is range(batch_size).


		Returns
		-------
		state: FilteringState
			A resumable state holding the normalized forward distribution and
			the cumulative log likelihood of this first block.
		"""

		emissions = _check_inputs(self, X, emissions, priors)
		if not self._initialized:
			self._initialize()

		n = emissions.shape[0]
		if sequence_ids is None:
			sequence_ids = tuple(range(n))
		elif not isinstance(sequence_ids, (list, tuple)):
			sequence_ids = (sequence_ids,)
		else:
			sequence_ids = tuple(sequence_ids)

		if len(sequence_ids) != n:
			raise ValueError("Expected {} sequence ids for a batch of {} "
				"sequences but got {}.".format(n, n, len(sequence_ids)))

		if len(set(sequence_ids)) != len(sequence_ids):
			raise ValueError("Sequence ids must be unique within a batch.")

		log_alpha = self.starts.unsqueeze(0) + emissions[:, 0]
		log_scale = torch.logsumexp(log_alpha, dim=1, keepdims=True)
		log_alpha = log_alpha - log_scale

		log_likelihood = log_scale[:, 0]
		if emissions.shape[1] > 1:
			log_alpha, block_scale = self._scaled_forward_block(log_alpha,
				emissions[:, 1:])
			log_likelihood = log_likelihood + block_scale

		state = FilteringState(self, log_alpha, log_likelihood, sequence_ids)
		return state

	def filter(self, state, X=None, emissions=None, priors=None,
		sequence_ids=None):
		"""Continue filtering from an existing state with a new block.

		The observations in the block extend the sequences already represented
		by `state`. Neither start probabilities nor end probabilities are
		applied: the filter simply continues from the state's current
		normalized forward distribution. This is mathematically identical to
		running the forward algorithm over the concatenation of all blocks.


		Parameters
		----------
		state: FilteringState
			A state returned by `filtering_state` or a previous `filter` call.

		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			The next block of observations. Does not need to be passed in if
			`emissions` are.

		emissions: torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities for the block.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
			Prior probabilities for assigning each symbol to each node.
			Default is None.

		sequence_ids: hashable, list, tuple or None, optional
			Must match, in order, the identities stored in the state. This
			guarantees that a block cannot be pushed into a state belonging
			to a different set of sequences. Default is None, which skips
			the explicit check but still verifies the batch size.


		Returns
		-------
		state: FilteringState
			The same state, updated in place, holding the normalized forward
			distribution and cumulative log likelihood after the new block.
		"""

		self._check_filtering_state(state)
		emissions = _check_inputs(self, X, emissions, priors)

		n = emissions.shape[0]
		if n != state.batch_size:
			raise ValueError("The state represents {} sequence(s) but the "
				"new block contains {}.".format(state.batch_size, n))

		if sequence_ids is not None:
			if not isinstance(sequence_ids, (list, tuple)):
				sequence_ids = (sequence_ids,)
			else:
				sequence_ids = tuple(sequence_ids)

			if sequence_ids != state.sequence_ids:
				raise ValueError("The sequence ids of the new block {} do "
					"not match the state's sequence ids {}. Blocks from "
					"different sequences cannot share a filtering state."
					.format(sequence_ids, state.sequence_ids))

		log_alpha, block_scale = self._scaled_forward_block(
			state._log_alpha, emissions)
		state._log_alpha = log_alpha
		state._log_likelihood = state._log_likelihood + block_scale
		return state

	def finish_filtering(self, state):
		"""Apply the end probabilities and return the sequence log likelihood.

		This is the only place end probabilities enter the streaming
		calculation. After this call the state is closed and cannot be
		extended; the returned value equals a single `log_probability` call on
		the full concatenated sequence.


		Parameters
		----------
		state: FilteringState
			A state returned by `filtering_state` and optionally extended with
			`filter`.


		Returns
		-------
		logp: torch.Tensor, shape=(batch_size,)
			The log probability of each full sequence, including the end
			probabilities.
		"""

		self._check_filtering_state(state)

		logp = state._log_likelihood + torch.logsumexp(
			state._log_alpha + self.ends, dim=1)
		state.closed = True
		return logp

	def reset_filtering_state(self, state):
		"""Explicitly rebind a state after the model parameters changed.

		Filtering states are normally invalidated when the model parameters
		are updated. Calling this method acknowledges that change and re-binds
		the state to the current parameter version, dtype/device, and state
		ordering without altering the stored filtering distribution or
		cumulative log likelihood. A closed state cannot be reset.


		Parameters
		----------
		state: FilteringState
			The state to rebind to this model.


		Returns
		-------
		state: FilteringState
			The same state, updated in place.
		"""

		if not isinstance(state, FilteringState):
			raise ValueError("`state` must be a FilteringState.")

		if state.model_name != self.name:
			raise ValueError("This {} cannot rebind a state created by a "
				"{}.".format(self.name, state.model_name))

		if state.closed:
			raise ValueError("A closed filtering state cannot be reset; "
				"start a new sequence with `filtering_state`.")

		if state.k != self.n_distributions:
			raise ValueError("Cannot rebind a state with {} states to a "
				"model with {} distributions.".format(state.k,
					self.n_distributions))

		state._log_alpha = state._log_alpha.to(dtype=self.dtype,
			device=self.device)
		state._log_likelihood = state._log_likelihood.to(dtype=self.dtype,
			device=self.device)
		state.dtype = self.dtype
		state.device = self.device
		state.parameter_version = self._parameter_version
		state.state_order = tuple(self.distributions)
		return state

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		self.from_summaries()
		self._reset_cache()



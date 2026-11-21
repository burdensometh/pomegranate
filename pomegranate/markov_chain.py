# markov_chain.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

from ._utils import _cast_as_tensor
from ._utils import _check_parameter

from .distributions._distribution import Distribution
from .distributions import Categorical
from .distributions import ConditionalCategorical


def _as_scalar(value):
	"""Convert a scalar-like value into a hashable Python object."""

	if isinstance(value, torch.Tensor):
		value = value.detach().cpu()
		if value.numel() != 1:
			raise ValueError("Sequence ids must be scalars.")
		return value.item()

	if isinstance(value, numpy.ndarray):
		if value.size != 1:
			raise ValueError("Sequence ids must be scalars.")
		return value.reshape(-1)[0].item()

	if isinstance(value, numpy.generic):
		return value.item()

	return value


def _as_scalar_tuple(values):
	"""Convert a list/tuple/array/tensor of scalars into a tuple."""

	if isinstance(values, (torch.Tensor, numpy.ndarray)):
		values = values.reshape(-1).tolist()

	return tuple(_as_scalar(value) for value in values)


def _as_list(values):
	"""Convert an array-like of scalars into a Python list."""

	if isinstance(values, torch.Tensor):
		return values.detach().cpu().reshape(-1).tolist()

	if isinstance(values, numpy.ndarray):
		return values.reshape(-1).tolist()

	return list(values)


class MarkovChain(Distribution):
	"""A Markov chain.

	A Markov chain is the simplest sequential model which factorizes the
	joint probability distribution P(X_{0} ... X_{t}) along a chain into the
	product of a marginal distribution P(X_{0}) P(X_{1} | X_{0}) ... with
	k conditional probability distributions for a k-th order Markov chain.

	Despite sometimes being thought of as an independent model, Markov chains
	are probability distributions over sequences just like hidden Markov
	models. Because a Markov chain has the same theoretical properties as a
	probability distribution, it can be used in any situation that a simpler
	distribution could, such as an emission distribution for a HMM or a
	component of a Bayes classifier.

	Sequences can also be summarized in chunks. When a chunk is marked with
	a sequence id and the model has already seen part of that sequence, the
	last `k` observations are used as context so that transitions spanning
	the boundary between chunks are counted exactly once. See the `summarize`
	method for more details.


	Parameters
	----------
	distributions: tuple or list or None
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Categorical()".

	k: int or None
		The number of conditional distributions to include in the chain, also
		the number of steps back to model in the sequence. This must be passed
		in if the distributions are not passed in.

	n_categories: list, tuple, or None
		A list or tuple containing the number of categories that each feature
		has.

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
	"""

	def __init__(self, distributions=None, k=None, n_categories=None,
		inertia=0.0, frozen=False, check_data=True):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "MarkovChain"

		distributions = _check_parameter(distributions, "distributions",
			dtypes=(list, tuple, torch.nn.ModuleList))

		if distributions is not None:
			self.distributions = torch.nn.ModuleList(distributions)
		else:
			self.distributions = None

		self.k = _check_parameter(_cast_as_tensor(k, dtype=torch.int32), "k",
			ndim=0)
		self.n_categories = _check_parameter(n_categories, "n_categories",
			dtypes=(list, tuple))

		if distributions is None and k is None:
			raise ValueError("Must provide one of 'distributions', or 'k'.")

		if distributions is not None:
			self.k = len(distributions) - 1

		self.d = None

		# The context of each sequence that is being summarized in chunks.
		# Each value stores the last `k` observations, the number of
		# observations seen so far, and how many of the shorter windows that
		# begin a sequence have already been extracted.
		self._sequence_context = {}
		self._context_schema = None

		self._initialized = distributions is not None and distributions[0]._initialized
		self._reset_cache()

	def _initialize(self, d, n_categories):
		"""Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.

		n_categories: int
			The maximum number of categories to model. This single number is
			used as the maximum across all features and all timesteps.
		"""

		if self.distributions is None:
			distributions = [Categorical()]
			distributions[0]._initialize(d, max(n_categories))

			for i in range(self.k):
				distribution = ConditionalCategorical()
				distribution._initialize(d, [[n_categories[j]]*(i+2)
					for j in range(d)])

				distributions.append(distribution)

			self.distributions = torch.nn.ModuleList(distributions)

		self.n_categories = n_categories
		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		if self._initialized and self.distributions is not None:
			for distribution in self.distributions:
				distribution._reset_cache()

	def sample(self, n):
		"""Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution. For a mixture model, this involves first
		sampling the component using the prior probabilities, and then sampling
		from the chosen distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.


		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""

		X = [self.distributions[0].sample(n)]

		for distribution in self.distributions[1:]:
			X_ = torch.stack(X).permute(1, 0, 2)
			samples = distribution.sample(n, X_[:, -self.k-1:])
			X.append(samples)

		return torch.stack(X).permute(1, 0, 2)

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 3D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of examples to evaluate.

		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""


		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3,
			check_parameter=self.check_data)
		self.d = X.shape[1]

		logps = self.distributions[0].log_probability(X[:, 0])
		for i, distribution in enumerate(self.distributions[1:-1]):
			logps += distribution.log_probability(X[:, :i+2])

		for i in range(X.shape[1] - self.k):
			j = i + self.k + 1
			logps += self.distributions[-1].log_probability(X[:, i:j])

		return logps

	def fit(self, X, sample_weight=None, sequence_ids=None, begin=None,
		end=None, state=None):
		"""Fit the model to optionally weighted sequences.

		This method will fit the provided distributions given the data and
		their weights. If only `k` has been provided, the relevant set of
		distributions will be initialized.

		The `sequence_ids`, `begin`, `end`, and `state` parameters describe
		how chunks relate to the sequences they are pieces of and are passed
		through to `summarize`. See that method for details.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of examples to evaluate.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.

		sequence_ids: list, tuple, numpy.ndarray, torch.Tensor, or None, optional
			An identifier for the sequence that each chunk belongs to.
			Default is None.

		begin: bool or list, tuple, numpy.ndarray, torch.Tensor, optional
			Whether each chunk starts a new sequence. Default is None.

		end: bool or list, tuple, numpy.ndarray, torch.Tensor, optional
			Whether each chunk ends its sequence. Default is None.

		state: str or list of str, optional
			An explicit per-chunk marker, one of 'begin', 'continue', or
			'end'. Default is None.


		Returns
		-------
		self
		"""

		self.summarize(X, sample_weight=sample_weight, sequence_ids=sequence_ids,
			begin=begin, end=end, state=state)
		self.from_summaries()
		return self

	def summarize(self, X, sample_weight=None, sequence_ids=None, begin=None,
		end=None, state=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache for each distribution
		in the network. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.

		Sequences can be summarized in chunks. Each chunk is associated with
		the sequence it is a piece of using `sequence_ids`, and whether it
		starts or ends that sequence using `begin`/`end` (or, equivalently,
		the explicit `state`). The last `k` observations of a sequence are
		retained as context so that transitions spanning a chunk boundary are
		counted exactly once, as they would be if the whole sequence had been
		passed in at once. Summing over any chunking of the data therefore
		yields the same sufficient statistics, and thus the same parameters
		after `from_summaries`, as a single call on the full sequences.

		When several sequences are interleaved in one call, each sequence
		keeps its own context and windows are never drawn across sequence
		boundaries. Empty chunks contribute nothing and never create
		transitions. A chunk that is not marked as a begin continues the
		sequence with the same id if one is stored and, if not, starts a new
		sequence.

		If none of `sequence_ids`, `begin`, `end`, or `state` are provided,
		the legacy behavior is used: `X` must be a 3D tensor where each row is
		an independent sequence, and no context is stored between calls.

		Weights belong to the true observations. A window of any order is
		weighted by the weight of the observation it predicts, i.e. its last
		element, so observations carried over as context are never counted
		twice.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of examples to summarize. This is a 3D tensor whose first
			dimension indexes independent chunks, a 2D chunk, or a list of 2D
			chunks that may have different lengths, when chunked input is used.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. Weights can be given per chunk,
			per observation, or per feature. Default is ones.

		sequence_ids: list, tuple, numpy.ndarray, torch.Tensor, or None, optional
			An identifier for the sequence that each chunk belongs to. Chunks
			with the same identifier are treated as consecutive pieces of one
			sequence. A single id may be given for all chunks. Default is None,
			which treats each chunk as its own independent sequence.

		begin: bool or list, tuple, numpy.ndarray, torch.Tensor, optional
			Whether each chunk starts a new sequence. Starting a sequence
			discards any stored context for that id before the chunk is
			summarized. Cannot be combined with `state`. Default is None.

		end: bool or list, tuple, numpy.ndarray, torch.Tensor, optional
			Whether each chunk ends its sequence. Ending a sequence discards
			its stored context after the chunk is summarized. Default is None.

		state: str or list of str, optional
			An explicit per-chunk marker, one of 'begin', 'continue', or
			'end'. Cannot be combined with `begin` or `end`. Default is None.


		Returns
		-------
		self
		"""

		if self.frozen:
			return self

		self._ensure_context_state()

		if state is not None and (begin is not None or end is not None):
			raise ValueError("'state' cannot be combined with 'begin' or "
				"'end'.")

		chunked = any(value is not None for value in
			(sequence_ids, begin, end, state))

		chunks = self._split_into_chunks(X, chunked=chunked)

		if len(chunks) == 0:
			return self

		if not chunked:
			# Legacy behavior: each row is an independent sequence and no
			# context is kept between rows or between calls. Unique sentinels
			# guarantee that stored contexts of real sequences are untouched.
			sequence_ids = [object() for _ in range(len(chunks))]
			begin = [True] * len(chunks)
			end = [True] * len(chunks)
		elif state is not None:
			begin, end = self._states_to_flags(state, len(chunks))
		else:
			begin = self._normalize_flags(begin, len(chunks), "begin")
			end = self._normalize_flags(end, len(chunks), "end")

		self._summarize_stream(chunks, sample_weight, sequence_ids, begin,
			end)

		return self

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Any sequence context is preserved so that later chunks of the same
		sequences can still be summarized. Use `clear_context` to discard it.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		if self.frozen:
			return

		for distribution in self.distributions:
			distribution.from_summaries()

	def _ensure_context_state(self):
		"""Make sure the context attributes exist on this instance.

		Objects unpickled from versions that predate chunked summarization do
		not have these attributes, so they are created lazily.
		"""

		if not hasattr(self, "_sequence_context"):
			self._sequence_context = {}

		if not hasattr(self, "_context_schema"):
			self._context_schema = None

	def clear_context(self, sequence_ids=None):
		"""Clear the stored context for one or more sequences.

		Contexts are stored automatically when sequences are summarized in
		chunks. This method discards them, for example when a sequence was
		abandoned before it was marked as ended, or when the schema of the
		model has been changed on purpose.


		Parameters
		----------
		sequence_ids: object or list, tuple, or None, optional
			A single sequence id or a collection of sequence ids to discard.
			If None, discard the context of all sequences. Default is None.


		Returns
		-------
		self
		"""

		self._ensure_context_state()

		if sequence_ids is None:
			self._sequence_context = {}
		else:
			if isinstance(sequence_ids, (str, bytes)):
				sequence_ids = [sequence_ids]
			else:
				try:
					iter(sequence_ids)
				except TypeError:
					sequence_ids = [sequence_ids]

			for sequence_id in sequence_ids:
				self._sequence_context.pop(_as_scalar(sequence_id), None)

		if len(self._sequence_context) == 0:
			self._context_schema = None

		return self

	def _split_into_chunks(self, X, chunked=False):
		"""Split the input into a list of 2D chunks."""

		if isinstance(X, torch.masked.MaskedTensor):
			X = X._masked_data

		if isinstance(X, (torch.Tensor, numpy.ndarray)):
			X = _cast_as_tensor(X)

			if X.ndim == 2 and chunked:
				return [X]
			elif X.ndim == 3:
				return [X[i] for i in range(X.shape[0])]

			raise ValueError("X must be a 3D array-like object with shape "
				"(-1, length, d) unless chunked input is used.")

		if isinstance(X, (list, tuple)):
			if not chunked:
				try:
					X_ = _cast_as_tensor(X)
				except Exception:
					X_ = None

				if not isinstance(X_, torch.Tensor) or X_.ndim != 3:
					raise ValueError("X must be a 3D array-like object with "
						"shape (-1, length, d) unless chunked input is used.")

				return [X_[i] for i in range(X_.shape[0])]

			try:
				X_ = _cast_as_tensor(X)
			except Exception:
				X_ = None

			if isinstance(X_, torch.Tensor):
				if X_.ndim == 2:
					return [X_]
				if X_.ndim == 3:
					return [X_[i] for i in range(X_.shape[0])]

			chunks = []
			for x in X:
				if isinstance(x, torch.masked.MaskedTensor):
					x = x._masked_data
				x = _cast_as_tensor(x)

				if not isinstance(x, torch.Tensor):
					raise ValueError("Each chunk must be array-like.")

				if x.ndim == 2:
					chunks.append(x)
				elif x.ndim == 3:
					chunks.extend(x[i] for i in range(x.shape[0]))
				else:
					raise ValueError("Each chunk must be a 2D array-like "
						"object.")

			return chunks

		raise ValueError("X must be a 3D array-like object with shape "
			"(-1, length, d) unless chunked input is used.")

	def _feature_dim(self):
		"""Return the number of features, if it is known."""

		if self.distributions is not None:
			for distribution in self.distributions:
				d = getattr(distribution, "d", None)
				if d is not None:
					return int(d)

		if self._initialized and self.d is not None:
			return int(self.d)

		return None

	def _category_limits(self):
		"""Return the number of categories allowed for each feature."""

		if not self._initialized or self.distributions is None:
			return None

		limits = None
		for distribution in self.distributions:
			probs = distribution.probs

			if probs is None:
				return None

			if isinstance(probs, torch.nn.ParameterList):
				values = tuple(int(prob.shape[-1]) for prob in probs)
			else:
				values = tuple([int(probs.shape[-1])] * int(probs.shape[-2]))

			if limits is None:
				limits = list(values)
			else:
				if len(values) != len(limits):
					return None

				limits = [max(a, b) for a, b in zip(limits, values)]

		return tuple(limits)

	def _stream_schema(self):
		"""Return the order and category schema of the model."""

		k = int(self.k)

		if not self._initialized:
			return (k, None, None, None)

		if self.n_categories is not None:
			n_categories = _as_scalar_tuple(self.n_categories)
		else:
			n_categories = None

		d = self._feature_dim()
		if d is None:
			d = self.d

		return (k, d, n_categories, self._category_limits())

	def _check_context_schema(self):
		"""Ensure that stored contexts are compatible with the model."""

		if not self._sequence_context:
			self._context_schema = None
			return

		schema = self._stream_schema()

		if self._context_schema is None:
			self._context_schema = schema
		elif schema != self._context_schema:
			raise ValueError("The order or category schema of the model has "
				"changed since the sequence context was stored (stored: {}, "
				"current: {}). Call clear_context() to discard the stored "
				"context before continuing with the new schema.".format(
					self._context_schema, schema))

	def _store_context(self, sequence_id, context):
		"""Store the context for a sequence."""

		self._sequence_context[sequence_id] = context

	def _drop_context(self, sequence_id):
		"""Remove the context for a sequence, if it exists."""

		if sequence_id in self._sequence_context:
			del self._sequence_context[sequence_id]

	def _normalize_sequence_ids(self, sequence_ids, n):
		"""Normalize sequence ids into one hashable value per chunk."""

		if sequence_ids is None:
			return [0] * n

		if isinstance(sequence_ids, (str, bytes)):
			return [_as_scalar(sequence_ids)] * n

		if isinstance(sequence_ids, (torch.Tensor, numpy.ndarray)):
			sequence_ids = _as_list(sequence_ids)
		else:
			try:
				sequence_ids = list(sequence_ids)
			except TypeError:
				return [_as_scalar(sequence_ids)] * n

		if len(sequence_ids) == 1:
			sequence_ids = sequence_ids * n
		elif len(sequence_ids) != n:
			raise ValueError("sequence_ids must have one entry per chunk.")

		return [_as_scalar(sequence_id) for sequence_id in sequence_ids]

	def _normalize_flags(self, flag, n, name):
		"""Normalize a begin/end parameter into one boolean per chunk."""

		if flag is None:
			return [False] * n

		if isinstance(flag, (bool, numpy.bool_)):
			return [bool(flag)] * n

		if isinstance(flag, (torch.Tensor, numpy.ndarray)):
			values = _as_list(flag)

			if len(values) == 1:
				return [bool(values[0])] * n
			elif len(values) != n:
				raise ValueError("{} must have one entry per chunk.".format(
					name))

			return [bool(value) for value in values]

		if isinstance(flag, (list, tuple)):
			if len(flag) == 1:
				return [bool(flag[0])] * n
			elif len(flag) != n:
				raise ValueError("{} must have one entry per chunk.".format(
					name))

			return [bool(value) for value in flag]

		raise ValueError("{} must be a boolean or an array of booleans.".format(
			name))

	def _states_to_flags(self, state, n):
		"""Convert explicit begin/continue/end states into flags."""

		if isinstance(state, str):
			states = [state] * n
		elif isinstance(state, (torch.Tensor, numpy.ndarray)):
			states = _as_list(state)

			if len(states) == 1:
				states = states * n
			elif len(states) != n:
				raise ValueError("state must have one entry per chunk.")
		elif isinstance(state, (list, tuple)):
			states = list(state)

			if len(states) == 1:
				states = states * n
			elif len(states) != n:
				raise ValueError("state must have one entry per chunk.")
		else:
			raise ValueError("state must be a string or a list of strings.")

		begin, end = [], []
		for value in states:
			if not isinstance(value, str):
				raise ValueError("state must be one of 'begin', 'continue', "
					"or 'end'.")

			value = value.lower()

			if value in ("begin", "start"):
				begin.append(True)
				end.append(False)
			elif value in ("continue", "middle"):
				begin.append(False)
				end.append(False)
			elif value in ("end", "stop", "finish"):
				begin.append(False)
				end.append(True)
			else:
				raise ValueError("state must be one of 'begin', 'continue', "
					"or 'end'.")

		return begin, end

	def _expand_weights(self, weight, length, d, dtype, device):
		"""Expand a single chunk's weights to shape (length, d)."""

		weight = _cast_as_tensor(weight, dtype=dtype)

		if weight.ndim == 0:
			return weight.expand(length, d).clone()

		if weight.ndim == 1:
			if weight.shape[0] == 1:
				return weight.expand(length, d).clone()

			if weight.shape[0] == length:
				return weight.reshape(length, 1).expand(length, d).clone()

			if weight.shape[0] == d:
				return weight.reshape(1, d).expand(length, d).clone()

		elif weight.ndim == 2:
			if weight.shape in ((1, 1), (1, d)):
				return weight.expand(length, d).clone()

			if weight.shape[0] == length and weight.shape[1] in (1, d):
				return weight.expand(length, d).clone()

		raise ValueError("sample_weight must be broadcastable to shape "
			"({}, {}).".format(length, d))

	def _prepare_stream_weights(self, sample_weight, chunks, dtype, device):
		"""Normalize weights to one (length, d) tensor per chunk."""

		lengths = [len(chunk) for chunk in chunks]
		n, d = len(chunks), chunks[0].shape[1]
		total = sum(lengths)

		if sample_weight is None:
			return [torch.ones(length, d, dtype=dtype, device=device)
				for length in lengths]

		if isinstance(sample_weight, (list, tuple)):
			try:
				sample_weight = _cast_as_tensor(sample_weight, dtype=dtype)
			except Exception:
				if len(sample_weight) != n:
					raise ValueError("sample_weight must have one entry per "
						"chunk.")

				return [self._expand_weights(weight, length, d, dtype, device)
					for weight, length in zip(sample_weight, lengths)]

		if not isinstance(sample_weight, torch.Tensor):
			sample_weight = _cast_as_tensor(sample_weight, dtype=dtype)

		if sample_weight.ndim == 0:
			return [sample_weight.expand(length, d).clone()
				for length in lengths]

		if sample_weight.ndim == 1:
			if sample_weight.shape[0] == total and total != n:
				weights, start = [], 0
				for length in lengths:
					weight = sample_weight[start:start+length]
					weights.append(weight.reshape(length, 1).expand(length, d)
						.clone())
					start += length

				return weights

			if sample_weight.shape[0] == n:
				return [weight.reshape(1, 1).expand(length, d).clone()
					for weight, length in zip(sample_weight, lengths)]

			if sample_weight.shape[0] == d:
				# A single per-feature weight vector shared by every chunk.
				return [sample_weight.reshape(1, d).expand(length, d).clone()
					for length in lengths]

		elif sample_weight.ndim == 2 and sample_weight.shape[0] == n and \
			sample_weight.shape[1] in (1, d):
			# A (n, 1) or (n, d) matrix gives one weight per chunk that is
			# broadcast across its observations, matching the per-sequence
			# weights accepted by a non-chunked call.
			return [sample_weight[i].reshape(1, -1).expand(length, d).clone()
				for i, length in enumerate(lengths)]

		elif sample_weight.ndim == 2:
			# Otherwise the weights may be laid out per observation over the
			# concatenation of all chunks, either as (total, 1) or (total, d).
			if sample_weight.numel() == total:
				values = sample_weight.reshape(-1)
				weights, start = [], 0

				for length in lengths:
					weight = values[start:start+length]
					weights.append(weight.reshape(length, 1).expand(length, d)
						.clone())
					start += length

				return weights

			if sample_weight.numel() == total * d:
				values = sample_weight.reshape(total, d)
				weights, start = [], 0

				for length in lengths:
					weights.append(values[start:start+length].clone())
					start += length

				return weights

		elif sample_weight.ndim == 3:
			if sample_weight.shape[0] != n:
				raise ValueError("sample_weight must have one entry per "
					"chunk.")

			if sample_weight.shape[2] not in (1, d):
				raise ValueError("sample_weight must have shape (-1, -1, {}) "
					"or (-1, -1, 1).".format(d))

			return [self._expand_weights(sample_weight[i, :length], length, d,
				dtype, device) for i, length in enumerate(lengths)]

		raise ValueError("sample_weight must be broadcastable to the chunks "
			"in X.")

	def _summarize_stream(self, chunks, sample_weight, sequence_ids, begin,
		end):
		"""Summarize chunks of one or more sequences using stored context."""

		k = int(self.k)
		device = chunks[0].device

		for i, chunk in enumerate(chunks):
			if chunk.ndim != 2:
				raise ValueError("Each chunk must be 2D, but chunk {} has {} "
					"dimensions.".format(i, chunk.ndim))

			if chunk.dtype not in (torch.int32, torch.int64):
				if self.check_data:
					raise ValueError("X must contain integer observations, "
						"but chunk {} has dtype {}.".format(i, chunk.dtype))

				chunk = chunk.type(torch.int32)

			if self.check_data and (chunk < 0).any():
				raise ValueError("X must contain non-negative observations.")

			chunks[i] = chunk.to(device)

		n, d = len(chunks), chunks[0].shape[1]
		lengths = [len(chunk) for chunk in chunks]

		for i, chunk in enumerate(chunks):
			if chunk.shape[1] != d:
				raise ValueError("All chunks must have {} features, but chunk "
					"{} has {}.".format(d, i, chunk.shape[1]))

		sequence_ids = self._normalize_sequence_ids(sequence_ids, n)
		begins = self._normalize_flags(begin, n, "begin")
		ends = self._normalize_flags(end, n, "end")

		if not self._initialized:
			if self.n_categories is not None:
				n_keys = _cast_as_tensor(self.n_categories, dtype=torch.int32)
			elif sum(lengths) == 0:
				raise ValueError("Cannot infer the number of categories from "
					"empty chunks. Pass 'n_categories' to the model or "
					"summarize a non-empty chunk first.")
			else:
				data = torch.cat([chunk for chunk in chunks if len(chunk) > 0],
					dim=0)
				n_keys = (torch.max(data, dim=0)[0] + 1).type(torch.int32)

			self._initialize(d, n_keys)

		feature_dim = self._feature_dim()
		if feature_dim is not None and feature_dim != d:
			raise ValueError("Chunks must have {} features, but found {}."
				.format(feature_dim, d))

		self._check_context_schema()

		limits = self._category_limits()
		if limits is not None and self.check_data:
			limits_ = torch.tensor(limits, device=device)

			for i, chunk in enumerate(chunks):
				if len(chunk) == 0:
					continue

				if (chunk >= limits_).any():
					raise ValueError("Observations fall outside the category "
						"schema of the model ({}). Pass a model with a larger "
						"'n_categories' or call clear_context() to discard the "
						"stored context if a new schema is intended.".format(
							limits))

		probs = self.distributions[0].probs
		dtype = torch.float32 if probs is None else probs.dtype
		weights = self._prepare_stream_weights(sample_weight, chunks, dtype,
			device)

		# Each entry in `pending` stores the windows for one distribution
		# together with the weight of the true observation they predict, i.e.
		# the last observation in each window.
		pending = [[] for _ in range(k+1)]

		for chunk, weight, sequence_id, is_begin, is_end in zip(chunks, weights,
			sequence_ids, begins, ends):

			length = len(chunk)

			if is_begin:
				self._drop_context(sequence_id)

			if length == 0:
				if is_end:
					self._drop_context(sequence_id)
				continue

			context = self._sequence_context.get(sequence_id)

			if context is None:
				context = {"tail": chunk.new_zeros((0, d)), "seen": 0,
					"prefix": 0}
				self._store_context(sequence_id, context)
				n_before = 0
			else:
				n_before = context["seen"]

			tail = context["tail"].to(device)
			tail_length = len(tail)
			extended = torch.cat([tail, chunk], dim=0)

			if k == 0:
				pending[0].append((chunk, weight))
			else:
				# The shorter distributions only model the windows that begin
				# a sequence, matching a single call on the full sequences.
				if n_before == 0:
					pending[0].append((chunk[0:1], weight[0:1]))

				prefix = context["prefix"]
				while prefix < k - 1 and n_before + length >= prefix + 2:
					m = prefix + 1
					window = extended[0:m+1].unsqueeze(0)
					window_weight = weight[m-n_before].reshape(1, -1)
					pending[m].append((window, window_weight))
					prefix = m

				context["prefix"] = prefix

				# Every window of order k is emitted the moment its last
				# observation arrives, so that no window is missed or counted
				# twice at a chunk boundary.
				if extended.shape[0] >= k + 1:
					n_windows = extended.shape[0] - k
					windows = extended.unfold(0, k + 1, 1).permute(0, 2, 1)
					start = k - tail_length
					pending[k].append((windows, weight[start:start+n_windows]))

			if k > 0:
				context["tail"] = extended[-k:].clone()
			else:
				context["tail"] = chunk.new_zeros((0, d))

			context["seen"] = n_before + length

			if is_end:
				self._drop_context(sequence_id)

		for i, values in enumerate(pending):
			if len(values) == 0:
				continue

			X_ = torch.cat([value[0] for value in values], dim=0)
			w_ = torch.cat([value[1] for value in values], dim=0)

			self.distributions[i].summarize(X_, sample_weight=w_)

		if self._sequence_context:
			self._context_schema = self._stream_schema()
		else:
			self._context_schema = None

	def get_extra_state(self):
		"""Return the sequence context so that it is part of the state."""

		self._ensure_context_state()

		context = {}
		for sequence_id, value in self._sequence_context.items():
			context[sequence_id] = {"tail": value["tail"].clone(),
				"seen": value["seen"], "prefix": value["prefix"]}

		return {"sequence_context": context, "context_schema":
			self._context_schema}

	def set_extra_state(self, state):
		"""Restore the sequence context from the state."""

		self._ensure_context_state()

		if state is None:
			self._sequence_context = {}
			self._context_schema = None
			return

		context = {}
		for sequence_id, value in state.get("sequence_context", {}).items():
			context[sequence_id] = {"tail": value["tail"].clone(),
				"seen": value["seen"], "prefix": value["prefix"]}

		self._sequence_context = context
		self._context_schema = state.get("context_schema")

	def load_state_dict(self, state_dict, strict=True, **kwargs):
		"""Load a state, tolerating states saved before context was added."""

		if "_extra_state" not in state_dict:
			state_dict = dict(state_dict)
			state_dict["_extra_state"] = {"sequence_context": {},
				"context_schema": None}

		return super().load_state_dict(state_dict, strict=strict, **kwargs)

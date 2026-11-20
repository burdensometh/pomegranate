# normal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _check_shapes

from ._distribution import Distribution


# Define some useful constants
NEGINF = float("-inf")
INF = float("inf")
SQRT_2_PI = 2.50662827463
LOG_2_PI = 1.83787706641


def _safe_cholesky(covs):
	"""Return a lower-triangular Cholesky factor of a (possibly degenerate) cov.

	When the covariance matrix is not positive definite it is regularized with
	an increasing amount of diagonal jitter before the factorization is retried.
	This makes the handling of degenerate covariance matrices explicit instead
	of silently producing linear-algebra errors. If the factorization still
	fails after several retries the Moore-Penrose pseudo-inverse based factor is
	returned so that the calculation can proceed.


	Parameters
	----------
	covs: torch.Tensor, shape=(d, d)
		The covariance matrix to factorize.


	Returns
	-------
	L: torch.Tensor, shape=(d, d)
		A lower-triangular matrix such that L @ L.T approximates `covs`.
	"""

	try:
		return torch.linalg.cholesky(covs)
	except Exception:
		pass

	eye = torch.eye(covs.shape[0], dtype=covs.dtype, device=covs.device)
	diag = torch.diagonal(covs)
	scale = torch.maximum(diag.max(), torch.tensor(1.0, dtype=covs.dtype,
		device=covs.device))

	for jitter in (1e-6, 1e-4, 1e-2, 1e-1):
		try:
			return torch.linalg.cholesky(covs + jitter * scale * eye)
		except Exception:
			pass

	# Final fallback: a factor built from the pseudo-inverse square root. This
	# path is only reached when every jittered factorization has failed.
	inv = torch.linalg.pinv(covs)
	L_inv = torch.linalg.cholesky(0.5 * (inv + inv.T) + 1e-6 * scale * eye)
	return torch.linalg.inv(L_inv).T


def _mask_patterns(mask):
	"""Group the rows of a mask by their observed-feature pattern.

	The Cholesky factorizations of the marginal and conditional covariance
	matrices depend only on which features are observed, not on the observed
	values themselves. Hence, rows sharing a mask pattern can reuse a single
	set of factorizations.


	Parameters
	----------
	mask: torch.Tensor, shape=(n, d), dtype=torch.bool
		A boolean tensor where True indicates an observed feature.


	Returns
	-------
	patterns: torch.Tensor, shape=(p, d), dtype=torch.int8
		The unique observed-feature patterns, one row per pattern.

	inverse: torch.Tensor, shape=(n,), dtype=torch.long
		The index of the pattern associated with each row of `mask`.
	"""

	return torch.unique(mask.to(torch.int8), dim=0, return_inverse=True)


class Normal(Distribution):
	"""A normal distribution object.

	A normal distribution models the probability of a variable occurring under
	a bell-shaped curve. It is described by a vector of mean values and a
	covariance value that can be zero, one, or two dimensional. This
	distribution can assume that features are independent of the others if
	the covariance type is 'diag' or 'sphere', but if the type is 'full' then
	the features are not independent.

	There are two ways to initialize this object. The first is to pass in
	the tensor of probability parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.

	Missing values are supported through `torch.masked.MaskedTensor` inputs,
	where `mask=True` marks an observed feature and `mask=False` marks a
	missing one. For full-covariance distributions the marginal likelihood is
	evaluated on the observed features only, rows that are entirely missing
	contribute a log probability of zero (i.e. carry no information beyond the
	priors), and during training the conditional mean and conditional covariance
	of the missing features given the observed ones are folded into the
	sufficient statistics used by the EM algorithm. Missing entries are never
	replaced with zeros and treated as observations.


	Parameters
	----------
	means: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
		The mean values of the distributions. Default is None.

	covs: list, numpy.ndarray, torch.Tensor, or None, optional
		The variances and covariances of the distribution. If covariance_type
		is 'full', the shape should be (self.d, self.d); if 'diag', the shape
		should be (self.d,); if 'sphere', it should be (1,). Note that this is
		the variances or covariances in all settings, and not the standard
		deviation, as may be more common for diagonal covariance matrices.
		Default is None.

	covariance_type: str, optional
		The type of covariance matrix. Must be one of 'full', 'diag', or
		'sphere'. Default is 'full'.

	min_cov: float or None, optional
		The minimum variance or covariance.

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
	"""

	def __init__(self, means=None, covs=None, covariance_type='full',
		min_cov=None, inertia=0.0, frozen=False, check_data=True):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
		self.name = "Normal"

		self.means = _check_parameter(_cast_as_parameter(means), "means",
			ndim=1)
		self.covs = _check_parameter(_cast_as_parameter(covs), "covs",
			ndim=(1, 2))

		_check_shapes([self.means, self.covs], ["means", "covs"])

		self.min_cov = _check_parameter(min_cov, "min_cov", min_value=0, ndim=0)
		self.covariance_type = covariance_type

		self._initialized = (means is not None) and (covs is not None)
		self.d = self.means.shape[-1] if self._initialized else None
		self._reset_cache()

	def _initialize(self, d):
		"""Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""

		self.means = _cast_as_parameter(torch.zeros(d, dtype=self.dtype,
			device=self.device))

		if self.covariance_type == 'full':
			self.covs = _cast_as_parameter(torch.zeros(d, d,
				dtype=self.dtype, device=self.device))
		elif self.covariance_type == 'diag':
			self.covs = _cast_as_parameter(torch.zeros(d, dtype=self.dtype,
				device=self.device))
		elif self.covariance_type == 'sphere':
			self.covs = _cast_as_parameter(torch.tensor(0, dtype=self.dtype,
				device=self.device))

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		if self._initialized == False:
			return

		self.register_buffer("_w_sum", torch.zeros(self.d, dtype=self.dtype,
			device=self.device))
		self.register_buffer("_xw_sum", torch.zeros(self.d, dtype=self.dtype,
			device=self.device))

		if self.covariance_type == 'full':
			self.register_buffer("_xxw_sum", torch.zeros(self.d, self.d,
				dtype=self.dtype, device=self.device))

			if self.covs.sum() > 0.0:
				chol = _safe_cholesky(self.covs)
				_inv_cov = torch.linalg.solve_triangular(chol, torch.eye(
					len(self.covs), dtype=self.dtype, device=self.device),
					upper=False).T
				_inv_cov_dot_mu = torch.matmul(self.means, _inv_cov)
				_log_det = -0.5 * torch.linalg.slogdet(self.covs)[1]
				_theta = _log_det - 0.5 * (self.d * LOG_2_PI)

				self.register_buffer("_inv_cov", _inv_cov)
				self.register_buffer("_inv_cov_dot_mu", _inv_cov_dot_mu)
				self.register_buffer("_log_det", _log_det)
				self.register_buffer("_theta", _theta)

		elif self.covariance_type in ('diag', 'sphere'):
			self.register_buffer("_xxw_sum", torch.zeros(self.d,
				dtype=self.dtype, device=self.device))

			if self.covs.sum() > 0.0:
				_log_sigma_sqrt_2pi = -torch.log(torch.sqrt(self.covs) *
					SQRT_2_PI)
				_inv_two_sigma = 1. / (2 * self.covs)

				self.register_buffer("_log_sigma_sqrt_2pi", _log_sigma_sqrt_2pi)
				self.register_buffer("_inv_two_sigma", _inv_two_sigma)

			if torch.any(self.covs < 0):
				raise ValueError("Variances must be positive.")

	def sample(self, n):
		"""Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution.


		Parameters
		----------
		n: int
			The number of samples to generate.


		Returns
		-------
		X: torch.tensor, shape=(n, self.d)
			Randomly generated samples.
		"""

		if self.covariance_type == 'diag':
			return torch.distributions.Normal(self.means, self.covs).sample([n])
		elif self.covariance_type == 'full':
			return torch.distributions.MultivariateNormal(self.means,
				self.covs).sample([n])

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format.

		When a `torch.masked.MaskedTensor` is provided, the mask determines
		which features are observed for each example. For full-covariance
		distributions, the marginal distribution over the observed features is
		evaluated separately for each distinct mask pattern, reusing the
		factorization within a pattern. Rows without any observed features get
		a log probability of zero because an empty observation carries no
		information; observed values are always used as-is and missing values
		are never filled in with zeros.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate, optionally as a masked tensor.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		X = _check_parameter(_cast_as_tensor(X, dtype=self.means.dtype), "X",
			ndim=2, shape=(-1, self.d), check_parameter=self.check_data)

		if not isinstance(X, torch.masked.MaskedTensor):
			if self.covariance_type == 'full':
				logp = torch.matmul(X, self._inv_cov) - self._inv_cov_dot_mu
				logp = self.d * LOG_2_PI + torch.sum(logp ** 2, dim=-1)
				logp = self._log_det - 0.5 * logp
				return logp

			elif self.covariance_type in ('diag', 'sphere'):
				return torch.sum(self._log_sigma_sqrt_2pi -
					((X - self.means) ** 2) * self._inv_two_sigma, dim=-1)

		mask = X._masked_mask
		data = X._masked_data
		n, d = mask.shape

		if self.covariance_type in ('diag', 'sphere'):
			# Features are independent, so missing features can simply be
			# skipped: masked arithmetic ignores the unobserved coordinates,
			# giving exactly the marginal log probability of the observed
			# coordinates. The result stays masked for rows without any
			# observation so that they do not pretend to carry evidence.
			logp = torch.sum(self._log_sigma_sqrt_2pi -
				((X - self.means) ** 2) * self._inv_two_sigma, dim=-1)
			return logp

		logp = torch.zeros(n, dtype=self.means.dtype, device=self.device)
		patterns, inverse = _mask_patterns(mask)

		for group_id, pattern in enumerate(patterns):
			idx = inverse == group_id
			observed = pattern.bool()

			# A fully missing row has an empty observation and therefore a
			# log probability of zero; it must not be scored as if the
			# missing entries had been observed as zeros.
			if not observed.any():
				continue

			mu_o = self.means[observed]
			cov_oo = self.covs[observed][:, observed]
			chol = _safe_cholesky(cov_oo)

			x_o = data[idx][:, observed]
			diff = x_o - mu_o

			Linv_diff = torch.linalg.solve_triangular(chol, diff.T,
				upper=False)
			log_det = torch.log(torch.diagonal(chol)).sum()

			k = observed.sum()
			logp[idx] = -0.5 * (k * LOG_2_PI +
				torch.sum(Linv_diff ** 2, dim=0)) - log_det

		return logp

	def summarize(self, X, sample_weight=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.

		When a `torch.masked.MaskedTensor` is given, the statistics incorporate
		the missing values through the EM conditional expectations. For each
		mask pattern the conditional mean of the missing features given the
		observed ones is used in place of the missing coordinates, and the
		conditional covariance is added to the weighted second-moment matrix
		so that the imputation uncertainty is not ignored. Entirely missing
		rows are imputed with the unconditional mean and covariance and so do
		not introduce any information beyond the current model itself.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize, optionally as a masked tensor.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""

		if self.frozen == True:
			return

		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		X = _cast_as_tensor(X, dtype=self.means.dtype)
		sample_weight = _cast_as_tensor(sample_weight, dtype=self.means.dtype)

		if not isinstance(X, torch.masked.MaskedTensor):
			if self.covariance_type == 'full':
				self._w_sum += torch.sum(sample_weight, dim=0)
				self._xw_sum += torch.sum(X * sample_weight, axis=0)
				self._xxw_sum += torch.matmul((X * sample_weight).T, X)

			elif self.covariance_type in ('diag', 'sphere'):
				self._w_sum[:] = self._w_sum + torch.sum(sample_weight, dim=0)
				self._xw_sum[:] = self._xw_sum + torch.sum(X * sample_weight,
					dim=0)
				self._xxw_sum[:] = self._xxw_sum + torch.sum(X ** 2 *
					sample_weight, dim=0)
			return

		mask = X._masked_mask
		data = X._masked_data
		n, d = mask.shape

		if self.covariance_type in ('diag', 'sphere'):
			# The features are assumed independent, so a missing feature
			# simply contributes no information about that feature: masked
			# arithmetic skips it in every statistic. Fully missing rows
			# therefore contribute nothing at all rather than a zero-filled
			# pseudo-observation.
			self._w_sum[:] = self._w_sum + torch.sum(sample_weight, dim=0)
			self._xw_sum[:] = self._xw_sum + torch.sum(X * sample_weight,
				dim=0)
			self._xxw_sum[:] = self._xxw_sum + torch.sum(X ** 2 *
				sample_weight, dim=0)
			return

		# Full covariance path: reduce the (possibly per-feature) weights to
		# one scalar weight per example. Per-feature weights are averaged
		# over the observed features; a fully missing row falls back to the
		# mean over all features so that its prior-based imputation still
		# carries the example-level weight that was provided.
		if isinstance(sample_weight, torch.masked.MaskedTensor):
			weight_data = sample_weight._masked_data
			weight_mask = sample_weight._masked_mask
		else:
			weight_data = sample_weight
			weight_mask = mask

		n_observed = weight_mask.sum(dim=1).clamp(min=1)
		row_weights = (weight_data * weight_mask).sum(dim=1) / n_observed
		full_missing = ~mask.any(dim=1)
		if full_missing.any():
			row_weights[full_missing] = weight_data[full_missing].mean(dim=1)

		# Process each distinct observed-feature pattern once, reusing the
		# marginal factorization across all rows that share it.
		patterns, inverse = _mask_patterns(mask)

		w_total = row_weights.sum()
		xw = torch.zeros(d, dtype=self.means.dtype, device=self.device)
		xxw = torch.zeros(d, d, dtype=self.means.dtype, device=self.device)

		for group_id, pattern in enumerate(patterns):
			idx = inverse == group_id
			observed = pattern.bool()
			missing = ~observed
			w_g = row_weights[idx]
			X_g = data[idx]

			if observed.all():
				x_imp = X_g
				cond_cov = None
			elif not observed.any():
				# Fully missing rows contribute the unconditional mean and
				# covariance, i.e. no information beyond the current model.
				x_imp = self.means.unsqueeze(0).repeat(idx.sum(), 1)
				cond_cov = self.covs
			else:
				cov_oo = self.covs[observed][:, observed]
				cov_mo = self.covs[missing][:, observed]
				cov_mm = self.covs[missing][:, missing]

				chol = _safe_cholesky(cov_oo)
				# Cross-covariance solved against the marginal factor.
				solved = torch.cholesky_solve(
					(X_g[:, observed] - self.means[observed]).T, chol).T
				cond_mean_m = self.means[missing] + solved @ cov_mo.T

				# Conditional covariance of the missing block:
				# C_mm - C_mo C_oo^{-1} C_om.
				cond_cov_mm = cov_mm - cov_mo @ torch.cholesky_solve(
					cov_mo.T, chol)
				cond_cov_mm = 0.5 * (cond_cov_mm + cond_cov_mm.T)

				x_imp = torch.empty(idx.sum(), d, dtype=self.means.dtype,
					device=self.device)
				x_imp[:, observed] = X_g[:, observed]
				x_imp[:, missing] = cond_mean_m

				cond_cov = torch.zeros(d, d, dtype=self.means.dtype,
					device=self.device)
				cond_cov[missing[:, None] & missing[None, :]] = \
					cond_cov_mm.reshape(-1)

			xw += (x_imp * w_g.unsqueeze(1)).sum(dim=0)
			xxw += torch.matmul((x_imp * w_g.unsqueeze(1)).T, x_imp)
			if cond_cov is not None:
				xxw += w_g.sum() * cond_cov

		# Every example contributes one full (imputed) vector, so the total
		# weight is the same for every feature even when individual
		# coordinates were missing.
		self._w_sum += w_total
		self._xw_sum += xw
		self._xxw_sum += xxw

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		if self.frozen == True:
			return

		# When at least one example contributes, features with zero
		# accumulated weight are unobserved in every summarized batch: keep
		# their previous parameters instead of producing NaNs. When nothing
		# has been summarized at all the original behavior (unprotected
		# division) is retained.
		any_weight = self._w_sum.sum() > 0

		if self.covariance_type == 'full':
			w = self._w_sum
			w_safe = torch.where(w > 0, w, torch.ones_like(w)) if \
				any_weight else w

			means = self._xw_sum / w_safe

			v = self._xw_sum.unsqueeze(0) * self._xw_sum.unsqueeze(1)
			covs = self._xxw_sum / w_safe - v / w_safe ** 2.0

			if any_weight:
				unobserved = w <= 0
				if unobserved.any():
					means[unobserved] = self.means[unobserved]
					covs[unobserved, :] = self.covs[unobserved, :]
					covs[:, unobserved] = self.covs[:, unobserved]

			if self.min_cov is not None:
				covs = covs + self.min_cov * torch.eye(self.d,
					dtype=covs.dtype, device=covs.device)

		elif self.covariance_type in ['diag', 'sphere']:
			w = self._w_sum
			w_safe = torch.where(w > 0, w, torch.ones_like(w)) if \
				any_weight else w
			covs = self._xxw_sum / w_safe - \
				self._xw_sum ** 2.0 / w_safe ** 2.0

			if any_weight:
				unobserved = w <= 0
				if unobserved.any():
					covs[unobserved] = self.covs if \
						self.covariance_type == 'sphere' else \
						self.covs[unobserved]

			if self.min_cov is not None:
				covs = torch.maximum(covs, torch.full_like(covs,
					float(self.min_cov)))

			if self.covariance_type == 'sphere':
				covs = covs.mean(dim=-1)

			means = self._xw_sum / w_safe
			if any_weight and (w <= 0).any():
				means[w <= 0] = self.means[w <= 0]

		_update_parameter(self.means, means, self.inertia)
		_update_parameter(self.covs, covs, self.inertia)
		self._reset_cache()

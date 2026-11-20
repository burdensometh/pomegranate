# test_normal_full_missing.py
# Tests for full-covariance Normal handling of partially missing features.
#
# The key property being checked is that a full-covariance Normal cannot just
# ignore missing columns and keep applying the full-vector formula: the
# marginal likelihood must be evaluated on the observed coordinates only and
# the EM sufficient statistics must include the conditional mean and
# conditional covariance of the missing coordinates. Missing entries are never
# replaced with zeros, fully missing rows carry no evidence beyond the model,
# and observed values are used unchanged.

import numpy
import torch

from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel

from numpy.testing import assert_array_almost_equal


LOG_2_PI = 1.83787706641


def _masked(X, mask):
	return torch.masked.MaskedTensor(torch.tensor(numpy.array(X),
		dtype=torch.float32), mask=torch.tensor(numpy.array(mask)))


def _marginal_logp(X, mask, mu, cov):
	"""Independent numpy reference for the masked marginal log probability."""
	X = numpy.array(X, dtype=numpy.float64)
	mu = numpy.array(mu, dtype=numpy.float64)
	cov = numpy.array(cov, dtype=numpy.float64)
	n, d = X.shape
	out = numpy.zeros(n)

	for i in range(n):
		o = numpy.array(mask[i], dtype=bool)
		if not o.any():
			out[i] = 0.0
			continue

		diff = X[i, o] - mu[o]
		c_oo = cov[numpy.ix_(o, o)]
		k = int(o.sum())
		out[i] = -0.5 * (k * LOG_2_PI +
			diff @ numpy.linalg.solve(c_oo, diff) +
			numpy.linalg.slogdet(c_oo)[1])

	return out


def _conditional_summaries(X, mask, mu, cov):
	"""Independent numpy reference for the EM conditional E-step statistics."""
	X = numpy.array(X, dtype=numpy.float64)
	mu = numpy.array(mu, dtype=numpy.float64)
	cov = numpy.array(cov, dtype=numpy.float64)
	n, d = X.shape

	Ex = numpy.zeros((n, d))
	Exx = numpy.zeros((n, d, d))
	for i in range(n):
		o = numpy.array(mask[i], dtype=bool)
		m = ~o
		ex = X[i].copy()

		if not o.any():
			ex = mu.copy()
			exx = cov + numpy.outer(mu, mu)
		else:
			exx = numpy.outer(X[i], X[i])
			if m.any():
				c_oo = cov[numpy.ix_(o, o)]
				c_mo = cov[numpy.ix_(m, o)]
				ex[m] = mu[m] + c_mo @ numpy.linalg.solve(
					c_oo, X[i, o] - mu[o])
				cond_cov = cov[numpy.ix_(m, m)] - c_mo @ numpy.linalg.solve(
					c_oo, c_mo.T)
				exx = numpy.outer(ex, ex)
				exx[numpy.ix_(m, m)] += cond_cov
			else:
				exx = numpy.outer(ex, ex)

		Ex[i] = ex
		Exx[i] = exx

	return Ex.sum(axis=0), Exx.sum(axis=0)


def _model():
	mu = [1.2, 1.8, 2.1]
	cov = [[0.30, 0.10, 0.00],
	       [0.10, 2.10, 0.60],
	       [0.00, 0.60, 1.20]]
	return mu, cov, Normal(mu, cov, covariance_type='full')


def _data():
	X = [[1.1, 2.1, 0.5],
	     [0.5, 0.2, 1.3],
	     [1.4, 1.1, 2.2],
	     [3.1, 2.1, 2.2],
	     [3.4, 1.0, 0.3],
	     [5.4, 1.9, 4.0],
	     [2.2, 1.3, 0.1]]

	# Row 2 is fully missing; rows 0, 4 are partially missing; row 3 is fully
	# observed.
	mask = [[True,  True,  False],
	        [True,  False, True ],
	        [False, False, False],
	        [True,  True,  True ],
	        [False, True,  False],
	        [True,  True,  True ],
	        [True,  False, True ]]
	return X, mask


def test_masked_marginal_log_probability():
	mu, cov, d = _model()
	X, mask = _data()
	Xm = _masked(X, mask)

	expected = _marginal_logp(X, mask, mu, cov)
	assert_array_almost_equal(d.log_probability(Xm), expected, 5)


def test_fully_missing_row_has_zero_log_probability():
	_, _, d = _model()
	X, mask = _data()
	Xm = _masked(X, mask)

	logp = d.log_probability(Xm)
	assert logp[2].item() == 0.0


def test_missing_is_not_treated_as_zero():
	mu, cov, d = _model()
	X, mask = _data()

	Xm = _masked(X, mask)
	X_zero = numpy.array(X, dtype=numpy.float32).copy()
	X_zero[~numpy.array(mask, dtype=bool)] = 0.0

	masked_logp = d.log_probability(Xm).numpy()
	zero_logp = d.log_probability(torch.tensor(X_zero)).numpy()
	assert not numpy.allclose(masked_logp, zero_logp)


def test_observed_values_kept_as_is():
	mu, cov, d = _model()
	X, mask = _data()

	Xm = _masked(X, mask)
	plain = d.log_probability(torch.tensor(numpy.array(X, dtype=numpy.float32)))
	masked = d.log_probability(Xm)

	# Fully observed rows must give identical scores either way.
	assert_array_almost_equal(masked[[3, 5]], plain[[3, 5]], 6)


def test_masked_equals_unmasked_when_everything_observed():
	mu, cov, d = _model()
	X, _ = _data()
	Xt = torch.tensor(numpy.array(X, dtype=numpy.float32))
	Xm = torch.masked.MaskedTensor(Xt,
		mask=torch.ones(Xt.shape, dtype=torch.bool))

	assert_array_almost_equal(d.log_probability(Xm), d.log_probability(Xt), 6)


def test_summarize_matches_conditional_sufficient_statistics():
	mu, cov, d = _model()
	X, mask = _data()
	Xm = _masked(X, mask)

	d.summarize(Xm)
	xw_ref, xxw_ref = _conditional_summaries(X, mask, mu, cov)

	assert_array_almost_equal(d._w_sum, numpy.full(3, len(X)), 6)
	assert_array_almost_equal(d._xw_sum, xw_ref, 5)
	assert_array_almost_equal(d._xxw_sum, xxw_ref, 5)


def test_summarize_reuses_factorizations_across_mask_patterns():
	mu, cov, d = _model()
	X, mask = _data()

	# Duplicating rows that share a pattern must double their contribution.
	X2 = X + X
	mask2 = mask + mask
	Xm = _masked(X2, mask2)
	d.summarize(Xm)

	xw_ref, xxw_ref = _conditional_summaries(X, mask, mu, cov)
	assert_array_almost_equal(d._xw_sum, 2 * xw_ref, 5)
	assert_array_almost_equal(d._xxw_sum, 2 * xxw_ref, 5)


def test_summarize_is_permutation_invariant():
	mu, cov, d = _model()
	X, mask = _data()

	perm = [6, 2, 0, 5, 1, 4, 3]
	Xp = [X[i] for i in perm]
	maskp = [mask[i] for i in perm]

	d.summarize(_masked(Xp, maskp))
	xw_ref, xxw_ref = _conditional_summaries(X, mask, mu, cov)
	assert_array_almost_equal(d._xw_sum, xw_ref, 5)
	assert_array_almost_equal(d._xxw_sum, xxw_ref, 5)


def test_from_summaries_recovers_conditional_mle():
	mu, cov, d = _model()
	X, mask = _data()
	Xm = _masked(X, mask)

	d.summarize(Xm)
	d.from_summaries()

	xw_ref, xxw_ref = _conditional_summaries(X, mask, mu, cov)
	n = len(X)
	mu_hat = xw_ref / n
	cov_hat = xxw_ref / n - numpy.outer(mu_hat, mu_hat)

	assert_array_almost_equal(d.means, mu_hat, 5)
	assert_array_almost_equal(d.covs, cov_hat, 5)


def test_fully_missing_rows_add_no_information():
	mu, cov, d = _model()
	X, mask = _data()

	# Summarize the full batch, then a batch where every row is fully
	# missing. Fully missing rows only contribute the model's own
	# unconditional mean and covariance (the EM imputation), never any
	# information from the (absent) observations.
	Xm = _masked(X, mask)
	d.summarize(Xm)
	xw_before = d._xw_sum.clone()
	xxw_before = d._xxw_sum.clone()
	w_before = d._w_sum.clone()

	empty_mask = numpy.zeros_like(mask, dtype=bool)
	d.summarize(_masked(X, empty_mask))

	n = len(X)
	mu_a = numpy.array(mu)
	cov_a = numpy.array(cov)

	assert_array_almost_equal(d._w_sum, w_before + n, 6)
	assert_array_almost_equal(d._xw_sum, xw_before + n * mu_a, 5)
	extra_xxw = n * (cov_a + numpy.outer(mu_a, mu_a))
	assert_array_almost_equal(d._xxw_sum, xxw_before + extra_xxw, 4)


def test_singular_covariance_is_regularized_not_crashing():
	# Rank-one covariance: ordinary Cholesky would fail on singular marginal
	# blocks. The explicit jitter strategy must keep every result finite.
	mu = numpy.array([0., 1., 2.], dtype=numpy.float32)
	v = numpy.array([1., 2., 3.], dtype=numpy.float32)
	singular = numpy.outer(v, v).astype(numpy.float32)

	d = Normal(mu, singular, covariance_type='full')
	X, mask = _data()
	logp = d.log_probability(_masked(X, mask))

	assert torch.isfinite(logp).all()
	assert logp[2].item() == 0.0  # fully missing row unaffected by jitter


def test_min_cov_floors_learned_covariance():
	d = Normal(covariance_type='full', min_cov=0.05)
	d.summarize(torch.randn(50, 3))
	d.from_summaries()

	eig_min = torch.linalg.eigvalsh(d.covs).min().item()
	assert eig_min >= 0.05 - 1e-6


def test_gmm_responsibilities_for_fully_missing_rows():
	mu0, cov0, d0 = _model()
	mu1 = [4.0, -1.0, 3.0]
	cov1 = [[1.20, 0.20, 0.10],
	        [0.20, 0.80, 0.00],
	        [0.10, 0.00, 1.50]]
	d1 = Normal(mu1, cov1, covariance_type='full')

	model = GeneralMixtureModel([d0, d1], priors=[0.7, 0.3], max_iter=5)
	X, mask = _data()

	proba = model.predict_proba(_masked(X, mask))

	# With an empty observation every component returns log prob zero, so the
	# posterior must collapse to the priors.
	assert_array_almost_equal(proba[2], [0.7, 0.3], 5)
	assert numpy.allclose(proba.sum(axis=1).numpy(), 1.0, atol=1e-6)


def test_gmm_fit_on_masked_data_stays_finite():
	torch.manual_seed(0)
	model = GeneralMixtureModel(
		[Normal(covariance_type='full'), Normal(covariance_type='full')],
		max_iter=10, tol=0.0, random_state=0)

	X, mask = _data()
	model.fit(_masked(X, mask))

	for component in model.distributions:
		assert torch.isfinite(component.means).all()
		assert torch.isfinite(component.covs).all()
	assert numpy.allclose(model.priors.numpy().sum(), 1.0)


def test_gmm_masked_fit_differs_from_zero_filled_fit():
	X, mask = _data()
	Xm = _masked(X, mask)
	X_zero = numpy.array(X, dtype=numpy.float32).copy()
	X_zero[~numpy.array(mask, dtype=bool)] = 0.0

	torch.manual_seed(0)
	model_masked = GeneralMixtureModel(
		[Normal(covariance_type='full'), Normal(covariance_type='full')],
		max_iter=10, tol=0.0, random_state=0).fit(Xm)

	torch.manual_seed(0)
	model_zero = GeneralMixtureModel(
		[Normal(covariance_type='full'), Normal(covariance_type='full')],
		max_iter=10, tol=0.0, random_state=0).fit(torch.tensor(X_zero))

	diff = max(
		(model_masked.distributions[k].means -
		 model_zero.distributions[k].means).abs().max().item()
		for k in range(2))
	assert diff > 1e-4

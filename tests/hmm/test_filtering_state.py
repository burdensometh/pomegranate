# test_filtering_state.py
# Tests for the resumable streaming filtering state shared by DenseHMM and
# SparseHMM. Both implementations must satisfy the exact same contract.

import numpy
import torch
import pytest

from pomegranate.hmm import DenseHMM
from pomegranate.hmm import SparseHMM
from pomegranate.hmm import FilteringState
from pomegranate.distributions import Exponential

from ..tools import assert_raises


def _make_dense():
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	edges = [[0.1, 0.8], [0.3, 0.6]]
	return DenseHMM(d, edges=edges, starts=[0.2, 0.8], ends=[0.1, 0.1])


def _make_sparse():
	d1 = Exponential([2.1, 0.3, 0.1])
	d2 = Exponential([1.5, 3.1, 2.2])

	model = SparseHMM()
	model.add_distributions([d1, d2])
	model.add_edge(model.start, d1, 0.2)
	model.add_edge(model.start, d2, 0.8)
	model.add_edge(d1, d1, 0.1)
	model.add_edge(d1, d2, 0.8)
	model.add_edge(d2, d1, 0.3)
	model.add_edge(d2, d2, 0.6)
	model.add_edge(d1, model.end, 0.1)
	model.add_edge(d2, model.end, 0.1)
	return model


@pytest.fixture
def X():
	torch.manual_seed(0)
	return torch.rand(3, 9, 3).abs() + 0.1


@pytest.fixture(params=["dense", "sparse"])
def model(request):
	if request.param == "dense":
		return _make_dense()
	return _make_sparse()


def _chunks(X, split):
	chunks, idx = [], 0
	for size in split:
		chunks.append(X[:, idx:idx + size])
		idx += size
	return chunks


def _stream(model, X, split, sequence_ids=None):
	chunks = _chunks(X, split)
	state = model.filtering_state(chunks[0], sequence_ids=sequence_ids)
	for block in chunks[1:]:
		model.filter(state, block, sequence_ids=sequence_ids)
	return state


def test_filtering_state_object(model, X):
	state = model.filtering_state(X[:, :2])

	assert isinstance(state, FilteringState)
	assert state.batch_size == 3
	assert len(state) == 3
	assert state.k == 2
	assert state.closed == False
	assert state.dtype == model.dtype
	assert state.device == model.device
	assert state.parameter_version == model._parameter_version


def test_alpha_is_normalized(model, X):
	state = _stream(model, X, (3, 3, 3))

	alpha = state.alpha
	assert alpha.shape == (3, 2)
	assert torch.all(alpha >= 0)
	assert torch.allclose(alpha.sum(dim=1), torch.ones(3), atol=1e-6)

	# log_alpha must be the log of alpha and itself be a log distribution
	assert torch.allclose(state.log_alpha, torch.log(alpha), atol=1e-6)
	assert torch.allclose(
		torch.logsumexp(state.log_alpha, dim=1), torch.zeros(3), atol=1e-6)


def test_chunk_equals_full_forward(model, X):
	full_forward = model.forward(X)
	full_logp = model.log_probability(X)

	for split in [(9,), (4, 5), (3, 3, 3), (2, 2, 2, 3), (1, 1, 7),
		(1, 2, 3, 1, 2)]:
		state = _stream(model, X, split)

		# Unnormalized forward log probs at the final time step must match the
		# single-shot forward algorithm, and the cumulative log likelihood
		# must equal logsumexp of that forward vector.
		assert torch.allclose(state.log_forward, full_forward[:, -1],
			atol=1e-5), split
		assert torch.allclose(state.log_likelihood,
			torch.logsumexp(full_forward[:, -1], dim=1), atol=1e-5), split

		# End probabilities are applied only when explicitly finishing.
		assert torch.allclose(model.finish_filtering(state), full_logp,
			atol=1e-5), split
		assert state.closed == True


def test_end_probabilities_not_applied_during_filtering(model, X):
	state = model.filtering_state(X[:, :4])
	prefix_forward = model.forward(X[:, :4])

	# Without finishing, the cumulative likelihood is the likelihood of the
	# observations only; finishing adds the end probabilities on top.
	observed_only = torch.logsumexp(prefix_forward[:, -1], dim=1)
	assert torch.allclose(state.log_likelihood, observed_only, atol=1e-6)

	with_end = observed_only + torch.logsumexp(
		prefix_forward[:, -1] - observed_only.unsqueeze(1) + model.ends, dim=1)
	assert torch.allclose(model.finish_filtering(state), with_end, atol=1e-6)


def test_no_repeated_starts(model, X):
	# A fresh state on each block would re-apply the start probabilities and
	# give a different (wrong) answer. Continuation must not do that.
	wrong = model.filtering_state(X[:, :3]).log_likelihood
	wrong = wrong + model.filtering_state(X[:, 3:6]).log_likelihood
	wrong = wrong + model.filtering_state(X[:, 6:]).log_likelihood

	state = _stream(model, X, (3, 3, 3))
	correct = model.finish_filtering(state)

	assert not torch.allclose(wrong, correct, atol=1e-3)
	assert torch.allclose(correct, model.log_probability(X), atol=1e-5)


def test_emissions_api(model, X):
	emissions = model._emission_matrix(X)

	state = model.filtering_state(emissions=emissions[:, :3])
	model.filter(state, emissions=emissions[:, 3:6])
	model.filter(state, emissions=emissions[:, 6:])

	assert torch.allclose(model.finish_filtering(state),
		model.log_probability(X), atol=1e-5)


def test_single_observation_blocks(model, X):
	state = model.filtering_state(X[:, :1])
	for i in range(1, X.shape[1]):
		model.filter(state, X[:, i:i + 1])

	assert torch.allclose(model.finish_filtering(state),
		model.log_probability(X), atol=1e-5)


def test_sequence_ids_roundtrip(model, X):
	ids = ("stream-a", "stream-b", "stream-c")
	state = model.filtering_state(X[:, :2], sequence_ids=ids)
	assert state.sequence_ids == ids

	model.filter(state, X[:, 2:4], sequence_ids=ids)
	model.filter(state, X[:, 4:], sequence_ids=ids)
	assert torch.allclose(model.finish_filtering(state),
		model.log_probability(X), atol=1e-5)

	# A single scalar id is wrapped into a one-element tuple.
	solo = model.filtering_state(X[:1, :2], sequence_ids="only")
	assert solo.sequence_ids == ("only",)
	model.filter(solo, X[:1, 2:4], sequence_ids="only")
	model.finish_filtering(solo)


def test_default_sequence_ids(model, X):
	state = model.filtering_state(X[:, :2])
	assert state.sequence_ids == (0, 1, 2)


def test_sequence_id_mismatch_raises(model, X):
	state = model.filtering_state(X[:, :2],
		sequence_ids=("a", "b", "c"))

	assert_raises(ValueError, model.filter, state, X[:, 2:4],
		sequence_ids=("a", "x", "c"))


def test_batch_size_mismatch_raises(model, X):
	state = model.filtering_state(X[:, :2])

	assert_raises(ValueError, model.filter, state, X[:2, 2:4])
	assert_raises(ValueError, model.filter, state, X[:1, 2:4])


def test_duplicate_sequence_ids_raises(model, X):
	assert_raises(ValueError, model.filtering_state, X[:, :2],
		sequence_ids=("a", "b", "a"))


def test_wrong_number_of_ids_raises(model, X):
	assert_raises(ValueError, model.filtering_state, X[:, :2],
		sequence_ids=("a", "b"))


def test_independent_streams_do_not_mix(model, X):
	# Two interleaved streams, advanced in alternation, must each produce the
	# same answer as if processed on their own.
	full = model.log_probability(X[:2])

	a = model.filtering_state(X[:1, :2], sequence_ids=("A",))
	b = model.filtering_state(X[1:2, :2], sequence_ids=("B",))

	for i in range(2, X.shape[1], 3):
		model.filter(a, X[:1, i:i + 3], sequence_ids=("A",))
		model.filter(b, X[1:2, i:i + 3], sequence_ids=("B",))

	logp_a = model.finish_filtering(a)
	logp_b = model.finish_filtering(b)

	assert torch.allclose(logp_a, full[:1], atol=1e-5)
	assert torch.allclose(logp_b, full[1:2], atol=1e-5)


def test_cross_model_state_raises(X):
	dense = _make_dense()
	sparse = _make_sparse()

	state = dense.filtering_state(X[:, :2])
	assert_raises(ValueError, sparse.filter, state, X[:, 2:4])


def test_non_state_object_raises(model, X):
	assert_raises(ValueError, model.filter, "not a state", X[:, :1])
	assert_raises(ValueError, model.finish_filtering, "not a state")
	assert_raises(ValueError, model.reset_filtering_state, "not a state")


def test_parameter_update_invalidates_state(model, X):
	state = model.filtering_state(X[:, :3])
	version_before = model._parameter_version

	model.summarize(X)
	model.from_summaries()

	assert model._parameter_version == version_before + 1
	assert_raises(ValueError, model.filter, state, X[:, 3:5])
	assert_raises(ValueError, model.finish_filtering, state)


def test_reset_filtering_state_rebinds(model, X):
	state = model.filtering_state(X[:, :3])
	model.summarize(X)
	model.from_summaries()

	assert_raises(ValueError, model.filter, state, X[:, 3:5])

	model.reset_filtering_state(state)
	assert state.parameter_version == model._parameter_version
	assert state.closed == False

	# After rebinding the state is usable again and the remaining suffix is
	# processed under the updated parameters without error.
	model.filter(state, X[:, 3:])
	assert torch.isfinite(model.finish_filtering(state)).all()


def test_dense_add_edge_bumps_version(X):
	model = _make_dense()
	state = model.filtering_state(X[:, :2])
	version_before = model._parameter_version

	d1, d2 = model.distributions
	model.add_edge(d1, d2, 0.5)

	assert model._parameter_version == version_before + 1
	assert_raises(ValueError, model.filter, state, X[:, 2:4])


def test_dtype_device_binding_raises(model, X):
	state = model.filtering_state(X[:, :2])
	model64 = model.double()

	assert_raises(ValueError, model64.filter, state, X[:, 2:4])

	model64.reset_filtering_state(state)
	assert state.dtype == torch.float64
	assert state.device == model64.device

	for i in range(2, X.shape[1], 2):
		model64.filter(state, X[:, i:i + 2])
	assert torch.allclose(model64.finish_filtering(state),
		model64.log_probability(X), atol=1e-10)


def test_closed_state_cannot_be_extended_or_reset(model, X):
	state = model.filtering_state(X[:, :2])
	model.finish_filtering(state)

	assert state.closed == True
	assert_raises(ValueError, model.filter, state, X[:, 2:4])
	assert_raises(ValueError, model.finish_filtering, state)
	assert_raises(ValueError, model.reset_filtering_state, state)


def test_long_sequence_numerical_stability(model):
	torch.manual_seed(1)
	X = torch.rand(2, 2000, 3) + 0.05

	full_forward = model.forward(X)
	full_logp = model.log_probability(X)

	state = model.filtering_state(X[:, :100])
	for i in range(100, X.shape[1], 100):
		model.filter(state, X[:, i:i + 100])

	assert torch.isfinite(state.alpha).all()
	assert torch.isfinite(state.log_likelihood).all()
	assert torch.allclose(state.log_forward, full_forward[:, -1], atol=1e-4)
	assert torch.allclose(model.finish_filtering(state), full_logp, atol=1e-4)


def test_dense_and_sparse_share_contract(X):
	dense = _make_dense()
	sparse = _make_sparse()

	# Same parameters, same chunking: both contracts must agree with each
	# other and with the single-shot log probability.
	split = (2, 4, 3)
	dense_state = _stream(dense, X, split)
	sparse_state = _stream(sparse, X, split)

	assert torch.allclose(dense_state.alpha, sparse_state.alpha, atol=1e-5)
	assert torch.allclose(dense_state.log_likelihood,
		sparse_state.log_likelihood, atol=1e-5)
	assert torch.allclose(dense.finish_filtering(dense_state),
		sparse.finish_filtering(sparse_state), atol=1e-5)
	assert torch.allclose(dense.log_probability(X),
		sparse.log_probability(X), atol=1e-5)

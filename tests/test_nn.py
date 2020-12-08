import minitorch
from hypothesis import given
from .strategies import tensors, assert_close
import pytest
import numpy as np


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t):
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0],
        sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0],
        sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0],
        sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t):
    t_np = t.to_numpy()

    compare_0 = t_np.max(axis=0)
    out_0 = minitorch.Max.apply(t, 0)
    assert out_0.shape == (1, 3, 4)
    assert np.array_equal(compare_0.reshape(-1), out_0.to_numpy().reshape(-1))
    minitorch.grad_check(lambda t: minitorch.Max.apply(t, 0), t + (minitorch.rand(t.shape) * 1e-5))

    compare_1 = t_np.max(axis=1)
    out_1 = minitorch.Max.apply(t, 1)
    assert out_1.shape == (2, 1, 4)
    assert np.array_equal(compare_1.reshape(-1), out_1.to_numpy().reshape(-1))
    minitorch.grad_check(lambda t: minitorch.Max.apply(t, 1), t + (minitorch.rand(t.shape) * 1e-5))

    compare_2 = t_np.max(axis=2)
    out_2 = minitorch.Max.apply(t, 2)
    assert out_2.shape == (2, 3, 1)
    assert np.array_equal(compare_2.reshape(-1), out_2.to_numpy().reshape(-1))
    minitorch.grad_check(lambda t: minitorch.Max.apply(t, 0), t + (minitorch.rand(t.shape) * 1e-5))


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_log_softmax(t):
    q = minitorch.softmax(t, 2)
    logmax_q = minitorch.logsoftmax(t, 2).exp()
    assert_close(logmax_q[0, 0, 0], q[0, 0, 0])


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t):
    out = minitorch.maxpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t):
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t):
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)

import tensorly as tl
from tensorly.tt_tensor import TTTensor
from tensorly.random import random_tt
from tensorly.testing import assert_array_almost_equal
from torch import int as pt_int

from ..tt_state import spins_to_tt_state, tt_norm
from ..tt_sum import tt_sum


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>


err_tol = 5 #decimals precision


def test_spins_to_tt_tensor():
    nqubits = 3
    perm = tl.sign(tl.randn((nqubits, 1))).type(pt_int)
    perm[perm == -1] = 0
    state = TTTensor(spins_to_tt_state(perm)).to_tensor().reshape(-1,1)
    def assign_qubit(ind):
        if not ind:
            return tl.tensor([1, 0])
        else:
            return tl.tensor([0, 1])
    manual_state = assign_qubit(perm[0])
    for ind in perm[1::]:
        manual_state = tl.kron(manual_state, assign_qubit(ind))
    assert_array_almost_equal(state, manual_state.reshape(-1, 1), decimal=err_tol)


def test_tt_norm():
    nqubits = 4
    nlayers = 1
    dims = tuple([2 for i in range(nqubits)])
    rank = [1] + [4 for i in range(nqubits-1)] + [1]
    state0 = random_tt(dims, rank=rank)
    state0.factors = [1j*factor for factor in state0.factors]
    state = tt_sum(random_tt(dims, rank=rank), state0)
    dense_state = state.to_tensor().reshape(-1,1)
    assert_array_almost_equal(tt_norm(state), tl.norm(dense_state), decimal=err_tol)

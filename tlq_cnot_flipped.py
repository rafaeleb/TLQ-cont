import tensorly as tl
tl.set_backend('pytorch')
from torch import randn, cos, sin, float32, complex64, exp
from torch.nn import Module, ModuleList, ParameterList, Parameter
from tensorly.tt_matrix import TTMatrix
from copy import deepcopy
from itertools import chain

from .tt_operators import identity
from .tt_precontraction import qubits_contract, _get_contrsets
from .tt_sum import tt_matrix_sum

class CNOT_flippedL(Module):

    # Left core of a flipped CNOT gate (right qubit control, left qubit target)
    
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        core, self.dtype, self.device = tl.zeros((1,2,2,2), dtype=dtype, device=device), dtype, device
        core[0,0,0,0] = 1.
        core[0,0,1,1] = 1.
        core[0,1,0,1] = 1.
        core[0,1,1,0] = 1.        
        self.core = core

    def forward(self):
        return self.core

    def reinitialize(self):
        pass

class CNOT_flippedR(Module):

    # Right core of a flipped CNOT gate (right qubit control, left qubit target)

    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        core, self.dtype, self.device = tl.zeros((2,2,2,1), dtype=dtype, device=device), dtype, device
        core[0,0,0,0] = 1.
        core[1,1,1,0] = 1.
        self.core =  core

    def forward(self):
        return self.core

    def reinitialize(self):
        pass
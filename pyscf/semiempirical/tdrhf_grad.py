
#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yu Zhang <zhy@lanl.gov>
#
# Ref:
# J. Chem. Phys. xx
#


from functools import reduce
import numpy
from pyscf      import lib
from pyscf.lib  import logger
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.scf  import cphf
from pyscf      import __config__


def sqm_grad_elec(td_grad, x_y, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO):
    '''
    Electronic part of TDA, TDHF nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    '''

    # SQM excited states gradients (TODO)

    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()
    
    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    x, y = x_y
    xpy = (x+y).reshape(nocc,nvir).T
    xmy = (x-y).reshape(nocc,nvir).T
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]
    
    # TODO


    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))

    return de


class Gradients(tdrhf_grad.Gradients):

    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20)
    cphf_conv_tol  = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)

    def __init__(self, td):
        assert isinstance(td, qed.tdscf.rhf.TDMixin)
        self.verbose    = td.verbose
        self.stdout     = td.stdout
        self.mol        = td.mol
        self.base       = td
        self.chkfile    = td.chkfile
        self.max_memory = td.max_memory
        self.state      = 1  # of which the gradients to be computed.
        self.atmlst     = None
        self.de         = None
        keys = set(('cphf_max_cycle', 'cphf_conv_tol'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s-%s ********', self.base.td_obj.__class__, self.base.cav_obj.__class__)
        log.info("QED-TDDFT analytical gradient: %s", "Yang, et. al. https://doi.org/10.26434/chemrxiv-2021-lf5m2")
        log.info('cphf_conv_tol  = %g', self.cphf_conv_tol)
        log.info('cphf_max_cycle = %d', self.cphf_max_cycle)
        log.info('chkfile        = %s', self.chkfile)
        log.info('State ID       = %d', self.state)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('\n')
        return self


    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, mn, singlet, atmlst=None):
        return grad_elec(self, xy, singlet, atmlst, self.max_memory, self.verbose)


    def kernel(self, xy=None, state=None, singlet=None, atmlst=None):
        '''
        Args:
            state : int
                Excited state ID.  state = 1 means the first excited state.
        '''
        if xy is None:






import numpy as np
from utils.io_corr import *
import time
from typing import List
from opt_einsum import contract

from pyquda import init, LatticeInfo
from pyquda.field import LatticeFermion, LatticePropagator, LatticeGauge
from pyquda_utils import core, gpt, gamma, phase
from pyquda.dirac import GaugeDirac


'''---------------------------------------------------------------'''
'''         Fermion flow for Pyquda LatticeFermion fields         '''
'''---------------------------------------------------------------'''

def propagator_laplacian_explicit(U: LatticeGauge, prop_in: LatticePropagator) -> LatticePropagator:
    """
    4D gauge-covariant Laplacian acting on the SINK leg of a propagator S(y|x):

      (ΔS)(x) = Σ_μ [ U_μ(x) S(x+μ) - 2 S(x) + U_μ†(x-μ) S(x-μ) ].

    Assumptions:
      - prop_in.data shape: (2, Lt, Lz, Ly, Lx//2, Ns_sink, Ns_src, Nc_sink, Nc_src)
      - prop_in.shift(±1, mu) shifts the SINK coordinates (t,z,y,xh)
      - U[mu] has shape (..., Nc, Nc), and U[mu].shift(-1, mu) is U_μ(x-μ)
      - Works with numpy / cupy via opt_einsum.
    """
    out = LatticePropagator(prop_in.latt_info)
    out.data[...] = 0

    for mu in range(4):
        # forward term: U_mu(x) · S(x+μ)
        S_f = prop_in.shift(+1, mu)           # S(x+μ)
        Uf  = U[mu].data                      # U_μ(x)   (..., Nc, Nc)
        term_f = contract('...ab,...sSbB->...sSaB', Uf, S_f.data)
        # indices: a,b  sink colors (contract b with Nc_sink)
        #          s,S  sink/source spins
        #          B    source color (left untouched)

        # backward term: U_mu†(x-μ) · S(x-μ)
        S_b = prop_in.shift(-1, mu)           # S(x-μ)
        Ub  = U[mu].shift(-1, mu).data        # U_μ(x-μ)
        Ub_d = Ub.swapaxes(-2, -1).conjugate()  # U_μ†(x-μ)
        term_b = contract('...ab,...sSbB->...sSaB', Ub_d, S_b.data)

        # accumulate stencil
        out.data[...] += term_f + term_b - 2.0 * prop_in.data

    return out


def flow_step_on_sink_prop_fixedU(U: LatticeGauge,
                                  prop: LatticePropagator,
                                  epsilon: float) -> LatticePropagator:
    """
    One fermion-flow step at fixed background gauge U(t):
      dχ/dt = Δ_cov[U(t)] χ(t)
    using the 3-stage coefficients analogous to Grid FermionFlow,
    but *without* updating U inside this function.
    """

    # φ0 = Δ[U] χ
    Phi0 = propagator_laplacian_explicit(U, prop)

    # φ1 = χ + 1/4 ε φ0
    Phi1 = LatticePropagator(prop.latt_info)
    Phi1.data[...] = prop.data + 0.25 * epsilon * Phi0.data

    # φ2 = χ + 8/9 ε Δ[U] φ1 - 2/9 ε φ0
    lap_Phi1 = propagator_laplacian_explicit(U, Phi1)
    Phi2 = LatticePropagator(prop.latt_info)
    Phi2.data[...] = prop.data + (8.0 / 9.0) * epsilon * lap_Phi1.data - (2.0 / 9.0) * epsilon * Phi0.data

    # χ(t+ε) = φ1 + 3/4 ε Δ[U] φ2
    lap_Phi2 = propagator_laplacian_explicit(U, Phi2)
    prop_new = LatticePropagator(prop.latt_info)
    prop_new.data[...] = Phi1.data + (3.0 / 4.0) * epsilon * lap_Phi2.data

    return prop_new

def flow_propagator_on_sink(U_t: LatticeGauge, prop_t: LatticePropagator, epsilon: float, Nstep: int):
    """
    返回每个 flow time 的 propagator。
    """
    for step in range(Nstep):

        t0 = time.time()
        energy = U_t.wilsonFlow(
        n_steps=1,
        epsilon=epsilon
        )

        prop_t = flow_step_on_sink_prop_fixedU(U_t, prop_t, epsilon)
        t1 = time.time()
        g.message(f"  Step {step+1}/{Nstep}: time={t1 - t0:.3f} s, Wilson flow energy={energy}")

    return U_t, prop_t
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gpt as g
import numpy as np
from opt_einsum import contract
import cupy as cp

from pyquda import getMPIComm
from pyquda.field import LatticeGauge, LatticePropagator, MultiLatticeFermion
from pyquda_utils import core, gpt, gamma, convert

GEN_SIMD_WIDTH = 64

D_gammas = [
    cp.asarray(gamma.gamma(1)),
    cp.asarray(gamma.gamma(2)),
    cp.asarray(gamma.gamma(4)),
    cp.asarray(gamma.gamma(8)),
]

# Please confirm this is gamma5 in your convention
G5 = cp.asarray(gamma.gamma(15))


def get_invD(U, invPara):
    mf, csw, prec, cgMax = invPara
    inv = g.algorithms.inverter
    pc = g.qcd.fermion.preconditioner
    cg = inv.bicgstab({"eps": prec, "maxiter": cgMax})
    w = g.qcd.fermion.wilson_clover(
        U,
        mass=mf,
        csw_r=csw,
        csw_t=csw,
        nu=1.0,
        xi_0=1.0,
        isAnisotropic=False,
        boundary_phases=[1, 1, 1, -1],
    )
    g.message("before invD")
    invD = w.propagator(inv.preconditioned(pc.eo1(), cg)).grouped(1)
    return invD, w


def _covdev_sym_prop(U_f: LatticeGauge, prop: LatticePropagator, mu: int):
    """
    Symmetric covariant derivative on propagator:
        0.5 * (D_{+mu} - D_{-mu})

    Do it column by column in MultiLatticeFermion space, then convert back
    to propagator.
    """
    U_f.gauge_dirac.loadGauge(U_f)

    mf = convert.propagatorToMultiFermion(prop)
    mf_covdev = convert.propagatorToMultiFermion(prop)

    for spin in range(4):
        for color in range(3):
            idx = spin * 3 + color
            Dp = U_f.pure_gauge.covDev(mf[idx], mu)
            Dm = U_f.pure_gauge.covDev(mf[idx], mu + 4)
            mf_covdev[idx] = 0.5 * (Dp - Dm)

    return convert.multiFermionToPropagator(mf_covdev)


def _left_covdev_dst2_from_dsty(U_f: LatticeGauge, dst_y: LatticePropagator, mu: int):
    """
    Construct left-acting covariant derivative on
        dst2 = gamma5 * adj(dst_y) * gamma5

    using
        leftD(dst2) = gamma5 * adj(D dst_y) * gamma5
    """
    D_y = _covdev_sym_prop(U_f, dst_y, mu)
    D_y_dag = D_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7)
    leftD_dst2 = contract("ab,...bcij,cd->...adij", G5, D_y_dag, G5)
    return leftD_dst2


def _flow_two_props_pyquda(
    U_f: LatticeGauge,
    dst_y: LatticePropagator,
    dst_seq: LatticePropagator,
    stepsize: float,
    flow_type: str = "wilson",
):
    """
    Flow two propagators together using PyQUDA gradientFlow, while U_f
    is updated in place.
    """
    mf_y = convert.propagatorToMultiFermion(dst_y)
    mf_seq = convert.propagatorToMultiFermion(dst_seq)

    L5_y = mf_y.L5
    L5_seq = mf_seq.L5
    assert L5_y == L5_seq

    packed = MultiLatticeFermion(
        U_f.latt_info,
        L5_y + L5_seq,
        cp.concatenate([mf_y.data, mf_seq.data], axis=0),
    )

    packed_flow = U_f.gradientFlow(packed, flow_type, 1, stepsize)

    mf_y_flow = MultiLatticeFermion(
        U_f.latt_info,
        L5_y,
        packed_flow.data[:L5_y].copy(),
    )
    mf_seq_flow = MultiLatticeFermion(
        U_f.latt_info,
        L5_seq,
        packed_flow.data[L5_y:L5_y + L5_seq].copy(),
    )

    dst_y_flow = convert.multiFermionToPropagator(mf_y_flow)
    dst_seq_flow = convert.multiFermionToPropagator(mf_seq_flow)

    return dst_y_flow, dst_seq_flow


def get_C3_chi_pyquda(
    U_f: LatticeGauge,
    dst_y: LatticePropagator,
    dst_seq: LatticePropagator,
    t0: int,
):
    """
    C3_chi(t) = Tr[ dst2 * dst_seq ]
    with dst2 = gamma5 * adj(dst_y) * gamma5
    Local contraction reduces wtzyx+spin+color to t,
    then core.gatherLattice handles MPI combination.
    """
    dst2 = contract(
        "ab,wtzyxbcij,cd->wtzyxadij",
        G5,
        dst_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7),
        G5,
    )

    scalar_t = contract("wtzyxabij,wtzyxbaji->t", dst2, dst_seq.data)

    slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
    slice_t = getMPIComm().bcast(slice_t, root=0)

    return np.roll(np.array(slice_t.real), -t0)

def get_C3_Tmunu_symmetrized_pyquda(
    U_f: LatticeGauge,
    dst_y: LatticePropagator,
    dst_seq: LatticePropagator,
    t0: int,
):
    Nt = U_f.latt_info.global_size[3]
    C3_Tmunu = np.zeros((4, 4, Nt), dtype=np.float64)

    # dst2 = gamma5 * adj(dst_y) * gamma5
    # dst_y.data: wtzyxabij
    # adj(dst_y): wtzyxbaji
    # dst2:       wtzyxadij
    dst2 = contract(
        "ab,wtzyxbcij,cd->wtzyxadij",
        G5,
        dst_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7),
        G5,
    )

    # first term: +1/2 Tr[ dst2 * gamma_nu * D_mu(dst_seq) ]
    for mu in range(4):
        D_seq = _covdev_sym_prop(U_f, dst_seq, mu)   # now guaranteed propagator-shaped

        for nu in range(4):
            # gamma_D_seq: wtzyxadij
            gamma_D_seq = contract(
                "ab,wtzyxbdij->wtzyxadij",
                D_gammas[nu],
                D_seq.data,
            )

            # trace over a,d,i,j and local sum over w,z,y,x -> keep t
            scalar_t = 0.5 * contract(
                "wtzyxadij,wtzyxdaji->t",
                dst2,
                gamma_D_seq,
            )

            slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
            slice_t = getMPIComm().bcast(slice_t, root=0)
            C3_Tmunu[mu, nu] += np.roll(np.array(slice_t.real), -t0)

    # second term: -1/2 Tr[ leftD_mu(dst2) * gamma_nu * dst_seq ]
    for mu in range(4):
        leftD_dst2 = _left_covdev_dst2_from_dsty(U_f, dst_y, mu)   # wtzyxadij

        for nu in range(4):
            gamma_dst_seq = contract(
                "ab,wtzyxbdij->wtzyxadij",
                D_gammas[nu],
                dst_seq.data,
            )

            scalar_t = -0.5 * contract(
                "wtzyxadij,wtzyxdaji->t",
                leftD_dst2,
                gamma_dst_seq,
            )

            slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
            slice_t = getMPIComm().bcast(slice_t, root=0)
            C3_Tmunu[mu, nu] += np.roll(np.array(slice_t.real), -t0)

    for mu in range(4):
        for nu in range(mu + 1, 4):
            C3_Tmunu[mu, nu] = 0.5 * (C3_Tmunu[mu, nu] + C3_Tmunu[nu, mu])
            C3_Tmunu[nu, mu] = C3_Tmunu[mu, nu]

    return C3_Tmunu


def C3_con_EMT_pyquda(
    gaugePara,
    invPara,
    flowPara,
    smearPara,
    Nsrc,
    sinkt_range,
    spin,
    datfile,
):
    """
    Hybrid migration version of quark EMT 3pt.

    - source/smear/inversion/sequential source remain in GPT
    - gauge + propagator gradient flow is done by PyQUDA
    - C3_chi and C3_Tmunu contractions are done in PyQUDA style
    - NO flattening of spin/color indices
    """
    assert spin in [0, 1, 2, 5]
    N_sinkt = len(sinkt_range)

    a, conf_id, U = gaugePara
    stepsize, Nsteps, improve, division = flowPara
    to_sm_src, to_sm_dst, sm_sigma, sm_steps = smearPara

    if improve:
        raise NotImplementedError(
            "Current PyQUDA flow path here only implements Wilson flow; improve=True not yet wired."
        )

    grid = U[0].grid
    Nx = grid.gdimensions[0]
    Ny = grid.gdimensions[1]
    Nz = grid.gdimensions[2]
    Nt = grid.gdimensions[3]

    invD = get_invD(U, invPara)[0]

    g.mem_report()

    C2 = np.zeros((Nsrc, Nt), dtype=np.float64)
    C3_chi = np.zeros((Nsrc, N_sinkt, Nsteps + 1, Nt), dtype=np.float64)
    C3_Tmunu = np.zeros((Nsrc, N_sinkt, Nsteps + 1, 4, 4, Nt), dtype=np.float64)
    src_locs = np.zeros((Nsrc, 4), dtype=np.uint16)

    for n_src in range(Nsrc):
        x0 = int(n_src % Nx)
        y0 = int(n_src % Ny)
        z0 = int(n_src % Nz)
        t0 = int(n_src % Nt)

        g.message("src", str(n_src), f"[{x0},{y0},{z0},{t0}]")
        src_locs[n_src] = np.array([x0, y0, z0, t0], dtype=np.uint16)

        src = g.mspincolor(grid)
        g.create.point(src, [x0, y0, z0, t0])

        if to_sm_src:
            smear = g.create.smear.gauss(U, sigma=sm_sigma, steps=sm_steps, dimensions=[0, 1, 2])
            g.message("source smearing starts")
            src = g(smear * src)
            g.message("source smearing ends")
            del smear

        dst_x = g(invD * src)
        del src

        dst_y_back = g.copy(dst_x)

        if to_sm_dst:
            smear = g.create.smear.gauss(U, sigma=sm_sigma, steps=sm_steps, dimensions=[0, 1, 2])
            g.message("first sink smearing starts")
            dst_x = g(smear * dst_x)

        tmp = g(g.gamma[5] * g.adj(dst_x) * g.gamma[5])
        C2[n_src] += np.roll(
            np.array(g.slice(g.trace(g.gamma[spin] * tmp * g.gamma[spin] * dst_x), 3)).real,
            -t0,
        )
        del tmp

        if to_sm_dst:
            g.message("second sink smearing starts")
            dst_x = g(smear * dst_x)
            g.message("sink smearings end")
            del smear

        for n_t, sink_t in enumerate(sinkt_range):
            g.message("create sequential source sink_t =", sink_t)

            src_seq = g.mspincolor(grid)
            src_seq[:] = 0
            src_seq[:, :, :, (sink_t + t0) % Nt] = dst_x[:, :, :, (sink_t + t0) % Nt]
            src_seq = g(g.gamma[spin] * src_seq * g.gamma[spin])

            dst_seq_gpt = g(invD * src_seq)
            del src_seq

            dst_y_gpt = g.copy(dst_y_back)
            if n_t == len(sinkt_range) - 1:
                del dst_x, dst_y_back

            Uap_gpt = g.copy(U)
            Uap_gpt[3][:, :, :, Nt - 1] *= -1

            U_f = gpt.LatticeGaugeGPT(Uap_gpt, GEN_SIMD_WIDTH)
            dst_y_py = gpt.LatticePropagatorGPT(dst_y_gpt, GEN_SIMD_WIDTH)
            dst_seq_py = gpt.LatticePropagatorGPT(dst_seq_gpt, GEN_SIMD_WIDTH)

            del Uap_gpt, dst_y_gpt, dst_seq_gpt

            g.mem_report()

            for step in range(Nsteps + 1):
                g.message("contraction for step", step)

                C3_chi[n_src, n_t, step] += get_C3_chi_pyquda(
                    U_f, dst_y_py, dst_seq_py, t0
                )

                C3_Tmunu[n_src, n_t, step] += get_C3_Tmunu_symmetrized_pyquda(
                    U_f, dst_y_py, dst_seq_py, t0
                )

                if step < Nsteps:
                    dst_y_py, dst_seq_py = _flow_two_props_pyquda(
                        U_f,
                        dst_y_py,
                        dst_seq_py,
                        stepsize,
                        flow_type="wilson",
                    )

            del U_f, dst_y_py, dst_seq_py

    np.save(f"{datfile}/C2_spin{spin}_HYP_SS_persrc.pyquda.npy", C2)
    np.save(f"{datfile}/C3_chi_spin{spin}_HYP_SS_persrc.pyquda.npy", C3_chi)
    np.save(f"{datfile}/C3_Tmunu_spin{spin}_HYP_SS_persrc.pyquda.npy", C3_Tmunu)
    np.save(f"{datfile}/src_locs_spin{spin}_HYP_SS.pyquda.npy", src_locs)

    C2 = np.mean(C2, axis=0)
    C3_chi = np.mean(C3_chi, axis=0)
    C3_Tmunu = np.mean(C3_Tmunu, axis=0)

    np.save(f"{datfile}/C2_spin{spin}_HYP_SS.pyquda.npy", C2)
    for n_t, sink_t in enumerate(sinkt_range):
        np.save(f"{datfile}/C3_chi_sinkt{sink_t}_spin{spin}_HYP_SS.pyquda.npy", C3_chi[n_t])
        np.save(f"{datfile}/C3_Tmunu_sinkt{sink_t}_spin{spin}_HYP_SS.pyquda.npy", C3_Tmunu[n_t])



def get_C3_Tmunu_symmetrized(Uap, dst2, dst_seq, t0):
    Nt = dst2.grid.gdimensions[3]
    C3_Tmunu = np.zeros((4,4,Nt),dtype=np.float64)

    for mu in range(4):
        tmp = 0.5*g( Uap[mu]*g.cshift(dst_seq,mu,1) - g.cshift( g.adj(Uap[mu])*dst_seq , mu,-1) )
        for nu in range(4):
            C3_Tmunu[mu][nu] += 0.5*np.roll( np.array( g.slice( g.trace(dst2*g.gamma[nu]*tmp) , 3 ) ).real ,-t0)

        tmp = 0.5*g( g.cshift(dst2,mu,1)*g.adj(Uap[mu]) - g.cshift( dst2*Uap[mu] ,mu,-1) )
        for nu in range(4):
            C3_Tmunu[mu][nu] -= 0.5*np.roll( np.array( g.slice( g.trace(tmp*g.gamma[nu]*dst_seq) , 3 ) ).real ,-t0)

    for mu in range(4):
        for nu in range(mu+1,4):
            C3_Tmunu[mu][nu] = ( C3_Tmunu[mu][nu] + C3_Tmunu[nu][mu] ) * 0.5
            C3_Tmunu[nu][mu] = C3_Tmunu[mu][nu]

    return C3_Tmunu



def C3_con_EMT(gaugePara, invPara, flowPara, smearPara, Nsrc, sinkt_range, spin, datfile):
    assert spin in [0,1,2,5] # 0,1,2 for Jpsi spin x,y,z; 5 for etac
    N_sinkt = len(sinkt_range)

    a, conf_id, U = gaugePara
    stepsize, Nsteps, improve, division = flowPara
    to_sm_src, to_sm_dst, sm_sigma, sm_steps = smearPara

    grid = U[0].grid
    L = np.array(grid.fdimensions)
    Nx = grid.gdimensions[0]
    Ny = grid.gdimensions[1]
    Nz = grid.gdimensions[2]
    Nt = grid.gdimensions[3]

    invD = get_invD(U, invPara)[0]
    #invD,w = get_invD(U, invPara)

    #check_repeat = np.array([ [ [ [ False for t0 in range(Nt) ] for z0 in range(Nz) ] for y0 in range(Ny) ] for x0 in range(Nx) ])
    g.mem_report()

    C2 = np.zeros((Nsrc,Nt),dtype=np.float64)
    C3_chi = np.zeros((Nsrc,N_sinkt,Nsteps+1,Nt),dtype=np.float64)
    C3_Tmunu = np.zeros((Nsrc,N_sinkt,Nsteps+1,4,4,Nt),dtype=np.float64)
    src_locs = np.zeros((Nsrc,4),dtype=np.uint16)
    for n_src in range(Nsrc):
        x0,y0,z0,t0 = int(n_src % Nx),int(n_src % Ny),int(n_src % Nz),int(n_src % Nt)
        g.message('src', str(n_src), f'[{x0},{y0},{z0},{t0}]')
        src_locs[n_src] = np.array([x0,y0,z0,t0],dtype=np.uint16)

        src = g.mspincolor(grid)
        g.create.point(src, [x0,y0,z0,t0])
        if to_sm_src:
            smear = g.create.smear.gauss(U, sigma=sm_sigma, steps=sm_steps, dimensions=[0,1,2])
            g.message('source smearing starts')
            src = g( smear * src )
            g.message('source smearing ends')
            del smear

        dst_x = g( invD * src )
        # r = g( w*dst_x-src )
        # g.message( '|r_true| = ', np.sqrt(g.sum(g.trace(g.adj(r)*r)).real) )
        del src
        dst_y_back = g.copy(dst_x)

        if to_sm_dst:
            smear = g.create.smear.gauss(U, sigma=sm_sigma, steps=sm_steps, dimensions=[0,1,2])
            g.message('first sink smearing starts')
            dst_x = g( smear * dst_x )

        tmp = g( g.gamma[5]*g.adj(dst_x)*g.gamma[5] )
        C2[n_src] += np.roll( np.array( g.slice( g.trace( g.gamma[spin] * tmp * g.gamma[spin] * dst_x ) , 3 ) ).real, -t0)
        del tmp

        if to_sm_dst:
            g.message('second sink smearing starts')
            dst_x = g( smear * dst_x )
            g.message('sink smearings end')
            del smear

        for n_t,sink_t in enumerate(sinkt_range):
            g.message('create sequential source sink_t =',sink_t)
            src_seq = g.mspincolor(grid)
            src_seq[:] = 0
            src_seq[:, :, :, (sink_t+t0)%Nt] = dst_x[:, :, :, (sink_t+t0)%Nt]
            src_seq = g( g.gamma[spin]*src_seq*g.gamma[spin] )
            dst_seq = g( invD*src_seq ) # dst_seq(y;x,x0)
            # r = g( w*dst_seq-src_seq )
            # g.message( '|r_true| = ', np.sqrt(g.sum(g.trace(g.adj(r)*r)).real) )
            del src_seq

            Uap = g.copy(U)
            Uap[3][:,:,:,Nt-1] *= -1
            dst_y = g.copy(dst_y_back)
            if n_t == len(sinkt_range)-1:
                del dst_x,dst_y_back
            g.mem_report()
            for step in range(Nsteps+1):
                g.message('contraction for step',step)
                dst2 = g(g.gamma[5]*g.adj(dst_y)*g.gamma[5])
                C3_chi[n_src,n_t,step] += np.roll( np.array( g.slice( g.trace(dst2 * dst_seq) , 3 ) ).real, -t0)
                C3_Tmunu[n_src,n_t,step] += get_C3_Tmunu_symmetrized(Uap,dst2,dst_seq,t0)
                del dst2

                dst_y = g.qcd.fermion.flow.Fermionflow_fixedstepsize(Uap, dst_y, stepsize, 1, Ncheckpoints=0, improvement=improve)[1]
                Uap, dst_seq = g.qcd.fermion.flow.Fermionflow_fixedstepsize(Uap, dst_seq, stepsize, 1, Ncheckpoints=0, improvement=improve)

            del Uap,dst_seq,dst_y

    np.save(f'{datfile}/C2_spin{spin}_HYP_SS_persrc.GPT.npy', C2)
    np.save(f'{datfile}/C3_chi_spin{spin}_HYP_SS_persrc.GPT.npy', C3_chi)
    np.save(f'{datfile}/C3_Tmunu_spin{spin}_HYP_SS_persrc.GPT.npy', C3_Tmunu)
    np.save(f'{datfile}/src_locs_spin{spin}_HYP_SS.GPT.npy', src_locs)

    C2 = np.mean( C2, axis=0 )
    C3_chi = np.mean( C3_chi, axis=0 )
    C3_Tmunu = np.mean( C3_Tmunu, axis=0 )

    np.save(f'{datfile}/C2_spin{spin}_HYP_SS.GPT.npy', C2)
    for n_t,sink_t in enumerate(sinkt_range):
        np.save(f'{datfile}/C3_chi_sinkt{sink_t}_spin{spin}_HYP_SS.GPT.npy', C3_chi[n_t])
        np.save(f'{datfile}/C3_Tmunu_sinkt{sink_t}_spin{spin}_HYP_SS.GPT.npy', C3_Tmunu[n_t])

import gpt as g
import numpy as np
from opt_einsum import contract

# load pyquda modules
from pyquda import getMPIComm
from pyquda.field import LatticeInfo, LatticeGauge, LatticePropagator, LatticeFermion
from pyquda_utils import core, gpt, gamma, source, phase, convert
from pyquda_comm.array import arrayIdentity, arrayZeros
import subprocess

GEN_SIMD_WIDTH = 64


def _impose_P_Breit_slice(U: LatticeGauge, complex_field, n_max, realize=False):
    g.message(f'impose_P_Breit_slice n_max = {n_max}')
    Nt = U.latt_info.global_size[3]
    results = np.zeros((n_max+1,n_max+1,n_max+1,Nt),dtype=np.complex128)
    for nx in range(n_max+1):
        for ny in range(n_max+1):
            for nz in range(n_max+1):
                qext_xyz = [[2 * nx, 2 * ny, 2 * nz]]
                phases_3pt = phase.MomentumPhase(U.latt_info).getPhases(qext_xyz, [0,0,0,0])
                slice_t = core.gatherLattice(contract("qwtzyx, wtzyx -> qt", phases_3pt, complex_field).get(), [1, -1, -1, -1])
                slice_t = getMPIComm().bcast(slice_t, root=0)
                results[nx,ny,nz] += slice_t[0]
    return results



def _get_Tmunu_symmetrized_P_Breit_slice(U_f: LatticeGauge, xi: LatticePropagator, eta: LatticePropagator, n_max: int):
    Nt = U_f[0].grid.gdimensions[3]

    # s term
    CHI = np.zeros([2,n_max+1,n_max+1,n_max+1,Nt], dtype=np.complex128)
    
    dot_xi_eta = contract('etzyxbc,etzyxbc->etzyx', xi.data.conj(), eta.data).real #! real to be removed
    CHI[0] = _impose_P_Breit_slice(U_f, dot_xi_eta, n_max, realize=True)
    dot_xi_xi  = contract('etzyxbc,etzyxbc->etzyx', xi.data.conj(), xi.data).real #! real to be removed
    CHI[1] = _impose_P_Breit_slice(U_f, dot_xi_xi, n_max, realize=True)

    # t term
    Tmunu = np.zeros([4,4,n_max+1,n_max+1,n_max+1,Nt], dtype=np.complex128)
    for mu in range(4):
        tmp = g( U_f[mu]*g.cshift(eta,mu,1) - g.cshift( g.adj(U_f[mu])*eta , mu,-1) )
        for nu in range(4):
            complex_field = g( g.adj(xi)*g.gamma[nu]*tmp )
            Tmunu[nu,mu] += -0.5*_impose_P_Breit_slice(U_f, complex_field, n_max, realize=True)

    # symmetrization
    for mu in range(4):
        for nu in range(mu+1,4):
            Tmunu[mu,nu] = ( Tmunu[mu,nu] + Tmunu[nu,mu] ) / 2
            Tmunu[nu,mu] = Tmunu[mu,nu]

    return Tmunu, CHI

def flowed_fermionic_EMT_pyquda(
    U: LatticeGauge, 
    dirac, 
    randPara,
    stepsize: float = 0.1,
    Nsteps: int = 20,
    datfile: str = "",
    n_max: int = 0,
    improve: bool = False
):

    Nv, n_input, randseed = randPara

    # --- 几何信息，从 PyQUDA 的 LatticeGauge 中取 ---
    global_size = U.latt_info.global_size
    Ns3 = global_size[0] * global_size[1] * global_size[2]
    Nt = global_size[3]

    #xi = source.fermion(U.latt_info, "point", [0,0,0,0])
    xi_GPT = g.vspincolor(U[0].grid)
    rng = g.random(randseed)

    Tmunu = np.zeros([Nv,4,4,n_max+1,n_max+1,n_max+1,Nsteps+1,Nt], dtype=np.complex128)
    CHI = np.zeros([Nv,2,n_max+1,n_max+1,n_max+1,Nsteps+1,Nt], dtype=np.complex128)
    for vec_picked in range(Nv):

        g.message('vec',vec_picked)
        rng.zn( xi_GPT , n=n_input )
        xi = gpt.LatticeGaugeGPT(xi_GPT, GEN_SIMD_WIDTH)
        eta = core.invertPropagator(dirac, xi, 1, 0)

        U_f = U.copy()

        for step in range(Nsteps+1):
            g.message('calc Tmunu, step =',step)
            
            tmpt,tmps = _get_Tmunu_symmetrized_P_Breit_slice(U_f, xi, eta, n_max)
            Tmunu[vec_picked,:,:,:,:,:,step,:] += tmpt
            CHI[vec_picked,:,:,:,:,step,:] += tmps

            if Nsteps > 0:
                if step == 0:

                    multi_fermion = convert.propagatorToMultiFermion(xi)
                    multi_fermion_wflow = U_f.gradientFlow(multi_fermion, "wilson", 10, stepsize / 10, True)
                    xi = convert.multiFermionToPropagator(multi_fermion_wflow)

                    multi_fermion = convert.propagatorToMultiFermion(eta)
                    multi_fermion_wflow = U_f.gradientFlow(multi_fermion, "wilson", 10, stepsize / 10, True)
                    eta = convert.multiFermionToPropagator(multi_fermion_wflow)

                    energy = U_f.wilsonFlow(10, epsilon=stepsize / 10)

                elif step < Nsteps:

                    multi_fermion = convert.propagatorToMultiFermion(xi)
                    multi_fermion_wflow = U_f.gradientFlow(multi_fermion, "wilson", 1, stepsize, True)
                    xi = convert.multiFermionToPropagator(multi_fermion_wflow)

                    multi_fermion = convert.propagatorToMultiFermion(eta)
                    multi_fermion_wflow = U_f.gradientFlow(multi_fermion, "wilson", 1, stepsize, True)
                    eta = convert.multiFermionToPropagator(multi_fermion_wflow)

                    energy = U_f.wilsonFlow(1, epsilon=stepsize)

    g.message(Nv,"random vectors done.")

    np.save(f'cTmunu_pervec_pyquda.npy', Tmunu)
    np.save(f'cCHI_pervec_pyquda.npy', CHI)
    
    Tmunu = np.mean(Tmunu,axis=0) / Ns3
    CHI = np.mean(CHI,axis=0) / Ns3
    for mu in range(4):
        for nu in range(mu,4):
            np.save(f'cT{mu+1}{nu+1}.npy', Tmunu[mu,nu])
    np.save(f'cCHI_pyquda.npy', CHI)

def get_invD(U, invPara):
    mf, csw, prec, cgMax = invPara
    inv = g.algorithms.inverter
    pc = g.qcd.fermion.preconditioner
    cg = inv.bicgstab({"eps": prec, "maxiter": cgMax})
    w = g.qcd.fermion.wilson_clover(U, mass=mf, csw_r=csw, csw_t=csw, nu=1.0, xi_0=1.0,
                                    isAnisotropic=False,
                                    boundary_phases=[1,1,1,-1])
    invD = w.propagator(inv.preconditioned(pc.eo1(), cg)).grouped(1)
    return invD,w


def impose_P_Breit_slice(complex_field, n_max, realize=False):
    g.message(f'impose_P_Breit_slice n_max = {n_max}')
    if realize:
        complex_field = g( (complex_field + g.adj(complex_field))/2 )
    L = np.array(complex_field.grid.fdimensions)
    Nt = L[3]
    results = np.zeros((n_max+1,n_max+1,n_max+1,Nt),dtype=np.complex128)
    for nx in range(n_max+1):
        for ny in range(n_max+1):
            for nz in range(n_max+1):
                P = g.exp_ixp( 2.0*np.pi*np.array([2*nx,2*ny,2*nz,0]) / L )
                results[nx,ny,nz] += np.array( g.slice( P*complex_field , 3 ) )
    return results



def get_Tmunu_symmetrized_P_Breit_slice(U_f, xi, eta, n_max):
    Nt = U_f[0].grid.gdimensions[3]

    # s term
    CHI = np.zeros([2,n_max+1,n_max+1,n_max+1,Nt], dtype=np.complex128)
    complex_field = g( g.adj(xi)*eta )
    CHI[0] = impose_P_Breit_slice(complex_field, n_max, realize=True)
    complex_field = g( g.adj(xi)*xi )
    CHI[1] = impose_P_Breit_slice(complex_field, n_max, realize=True)

    # t term
    Tmunu = np.zeros([4,4,n_max+1,n_max+1,n_max+1,Nt], dtype=np.complex128)
    for mu in range(4):
        tmp = g( U_f[mu]*g.cshift(eta,mu,1) - g.cshift( g.adj(U_f[mu])*eta , mu,-1) )
        for nu in range(4):
            complex_field = g( g.adj(xi)*g.gamma[nu]*tmp )
            Tmunu[nu,mu] += -0.5*impose_P_Breit_slice(complex_field, n_max, realize=True)

    # symmetrization
    for mu in range(4):
        for nu in range(mu+1,4):
            Tmunu[mu,nu] = ( Tmunu[mu,nu] + Tmunu[nu,mu] ) / 2
            Tmunu[nu,mu] = Tmunu[mu,nu]

    return Tmunu, CHI



def flowed_fermionic_EMT(gaugePara, randPara, invPara, flowPara, n_max):
    a, conf_id, U = gaugePara
    Nv, n_input, randseed = randPara
    stepsize, Nsteps, improve, division = flowPara

    assert len(U) == 4
    Nlat = U[0].grid.fsites
    Nt = U[0].grid.gdimensions[3]
    Ns3 = int(Nlat/Nt)
    g.message(Nt,Ns3,Nlat)

    invD = get_invD(U, invPara)[0]
    xi = g.vspincolor(U[0].grid)
    rng = g.random(randseed)
    g.mem_report()

    Tmunu = np.zeros([Nv,4,4,n_max+1,n_max+1,n_max+1,Nsteps+1,Nt], dtype=np.complex128)
    CHI = np.zeros([Nv,2,n_max+1,n_max+1,n_max+1,Nsteps+1,Nt], dtype=np.complex128)
    for vec_picked in range(Nv):
        g.message('vec',vec_picked)
        rng.zn( xi , n=n_input )
        eta = g( invD*xi )
        U_f = g.copy(U)
        U_f[3][:,:,:,Nt-1] *= -1

        for step in range(Nsteps+1):
            g.message('calc Tmunu, step =',step)
            
            tmpt,tmps = get_Tmunu_symmetrized_P_Breit_slice(U_f, xi, eta, n_max)
            Tmunu[vec_picked,:,:,:,:,:,step,:] += tmpt
            CHI[vec_picked,:,:,:,:,step,:] += tmps

            if Nsteps > 0:
                if step == 0:
                    xi = g.qcd.fermion.flow.Fermionflow_fixedstepsize(U_f, xi, stepsize/(10*division), 10*division, Ncheckpoints=0, improvement=improve)[1]
                    U_f, eta = g.qcd.fermion.flow.Fermionflow_fixedstepsize(U_f, eta, stepsize/(10*division), 10*division, Ncheckpoints=0, improvement=improve)
                elif step < Nsteps:
                    xi = g.qcd.fermion.flow.Fermionflow_fixedstepsize(U_f, xi, stepsize/division, division, Ncheckpoints=0, improvement=improve)[1]
                    U_f, eta = g.qcd.fermion.flow.Fermionflow_fixedstepsize(U_f, eta, stepsize/division, division, Ncheckpoints=0, improvement=improve)

    g.message(Nv,"random vectors done.")

    np.save(f'cTmunu_{a}_{conf_id}_pervec.npy', Tmunu)
    np.save(f'cCHI_{a}_{conf_id}_pervec.npy', CHI)
    
    Tmunu = np.mean(Tmunu,axis=0) / Ns3
    CHI = np.mean(CHI,axis=0) / Ns3
    for mu in range(4):
        for nu in range(mu,4):
            np.save(f'cT{mu+1}{nu+1}_{a}_{conf_id}.npy', Tmunu[mu,nu])
    np.save(f'cCHI_{a}_{conf_id}.npy', CHI)
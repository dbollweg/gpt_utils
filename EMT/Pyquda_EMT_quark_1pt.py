import gpt as g
import numpy as np
from opt_einsum import contract
import cupy as cp

# load pyquda modules
from pyquda import getMPIComm
from pyquda.field import LatticeInfo, LatticeGauge, LatticePropagator, LatticeFermion
from pyquda_utils import core, gpt, gamma, source, phase, convert
from pyquda_comm.array import arrayIdentity, arrayZeros
import subprocess

GEN_SIMD_WIDTH = 64

D_gammas = [gamma.gamma(1),gamma.gamma(2),gamma.gamma(4),gamma.gamma(8)]

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



def _get_Tmunu_symmetrized_P_Breit_slice(U_f: LatticeGauge, xi: LatticeFermion, eta: LatticeFermion, n_max: int):
    Nt = U_f.latt_info.global_size[3]

    # s term
    CHI = np.zeros([2,n_max+1,n_max+1,n_max+1,Nt], dtype=np.complex128)
    
    dot_xi_eta = contract('etzyxbc,etzyxbc->etzyx', xi.data.conj(), eta.data) 
    CHI[0] = _impose_P_Breit_slice(U_f, dot_xi_eta, n_max, realize=True)
    dot_xi_xi  = contract('etzyxbc,etzyxbc->etzyx', xi.data.conj(), xi.data) 
    CHI[1] = _impose_P_Breit_slice(U_f, dot_xi_xi, n_max, realize=True)

    # t term
    Tmunu = np.zeros([4,4,n_max+1,n_max+1,n_max+1,Nt], dtype=np.complex128)
    U_f.gauge_dirac.loadGauge(U_f)
    for mu in range(4):
        #\psi'(x)=U_\mu(x)\psi(x+\hat\mu)0,1,2,3 for x,y,z,t; 4,5,6,7 for -x,-y,-z,-t
        tmp = U_f.pure_gauge.covDev(eta, mu) - U_f.pure_gauge.covDev(eta, mu+4) 
        print('Pyquda data U_f',mu,(U_f.lexico())[mu,7,7,7,7])
        print('Pyquda data tmp',mu,(tmp.lexico())[7,7,7,7])
        print('Pyquda data xi',mu,(xi.lexico())[7,7,7,7])
        pyquda_tmp = contract('...sc,...sc->...', tmp.data.conj(), xi.data)
        pyquda_tmp = _impose_P_Breit_slice(U_f, pyquda_tmp, n_max, realize=True)
        g.message('Pyquda tmp.',mu,pyquda_tmp)
        for nu in range(4):
            Y = contract('ab,...bc->...ac', cp.asarray(D_gammas[nu]), tmp.data)
            complex_field = contract('...sc,...sc->...', xi.data.conj(), Y)
            Tmunu[nu,mu] += -0.5*_impose_P_Breit_slice(U_f, complex_field, n_max, realize=True)
            g.message('Pyquda Tmunu',mu,nu,Tmunu[mu,mu], Tmunu[nu,mu])

    # symmetrization
    for mu in range(4):
        for nu in range(mu+1,4):
            Tmunu[mu,nu] = ( Tmunu[mu,nu] + Tmunu[nu,mu] ) / 2
            Tmunu[nu,mu] = Tmunu[mu,nu]
            g.message('Pyquda symmetrization Tmunu',mu,nu,Tmunu[mu,mu], Tmunu[nu,mu])

    return Tmunu, CHI

def flowed_fermionic_EMT_pyquda(
    U: LatticeGauge, 
    U_GPT,
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

    invD = get_invD(U_GPT, [0.236, 1.0372, 1e-15, 300])[0]
    xi_GPT = g.vspincolor(U_GPT[0].grid)
    rng = g.random(randseed)

    #xi = source.fermion(U.latt_info, "point", [0,0,0,0])
    #xi_GPT = g.vspincolor(U_GPT[0].grid)

    Tmunu = np.zeros([Nv,4,4,n_max+1,n_max+1,n_max+1,Nsteps+1,Nt], dtype=np.complex128)
    CHI = np.zeros([Nv,2,n_max+1,n_max+1,n_max+1,Nsteps+1,Nt], dtype=np.complex128)
    for vec_picked in range(Nv):

        g.message('vec',vec_picked)
        rng.zn( xi_GPT , n=n_input )
        eta_GPT = g( invD*xi_GPT )

        xi_M = g.mspincolor(U_GPT[0].grid)
        eta_M = g.mspincolor(U_GPT[0].grid)
        xi_M[:] = 0
        eta_M[:] = 0
        for s_col in range(4):
            for c_col in range(3):
                for s_row in range(4):
                    for c_row in range(3):
                        xi_M[:, :, :, :, s_row, s_col, c_row, c_col] = xi_GPT[:, :, :, :, s_row, c_row]
                        eta_M[:, :, :, :, s_row, s_col, c_row, c_col] = eta_GPT[:, :, :, :, s_row, c_row]
        xi_tmp = gpt.LatticePropagatorGPT(xi_M, GEN_SIMD_WIDTH)
        eta_tmp= gpt.LatticePropagatorGPT(eta_M, GEN_SIMD_WIDTH)
        xi = LatticeFermion(U.latt_info, xi_tmp.data[:,:,:,:,:,:,0,:,0])
        eta = LatticeFermion(U.latt_info, eta_tmp.data[:,:,:,:,:,:,0,:,0])

        xi_test = LatticeFermion(U.latt_info, xi_tmp.data[:,:,:,:,:,:,1,:,1])
        eta_test = LatticeFermion(U.latt_info, eta_tmp.data[:,:,:,:,:,:,1,:,1])
        g.message("xi_test norm:", (xi-xi_test).norm2()**0.5)
        g.message("eta_test norm:", (eta-eta_test).norm2()**0.5)

        #U_f = U.copy()
        U_f = gpt.LatticeGaugeGPT(U_GPT, GEN_SIMD_WIDTH)

        for step in range(Nsteps+1):
            g.message('calc Tmunu, step =',step)
            U_f.gauge_dirac.loadGauge(U_f)
            
            tmpt,tmps = _get_Tmunu_symmetrized_P_Breit_slice(U_f, xi, eta, n_max)
            Tmunu[vec_picked,:,:,:,:,:,step,:] += tmpt
            CHI[vec_picked,:,:,:,:,step,:] += tmps

            if Nsteps > 0:

                # Multi_xi = core.MultiLatticeFermion(U.latt_info, 1, cp.array([xi.data]))
                # Multi_eta = core.MultiLatticeFermion(U.latt_info, 1, cp.array([eta.data]))

                print(f'Pyquda step{step} with epsilon = {stepsize} data 0 U_f0',0,(U_f.lexico())[0,7,7,7,7])
                print(f'Pyquda step{step} with epsilon = {stepsize} data 0 U_f1',0,(U_f.lexico())[1,7,7,7,7])
                print(f'Pyquda step{step} with epsilon = {stepsize} data 0 U_f2',0,(U_f.lexico())[2,7,7,7,7])
                print(f'Pyquda step{step} with epsilon = {stepsize} data 0 U_f3',0,(U_f.lexico())[3,7,7,7,7])
                print(f'Pyquda step{step} with epsilon = {stepsize} data 0 xi',0,(xi.lexico())[7,7,7,7])

                temp = core.MultiLatticeFermion(U.latt_info, 2, cp.array([xi.data, eta.data]))
                temp_flow = U_f.gradientFlow(temp, "wilson", 1, stepsize)
                xi, eta = temp_flow[0], temp_flow[1]

                # Multi_xi = U_f.gradientFlow(Multi_xi, "wilson", 1, stepsize, True)
                # Multi_eta = U_f.gradientFlow(Multi_eta, "wilson", 1, stepsize, True)
                # energy = U_f.wilsonFlow(1, epsilon=stepsize)

                # xi = LatticeFermion(U.latt_info, Multi_xi.data[0, :, :, :, :, :, :, :])
                # eta = LatticeFermion(U.latt_info, Multi_eta.data[0, :, :, :, :, :, :, :])

                print(f'Pyquda step{step} with epsilon = {stepsize} data 1 U_f0',0,(U_f.lexico())[0,7,7,7,7])
                print(f'Pyquda step{step} with epsilon = {stepsize} data 1 U_f1',0,(U_f.lexico())[1,7,7,7,7])
                print(f'Pyquda step{step} with epsilon = {stepsize} data 1 U_f2',0,(U_f.lexico())[2,7,7,7,7])
                print(f'Pyquda step{step} with epsilon = {stepsize} data 1 U_f3',0,(U_f.lexico())[3,7,7,7,7])
                print(f'Pyquda step{step} with epsilon = {stepsize} data 1 xi',0,(xi.lexico())[7,7,7,7])

    g.message(Nv,"random vectors done.")

    np.save(f'{datfile}/cTmunu_pervec.pyquda.npy', Tmunu)
    np.save(f'{datfile}/cCHI_pervec.pyquda.npy', CHI)
    
    Tmunu = np.mean(Tmunu,axis=0) / Ns3
    CHI = np.mean(CHI,axis=0) / Ns3
    for mu in range(4):
        for nu in range(mu,4):
            np.save(f'{datfile}/cT{mu+1}{nu+1}.pyquda.npy', Tmunu[mu,nu])
    np.save(f'{datfile}/cCHI.pyquda.npy', CHI)







def impose_P_Breit_slice(complex_field, n_max, realize=False):
    g.message(f'impose_P_Breit_slice n_max = {n_max}')
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
        print('GPT data U_f',mu,U_f[mu][7,7,7,7])
        print('GPT data tmp',mu,tmp[7,7,7,7])
        print('GPT data xi',mu,xi[7,7,7,7])
        gpt_tmp = g( g.adj(tmp)*xi )
        gpt_tmp = impose_P_Breit_slice(gpt_tmp, n_max, realize=True)
        g.message('GPT tmp',mu, gpt_tmp)
        for nu in range(4):
            complex_field = g( g.adj(xi)*g.gamma[nu]*tmp )
            Tmunu[nu,mu] += -0.5*impose_P_Breit_slice(complex_field, n_max, realize=True)
            g.message('Tmunu',mu,nu,Tmunu[mu,mu], Tmunu[nu,mu])

    # symmetrization
    for mu in range(4):
        for nu in range(mu+1,4):
            Tmunu[mu,nu] = ( Tmunu[mu,nu] + Tmunu[nu,mu] ) / 2
            Tmunu[nu,mu] = Tmunu[mu,nu]
            g.message('symmetrization Tmunu',mu,nu,Tmunu[mu,mu], Tmunu[nu,mu])

    return Tmunu, CHI



def flowed_fermionic_EMT(gaugePara, randPara, invPara, flowPara, datfile='', n_max = 0):
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
        #U_f[3][:,:,:,Nt-1] *= -1 #! FIXME, what is this?
        g.message('tmp no U_f[3][:,:,:,Nt-1] *= -1')

        for step in range(Nsteps+1):
            g.message('calc Tmunu, step =',step)
            
            tmpt,tmps = get_Tmunu_symmetrized_P_Breit_slice(U_f, xi, eta, n_max)
            Tmunu[vec_picked,:,:,:,:,:,step,:] += tmpt
            CHI[vec_picked,:,:,:,:,step,:] += tmps

            if Nsteps > 0:

                print(f'GPT step{step} with epsilon = {stepsize} data 0 U_f0',0,U_f[0][7,7,7,7])
                print(f'GPT step{step} with epsilon = {stepsize} data 0 U_f1',0,U_f[1][7,7,7,7])
                print(f'GPT step{step} with epsilon = {stepsize} data 0 U_f2',0,U_f[2][7,7,7,7])
                print(f'GPT step{step} with epsilon = {stepsize} data 0 U_f3',0,U_f[3][7,7,7,7])
                print(f'GPT step{step} with epsilon = {stepsize} data 0 xi',0,xi[7,7,7,7])
                xi = g.qcd.fermion.flow.Fermionflow_fixedstepsize(U_f, xi, stepsize, 1, Ncheckpoints=0, improvement=False)[1]
                U_f, eta = g.qcd.fermion.flow.Fermionflow_fixedstepsize(U_f, eta, stepsize, 1, Ncheckpoints=0, improvement=False)
                print(f'GPT step{step} with epsilon = {stepsize} data 1 U_f0',0,U_f[0][7,7,7,7])
                print(f'GPT step{step} with epsilon = {stepsize} data 1 U_f1',0,U_f[1][7,7,7,7])
                print(f'GPT step{step} with epsilon = {stepsize} data 1 U_f2',0,U_f[2][7,7,7,7])
                print(f'GPT step{step} with epsilon = {stepsize} data 1 U_f3',0,U_f[3][7,7,7,7])
                print(f'GPT step{step} with epsilon = {stepsize} data 1 xi',0,xi[7,7,7,7])

    g.message(Nv,"random vectors done.")

    np.save(f'{datfile}/cTmunu_pervec.GPT.npy', Tmunu)
    np.save(f'{datfile}/cCHI_pervec.GPT.npy', CHI)
    
    Tmunu = np.mean(Tmunu,axis=0) / Ns3
    CHI = np.mean(CHI,axis=0) / Ns3
    for mu in range(4):
        for nu in range(mu,4):
            np.save(f'{datfile}/cT{mu+1}{nu+1}.GPT.npy', Tmunu[mu,nu])
    np.save(f'{datfile}/cCHI.GPT.npy', CHI)

    

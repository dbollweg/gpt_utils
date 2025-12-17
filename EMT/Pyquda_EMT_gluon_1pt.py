import gpt as g
import numpy as np
from opt_einsum import contract

# load pyquda modules
from pyquda.field import LatticeInfo, LatticeGauge
from pyquda_utils import core, gpt, gamma, source, phase
from pyquda_comm.array import arrayIdentity, arrayZeros
import subprocess

GEN_SIMD_WIDTH = 64

def _F_clover_traceless(U: LatticeGauge, mu: int, nu: int):
    """
    Clover 1x1 field-strength-like tensor F_{mu,nu}:
      - build 4 plaquettes (clover) from U.loop
      - average over corners
      - project to anti-Hermitian traceless
      - multiply overall factor (-i/2)
    Result is stored in-place in a LatticeGauge-like object, same type as U.
    """
    loops_one = [
        [mu,   nu,   mu+4, nu+4],
        [nu,   mu+4, nu+4, mu],
        [mu+4, nu+4, mu,   nu],
        [nu+4, mu,   nu,   mu+4],
    ]

    # 1) sum over clover corners
    F = U.loop([loops_one] * 4, coeff=[1.0, 1.0, 1.0, 1.0])   # same type as U
    data = F.data                                             # (..., Nc, Nc)

    # 2) anti-Hermitian part: A = 1/8 (F - F†) (including average clover factor 1/4)
    A = 0.125 * (data - data.swapaxes(-2, -1).conjugate())

    # 3) traceless: A -> A - tr(A)/Nc * I
    Nc = A.shape[-1]
    trA = contract('...ii->...', A)                  # (...,)
    I = arrayIdentity(Nc, A.dtype, F.location)       # (Nc, Nc)
    A -= trA[..., None, None] * I / Nc

    # 4) overall normalization: F_mu_nu = (-i) * A
    data[...] = (-1j) * A

    return F

def _all_F_clover_traceless(U: LatticeGauge,):
    F = [[None]*4 for _ in range(4)]
    planes = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    for mu, nu in planes:
        F_mu_nu = (_F_clover_traceless(U, mu, nu)).data[0]
        F[mu][nu] = F_mu_nu
        F[nu][mu] = -F_mu_nu

    return F

def flowed_gluonic_EMT_P_pyquda(
    U: LatticeGauge,
    stepsize: float = 0.1,
    Nsteps: int = 20,
    datfile: str = "",
    n_max: int = 0,
    improve: bool = False,
):
    """
    PyQUDA version of flowed_gluonic_EMT_P, using
    _F_clover_traceless / _all_F_clover_traceless and `contract`
    for all color traces and contractions.

    U: LatticeGauge (4D gauge field, with geometry in U.latt_info)
    返回值: T_{mu nu}(n_x,n_y,n_z,flow_step,t) 的 numpy 数组
           形状为 (4,4,n_max+1,n_max+1,n_max+1,Nsteps+1,Nt)
    """
    # --- 几何信息，从 PyQUDA 的 LatticeGauge 中取 ---
    global_size = U.latt_info.global_size
    Lx, Ly, Lz, Lt = U.latt_info.size
    Ns3 = global_size[0] * global_size[1] * global_size[2]

    g.message("Lx, Ly, Lz, Lt, Ns3 =", Lx, Ly, Lz, Lt, Ns3)

    # Tmunu_t[mu,nu,nx,ny,nz,step,t]
    Tmunu_t = np.zeros(
        (4, 4, n_max + 1, n_max + 1, n_max + 1, Nsteps + 1, global_size[3]),
        dtype=np.complex128,
    )

    # 为了不修改输入的 U，可以复制一份（如果你不介意 in-place flow，可以直接用 U）
    U_flow = U.copy()

    for step in range(Nsteps + 1):
        g.message("step", step, "calculate F")

        # --------- 1. 计算所有 F_{mu,nu}（clover + traceless） ---------
        # _all_F_clover_traceless(U) 返回 F[mu][nu] = array(shape=(Nlat, Nc,Nc))
        F = _all_F_clover_traceless(U_flow)

        g.message("step", step, "calculate T")

        # --------- 2. 用 F 构造 EMT T_{mu,nu}(x) 并做动量投影 ---------
        # 原来是:
        # tmp(x) = sum_{rho != mu,nu} tr_c [ F_{mu,rho}(x) * F_{nu,rho}(x) ]
        # 然后做 P(p) * tmp 的三维和, slice over time
        for mu in range(4):
            for nu in range(mu, 4):
                # tmp: site 上的复数场，shape=(Nlat,)
                tmp = arrayZeros((2, Lt, Lz, Ly, Lx // 2), U.data.dtype, U.location)       # (Nc, Nc)

                for rho in range(4):
                    if rho == mu or rho == nu:
                        continue

                    # F_{mu,rho}(x), F_{nu,rho}(x), shape = (Nlat, Nc, Nc)
                    F_mr = F[mu][rho]
                    F_nr = F[nu][rho]
                    g.message('DEBUG F_mr shape:', F_mr.shape)
                    g.message('DEBUG F_nr shape:', F_nr.shape)

                    # color trace of matrix product:
                    # tr(F_mr * F_nr) = sum_{a,b} F_mr[a,b] * F_nr[b,a]
                    # 直接用 einsum: '...ab,...ba->...'
                    tmp += contract("...ab,...ba->...", F_mr, F_nr)

                # ---- 3. 对每个 (n_x,n_y,n_z) 做平面波投影并在空间上求和 ----
                # 原始代码：P = g.exp_ixp(2π * [2nx,2ny,2nz,0]/L)，然后 g.slice(P*tmp, 3)
                # 这里手动构造 P(x)=exp(i p·x)，再对 (x,y,z) 求和得到各个 t 的值
                for nx in range(n_max + 1):
                    for ny in range(n_max + 1):
                        for nz in range(n_max + 1):
                            # p_mu = 2π * (2n_mu / L_mu) ，最后分量固定为 0
                            qext_xyz = [[2 * nx, 2 * ny, 2 * nz]]

                            # phase(x) = p · x
                            phases_3pt = phase.MomentumPhase(U.latt_info).getPhases(qext_xyz, [0,0,0,0])
                            
                            # 对 (x,y,z) 求和，保留 t 维度，shape=(Lt,)
                            slice_t = core.gatherLattice(contract("qwtzyx, wtzyx -> qt", phases_3pt, tmp).get(), [1, -1, -1, -1])

                            if U.latt_info.mpi_rank == 0:
                                Tmunu_t[mu, nu, nx, ny, nz, step, :] += 2.0 * slice_t[0]

        # --------- 4. 做 Wilson flow / Zeuthen flow 更新 U_flow ---------
        # TODO add improve option
        if Nsteps > 0:
            if step == 0:
                g.message("wilsonFlow step =", step)
                energy = U_flow.wilsonFlow(10, epsilon=stepsize / 10)
            elif step < Nsteps:
                g.message("wilsonFlow step =", step)
                energy = U_flow.wilsonFlow(1, epsilon=stepsize)

    # --------- 5. 归一化 & 存盘 ---------
    # 原代码最后除以 Ns3
    Tmunu_t /= Ns3

    for mu in range(4):
        for nu in range(mu, 4):
            suffix = f".T{mu+1}{nu+1}.n_max{n_max}.pyquda.npy"
            np.save(datfile + suffix, Tmunu_t[mu, nu])

    return Tmunu_t
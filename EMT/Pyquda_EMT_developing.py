import gpt as g
import numpy as np
from utils.io_corr import *
import time
from typing import List
from opt_einsum import contract

from pyquda import init, LatticeInfo, LatticeGauge
from pyquda.field import LatticeFermion
from pyquda_utils import core, gpt, gamma, phase
from pyquda.dirac import GaugeDirac

my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]

ordered_list_of_gammas = [g.gamma[5], g.gamma["T"], g.gamma["T"]*g.gamma[5],
                                      g.gamma["X"], g.gamma["X"]*g.gamma[5], 
                                      g.gamma["Y"], g.gamma["Y"]*g.gamma[5],
                                      g.gamma["Z"], g.gamma["Z"]*g.gamma[5], 
                                      g.gamma["I"], g.gamma["SigmaXT"], 
                                      g.gamma["SigmaXY"], g.gamma["SigmaXZ"], 
                                      g.gamma["SigmaZT"]
                            ]
GEN_SIMD_WIDTH = 64

'''---------------------------------------------------------------'''
'''         Pyquda EMT quark 1pt --> Ringed quark fields          '''
'''---------------------------------------------------------------'''

# ---------------------------
# 与 Grid 版同义：<xi^\dagger gamma_nu tmp> 的 (t)-slice 和空间求和
# xi,tmp: LatticeFermion
# 假设 data shape = (Lt, Lz, Ly, Lx, Nc, Ns)
# ---------------------------
def _bilinear_slice_t(xi, tmp, nu):
    xp = xi.backend
    gammas = _gamma_matrices(xp, xi.data.dtype)[nu]  # (4,4)

    # ---- 把自旋-颜色放到最后两维： (..., Nc, Ns)
    # 如果你的布局不同，请把这两行 reshape/axis 顺序对应改掉
    Xi  = xi.data      # (Lt,Lz,Ly,Lx,Nc,Ns)
    Tmp = tmp.data

    # 颜色与自旋收缩：sum_{c,s,s'} xi^*_{c s} (gamma_nu)_{s s'} tmp_{c s'}
    # 先乘自旋，再 sum 颜色、自旋
    #   Y_{... , c, s} = sum_{s'} (gamma)_{s s'} Tmp_{..., c, s'}
    Y = contract('ab,...csb->...csa', gammas, Tmp)          # (...,Nc,Ns)

    #   Z_{...} = sum_{c,s} Xi^*_{..., c, s} * Y_{..., c, s}
    Z = contract('...cs,...cs->...', Xi.conj(), Y)          # (...,)

    # 按 (t) 切片并对空间求和
    # reshape → (Lt, Lz*Ly*Lx)
    Lt, Lz, Ly, Lx = xi.latt_info.size[3], xi.latt_info.size[2], xi.latt_info.size[1], xi.latt_info.size[0]
    Z4 = Z.reshape(Lt, Lz*Ly*Lx)
    return Z4.real.sum(axis=1)   # (Lt,)


# -------- Z_n 随机源，形状严格匹配 (2,Lt,Lz,Ly,Xh,Ns,Nc) ----------
def _zn_noise_like(latt_info, backend, dtype, n_input: int, seed: int):
    # 构造空的 LatticeFermion 容器
    src = LatticeFermion(latt_info)
    xp  = backend
    shape = src.data.shape
    # RandomState for numpy/cupy 一致；cupy 也支持 RandomState
    rng = xp.random.RandomState(seed)
    k = rng.randint(0, n_input, size=shape)        # 整型索引
    phases = xp.exp(1j * (2.0 * xp.pi) * (k / float(n_input)))
    src.data[...] = phases.astype(dtype)
    return src

# -------- 主函数：费米子 EMT (disconnected) ----------
def flowed_fermionic_EMT_pyquda(
    gauge,                  # LatticeGauge
    dirac,                  # 你构造并 loadGauge 的 Dirac 对象: core.getDirac(...)
    Nv: int, n_input: int, randseed: int,
    stepsize: float, Nsteps: int, division: int,
    out_prefix: str,
):
    xp            = gauge.backend
    Lx, Ly, Lz, Lt = gauge.latt_info.size
    Ns3          = Lx * Ly * Lz

    # 输出数组（实部）
    Tmunu = xp.zeros((Nsteps+1, 4, 4, Lt), dtype=xp.float64)
    CHI   = xp.zeros((Nsteps+1, 2, Lt),   dtype=xp.float64)

    gdir = gauge.gauge_dirac   # 用于 covDev 和 Wilson flow

    for v in range(Nv):
        # ---- 随机源 xi (Z_n) ----
        xi = _zn_noise_like(gauge.latt_info, gauge.backend, xp.complex128, n_input, randseed + 7919*v)

        # ---- 初始反演 eta = D^{-1} xi ----
        # 你给的接口：core.invertPropagator(dirac, b, 1, 0)
        # 若返回值是新费米子，就直接接收；若是写到 b 或 out，需要对应调整
        eta = LatticeFermion(gauge.latt_info)
        dirac.core.invertPropagator(dirac, xi, 1, 0, out=eta)  # 如果没有 out=，就 eta = core.invertPropagator(...)

        # ---- flow + 观测 ----
        for step in range(Nsteps+1):
            # EMT: -1/2 * <xi† γ_ν ∇_μ eta>
            for mu in range(4):
                tmp = gdir.covDev(eta, mu)  # LatticeFermion，与 xi 同布局
                for nu in range(4):
                    Tmunu[step, nu, mu, :] += -0.5 * _bilinear_slice_t(xi, tmp, nu)

            # χ 通道
            # CHI[0] = Re <xi† eta>_space-sum ; CHI[1] = Re <xi† xi>_space-sum
            # 逐点先收缩 s,c → 得到 (2,Lt,Lz,Ly,Xh)
            dot_xi_eta = contract('etzyxbc,etzyxbc->etzyx', xi.data.conj(), eta.data).real
            dot_xi_xi  = contract('etzyxbc,etzyxbc->etzyx', xi.data.conj(), xi.data).real
            # 再对 EO 与空间求和
            CHI[step, 0, :] += dot_xi_eta.sum(axis=(0,2,3,4))
            CHI[step, 1, :] += dot_xi_xi .sum(axis=(0,2,3,4))

            # Wilson flow 规范场，然后重载并重新反演 eta
            if Nsteps > 0 and step < Nsteps:
                if step == 0:
                    gdir.wilsonFlow(n_steps=10*division, epsilon=stepsize/(10*division),
                                    t0=0.0, restart=False, compute_plaquette=False, compute_qcharge=False)
                else:
                    gdir.wilsonFlow(n_steps=division,      epsilon=stepsize/division,
                                    t0=0.0, restart=False, compute_plaquette=False, compute_qcharge=False)
                # 将 flow 后的 gauge 住进 dirac，并重新反演
                dirac.loadGauge(gauge)
                eta = LatticeFermion(gauge.latt_info)
                dirac.core.invertPropagator(dirac, xi, 1, 0, out=eta)

    # 归一化（按体积与随机源数）
    Tmunu /= (Ns3 * Nv)
    CHI   /= (Ns3 * Nv)

    # 保存（cupy → numpy）
    def _to_numpy(a):
        if type(a).__module__.startswith('cupy'):
            import cupy as cp
            return cp.asnumpy(a)
        return a

    np.save(f'{out_prefix}_fEMT.npy', _to_numpy(Tmunu))
    np.save(f'{out_prefix}_CHI.npy',  _to_numpy(CHI))

    return Tmunu, CHI




class EMT_measurement():
    def __init__(self, parameters):
        self.quark_mom = parameters["quark_mom"]
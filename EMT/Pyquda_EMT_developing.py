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
'''    Pyquda EMT gluonic 1pt --> gluonic disconnected diagram    '''
'''---------------------------------------------------------------'''

# ---------- 共轭转置（用 transpose，而不是 contract/swapaxes） ----------
def _adj_last2(M):
    # 把最后两轴对调：(..., i, j) -> (..., j, i)
    axes = tuple(range(M.ndim - 2)) + (M.ndim - 1, M.ndim - 2)
    return M.conj().transpose(axes)

# ---------- 反厄米去迹（不造单位阵，直接改对角线） ----------
def _proj_traceless_antiherm_data(M):
    Nc  = M.shape[-1]
    A   = 0.5 * (M - _adj_last2(M))          # 反厄米
    trA = contract('...ii->...', A)          # 逐点迹
    Aii = A.diagonal(0, -2, -1)              # (..., Nc) 视图，可写
    Aii -= trA[..., None] / Nc               # 去迹
    return A

# ---------- 颜色迹 tr_c(F_mu_rho F_nu_rho) ----------
def _color_trace_FF(F_mu_rho, F_nu_rho):
    prod = contract('...ik,...kj->...ij', F_mu_rho, F_nu_rho)
    return contract('...ii->...', prod)      # (...,)

# ---------- clover 1x1 的 F_{mu,nu}（只用 LatticeGauge.loop） ----------
def _F_clover_traceless(U, mu: int, nu: int):
    loops_one = [
        [mu,   nu,   mu+4, nu+4],
        [nu,   mu+4, nu+4, mu],
        [mu+4, nu+4, mu,   nu],
        [nu+4, mu,   nu,   mu+4],
    ]
    # 先拿到 clover 四角求和的容器
    F = U.loop([loops_one]*4, coeff=[1.0, 1.0, 1.0, 1.0])   # F 是一个 LatticeGauge

    # 直接把 F.data 原地改成代数化的结果：
    # 1) clover 平均
    F.data[...] *= 0.25
    # 2) 反厄米 + 去迹
    A = _proj_traceless_antiherm_data(F.data)               # 返回的是视图/新张量都行
    # 3) 规范因子
    F.data[...] = (-1j) * 0.5 * A

    return F

# ---------- 空间 Fourier 投影 ----------
def _fourier_project_ts(tmp_point_scalar, U, n_max):
    """
    输入:
      - tmp_point_scalar: 逐点复标量，shape 可 reshape 为 (Lt, Lz, Ly, Lx)，当前 backend
      - U: LatticeGauge（提供 latt_info / backend）
      - n_max: 动量最大整数
    输出:
      - (n_max+1, n_max+1, n_max+1, Lt)，与原实现一致（含“×2”的约定）
    """

    Lx, Ly, Lz, Lt = U.latt_info.size

    # 1) 构造动量列表（双倍动量以匹配你原来的定义）
    mom_list = [(2*nx, 2*ny, 2*nz)
                for nx in range(n_max+1)
                for ny in range(n_max+1)
                for nz in range(n_max+1)]

    # 2) 生成相位 (K, 2, Lt, Lz, Ly, Lx//2)（自动在正确 backend）
    phases = phase.MomentumPhase(U.latt_info).getPhases(mom_list)   # complex128 by default

    # 3) 把 tmp 转为 EO 布局，与 phases 对齐: (2, Lt, Lz, Ly, Lx//2)
    tmp4d = tmp_point_scalar.reshape(Lt, Lz, Ly, Lx)
    tmp_eo = U.latt_info.evenodd(tmp4d, True)   # True: complex input → (2, Lt, Lz, Ly, Lx//2)

    # 4) 向量化做所有动量的空间+奇偶求和，保留 Lt:
    #    phases: (K, 2, Lt, Lz, Ly, Xh) ; tmp_eo: (2, Lt, Lz, Ly, Xh)
    #    sum over (p,z,y,xh) → (K, Lt)
    proj_kt = contract('kptzyx,ptzyx->kt', phases, tmp_eo)

    # 5) 恢复到 (nx, ny, nz, Lt)，并保留你原始实现里的 ×2 约定
    out = proj_kt.reshape(n_max+1, n_max+1, n_max+1, Lt)
    return 2.0 * out


# ---------- Wilson flow 版本：flowed_gluonic_EMT_P ----------
def flowed_gluonic_EMT_P_pyquda(U, stepsize=0.1, Nsteps=20, division=1, datfile='', n_max=0):
    """
    仅用 Wilson flow（不包含 Zeuthen flow）；张量操作驻留在 U.backend（numpy/cupy）。
    返回 Tmunu_t[mu,nu,nx,ny,nz,step,t]
    """
    xp = U.backend
    Lx, Ly, Lz, Lt = U.latt_info.size
    Ns3 = Lx * Ly * Lz

    Tmunu_t = xp.zeros((4, 4, n_max+1, n_max+1, n_max+1, Nsteps+1, Lt), dtype=xp.complex128)
    gdir = U.gauge_dirac

    for step in range(Nsteps+1):
        # -- F_{mu,nu}
        # 预计算 6 个 F_{mu,nu}（mu<nu）的 data
        F_up = {}
        for mu in range(4):
            for nu in range(mu+1, 4):
                F_up[(mu, nu)] = _F_clover_traceless(U, mu, nu).data

        def F_data(mu, nu):
            if mu < nu:
                return F_up[(mu, nu)]
            elif mu > nu:
                return -F_up[(nu, mu)]
            else:
                raise RuntimeError("F_{mu,mu} 不应被访问")

        # 之后在 T 的循环里这样用：
        for mu in range(4):
            for nu in range(mu, 4):
                acc = None
                for rho in range(4):
                    if rho == mu or rho == nu: 
                        continue
                    s = _color_trace_FF(F_data(mu, rho), F_data(nu, rho))
                    acc = s if acc is None else (acc + s)
                Pproj = _fourier_project_ts(acc, U, n_max)
                Tmunu_t[mu, nu, :, :, :, step, :] += Pproj


        # -- Wilson flow（沿用 step=0 细步长、其后 division 的策略）
        if Nsteps > 0 and step < Nsteps:
            if step == 0:
                gdir.wilsonFlow(n_steps=10*division, epsilon=stepsize/(10*division),
                                t0=0.0, restart=False, compute_plaquette=False, compute_qcharge=False)
            else:
                gdir.wilsonFlow(n_steps=division, epsilon=stepsize/division,
                                t0=0.0, restart=False, compute_plaquette=False, compute_qcharge=False)

    # 空间平均
    Tmunu_t /= Ns3

    # 保存
    def _to_numpy(a):
        if type(a).__module__.startswith('cupy'):
            import cupy as cp
            return cp.asnumpy(a)
        return a

    for mu in range(4):
        for nu in range(mu, 4):
            tag = f'.T{mu+1}{nu+1}.n_max{n_max}'
            np.save(datfile + tag + '.slice.npy', _to_numpy(Tmunu_t[mu, nu]))

    return Tmunu_t


'''---------------------------------------------------------------'''
'''         Fermion flow for Pyquda LatticeFermion fields         '''
'''---------------------------------------------------------------'''

# -----------------------------
# 4D 规范协变拉普拉斯：Δ_cov ψ ≈ -∑_μ ∇_μ ( ∇_μ ψ )
# 这里 ∇_μ 用 QUDA 的 COVDEV_DSLASH (GaugeDirac.covDev)
# -----------------------------
def covariant_laplacian_4d(gdir, psi: LatticeFermion) -> LatticeFermion:
    out = LatticeFermion(psi.latt_info)
    out.data[...] = 0
    for mu in range(4):
        d1 = gdir.covDev(psi, mu)   # ∇_μ ψ
        d2 = gdir.covDev(d1,  mu)   # ∇_μ (∇_μ ψ)
        out.data[...] -= d2.data
    return out


# -----------------------------
# 单步三段（W0→W1→W2）费米子流，与规范流同步
#   输入:  resident gauge = W0，费米子 chi
#   输出:  resident gauge 更新为 W2，返回 chi_new
# -----------------------------
def fermion_flow_step_sync(gdir,
                           gauge: LatticeGauge,
                           chi: LatticeFermion,
                           epsilon: float) -> LatticeFermion:
    # --- W0 阶段
    phi0 = covariant_laplacian_4d(gdir, chi)                 # φ0 = Δ[W0] χ
    phi1 = LatticeFermion(chi.latt_info)
    phi1.data[...] = chi.data + (0.25 * epsilon) * phi0.data # φ1

    # --- 推进到 W1（一次 Wilson flow 子步）
    gdir.wilsonFlow(n_steps=1, epsilon=epsilon, t0=0.0, restart=False,
                    compute_plaquette=False, compute_qcharge=False)

    # --- W1 阶段
    lap_W1_phi1 = covariant_laplacian_4d(gdir, phi1)
    phi2 = LatticeFermion(chi.latt_info)
    phi2.data[...] = chi.data + (8.0/9.0)*epsilon*lap_W1_phi1.data - (2.0/9.0)*epsilon*phi0.data

    # --- 推进到 W2（再一次 Wilson flow 子步）
    gdir.wilsonFlow(n_steps=1, epsilon=epsilon, t0=0.0, restart=False,
                    compute_plaquette=False, compute_qcharge=False)

    # --- W2 阶段 & 合成 χ^{new}
    lap_W2_phi2 = covariant_laplacian_4d(gdir, phi2)
    chi_new = LatticeFermion(chi.latt_info)
    chi_new.data[...] = phi1.data + (3.0/4.0)*epsilon*lap_W2_phi2.data

    # resident gauge 已是 W2
    return chi_new


# -----------------------------
# N 步费米子流（总流时 t = N * eps）
# 可在每步后做观测（callback(step, t_flow, gauge, chi)）
# -----------------------------
def fermion_flow(gauge: LatticeGauge,
                 chi_in: LatticeFermion,
                 steps: int,
                 epsilon: float,
                 callback: Optional[Callable[[int, float, LatticeGauge, LatticeFermion], None]] = None
                 ) -> LatticeFermion:
    gdir = gauge.gauge_dirac
    chi = LatticeFermion(chi_in.latt_info)
    chi.data[...] = chi_in.data
    t_flow = 0.0

    # 确保 gauge 已 resident
    gdir.loadGauge(gauge)

    for s in range(steps):
        chi = fermion_flow_step_sync(gdir, gauge, chi, epsilon)  # W0→W2，chi 更新
        t_flow += epsilon
        if callback is not None:
            callback(s+1, t_flow, gauge, chi)

    # 用完可选择释放 resident gauge
    # gdir.freeGauge()
    return chi


# -----------------------------
# （可选）伴随步（从 t+ε 回到 t），结构按 Grid::evolve_step_adjoint 对应
# 注意：严格的伴随需要 W0/W1/W2 的时间反演轨迹。
# 这里提供一个“同步近似版”：在当前 resident gauge 上做镜像组合。
# 若你要严格版，请配合 checkpoint 重播规范场（见下注）
# -----------------------------
def fermion_flow_step_adj_sync(gdir,
                               gauge: LatticeGauge,
                               chi: LatticeFermion,
                               epsilon: float) -> LatticeFermion:
    # —— 近似镜像（需要严格版本时：用 checkpoint 重建 W2、W1、W0，逐段套 Δ，与 Grid 保持一致）
    # 这里简化：在当前 resident gauge 上用与前向相同的 Δ 组合一次
    # （实际科研用途建议按 Grid 的 checkpoint 策略实现严格反向步）
    lap = covariant_laplacian_4d(gdir, chi)
    chi_prev = LatticeFermion(chi.latt_info)
    chi_prev.data[...] = chi.data - (0.75)*epsilon*lap.data
    return chi_prev


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
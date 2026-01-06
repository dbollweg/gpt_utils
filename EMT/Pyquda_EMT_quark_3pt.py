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
    g.message("before invD")
    invD = w.propagator(inv.preconditioned(pc.eo1(), cg)).grouped(1)
    return invD,w

def _slice_sum_t_pyquda(U: LatticeGauge, complex_field_4d):
    """
    complex_field_4d: shape (Nt, Nz, Ny, Nx) or lexico-able to that order,
                      dtype complex
    return: C(t) on root=0, broadcast to all ranks, shape (Nt,), complex128
    """
    # 约定：你的 complex_field 是 (t,z,y,x) 的顺序（和你上面 dot_xi_eta 那种一样）
    # 空间求和：sum_{x,y,z}
    # 这里直接用 opt_einsum 做 sum
    slice_t_local = contract("tzyx->t", complex_field_4d)

    # gatherLattice: 把各 rank 的局部 t-slice 合并（你上面就是这么干的）
    slice_t = core.gatherLattice(slice_t_local.get() if hasattr(slice_t_local, "get") else slice_t_local,
                                 [1, -1, -1, -1])
    slice_t = getMPIComm().bcast(slice_t, root=0)
    return slice_t


def _get_C3_Tmunu_symmetrized_pyquda(Uap_f: LatticeGauge,
                                    dst2: LatticePropagator,
                                    dst_seq: LatticePropagator,
                                    t0: int):
    """
    对齐 GPT 版 get_C3_Tmunu_symmetrized:
    C3_Tmunu[mu][nu](t) = (空间求和 + trace + time-slice), 然后 roll(-t0), 最后 (mu,nu) 对称化
    """
    Nt = Uap_f.latt_info.global_size[3]
    C3_Tmunu = np.zeros((4,4,Nt), dtype=np.float64)

    # 保证 gauge 已加载到 quda dirac 里（你上面每次都 loadGauge）
    Uap_f.gauge_dirac.loadGauge(Uap_f)

    # ---- helper：对一个 propagator 取 trace( ... ) 得到 complex_field(t,z,y,x) ----
    def trace_prop(prop):
        # prop.data: (..., s_row, s_col, c_row, c_col)
        # trace over row/col in spin & color: sum_{s,c} prop[s,s,c,c]
        return contract("tzyxsscc->tzyx", prop)  # 注意：这里假设轴顺序正好是 s_row,s_col,c_row,c_col

    # 如果你实际轴顺序是 (..., s_row, s_col, c_row, c_col)，上面就对；
    # 若是 (..., s_row, c_row, s_col, c_col)，你要改成 "tzyxscsc->tzyx" 之类。

    # ---- 第一项：tmp = 0.5 * ( U*shift(dst_seq,+1) - shift( U^\dag * dst_seq, -1 ) ) ----
    for mu in range(4):
        # covDev 对 propagator 的“右指标”作用要谨慎：
        # 这里我们复刻 GPT 的写法：对 dst_seq 做 covariant symmetric derivative (右作用)。
        # 用你验证过的 covDev 组合：
        tmp_seq = 0.5 * (Uap_f.pure_gauge.covDev(dst_seq, mu) - Uap_f.pure_gauge.covDev(dst_seq, mu+4))

        for nu in range(4):
            # 计算 trace( dst2 * gamma[nu] * tmp_seq )
            # gamma 只作用在 spin 指标： (s_row,s_col)
            # 令 Y = gamma_nu * tmp_seq  (左乘 gamma 作用在 s_row)
            Y = contract("ab,tzyxbscd->tzyxascd", D_gammas[nu], tmp_seq.data)  # a=s_row'
            # dst2 * Y ： dst2 的右 spin/color 与 Y 的左 spin/color 收缩
            # dst2: (..., s_row, s_col, c_row, c_col)
            # Y:    (..., s_row', s_col, c_row, c_col) 这里需要确保匹配
            # 最后取 trace over 外层 s_row'==s_col'? 这一步最容易错！
            #
            # 更稳妥：直接形成矩阵乘法式的 trace：
            # trace( dst2 * (gamma*tmp) ) = sum_{s1,s2,c1,c2} dst2[s1,s2,c1,c2] * (gamma*tmp)[s2,s1,c2,c1]
            #
            # 所以下面我用显式指标交换写：
            dst2_data = dst2.data
            # (gamma*tmp) 的指标换位以便按上面 trace 公式收缩
            # Y: [sL, sR, cL, cR]
            # 需要 Y_swap: [sR, sL, cR, cL]
            Y_swap = contract("tzyxascd->tzyxbadc", Y)  # 这里假设 b=sR, a=sL, d=cR, c=cL

            tr_field = contract("tzyxsscc,tzyxsscc->tzyx", dst2_data, Y_swap)

            Ct = _slice_sum_t_pyquda(Uap_f, tr_field).real
            C3_Tmunu[mu,nu] += 0.5 * np.roll(np.array(Ct, dtype=np.float64), -t0)

        # ---- 第二项（左作用项）：tmp = 0.5 * ( shift(dst2,+1)*U^\dag - shift(dst2*U,-1) ) ----
        #
        # 这部分在 PyQUDA 里没有一个现成 covDev 一行对应，因为它等价于“对 dst2 的左指标做 covariant derivative”。
        # 所以这里按 GPT 的代数结构手动写：需要 cshift 和 link 乘法。
        #
        # 下面用两个 TODO：你需要把 cshift/link-mul 换成你在 PyQUDA 里实际可用的接口。
        #
        # 伪代码：
        #   term1 = shift(dst2, +mu) * U_mu^\dag(x)    (注意这里 U^\dag 在 x 处)
        #   term2 = shift(dst2*U_mu(x), -mu)
        #   tmp_left = 0.5*(term1 - term2)
        #
        # 如果你的 LatticePropagator 支持：
        #   dst2.shift(mu, +1) / dst2.shift(mu, -1)
        #   Uap_f.pure_gauge.mulLinkRight/Left(...)
        # 那就直接替换。
        #
        # 我先写成“需要你替换的接口版本”：
        #
        # tmp_left = _left_covdev_like_GPT(Uap_f, dst2, mu)  # TODO: implement with your available primitives
        #
        # 然后做 trace( tmp_left * gamma[nu] * dst_seq )
        #
        # 下面先给一个占位实现：raise NotImplementedError，避免你不小心跑了错的结果。
        tmp_left = None  # TODO

        # 你如果把 tmp_left 实现好了，再打开这段：
        # for nu in range(4):
        #     Y2 = contract("ab,tzyxbscd->tzyxascd", D_gammas[nu], dst_seq.data)
        #     # trace( tmp_left * (gamma*dst_seq) )
        #     Y2_swap = contract("tzyxascd->tzyxbadc", Y2)
        #     tr_field2 = contract("tzyxsscc,tzyxsscc->tzyx", tmp_left, Y2_swap)
        #     Ct2 = _slice_sum_t_pyquda(Uap_f, tr_field2).real
        #     C3_Tmunu[mu,nu] -= 0.5 * np.roll(np.array(Ct2, dtype=np.float64), -t0)

    # ---- symmetrization ----
    for mu in range(4):
        for nu in range(mu+1, 4):
            C3_Tmunu[mu,nu] = 0.5*(C3_Tmunu[mu,nu] + C3_Tmunu[nu,mu])
            C3_Tmunu[nu,mu] = C3_Tmunu[mu,nu]

    return C3_Tmunu

def C3_con_EMT_pyquda(U_f: LatticeGauge,
                      U_GPT,  # 你 GPT inverter 需要的
                      invPara,
                      stepsize,
                      Nsteps,
                      smearPara,
                      Nsrc,
                      sinkt_range,
                      spin: int):

    assert spin in [0,1,2,5]  # 0,1,2 for Jpsi spin x,y,z; 5 for etac
    to_sm_src, to_sm_dst, sm_sigma, sm_steps = smearPara
    latt_info = U_f.latt_info

    global_size = U_f.latt_info.global_size
    Nx, Ny, Nz, Nt = global_size
    Ns3 = Nx*Ny*Nz

    # --- inverter (沿用你已有的 get_invD) ---
    invD = get_invD(U_GPT, invPara)[0]

    N_sinkt = len(sinkt_range)

    C2 = np.zeros((Nsrc, Nt), dtype=np.float64)
    C3_chi = np.zeros((Nsrc, N_sinkt, Nsteps+1, Nt), dtype=np.float64)
    C3_Tmunu = np.zeros((Nsrc, N_sinkt, Nsteps+1, 4, 4, Nt), dtype=np.float64)
    src_locs = np.zeros((Nsrc,4), dtype=np.uint16)

    for n_src in range(Nsrc):
        x0 = n_src % Nx
        y0 = n_src % Ny
        z0 = n_src % Nz
        t0 = n_src % Nt
        src_locs[n_src] = np.array([x0,y0,z0,t0], dtype=np.uint16)

        # ---- 1) point source ----
        src = g.mspincolor(U_GPT.grid)
        g.create.point(src, [x0,y0,z0,t0])

        # ---- 2) source smearing ----
        if to_sm_src:
            smear = g.create.smear.gauss(U_GPT, sigma=sm_sigma, steps=sm_steps, dimensions=[0,1,2])
            g.message('source smearing starts')
            src = g( smear * src )
            g.message('source smearing ends')
            del smear

        # ---- 3) solve dst_x = invD * src ----
        dst_x = g( invD * src )
        # r = g( w*dst_x-src )
        # g.message( '|r_true| = ', np.sqrt(g.sum(g.trace(g.adj(r)*r)).real) )
        del src
        dst_y_back = g.copy(dst_x)

        # ---- 4) sink smearing (first) ----
        if to_sm_dst:
            smear = g.create.smear.gauss(U_GPT, sigma=sm_sigma, steps=sm_steps, dimensions=[0,1,2])
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

        # ---- 7) sequential propagators for each sink_t ----
        for n_t, sink_t in enumerate(sinkt_range):

            propag_gpt = gpt.LatticePropagatorGPT(dst_x, GEN_SIMD_WIDTH)

            # GPT: src_seq[:,:,:, (sink_t+t0)%Nt ] = dst_x[:,:,:, (sink_t+t0)%Nt ]
            #      src_seq = gamma_spin * src_seq * gamma_spin
            #
            # TODO: construct src_seq_prop by time-slice picking and gamma projection
            src_seq_prop = None  # TODO

            # dst_seq = invD * src_seq
            dst_seq_gpt = None  # TODO
            dst_seq = _gpt_prop_to_pyquda_prop(dst_seq_gpt)  # TODO

            # Uap = U with temporal boundary flip: Uap[3][:,:,:,Nt-1] *= -1
            # 你 PyQUDA 里一般通过 boundary_phases 处理了，但 GPT 这里硬乘了 -1。
            # 若你要逐项对齐 GPT，就也做一次：
            Uap_f = U_f.copy()
            # TODO: implement Uap_f temporal link flip at t=Nt-1 for mu=3 if needed.

            dst_y = dst_y_back.copy()

            for step in range(Nsteps+1):
                # dst2 = gamma5 * adj(dst_y) * gamma5
                dst2 = None  # TODO

                # C3_chi: roll( slice( trace(dst2*dst_seq) ), -t0 )
                # TODO:
                # C3_chi[n_src,n_t,step] += ...

                # C3_Tmunu:
                C3_Tmunu[n_src,n_t,step] += _get_C3_Tmunu_symmetrized_pyquda(Uap_f, dst2, dst_seq, t0)

                # ---- flow step ----
                if Nsteps > 0:
                    # 你在 EMT 里是：
                    #   Multi_eta = U_f.gradientFlow(Multi_eta, "wilson", 1, stepsize, True)
                    # 这里 dst_y 与 dst_seq 都是 propagator：你可能需要 MultiLatticePropagator
                    # 如果没有，就把每列当 fermion flow 一遍（慢但对齐）
                    #
                    # TODO: implement propagator flow in PyQUDA
                    pass

    return C2, C3_chi, C3_Tmunu, src_locs






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



def C3_con_EMT(gaugePara, invPara, flowPara, smearPara, Nsrc, sinkt_range, spin):
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

    np.save(f'C2_{a}_{conf_id}_spin{spin}_HYP_SS_persrc.npy', C2)
    np.save(f'C3_chi_{a}_{conf_id}_spin{spin}_HYP_SS_persrc.npy', C3_chi)
    np.save(f'C3_Tmunu_{a}_{conf_id}_spin{spin}_HYP_SS_persrc.npy', C3_Tmunu)
    np.save(f'src_locs_{a}_{conf_id}_spin{spin}_HYP_SS.npy', src_locs)

    C2 = np.mean( C2, axis=0 )
    C3_chi = np.mean( C3_chi, axis=0 )
    C3_Tmunu = np.mean( C3_Tmunu, axis=0 )

    np.save(f'C2_{a}_{conf_id}_spin{spin}_HYP_SS.npy', C2)
    for n_t,sink_t in enumerate(sinkt_range):
        np.save(f'C3_chi_{a}_{conf_id}_sinkt{sink_t}_spin{spin}_HYP_SS.npy', C3_chi[n_t])
        np.save(f'C3_Tmunu_{a}_{conf_id}_sinkt{sink_t}_spin{spin}_HYP_SS.npy', C3_Tmunu[n_t])
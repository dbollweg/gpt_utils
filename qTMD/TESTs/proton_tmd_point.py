import gpt as g 
import os
import sys
import numpy as np
import math


from tools import *
from io_corr import *

#ADD: Add coulomb gauge version: set longitudinal links to one
config_number = g.default.get_int("--config_num", 0)

groups = {
    "Frontier_batch_0": {
        "confs": [
            str(config_number),
        ],
#        "evec_fmt": "/lustre/orion/proj-shared/nph159/data/64I/%s.evecs/lanczos.output",
        "evec_fmt": "/lustre/orion/proj-shared/nph159/data/64I/job-0%s/lanczos.output",
        "conf_fmt": "/ccs/home/xiangg/latwork/DWF/TEST/nucleon_TMD/random-gauge.nersc",
    },
}


jobs = {
    "proton_TMD_AMA": {
        "exact": 1,
        "sloppy": 1,
        "low": 0,
    },
}


jobs_per_run = g.default.get_int("--gpt_jobs", 1)

# find jobs for this run
def get_job(only_on_conf=None):
    # statistics
    n = 0
    for group in groups:
        for job in jobs:
            for conf in groups[group]["confs"]:
                n += 1

    jid = -1
    for group in groups:
        for conf in groups[group]["confs"]:
            for job in jobs:
                jid += 1
                if only_on_conf is not None and only_on_conf != conf:
                    continue
                return group, job, conf, jid, n

    return None

if g.rank() == 0:
    first_job = get_job()
    run_jobs = str(
        list(
            filter(
                lambda x: x is not None,
                [first_job] + [get_job(first_job[2]) for i in range(1, jobs_per_run)],
            )
        )
    ).encode("utf-8")
else:
    run_jobs = bytes()
run_jobs = eval(g.broadcast(0, run_jobs).decode("utf-8"))

"""
================================================================================
            Every node now knows what to do -> Now initialization
================================================================================
"""

# configuration needs to be the same for all jobs, so load eigenvectors and configuration
conf = run_jobs[0][2]
group = run_jobs[0][0]


# loading gauge configuration
g.message("Loading ")
U = g.load(groups[group]["conf_fmt"])
g.message("finished loading gauge config")

# use the gauge configuration grid
grid = U[0].grid
L = np.array(grid.fdimensions)

g.message("Plaquette:", g.qcd.gauge.plaquette(U))
g.message("Lattice size:", U[0].grid.fdimensions)

# wilson
p = {
    "kappa": 0.139727,
    #"mass": 0.1,
    "csw_r": 0,
    "csw_t": 0,
    "xi_0": 0,
    "nu": 0,
    "isAnisotropic": False,
    "boundary_phases": [1, 1, 1, -1],
}
w = g.qcd.fermion.wilson_clover(U, p)


# create point source
src = g.mspincolor(grid)
pos = [0, 0, 0, 0]
g.create.point(src, pos)
g.message("src:", src[0,0,0,0])

# even-odd preconditioned matrix
eo = g.qcd.fermion.preconditioner.eo2_ne()

# build solver
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-12, "maxiter": 10000})
propagator = w.propagator(inv.preconditioned(pc.eo1_ne(), cg))

# propagator
prop_f = g.mspincolor(grid)
prop_f @= propagator * src
g.message("fw prop [0,0,1,0]:", prop_f[0,0,1,0])

"""
================================================================================
                Gamma structures and Projection of nucleon states
================================================================================
"""
### Gamma structures
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]

### Projection of nucleon states
Cg5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma[5].tensor()
Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.5 # FIXME 0.25
Unpol = Pp
Szp = (g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Szm = (g.gamma["I"].tensor() + 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Sxp = (g.gamma["I"].tensor() - 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
Sxm = (g.gamma["I"].tensor() + 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
PpSzp = Pp * Szp
PpSzm = Pp * Szm
PpSxp = Pp * Sxp
PpSxm = Pp * Sxm
#my_projections=["PpSzp", "PpSxp", "PpSxm"]
#my_projections=["PpSzp", "PpSzm", "PpSxp"]
#PolProjections = [PpSzp, PpSxp, PpSxm]
#PolProjections = [PpSzp, PpSzm, PpSxp]
PolProjections = {
    "PpSzp": PpSzp,
    "PpSzm": PpSzm,
    "PpSxp": PpSxp,
    "PpSxm": PpSxm,  
    "Unpol": Unpol,
}

def down_quark_insertion(Q, Gamma, P):
    #eps_abc eps_a'b'c'Gamma_{beta alpha}Gamma_{beta'alpha'}P_{gamma gamma'}
    # * ( Q^beta'beta_b'b Q^gamma'gamma_{c'c} -  Q^beta'gamma_b'c Q^gamma'beta_{c'b} )
    
    eps = g.epsilon(Q.otype.shape[2])
    
    R = g.lattice(Q)
    
    PDu = g(g.spin_trace(P*Q))

    GtDG = g.eval(g.transpose(Gamma)*Q*Gamma)

    GtDG = g.separate_color(GtDG)
    PDu = g.separate_color(PDu)
    
    GtD = g.eval(g.transpose(Gamma)*Q)
    PDG = g.eval(P*Q*Gamma)
    
    GtD = g.separate_color(GtD)
    PDG = g.separate_color(PDG)
    
    D = {x: g.lattice(GtDG[x]) for x in GtDG}

    for d in D:
        D[d][:] = 0
        
    for i1, sign1 in eps:
        for i2, sign2 in eps:
            D[i1[0], i2[0]] += sign1 * sign2 * g.transpose((PDu[i2[2], i1[2]] * GtDG[i2[1], i1[1]] - GtD[i2[1],i1[2]] * PDG[i2[2], i1[1]]))
            
    g.merge_color(R, D)
    return R

def up_quark_insertion(Qu, Qd, Gamma, P):

    eps = g.epsilon(Qu.otype.shape[2])
    R = g.lattice(Qu)
    Dut = g.lattice(Qu)

    Du_sep = g.separate_color(Qu)
    GDd = g.eval(Gamma * Qd)
    GDd = g.separate_color(GDd)

    '''
    PDu = g.eval(P*Qu)
    PDu = g.separate_color(PDu)

    # ut
    DuP = g.eval(Qu * P)
    DuP = g.separate_color(DuP)
    TrDuP = g(g.spin_trace(Qu * P))
    TrDuP = g.separate_color(TrDuP)
    
    # s2ds1b
    GtDG = g.eval(g.transpose(Gamma)*Qd*Gamma)
    GtDG = g.separate_color(GtDG)

    #sum color indices
    D = {x: g.lattice(GDd[x]) for x in GDd}
    for d in D:
        D[d][:] = 0

    for i1, sign1 in eps:
        for i2, sign2 in eps:
            D[i2[2], i1[2]] += sign1 * sign2 * (P * g.spin_trace(GtDG[i1[1],i2[1]]*g.transpose(Du_sep[i1[0],i2[0]]))
                                + g.transpose(TrDuP[i1[0],i2[0]] * GtDG[i1[1],i2[1]])
                                #+ GtDG[i1[1],i2[1]] * g.transpose(PDu[i1[0],i2[0]])
                                + PDu[i1[0],i2[0]] * g.transpose(GtDG[i1[1],i2[1]])
                                + g.transpose(GtDG[i1[0],i2[0]]) * DuP[i1[1],i2[1]])
    
    g.merge_color(R, D)

    return R

    '''
    #first term & second term
    GDd = g.eval(Gamma * Qd)
    GDd = g.separate_color(GDd)

    DuG = g.eval(Qu * Gamma)
    DuG = g.separate_color(DuG)

    #third term
    Du_sep = g.separate_color(Qu)
    Du_spintransposed = {x: g.lattice(Du_sep[x]) for x in Du_sep}
    for d in Du_spintransposed:
        Du_spintransposed[d] = g(g.transpose(Du_sep[d]))
    g.merge_color(Dut,Du_spintransposed)

    PDut = g.eval(g.transpose(P) * Dut)
    PDut = g.separate_color(PDut)
    GDuG = g.eval(Gamma * Qu * Gamma)
    GDuG = g.separate_color(GDuG)    

    #fourth term
    #GDuG = g.eval(Gamma * Qu * Gamma)
    #GDuG = g.separate_color(GDuG)
    DuP_trace = g(g.spin_trace(Qu * P))
    DuP_trace = g.separate_color(DuP_trace)

    #sum color indices
    D = {x: g.lattice(GDd[x]) for x in GDd}
    for d in D:
        D[d][:] = 0

    for i1, sign1 in eps:
        for i2, sign2 in eps:
            tmp = sign1 * sign2 * (GDd[i1[1],i2[1]] * g.transpose(DuG[i1[0],i2[0]]) * g.transpose(P)
                                + g.spin_trace(GDd[i1[1],i2[1]] * g.transpose(DuG[i1[0],i2[0]])) * g.transpose(P)
                                - PDut[i1[1],i2[1]] * GDuG[i1[0],i2[0]]
                                - DuP_trace[i1[0],i2[0]] * GDuG[i1[1],i2[1]])
            D[i2[2], i1[2]] += g.transpose(tmp)
    
    g.merge_color(R, D)
    return R


def create_bw_seq(inverter, prop, flavor, origin=None):
    
    pf = [0,0,0,0]
    pol_list = ["Unpol", "PpSzp", "PpSzm"]
    t_insert = 1

    pp = 2.0 * np.pi * np.array(pf) / prop.grid.fdimensions
    P = g.exp_ixp(pp, origin)
    
    src_seq = [g.mspincolor(prop.grid) for i in range(len(pol_list))]
    dst_seq = []
    dst_tmp = g.mspincolor(prop.grid)
    
    #g.qcd.baryon.proton_seq_src(prop, src_seq, self.t_insert, flavor)
    for i, pol in enumerate(pol_list):

        if (flavor == 1): 
            g.message("starting diquark contractions for up quark insertion and Polarization ", i)

            src_seq[i] = up_quark_insertion(prop, prop, Cg5, PolProjections[pol])
        elif (flavor == 2):
            g.message("starting diquark contractions for down quark insertion and Polarization ", i)

            src_seq[i] = down_quark_insertion(prop, Cg5, PolProjections[pol])
        else: 
            raise Exception("Unknown flavor for backward sequential src construction")
    
        # sequential solve through t=t_insert
        src_seq_t = g.lattice(src_seq[i])
        src_seq_t[:] = 0
        # FIXME
        src_seq_t[:, :, :, origin[3]+t_insert] = src_seq[i][:, :, :, origin[3]+t_insert]

        g.message("diquark contractions for Polarization ", i, " done")
    
        # FIXME
        #smearing_input = g.eval(g.gamma[5]*P*g.conj(src_seq_t))
        smearing_input = g.eval(g.gamma[5]*P*g.adj(src_seq_t))

        tmp_prop = smearing_input

        dst_tmp = g.eval(inverter * tmp_prop)           
        # FIXME dst_seq.append(g.eval(g.gamma[5] * g.conj(dst_tmp)))
        dst_seq.append(g.eval(g.adj(dst_tmp) * g.gamma[5]))
        #dst_seq.append(g.eval(g.gamma[5] * g.conj(dst_tmp)))

    g.message("bw. seq propagator done")
    return dst_seq

sequential_bw_prop_up = create_bw_seq(propagator, prop_f, 1, pos)
sequential_bw_prop_down = create_bw_seq(propagator, prop_f, 2, pos)

g.message("bw prop U (Unpol) [0,0,1,0]:")
print(sequential_bw_prop_up[0][0,0,1,0])
g.message("bw prop U (PpSzp) [0,0,1,0]:")
print(sequential_bw_prop_up[1][0,0,1,0])
g.message("bw prop U (PpSzm) [0,0,1,0]:")
print(sequential_bw_prop_up[2][0,0,1,0])
g.message("bw prop D (Unpol) [0,0,1,0]:")
print(sequential_bw_prop_down[0][0,0,1,0])
g.message("bw prop D (PpSzp) [0,0,1,0]:")
print(sequential_bw_prop_down[1][0,0,1,0])
g.message("bw prop D (PpSzm) [0,0,1,0]:")
print(sequential_bw_prop_down[2][0,0,1,0])

g.message("bw prop U (Unpol) [0,0,2,0]:")
print(sequential_bw_prop_up[0][0,0,2,0])
g.message("bw prop U (PpSzp) [0,0,2,0]:")
print(sequential_bw_prop_up[1][0,0,2,0])
g.message("bw prop U (PpSzm) [0,0,2,0]:")
print(sequential_bw_prop_up[2][0,0,2,0])
g.message("bw prop D (Unpol) [0,0,2,0]:")
print(sequential_bw_prop_down[0][0,0,2,0])
g.message("bw prop D (PpSzp) [0,0,2,0]:")
print(sequential_bw_prop_down[1][0,0,2,0])
g.message("bw prop D (PpSzm) [0,0,2,0]:")
print(sequential_bw_prop_down[2][0,0,2,0])
#for src_ic in range(0, 3):
#    for src_is in range(0, 4):
#        for sink_ic in range(0, 3):
#            for sink_is in range(0, 4):
#                print(src_ic, src_is, sink_ic, sink_is, bw_prop[sink_is][src_is][sink_ic][src_ic])
#g.message("bw prop U (PpSzm):", sequential_bw_prop_up[1][0,0,0,0])
#g.message("bw prop D (PpSzp):", sequential_bw_prop_down[0][0,0,0,0])
#g.message("bw prop D (PpSzm):", sequential_bw_prop_down[1][0,0,0,0])